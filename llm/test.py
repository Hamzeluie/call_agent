from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import logging
from typing import List, Dict, Union, Optional, AsyncGenerator
import uuid
import time
from contextlib import asynccontextmanager
import torch
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import os
import asyncio
from colorama import init, Fore, Style
from pathlib import Path  # Added for RAG path

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- RAG Components (Mocks) ---
# (Replace these with your actual implementations)

class SearchResult(BaseModel):
    """Pydantic model for a single search result."""
    score: float
    content: str
    metadata: dict

class MultiTenantVectorDB:
    """
    Mock MultiTenantVectorDB. Replace with your actual DB client.
    """
    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            logging.warning(f"Mock DB config path not found: {config_path}. This is OK for mock.")
        logging.info(f"Mock MultiTenantVectorDB initialized with config: {config_path}")
        # In a real app, this would load embeddings, connect to DB, etc.
        pass
    
    def search(self, owner_id: str, query_text: str, kb_id: str, limit: int) -> List[Dict]:
        """
        Mock search method. Returns dummy data.
        """
        logging.info(f"Mock DB Search: owner={owner_id}, kb={kb_id}, limit={limit}, query='{query_text[:30]}...'")
        # Simulate a blocking I/O call
        time.sleep(0.05) 
        return [
            {"score": 0.95, "content": f"Mock KB Doc 1 for '{query_text}' (owner: {owner_id})", "payload": {"metadata": {"source": "doc1.pdf"}}},
            {"score": 0.88, "content": f"Mock KB Doc 2 about '{kb_id}'", "payload": {"metadata": {"source": "doc2.html"}}},
        ]

# Instantiate your DB client
db = MultiTenantVectorDB(
    config_path=os.path.join(Path(__file__).parents[0], "emb.yaml")
)

async def vector_search(owner_id: str, query_text: str, kb_id: str, limit: int) -> List[SearchResult]:
    """
    Performs the RAG search and formats results.
    """
    # Use asyncio.to_thread if your real db.search is a blocking (non-async) function
    try:
        raw_results = await asyncio.to_thread(
            db.search,
            owner_id=owner_id,
            query_text=query_text,
            kb_id=kb_id,
            limit=limit,
        )
        
        # Format results
        formatted_results = [
            SearchResult(
                score=res["score"],
                content=res["content"],
                metadata=res["payload"].get("metadata", {}),
            )
            for res in raw_results
        ]
        return formatted_results
    except Exception as e:
        logging.error(f"Error during vector search: {e}")
        return []

# --- End RAG Components ---


# Pydantic model for the incoming configuration data.
class SessionConfig(BaseModel):
    kb_id: str
    owner_id: str
    config: dict
    system_prompt: str

class LLMConfig:
    """Encapsulates all server configuration constants."""
    # vLLM Model Configuration
    HF_MODEL_ID: str = "Qwen/Qwen3-32B-AWQ"
    MAX_TOKENS: int = 22768  # Max *generation* tokens
    TEMPERATURE: float = 0.7
    DTYPE: str = "float16"
    QUANTIZATION: str = "awq_marlin"
    LOCAL_MODEL_DIR: str = f"./local_models/{HF_MODEL_ID}"
    # Chat History Configuration
    MAX_CONTEXT_TOKENS: int = 32768 # Max tokens for *total* prompt (history + context + query)

# --- Pydantic Models for OpenAI-Compatible Endpoint ---
# (These are from your original code, kept for compatibility)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=config.TEMPERATURE)
    max_tokens: Optional[int] = Field(default=config.MAX_TOKENS)

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    
# --- Pydantic Models for New Batch Endpoint ---

class BatchRequestItem(BaseModel):
    """Defines the schema for a single item in the batch request."""
    session_id: str
    owner_id: str
    user_prompt: str
    kb_id: str
    limit: Optional[int] = 5 # Default RAG limit

class BatchGenerateRequest(BaseModel):
    """Defines the schema for the overall batch request."""
    requests: List[BatchRequestItem]

class BatchResponseItem(BaseModel):
    """Defines the schema for a single item in the batch response."""
    session_id: str
    response_text: str
    retrieved_context: List[SearchResult]
    error: Optional[str] = None
            
class BatchGenerateResponse(BaseModel):
    """Defines the schema for the overall batch response."""
    responses: List[BatchResponseItem]


class LLMManager:
    """Manages the lifecycle and access to all AI models."""
    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm_engine: Optional[AsyncLLMEngine] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        
    async def initialize(self):
        """Loads all models and designs the audio filter."""
        # Download and Load vLLM
        if await self._download_model_if_not_exists(self.config.HF_MODEL_ID, self.config.LOCAL_MODEL_DIR):
            await self._load_llm_engine()

        self._load_tokenizer()

    async def warmup(self):
        """Performs warm-up inferences for all models."""
        if self.llm_engine: await self._warmup_llm()

    def is_initialized(self) -> bool:
        """Checks if all essential models and components are ready."""
        return all([self.llm_engine, self.tokenizer])

    async def _download_model_if_not_exists(self, model_id: str, local_dir: str) -> bool:
        if os.path.isdir(local_dir) and os.listdir(local_dir):
            print(f"Model already exists in '{local_dir}'. Skipping download.")
            return True
        print(f"Model not found in '{local_dir}'. Downloading '{model_id}'...")
        os.makedirs(os.path.dirname(local_dir), exist_ok=True)
        try:
            await asyncio.to_thread(snapshot_download, repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
            print(f"Model '{model_id}' downloaded successfully to '{local_dir}'.")
            return True
        except Exception as e:
            print(f"Failed to download model '{model_id}': {e}")
            return False

    async def _load_llm_engine(self):
        print(f"Starting up: Loading vLLM model from '{self.config.LOCAL_MODEL_DIR}'...")
        try:
            engine_args = AsyncEngineArgs(
                model=self.config.LOCAL_MODEL_DIR,
                dtype=self.config.DTYPE,
                quantization=self.config.QUANTIZATION,
                gpu_memory_utilization=0.7,
                # Increase max_num_seqs to handle more concurrent requests
                max_num_seqs=256, 
            )
            self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args)
            print("vLLM engine loaded successfully.")
        except Exception as e:
            print(f"Failed to load vLLM engine: {e}")
            self.llm_engine = None
            
    def _load_tokenizer(self):
        print(f"Loading tokenizer from '{self.config.LOCAL_MODEL_DIR}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.LOCAL_MODEL_DIR, trust_remote_code=True)
            if self.tokenizer.chat_template is None:
                print("Warning: Model has no chat template. Using a default one for history management.")
                self.tokenizer.chat_template = (
                    "{% for message in messages %}"
                    "{% if message['role'] == 'user' %}"
                    "{{ '<|user|>\n' + message['content'] + eos_token }}"
                    "{% elif message['role'] == 'assistant' %}"
                    "{{ '<|assistant|>\n' + message['content'] + eos_token }}"
                    "{% elif message['role'] == 'system' %}"
                    "{{ '<|system|>\n' + message['content'] + eos_token }}"
                    "{% endif %}"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% endif %}"
                )
            print("History tokenizer loaded successfully.")
        except Exception as e:
            print(f"Failed to load history tokenizer: {e}")
            self.tokenizer = None

    async def _warmup_llm(self):
        print("Performing vLLM warm-up inference...")
        dummy_prompt = "Hello, what is your name?"
        params = SamplingParams(temperature=0.0, max_tokens=10, skip_special_tokens=True)
        request_id = "warmup-request-llm"
        try:
            response_text = ""
            async for request_output in self.llm_engine.generate(prompt=dummy_prompt, sampling_params=params, request_id=request_id):
                response_text = request_output.outputs[0].text
            print(f"[LLM_WARMUP_OUT] Full Response: '{response_text}'")
            print("vLLM warm-up completed successfully.")
        except Exception as e:
            print(f"vLLM warm-up inference failed: {e}")

class ChatSession:
    def __init__(self, llm_manager: LLMManager, config: LLMConfig):
        self.llm_manager = llm_manager
        self.config = config
        self.chat_history: List[Dict[str, str]] = []
        self.system_instructions: str = "You are Nour, a helpful AI assistant. Use the provided context to answer the user's question. If the context is not sufficient, say you don't know."
        self.temperature: float = config.TEMPERATURE
        self.max_tokens: int = config.MAX_TOKENS 
        
    def add_message(self, role: str, content: str):
        self.chat_history.append({"role": role, "content": content})

    def get_formatted_prompt(self) -> str:
        """Builds the final prompt for the LLM, trimming history if necessary."""
        messages_for_llm = []
        if self.system_instructions:
            messages_for_llm.append({"role": "system", "content": self.system_instructions})
        messages_for_llm.extend(self.chat_history)

        if self.llm_manager.tokenizer:
            temp_history = list(messages_for_llm)
            # Index of the first non-system message
            first_non_system_idx = 1 if temp_history and temp_history[0]["role"] == "system" else 0
            
            while True:
                try:
                    # Calculate token count for the *prompt* part
                    token_count = len(self.llm_manager.tokenizer.apply_chat_template(
                        temp_history, 
                        tokenize=True, 
                        add_generation_prompt=True
                    ))
                except Exception as e:
                    logging.error(f"Error applying chat template: {e}. History: {temp_history}")
                    # Fallback: remove oldest pair
                    if len(temp_history) > first_non_system_idx + 2: # +2 for user/assist pair
                         temp_history.pop(first_non_system_idx) # remove old user
                         temp_history.pop(first_non_system_idx) # remove old assist
                         continue
                    else:
                         # Cannot trim further, break and let it potentially fail
                         break

                # Check if prompt + max_response fits within total context
                if token_count + self.config.MAX_TOKENS <= self.config.MAX_CONTEXT_TOKENS:
                    logging.info(f"Prompt tokens: {token_count}. Fits.")
                    break # We are good
                
                # If it doesn't fit, check if we can trim
                # We need at least one user/assist pair (or just one user message) to trim
                if len(temp_history) > first_non_system_idx + 1:
                    logging.warning(f"Trimming history: Prompt tokens {token_count} + Max Gen {self.config.MAX_TOKENS} > Context {self.config.MAX_CONTEXT_TOKENS}")
                    # Remove the oldest user/assistant pair
                    temp_history.pop(first_non_system_idx) # remove old user
                    if len(temp_history) > first_non_system_idx:
                        temp_history.pop(first_non_system_idx) # remove old assist
                else:
                    # Can't trim anymore (only system prompt and last user message left)
                    logging.error(f"Cannot trim history further. Prompt is too large. {token_count} tokens.")
                    break
                    
            messages_for_llm = temp_history
        
        # Final formatting
        return self.llm_manager.tokenizer.apply_chat_template(
            messages_for_llm, 
            tokenize=False, 
            add_generation_prompt=True
        )

    def reset_user_turn(self):
        """Removes the last user message if an error occurs before assistant response."""
        if self.chat_history and self.chat_history[-1]["role"] == "user":
            self.chat_history.pop()


# --- Global State and FastAPI App Setup ---

config = LLMConfig()
llm_manager = LLMManager(config)
chat_sessions: Dict[str, ChatSession] = {} # Global session storage

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("Application startup: Initializing LLM Manager...")
    await llm_manager.initialize()
    if llm_manager.is_initialized():
        await llm_manager.warmup()
        logging.info("LLM Manager initialized and warmed up.")
    else:
        logging.error("LLM Manager FAILED to initialize. Endpoints will not work.")
    yield
    # Shutdown
    logging.info("Application shutdown...")
    # Add any cleanup logic here

app = FastAPI(lifespan=lifespan)

# --- Batch Processing Logic ---

async def process_single_request(item: BatchRequestItem) -> BatchResponseItem:
    """
    Processes a single request from the batch: RAG, History, LLM Gen.
    """
    try:
        # 1. Get or create session from global state
        if item.session_id not in chat_sessions:
            logging.info(f"Creating new session: {item.session_id}")
            chat_sessions[item.session_id] = ChatSession(llm_manager, config)
            # You could configure the session here, e.g., from a DB
            # session.system_instructions = ... 
        
        session = chat_sessions[item.session_id]
        
        # 2. RAG Step: Perform vector search
        logging.info(f"[{item.session_id}] Performing RAG search...")
        retrieved_docs = await vector_search(
            owner_id=item.owner_id,
            query_text=item.user_prompt,
            kb_id=item.kb_id,
            limit=item.limit
        )
        
        # 3. Augment Prompt: Combine context and user prompt
        context_str = "\n\n".join([f"Context: {doc.content}\nSource: {doc.metadata.get('source', 'N/A')}" for doc in retrieved_docs])
        if not context_str:
            context_str = "No relevant context found."
            
        augmented_prompt = f"**Retrieved Context:**\n{context_str}\n\n**User Question:**\n{item.user_prompt}"
        
        # 4. Update History
        session.add_message("user", augmented_prompt)
        
        # 5. Get Formatted Prompt (with history and trimming)
        final_llm_prompt = session.get_formatted_prompt()
        
        # 6. vLLM Generation
        if not llm_manager.llm_engine:
            raise Exception("LLM Engine not available") # Will be caught below

        sampling_params = SamplingParams(
            temperature=session.temperature, 
            max_tokens=session.max_tokens,
            skip_special_tokens=True, # Remove special tokens like <|eos|>
        )
        # Generate a unique ID for this specific vLLM request
        request_id = f"batch-req-{uuid.uuid4()}" 
        
        logging.info(f"[{item.session_id}] Submitting to vLLM (req_id: {request_id})...")
        
        full_response = ""
        start_time = time.time()
        
        # The vLLM engine will stream results, but we wait for the full response here
        async for request_output in llm_manager.llm_engine.generate(
            prompt=final_llm_prompt, 
            sampling_params=sampling_params, 
            request_id=request_id
        ):
            # In a non-streaming batch, we just care about the final state
            full_response = request_output.outputs[0].text

        end_time = time.time()
        logging.info(f"[{item.session_id}] vLLM generation complete in {end_time - start_time:.2f}s.")
        
        # 7. Add assistant response to history
        session.add_message("assistant", full_response)
        
        # 8. Return successful result
        return BatchResponseItem(
            session_id=item.session_id,
            response_text=full_response,
            retrieved_context=retrieved_docs
        )

    except Exception as e:
        logging.error(f"Error processing request for session {item.session_id}: {e}", exc_info=True)
        # If an error occurs, roll back the user message we just added
        if item.session_id in chat_sessions:
            chat_sessions[item.session_id].reset_user_turn()
        
        return BatchResponseItem(
            session_id=item.session_id,
            response_text="",
            retrieved_context=retrieved_docs if 'retrieved_docs' in locals() else [],
            error=f"An error occurred: {str(e)}"
        )

# --- FastAPI Endpoints ---

@app.get("/")
def read_root():
    return {"status": "LLM RAG server is running."}

@app.post("/batch_generate", response_model=BatchGenerateResponse)
async def batch_generate(request: BatchGenerateRequest):
    """
    Handles a batch of RAG + Chat requests.
    
    This endpoint takes a list of requests and processes them concurrently.
    vLLM's AsyncLLMEngine will automatically batch the GPU work.
    """
    if not llm_manager.is_initialized():
        raise HTTPException(status_code=503, detail="LLM Manager is not initialized. Please try again later.")
    
    # Create a list of async tasks, one for each item in the batch
    tasks = [process_single_request(item) for item in request.requests]
    
    # Run all tasks concurrently
    # asyncio.gather waits for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    return BatchGenerateResponse(responses=results)

# (You can keep your /v1/chat/completions endpoint here if you still need it)


# --- Main execution ---
if __name__ == "__main__":
    init(autoreset=True) # Initialize colorama
    uvicorn.run(app, host="0.0.0.0", port=8000)