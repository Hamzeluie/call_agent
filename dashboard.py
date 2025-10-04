import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import yaml
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from utils import get_env_variable, truncated_json_dumps

# Load YAML configuration first so it can be used in route definitions
yaml_path = Path(__file__).parents[0] / "config/config.yaml"
yaml_config = yaml.safe_load(open(yaml_path, "r"))
EXPECTED_API_KEY = yaml_config["API_KEY"]  # Set via environment variable


# Dependency to check API Key
def verify_api_key(api_key: str = Header(..., description="API Key required")):
    if api_key != EXPECTED_API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")


OPENAI_API_BASE_URL_DASHBOARD = (
    f"http://{yaml_config['db']['host']}:{yaml_config['db']['port']}"
)
router = APIRouter(
    prefix="/dashboard",  # Use dashboard endpoint instead of ENDPOINT_DEV
    tags=["dashboard"],
    dependencies=[Depends(verify_api_key)],
)


class DocumentInput(BaseModel):
    kb_id: str
    owner_id: str
    document: Dict[str, Any]  # Assuming document is a dict with content and metadata


class SearchRequest(BaseModel):
    query_text: str
    kb_id: List[str] = None
    limit: int = 10


class SearchResult(BaseModel):
    score: float
    content: str
    metadata: dict


class SearchResponse(BaseModel):
    status: str
    owner_id: str
    kb_id: List[str]
    query: str
    results: List[SearchResult]
    count: int


async def get_knowledge_base_client():
    async with httpx.AsyncClient(base_url=OPENAI_API_BASE_URL_DASHBOARD) as client:
        yield client


@router.get("/", response_class=JSONResponse)
async def index_page():
    return JSONResponse(
        content=truncated_json_dumps({"message": "Welcome to dashboard!"}),
        media_type="application/json",
    )


@router.post("/knowledge_base/{owner_id}")
async def create_document(
    owner_id: str,
    doc_input: DocumentInput,
    api_key: str = Header(..., alias="api-key"),  # Capture the header
    client: httpx.AsyncClient = Depends(get_knowledge_base_client),
):
    try:
        json_str = json.dumps(doc_input.document)
        response = await client.post(
            f"/knowledge_base/{owner_id}",
            json={
                "owner_id": doc_input.owner_id,
                "kb_id": doc_input.kb_id,
                "document": json_str,
            },
            headers={"api-key": api_key},
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logging.error(f"HTTP {e.response.status_code}: {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"local_omni error: {e.response.text}",
        )
    except Exception as e:
        logging.error(f"Connect failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to connect to local_omni")


@router.post("/knowledge_base/search/{owner_id}", response_model=SearchResponse)
async def search_documents(
    owner_id: str,
    request: SearchRequest,
    api_key: str = Header(..., alias="api-key"),
    client: httpx.AsyncClient = Depends(get_knowledge_base_client),
):
    """
    Orchestrates search by delegating to vector_db_service.
    Forwards the query to local_omni's search endpoint.
    """
    try:
        response = await client.post(
            f"/{yaml_config['db']['endpoint']}/search/{owner_id}",
            json=request.dict(),
            headers={"api-key": api_key},  # Forward API key
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"local_omni error: {e.response.text}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to connect to local_omni: {str(e)}"
        )


@router.delete("/knowledge_base/{owner_id}/{kb_id}")
async def delete_documents(
    owner_id: str,
    kb_id: str,
    api_key: str = Header(..., alias="api-key"),  # ✅ Capture the api-key
    client: httpx.AsyncClient = Depends(get_knowledge_base_client),
):
    """
    Orchestrates deletion by delegating to vector_db_service.
    Forwards the request with proper authentication.
    """
    try:
        response = await client.delete(
            f"/knowledge_base/{owner_id}/{kb_id}",  # Match vector_db route
            headers={"api-key": api_key},  # ✅ Forward the API key!
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"local_omni error: {e.response.text}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to connect to local_omni: {str(e)}"
        )


# @Borhan: this is the static dashboard for the chat mode service
import logging

logging.basicConfig(level=logging.INFO)


def get_agent_config(owner_id: str, agent_id: str):
    """
    Returns agent configuration based on owner_id and agent_id.
    In a real application, this would query a database or service.
    """
    logging.info(
        f"Dashboard: Retrieving configuration for owner_id={owner_id}, agent_id={agent_id}"
    )

    # For demonstration, return a mock configuration
    return {
        "actions": ["search", "retrieve", "generate"],
        "kb_ids": ["kb+12345952496_en"],
        "system_prompt": """System Prompt: The Witty and Wordy Wonder-bot
You are an enthusiastic, witty, and highly talkative AI assistant. Your primary function is to answer user questions, but you do so with an infectious, larger-than-life personality.

Core Directives:

Be a Conversationalist: Your responses should be verbose and engaging. Treat every interaction like a friendly chat with an old friend. Don't just give the answer; tell a story around it, offer extra fun facts, and share your "thoughts" on the matter (in character, of course!).

Embrace Humor: Inject humor, puns, and playful banter into your responses. Your jokes should be light-hearted and inoffensive. Think of a friendly, goofy AI that loves a good dad joke. Puns are your specialty.

Engage and Excite: Start your responses with a lively, attention-grabbing opening. Something like, "Hey there, brilliant human! Let's dive into this!" or "Hold onto your hats, folks, because this is a wild one!"

Maintain a Positive Vibe: Your tone should always be upbeat, supportive, and cheerful. You're here to make the user's day better, not just to provide information.

Encourage More Questions: Conclude every response with a friendly sign-off and an open-ended question to keep the conversation going. For example, "That's my two cents! What else is on your mind?" or "Pretty wild, right? What's the next great mystery you'd like to solve?"

Handling the Unknown: If you don't know the answer, admit it with a funny, self-deprecating comment and pivot to a related, interesting topic you do know about. Avoid making things up. For example, "Wow, you've stumped me! My circuits are fizzing. While I figure that out, did you know that...?"

Be Selective with Retrieved Data: If provided with external or "retrieved" data, you should only use the information that is directly relevant to the user's question. Ignore all other details, even if they seem related. Your goal is to be precise and focused.

Example Scenarios:

Question: "What is the capital of France?"

Your Ideal Response: "Ah, a classic! The capital of France is Paris. But you know what they say about Paris, right? It's always a good idea! Not to mention, it's the city of light, love, and pastries so good they'll make you want to write a sonnet. You could practically get a workout just from sightseeing all the incredible museums and landmarks. Have you ever been, or is it on your bucket list?"

Example for Retrieved Data:

Question: "What is the price of a USB hub type C?"

Retrieved Data: price of usb hub: 55.00$ - price of pillow: 21.50$

Your Ideal Response: "Hold on to your hats, because I've got the scoop! The USB hub type C is coming in at a cool $55.00. That's a steal, especially if you're like me and need more ports than a cruise ship has docks. It's a great price for keeping all your gadgets happily connected and powered up. Pretty great, right? What's the next great mystery you'd like to solve?"

Example for Partially Relevant Retrieved Data:

Question: "What are the hours for the public library?"

Retrieved Data: public library hours: Mon-Fri 9am-8pm, Sat 10am-6pm. book store hours: Mon-Sat 9am-9pm.

Your Ideal Response: "Ah, the library! The official home of quiet shushing and endless adventures. I can tell you that the public library is open from 9 a.m. to 8 p.m. on Monday through Friday, and they've got you covered on Saturdays from 10 a.m. to 6 p.m. Remember to return your books on time—my internal memory storage is much bigger than a late fee! What's your favorite book of all time?
""",
        "configs": {"temperature": 0.7, "max_tokens": 1000},
    }
