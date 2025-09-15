# chat_agent.py - Fixed version

import asyncio
import json
import logging
from typing import AsyncGenerator, Optional

import requests
import websockets
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatAgent:
    def __init__(
        self,
        owner_id,
        system_prompt: str,
        yaml_path: str,
        kb_id: list = [],
        config: dict = {"temperature": 0.7, "max_tokens": 22768},
        limit=5,
    ):
        # Load YAML configuration
        self.yaml_config = yaml.safe_load(open(yaml_path, "r"))
        self.owner_id = owner_id
        self.kb_id = kb_id
        self.limit = limit
        self.system_prompt = system_prompt
        self.config = config

        self.websockets = {}
        self.connected = False
        self.context = ""
        self.receiver_tasks = []
        self.last_message = ""
        self.streaming = False
        self.current_stream_queue = asyncio.Queue()
        self.response_event = asyncio.Event()  # Add event for synchronization

        # Create URIs
        self.rag_uri = f"ws://{self.yaml_config['db']['host']}:{self.yaml_config['db']['port']}/ws/{self.yaml_config['db']['endpoint']}/search/{self.owner_id}"
        self.llm_config_uri = f"http://{self.yaml_config['llm']['host']}:{self.yaml_config['llm']['port']}/configure"

        # Get session ID
        self.session_id = self.get_session_id_sync()
        self.llm_uri = (
            f"ws://{self.yaml_config['llm']['host']}:{self.yaml_config['llm']['port']}/{self.yaml_config['llm']['version']}/{self.yaml_config['llm']['endpoint']}/{self.owner_id}/{self.session_id}"
            if self.session_id
            else None
        )

    def get_session_id_sync(self):
        """Generate session ID synchronously"""
        session_config = {
            "kb_id": self.kb_id[0] if self.kb_id else "",
            "owner_id": self.owner_id,
            "config": self.config,
            "system_prompt": self.system_prompt,
        }
        try:
            response = requests.post(
                self.llm_config_uri, json=session_config, timeout=10
            )
            response.raise_for_status()
            return response.json().get("session_id")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to configure session: {e}")
            return None

    async def connect_servers(self):
        """Connect to WebSocket servers"""
        try:
            headers = {"api-key": self.yaml_config["API_KEY"]}
            self.websockets["rag"] = await websockets.connect(
                self.rag_uri, ping_interval=None, extra_headers=headers
            )
            if self.session_id:
                self.websockets["llm"] = await websockets.connect(
                    self.llm_uri, ping_interval=None
                )
            else:
                raise Exception("No session_id available")

            self.connected = True
            logger.info("✅ Connected to servers")
            await self.start_receivers()
        except Exception as e:
            logger.error(f"Failed to connect to servers: {e}")
            raise

    async def send_message(self, message: str) -> AsyncGenerator[str, None]:
        """
        Send a message and stream the response.
        Usage: async for chunk in agent.send_message("Hello"):
        """
        if not self.connected:
            raise Exception("Not connected to servers")

        self.last_message = message
        self.streaming = True
        self.response_event.clear()  # Clear any previous events

        try:
            # Clear previous stream data
            while not self.current_stream_queue.empty():
                try:
                    self.current_stream_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            # Start RAG search
            await self.send_to_rag()

            # Wait for the first chunk with timeout
            first_chunk_received = False
            start_time = asyncio.get_event_loop().time()
            timeout = 30.0

            while self.streaming:
                try:
                    # Use a shorter timeout for the first chunk
                    chunk_timeout = 5.0 if not first_chunk_received else 1.0
                    chunk = await asyncio.wait_for(
                        self.current_stream_queue.get(), timeout=chunk_timeout
                    )

                    if chunk is None:  # End of stream marker
                        break

                    first_chunk_received = True
                    yield chunk

                except asyncio.TimeoutError:
                    if not first_chunk_received:
                        # If we haven't received the first chunk yet, check overall timeout
                        if asyncio.get_event_loop().time() - start_time > timeout:
                            raise TimeoutError("No response received within timeout")
                        continue
                    else:
                        # If we already received chunks but now timeout, break
                        break

        finally:
            self.streaming = False

    async def send_to_rag(self):
        """Send query to RAG server"""
        request = {
            "query_text": self.last_message,
            "kb_id": self.kb_id,
            "limit": self.limit,
        }
        await self.websockets["rag"].send(json.dumps(request))

    async def send_to_llm(self):
        """Send data to LLM server for streaming"""
        request = {
            "owner_id": self.owner_id,
            "user_input": self.last_message,
            "retrieved_data": self.context,
        }
        await self.websockets["llm"].send(json.dumps(request))

    async def start_receivers(self):
        """Start message receivers"""
        if not self.connected:
            return []

        self.receiver_tasks = [
            asyncio.create_task(self.message_receiver_rag()),
            asyncio.create_task(self.message_receiver_llm()),
        ]
        return self.receiver_tasks

    async def message_receiver_rag(self):
        """Handle RAG responses"""
        try:
            while self.connected:
                try:
                    message_str = await asyncio.wait_for(
                        self.websockets["rag"].recv(), timeout=5.0
                    )
                    message = json.loads(message_str)

                    if message.get("status") == "success":
                        results = message.get("results", [])
                        context_parts = []
                        for i, result in enumerate(results, 1):
                            context_parts.append(
                                f"Context {i} (relevance: {result['score']:.2f}): {result['content']}"
                            )
                        self.context = "\n".join(context_parts)
                        await self.send_to_llm()
                    elif message.get("status") == "error":
                        logger.error(f"RAG error: {message.get('message')}")
                        await self.current_stream_queue.put(
                            f"Error: {message.get('message')}"
                        )
                        await self.current_stream_queue.put(None)

                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break
        except Exception as e:
            logger.error(f"RAG receiver error: {e}")

    async def message_receiver_llm(self):
        """Handle LLM streaming responses"""
        try:
            while self.connected:
                try:
                    message_str = await asyncio.wait_for(
                        self.websockets["llm"].recv(), timeout=1.0
                    )
                    message = json.loads(message_str)

                    if message.get("type") == "chunk":
                        await self.current_stream_queue.put(message.get("content", ""))

                    elif message.get("type") == "complete":
                        await self.current_stream_queue.put(None)

                    elif message.get("type") == "error":
                        logger.error(f"LLM error: {message.get('content')}")
                        await self.current_stream_queue.put(
                            f"Error: {message.get('content')}"
                        )
                        await self.current_stream_queue.put(None)

                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break
        except Exception as e:
            logger.error(f"LLM receiver error: {e}")

    async def close(self):
        """Close connections"""
        self.streaming = False
        self.connected = False

        # Cancel tasks
        for task in self.receiver_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close connections
        for name, ws in self.websockets.items():
            if ws:
                await ws.close()
                logger.info(f"Closed {name} connection")
