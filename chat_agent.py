import asyncio
import json
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List

import redis.asyncio as redis
from agent_architect.datatype_abstraction import TextFeatures
from agent_architect.models_abstraction import (
    AbstractInferenceClient,
    AbstractQueueManagerClient,
    DynamicBatchManager,
)
from agent_architect.session_abstraction import AgentSessions, SessionStatus
from agent_architect.utils import get_all_channels, go_next_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

agent_type = "chat"
SERVICE_NAMES = ["RAG"]
CHANNEL_STEPS = {"RAG": ["high", "low"]}
INPUT_CHANNEL = f"{SERVICE_NAMES[0]}:{CHANNEL_STEPS[SERVICE_NAMES[0]][1]}"
OUTPUT_CHANNEL = f"{agent_type.lower()}:output"


class RedisQueueManager(AbstractQueueManagerClient):
    """
    Manages Redis-based async queue for inference requests
    """

    def __init__(
        self,
        agent_type: str,
        service_names: str,
        channels_steps: str,
        input_channel: str,
        output_channel: str,
        redis_url: str = "redis://localhost:6379",
        timeout: float = 30.0,
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
        self.rag_uri = f"ws://{self.yaml_config['db']['host']}:{self.yaml_config['db']['port']}/ws/search/{self.owner_id}"
        self.llm_config_uri = f"http://{self.yaml_config['llm']['host']}:{self.yaml_config['llm']['port']}/configure"

        # Get session ID
        self.session_id = self.get_session_id_sync()
        self.llm_uri = (
            f"ws://{self.yaml_config['llm']['host']}:{self.yaml_config['llm']['port']}/ws/llm/{self.owner_id}/{self.session_id}"
            if self.session_id
            else None
        )
        logger.info(f"Session {sid} started")

        print(f"RAG URI: {self.rag_uri}")
        print(f"LLM URI: {self.llm_uri}")
        print(f"Session ID: {self.session_id}")
        print(f"llm_config_uri : {self.llm_config_uri}")

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

        raw = await self.redis_client.hget(self.active_sessions_key, sid)
        if raw is None:
            return None
        return AgentSessions.from_json(raw)

    async def cleanup_stopped_sessions(self):
        """Remove all sessions with status 'stop' from active_sessions."""
        sessions = await self.redis_client.hgetall(self.active_sessions_key)
        for sid, raw_status in sessions.items():
            try:
                status_obj = AgentSessions.from_json(raw_status)
                if status_obj.status == SessionStatus.STOP or status_obj.is_expired():
                    print(f"sid {status_obj.sid} is stoped")
                    await self.stop_session(status_obj.sid)
            except Exception as e:
                logger.warning(f"Failed to parse session {sid}: {e}")

    async def cleanup_interrupt_sessions(self):
        """clean all queue with status 'interrupt'."""
        try:
            try:
                headers = {"api-key": self.yaml_config["API_KEY"]}
                self.websockets["rag"] = await websockets.connect(
                    self.rag_uri, ping_interval=None, extra_headers=headers
                )
                print("✅ Connected to KB server")
            except Exception as e:
                logger.error(f"Failed to connect to RAG server: {e}")
                raise
            
            try:
                if self.session_id:
                    self.websockets["llm"] = await websockets.connect(
                        self.llm_uri, ping_interval=None
                    )
                    print("✅ Connected to LLM server")
                else:
                    raise Exception("No session_id available")
            except Exception as e:
                logger.error(f"Failed to connect to LLM server: {e}")
                raise

            self.connected = True
            logger.info("✅ Connected to servers")
            await self.start_receivers()
        except Exception as e:
            logger.warning(f"Failed to interrupt session {sid}: {e}")

    async def stop_session(self, sid: str):
        """Mark a session as inactive and cleanup its requests"""

        status_obj = await self.get_status_object(sid=sid)
        deleted_count = await self.redis_client.hdel(self.active_sessions_key, sid)
        if deleted_count > 0:
            await self.cleanup_session_requests(status_obj)
            logger.info(f"Session {sid} stopped (deleted from Redis)")
        else:
            logger.warning(
                f"Session {sid} was not active - may have been already stopped"
            )

    async def is_session_active(self, sid: str) -> bool:
        """Check if a session is active"""
        status_obj = await self.get_status_object(sid)
        if status_obj is None:
            return False
        if status_obj.is_expired():
            await self.stop_session(sid)
            return False
        return True

    async def cleanup_session_requests(self, req: AgentSessions):
        """Remove any queued requests for stopped session"""
        sid = req.sid
        for queue in get_all_channels(req):
            items = await self.redis_client.lrange(queue, 0, -1)
            for item in items:
                try:
                    request_data = json.loads(item)
                    if request_data.get("sid") == sid:
                        await self.redis_client.lrem(queue, 1, item)
                        print(f"remove from channel:{queue}, sid:{sid}")
                except json.JSONDecodeError:
                    continue

    async def submit_data_request(self, request_data: TextFeatures, sid: str) -> str:
        status_obj = await self.get_status_object(sid=sid)
        next_service = go_next_service(
            current_stage_name="start",
            service_names=status_obj.service_names,
            channels_steps=status_obj.channels_steps,
            last_channel=status_obj.last_channel,
            prioriry=request_data.priority,
        )
        await self.redis_client.lpush(next_service, request_data.to_json())
        logger.info(f"Request {sid} submitted to {next_service}")

    async def listen_for_result(self, sid: str) -> AsyncGenerator[str, None]:
        """
        Listen indefinitely for a result matching the given session ID.
        Results must be pushed to the session's output channel via LPUSH/RPUSH.
        """
        status_obj = await self.get_status_object(sid=sid)
        if not status_obj:
            raise ValueError(f"Session {sid} not found")

        # Use a dedicated client for clean resource management
        temp_client = await redis.from_url(self.redis_url, decode_responses=True)

        try:
            while True:
                # Check if session was stopped during processing
                if not await self.is_session_active(sid):
                    raise Exception(f"Session {sid} was stopped during processing")

                # Block indefinitely until an item is available
                result = await temp_client.brpop(status_obj.last_channel, timeout=0)
                print(f"result: {result}")
                # BRPOP with timeout=0 blocks forever until an item arrives
                if result is None:
                    # This should never happen with timeout=0, but included for safety
                    continue

                _, raw_data = result

                try:
                    result_data = TextFeatures.from_json(raw_data)
                    print(f"✅brpop result: {result_data.priority}")
                    if result_data.sid == sid:
                        yield result_data
                    # Optionally: log mismatched SID or re-queue if needed
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(
                        f"Failed to parse message from {status_obj.last_channel}: {e}"
                    )
                    # Skip invalid messages and keep listening
                    continue

        finally:
            await temp_client.close()

    async def close(self):
        """Cleanup resources"""
        if self.pubsub:
            await self.pubsub.unsubscribe(self.output_channel)
            await self.pubsub.close()
        if self.redis_client:
            await self.redis_client.close()


class InferenceService(AbstractInferenceClient):
    def __init__(
        self,
        agent_type: str,
        service_names: str,
        channels_steps: str,
        input_channel: str,
        output_channel: str,
        redis_url: str = "redis://localhost:6379",
        max_batch_size: int = 16,
        max_wait_time: float = 0.1,
        timeout: float = 30.0,
    ):
        super().__init__()
        self.agent_type = agent_type
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.queue_manager = RedisQueueManager(
            redis_url=redis_url,
            agent_type=agent_type,
            service_names=service_names,
            channels_steps=channels_steps,
            input_channel=input_channel,
            output_channel=output_channel,
            timeout=timeout,
        )
        self.batch_manager = DynamicBatchManager(max_batch_size, max_wait_time)

    async def start_session(self, sid: str, agent_id: str, owner_id: str):
        """Mark a session as active"""
        await self.queue_manager.start_session(sid, agent_id, owner_id)

    async def stop_session(self, sid: str):
        """Stop a specific client session"""
        await self.queue_manager.stop_session(sid)

    async def is_session_active(self, sid: str) -> bool:
        """Check if a session is active"""
        return await self.queue_manager.is_session_active(sid)

    async def _initialize_components(self):
        await self.queue_manager.initialize()

    async def _cleanup_components(self):
        await self.queue_manager.close()

    async def _cleanup_stopped_sessions_loop(self):
        """Background loop to clean up stopped sessions."""
        while self.is_running:
            try:
                await self.queue_manager.cleanup_stopped_sessions()
            except Exception as e:
                logger.error(f"Error in stopped session cleanup: {e}")
            await asyncio.sleep(1.0)  # Check every second (adjust as needed)

    async def start(self) -> None:
        """Start the inference service."""
        await self._initialize_components()
        self.is_running = True
        asyncio.create_task(self._cleanup_stopped_sessions_loop())
        # asyncio.create_task(self.queue_manager.cleanup_interrupt_sessions())

    async def predict(self, sid: str) -> Any:
        if not await self.is_session_active(sid):
            raise Exception(f"Session is not active or has been stopped")
        async for chunk in self.queue_manager.listen_for_result(sid):
            yield chunk

    async def send_chunk(self, sid: str, input_data: str) -> None:
        if not await self.is_session_active(sid):
            raise Exception(f"Session is not active or has been stopped")
        first_service, priority = self.input_channel.split(":")
        print("priority:", priority)
        input_request = TextFeatures(
            sid=sid,
            agent_type=self.agent_type,
            is_final=False,
            priority=priority,
            created_at=None,
            text=input_data,
        )
        await self.queue_manager.submit_data_request(
            request_data=input_request, sid=sid
        )
