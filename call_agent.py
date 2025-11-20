import asyncio
import json
import logging
import wave
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np
import redis.asyncio as redis

from agent_architect.datatype_abstraction import AudioFeatures
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


def save_wav(pcm_bytes: bytes, sample_rate: int, output_path: Path):
    audio_np = np.frombuffer(pcm_bytes, dtype=np.float32)
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    logger.info(f"Saved {output_path}")


def get_interrupted_channels(req: AgentSessions):
    middle_channels = []
    for service in req.service_names:
        if (
            service in req.channels_steps
            and not service == req.first_channel.split(":")[0]
        ):
            for priority in req.channels_steps[service]:
                middle_channels.append(f"{service}:{priority}")
    return middle_channels + [req.last_channel]


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
        self.redis_url = redis_url
        self.agent_type = agent_type
        self.service_names = service_names
        self.channels_steps = channels_steps
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.timeout = timeout
        self.redis_client = None
        self.pubsub = None
        self.active_sessions_key = f"{self.agent_type}:active_sessions"

    async def initialize(self):
        """Initialize Redis connection"""
        self.redis_client = await redis.from_url(self.redis_url, decode_responses=False)
        self.pubsub = self.redis_client.pubsub()
        await self.pubsub.subscribe(self.output_channel)
        logger.info(f"Redis queue manager initialized for queue")

    async def start_session(self, sid: str, agent_id: str, owner_id: str):
        """Mark a session as active"""
        await self.stop_session(sid)
        agent_session = AgentSessions(
            sid=sid,
            agent_type=self.agent_type,
            agent_id=agent_id,
            service_names=self.service_names,
            channels_steps=self.channels_steps,
            owner_id=owner_id,
            status=SessionStatus.ACTIVE,
            first_channel=self.input_channel,
            last_channel=self.output_channel,
            timeout=self.timeout,
        )
        await self.redis_client.hset(
            self.active_sessions_key, sid, agent_session.to_json()
        )
        logger.info(f"Session {sid} started")

    async def get_status_object(self, sid: str) -> AgentSessions:

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
                    await self.stop_session(status_obj.sid)
            except Exception as e:
                logger.warning(f"Failed to parse session {sid}: {e}")

    async def cleanup_interrupt_sessions(self):
        """clean all queue with status 'interrupt'."""
        try:
            sessions = await self.redis_client.hgetall(self.active_sessions_key)
            for sid, raw_status in sessions.items():
                status_obj = AgentSessions.from_json(raw_status)
                if status_obj.status == SessionStatus.INTERRUPT:
                    # print("🦓🦓🦓🦓🦓")
                    await self.interrupt_session_requests(status_obj)
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

    async def is_session_interrupt(self, sid: str) -> bool:
        """Check if a session is active"""
        status_obj = await self.get_status_object(sid)
        # print(f"Checking interrupt for session {sid}, status_obj: {status_obj}")
        if status_obj is None:
            return True
        if status_obj.is_expired():
            await self.stop_session(sid)
            return True
        if status_obj.status == SessionStatus.INTERRUPT:
            return True
        return False

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

    async def interrupt_session_requests(self, req: AgentSessions):
        """Remove any queued requests for stopped session"""
        sid = req.sid
        for queue in get_interrupted_channels(req):
            items = await self.redis_client.lrange(queue, 0, -1)
            for item in items:
                try:
                    request_data = json.loads(item)
                    if request_data.get("sid") == sid:
                        await self.redis_client.lrem(queue, 1, item)
                        print(f"remove from channel:{queue}, sid:{sid}")
                except json.JSONDecodeError:
                    continue

    async def submit_data_request(self, request_data: AudioFeatures, sid: str) -> str:
        status_obj = await self.get_status_object(sid=sid)
        next_service = go_next_service(
            current_stage_name="start",
            service_names=status_obj.service_names,
            channels_steps=status_obj.channels_steps,
            last_channel=status_obj.last_channel,
            prioriry="input",
        )
        print("=>**", next_service)
        await self.redis_client.lpush(next_service, request_data.to_json())
        logger.info(f"Request {sid} submitted to {next_service}")

    async def listen_for_result(self, sid: str) -> AsyncGenerator[AudioFeatures, None]:
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

                # Check if session was interrupted during processing
                if not await self.is_session_interrupt(sid):
                    raw = await self.redis_client.hget(self.active_sessions_key, sid)
                    if raw is not None:
                        logger.info(
                            f"Session {sid} interrupted — stopping audio stream"
                        )
                        status_obj = AgentSessions.from_json(raw)
                        if status_obj.status == SessionStatus.INTERRUPT:
                            await self.interrupt_session_requests(status_obj)

                # Check if session was stopped during processing
                if not await self.is_session_active(sid):
                    raise Exception(f"Session {sid} was stopped during processing")

                # Block indefinitely until an item is available
                result = await temp_client.brpop(status_obj.last_channel, timeout=0)

                # BRPOP with timeout=0 blocks forever until an item arrives
                if result is None:
                    # This should never happen with timeout=0, but included for safety
                    continue

                _, raw_data = result

                try:
                    result_data = AudioFeatures.from_json(raw_data)

                    # Save audio file before processing
                    print("💾 Saving audio file for session:", result_data.sid)
                    # Save audio to directory (minimal version)
                    audio_dir = Path("/home/ubuntu/borhan/whole_pipeline/vexu/outputs")
                    audio_dir.mkdir(exist_ok=True)
                    file_count = len(list(audio_dir.glob("audio_*.wav"))) + 1
                    filename = f"audio_{file_count:06d}.wav"
                    save_wav(result_data.audio, 24000, str(audio_dir / filename))
                    logger.info(f"Audio saved as {filename}")
                    print("💾 Saved audio file for session:", result_data.sid)

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

    async def is_session_interrupt(self, sid: str) -> bool:
        """Check if a session is active"""
        return await self.queue_manager.is_session_interrupt(sid)

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

    async def _cleanup_interrupted_sessions_loop(self):
        """Background loop to clean up stopped sessions."""
        while self.is_running:
            try:
                await self.queue_manager.cleanup_interrupt_sessions()
            except Exception as e:
                logger.error(f"Error in stopped session cleanup: {e}")
            await asyncio.sleep(1.0)  # Check every second (adjust as needed)

    async def start(self) -> None:
        """Start the inference service."""
        await self._initialize_components()
        self.is_running = True
        asyncio.create_task(self._cleanup_stopped_sessions_loop())
        asyncio.create_task(self._cleanup_interrupted_sessions_loop())

    async def predict(self, sid: str) -> AsyncGenerator[AudioFeatures, None]:
        if not await self.is_session_active(sid):
            raise Exception(f"Session is not active or has been stopped")
        async for chunk in self.queue_manager.listen_for_result(sid):
            yield chunk

    async def send_chunk(self, sid: str, audio_b64: bytes) -> None:
        """Submit a single audio chunk for processing."""
        if not await self.is_session_active(sid):
            raise Exception(f"Session {sid} is not active")
        input_request = AudioFeatures(
            sid=sid,
            agent_type=self.agent_type,
            is_final=False,
            priority="input",
            created_at=None,
            audio=audio_b64,  # base64 string
            sample_rate=16000,
        )
        print("=>**submit_data_request", type(audio_b64))
        await self.queue_manager.submit_data_request(input_request, sid)
