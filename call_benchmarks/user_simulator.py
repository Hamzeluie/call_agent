import asyncio
import base64
import json
import os
import uuid
import wave
from typing import List, Optional

import websockets
from metrics_collector import CallMetrics, MetricsCollector


class UserSimulator:
    def __init__(
        self,
        user_id: int,
        base_url: str,
        audio_files: List[str],
        metrics_collector: MetricsCollector,
    ):
        self.user_id = user_id
        self.base_url = base_url
        self.audio_files = audio_files
        self.metrics = metrics_collector
        self.ws = None
        self.call_sid = None
        self.call_metrics = None

    async def simulate_call(self, owner_id: str = "5", user_id: str = "+15551234567"):
        """Simulate a complete call lifecycle"""
        try:
            # Step 1: Initiate call
            call_data = await self._initiate_call(owner_id, user_id)
            self.call_sid = call_data["call_sid"]
            self.call_metrics = self.metrics.start_call(
                f"user_{self.user_id}_{self.call_sid}"
            )

            # Step 2: Connect WebSocket
            await self._connect_websocket()

            # Step 3: Send audio
            await self._send_audio_stream()

            # Step 4: Wait for responses and then end call
            await asyncio.sleep(10)  # Listen for responses
            await self._end_call()

        except Exception as e:
            error_msg = f"User {self.user_id} error: {str(e)}"
            print(error_msg)
            if self.call_metrics:
                self.metrics.end_call(self.call_metrics.call_id, error_msg)

    async def _initiate_call(self, owner_id: str, user_id: str) -> dict:
        """Initiate call through REST API"""
        import aiohttp

        url = f"{self.base_url}/api/initiate-call"
        payload = {
            "owner_id": owner_id,
            "user_id": f"{user_id}_{self.user_id}",  # Make unique per user
            "agent_id": "0",
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Failed to initiate call: {response.status}")

    async def _connect_websocket(self):
        """Connect to WebSocket for audio streaming"""
        ws_url = f"ws://localhost:8000/ws/{self.call_sid}"  # Adjust port as needed
        self.ws = await websockets.connect(ws_url)

        # Wait for connection confirmation
        response = await self.ws.recv()
        data = json.loads(response)
        if data.get("type") == "connected":
            print(f"User {self.user_id}: WebSocket connected")

    async def _send_audio_stream(self):
        """Stream audio files to the WebSocket"""
        if not self.audio_files:
            raise Exception("No audio files available")

        # Use first audio file for simulation
        audio_file = self.audio_files[0]
        audio_data = self._load_audio_file(audio_file)

        # Mark first audio sent
        self.metrics.mark_first_audio(self.call_metrics.call_id)

        # Simulate real-time audio streaming
        chunk_size = 1600  # 100ms of 16kHz 16-bit audio
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]
            if len(chunk) < chunk_size:
                # Pad last chunk if needed
                chunk += b"\x00" * (chunk_size - len(chunk))

            # Convert to base64 as expected by your frontend
            audio_b64 = base64.b64encode(chunk).decode("utf-8")

            message = {"type": "audio", "audio": audio_b64}

            await self.ws.send(json.dumps(message))
            self.metrics.add_audio_sent(self.call_metrics.call_id, len(chunk))

            # Simulate real-time delay
            await asyncio.sleep(0.1)  # 100ms chunks

            # Check for responses
            try:
                response = await asyncio.wait_for(self.ws.recv(), timeout=0.01)
                data = json.loads(response)
                if data.get("type") == "audio":
                    self.metrics.add_audio_received(
                        self.call_metrics.call_id, len(data["audio"])
                    )
                    if not self.call_metrics.first_transcript_time:
                        self.metrics.mark_first_transcript(self.call_metrics.call_id)
            except asyncio.TimeoutError:
                pass

    async def _end_call(self):
        """Properly end the call"""
        if self.ws:
            await self.ws.send(json.dumps({"type": "stop"}))
            await self.ws.close()

        if self.call_metrics:
            self.metrics.end_call(self.call_metrics.call_id)

    def _load_audio_file(self, file_path: str) -> bytes:
        """Load and convert audio file to PCM format"""
        try:
            with wave.open(file_path, "rb") as wav_file:
                # Check if it's already in the right format
                if wav_file.getsampwidth() == 2 and wav_file.getframerate() == 8000:
                    frames = wav_file.readframes(wav_file.getnframes())
                    return frames
                else:
                    # Simple conversion - for production use proper audio conversion
                    frames = wav_file.readframes(wav_file.getnframes())
                    return frames  # You might need proper resampling here
        except Exception as e:
            # Fallback: generate synthetic audio
            print(f"Warning: Using synthetic audio for {file_path}: {e}")
            return self._generate_synthetic_audio()

    def _generate_synthetic_audio(self, duration_seconds: int = 10) -> bytes:
        """Generate synthetic PCM audio for testing"""
        sample_rate = 8000
        samples = duration_seconds * sample_rate
        # Generate a simple sine wave
        import numpy as np

        t = np.linspace(0, duration_seconds, samples, False)
        tone = np.sin(2 * np.pi * 440 * t) * 0.3  # 440 Hz sine wave
        audio_data = (tone * 32767).astype(np.int16).tobytes()
        return audio_data
