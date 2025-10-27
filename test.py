# test.py
import asyncio
import base64
import threading
import time

import numpy as np
import pyaudio
from call_agent import CallAgent  # assuming your class is in callagent.py

# Audio settings (must match your VAD server)
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Must match VAD sample_rate in YAML
CHUNK = 1024  # Small chunks for low latency


class LocalVoiceInterface:
    def __init__(self, call_agent: CallAgent):
        self.call_agent = call_agent
        self.audio = pyaudio.PyAudio()
        self.mic_stream = None
        self.speaker_stream = None
        self.running = False
        self.playback_queue = asyncio.Queue()
        self.loop = None

    async def start(self):
        self.running = True
        self.loop = asyncio.get_running_loop()

        # Start mic recording in a thread (PyAudio is blocking)
        self.mic_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=self._mic_callback,
        )
        self.mic_stream.start_stream()

        # Start speaker playback
        self.speaker_stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            output=True,
            frames_per_buffer=CHUNK,
        )

        # Start playback loop
        asyncio.create_task(self._playback_loop())

        print("🎤 Voice interface started. Speak into your microphone...")
        print("Press Ctrl+C to stop.")

    def _mic_callback(self, in_data, frame_count, time_info, status):
        if self.running and self.loop:
            # Schedule sending audio chunk to agent
            asyncio.run_coroutine_threadsafe(self._send_audio_chunk(in_data), self.loop)
        return (None, pyaudio.paContinue)

    async def _send_audio_chunk(self, audio_bytes: bytes):
        try:
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            await self.call_agent.send_audio_chunk(audio_b64)
        except Exception as e:
            print(f"Error sending audio: {e}")

    async def _playback_loop(self):
        while self.running:
            try:
                # Wait for audio from TTS
                audio_bytes = await asyncio.wait_for(
                    self.playback_queue.get(), timeout=1.0
                )
                if audio_bytes == b"STOP":
                    break
                # Play audio
                self.speaker_stream.write(audio_bytes)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Playback error: {e}")

    def play_audio(self, audio_bytes: bytes):
        """Called from TTS receiver to enqueue audio for playback"""
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self.playback_queue.put(audio_bytes), self.loop
            )

    async def stop(self):
        self.running = False
        if self.mic_stream:
            self.mic_stream.stop_stream()
            self.mic_stream.close()
        if self.speaker_stream:
            self.speaker_stream.close()
        self.audio.terminate()
        await self.playback_queue.put(b"STOP")
        print("Voice interface stopped.")


# Monkey-patch CallAgent to output audio to our speaker
original_tts_receiver = CallAgent.message_receiver_tts


async def patched_tts_receiver(self):
    try:
        audio_complete = False
        while self.connected and "tts" in self.websockets and not audio_complete:
            try:
                message = await self.websockets["tts"].recv()
                if isinstance(message, bytes):
                    # Forward to local speaker
                    if hasattr(self, "_voice_interface"):
                        self._voice_interface.play_audio(message)
                elif message == "AUDIO_COMPLETE":
                    audio_complete = True
                    break
            except asyncio.TimeoutError:
                continue
    except Exception as e:
        print(f"TTS receiver error: {e}")


CallAgent.message_receiver_tts = patched_tts_receiver


async def main():
    # Load your YAML config (must point to localhost due to SSH forwarding)
    yaml_path = "config/config.yaml"  # Update this path

    # Initialize CallAgent
    agent = CallAgent(
        owner_id="test_user",
        system_prompt="You are a helpful assistant.",
        yaml_path=yaml_path,
        kb_id=[],
        config={"temperature": 0.7, "max_tokens": 1024},
    )

    # Connect to remote services (via localhost due to SSH tunnel)
    await agent.connect_servers()

    # Start receivers
    await agent.start_receivers()

    # Attach voice interface
    voice = LocalVoiceInterface(agent)
    agent._voice_interface = voice  # So TTS can access it

    try:
        await voice.start()
        # Keep running
        while voice.running:
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        await voice.stop()
        await agent.close()


if __name__ == "__main__":
    asyncio.run(main())
