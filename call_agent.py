import asyncio
import base64
import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import websockets
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CallAgent:

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

        self.websockets = {}  # Store all websockets in a dictionary
        self.connected = False
        self.last_chunk_audio = None
        self.last_transcript = None
        self.context = ""
        self.sample_rate = 0
        self.receiver_tasks = []  # Store all receiver tasks
        self.llm_response = ""
        self.audio_output = bytearray()
        self.interrupt_requested = False
        self.end_call = False

        # Create URIs for different services
        self.llm_config_uri = f"http://{self.yaml_config['llm']['host']}:{self.yaml_config['llm']['port']}/configure"
        self.vad_uri = ""
        self.stt_uri = ""
        self.rag_uri = ""
        self.llm_uri = ""
        self.tts_uri = ""
        self.formatted_audio_responses = []  # Store formatted audio responses
        self.tts_responses = []  # Store TTS completion responses
        self.current_item_id = str(
            uuid.uuid4()
        )  # Generate a unique ID for the current conversation

    def clean_markdown_formatting(self, text):
        """Remove markdown formatting from text to improve TTS readability"""
        import re

        # Replace bold/italic markdown with plain text
        # Replace **text** or *text* with just text
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
        text = re.sub(r"\*(.*?)\*", r"\1", text)

        # Handle currency symbols with appropriate spacing
        # Make sure $ symbol is pronounced by adding a space after it if followed by a number
        text = re.sub(r"\$(\d)", r"$ \1", text)

        # Handle other markdown elements as needed
        # Remove backticks for code
        text = re.sub(r"`(.*?)`", r"\1", text)

        return text

    def get_session_id_sync(self):
        """Generate a unique session ID if not provided (synchronous version)"""
        session_config = {
            "kb_id": self.kb_id[0] if len(self.kb_id) > 0 else "",
            "owner_id": self.owner_id,
            "config": self.config,
            "system_prompt": self.system_prompt,
        }
        try:
            response = requests.post(
                self.llm_config_uri, json=session_config, timeout=10
            )
            response.raise_for_status()
            response_data = response.json()
            return response_data.get("session_id")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to configure session: {e}")
            return None

    async def connect_servers(self):
        """Connect to WebSocket servers"""
        try:
            # Get session ID synchronously
            self.session_id = self.get_session_id_sync()
            # Create URIs for different services
            self.vad_uri = f"ws://{self.yaml_config['vad']['host']}:{self.yaml_config['vad']['port']}/{self.yaml_config['vad']['version']}/{self.yaml_config['vad']['endpoint']}"
            self.stt_uri = f"ws://{self.yaml_config['stt']['host']}:{self.yaml_config['stt']['port']}/{self.yaml_config['stt']['version']}/{self.yaml_config['stt']['endpoint']}"
            self.rag_uri = f"ws://{self.yaml_config['db']['host']}:{self.yaml_config['db']['port']}/ws/{self.yaml_config['db']['endpoint']}/search/{self.owner_id}"
            self.tts_uri = f"ws://{self.yaml_config['tts']['host']}:{self.yaml_config['tts']['port']}"
            headers = {"api-key": self.yaml_config["API_KEY"]}
            self.llm_uri = (
                f"ws://{self.yaml_config['llm']['host']}:{self.yaml_config['llm']['port']}/{self.yaml_config['llm']['version']}/{self.yaml_config['llm']['endpoint']}/{self.owner_id}/{self.session_id}"
                if self.session_id
                else None
            )
            print(f"Connecting to servers")
            self.websockets["vad"] = await websockets.connect(
                self.vad_uri, ping_interval=None
            )
            print(f"Connected to servers VAD")
            self.websockets["stt"] = await websockets.connect(
                self.stt_uri, ping_interval=None
            )
            print(f"Connected to servers STT")
            self.websockets["rag"] = await websockets.connect(
                self.rag_uri, ping_interval=None, extra_headers=headers
            )
            print(f"Connected to servers RAG")
            if self.session_id:
                self.websockets["llm"] = await websockets.connect(
                    self.llm_uri, ping_interval=None
                )
                print(f"Connected to servers LLM")
            else:
                logger.error("Cannot connect to LLM server: No session_id available")

            self.websockets["tts"] = await websockets.connect(
                self.tts_uri, ping_interval=None
            )
            print(f"Connected to servers TTS")
            self.connected = True
            logger.info("✅ Connected to servers")
        except Exception as e:
            logger.error(f"Failed to connect to servers: {e}")
            raise

    async def send_audio_chunk(self, audio_b64):
        """Send a single audio chunk to the VAD server"""
        if not self.connected or "vad" not in self.websockets:
            raise Exception("VAD WebSocket not connected")
        try:
            request = {"type": "input_audio_buffer.append", "audio": audio_b64}
            await self.websockets["vad"].send(json.dumps(request))
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            raise
        except Exception as e:
            logger.error(f"Error sending audio chunk to VAD: {e}")
            raise

    async def send_to_stt(self):
        """Send audio chunk to the STT server"""
        if not self.connected or "stt" not in self.websockets:
            raise Exception("STT WebSocket not connected")
        try:
            # Get sample rate from the speech_segment message
            self.sample_rate = self.yaml_config["vad"]["sample_rate"]

            # The audio is already base64-encoded, we need to decode it
            audio_bytes = base64.b64decode(self.last_chunk_audio)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

            request = {
                "type": "audio.append",
                "sample_rate": self.sample_rate,
                "audio": audio_int16.tolist(),
            }
            await self.websockets["stt"].send(json.dumps(request))
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            raise
        except Exception as e:
            logger.error(f"Error sending audio chunk to STT: {e}")
            raise

    async def send_to_rag(self):
        """Send query to the RAG server"""
        if not self.connected or "rag" not in self.websockets:
            raise Exception("RAG WebSocket not connected")
        try:
            request = {
                "query_text": self.last_transcript,
                "kb_id": self.kb_id,
                "limit": self.limit,
            }
            await self.websockets["rag"].send(json.dumps(request))
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            raise
        except Exception as e:
            logger.error(f"Error sending query to RAG: {e}")
            raise

    async def send_to_llm(self):
        """Send data to the LLM server"""
        if not self.connected or "llm" not in self.websockets:
            raise Exception("LLM WebSocket not connected")
        try:
            request = {
                "owner_id": self.owner_id,
                "user_input": self.last_transcript,
                "retrieved_data": self.context,
                "interrupt": self.interrupt_requested,
            }
            await self.websockets["llm"].send(json.dumps(request))
            # Reset interrupt flag after sending
            if self.interrupt_requested:
                self.interrupt_requested = False
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            raise
        except Exception as e:
            logger.error(f"Error sending data to LLM: {e}")
            raise

    async def send_to_tts(self):
        """Send a single audio chunk to the TTS server"""
        if not self.connected or "tts" not in self.websockets:
            raise Exception("TTS WebSocket not connected")
        try:
            await self.websockets["tts"].send(self.llm_response)
        except websockets.exceptions.ConnectionClosed:
            self.connected = False
            raise
        except Exception as e:
            logger.error(f"Error sending audio chunk to TTS: {e}")
            raise

    async def start_receivers(self):
        """Start all message receivers in background"""
        if not self.connected:
            return []

        # Create tasks for all receivers
        self.receiver_tasks = [
            asyncio.create_task(self.message_receiver_vad()),
            asyncio.create_task(self.message_receiver_stt()),
            asyncio.create_task(self.message_receiver_rag()),
            asyncio.create_task(self.message_receiver_llm()),
            asyncio.create_task(self.message_receiver_tts()),
        ]

        return self.receiver_tasks

    async def message_receiver_vad(self):
        """Listens for messages from the VAD server."""
        print("👂 Listening for messages from the VAD server...")
        try:
            while self.connected and "vad" in self.websockets:
                try:
                    message_str = await asyncio.wait_for(
                        self.websockets["vad"].recv(), timeout=1.0
                    )
                    message = json.loads(message_str)
                    msg_type = message.get("type")
                    if msg_type == "input_audio_buffer.speech_started":
                        print("SERVER DETECTED: 🟢 Speech Started")
                        # await self.interrupt_llm_response()
                        # await self.interrupt_tts_response()  # Interrupt TTS
                    elif msg_type == "input_audio_buffer.speech_stopped":
                        print("SERVER DETECTED: 🔴 Speech Stopped")
                    elif msg_type == "error":
                        print(f"VAD SERVER ERROR: {message.get('message')}")
                    elif msg_type == "speech_segment":
                        print(f"speech_segment 🟢🟢🟢")
                        self.last_chunk_audio = message.get("audio")
                        self.interrupt_requested = False
                        try:
                            await self.send_to_stt()
                        except Exception as e:
                            print(f"Failed to send audio to STT service: {e}")
                    else:
                        print(f"Received unknown VAD message: {message}")
                except asyncio.TimeoutError:
                    # Timeout is normal, just continue listening
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break
        except Exception as e:
            print(f"An error occurred in VAD receiver: {e}")
        finally:
            print("VAD receiver task finished.")

    async def message_receiver_stt(self):
        """Listens for messages from the STT server."""
        try:
            while self.connected and "stt" in self.websockets:
                try:
                    message_str = await asyncio.wait_for(
                        self.websockets["stt"].recv(), timeout=1.0
                    )
                    message = json.loads(message_str)

                    msg_type = message.get("type")
                    if msg_type == "response.audio_transcript.done":
                        self.last_transcript = message.get("transcript")
                        print(f"✒️ Transcription: {self.last_transcript}")
                        if (
                            "bye" in self.last_transcript.lower()
                            or "goodbye" in self.last_transcript.lower()
                        ):
                            print("#" * 50)
                            # self.end_call = True
                            await self.close()
                        await self.send_to_rag()
                    elif msg_type == "error":
                        print(f"STT SERVER ERROR: {message.get('message')}")
                    else:
                        print(f"Received unknown STT message: {message}")
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break
        except Exception as e:
            print(f"An error occurred in STT receiver: {e}")
        finally:
            print("STT receiver task finished.")

    async def message_receiver_rag(self):
        """Listens for messages from the RAG server."""
        try:
            while self.connected and "rag" in self.websockets:
                try:
                    message_str = await asyncio.wait_for(
                        self.websockets["rag"].recv(), timeout=1.0
                    )
                    message = json.loads(message_str)
                    status = message.get("status")
                    if status == "success":
                        try:
                            results = message.get("results", [])
                            if message.get("status") == "success":
                                context_parts = []
                                for i, result in enumerate(results, 1):
                                    context_parts.append(
                                        f"Context {i} (relevance: {result['score']:.2f}): {result['content']}"
                                    )
                                self.context = "\n".join(context_parts)
                                print("🔍 Searching in DB context: ", self.context)
                                await self.send_to_llm()
                            else:
                                print("No results found.")
                        except Exception as e:
                            print(f"Error processing search results: {e}")
                    elif status == "error":
                        print(f"RAG SERVER ERROR: {message.get('message')}")
                    else:
                        print(f"Received unknown RAG message: {message}")
                except asyncio.TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break
        except Exception as e:
            print(f"An error occurred in RAG receiver: {e}")
        finally:
            print("RAG receiver task finished.")

    async def message_receiver_llm(self):
        """Listens for messages from the LLM server."""
        try:
            while self.connected and "llm" in self.websockets:
                try:
                    message_str = await asyncio.wait_for(
                        self.websockets["llm"].recv(), timeout=1.0
                    )
                    message = json.loads(message_str)
                    if message.get("type") == "chunk" and not self.interrupt_requested:
                        if (
                            "bye" in message["content"].lower()
                            or "goodbye" in message["content"].lower()
                        ):
                            print("#" * 50)
                            self.end_call = True
                            # continue
                        chunk = message["content"]
                        cleaned_chunk = self.clean_markdown_formatting(chunk)
                        # Send to TTS immediately
                        await self.websockets["tts"].send(cleaned_chunk)

                    elif message.get("type") == "response_complete":
                        # print('#' * 50)
                        # Signal TTS that text generation is complete
                        # await self.websockets["tts"].send("END_OF_TEXT")
                        if self.end_call:
                            self.end_call = False
                            await self.close()

                        print("LLM response complete")

                    elif "error" in message:
                        print(f"LLM SERVER ERROR: {message.get('error')}")

                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            print(f"An error occurred in LLM receiver: {e}")

    async def message_receiver_tts(self):
        """Listens for audio from the TTS server."""
        try:
            audio_complete = False
            while self.connected and "tts" in self.websockets and not audio_complete:
                try:
                    message = await self.websockets["tts"].recv()

                    if isinstance(message, bytes):
                        # Process audio chunk
                        audio_chunk_b64 = base64.b64encode(message).decode("utf-8")
                        tts_response = {
                            "type": "response.audio.delta",
                            "delta": audio_chunk_b64,
                            "item_id": str(uuid.uuid4()),
                        }
                        self.formatted_audio_responses.append(tts_response)

                    elif message == "AUDIO_COMPLETE":
                        # TTS signals all audio for the current text is done
                        audio_complete = True
                        self.tts_responses.append(
                            {
                                "type": "response.done",
                                "item_id": self.current_item_id,
                            }
                        )
                        break

                except asyncio.TimeoutError:
                    continue
        except Exception as e:
            print(f"An error occurred in TTS receiver: {e}")

    async def close(self):
        """Close all WebSocket connections and cancel tasks"""
        print("Closing all connections...")

        # Cancel all receiver tasks
        for task in self.receiver_tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close all websocket connections
        for name, ws in self.websockets.items():
            if ws:
                await ws.close()
                print(f"Closed {name} connection")

        self.connected = False
        print("All connections closed.")

    async def init_call(self, message):
        self.last_transcript = message
        await self.send_to_llm()

    async def interrupt_llm_response(self):
        """Send an interruption message to the LLM server"""
        if not self.connected or "llm" not in self.websockets:
            return

        try:
            interruption_message = {
                "owner_id": self.owner_id,
                "user_input": ".",  # Empty since we're interrupting
                "retrieved_data": "",
                "interrupt": True,  # Signal interruption
            }
            await self.websockets["llm"].send(json.dumps(interruption_message))
            logger.info("Interruption signal sent to LLM server")
        except Exception as e:
            logger.error(f"Error sending interruption to LLM: {e}")

    async def interrupt_tts_response(self):
        """Send interruption command to TTS server and handle cleanup"""
        if not self.connected or "tts" not in self.websockets:
            return

        try:
            # Send interruption command
            await self.websockets["tts"].send("INTERRUPT_TTS")
            logger.info("Interruption command sent to TTS server")

            # Clear local buffers
            await self.interrupt_current_response()

        except Exception as e:
            logger.error(f"Error sending interruption to TTS: {e}")

    async def interrupt_current_response(self):
        """Interrupt the current response pipeline"""
        # Clear all buffers and reset state
        self.formatted_audio_responses.clear()
        self.tts_responses.clear()
        self.audio_output = bytearray()
        self.llm_response = ""
        self.context = ""
        self.last_transcript = ""
        print("✅ Response interrupted and buffers cleared")
        self.is_responding = False
        self.interrupt_requested = True
