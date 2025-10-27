import asyncio
import audioop
import base64
import json

# NEW: Imports for audio logging

# from faster_whisper import WhisperModel # REMOVED
import urllib.parse
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp
from call_agent import CallAgent
from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect

# Import the logger
from logging_config import get_logger
from twilio.twiml.voice_response import Connect, VoiceResponse
from utils import (  # Ensure truncated_json_dumps is available in utils
    get_env_variable,
    truncated_json_dumps,
)


# Initialize the logger for this module
logger = get_logger(__name__)  # e.g., 'call_agent_vexu.development'

ENDPOINT_DEV = get_env_variable("ENDPOINT_DEV")
MODEL_DEV = get_env_variable("MODEL_DEV")
TEMPERATURE_DEV = get_env_variable("TEMPERATURE_DEV", var_type=float)

# REMOVED WHISPER_MODEL_DEV, WHISPER_QUANTIZE_DEV, WHISPER_BEAM_DEV
DEVICE_DEV = get_env_variable(
    "DEVICE_DEV"
)  # Keep DEVICE_DEV if used for VAD parameters
SHOW_TIMING_MATH_DEV = get_env_variable("SHOW_TIMING_MATH_DEV", var_type=bool)

OPENAI_API_KEY_DEV = get_env_variable(
    "OPENAI_API_KEY_DEV"
)  # Kept for general OpenAI API usage if any, but not for local Realtime WS
SYSTEM_MESSAGE_TEMPLATE_DEV = get_env_variable("SYSTEM_MESSAGE_TEMPLATE_DEV")


VOICE_DEV = get_env_variable("VOICE_DEV")

OPENAI_API_BASE_URL_DEV = get_env_variable("OPENAI_API_BASE_URL_DEV")
OPENAI_REALTIME_WS_BASE_URL_DEV = get_env_variable(
    "OPENAI_REALTIME_WS_BASE_URL_DEV"
)  # NEW: Load the local WebSocket URL
OPENAI_SUMMARY_MODEL_DEV = get_env_variable("OPENAI_SUMMARY_MODEL_DEV")
OPENAI_SUMMARY_TEMPERATURE_DEV = get_env_variable(
    "OPENAI_SUMMARY_TEMPERATURE_DEV", var_type=float
)
OPENAI_SUMMARY_PROMPT_TEMPLATE_DEV = get_env_variable(
    "OPENAI_SUMMARY_PROMPT_TEMPLATE_DEV"
)

# NEW: Urgency Check Configuration
OPENAI_URGENCY_MODEL_DEV = get_env_variable(
    "OPENAI_URGENCY_MODEL_DEV"
)  # Use qwen-chat as default, or a specific one
OPENAI_URGENCY_TEMPERATURE_DEV = get_env_variable(
    "OPENAI_URGENCY_TEMPERATURE_DEV", var_type=float
)  # Low temperature for classification
OPENAI_URGENCY_PROMPT_TEMPLATE_DEV = get_env_variable(
    "OPENAI_URGENCY_PROMPT_TEMPLATE_DEV"
)


VAD_THRESHOLD_DEV = get_env_variable("VAD_THRESHOLD_DEV", var_type=float)
VAD_MIN_SILENCE_DEV = get_env_variable("VAD_MIN_SILENCE_DEV", var_type=int)
VAD_MIN_SPEECH_DEV = get_env_variable("VAD_MIN_SPEECH_DEV", var_type=int)
VAD_SPEECH_PAD_DEV = get_env_variable("VAD_SPEECH_PAD_DEV", var_type=int)

# NEW: Padding for audio sent to Vexu
VEXU_PRE_SPEECH_PAD_MS = 700  # Add 700ms of audio before VAD start for Vexu

# Define Hangup Keywords (Formal, one per language)
ENGLISH_FORMAL_BYE_KEYWORD = "goodbye"
ARABIC_FORMAL_BYE_KEYWORD = "مع السلامة"  # "ma'a as-salama" - (with peace/safety)

# Normalize keywords to lowercase for case-insensitive matching
HANGUP_KEYWORDS = [
    ENGLISH_FORMAL_BYE_KEYWORD.lower(),
    ARABIC_FORMAL_BYE_KEYWORD.lower(),  # Arabic script itself doesn't have case, but good practice if transliteration was used.
]

router = APIRouter(prefix=ENDPOINT_DEV, tags=["development"])
vad_parameters = {
    "threshold": VAD_THRESHOLD_DEV,  # Speech activity threshold (0.0 to 1.0)
    "min_silence_duration_ms": VAD_MIN_SILENCE_DEV,  # Minimum silence duration in ms to consider as a split point
    "min_speech_duration_ms": VAD_MIN_SPEECH_DEV,  # Minimum duration of a speech segment in ms
    "speech_pad_ms": VAD_SPEECH_PAD_DEV,  # Pad speech segments with silence at the beginning and end (in ms)
    # "window_size_samples": 1024,      # Size of audio chunks processed by VAD (512, 1024, 1536 for Silero VAD)
    # "max_speech_duration_s": float("inf") # Maximum duration of a speech segment in seconds
}


LOG_EVENT_TYPES = [
    "error",
    "response.content.done",
    "rate_limits.updated",
    "response.done",
    "input_audio_buffer.committed",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.speech_started",
    "session.created",
]

call_buffer = {}

# --- NEW: Audio Logging Configuration and Class ---
SAVE_AUDIO_DEV = True
AUDIO_LOG_DIR_DEV = "./call_audio_logs_dev"


class CallAudioLogger:
    def __init__(self, call_sid: str, base_log_dir: str):
        self.call_sid = call_sid
        self.log_dir = Path(base_log_dir) / call_sid
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._twilio_input_buffer = bytearray()  # Stores ulaw from Twilio
        self._omni_input_buffer = bytearray()  # Stores PCM sent to Omni
        self._agent_output_buffer = bytearray()  # Stores PCM from Omni (agent's speech)

        logger.info(
            f"Initialized CallAudioLogger for call {call_sid}. Log directory: {self.log_dir}",
            extra={"call_sid": call_sid, "log_dir": str(self.log_dir)},
        )

    async def _write_full_audio_file(self, file_path: Path, audio_bytes: bytes):
        if not audio_bytes:
            logger.debug(
                f"No audio data to save for {file_path}. Skipping file write.",
                extra={"call_sid": self.call_sid, "file_path": str(file_path)},
            )
            return
        try:
            # Use asyncio.to_thread to run blocking file I/O in a separate thread
            await asyncio.to_thread(file_path.write_bytes, audio_bytes)
            logger.info(
                f"Saved aggregated audio file to {file_path} (size: {len(audio_bytes)} bytes).",
                extra={
                    "call_sid": self.call_sid,
                    "file_path": str(file_path),
                    "total_size_bytes": len(audio_bytes),
                },
            )
        except Exception as e:
            logger.error(
                f"Failed to save aggregated audio file to {file_path}: {e}",
                exc_info=True,
                extra={"call_sid": self.call_sid, "file_path": str(file_path)},
            )

    async def log_twilio_input(self, audio_bytes_ulaw: bytes):
        if not SAVE_AUDIO_DEV:
            return
        self._twilio_input_buffer.extend(audio_bytes_ulaw)
        # logger.debug(f"Appended {len(audio_bytes_ulaw)} ulaw bytes to Twilio input buffer. Current size: {len(self._twilio_input_buffer)}", extra={"call_sid": self.call_sid})

    async def log_omni_input(self, audio_bytes_pcm: bytes):
        if not SAVE_AUDIO_DEV:
            return
        self._omni_input_buffer.extend(audio_bytes_pcm)
        # logger.debug(f"Appended {len(audio_bytes_pcm)} PCM bytes to Omni input buffer. Current size: {len(self._omni_input_buffer)}", extra={"call_sid": self.call_sid})

    async def log_agent_output(self, audio_bytes_pcm: bytes):
        if not SAVE_AUDIO_DEV:
            return
        self._agent_output_buffer.extend(audio_bytes_pcm)
        # logger.debug(f"Appended {len(audio_bytes_pcm)} PCM bytes to Agent output buffer. Current size: {len(self._agent_output_buffer)}", extra={"call_sid": self.call_sid})

    async def finalize_and_save(self):
        if not SAVE_AUDIO_DEV:
            return
        logger.info(
            f"Finalizing and saving all audio buffers for call {self.call_sid}.",
            extra={"call_sid": self.call_sid},
        )

        # Save Twilio input (ulaw)
        if self._twilio_input_buffer:
            file_name = "twilio_input_full_call.ulaw"
            await self._write_full_audio_file(
                self.log_dir / file_name, bytes(self._twilio_input_buffer)
            )
            self._twilio_input_buffer.clear()

        # Save Omni input (PCM)
        if self._omni_input_buffer:
            file_name = "omni_input_full_call.pcm"
            await self._write_full_audio_file(
                self.log_dir / file_name, bytes(self._omni_input_buffer)
            )
            self._omni_input_buffer.clear()

        # Save Agent output (PCM)
        if self._agent_output_buffer:
            file_name = "agent_output_full_call.pcm"
            await self._write_full_audio_file(
                self.log_dir / file_name, bytes(self._agent_output_buffer)
            )
            self._agent_output_buffer.clear()


# --- End NEW: Audio Logging Configuration and Class ---


@router.get("/", response_class=JSONResponse)
async def index_page():
    return JSONResponse(
        content=truncated_json_dumps(
            {"message": "Twilio Development Media Stream Server is running!"}
        ),
        media_type="application/json",
    )


@router.api_route("/answer", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    response = VoiceResponse()
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f"wss://{host}{ENDPOINT_DEV}/media-stream")
    response.append(connect)
    form = await request.form()
    twilio_reciever_number = form.get("Called")
    twilio_call_sid = form.get("CallSid")
    caller_phone = form.get("From")

    # Define log_context for this function scope
    # request_id might be added by middleware; fetch it if available
    request_id_header = request.headers.get("X-Request-ID")
    log_context_call = {
        "twilio_call_sid": twilio_call_sid,
        "caller_phone": caller_phone,
        "twilio_reciever_number": twilio_reciever_number,
        "request_id": request_id_header,
    }
    if not twilio_call_sid:
        logger.critical(
            "CRITICAL ERROR: CallSid is missing from Twilio request (DEV).",
            extra={
                "form_params_preview": str(form)[:200],
                "request_id": request_id_header,
            },
        )
        raise ValueError("CallSid is missing from Twilio request.")

    # Update log_context_call once twilio_call_sid is confirmed to exist
    log_context_call["twilio_call_sid"] = twilio_call_sid

    if not twilio_reciever_number:
        logger.critical(
            "CRITICAL ERROR: 'Called' number (twilio_reciever_number) not available from Twilio (DEV). Cannot fetch Vexu user details.",
            extra=log_context_call,
        )
        raise ValueError(
            f"twilio_reciever_number is required to fetch Vexu user details."
        )

    logger.info(
        f"Incoming call to Vexu number (DEV): {twilio_reciever_number}. Fetching user details.",
        extra=log_context_call,
    )

    call_buffer[twilio_call_sid] = {
        "caller_phone": caller_phone,
        "agent_initiated_hangup": False,
        "twilio_reciever_number": twilio_reciever_number,
        # NEW: Audio buffering and VAD state for caller's turn
        "current_caller_turn_ulaw_buffer": bytearray(),
        "first_media_timestamp_in_turn": None,
        "vad_speech_start_twilio_timestamp": None,
        "vad_speech_end_twilio_timestamp": None,
        "last_caller_audio_for_vexu": b"",  # Stores PCM 16-bit for Vexu after trimming
        "latest_media_timestamp_twilio": 0,  # Tracks the latest Twilio media timestamp
    }
    return HTMLResponse(content=str(response), media_type="application/xml")


@router.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    ws_session_id = str(uuid.uuid4())
    log_context = {  # Base context for this WebSocket connection
        "ws_session_id": ws_session_id,
        "client_host": websocket.client.host,
        "client_port": websocket.client.port,
        "call_sid": None,  # Will be updated
        "stream_sid": None,  # Will be updated
    }
    logger.info("WebSocket client connected (DEV).", extra=log_context)
    gpt_output_buffer = (
        b""  # This buffer will now store PCM 16-bit audio from OpenAI (agent's speech)
    )
    await websocket.accept()
    logger.info("WebSocket connection accepted by server (DEV).", extra=log_context)
    # caller_audio_buffer = bytearray() # REMOVED: Now managed per-call in call_buffer

    call_sid = None
    stream_sid = None
    audio_logger_instance: Optional[CallAudioLogger] = (
        None  # Declare here to be accessible in finally
    )
    try:
        logger.info(
            f"Attempting to connect to local Realtime API WebSocket (DEV) at {OPENAI_REALTIME_WS_BASE_URL_DEV}.",
            extra=log_context,
        )

        call_agent = CallAgent(owner_id="", system_prompt="", yaml_path=yaml_path)
        call_agent_ready = asyncio.Event()
        logger.info(
            "Local Realtime API WebSocket connection established (DEV).",
            extra=log_context,
        )

        # latest_media_timestamp = 0 # REMOVED: Now in call_buffer[call_sid]["latest_media_timestamp_twilio"]
        last_assistant_item = None
        mark_queue = []
        response_start_timestamp_twilio = None

        async def receive_from_twilio():
            nonlocal call_sid, stream_sid, response_start_timestamp_twilio, last_assistant_item, mark_queue, log_context, audio_logger_instance
            session_initialized_for_this_call = False
            try:
                logger.info(
                    "Starting to receive messages from Twilio (DEV).",
                    extra=log_context,
                )
                async for message in websocket.iter_text():
                    data = json.loads(message)
                    if data["event"] == "start":

                        stream_sid = data["start"]["streamSid"]
                        call_sid = data["start"]["callSid"]
                        twilio_reciever_number = call_buffer[call_sid][
                            "twilio_reciever_number"
                        ]
                        log_context["stream_sid"] = stream_sid
                        log_context["call_sid"] = call_sid
                        logger.info(
                            "Incoming Twilio media stream started (DEV).",
                            extra=log_context,
                        )
                        response_start_timestamp_twilio = None
                        # latest_media_timestamp = 0 # REMOVED
                        last_assistant_item = None

                        if not call_sid or call_sid not in call_buffer:
                            logger.error(
                                "call_sid not found in call_buffer (DEV). Cannot initialize local Realtime API session.",
                                extra=log_context,
                            )
                            # if openai_ws.open:
                            #     await openai_ws.close()
                            await websocket.close(
                                code=1011, reason="Call SID not found in buffer"
                            )
                            return

                        call_data = call_buffer[
                            call_sid
                        ]  # Get call_data for this specific call_sid
                        # NEW: Reset audio state for a new stream/turn
                        call_data["current_caller_turn_ulaw_buffer"] = bytearray()
                        call_data["first_media_timestamp_in_turn"] = None
                        call_data["vad_speech_start_twilio_timestamp"] = None
                        call_data["vad_speech_end_twilio_timestamp"] = None
                        call_data["last_caller_audio_for_vexu"] = b""
                        call_data["latest_media_timestamp_twilio"] = (
                            0  # Reset latest timestamp
                        )
                        logger.debug(
                            "Call data audio state reset for new stream (DEV).",
                            extra=log_context,
                        )

                        if "owner_name" not in call_data or not call_data["owner_name"]:
                            logger.error(
                                f"Owner name not found or empty in call_buffer for call_sid {call_sid} (DEV). Cannot initialize local Realtime API session.",
                                extra=log_context,
                            )
                            # if openai_ws.open:
                            #     await openai_ws.close()
                            await websocket.close(
                                code=1011,
                                reason="Owner name missing in call buffer",
                            )
                            return

                        # --- NEW: Initialize CallAudioLogger and store in call_buffer ---
                        if (
                            SAVE_AUDIO_DEV and "audio_logger" not in call_data
                        ):  # Only initialize once per call
                            call_data["audio_logger"] = CallAudioLogger(
                                call_sid, AUDIO_LOG_DIR_DEV
                            )
                            audio_logger_instance = call_data[
                                "audio_logger"
                            ]  # Store reference
                        # --- End NEW ---
                        # @Borhan: the owner_name is needed for the prompt, we must get it from the dashboard
                        owner_name_for_prompt = call_data["owner_name"]
                        caller_name_from_buffer = call_data.get("caller_name")
                        caller_name_greeting_segment = (
                            caller_name_from_buffer
                            if caller_name_from_buffer
                            else "Unknown"
                        )
                        if not session_initialized_for_this_call:
                            # @Borhan: add the owner_name HERE!
                            dynamic_system_prompt = SYSTEM_MESSAGE_TEMPLATE_DEV.format(
                                owner_name=owner_name_for_prompt,
                                owner_subject_pronouns="he",
                                owner_object_pronouns="him",
                                owner_possessive_adjectives="his",
                            )
                            logger.info(
                                f"Initializing local Realtime API session (DEV) with dynamic prompt for owner: '{owner_name_for_prompt}'. Caller info: '{caller_name_greeting_segment}'.",
                                extra=log_context,
                            )

                            # Connect to all servers
                            call_agent.owner_id = twilio_reciever_number
                            call_agent.system_prompt = dynamic_system_prompt
                            # @Borhan Dashboard kb id
                            call_agent.kb_id = ["kb+12345952496_en"]
                            await call_agent.connect_servers()

                            # Start all receivers
                            receiver_tasks = await call_agent.start_receivers()
                            call_agent_ready.set()

                            # initialize call
                            await call_agent.init_call(
                                f"The caller name is {caller_name_from_buffer}, Please greet the caller based on your instructions"
                            )

                            logger.info(
                                "Local Realtime API session initialized and initial item sent (DEV).",
                                extra=log_context,
                            )
                            session_initialized_for_this_call = True
                        else:
                            logger.debug(
                                "Local Realtime API Session already initialized for this call (DEV).",
                                extra=log_context,
                            )

                    # elif data["event"] == "media" and openai_ws.open:
                    elif data["event"] == "media":
                        media_payload = data["media"][
                            "payload"
                        ]  # This is base64 encoded ulaw from Twilio
                        audio_bytes_ulaw = base64.b64decode(
                            media_payload
                        )  # Decode to raw ulaw

                        # NEW: Update latest_media_timestamp_twilio in call_data
                        call_data = call_buffer.get(call_sid)
                        if call_data:
                            call_data["latest_media_timestamp_twilio"] = int(
                                data["media"]["timestamp"]
                            )
                            if call_data["first_media_timestamp_in_turn"] is None:
                                # Approximate start of this first chunk in the turn
                                call_data["first_media_timestamp_in_turn"] = call_data[
                                    "latest_media_timestamp_twilio"
                                ] - (len(audio_bytes_ulaw) * 1000 // 8000)
                                logger.debug(
                                    f"First media timestamp in turn set to {call_data['first_media_timestamp_in_turn']} (DEV).",
                                    extra=log_context,
                                )

                            # --- NEW: Debugging and Saving Twilio Input ---
                            logger.debug(
                                f"Twilio Input (DEV): Received {len(audio_bytes_ulaw)} bytes of ulaw audio.",
                                extra={
                                    **log_context,
                                    "audio_source": "twilio_input",
                                    "ulaw_bytes_len": len(audio_bytes_ulaw),
                                },
                            )
                            if audio_logger_instance:  # Use the instance directly
                                asyncio.create_task(
                                    audio_logger_instance.log_twilio_input(
                                        audio_bytes_ulaw
                                    )
                                )
                            # --- End NEW ---

                            # NEW: Extend the current_caller_turn_ulaw_buffer
                            call_data["current_caller_turn_ulaw_buffer"].extend(
                                audio_bytes_ulaw
                            )

                            # Convert ulaw to PCM 16-bit for OpenAI
                            pcm_audio_from_twilio = audioop.ulaw2lin(
                                audio_bytes_ulaw, 2
                            )  # 2 means 16-bit output

                            # --- NEW: Debugging and Saving Omni Input ---
                            logger.debug(
                                f"Omni Input (DEV): Converted {len(audio_bytes_ulaw)} ulaw bytes to {len(pcm_audio_from_twilio)} PCM bytes for Omni.",
                                extra={
                                    **log_context,
                                    "audio_source": "omni_input",
                                    "ulaw_bytes_len": len(audio_bytes_ulaw),
                                    "pcm_bytes_len": len(pcm_audio_from_twilio),
                                },
                            )
                            if audio_logger_instance:  # Use the instance directly
                                asyncio.create_task(
                                    audio_logger_instance.log_omni_input(
                                        pcm_audio_from_twilio
                                    )
                                )
                            # --- End NEW ---

                            # Base64 encode PCM for OpenAI
                            pcm_payload_for_openai = base64.b64encode(
                                pcm_audio_from_twilio
                            ).decode("utf-8")

                            # await openai_ws.send(json.dumps(audio_append))
                            await call_agent.send_audio_chunk(pcm_payload_for_openai)
                        else:
                            logger.warning(
                                "Media received but call_data not found in buffer (DEV). Skipping audio processing.",
                                extra=log_context,
                            )

                    elif data["event"] == "mark":
                        received_mark_name = data["mark"].get("name")
                        if mark_queue:
                            popped_mark = mark_queue.pop(0)
                            logger.debug(
                                f"Mark '{received_mark_name}' (expected '{popped_mark}') processed by Twilio (DEV). Mark queue size: {len(mark_queue)}.",
                                extra=log_context,
                            )
                        else:
                            logger.warning(
                                f"Received mark '{received_mark_name}' from Twilio, but local mark_queue was empty (DEV).",
                                extra=log_context,
                            )

                        buffer_entry = call_buffer.get(call_sid)
                        if buffer_entry:
                            is_hangup_flagged = buffer_entry.get(
                                "agent_initiated_hangup", False
                            )
                            if is_hangup_flagged and not mark_queue:
                                logger.info(
                                    "Agent hangup flagged and mark queue empty (DEV). Finalizing call after ensuring audio playback.",
                                    extra=log_context,
                                )

                                if call_agent.open:
                                    try:
                                        await call_agent.close(
                                            code=1000,
                                            reason="Agent hangup sequence finalized after audio playback",
                                        )
                                        logger.info(
                                            "Local Realtime API WebSocket closed by receive_from_twilio after hangup sequence (DEV).",
                                            extra=log_context,
                                        )
                                    except Exception as e_ows_close:
                                        logger.error(
                                            "Exception closing Local Realtime API WebSocket in receive_from_twilio (DEV).",
                                            exc_info=True,
                                            extra=log_context,
                                        )

                                try:
                                    await websocket.close(
                                        code=1000,
                                        reason="Agent hangup sequence finalized after audio playback",
                                    )
                                    logger.info(
                                        "Twilio WebSocket closed by receive_from_twilio after hangup sequence (DEV).",
                                        exc_info=True,
                                        extra=log_context,
                                    )
                                except Exception as e_ws_close:
                                    logger.error(
                                        "Exception closing Twilio WebSocket in receive_from_twilio (DEV).",
                                        exc_info=True,
                                        extra=log_context,
                                    )
                                return

                    elif data["event"] == "stop":
                        # await call_agent.stop()
                        await call_agent.close()
            except WebSocketDisconnect:
                logger.info(
                    "Client disconnected - Twilio WebSocket closed (DEV).",
                    extra=log_context,
                )
                # if openai_ws.open:
                #     logger.info(
                #         "Closing Local Realtime API WebSocket due to Twilio disconnect (DEV).",
                #         extra=log_context,
                #     )
                #     await openai_ws.close()
                await call_agent.close()

            except Exception as e:
                logger.error(
                    "Unexpected error in receive_from_twilio (DEV).",
                    exc_info=True,
                    extra=log_context,
                )
                # if openai_ws.open:
                #     await openai_ws.close()
                await call_agent.close()
            finally:
                current_vexu_call_id = None
                retrieved_caller_name_for_summary = None

                # --- NEW: Finalize and save audio logs ---
                if audio_logger_instance:
                    await audio_logger_instance.finalize_and_save()
                # --- End NEW ---

                # if openai_ws.open:
                #     await openai_ws.close()

        async def send_to_twilio():
            nonlocal call_sid, stream_sid, last_assistant_item, response_start_timestamp_twilio, gpt_output_buffer, mark_queue, log_context, audio_logger_instance

            try:
                logger.info(
                    "Starting to receive messages from Local Realtime API (DEV).",
                    extra=log_context,
                )
                await call_agent_ready.wait()
                if not call_agent:
                    logger.error(
                        "Call agent still None after ready event", extra=log_context
                    )
                    return
                # Initialize formatted_audio_responses if needed
                if not hasattr(call_agent, "formatted_audio_responses"):
                    logger.warning(
                        "Call agent has no formatted_audio_responses attribute",
                        extra=log_context,
                    )
                    call_agent.formatted_audio_responses = []
                # Initialize interruption tracking if not exists
                if not hasattr(call_agent, "interrupt_requested"):
                    call_agent.interrupt_requested = False
                if not hasattr(call_agent, "is_responding"):
                    call_agent.is_responding = False

                while True:
                    # Wait for responses to be available
                    if not call_agent.formatted_audio_responses:
                        await asyncio.sleep(0.1)  # Short sleep to avoid CPU spinning
                        continue

                    response = call_agent.formatted_audio_responses.pop(0)
                    if response["type"] in LOG_EVENT_TYPES:
                        event_data_log = (
                            truncated_json_dumps(response, max_string_len=150)
                            if response["type"]
                            not in [
                                "response.audio.delta",
                                "input_audio_buffer.append",
                            ]
                            else {
                                "type": response["type"],
                                "details": "suppressed for brevity",
                            }
                        )
                        logger.debug(
                            f"Local Realtime API Event (DEV): {response['type']}.",
                            extra={
                                **log_context,
                                "openai_event_type": response["type"],
                                "openai_event_data": event_data_log,
                            },
                        )

                    if (
                        response.get("type") == "response.audio.delta"
                        and "delta" in response
                    ):
                        # OpenAI sends PCM 16-bit, base64 encoded
                        pcm_bytes_from_openai = base64.b64decode(response["delta"])

                        # --- NEW: Debugging and Saving Agent Output ---
                        logger.debug(
                            f"Agent Output (DEV): Received {len(pcm_bytes_from_openai)} bytes of PCM audio from Omni.",
                            extra={
                                **log_context,
                                "audio_source": "agent_output",
                                "pcm_bytes_len": len(pcm_bytes_from_openai),
                            },
                        )
                        if audio_logger_instance:  # Use the instance directly
                            asyncio.create_task(
                                audio_logger_instance.log_agent_output(
                                    pcm_bytes_from_openai
                                )
                            )

                        # Convert PCM 16-bit to ulaw for Twilio
                        ulaw_bytes_for_twilio = audioop.lin2ulaw(
                            pcm_bytes_from_openai, 2
                        )

                        # Base64 encode ulaw for Twilio
                        audio_payload_for_twilio = base64.b64encode(
                            ulaw_bytes_for_twilio
                        ).decode("utf-8")

                        audio_delta_twilio = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": audio_payload_for_twilio
                            },  # Send ulaw to Twilio
                        }
                        await websocket.send_json(audio_delta_twilio)
                        gpt_output_buffer += (
                            pcm_bytes_from_openai  # Accumulate PCM 16-bit directly
                        )
                        call_data = call_buffer.get(call_sid)
                        if call_data:  # Use latest_media_timestamp from call_data
                            if response_start_timestamp_twilio is None:
                                response_start_timestamp_twilio = call_data[
                                    "latest_media_timestamp_twilio"
                                ]
                        else:
                            logger.warning(
                                "Call data not found for response.audio.delta (DEV). Cannot set response_start_timestamp_twilio.",
                                extra=log_context,
                            )

                        if response.get("item_id"):
                            last_assistant_item = response["item_id"]
                        await send_mark(websocket, stream_sid)

                    elif (
                        response.get("type") == "response.audio_transcript.done"
                        and "transcript" in response
                    ):
                        current_vexu_call_id = call_buffer.get(call_sid, {}).get(
                            "vexu_call_id"
                        )
                        if not current_vexu_call_id:
                            logger.error(
                                "Cannot send transcript to Vexu (DEV): Vexu Call ID missing.",
                                extra=log_context,
                            )

                        # Determine if this transcript is for the agent or the caller
                        # @Borhan: The transcript from both caller and agent will be sent to app
                        # if gpt_output_buffer:  # This is the agent's response
                        #     agent_audio_for_vexu_pcm16_8khz = gpt_output_buffer
                        #     asyncio.create_task(
                        #         post_vexu_message_async(
                        #             vexu_call_id=current_vexu_call_id,
                        #             text=response["transcript"],
                        #             sender="agent",
                        #             audio_pcm16_8khz_bytes=agent_audio_for_vexu_pcm16_8khz,
                        #         )
                        #     )
                        #     logger.info(
                        #         "Sent agent message to Vexu AI (DEV).",
                        #         extra={
                        #             **log_context,
                        #             "vexu_call_id": current_vexu_call_id,
                        #             "transcript_preview": response["transcript"][:100],
                        #         },
                        #     )
                        #     gpt_output_buffer = b""  # Clear agent's audio buffer

                        #     agent_transcript_lower = response["transcript"].lower()
                        #     if (
                        #         call_sid
                        #         and call_sid in call_buffer
                        #         and any(
                        #             keyword in agent_transcript_lower
                        #             for keyword in HANGUP_KEYWORDS
                        #         )
                        #     ):
                        #         logger.info(
                        #             f"Agent said bye keyword (DEV): '{response['transcript']}'. Flagging for hangup after audio playback.",
                        #             extra={
                        #                 **log_context,
                        #                 "transcript": response["transcript"],
                        #             },
                        #         )
                        #         call_buffer[call_sid]["agent_initiated_hangup"] = True
                        # else:  # This is the caller's response (transcribed by server.py)
                        #     # NEW: Retrieve the pre-processed and trimmed audio from call_buffer
                        #     call_data = call_buffer.get(call_sid)
                        #     pcm_audio_caller_for_vexu = b""
                        #     if call_data:
                        #         pcm_audio_caller_for_vexu = call_data.pop(
                        #             "last_caller_audio_for_vexu", b""
                        #         )
                        #         if not pcm_audio_caller_for_vexu:
                        #             logger.warning(
                        #                 "No trimmed caller audio found for Vexu message (DEV). Sending empty audio.",
                        #                 extra=log_context,
                        #             )
                        #     else:
                        #         logger.warning(
                        #             "Call data not found for Vexu message (DEV). Sending empty audio.",
                        #             extra=log_context,
                        #         )

                        #     logger.info(
                        #         "Sent caller message to Vexu (DEV) using server's transcript and trimmed audio.",
                        #         extra={
                        #             **log_context,
                        #             "vexu_call_id": current_vexu_call_id,
                        #             "transcript_preview": response["transcript"][:100],
                        #             "audio_bytes_len": len(pcm_audio_caller_for_vexu),
                        #         },
                        #     )
                        #     # caller_audio_buffer.clear() # REMOVED: Buffer is now cleared in speech_stopped


                    elif response.get("type") == "input_audio_buffer.speech_stopped":

                        logger.info(
                            "Caller speech stopped detected by server (DEV). Processing audio for Vexu.",
                            extra=log_context,
                        )
                        call_data = call_buffer.get(call_sid)
                        if (
                            call_data
                            and call_data["vad_speech_start_twilio_timestamp"]
                            is not None
                            and call_data["first_media_timestamp_in_turn"] is not None
                        ):
                            # Calculate offsets and trim audio
                            start_offset_ms = (
                                call_data["vad_speech_start_twilio_timestamp"]
                                - call_data["first_media_timestamp_in_turn"]
                            )
                            end_offset_ms = (
                                call_data["latest_media_timestamp_twilio"]
                                - call_data["first_media_timestamp_in_turn"]
                            )  # Use latest_media_timestamp from call_data

                            # NEW: Apply pre-speech padding for Vexu
                            padded_start_offset_ms = max(
                                0, start_offset_ms - VEXU_PRE_SPEECH_PAD_MS
                            )

                            # Ensure offsets are non-negative and within buffer bounds
                            # Use padded_start_offset_ms for the start
                            start_byte_offset = int(
                                padded_start_offset_ms * 8000 / 1000
                            )  # 8000 samples/sec, 1 byte/sample for ulaw
                            end_byte_offset = min(
                                len(call_data["current_caller_turn_ulaw_buffer"]),
                                int(end_offset_ms * 8000 / 1000),
                            )

                            trimmed_ulaw_audio_for_vexu = call_data[
                                "current_caller_turn_ulaw_buffer"
                            ][start_byte_offset:end_byte_offset]
                            pcm_audio_caller_for_vexu = audioop.ulaw2lin(
                                bytes(trimmed_ulaw_audio_for_vexu), 2
                            )

                            call_data["last_caller_audio_for_vexu"] = (
                                pcm_audio_caller_for_vexu
                            )
                            logger.info(
                                f"Trimmed caller audio for Vexu (DEV): {len(pcm_audio_caller_for_vexu)} PCM bytes. "
                                f"Original ulaw buffer size: {len(call_data['current_caller_turn_ulaw_buffer'])} bytes. "
                                f"Padded Start offset: {padded_start_offset_ms}ms, End offset: {end_offset_ms}ms.",
                                extra={
                                    **log_context,
                                    "trimmed_audio_len": len(pcm_audio_caller_for_vexu),
                                    "original_ulaw_len": len(
                                        call_data["current_caller_turn_ulaw_buffer"]
                                    ),
                                },
                            )
                        else:
                            logger.warning(
                                "Cannot trim caller audio for Vexu: VAD start/end timestamps or first media timestamp missing (DEV). Sending untrimmed as fallback.",
                                extra=log_context,
                            )
                            # Fallback: if VAD info is missing, send the whole accumulated buffer
                            if (
                                call_data
                                and call_data["current_caller_turn_ulaw_buffer"]
                            ):
                                pcm_audio_caller_for_vexu = audioop.ulaw2lin(
                                    bytes(call_data["current_caller_turn_ulaw_buffer"]),
                                    2,
                                )
                                call_data["last_caller_audio_for_vexu"] = (
                                    pcm_audio_caller_for_vexu
                                )
                                logger.warning(
                                    "Sending untrimmed caller audio to Vexu as fallback (DEV).",
                                    extra=log_context,
                                )
                            else:
                                call_data["last_caller_audio_for_vexu"] = b""
                                logger.warning(
                                    "No caller audio in buffer to send to Vexu (DEV).",
                                    extra=log_context,
                                )

                        # Clear the buffer and reset timestamps for the next turn
                        if call_data:
                            call_data["current_caller_turn_ulaw_buffer"].clear()
                            call_data["first_media_timestamp_in_turn"] = None
                            call_data["vad_speech_start_twilio_timestamp"] = None
                            call_data["vad_speech_end_twilio_timestamp"] = None

                    if response.get("type") == "input_audio_buffer.speech_started":
                        logger.info(
                            "Caller speech started detected (DEV).",
                            extra=log_context,
                        )
                        call_data = call_buffer.get(call_sid)
                        if call_data:
                            call_data["vad_speech_start_twilio_timestamp"] = call_data[
                                "latest_media_timestamp_twilio"
                            ]  # Set the VAD start timestamp
                            logger.debug(
                                f"VAD speech start timestamp set to {call_data['vad_speech_start_twilio_timestamp']} (DEV).",
                                extra=log_context,
                            )
                        if last_assistant_item:
                            await handle_speech_started_event(
                                call_agent, call_sid
                            )  # Pass call_sid

            except Exception as e:
                logger.error(
                    "Error in send_to_twilio (DEV).",
                    exc_info=True,
                    extra=log_context,
                )
            finally:
                logger.debug("send_to_twilio task ended (DEV).", extra=log_context)

        async def handle_speech_started_event(
            openai_ws_param, current_call_sid: str
        ):  # Added current_call_sid
            nonlocal stream_sid, response_start_timestamp_twilio, last_assistant_item, mark_queue, log_context

            call_data = call_buffer.get(current_call_sid)
            if not call_data:
                logger.warning(
                    f"handle_speech_started_event: call_data not found for {current_call_sid} (DEV). Aborting.",
                    extra=log_context,
                )
                return

            current_latest_media_timestamp = call_data[
                "latest_media_timestamp_twilio"
            ]  # Get from call_data

            logger.info(
                "Handling speech started event (interruption) (DEV).",
                extra=log_context,
            )
            if mark_queue and response_start_timestamp_twilio is not None:
                elapsed_time = (
                    current_latest_media_timestamp - response_start_timestamp_twilio
                )
                if SHOW_TIMING_MATH_DEV:
                    logger.debug(
                        f"Calculating elapsed time for truncation (DEV): {current_latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms.",
                        extra=log_context,
                    )

                if last_assistant_item:
                    if SHOW_TIMING_MATH_DEV:
                        logger.debug(
                            f"Truncating Local Realtime API item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms (DEV).",
                            extra={
                                **log_context,
                                "last_assistant_item_id": last_assistant_item,
                                "elapsed_time_ms_truncation": elapsed_time,
                            },
                        )
                    truncate_event = {
                        "type": "conversation.item.truncate",
                        "item_id": last_assistant_item,
                        "content_index": 0,
                        "audio_end_ms": elapsed_time,
                    }
                    await openai_ws_param.send(json.dumps(truncate_event))

                await websocket.send_json({"event": "clear", "streamSid": stream_sid})
                mark_queue.clear()
                last_assistant_item = None
                response_start_timestamp_twilio = None

        async def send_mark(connection, current_stream_sid_param):
            if current_stream_sid_param:
                mark_event = {
                    "event": "mark",
                    "streamSid": current_stream_sid_param,
                    "mark": {"name": "responsePart"},
                }
                await connection.send_json(mark_event)
                mark_queue.append("responsePart")

        logger.debug(
            "Starting to gather Twilio receive and Local Realtime API send tasks (DEV).",
            extra=log_context,
        )
        try:
            await asyncio.gather(receive_from_twilio(), send_to_twilio())
        except Exception as e:
            logger.error(
                "Error in asyncio.gather for Twilio/Local Realtime API tasks (DEV).",
                exc_info=True,
                extra=log_context,
            )
        finally:
            logger.debug(
                "asyncio.gather completed - exiting Local Realtime API WebSocket context (DEV).",
                extra=log_context,
            )

        logger.info(
            "Local Realtime API WebSocket context exited - connection presumed closed (DEV).",
            extra=log_context,
        )
    except Exception as e:
        logger.error(
            "Error in main WebSocket handler (DEV).", exc_info=True, extra=log_context
        )
    finally:
        logger.debug(
            "Entering final cleanup section of handle_media_stream (DEV).",
            extra=log_context,
        )
        if call_sid and call_sid in call_buffer:
            # --- NEW: Finalize and save audio logs before removing from buffer ---
            if "audio_logger" in call_buffer[call_sid]:
                audio_logger_instance = call_buffer[call_sid]["audio_logger"]
                await audio_logger_instance.finalize_and_save()
                del call_buffer[call_sid]["audio_logger"]
                logger.debug(
                    "Audio logger instance removed from call_buffer (DEV).",
                    extra=log_context,
                )
            # --- End NEW ---
            del call_buffer[call_sid]
            logger.debug("Call data removed from call_buffer (DEV).", extra=log_context)
        elif call_sid:
            logger.debug(
                "Call data was not found in call_buffer for removal (DEV) (already removed or error).",
                extra=log_context,
            )
        else:
            logger.debug(
                "No call_sid established, skipping call_buffer cleanup (DEV).",
                extra=log_context,
            )
        logger.debug(
            "WebSocket handler final cleanup completed (DEV).", extra=log_context
        )

    logger.info("WebSocket handler fully completed (DEV).", extra=log_context)
