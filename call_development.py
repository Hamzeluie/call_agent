import asyncio
import audioop
import base64
import json

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

# NEW: Imports for audio logging


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

VEXU_API_BASE_URL_DEV = get_env_variable("VEXU_API_BASE_URL_DEV")
VEXU_API_TOKEN_DEV = get_env_variable("VEXU_API_TOKEN_DEV")
VEXU_CALLER_NAME_CONSTANT_DEV = get_env_variable("VEXU_CALLER_NAME_CONSTANT_DEV")

OPENAI_API_KEY_DEV = get_env_variable(
    "OPENAI_API_KEY_DEV"
)  # Kept for general OpenAI API usage if any, but not for local Realtime WS
SYSTEM_MESSAGE_TEMPLATE_DEV = get_env_variable("SYSTEM_MESSAGE_TEMPLATE_DEV")

VEXU_USERS_API_BASE_URL_DEV = get_env_variable("VEXU_USERS_API_BASE_URL_DEV")
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

# REMOVED WHISPER MODEL INITIALIZATION AND WARMUP
# STT_MODEL = WhisperModel(WHISPER_MODEL_DEV, device=DEVICE_DEV, compute_type=WHISPER_QUANTIZE_DEV)
# logger.info("Warming up the Whisper model (DEV)...")
# for i in range(3):
#     warmup_start_time = time.time()
#     segments, info = STT_MODEL.transcribe("./arabic-warmup-audio.wav", beam_size=WHISPER_BEAM_DEV, language=['ar', 'en'], vad_filter=True, vad_parameters=vad_parameters)
#     warmup_end_time = time.time()
#     logger.info(f"Whisper model warmup iteration {i+1} (DEV) took {warmup_end_time - warmup_start_time:.2f} seconds.")
# logger.info("Whisper model (DEV) warmed up.")
# logger.info(
#     "Whisper warmup (DEV): Detected language '%s' with probability %f",
#     info.language,
#     info.language_probability,
#     extra={"language": info.language, "probability": float(f"{info.language_probability:.4f}")}
# )
# for segment_idx, segment in enumerate(segments):
#     logger.debug(
#         "Whisper warmup (DEV) segment %d: [%.2fs -> %.2fs] %s",
#         segment_idx, segment.start, segment.end, segment.text,
#         extra={"segment_start_s": segment.start, "segment_end_s": segment.end}
#     )

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


async def get_vexu_user_details(phone_number: str):
    """Fetches user details (name and user_id) from Vexu Users API."""
    if not phone_number:
        logger.warning(
            "Vexu Users API (DEV): Cannot fetch user details, phone_number is missing."
        )
        return None

    headers = {"accept": "application/json", "X-AI-Service-Token": VEXU_API_TOKEN_DEV}

    encoded_phone_number = urllib.parse.quote(phone_number)
    url = f"{VEXU_USERS_API_BASE_URL_DEV}{encoded_phone_number}"
    timeout = aiohttp.ClientTimeout(total=10)

    logger.debug(
        "Vexu Users API (DEV): Attempting to fetch user details.",
        extra={
            "url": url,
            "phone_number_encoded": encoded_phone_number,
            "phone_number_raw": phone_number,
        },
    )
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(url, headers=headers) as response:
                print("SESSION::", url, response.status)
                if response.status == 200:
                    user_data = await response.json()
                    logger.info(
                        "Vexu Users API (DEV): Successfully fetched user details.",
                        extra={
                            "phone_number": phone_number,
                            "user_data_preview": truncated_json_dumps(
                                user_data, max_string_len=150
                            ),
                        },
                    )
                    if (
                        "name" in user_data
                        and user_data["name"]
                        and "user_id" in user_data
                        and user_data["user_id"]
                    ):
                        return {
                            "name": user_data["name"],
                            "user_id": user_data["user_id"],
                        }
                    else:
                        logger.warning(
                            f"Vexu Users API (DEV): User data for {phone_number} is incomplete (missing name or user_id).",
                            extra={
                                "phone_number": phone_number,
                                "user_data_received": truncated_json_dumps(user_data),
                            },
                        )
                        return None
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Vexu Users API (DEV): Error fetching user details for {phone_number}.",
                        extra={
                            "phone_number": phone_number,
                            "status_code": response.status,
                            "response_text_preview": error_text[:300],
                            "url": url,
                        },
                    )
                    return None
        except Exception as e:
            logger.error(
                f"Vexu Users API (DEV): Exception during request for {phone_number}.",
                exc_info=True,
                extra={"phone_number": phone_number, "url": url},
            )
            return None


async def post_to_vexu_api(endpoint: str, vexu_call_id: str = None, data: dict = None):
    headers = {
        "X-AI-Service-Token": VEXU_API_TOKEN_DEV,
        "Content-Type": "application/json",
    }
    if vexu_call_id:
        headers["call_id"] = vexu_call_id

    url = f"{VEXU_API_BASE_URL_DEV}{endpoint}"
    timeout = aiohttp.ClientTimeout(total=10)

    # Define log_extra_vexu_api for this scope
    log_extra_vexu_api = {
        "endpoint": endpoint,
        "vexu_call_id": vexu_call_id,
        "url": url,
    }

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(
            url, headers=headers, json=data if data is not None else {}
        ) as response:
            if response.status >= 200 and response.status < 300:
                logger.info(
                    f"Vexu AI (DEV): Successfully posted to endpoint '{endpoint}'.",
                    extra={**log_extra_vexu_api, "status_code": response.status},
                )
                try:
                    response_data = await response.json()
                    logger.debug(
                        "Vexu AI (DEV): Response data from POST.",
                        extra={
                            **log_extra_vexu_api,
                            "response_data_preview": truncated_json_dumps(
                                response_data, max_string_len=150
                            ),
                        },
                    )
                    return response_data
                except aiohttp.ContentTypeError:
                    response_data = await response.text()
                    logger.warning(
                        f"Vexu AI (DEV): Response from POST to endpoint '{endpoint}' not JSON.",
                        extra={
                            **log_extra_vexu_api,
                            "response_body_preview": response_data[:300],
                        },
                    )
                    return response_data
            else:
                error_text = await response.text()
                logger.error(
                    f"Vexu AI (DEV): Error posting to endpoint '{endpoint}'.",
                    extra={
                        **log_extra_vexu_api,
                        "status_code": response.status,
                        "response_text_preview": error_text[:300],
                        "request_payload_preview": truncated_json_dumps(
                            data, max_string_len=150
                        ),
                    },
                )
                return None


async def post_vexu_start_call(
    twilio_call_sid: str, caller_phone: str, dynamic_vexu_user_id: str
):
    log_context = {
        "twilio_call_sid": twilio_call_sid,
        "caller_phone": caller_phone,
        "dynamic_vexu_user_id": dynamic_vexu_user_id,
    }
    if not dynamic_vexu_user_id:
        logger.critical(
            "CRITICAL ERROR: dynamic_vexu_user_id is missing for post_vexu_start_call (DEV).",
            extra=log_context,
        )
        return None

    vexu_call_id_generated = str(uuid.uuid4())
    start_time_iso = (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )
    payload = {
        "user_id": dynamic_vexu_user_id,
        "call_sid": vexu_call_id_generated,
        "contact_id": None,
        "caller_name": None,
        "caller_phone": caller_phone if caller_phone else "Unknown",
        "start_time": start_time_iso,
        "end_time": start_time_iso,
        "duration": 1,
        "transcript": "",
        "audio_base64": "",
        "summary": "",
        "is_encrypted": False,
        "is_emergency": False,  # Initialize as False
    }
    logger.info(
        f"Posting start call to Vexu AI (DEV). Vexu Call ID in payload: {vexu_call_id_generated}.",
        extra={**log_context, "vexu_call_id_generated": vexu_call_id_generated},
    )
    logger.debug(
        "Vexu start call payload (DEV):",
        extra={**log_context, "payload": truncated_json_dumps(payload)},
    )

    response_data = await post_to_vexu_api("", data=payload)

    if (
        response_data
        and isinstance(response_data, dict)
        and "id" in response_data
        and response_data["id"]
    ):
        vexu_call_id_from_response = response_data["id"]
        caller_name_from_response = response_data.get("caller_name")
        logger.info(
            f"Vexu Call created (DEV). API responded with ID: {vexu_call_id_from_response}.",
            extra={
                **log_context,
                "vexu_call_id_from_response": vexu_call_id_from_response,
                "caller_name_from_response": caller_name_from_response,
            },
        )
        return {
            "vexu_call_id": vexu_call_id_from_response,
            "caller_name": caller_name_from_response,
        }
    else:
        logger.error(
            "Error: Failed to get 'id' from Vexu AI post_call response or 'id' is empty (DEV).",
            extra={
                **log_context,
                "response_data_preview": truncated_json_dumps(
                    response_data, max_string_len=150
                ),
            },
        )
        return None


async def post_vexu_message_async(
    vexu_call_id: str,
    text: str,
    sender: str = "caller",
    audio_pcm16_8khz_bytes: bytes = None,
):
    log_context = {
        "vexu_call_id": vexu_call_id,
        "sender": sender,
        "text_preview": text[:50],
    }
    if not vexu_call_id:
        logger.warning(
            "Vexu AI (DEV): Cannot post message, Vexu Call ID is missing.",
            extra=log_context,
        )
        return
    current_time = datetime.now(timezone.utc)
    if sender == "agent":
        current_time += timedelta(milliseconds=500)
    timestamp_iso = current_time.isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )
    audio_base64_payload = ""
    if audio_pcm16_8khz_bytes:
        try:
            if len(audio_pcm16_8khz_bytes) > 0:
                audio_base64_payload = base64.b64encode(audio_pcm16_8khz_bytes).decode(
                    "utf-8"
                )
            else:
                logger.debug(
                    "Vexu AI (DEV): Received empty audio bytes, sending empty audio payload for message.",
                    extra=log_context,
                )
        except Exception as e:
            logger.error(
                "Vexu AI (DEV): Error base64 encoding audio for message.",
                exc_info=True,
                extra=log_context,
            )

    payload = {
        "sender": sender,
        "text": text,
        "timestamp": timestamp_iso,
        "message_type": "regular",
        "audio_base64": audio_base64_payload,
        "is_encrypted": False,
    }
    endpoint = f"/{vexu_call_id}/messages"
    asyncio.create_task(
        post_to_vexu_api(endpoint, vexu_call_id=vexu_call_id, data=payload)
    )


async def post_vexu_end_call(vexu_call_id: str):
    log_context = {"vexu_call_id": vexu_call_id}
    if not vexu_call_id:
        logger.warning(
            "Vexu AI (DEV): Cannot post end call, Vexu Call ID is missing.",
            extra=log_context,
        )
        return
    endpoint = f"/{vexu_call_id}/end"
    logger.info("Vexu AI (DEV): Posting end call.", extra=log_context)
    await post_to_vexu_api(endpoint, vexu_call_id=vexu_call_id, data={})


async def get_openai_summary(
    conversation_text: str, call_sid: Optional[str] = "N/A"
) -> Optional[str]:
    log_context = {"twilio_call_sid_for_summary": call_sid}
    if not conversation_text:
        logger.info(
            "OpenAI Summary (DEV): No conversation text provided.", extra=log_context
        )
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_DEV}",
    }

    data = {
        "model": OPENAI_SUMMARY_MODEL_DEV,
        "messages": [
            # {"role": "system", "content": "You are a helpful assistant designed to summarize call transcripts."},
            {
                "role": "user",
                "content": OPENAI_SUMMARY_PROMPT_TEMPLATE_DEV.format(
                    conversation_text=conversation_text
                ),
            }
        ],
        "temperature": OPENAI_SUMMARY_TEMPERATURE_DEV,
    }
    url = f"{OPENAI_API_BASE_URL_DEV}/chat/completions"
    timeout = aiohttp.ClientTimeout(total=30)

    logger.info(
        "OpenAI Summary (DEV): Requesting summary.",
        extra={**log_context, "model": OPENAI_SUMMARY_MODEL_DEV, "url": url},
    )
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    summary_response = await response.json()
                    if (
                        summary_response.get("choices")
                        and len(summary_response["choices"]) > 0
                    ):
                        message_content = (
                            summary_response["choices"][0]
                            .get("message", {})
                            .get("content")
                        )
                        if message_content:
                            logger.info(
                                "OpenAI Summary (DEV): Successfully received summary.",
                                extra={
                                    **log_context,
                                    "summary_preview": message_content[:100],
                                },
                            )
                            return message_content.strip()
                        else:
                            logger.warning(
                                "OpenAI Summary (DEV): 'content' field missing in choice's message.",
                                extra=log_context,
                            )
                            logger.debug(
                                "OpenAI Summary (DEV): Faulty choice object details.",
                                extra={
                                    **log_context,
                                    "faulty_choice_object": truncated_json_dumps(
                                        summary_response.get("choices", [{}])[0]
                                    ),
                                },
                            )
                            return None
                    else:
                        logger.warning(
                            "OpenAI Summary (DEV): 'choices' field missing or empty in API response.",
                            extra={
                                **log_context,
                                "api_response_preview": truncated_json_dumps(
                                    summary_response, max_string_len=150
                                ),
                            },
                        )
                        return None
                else:
                    error_text = await response.text()
                    logger.error(
                        "OpenAI Summary (DEV): Error from API.",
                        extra={
                            **log_context,
                            "status_code": response.status,
                            "response_text_preview": error_text[:300],
                            "url": url,
                        },
                    )
                    return None
        except Exception as e:
            logger.error(
                "OpenAI Summary (DEV): Exception during request.",
                exc_info=True,
                extra=log_context,
            )
            return None


# NEW FUNCTION: Update Vexu Call Emergency Status
async def update_vexu_call_emergency_status(
    vexu_call_id: str, is_emergency_status: bool
):
    log_context = {
        "vexu_call_id": vexu_call_id,
        "is_emergency_status": is_emergency_status,
    }
    if not vexu_call_id:
        logger.warning(
            "Vexu AI (DEV): Cannot update emergency status, Vexu Call ID is missing.",
            extra=log_context,
        )
        return None

    current_call_data = await get_vexu_call_details(vexu_call_id)
    if not current_call_data:
        logger.error(
            f"Vexu AI (DEV): Failed to fetch existing call data for {vexu_call_id} to update emergency status. Aborting update.",
            extra=log_context,
        )
        return None

    # Only update if the status is changing to avoid unnecessary PUT requests
    if current_call_data.get("is_emergency") == is_emergency_status:
        logger.debug(
            f"Vexu AI (DEV): is_emergency status for {vexu_call_id} is already {is_emergency_status}. No update needed.",
            extra=log_context,
        )
        return None

    payload = current_call_data.copy()
    payload["is_emergency"] = is_emergency_status

    headers = {
        "accept": "application/json",
        "X-AI-Service-Token": VEXU_API_TOKEN_DEV,
        "Content-Type": "application/json",
    }
    url = f"{VEXU_API_BASE_URL_DEV}/{vexu_call_id}"
    timeout = aiohttp.ClientTimeout(total=10)

    logger.info(
        f"Vexu AI (DEV): Attempting to set is_emergency to {is_emergency_status} for call {vexu_call_id}.",
        extra={**log_context, "url": url},
    )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.put(url, headers=headers, json=payload) as response:
                if response.status >= 200 and response.status < 300:
                    logger.info(
                        f"Vexu AI (DEV): Successfully updated is_emergency status for call {vexu_call_id}.",
                        extra={
                            **log_context,
                            "status_code": response.status,
                            "url": url,
                        },
                    )
                    try:
                        response_data = await response.json()
                        logger.debug(
                            "Vexu AI (DEV): Update emergency status response data preview.",
                            extra={
                                **log_context,
                                "url": url,
                                "response_data_preview": truncated_json_dumps(
                                    response_data, max_string_len=150
                                ),
                            },
                        )
                        return response_data
                    except aiohttp.ContentTypeError:
                        text_response = await response.text()
                        logger.warning(
                            f"Vexu AI (DEV): Update emergency status response from {vexu_call_id} not JSON.",
                            extra={
                                **log_context,
                                "url": url,
                                "response_body_preview": text_response[:300],
                            },
                        )
                        return text_response
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Vexu AI (DEV): Error updating is_emergency status for call {vexu_call_id}.",
                        extra={
                            **log_context,
                            "status_code": response.status,
                            "response_text_preview": error_text[:300],
                            "request_payload_preview": truncated_json_dumps(
                                payload, max_string_len=150
                            ),
                        },
                    )
                    return None
        except Exception as e:
            logger.error(
                f"Vexu AI (DEV): Exception during PUT request for emergency status update {vexu_call_id}.",
                exc_info=True,
                extra={**log_context, "url": url},
            )
            return None


# MODIFIED FUNCTION: Check Call Urgency
async def check_call_urgency(vexu_call_id: str, twilio_call_sid: str):
    log_context = {"vexu_call_id": vexu_call_id, "twilio_call_sid": twilio_call_sid}
    logger.info("Checking call urgency (DEV).", extra=log_context)

    messages = await get_vexu_messages(vexu_call_id)
    if not messages:
        logger.warning(
            "No messages found or error fetching for Vexu Call ID (DEV). Cannot check urgency.",
            extra=log_context,
        )
        return

    try:
        messages.sort(
            key=lambda m: (
                datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00"))
                if "timestamp" in m and m["timestamp"]
                else datetime.min.replace(tzinfo=timezone.utc)
            )
        )
    except Exception as e:
        logger.warning(
            "Could not sort messages by timestamp for urgency check (DEV). Proceeding with original order.",
            exc_info=True,
            extra=log_context,
        )

    conversation_parts = [
        f"{msg.get('sender', 'Unknown').capitalize()}: {msg.get('text', '').strip()}"
        for msg in messages
        if msg.get("text", "").strip()
    ]
    full_conversation_text = "\n".join(conversation_parts)

    if not full_conversation_text.strip():
        logger.info(
            "No text content in messages for Vexu Call ID (DEV). Cannot check urgency.",
            extra=log_context,
        )
        return

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY_DEV}",  # This key is sent, but the local server doesn't use it
    }

    data = {
        "model": OPENAI_URGENCY_MODEL_DEV,
        "messages": [
            {
                "role": "user",
                "content": OPENAI_URGENCY_PROMPT_TEMPLATE_DEV.format(
                    conversation_text=full_conversation_text
                ),
            }
        ],
        "temperature": OPENAI_URGENCY_TEMPERATURE_DEV,
    }
    url = f"{OPENAI_API_BASE_URL_DEV}/chat/completions"
    timeout = aiohttp.ClientTimeout(total=10)  # Shorter timeout for quick check

    logger.debug(
        "OpenAI Urgency Check (DEV): Requesting urgency analysis.",
        extra={
            **log_context,
            "model": OPENAI_URGENCY_MODEL_DEV,
            "url": url,
            "prompt_preview": data["messages"][0]["content"][:150],
        },
    )
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    urgency_response = await response.json()
                    if (
                        urgency_response.get("choices")
                        and len(urgency_response["choices"]) > 0
                    ):
                        message_content = (
                            urgency_response["choices"][0]
                            .get("message", {})
                            .get("content")
                        )
                        if message_content:
                            urgency_result = (
                                message_content.strip().upper()
                            )  # Convert to uppercase for robust check
                            logger.info(
                                "Call Urgency Check (DEV): Result received.",
                                extra={**log_context, "urgency_result": urgency_result},
                            )

                            # Check if the response contains "URGENT"
                            if "it is related" in urgency_result.lower():
                                logger.warning(
                                    f"Call {vexu_call_id} detected as URGENT! Updating Vexu record.",
                                    extra=log_context,
                                )
                                asyncio.create_task(
                                    update_vexu_call_emergency_status(
                                        vexu_call_id, True
                                    )
                                )
                            else:
                                logger.info(
                                    f"Call {vexu_call_id} detected as NOT URGENT.",
                                    extra=log_context,
                                )
                                # Optionally, you could set it to False if it was previously True,
                                # but typically urgency is a one-way flag for a call.
                                # asyncio.create_task(update_vexu_call_emergency_status(vexu_call_id, False))
                            return urgency_result
                        else:
                            logger.warning(
                                "Call Urgency Check (DEV): 'content' field missing in choice's message.",
                                extra=log_context,
                            )
                            return None
                    else:
                        logger.warning(
                            "Call Urgency Check (DEV): 'choices' field missing or empty in API response.",
                            extra={
                                **log_context,
                                "api_response_preview": truncated_json_dumps(
                                    urgency_response, max_string_len=150
                                ),
                            },
                        )
                        return None
                else:
                    error_text = await response.text()
                    logger.error(
                        "Call Urgency Check (DEV): Error from API.",
                        extra={
                            **log_context,
                            "status_code": response.status,
                            "response_text_preview": error_text[:300],
                            "url": url,
                        },
                    )
                    return None
        except Exception as e:
            logger.error(
                "Call Urgency Check (DEV): Exception during request.",
                exc_info=True,
                extra=log_context,
            )
            return None


async def get_vexu_call_details(vexu_call_id: str) -> Optional[Dict[str, Any]]:
    log_context = {"vexu_call_id": vexu_call_id}
    if not vexu_call_id:
        logger.warning(
            "Vexu AI (DEV): Cannot get call details, Vexu Call ID is missing.",
            extra=log_context,
        )
        return None

    headers = {"accept": "application/json", "X-AI-Service-Token": VEXU_API_TOKEN_DEV}
    url = f"{VEXU_API_BASE_URL_DEV}/{vexu_call_id}"
    timeout = aiohttp.ClientTimeout(total=10)

    logger.debug(
        f"Vexu AI (DEV): Attempting to fetch call details.",
        extra={**log_context, "url": url},
    )
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    call_data = await response.json()
                    logger.info(
                        "Vexu AI (DEV): Successfully fetched call details.",
                        extra={
                            **log_context,
                            "call_data_preview": truncated_json_dumps(
                                call_data, max_string_len=150
                            ),
                        },
                    )
                    return call_data
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Vexu AI (DEV): Error fetching call details for {vexu_call_id}.",
                        extra={
                            **log_context,
                            "status_code": response.status,
                            "response_text_preview": error_text[:300],
                            "url": url,
                        },
                    )
                    return None
        except Exception as e:
            logger.error(
                f"Vexu AI (DEV): Exception during GET request for call details {vexu_call_id}.",
                exc_info=True,
                extra={**log_context, "url": url},
            )
            return None


async def update_vexu_call_summary(
    vexu_call_id: str,
    summary_text: str,
    caller_name_override: Optional[str] = None,
    existing_call_data_fetched: Optional[Dict[str, Any]] = None,
):
    log_context = {
        "vexu_call_id": vexu_call_id,
        "caller_name_override": caller_name_override,
    }
    if not vexu_call_id:
        logger.warning(
            "Vexu AI (DEV): Cannot update call summary, Vexu Call ID is missing.",
            extra=log_context,
        )
        return None
    if not summary_text:
        logger.warning(
            f"Vexu AI (DEV): Cannot update call summary for {vexu_call_id}, summary text is empty.",
            extra=log_context,
        )
        return None

    current_call_data = existing_call_data_fetched
    if not current_call_data:
        current_call_data = await get_vexu_call_details(vexu_call_id)
        if not current_call_data:
            logger.critical(
                f"Vexu AI (DEV): Failed to fetch existing call data for {vexu_call_id} to update summary. Cannot update accurately.",
                extra=log_context,
            )
            return None

    payload = current_call_data.copy()
    payload["summary"] = summary_text

    headers = {
        "accept": "application/json",
        "X-AI-Service-Token": VEXU_API_TOKEN_DEV,
        "Content-Type": "application/json",
    }
    url = f"{VEXU_API_BASE_URL_DEV}/{vexu_call_id}"
    timeout = aiohttp.ClientTimeout(total=10)

    logger.debug(
        f"Vexu AI (DEV): Attempting to update call {vexu_call_id} with summary. Payload preview.",
        extra={
            **log_context,
            "url": url,
            "payload_preview": truncated_json_dumps(payload, max_string_len=150),
        },
    )

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.put(url, headers=headers, json=payload) as response:
                if response.status >= 200 and response.status < 300:
                    logger.info(
                        f"Vexu AI (DEV): Successfully updated call {vexu_call_id} with summary.",
                        extra={
                            **log_context,
                            "status_code": response.status,
                            "url": url,
                        },
                    )
                    try:
                        response_data = await response.json()
                        logger.debug(
                            "Vexu AI (DEV): Update summary response data preview.",
                            extra={
                                **log_context,
                                "url": url,
                                "response_data_preview": truncated_json_dumps(
                                    response_data, max_string_len=150
                                ),
                            },
                        )
                        return response_data
                    except aiohttp.ContentTypeError:
                        text_response = (
                            await response.text()
                        )  # Capture text before logging
                        logger.warning(
                            f"Vexu AI (DEV): Update summary response from {vexu_call_id} not JSON.",
                            extra={
                                **log_context,
                                "url": url,
                                "response_body_preview": text_response[:300],
                            },
                        )
                        return text_response
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Vexu AI (DEV): Error updating call {vexu_call_id} with summary.",
                        extra={
                            **log_context,
                            "status_code": response.status,
                            "response_text_preview": error_text[:300],
                            "request_payload_preview": truncated_json_dumps(
                                payload, max_string_len=150
                            ),
                        },
                    )
                    return None
        except Exception as e:
            logger.error(
                f"Vexu AI (DEV): Exception during PUT request for call summary update {vexu_call_id}.",
                exc_info=True,
                extra={**log_context, "url": url},
            )
            return None


async def get_vexu_messages(vexu_call_id: str) -> List[Dict[str, Any]]:
    log_context = {"vexu_call_id": vexu_call_id}
    if not vexu_call_id:
        logger.warning(
            "Vexu AI (DEV): Cannot get messages, vexu_call_id is missing.",
            extra=log_context,
        )
        return []

    headers = {"accept": "application/json", "X-AI-Service-Token": VEXU_API_TOKEN_DEV}
    url = f"{VEXU_API_BASE_URL_DEV}/{vexu_call_id}/messages?skip=0&limit=100"
    timeout = aiohttp.ClientTimeout(total=10)
    messages = []

    logger.debug(
        f"Vexu AI (DEV): Attempting to fetch messages for call_id: {vexu_call_id}.",
        extra={**log_context, "url": url},
    )
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    messages_data = await response.json()
                    logger.info(
                        f"Vexu AI (DEV): Successfully fetched {len(messages_data)} messages for {vexu_call_id}.",
                        extra=log_context,
                    )
                    logger.debug(
                        "Vexu AI (DEV): Fetched messages data preview.",
                        extra={
                            **log_context,
                            "messages_preview": truncated_json_dumps(
                                messages_data, max_string_len=150
                            ),
                        },
                    )
                    return messages_data
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Vexu AI (DEV): Error fetching messages for {vexu_call_id}.",
                        extra={
                            **log_context,
                            "status_code": response.status,
                            "response_text_preview": error_text[:300],
                            "url": url,
                        },
                    )
        except Exception as e:
            logger.error(
                f"Vexu AI (DEV): Exception during request for messages for {vexu_call_id}.",
                exc_info=True,
                extra={**log_context, "url": url},
            )
    return messages


async def process_call_summary_and_update(
    vexu_call_id: str,
    twilio_call_sid: Optional[str],
    caller_name_for_summary: Optional[str],
):
    log_context = {
        "vexu_call_id": vexu_call_id,
        "twilio_call_sid": twilio_call_sid,
        "caller_name_for_summary": caller_name_for_summary,
    }
    logger.info("Starting post-call summarization process (DEV).", extra=log_context)

    messages = await get_vexu_messages(vexu_call_id)
    if not messages:
        logger.warning(
            "No messages found or error fetching for Vexu Call ID (DEV). Aborting summarization.",
            extra=log_context,
        )
        return

    try:
        messages.sort(
            key=lambda m: (
                datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00"))
                if "timestamp" in m and m["timestamp"]
                else datetime.min.replace(tzinfo=timezone.utc)
            )
        )
    except Exception as e:
        logger.warning(
            "Could not sort messages by timestamp due to format error (DEV). Proceeding with original order.",
            exc_info=True,
            extra=log_context,
        )

    conversation_parts = [
        f"{msg.get('sender', 'Unknown').capitalize()}: {msg.get('text', '').strip()}"
        for msg in messages
        if msg.get("text", "").strip()
    ]
    full_conversation_text = "\n".join(conversation_parts)

    if not full_conversation_text.strip():
        logger.info(
            "No text content in messages for Vexu Call ID (DEV). Aborting summarization.",
            extra=log_context,
        )
        return

    # IMPORTANT: This `get_openai_summary` still uses the *real* OpenAI API
    # via OPENAI_API_BASE_URL_DEV (which is `http://localhost:8000` from .env,
    # so it would go to server.py's FastAPI routes, not its websocket).
    # This is distinct from the Realtime API WebSocket.
    summary_text_generated = await get_openai_summary(
        full_conversation_text, call_sid=twilio_call_sid
    )
    if not summary_text_generated:
        logger.warning(
            "Failed to generate summary for Vexu Call ID (DEV). Aborting update.",
            extra=log_context,
        )
        return

    logger.info(
        "Generated summary for Vexu Call ID (DEV).",
        extra={**log_context, "summary_text_preview": summary_text_generated[:100]},
    )

    existing_call_data = await get_vexu_call_details(vexu_call_id)
    if not existing_call_data:
        logger.critical(
            "CRITICAL: Failed to retrieve existing call details for summary update (DEV). Summary cannot be updated with full context. Aborting PUT.",
            extra=log_context,
        )
        return

    await update_vexu_call_summary(
        vexu_call_id=vexu_call_id,
        summary_text=summary_text_generated,
        caller_name_override=caller_name_for_summary,
        existing_call_data_fetched=existing_call_data,
    )


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
    vexu_user_data = await get_vexu_user_details(twilio_reciever_number)

    if (
        not vexu_user_data
        or not vexu_user_data.get("name")
        or not vexu_user_data.get("user_id")
    ):
        logger.critical(
            f"CRITICAL ERROR: Could not fetch complete dynamic user details (name, user_id) from Vexu API (DEV) for {twilio_reciever_number}.",
            extra={
                **log_context_call,
                "vexu_user_data_received": truncated_json_dumps(vexu_user_data),
            },
        )
        raise ValueError(
            f"Failed to fetch complete Vexu user details (name and user_id) for {twilio_reciever_number}."
        )

    owner_name_for_call = vexu_user_data["name"]
    user_id_for_call = vexu_user_data["user_id"]
    logger.info(
        f"Using dynamic owner (DEV): '{owner_name_for_call}', user_id: '{user_id_for_call}'.",
        extra={
            **log_context_call,
            "owner_name_for_call": owner_name_for_call,
            "user_id_for_call": user_id_for_call,
        },
    )
    start_call_response = await post_vexu_start_call(
        twilio_call_sid=twilio_call_sid,
        caller_phone=caller_phone,
        dynamic_vexu_user_id=user_id_for_call,
    )

    if not start_call_response or not start_call_response.get("vexu_call_id"):
        logger.critical(
            "CRITICAL ERROR: Failed to obtain vexu_call_id from Vexu after posting start call (DEV).",
            extra={
                **log_context_call,
                "user_id_used": user_id_for_call,
                "start_call_response_preview": truncated_json_dumps(
                    start_call_response
                ),
            },
        )
        raise RuntimeError(f"Failed to create Vexu call record (vexu_call_id is null).")

    vexu_call_id = start_call_response["vexu_call_id"]
    retrieved_caller_name = start_call_response.get("caller_name")

    logger.info(
        f"Vexu Call ID established (DEV): {vexu_call_id}. Retrieved Caller Name: {retrieved_caller_name}.",
        extra={
            **log_context_call,
            "vexu_call_id": vexu_call_id,
            "retrieved_caller_name": retrieved_caller_name,
        },
    )
    call_buffer[twilio_call_sid] = {
        "vexu_call_id": vexu_call_id,
        "caller_phone": caller_phone,
        "owner_name": owner_name_for_call,
        "vexu_user_id": user_id_for_call,
        "caller_name": retrieved_caller_name,
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

        async def receive_from_twilio(): # receive from the ws from orchestrator ws/sid @Borhan
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

                        owner_name_for_prompt = call_data["owner_name"]
                        caller_name_from_buffer = call_data.get("caller_name")
                        caller_name_greeting_segment = (
                            caller_name_from_buffer
                            if caller_name_from_buffer
                            else "Unknown"
                        )
                        if not session_initialized_for_this_call:
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

                if call_sid and call_sid in call_buffer:
                    current_vexu_call_id = call_buffer[call_sid].get("vexu_call_id")
                    retrieved_caller_name_for_summary = call_buffer[call_sid].get(
                        "caller_name"
                    )
                    if current_vexu_call_id:
                        logger.info(
                            "Posting Vexu end call (DEV).",
                            extra={
                                **log_context,
                                "vexu_call_id": current_vexu_call_id,
                            },
                        )
                        await post_vexu_end_call(vexu_call_id=current_vexu_call_id)
                        await process_call_summary_and_update(
                            vexu_call_id=current_vexu_call_id,
                            twilio_call_sid=call_sid,
                            caller_name_for_summary=retrieved_caller_name_for_summary,
                        )
                    else:
                        logger.warning(
                            "Vexu Call ID not found in buffer for end call/summarization (DEV).",
                            extra=log_context,
                        )
                else:
                    logger.warning(
                        "Call SID not in buffer or not set for end call/summarization (DEV).",
                        extra=log_context,
                    )

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

                    # # @Borhan: Interupt added (the if statement)
                    # if call_agent.interrupt_requested:
                    #     logger.info(
                    #         "Interruption requested - clearing audio buffer and resetting state",
                    #         extra=log_context
                    #     )

                    #     # Clear all pending audio responses
                    #     call_agent.formatted_audio_responses.clear()

                    #     # Reset interruption state
                    #     call_agent.interrupt_requested = False
                    #     call_agent.is_responding = False

                    #     # Clear local buffer
                    #     gpt_output_buffer = b''

                    #     logger.info(
                    #         "Interruption completed - ready for new input",
                    #         extra=log_context
                    #     )
                    #     continue

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
                        if gpt_output_buffer:  # This is the agent's response
                            agent_audio_for_vexu_pcm16_8khz = gpt_output_buffer
                            asyncio.create_task(
                                post_vexu_message_async(
                                    vexu_call_id=current_vexu_call_id,
                                    text=response["transcript"],
                                    sender="agent",
                                    audio_pcm16_8khz_bytes=agent_audio_for_vexu_pcm16_8khz,
                                )
                            )
                            logger.info(
                                "Sent agent message to Vexu AI (DEV).",
                                extra={
                                    **log_context,
                                    "vexu_call_id": current_vexu_call_id,
                                    "transcript_preview": response["transcript"][:100],
                                },
                            )
                            gpt_output_buffer = b""  # Clear agent's audio buffer

                            agent_transcript_lower = response["transcript"].lower()
                            if (
                                call_sid
                                and call_sid in call_buffer
                                and any(
                                    keyword in agent_transcript_lower
                                    for keyword in HANGUP_KEYWORDS
                                )
                            ):
                                logger.info(
                                    f"Agent said bye keyword (DEV): '{response['transcript']}'. Flagging for hangup after audio playback.",
                                    extra={
                                        **log_context,
                                        "transcript": response["transcript"],
                                    },
                                )
                                call_buffer[call_sid]["agent_initiated_hangup"] = True
                        else:  # This is the caller's response (transcribed by server.py)
                            # NEW: Retrieve the pre-processed and trimmed audio from call_buffer
                            call_data = call_buffer.get(call_sid)
                            pcm_audio_caller_for_vexu = b""
                            if call_data:
                                pcm_audio_caller_for_vexu = call_data.pop(
                                    "last_caller_audio_for_vexu", b""
                                )
                                if not pcm_audio_caller_for_vexu:
                                    logger.warning(
                                        "No trimmed caller audio found for Vexu message (DEV). Sending empty audio.",
                                        extra=log_context,
                                    )
                            else:
                                logger.warning(
                                    "Call data not found for Vexu message (DEV). Sending empty audio.",
                                    extra=log_context,
                                )

                            asyncio.create_task(
                                post_vexu_message_async(
                                    vexu_call_id=current_vexu_call_id,
                                    text=response["transcript"],
                                    sender="caller",
                                    audio_pcm16_8khz_bytes=pcm_audio_caller_for_vexu,
                                )
                            )
                            logger.info(
                                "Sent caller message to Vexu (DEV) using server's transcript and trimmed audio.",
                                extra={
                                    **log_context,
                                    "vexu_call_id": current_vexu_call_id,
                                    "transcript_preview": response["transcript"][:100],
                                    "audio_bytes_len": len(pcm_audio_caller_for_vexu),
                                },
                            )
                            # caller_audio_buffer.clear() # REMOVED: Buffer is now cleared in speech_stopped

                        # Check urgency after ANY message (caller or agent)
                        asyncio.create_task(
                            check_call_urgency(
                                vexu_call_id=current_vexu_call_id,
                                twilio_call_sid=call_sid,
                            )
                        )

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
