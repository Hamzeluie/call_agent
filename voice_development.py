import asyncio
import audioop
import base64
import contextlib
import json
from typing import Dict, List
from uuid import uuid4

import numpy as np
from agent_architect.datatype_abstraction import TextFeatures
from call_agent import InferenceService
from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from logging_config import get_logger
from pydantic import BaseModel
from utils import get_env_variable

# Initialize the logger for this module
logger = get_logger(__name__)

ENDPOINT_VOICE = get_env_variable("ENDPOINT_VOICE")


router = APIRouter(prefix=ENDPOINT_VOICE, tags=["voice"])

# Store active chat sessions (not strictly necessary since InferenceService manages sessions)
active_sessions: Dict[str, InferenceService] = {}


# Pydantic models for data validation
class ChatInitRequest(BaseModel):
    owner_id: str
    user_id: str
    agent_id: str


class SessionConfig(BaseModel):
    kb_id: str
    user_id: str
    config: dict
    system_prompt: str


MAX_CONCURRENT_CLIENTS = 10  # Optional: limit concurrency if needed

# Initialize InferenceService
voice_agent = InferenceService(
    agent_type="voice",
    service_names=["VAD", "STT", "RAG", "TTS"],
    channels_steps={
        "VAD": ["input"],
        "STT": ["high", "low"],
        "RAG": ["high", "low"],
        "TTS": ["high", "low"],
    },
    input_channel="VAD:input",
    output_channel="voice:output",
    timeout=30.0,
)


# Start the InferenceService
@router.on_event("startup")
async def startup_event():
    asyncio.create_task(voice_agent.start())
    logger.info("InferenceService started")


# Cleanup on shutdown
@router.on_event("shutdown")
async def shutdown_event():
    await voice_agent.stop()
    logger.info("InferenceService stopped")


@router.get("/", response_class=HTMLResponse)
async def get_html():
    """
    Serves the static HTML page for the Q&A interface from the separate HTML file.
    """
    try:
        with open("static/chat.html", "r") as file:
            html_content = file.read()
        return HTMLResponse(content=html_content, status_code=200)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="HTML file not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading HTML file: {str(e)}"
        )


@router.api_route("/answer", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    # ... (form processing and sid generation remains the same)
    form = await request.form()
    owner_id = form.get("owner_id")
    agent_id = form.get("agent_id")
    user_id = form.get("user_id")
    sid = form.get("sid")
    await voice_agent.start_session(sid=sid, owner_id=owner_id, agent_id=agent_id)

    # --- Send SID in an HTTP Header ---
    return JSONResponse(
        {
            "sid": sid,
            "websocket_url": f"wss://{request.url.hostname}{ENDPOINT_VOICE}/ws/{sid}",
        }
    )


@router.websocket("/ws/{sid}")
async def websocket_endpoint(websocket: WebSocket, sid: str):
    await websocket.accept()

    # Optional: confirm session exists (defensive)
    # if not await voice_agent.session_exists(sid):
    #     await websocket.close(code=1008)
    #     return

    send_task = asyncio.create_task(send_to_frontend(websocket, sid))
    try:
        async for message in websocket.iter_text():
            try:

                if await voice_agent.is_session_interrupt(sid):
                    await websocket.send_json(
                        {
                            "event": "interrupt",
                            "message": "Your session has interrupt due to inactivity.",
                        }
                    )

                data = json.loads(message)
                event_type = data.get("event")

                if event_type == "start":
                    logger.info(f"Call started for session {sid}: {data}")
                    continue  # No audio yet

                elif event_type == "stop":
                    logger.info(f"Call stopped for session {sid}: {data}")
                    break  # End the loop

                elif event_type == "media":
                    await receive_from_frontend(
                        sid, data
                    )  # Note: removed unused `websocket` arg

                else:
                    logger.debug(
                        f"Ignoring unknown event: {event_type} in session {sid}"
                    )
                    continue

            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(
                    f"Invalid or unexpected message format: {message[:200]}..."
                )
                continue  # Don't crash on bad input

    except WebSocketDisconnect:
        logger.info(f"Client {sid} disconnected")
    except Exception as e:
        logger.exception(f"Unexpected error in WebSocket for session {sid}")
    finally:
        send_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await send_task
        await voice_agent.stop_session(sid=sid)


async def receive_from_frontend(sid: str, data: dict):
    # Safe: only called when event == "media"
    media_payload = data["media"]["payload"]
    audio_bytes_ulaw = base64.b64decode(media_payload)
    pcm_audio = audioop.ulaw2lin(audio_bytes_ulaw, 2)

    # Save to file (append mode)
    # session_info = active_sessions.get(sid)
    # if session_info:
    # with open(
    #     "/home/ubuntu/borhan/whole_pipeline/vexu/outputs/recive/a.pcm", "ab"
    # ) as f:
    #     f.write(pcm_audio)

    pcm_payload = base64.b64encode(pcm_audio).decode("utf-8")
    await voice_agent.send_chunk(sid, pcm_payload)


async def send_to_frontend(websocket: WebSocket, sid: str):
    try:
        async for chunk in voice_agent.predict(sid):
            # 1. Decode the base64 audio (this is float32 PCM at 24kHz)
            if await voice_agent.is_session_interrupt(sid):
                logger.info(
                    f"Interrupt detected in send_to_frontend for {sid}, dropping chunk"
                )

            float32_bytes = base64.b64decode(chunk.audio)

            if not float32_bytes:
                continue

            # 2. Convert bytes to numpy float32 array
            audio_float32 = np.frombuffer(float32_bytes, dtype=np.float32)

            # 3. Clip and convert to int16
            # Fish TTS output is typically in the range [-1.0, 1.0]
            audio_int16 = (np.clip(audio_float32, -1.0, 1.0) * 32767).astype(np.int16)

            # 4. Resample from 24kHz to 8kHz
            # Simple decimation by a factor of 3 (24000 / 8000 = 3)
            # This is fast and sufficient for this specific ratio.
            if len(audio_int16) % 3 != 0:
                # Trim to make it divisible by 3 for clean decimation
                audio_int16 = audio_int16[: -(len(audio_int16) % 3)]
            audio_8k_int16 = audio_int16[::3]

            # 5. Convert to bytes for audioop
            pcm_8k_bytes = audio_8k_int16.tobytes()
            # with open(
            #     "/home/ubuntu/borhan/whole_pipeline/vexu/outputs/send/a.pcm", "ab"
            # ) as f:
            #     f.write(pcm_8k_bytes)

            # 6. Now it's safe to convert to ulaw
            ulaw_bytes_for_twilio = audioop.lin2ulaw(pcm_8k_bytes, 2)
            audio_payload_for_twilio = base64.b64encode(ulaw_bytes_for_twilio).decode(
                "utf-8"
            )

            audio_delta_twilio = {
                "event": "media",
                "streamSid": sid,
                "media": {"payload": audio_payload_for_twilio},
            }
            await websocket.send_json(audio_delta_twilio)

    except Exception as e:
        logger.exception("Error in send_to_frontend")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "vexu-chat-api",
        "active_sessions": len(
            active_sessions
        ),  # Note: May not reflect actual sessions in Redis
    }
