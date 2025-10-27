import asyncio
import json
from typing import Dict, List
from uuid import uuid4

from agent_architect.datatype_abstraction import TextFeatures
from chat_agent import InferenceService
from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from logging_config import get_logger
from pydantic import BaseModel
from utils import get_env_variable

# Initialize the logger for this module
logger = get_logger(__name__)

ENDPOINT_MESSAGE = get_env_variable("ENDPOINT_MESSAGE")


router = APIRouter(prefix=ENDPOINT_MESSAGE, tags=["messenger"])

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


# Initialize InferenceService
chat_agent = InferenceService(
    agent_type="chat",
    service_names=["RAG"],
    channels_steps={"RAG": ["high", "low"]},
    input_channel="RAG:low",
    output_channel="chat:output",
    timeout=30.0,
)


# Start the InferenceService
@router.on_event("startup")
async def startup_event():
    asyncio.create_task(chat_agent.start())
    logger.info("InferenceService started")


# Cleanup on shutdown
@router.on_event("shutdown")
async def shutdown_event():
    await chat_agent.stop()
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


@router.post("/configure")
async def configure_session(session_config: SessionConfig):
    """
    Configures session using InferenceService.
    """
    logger.info(f"Received configuration: {session_config}")
    try:
        # Note: InferenceService does not have a direct configure method.
        # Assuming configuration is handled via session initialization or external storage.
        # For now, return a success message with a session key derived from user_id and kb_id.
        session_key = f"{session_config.user_id}_{session_config.kb_id}"
        return {"message": "Configuration successful.", "session_key": session_key}
    except Exception as e:
        logger.error(f"Error configuring session: {e}")
        raise HTTPException(
            status_code=500, detail=f"Could not configure session: {str(e)}"
        )


@router.post("/chat/init")
async def init_chat(request: ChatInitRequest):
    """
    Initializes chat session using InferenceService.
    """
    try:
        sid = f"{request.owner_id}:{request.agent_id}:{request.user_id}:{str(uuid4().hex)}"
        await chat_agent.start_session(
            sid=sid, agent_id=request.agent_id, owner_id=request.owner_id
        )
        logger.info(f"Chat session initialized: {sid}")
        return {
            "status": "success",
            "message": f"Session {sid} initialized",
            "session_id": sid,
        }
    except Exception as e:
        logger.error(f"Error initializing session {sid}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error initializing session: {str(e)}"
        )


@router.websocket("/ws/{sid}")
async def websocket_endpoint(websocket: WebSocket, sid: str):
    await websocket.accept()
    try:
        # Send connection confirmation
        await websocket.send_text(
            json.dumps({"type": "connected", "message": "Chat session established"})
        )

        while True:
            if not await chat_agent.is_session_active(sid):
                await websocket.send_text(json.dumps({"type": "timeout", "message": "yout session is expired. in case of inactivity"}))
                raise Exception(f"Session {sid} is not active or has been stopped")
            # Receive message from client
            user_message = await websocket.receive_text()
            logger.info(f"Received from client ({sid}): '{user_message}'")

            # Send typing indicator
            await websocket.send_text(
                json.dumps({"type": "typing", "message": "AI is thinking..."})
            )

            try:
                response = []
                async for chunk in chat_agent.predict(user_message, sid):
                    if chunk.is_final:
                        break
                    response.append(chunk.text)
                    # Send response as a single chunk to maintain compatibility with client expectations
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "chunk",
                                "content": chunk.text,
                                # "is_first": True,
                            }
                        )
                    )
                # Send completion message
                await websocket.send_text(
                    json.dumps({"type": "complete", "message": "".join(response)})
                )

            except asyncio.TimeoutError:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Request timed out"})
                )

            except Exception as e:
                logger.error(f"Error processing message for {sid}: {e}")
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": f"Error processing message: {str(e)}",
                        }
                    )
                )

    except WebSocketDisconnect:
        logger.info(f"Client disconnected from {sid}")
    except Exception as e:
        logger.error(f"Error in session {sid}: {e}")
        await websocket.send_text(json.dumps({"error": str(e)}))
    finally:
        await websocket.close()
        logger.info(f"WebSocket closed for {sid}")


@router.delete("/chat/{session_id}")
async def end_chat_session(session_id: str):
    """End a chat session and clean up resources"""
    try:
        await chat_agent.stop_session(sid=session_id)
        logger.info(f"Session {session_id} ended successfully")
        return {"status": "success", "message": "Session ended"}
    except Exception as e:
        logger.error(f"Error ending session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error ending session: {str(e)}")


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
