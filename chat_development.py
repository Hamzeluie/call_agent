import asyncio
import base64
import json
import logging
import os
import urllib.parse
from pathlib import Path
from typing import Dict, Set

import requests
import uvicorn
import websockets

# Import your ChatAgent
from chat_agent import ChatAgent
from dashboard import get_agent_config
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
from fastapi.staticfiles import StaticFiles
from logging_config import get_logger
from pydantic import BaseModel
from utils import get_env_variable, truncated_json_dumps

# Initialize the logger for this module
logger = get_logger(__name__)

ENDPOINT_MESSAGE = get_env_variable("ENDPOINT_MESSAGE")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

router = APIRouter(prefix=ENDPOINT_MESSAGE, tags=["messenger"])

# Store active chat sessions with ChatAgent instances
active_sessions: Dict[str, ChatAgent] = {}


# Pydantic models for data validation
class ChatInitRequest(BaseModel):
    user_id: str
    agent_id: str
    session_id: str


class SessionConfig(BaseModel):
    kb_id: str
    user_id: str
    config: dict
    system_prompt: str


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
    Configures session using ChatAgent.
    """
    logging.info(f"Received configuration: {session_config}")

    try:
        session_key = f"{session_config.user_id}:{session_config.kb_id}"

        # Create and configure ChatAgent
        yaml_path = Path(__file__).parent / "config.yaml"  # Adjust path as needed
        agent = ChatAgent(
            owner_id=session_config.user_id,
            system_prompt=session_config.system_prompt,
            yaml_path=yaml_path,
            kb_id=[session_config.kb_id],
            config=session_config.config,
        )

        # Connect to servers
        await agent.connect_servers()

        # Store the agent
        active_sessions[session_key] = agent

        logging.info(f"Session {session_key} configured successfully")
        return {"message": "Configuration successful.", "session_key": session_key}

    except Exception as e:
        logging.error(f"Error configuring session: {e}")
        raise HTTPException(
            status_code=500, detail=f"Could not configure session: {str(e)}"
        )


@router.post("/chat/init")
async def init_chat(request: ChatInitRequest):
    """
    Initializes chat session using ChatAgent. This must be called before WebSocket connection.
    """
    logging.info(f"Received chat init request: {request}")

    try:
        # Get agent configuration from dashboard
        agent_config = get_agent_config(request.user_id, request.agent_id)
        logging.info(f"Agent configuration: {agent_config}")

        if not isinstance(agent_config, dict) or not all(
            key in agent_config for key in ["kb_ids", "configs", "system_prompt"]
        ):
            raise ValueError("Invalid agent configuration format")

        session_key = f"{request.user_id}:{request.session_id}"

        # Check if session already exists
        if session_key in active_sessions:
            logging.info(f"Session {session_key} already exists, reusing...")
            return {
                "status": "success",
                "message": "Chat session already initialized",
                "session_id": request.session_id,
                "session_key": session_key,
            }

        # Create SessionConfig for the configure endpoint
        session_config = SessionConfig(
            kb_id=",".join(agent_config["kb_ids"]),
            user_id=request.user_id,
            config=agent_config["configs"],
            system_prompt=agent_config["system_prompt"],
        )

        # Call configure endpoint to set up the ChatAgent
        configure_response = await configure_session(session_config)

        # Update session key to match the WebSocket format
        websocket_session_key = f"{request.user_id}:{request.session_id}"
        configured_session_key = configure_response.get("session_key")

        # If the session key from configure is different, update our active_sessions
        if configured_session_key and configured_session_key != websocket_session_key:
            active_sessions[websocket_session_key] = active_sessions.pop(
                configured_session_key
            )
            logging.info(
                f"Updated session key from {configured_session_key} to {websocket_session_key}"
            )

        logging.info(f"Session {websocket_session_key} initialized successfully")

        return {
            "status": "success",
            "message": "Chat session initialized",
            "session_id": request.session_id,
            "session_key": websocket_session_key,
        }

    except Exception as e:
        logging.error(f"Error initializing chat: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error initializing chat: {str(e)}"
        )


@router.websocket("/ws/{user_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str, session_id: str):
    await websocket.accept()

    session_key = f"{user_id}:{session_id}"

    if session_key not in active_sessions:
        error_msg = "Session not configured. Please call /chat/init first."
        await websocket.send_text(json.dumps({"type": "error", "message": error_msg}))
        await websocket.close(1008)
        logging.error(f"Session {session_key} not found in active_sessions")
        return

    logging.info(f"Client connected for {session_key}")

    try:
        agent = active_sessions[session_key]

        # Send connection confirmation
        await websocket.send_text(
            json.dumps({"type": "connected", "message": "Chat session established"})
        )

        while True:
            # Receive message from client
            user_message = await websocket.receive_text()
            logging.info(f"Received from client ({session_key}): '{user_message}'")

            # Send typing indicator
            await websocket.send_text(
                json.dumps({"type": "typing", "message": "AI is thinking..."})
            )

            try:
                # Process message through ChatAgent with streaming
                full_response = ""
                is_first_chunk = True

                async for chunk in agent.send_message(user_message):
                    if chunk:
                        # Send each chunk as it arrives
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "type": "chunk",
                                    "content": chunk,
                                    "is_first": is_first_chunk,
                                }
                            )
                        )
                        full_response += chunk
                        is_first_chunk = False

                # Send completion message
                await websocket.send_text(
                    json.dumps({"type": "complete", "message": full_response})
                )

            except asyncio.TimeoutError:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": "Request timed out"})
                )

            except Exception as e:
                logging.error(f"Error processing message for {session_key}: {e}")
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "message": f"Error processing message: {str(e)}",
                        }
                    )
                )

    except WebSocketDisconnect:
        logging.info(f"Client disconnected from {session_key}")
    except Exception as e:
        logging.error(f"Error in session {session_key}: {e}")
        await websocket.send_text(json.dumps({"error": str(e)}))
    finally:
        # Keep the session active for multiple messages
        await websocket.close()
        logging.info(f"WebSocket closed for {session_key}")


@router.delete("/chat/{user_id}/{session_id}")
async def end_chat_session(user_id: str, session_id: str):
    """End a chat session and clean up resources"""
    session_key = f"{user_id}:{session_id}"

    if session_key in active_sessions:
        try:
            await active_sessions[session_key].close()
            del active_sessions[session_key]
            logging.info(f"Session {session_key} ended successfully")
            return {"status": "success", "message": "Session ended"}
        except Exception as e:
            logging.error(f"Error ending session {session_key}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Error ending session: {str(e)}"
            )
    else:
        return {"status": "not_found", "message": "Session not found"}


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "vexu-chat-api",
        "active_sessions": len(active_sessions),
    }


app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5100)
