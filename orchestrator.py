import asyncio
import audioop
import base64
import json
import re
import time
import uuid
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import numpy as np
import websockets
import yaml
from call_development import router as development_router
from chat_development import router as messages_router
from dashboard import router as dashboard_router
from fastapi import FastAPI, HTTPException, Request, WebSocket, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from logging_config import get_logger, setup_application_logging
from pydantic import BaseModel

# Add these imports for audio processing
from utils import get_env_variable

# import audioop  # Remove this deprecated import

yaml_path = Path(__file__).parents[0] / "config.yaml"
yaml_config = yaml.safe_load(open(yaml_path, "r"))

# Define HOST and PORT from the YAML configuration
HOST = yaml_config["orchestrator"]["host"]
PORT = yaml_config["orchestrator"]["port"]

# Call simulator configuration
TWILIO_SERVER_BASE_URL = "https://ols.vexu.ai"
DEVELOPMENT_ENDPOINT = get_env_variable("ENDPOINT_DEV")
MESSAGE_ENDPOINT = get_env_variable("ENDPOINT_MESSAGE")


class CallRequest(BaseModel):
    owner_id: str = "+12345952496"  # owner number
    user_id: str = "+201140099226"  # user number
    agent_id: str = "agent_1"


class CallSession:
    def __init__(self, call_sid: str, owner_id: str, user_id: str):
        self.call_sid = call_sid
        self.owner_id = owner_id
        self.user_id = user_id
        self.stream_sid = str(uuid.uuid4())
        self.twilio_ws = None
        self.client_ws = None
        self.is_active = False
        self.last_media_duration_s = 0.0
        self.backend_ws_url = None
        # START MODIFICATION: Add a queue and a task handle for sequential mark processing
        self.mark_ack_queue = asyncio.Queue()
        self.mark_processor_task = None
        # END MODIFICATION


class MessageRequest(BaseModel):
    owner_id: str
    user_id: str
    agent_id: str


class MessageResponse(BaseModel):
    call_sid: str
    status: str


# Store active calls
active_calls = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup logging as the first step
    setup_application_logging()
    logger = get_logger()
    logger.info("Application startup: Logging configured.")
    logger.info(f"Service starting on {HOST}:{PORT}")

    # Ensure static directory exists
    static_dir = Path("static")
    if not static_dir.exists():
        static_dir.mkdir()
        logger.info("Created static directory")

    yield
    logger.info("Application shutdown: Process finished.")


app = FastAPI(lifespan=lifespan)

# Get a logger for this module
module_logger = get_logger(__name__)


@app.middleware("http")
async def log_requests_middleware(request: Request, call_next):
    # Skip logging for static files to reduce noise
    if request.url.path.startswith("/static/"):
        return await call_next(request)

    # Generate a unique request ID if not provided
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Prepare fields for structured logging
    log_extra_fields = {
        "request_id": request_id,
        "method": request.method,
        "url": str(request.url),
        "client_host": request.client.host if request.client else "N/A",
        "client_port": request.client.port if request.client else "N/A",
    }

    # Log request start
    main_app_logger = get_logger()
    main_app_logger.info(
        f"Incoming request: {request.method} {request.url}", extra=log_extra_fields
    )

    start_time = time.time()

    try:
        response = await call_next(request)
    except Exception as e:
        process_time_ms = (time.time() - start_time) * 1000
        log_extra_fields["process_time_ms"] = f"{process_time_ms:.2f}"

        module_logger.error(
            f"Unhandled exception during request processing: {e}",
            exc_info=True,
            extra=log_extra_fields,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"request_id": request_id, "detail": "Internal Server Error"},
        )

    process_time_ms = (time.time() - start_time) * 1000
    log_extra_fields["status_code"] = response.status_code
    log_extra_fields["process_time_ms"] = f"{process_time_ms:.2f}"

    main_app_logger.info(
        f"Request completed: {request.method} {request.url} - Status {response.status_code}",
        extra=log_extra_fields,
    )

    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    return response


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
# app.include_router(production_router)
app.include_router(development_router)
# app.include_router(monitoring_router)
app.include_router(dashboard_router)
app.include_router(messages_router)
# app.include_router(chat_router)


# Call simulator API endpoints
@app.get("/api/health")
async def api_health_check():
    """API health check endpoint for the call simulator"""
    return {"status": "healthy"}


@app.post("/api/initiate-call")
async def initiate_call(call_request: CallRequest):
    """Initiate a simulated Twilio call"""
    call_sid = f"CA{uuid.uuid4().hex}"

    # Prepare the form data that Twilio would send
    form_data = {
        "CallSid": call_sid,
        "owner_id": call_request.owner_id,
        "From": call_request.user_id,
        "CallStatus": "in-progress",
        "Direction": "inbound",
        "AccountSid": f"AC{uuid.uuid4().hex[:32]}",
    }

    try:
        # Make the request to the /answer endpoint
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{TWILIO_SERVER_BASE_URL}{DEVELOPMENT_ENDPOINT}/answer",
                data=form_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()

        # Parse the TwiML response
        twiml = response.text.strip()

        # Remove BOM if present
        if twiml.startswith("\ufeff"):
            twiml = twiml[1:]

        try:
            root = ET.fromstring(twiml)
        except ET.ParseError as e:
            # Try to extract XML if there's extra content
            xml_match = re.search(r"<\?xml.*?</Response>", twiml, re.DOTALL)
            if xml_match:
                twiml = xml_match.group(0)
                root = ET.fromstring(twiml)
            else:
                raise HTTPException(
                    status_code=500, detail=f"Failed to parse TwiML: {str(e)}"
                )

        # Find the Stream element
        ws_url = None

        # Look for Stream inside Connect
        for connect in root.iter():
            if connect.tag == "Connect" or connect.tag.endswith("}Connect"):
                for child in connect:
                    if child.tag == "Stream" or child.tag.endswith("}Stream"):
                        ws_url = child.get("url") or child.get("URL")
                        if ws_url:
                            break
            if ws_url:
                break

        # Direct search for Stream element
        if not ws_url:
            for elem in root.iter():
                if elem.tag == "Stream" or elem.tag.endswith("}Stream"):
                    ws_url = elem.get("url") or elem.get("URL")
                    if ws_url:
                        break

        if not ws_url:
            raise HTTPException(
                status_code=500, detail="No WebSocket URL found in TwiML response"
            )

        # Convert relative URL to absolute if needed
        parsed_url = urlparse(ws_url)
        if not parsed_url.scheme:
            if ws_url.startswith("/"):
                base_parsed = urlparse(TWILIO_SERVER_BASE_URL)
                ws_url = f"wss://{base_parsed.netloc}{ws_url}"
            else:
                ws_url = f"wss://{parsed_url.netloc or urlparse(TWILIO_SERVER_BASE_URL).netloc}/{ws_url}"

        # Create call session
        session = CallSession(call_sid, call_request.owner_id, call_request.user_id)
        session.backend_ws_url = ws_url
        active_calls[call_sid] = session

        return {"call_sid": call_sid, "ws_url": ws_url, "status": "initiated"}

    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to initiate call: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


async def process_mark_acknowledgments(session: CallSession):
    """A worker that processes mark acknowledgments sequentially from a queue."""
    while session.is_active:
        try:
            # Wait for a mark to be added to the queue
            mark_data, delay_s = await session.mark_ack_queue.get()

            # Wait for the simulated playback duration
            await asyncio.sleep(delay_s)

            if not session.is_active:
                module_logger.info(
                    "Session became inactive while waiting to send mark. Aborting."
                )
                session.mark_ack_queue.task_done()
                continue

            mark_name = mark_data.get("mark", {}).get("name")
            module_logger.info(
                f"Sending sequential 'mark' acknowledgment for '{mark_name}'."
            )

            response_mark_event = {
                "event": "mark",
                "streamSid": session.stream_sid,
                "mark": {"name": mark_name},
            }

            if session.twilio_ws and not session.twilio_ws.closed:
                await session.twilio_ws.send(json.dumps(response_mark_event))
            else:
                module_logger.warning(
                    f"Could not send sequential mark for '{mark_name}', WebSocket is closed."
                )

            # Notify the queue that the task is done
            session.mark_ack_queue.task_done()

        except asyncio.CancelledError:
            module_logger.info("Mark processor task cancelled.")
            break
        except Exception as e:
            module_logger.error(f"Error in mark processor: {e}")


@app.websocket("/ws/{call_sid}")
async def websocket_endpoint(websocket: WebSocket, call_sid: str):
    """WebSocket endpoint for browser client"""
    await websocket.accept()

    session = active_calls.get(call_sid)
    if not session:
        await websocket.send_json({"type": "error", "message": "Invalid call session"})
        await websocket.close()
        return

    session.client_ws = websocket

    try:
        ws_url = session.backend_ws_url
        if not ws_url:
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "Backend WebSocket URL not found in session.",
                }
            )
            await websocket.close()
            return

        # Connect to Twilio WebSocket
        async with websockets.connect(
            ws_url, ping_interval=20, ping_timeout=10
        ) as twilio_ws:
            session.twilio_ws = twilio_ws
            session.is_active = True

            # START MODIFICATION: Start the sequential mark processor task
            session.mark_processor_task = asyncio.create_task(
                process_mark_acknowledgments(session)
            )
            # END MODIFICATION

            # Send start event
            start_event = {
                "event": "start",
                "sequenceNumber": "1",
                "start": {
                    "streamSid": session.stream_sid,
                    "callSid": call_sid,
                    "accountSid": f"AC{uuid.uuid4().hex[:32]}",
                    "tracks": ["inbound"],
                    "mediaFormat": {
                        "encoding": "audio/x-mulaw",
                        "sampleRate": 8000,
                        "channels": 1,
                    },
                },
                "streamSid": session.stream_sid,
            }
            await twilio_ws.send(json.dumps(start_event))

            # Notify client that connection is established
            await websocket.send_json({"type": "connected", "status": "Call connected"})

            # Create tasks for bidirectional communication
            client_to_twilio_task = asyncio.create_task(
                forward_client_to_twilio(session)
            )
            twilio_to_client_task = asyncio.create_task(
                forward_twilio_to_client(session)
            )

            # Wait for tasks to complete
            await asyncio.gather(client_to_twilio_task, twilio_to_client_task)

    except Exception as e:
        await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        session.is_active = False
        # START MODIFICATION: Cleanly shut down the mark processor task
        if (
            session
            and session.mark_processor_task
            and not session.mark_processor_task.done()
        ):
            session.mark_processor_task.cancel()
        # END MODIFICATION
        if call_sid in active_calls:
            del active_calls[call_sid]
        await websocket.close()


async def forward_client_to_twilio(session: CallSession):
    """Forward audio from browser client to Twilio WebSocket"""
    sequence_number = 2
    media_timestamp = 0

    try:
        while session.is_active:
            message = await session.client_ws.receive_json()

            if message["type"] == "audio":
                # Browser sends PCM 16-bit audio at 8kHz as base64
                pcm_data = base64.b64decode(message["audio"])

                # Convert PCM to μ-law using our new function
                pcm_array = np.frombuffer(pcm_data, dtype=np.int16)
                mulaw_data = audioop.lin2ulaw(pcm_data, 2)

                # Send to Twilio as media event
                media_event = {
                    "event": "media",
                    "sequenceNumber": str(sequence_number),
                    "media": {
                        "timestamp": str(int(media_timestamp)),
                        "payload": base64.b64encode(mulaw_data).decode("utf-8"),
                        "track": "inbound",
                    },
                    "streamSid": session.stream_sid,
                }
                await session.twilio_ws.send(json.dumps(media_event))

                sequence_number += 1
                media_timestamp += (len(pcm_array) * 1000) / 8000

            elif message["type"] == "stop":
                print(
                    f"Client sent 'stop' for call {session.call_sid}. Closing connection to backend."
                )
                stop_event = {
                    "event": "stop",
                    "sequenceNumber": str(sequence_number),
                    "streamSid": session.stream_sid,
                }
                if session.twilio_ws and not session.twilio_ws.closed:
                    try:
                        await session.twilio_ws.send(json.dumps(stop_event))
                        await session.twilio_ws.close()
                    except Exception as e:
                        print(
                            f"Exception while closing backend websocket on 'stop': {e}"
                        )
                session.is_active = False
                break

    except Exception as e:
        print(f"Error in forward_client_to_twilio: {e}")
        session.is_active = False


async def forward_twilio_to_client(session: CallSession):
    """Forward messages from Twilio WebSocket to browser client and handle marks."""
    try:
        while session.is_active:
            message = await session.twilio_ws.recv()
            data = json.loads(message)

            if data["event"] == "media":
                mulaw_payload = data["media"]["payload"]
                mulaw_data = base64.b64decode(mulaw_payload)
                session.last_media_duration_s = len(mulaw_data) / 8000.0

                # Convert mulaw to PCM using our new function
                pcm_data = audioop.ulaw2lin(mulaw_data, 2)
                pcm_base64 = base64.b64encode(pcm_data).decode("utf-8")

                await session.client_ws.send_json(
                    {
                        "type": "audio",
                        "audio": pcm_base64,
                        "timestamp": data["media"].get("timestamp", "0"),
                    }
                )

            elif data["event"] == "mark":
                # START MODIFICATION: Instead of creating a task, put the mark on the queue
                delay_with_buffer = session.last_media_duration_s + 0.15
                mark_name = data.get("mark", {}).get("name")
                module_logger.info(
                    f"Simulator received 'mark' ('{mark_name}'), queueing for acknowledgment with {delay_with_buffer:.2f}s delay."
                )
                await session.mark_ack_queue.put((data, delay_with_buffer))
                # END MODIFICATION

            elif data["event"] == "clear":
                print(
                    f"Forwarding 'clear' event for streamSid: {data.get('streamSid')}"
                )
                await session.client_ws.send_json({"type": "clear_audio"})

            elif data["event"] == "stop":
                await session.client_ws.send_json({"type": "stop"})
                session.is_active = False
                break

    except Exception as e:
        print(f"Error in forward_twilio_to_client: {e}")
        session.is_active = False


@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Serve the demo web app at the root URL"""
    req_id = request.headers.get("X-Request-ID", "N/A")
    module_logger.info(
        "Root path '/' accessed - serving demo app.", extra={"request_id": req_id}
    )

    # Check if index.html exists
    index_path = Path("static/index.html")
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        # Return a simple page if index.html doesn't exist
        return HTMLResponse(
            content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Twilio Call Simulator</title>
        </head>
        <body>
            <h1>Please create static/index.html with the call simulator UI</h1>
        </body>
        </html>
        """
        )


@app.get("/error-test")
async def error_test(request: Request):
    """Test endpoint to verify error logging"""
    req_id = request.headers.get("X-Request-ID", "N/A")
    try:
        x = 1 / 0
    except ZeroDivisionError:
        module_logger.error(
            "A test error occurred: Division by zero.",
            exc_info=True,
            extra={"request_id": req_id},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Intentional error for testing logging",
                "request_id": req_id,
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
