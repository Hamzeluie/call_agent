# web_ui.py
import asyncio
import base64
import json
import logging
from typing import Dict

import librosa
import numpy as np
from call_agent import CallAgent
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
active_agents: Dict[str, CallAgent] = {}
YAML_PATH = "config/config.yaml"


def resample_audio_if_needed(
    audio_bytes: bytes, original_rate: int, target_rate: int = 16000
) -> bytes:
    if original_rate == target_rate:
        return audio_bytes
    samples = np.frombuffer(audio_bytes, dtype=np.int16)
    samples_float = samples.astype(np.float32) / 32768.0
    resampled = librosa.resample(
        samples_float, orig_sr=original_rate, target_sr=target_rate
    )
    resampled_int16 = (resampled * 32768).astype(np.int16)
    return resampled_int16.tobytes()


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Agent</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; }
            button { padding: 12px 24px; font-size: 18px; margin: 10px; cursor: pointer; }
            #status { margin: 20px 0; padding: 10px; background: #f0f0f0; border-radius: 4px; }
            #transcript, #response { margin: 10px 0; padding: 10px; background: #eef; border-left: 4px solid #007; }
        </style>
    </head>
    <body>
        <h1>🎙️ Voice Agent</h1>
        <button id="startBtn">Start Call</button>
        <button id="endBtn" disabled>End Call</button>
        
        <div id="status">Status: Idle</div>
        <div id="transcript"><strong>You:</strong> <span id="userText">-</span></div>
        <div id="response"><strong>Agent:</strong> <span id="agentText">-</span></div>

        <script>
            let userAudioContext = null;
            let processor = null;
            let websocket = null;
            let isProcessing = false;
            const playbackContext = new (window.AudioContext || window.webkitAudioContext)();

            const startBtn = document.getElementById('startBtn');
            const endBtn = document.getElementById('endBtn');
            const statusDiv = document.getElementById('status');
            const userText = document.getElementById('userText');
            const agentText = document.getElementById('agentText');

            function floatTo16BitPCM(input) {
                const buffer = new ArrayBuffer(input.length * 2);
                const view = new DataView(buffer);
                for (let i = 0; i < input.length; i++) {
                    const s = Math.max(-1, Math.min(1, input[i]));
                    view.setInt16(i * 2, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                }
                return buffer;
            }

            function playTtsAudio(base64Audio) {
                try {
                    const binary = atob(base64Audio);
                    const bytes = new Uint8Array(binary.length);
                    for (let i = 0; i < binary.length; i++) {
                        bytes[i] = binary.charCodeAt(i);
                    }
                    const buffer = playbackContext.createBuffer(1, bytes.length / 2, 16000);
                    const channelData = buffer.getChannelData(0);
                    for (let i = 0; i < channelData.length; i++) {
                        const int16 = (bytes[i * 2] | (bytes[i * 2 + 1] << 8));
                        channelData[i] = int16 / 32768.0;
                    }
                    const source = playbackContext.createBufferSource();
                    source.buffer = buffer;
                    source.connect(playbackContext.destination);
                    source.start();
                } catch (e) {
                    console.error("TTS playback error:", e);
                }
            }

            startBtn.onclick = async () => {
                try {
                    statusDiv.textContent = "Status: Connecting...";
                    websocket = new WebSocket(`ws://${window.location.host}/ws`);

                    websocket.onopen = () => {
                        statusDiv.textContent = "Status: Connected!";
                        startAudioCapture();
                        startBtn.disabled = true;
                        endBtn.disabled = false;
                    };

                    websocket.onmessage = (event) => {
                        const msg = JSON.parse(event.data);
                        if (msg.type === "transcript") {
                            userText.textContent = msg.text;
                        } else if (msg.type === "response") {
                            agentText.textContent = msg.text;
                        } else if (msg.type === "tts_audio") {
                            playTtsAudio(msg.audio);  // 👈 PLAY TTS AUDIO
                        } else if (msg.type === "status") {
                            statusDiv.textContent = "Status: " + msg.text;
                        }
                    };

                    websocket.onclose = () => {
                        statusDiv.textContent = "Status: Disconnected";
                        stopAudioCapture();
                        startBtn.disabled = false;
                        endBtn.disabled = true;
                    };
                } catch (e) {
                    console.error(e);
                    statusDiv.textContent = "Status: Error - " + e.message;
                }
            };

            endBtn.onclick = () => {
                if (websocket) websocket.close();
            };

            async function startAudioCapture() {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                // Force 16kHz if possible
                const contextOptions = { sampleRate: 16000 };
                userAudioContext = new (window.AudioContext || window.webkitAudioContext)(contextOptions);
                const source = userAudioContext.createMediaStreamSource(stream);
                processor = userAudioContext.createScriptProcessor(4096, 1, 1);

                processor.onaudioprocess = (e) => {
                    if (!isProcessing || !websocket || websocket.readyState !== WebSocket.OPEN) return;
                    const inputData = e.inputBuffer.getChannelData(0);
                    const volume = Math.sqrt(inputData.reduce((sum, x) => sum + x * x, 0) / inputData.length);
                    if (volume < 0.01) return; // Skip silence
                    const pcmBuffer = floatTo16BitPCM(inputData);
                    const pcmBase64 = btoa(String.fromCharCode(...new Uint8Array(pcmBuffer)));
                    websocket.send(JSON.stringify({ type: "audio", audio: pcmBase64 }));
                };

                source.connect(processor);
                processor.connect(userAudioContext.destination);
                isProcessing = true;
            }

            function stopAudioCapture() {
                isProcessing = false;
                if (processor) {
                    processor.disconnect();
                    processor = null;
                }
                if (userAudioContext) {
                    userAudioContext.close();
                    userAudioContext = null;
                }
            }

            window.onbeforeunload = () => {
                if (websocket) websocket.close();
            };
        </script>
    </body>
    </html>
    """


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(id(websocket))
    agent = None
    try:
        agent = CallAgent(
            owner_id="web_user",
            system_prompt="You are a helpful voice assistant. Keep responses concise.",
            yaml_path=YAML_PATH,
            kb_id=[],
            config={"temperature": 0.7, "max_tokens": 500},
        )
        await agent.connect_servers()
        active_agents[session_id] = agent

        receiver_tasks = await agent.start_receivers()

        # 👇 SEND WELCOME MESSAGE
        welcome_text = "Hello! I am an AI assistant. How can I help you today?"
        await websocket.send_json({"type": "response", "text": welcome_text})

        # Convert text to speech via TTS server
        if "tts" in agent.websockets:
            await agent.websockets["tts"].send(welcome_text)
            # Wait briefly for TTS to process
            await asyncio.sleep(0.1)

        async def forward_messages():
            last_transcript = None
            last_response_text = welcome_text  # 👈 include welcome
            sent_audio_items = set()

            while not agent.end_call:
                if agent.last_transcript and agent.last_transcript != last_transcript:
                    await websocket.send_json(
                        {"type": "transcript", "text": agent.last_transcript}
                    )
                    last_transcript = agent.last_transcript

                if agent.llm_response != last_response_text:
                    await websocket.send_json(
                        {"type": "response", "text": agent.llm_response}
                    )
                    last_response_text = agent.llm_response

                # Forward TTS audio chunks
                for tts_msg in list(agent.formatted_audio_responses):
                    item_id = tts_msg.get("item_id")
                    if item_id and item_id not in sent_audio_items:
                        await websocket.send_json(
                            {"type": "tts_audio", "audio": tts_msg["delta"]}
                        )
                        sent_audio_items.add(item_id)
                        agent.formatted_audio_responses.remove(tts_msg)

                await asyncio.sleep(0.05)

            await websocket.send_json({"type": "status", "text": "Call ended"})

        forward_task = asyncio.create_task(forward_messages())

        # ... rest of audio handling
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg["type"] == "audio":
                try:
                    # Decode base64 to bytes
                    audio_bytes = base64.b64decode(msg["audio"])
                    # Most browsers use 48kHz by default
                    samples = np.frombuffer(audio_bytes, dtype=np.int16)
                    samples_float = samples.astype(np.float32) / 32768.0
                    resampled = librosa.resample(
                        samples_float, orig_sr=48000, target_sr=16000
                    )
                    resampled_int16 = (resampled * 32768).astype(np.int16)
                    audio_b64 = base64.b64encode(resampled_int16.tobytes()).decode(
                        "utf-8"
                    )
                    await agent.send_audio_chunk(audio_b64)
                except Exception as e:
                    logger.error(f"Resampling failed: {e}")

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if agent:
            await agent.close()
        if session_id in active_agents:
            del active_agents[session_id]
        await websocket.close()
