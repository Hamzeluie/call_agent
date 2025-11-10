# call_agent
This repository is orchestrator of call and message AI agent.

call agent component:

* **VAD**: need **voice activity detection** to detect where user speack and where not speack.
* **STT**: need to transcription of **Speech To Text**.
* **KB**: for using a rag system we need a **Knowledge Base** database.
* **LLM**: an **LLM** should be core or brain of the agent.
* **TTS**: need transcription of **Text To Speech**. to responce as voice to user.

chat agent conponent:
* **KB**: for using a rag system we need a **Knowledge Base** database.
* **LLM**: an **LLM** should be core or brain of the agent.


## Repository files

**orchestrator.py**: starting point of the repository.

**call_development.py**: call router file which is called in the **orchestrator.py**

**call_agent.py**: locate class of call agent

**chat_development.py**: chat router file which is called in the **orchestrator.py**

**chat_agent.py**: locate class of chat agent

**dashboard.py**: router of the dashboard. (input context from 
users)



## parameters

there is 5 microservice (VAD,STT,KB,LLM,TTS). connected to the repository. each microservice **HOST** and **PORT** should be declare in the **config.yaml**


# Install .venv
you can install virtual environment with **poetry**

        poetry install --no-root

# Run orchestrator.py
if you use poetry you can run main.py with:

        poetry run python orchestrator.py



# Docker
you can build docker image and run a container with the image.
Docker file Expose port is **8000**

with:


        docker build --no-cache -t orchestrator-server .
        docker run -it --gpus all -p 8000:8000 orchestrator-server


# Input/Output structure
there is five microservice used in this repository totally. in this section i introduce input/output structure of each service.

## VAD
"ws://{self.yaml_config['vad']['host']}:{self.yaml_config['vad']['port']}/ws/vad"
### Input structure

        {
                "type": "input_audio_buffer.append", 
                "audio": audio_b64
        }
### Output structure

## TTS
"ws://{self.yaml_config['stt']['host']}:{self.yaml_config['stt']['port']}/ws/stt"
### Input structure

        {
                "type": "audio.append",
                "sample_rate": integer,
                "audio": audio_int16,
        }
### Output structure


## KB
"ws://{self.yaml_config['db']['host']}:{self.yaml_config['db']['port']}/ws/search/{self.owner_id}"
### Input structure

        {
                "query_text": string,
                "kb_id": integer,
                "limit": integer,
        }
### Output structure

## LLM
"ws://{self.yaml_config['llm']['host']}:{self.yaml_config['llm']['port']}/ws/llm/{self.owner_id}/{self.session_id}"
### Input structure

        {
                "owner_id": integer,
                "user_input": string,
                "retrieved_data": string,
                "interrupt": boolian,
        }
### Output structure
in starting speech

        {
                "type": "input_audio_buffer.speech_started"
        }

in stopping speech

        {
                "type": "input_audio_buffer.speech_stopped"
        }
in speech segment

        {
                "type": "speech_segment",
                "audio": segment_b64,
                "sample_rate": VAD_SAMPLE_RATE
        }
OR in ERROR

        {
                "type": "error", 
                "message": "Server models not initialized."
        }


## STT
"ws://{self.yaml_config['tts']['host']}:{self.yaml_config['tts']['port']}"
### Input structure
simple text

### Output structure

        {
                "type": "response.audio.delta",
                "delta": audio_chunk_b64,
                "item_id": string,
        }
in end of speech

        {
                "type": "response.done",
                "item_id": integer,
        }