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
