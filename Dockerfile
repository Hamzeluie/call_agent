# Use Ubuntu 22.04 as base image
FROM ubuntu:22.04

# Set environment variables to avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    POETRY_VERSION=2.1.3

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.10 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Make python3.11 default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set work directory
WORKDIR /app

# Copy only requirements files first to leverage Docker cache
COPY pyproject.toml poetry.lock* ./

# Install project dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-root

# Copy the rest of the application
COPY . .

# Expose the port Uvicorn will run on
EXPOSE 80

COPY .env.defaults .
# Command to run the application with hot reload
CMD ["poetry", "run", "python", "orchestrator.py"]
