FROM python:3.10-slim

# System dependencies, creates env files needed for llama-ccp and others
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libsndfile1 \
    build-essential \
    cmake \
    curl \
    unzip \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /models

# Install python deps + models
COPY requirements.txt .
RUN pip install -r requirements.txt


RUN echo "Downloading Vosk STT model..." && \
    curl -L -o /models/vosk-model.zip \
      https://huggingface.co/rhasspy/vosk-models/resolve/main/en/vosk-model-small-en-us-0.15.zip && \
    unzip /models/vosk-model.zip -d /models && \
    rm /models/vosk-model.zip


RUN echo "Downloading Phi-3 Mini GGUF..." && \
    curl -L -o /models/Phi-3-mini-4k-instruct-q4.gguf \
        https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf



# Copy server code & models
COPY app/server.py .
COPY models /models

# Workspace folder for saving outputs
RUN mkdir -p /workspace

# if you need to change port:
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
