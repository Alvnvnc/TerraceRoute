# Unified TerraceRoute submission: Track 1 text routing + Track 2 video captioning.
# The root dispatcher selects a track from /input/tasks.json and each track keeps its
# own process, configuration, watchdog, and output contract.

FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt /tmp/requirements.txt
RUN CMAKE_ARGS="-DGGML_NATIVE=OFF" pip install --no-cache-dir -r /tmp/requirements.txt


FROM python:3.11-slim

ARG MODEL_REPO=bartowski/Qwen2.5-3B-Instruct-GGUF
ARG MODEL_FILE=Qwen2.5-3B-Instruct-Q4_K_M.gguf
ARG ESCALATION_LEVEL=0
ARG RUNTIME_LAYER_REV=1

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH=/opt/venv/bin:$PATH \
    LOCAL_BACKEND=llamacpp \
    LLAMACPP_MODEL_PATH=/models/local.gguf \
    LLAMACPP_THREADS=2 \
    LLAMACPP_CTX=4096 \
    ESCALATION_LEVEL=${ESCALATION_LEVEL} \
    WATCHDOG_S=510

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && printf '%s\n' "$RUNTIME_LAYER_REV" > /etc/terraceroute-runtime-rev

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

RUN python -c "from huggingface_hub import hf_hub_download; import os, shutil; \
os.makedirs('/models', exist_ok=True); \
p=hf_hub_download(repo_id='${MODEL_REPO}', filename='${MODEL_FILE}'); \
shutil.copy(p, '/models/local.gguf')" \
 && rm -rf /root/.cache/huggingface

COPY entrypoint.py /app/entrypoint.py
COPY track1/agent/ /app/track1/agent/
COPY track2/agent/ /app/track2/agent/

RUN python -c "from llama_cpp import Llama; Llama(model_path='/models/local.gguf', n_ctx=512, n_threads=2, verbose=False); print('warm-up OK')"

ENTRYPOINT ["python", "-B", "/app/entrypoint.py"]
