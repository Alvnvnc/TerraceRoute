# TerraceRoute Track-1 agent — CPU-only, targets the 4 GB RAM / 2 vCPU scoring env.
# Batch: read /input/tasks.json → write /output/results.json. See docs/escalation-math.md.
#
# Multi-stage: the builder compiles llama-cpp-python from source for glibc (the prebuilt
# "cpu" wheel is musl-linked → won't run on the slim/glibc image); the runtime stays lean.

# ---------- builder ----------
FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

COPY requirements.txt /tmp/req.txt
# GGML_NATIVE=OFF → portable (don't bake in build-time host CPU instructions that may be
# absent on the scoring machine). Compiling from source keeps it glibc-compatible.
RUN CMAKE_ARGS="-DGGML_NATIVE=OFF" pip install --no-cache-dir -r /tmp/req.txt

# ---------- runtime ----------
FROM python:3.11-slim

ARG MODEL_REPO=bartowski/Qwen2.5-3B-Instruct-GGUF
ARG MODEL_FILE=Qwen2.5-3B-Instruct-Q4_K_M.gguf

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    PATH=/opt/venv/bin:$PATH \
    LOCAL_BACKEND=llamacpp \
    LLAMACPP_MODEL_PATH=/models/local.gguf \
    LLAMACPP_THREADS=2 \
    LLAMACPP_CTX=4096 \
    ESCALATION_LEVEL=0 \
    WATCHDOG_S=510

# libgomp1 = runtime OpenMP for the compiled llama.cpp (libstdc++6 ships in the base image).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Download the model at build time → self-contained image (no internet needed at scoring).
RUN python -c "from huggingface_hub import hf_hub_download; import shutil, os; \
os.makedirs('/models', exist_ok=True); \
p=hf_hub_download(repo_id='${MODEL_REPO}', filename='${MODEL_FILE}'); \
shutil.copy(p, '/models/local.gguf')" \
 && rm -rf /root/.cache/huggingface

COPY agent/ /app/agent/

# Warm-up: fail fast at BUILD time if the model cannot load (rather than at scoring time).
RUN python -c "from llama_cpp import Llama; Llama(model_path='/models/local.gguf', n_ctx=512, n_threads=2, verbose=False); print('warm-up OK')"

# Declared last so leaderboard walk-down variants (L0/L2/L3) share every heavy layer above.
ARG ESCALATION_LEVEL=0
ENV ESCALATION_LEVEL=${ESCALATION_LEVEL}

ENTRYPOINT ["python", "-m", "agent.run"]
