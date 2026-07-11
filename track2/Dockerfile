# Track 2 — Video Captioning Agent. CPU-only, API-driven (self-hosted Radeon backend).
# Batch: read /input/tasks.json → write /output/results.json. See plan.md.
#
# No Python dependencies beyond the stdlib; ffmpeg does the frame sampling.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    WATCHDOG_S=510

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY agent/ /app/agent/

ENTRYPOINT ["python", "-m", "agent.run"]
