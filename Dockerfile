FROM python:3.11-slim

WORKDIR /app

# Dependency dulu supaya layer cache efektif
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY eval/ ./eval/
COPY artifacts/ ./artifacts/

EXPOSE 8000

# OLLAMA_HOST & FIREWORKS_API_KEY di-inject saat runtime (compose / -e)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
