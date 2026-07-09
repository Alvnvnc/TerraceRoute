"""Client Ollama lokal: chat (+ logprobs bila didukung) & embeddings.

Ollama >= 0.12.11 mengembalikan logprobs. Versi lebih lama tidak — dalam kasus
itu `avg_logprob` = None dan router jatuh ke self-consistency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import httpx

from .config import settings


@dataclass
class LocalGeneration:
    text: str
    avg_logprob: Optional[float]   # rata-rata log-prob token output, None bila tak tersedia
    min_logprob: Optional[float]   # log-prob token paling ragu (sinyal error tertajam)
    prompt_tokens: int
    completion_tokens: int


class OllamaClient:
    def __init__(self, host: Optional[str] = None, model: Optional[str] = None):
        self.host = (host or settings.ollama_host).rstrip("/")
        self.model = model or settings.local_model

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        want_logprobs: Optional[bool] = None,
        num_predict: Optional[int] = None,
    ) -> LocalGeneration:
        want = settings.ollama_logprobs if want_logprobs is None else want_logprobs
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options = {"temperature": temperature}
        if num_predict is not None:
            options["num_predict"] = num_predict     # cap output (mis. judge YES/NO)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if want:
            # Didukung di Ollama baru; diabaikan diam-diam oleh versi lama.
            payload["logprobs"] = True

        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(f"{self.host}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

        text = (data.get("message") or {}).get("content", "")
        lps = _extract_token_logprobs(data)
        avg_logprob = sum(lps) / len(lps) if lps else None
        min_logprob = min(lps) if lps else None
        return LocalGeneration(
            text=text,
            avg_logprob=avg_logprob,
            min_logprob=min_logprob,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
        )

    async def embed(self, text: str) -> list[float]:
        text = (text or "").strip()[:4000]      # guard: input kosong/terlalu panjang -> 500
        if not text:
            return []
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.host}/api/embeddings",
                json={"model": settings.embed_model, "prompt": text},
            )
            resp.raise_for_status()
            return resp.json().get("embedding", [])


def _extract_token_logprobs(data: dict) -> Optional[list[float]]:
    """Ambil daftar logprob per-token dari respons Ollama.

    Format logprobs Ollama masih berkembang; kita defensif terhadap beberapa
    bentuk yang mungkin. Kembalikan None kalau tak ditemukan.
    """
    msg = data.get("message") or {}
    lp = msg.get("logprobs") or data.get("logprobs")
    if not lp:
        return None
    # Bentuk 1: list of {token, logprob}
    if isinstance(lp, list) and lp and isinstance(lp[0], dict) and "logprob" in lp[0]:
        vals = [x["logprob"] for x in lp if x.get("logprob") is not None]
        return vals or None
    # Bentuk 2: {"content": [{logprob: ...}, ...]} (ala OpenAI)
    content = lp.get("content") if isinstance(lp, dict) else None
    if isinstance(content, list) and content:
        vals = [x["logprob"] for x in content if isinstance(x, dict) and x.get("logprob") is not None]
        return vals or None
    return None
