"""Model lokal (GRATIS — token lokal tidak dihitung skor).

Dua backend, dipilih via LOCAL_BACKEND:
  - "ollama":   dev cepat, tembak server Ollama host (OpenAI-compatible /v1).
  - "llamacpp": prod in-process, GGUF kecil (Qwen2.5-3B Q4) via llama-cpp-python.

Antarmuka seragam: .chat(system, user, max_tokens, temperature) -> str
"""
from __future__ import annotations

import json
from typing import Optional

from .config import config
from .http import post_json


class LocalLLM:
    def __init__(self) -> None:
        self.backend = config.local_backend
        self._llama = None  # lazy untuk llamacpp

    # --- backend: ollama (dev) ---
    def _chat_ollama(self, system: Optional[str], user: str, max_tokens: int,
                     temperature: float) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        payload = {
            "model": config.local_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        url = config.ollama_host.rstrip("/") + "/v1/chat/completions"
        data = post_json(url, payload, timeout=120)
        return data["choices"][0]["message"]["content"] or ""

    # --- backend: llamacpp (prod) ---
    def _ensure_llama(self):
        if self._llama is None:
            from llama_cpp import Llama  # import lokal agar dev tanpa lib tetap jalan
            self._llama = Llama(
                model_path=config.llamacpp_model_path,
                n_ctx=config.llamacpp_ctx,
                n_threads=config.llamacpp_threads,
                n_gpu_layers=0,          # CPU-only (scoring env tanpa GPU)
                verbose=False,
            )
        return self._llama

    def _chat_llamacpp(self, system: Optional[str], user: str, max_tokens: int,
                       temperature: float) -> str:
        llama = self._ensure_llama()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        out = llama.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature,
        )
        return out["choices"][0]["message"]["content"] or ""

    def chat(self, user: str, system: Optional[str] = None, max_tokens: int = 512,
             temperature: float = 0.0) -> str:
        try:
            if self.backend == "llamacpp":
                return self._chat_llamacpp(system, user, max_tokens, temperature).strip()
            return self._chat_ollama(system, user, max_tokens, temperature).strip()
        except Exception:
            return ""  # jangan pernah crash pipeline karena model lokal

    def json_chat(self, user: str, system: Optional[str] = None, max_tokens: int = 256) -> Optional[dict]:
        """Minta output JSON; parse toleran (ambil objek {...} pertama)."""
        raw = self.chat(user, system=system, max_tokens=max_tokens, temperature=0.0)
        if not raw:
            return None
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            return None
