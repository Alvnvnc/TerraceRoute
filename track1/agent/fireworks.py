"""Fireworks AI escalation client (OpenAI-compatible).

EVERY call must go through FIREWORKS_BASE_URL (rule) and use a model from
ALLOWED_MODELS (read at runtime, NEVER hardcoded). Input + output tokens both count
toward the score, so keep the prompt as dense as possible and cap max_tokens.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from .config import config
from .http import post_json


@dataclass
class RemoteResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Fireworks:
    def __init__(self) -> None:
        self.base_url = config.fireworks_base_url.rstrip("/")
        self.api_key = config.fireworks_api_key
        self.models = config.allowed_models

    def pick_model(self, prefer: Optional[str] = None) -> Optional[str]:
        """Pick a model from ALLOWED_MODELS. `prefer` = substring match (e.g. 'code')."""
        if not self.models:
            return None
        if prefer:
            for m in self.models:
                if prefer.lower() in m.lower():
                    return m
        return self.models[0]

    def generate(self, prompt: str, system: Optional[str] = None, *,
                 model: Optional[str] = None, temperature: float = 0.0,
                 max_tokens: Optional[int] = None, retries: int = 2) -> Optional[RemoteResult]:
        model = model or self.pick_model()
        if not model or not self.api_key:
            return None
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or config.remote_max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"

        last_exc = None
        for attempt in range(retries + 1):
            try:
                data = post_json(url, payload, headers=headers, timeout=config.remote_timeout_s)
                text = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage") or {}
                return RemoteResult(
                    text=text.strip(),
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    model=model,
                )
            except Exception as exc:  # noqa: BLE001 - never kill the pipeline
                last_exc = exc
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        _ = last_exc
        return None
