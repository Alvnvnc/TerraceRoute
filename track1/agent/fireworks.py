"""Fireworks AI escalation client (OpenAI-compatible).

EVERY call must go through FIREWORKS_BASE_URL (rule) and use a model from
ALLOWED_MODELS (read at runtime, NEVER hardcoded). Input + output tokens both count
toward the score, so keep the prompt as dense as possible and cap max_tokens.
"""
from __future__ import annotations

import time
import urllib.error
from dataclasses import dataclass
from typing import Optional, Sequence, Union

from .config import config
from .http import post_json


@dataclass
class RemoteResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    finish_reason: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def truncated(self) -> bool:
        """max_tokens cut the response off. With 2026 reasoning models this usually
        means the tokens went to thinking and the text is mid-reasoning garbage —
        never submit it over a local answer."""
        return self.finish_reason == "length"


class Fireworks:
    def __init__(self) -> None:
        self.base_url = config.fireworks_base_url.rstrip("/")
        self.api_key = config.fireworks_api_key
        self.models = config.allowed_models
        # (model, effort) pairs rejected with HTTP 400 — don't send that pair again.
        # Keyed per pair because gpt-oss rejects "none" yet accepts "low".
        self._no_effort_param: set[str] = set()

    def pick_model(self, prefer: Union[str, Sequence[str], None] = None) -> Optional[str]:
        """Pick a model from ALLOWED_MODELS. `prefer` = substring(s) tried in order,
        so callers express a ranked preference ('code' first, then 'glm', ...) while
        the runtime-injected allowlist stays authoritative."""
        if not self.models:
            return None
        prefs = [prefer] if isinstance(prefer, str) else list(prefer or ())
        for p in prefs:
            for m in self.models:
                if p.lower() in m.lower():
                    return m
        return self.models[0]

    def generate(self, prompt: str, system: Optional[str] = None, *,
                 model: Optional[str] = None, temperature: float = 0.0,
                 max_tokens: Optional[int] = None, retries: int = 2,
                 reasoning_effort: Optional[str] = None) -> Optional[RemoteResult]:
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
        # Measured (fw_probe2, 2026-07-11): every current serverless model is a
        # reasoning model; without this, thinking eats the tiny per-category token
        # caps and the answer arrives truncated or empty. glm/deepseek/kimi accept
        # "none" (answers drop to 2-7 completion tokens); models that 400 on the
        # param (gpt-oss rejects "none" but takes "low") are remembered and called
        # without it. Per-call override: logical puzzles NEED some thinking.
        effort = reasoning_effort or config.reasoning_effort
        if effort and f"{model}:{effort}" not in self._no_effort_param:
            payload["reasoning_effort"] = effort
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = f"{self.base_url}/chat/completions"

        last_exc = None
        for attempt in range(retries + 1):
            try:
                data = post_json(url, payload, headers=headers, timeout=config.remote_timeout_s)
                choice = data["choices"][0]
                text = choice["message"]["content"] or ""
                usage = data.get("usage") or {}
                return RemoteResult(
                    text=text.strip(),
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    model=model,
                    finish_reason=str(choice.get("finish_reason") or ""),
                )
            except urllib.error.HTTPError as exc:
                if exc.code == 400 and "reasoning_effort" in payload:
                    # Value unsupported by this model — drop it and retry immediately.
                    self._no_effort_param.add(f"{model}:{payload['reasoning_effort']}")
                    payload.pop("reasoning_effort")
                    continue
                last_exc = exc
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
            except Exception as exc:  # noqa: BLE001 - never kill the pipeline
                last_exc = exc
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        _ = last_exc
        return None
