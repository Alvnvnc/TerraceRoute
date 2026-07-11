"""OpenAI-compatible chat client for the self-hosted Radeon backend (+ fallbacks).

Same shape as Track 1's fireworks.py: bounded retries, never raises, the caller always
gets None on failure and falls back. Fireworks is reserved for generator calls so a
Gemma fallback never grades its own captions. keep_alive=-1 pins the primary models in
VRAM (48GB W7900 holds generator + checker together) so no call pays model-load latency
twice.
"""
from __future__ import annotations

import time
from typing import Optional

from .config import config
from .http import post_json


def _messages(prompt: str, images_b64: Optional[list[str]] = None,
              system: Optional[str] = None,
              image_labels: Optional[list[str]] = None) -> list[dict]:
    content: list[dict] | str
    if images_b64:
        # Interleave a text label BEFORE each image: VLMs mis-count bare image
        # sequences, and a per-image timestamp binds each frame to its moment.
        content = [{"type": "text", "text": prompt}]
        for i, img in enumerate(images_b64):
            if image_labels and i < len(image_labels):
                content.append({"type": "text", "text": image_labels[i]})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img}"}})
    else:
        content = prompt
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": content})
    return msgs


def _text(data: dict) -> Optional[str]:
    return (data["choices"][0]["message"]["content"] or "").strip() or None


def _fireworks_response_format(json_schema: Optional[dict]) -> Optional[dict]:
    if not json_schema:
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "caption_response",
            "schema": json_schema,
        },
    }


def _fireworks_chat(payload: dict, json_schema: Optional[dict],
                    deadline: Optional[float] = None) -> Optional[str]:
    """Send a generator request to the configured Fireworks Gemma deployment."""
    if not config.has_fireworks_gemma:
        return None
    try:
        fireworks_payload = {
            **payload,
            "model": config.fireworks_gemma_model,
        }
        if json_schema:
            fireworks_payload["response_format"] = _fireworks_response_format(json_schema)
        data = post_json(
            f"{config.fireworks_base_url.rstrip('/')}/chat/completions",
            fireworks_payload,
            headers={"Authorization": f"Bearer {config.fireworks_api_key}"},
            timeout=config.request_timeout_s, deadline=deadline)
        return _text(data)
    except Exception:  # noqa: BLE001
        return None


def chat(prompt: str, *, images_b64: Optional[list[str]] = None,
          image_labels: Optional[list[str]] = None,
          system: Optional[str] = None, model: Optional[str] = None,
          temperature: float = 0.4, max_tokens: int = 400,
          json_mode: bool = False, json_schema: Optional[dict] = None,
          allow_fireworks: bool = True, prefer_fireworks: bool = False,
          budget_s: Optional[float] = None) -> Optional[str]:
    """One chat completion against the primary backend, then the fallback. None on failure.

    `budget_s` bounds the WHOLE call (all providers, all retries) so a slow provider
    cannot spend another stage's share of the clip budget."""
    deadline = (time.monotonic() + budget_s) if budget_s else None

    def out_of_budget() -> bool:
        return deadline is not None and deadline - time.monotonic() <= 2.0

    payload: dict = {
        "messages": _messages(prompt, images_b64, system, image_labels),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    # Verifier-guided repair deliberately asks the stronger remote Gemma to look at
    # the original frames again before the local description loses visual detail.
    if allow_fireworks and prefer_fireworks and not out_of_budget():
        text = _fireworks_chat(payload, json_schema, deadline)
        if text:
            return text

    # Primary: self-hosted Ollama on the Radeon (token as query param).
    url = f"{config.vlm_base_url.rstrip('/')}/chat/completions"
    if config.vlm_token:
        url += f"?token={config.vlm_token}"
    if not out_of_budget():
        try:
            data = post_json(url, {**payload, "model": model or config.vlm_model,
                                   "keep_alive": -1},
                             timeout=config.request_timeout_s, deadline=deadline)
            text = (data["choices"][0]["message"]["content"] or "").strip()
            if text:
                return text
        except Exception:  # noqa: BLE001
            pass

    # First fallback: an image-capable Gemma deployment on Fireworks. Structured
    # outputs make partial style maps a retriable provider failure, not a silent score loss.
    if allow_fireworks and not prefer_fireworks and not out_of_budget():
        text = _fireworks_chat(payload, json_schema, deadline)
        if text:
            return text

    # Last fallback: a separately configured OpenAI-compatible provider. The checker
    # gets a distinct model name so it cannot silently use the caption generator.
    fallback_model = config.fb_model if model is None else config.fb_checker_model
    has_fallback = config.has_fallback if model is None else config.has_checker_fallback
    if has_fallback and not out_of_budget():
        try:
            data = post_json(
                f"{config.fb_base_url.rstrip('/')}/chat/completions",
                {**payload, "model": fallback_model},
                headers={"Authorization": f"Bearer {config.fb_api_key}"},
                timeout=config.request_timeout_s, deadline=deadline)
            return _text(data)
        except Exception:  # noqa: BLE001
            pass
    return None


def warm_up() -> None:
    """Load generator + checker into VRAM before the first real task (fire-and-forget)."""
    try:
        chat("Reply with OK.", model=config.vlm_model, max_tokens=4, temperature=0.0)
    except Exception:  # noqa: BLE001
        pass
    try:
        chat("Reply with OK.", model=config.checker_model, max_tokens=4,
             temperature=0.0, allow_fireworks=False)
    except Exception:  # noqa: BLE001
        pass
