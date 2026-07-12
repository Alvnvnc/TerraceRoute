"""Tiny stdlib-based HTTP (urllib) — no external dependencies.

Keeps the image lean and the network layer free of third-party installs.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Optional


def post_json(url: str, payload: dict, headers: Optional[dict] = None,
              timeout: float = 60.0, retries: int = 2,
              deadline: Optional[float] = None) -> dict[str, Any]:
    """POST JSON, return the parsed dict. Raises on HTTP/network error.

    Short retry (backoff 1s, 3s) on transient errors — including non-JSON replies
    (a proxy/gateway returning an HTML error page). The final attempt raises as-is
    so the caller can fall back.

    `deadline` (time.monotonic() value) hard-bounds the whole call including
    retries: each attempt's socket timeout shrinks to the remaining budget and no
    retry starts once the deadline has passed — a hung provider can no longer eat
    another clip's share of the 10-minute wall clock.
    """
    body = json.dumps(payload).encode("utf-8")
    # Cloudflare may reject Python's default urllib signature (edge code 1010).
    hdrs = {"Content-Type": "application/json", "User-Agent": "TerraceRoute/1.0"}
    if headers:
        hdrs.update(headers)
    last_exc: Exception = RuntimeError("unreachable")
    for attempt in range(retries + 1):
        eff_timeout = timeout
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 2.0:
                raise last_exc if attempt else TimeoutError("budget exhausted")
            eff_timeout = min(timeout, remaining)
        try:
            req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
            with urllib.request.urlopen(req, timeout=eff_timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1.0 + 2.0 * attempt)
    raise last_exc


def download(url: str, dest: str, timeout: float = 60.0, retries: int = 1) -> bool:
    """Download a file to dest. True on success; never raises."""
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return True
        except Exception:  # noqa: BLE001
            if attempt < retries:
                time.sleep(1.0)
    return False
