"""HTTP kecil berbasis stdlib (urllib) — tanpa dependency eksternal.

Membuat image ramping & bebas instalasi pihak ketiga untuk lapisan jaringan.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Optional


def post_json(url: str, payload: dict, headers: Optional[dict] = None,
              timeout: float = 60.0, retries: int = 2) -> dict[str, Any]:
    """POST JSON, kembalikan dict hasil parse. Melempar pada error HTTP/jaringan.

    Retry singkat (backoff 1s, 3s) pada error transien — termasuk balasan non-JSON
    (proxy/gateway yang membalas halaman HTML error). Percobaan terakhir melempar
    apa adanya agar caller bisa fallback.
    """
    body = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    last_exc: Exception = RuntimeError("unreachable")
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1.0 + 2.0 * attempt)
    raise last_exc
