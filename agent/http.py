"""HTTP kecil berbasis stdlib (urllib) — tanpa dependency eksternal.

Membuat image ramping & bebas instalasi pihak ketiga untuk lapisan jaringan.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Optional


def post_json(url: str, payload: dict, headers: Optional[dict] = None,
              timeout: float = 60.0) -> dict[str, Any]:
    """POST JSON, kembalikan dict hasil parse. Melempar pada error HTTP/jaringan."""
    body = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(url, data=body, headers=hdrs, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
