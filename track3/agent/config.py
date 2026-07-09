"""Configuration from environment (+ optional .env), dependency-free.

Credentials never live in the repo. Copy ``.env.example`` to ``.env`` and fill
in a *test* zone's token (scoped to Account: Cloudflare Tunnel Edit + Zone: DNS
Edit on that one zone).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    account_id: str = ""
    zone_id: str = ""
    api_token: str = ""
    metrics_addr: str = "127.0.0.1:20241"
    journal_path: str = "artifacts/journal.jsonl"

    @property
    def has_credentials(self) -> bool:
        return bool(self.account_id and self.zone_id and self.api_token)


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines); does not overwrite real env vars."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def load_config(dotenv: Optional[str] = ".env") -> Config:
    if dotenv:
        load_dotenv(dotenv)
    return Config(
        account_id=os.environ.get("CF_ACCOUNT_ID", ""),
        zone_id=os.environ.get("CF_ZONE_ID", ""),
        api_token=os.environ.get("CF_API_TOKEN", ""),
        metrics_addr=os.environ.get("CF_METRICS_ADDR", "127.0.0.1:20241"),
        journal_path=os.environ.get("TG_JOURNAL", "artifacts/journal.jsonl"),
    )
