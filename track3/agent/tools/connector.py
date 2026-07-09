"""Manage the local ``cloudflared`` connector process and read its health.

Split into pure parsers (testable without a subprocess) and thin runtime
helpers (subprocess + local HTTP to the metrics server).
"""

from __future__ import annotations

import json
import re
import subprocess
import urllib.error
import urllib.request
from typing import Optional

_QUICK_URL_RE = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com")
_METRICS_ADDR_RE = re.compile(r"metrics server on\s+([0-9.]+:\d+)")


# --------------------------------------------------------------------------- #
# Pure parsers
# --------------------------------------------------------------------------- #
def parse_quick_tunnel_url(text: str) -> Optional[str]:
    """Extract the ``https://<random>.trycloudflare.com`` URL cloudflared prints
    to stderr for a Quick Tunnel."""
    m = _QUICK_URL_RE.search(text or "")
    return m.group(0) if m else None


def parse_metrics_addr(text: str) -> Optional[str]:
    """Extract the metrics server address from cloudflared's startup log line
    ``Starting metrics server on 127.0.0.1:20241/metrics``."""
    m = _METRICS_ADDR_RE.search(text or "")
    return m.group(1) if m else None


def parse_log_signatures(text: str) -> set[str]:
    """Detect known failure markers in cloudflared output (text or JSON logs)."""
    sigs: set[str] = set()
    low = (text or "").lower()
    if "connection refused" in low or "actively refused" in low:
        sigs.add("connection_refused")
    if "x509" in low:
        sigs.add("x509")
    if "unauthorized" in low or "failed to authenticate" in low:
        sigs.add("unauthorized")
    if "credentials file" in low and ("doesn't exist" in low or "not a file" in low):
        sigs.add("credentials_missing")
    return sigs


def parse_ha_connections(metrics_text: str) -> Optional[int]:
    """Read the ``cloudflared_tunnel_ha_connections`` gauge from /metrics text."""
    for line in (metrics_text or "").splitlines():
        if line.startswith("cloudflared_tunnel_ha_connections"):
            try:
                return int(float(line.split()[-1]))
            except (ValueError, IndexError):
                return None
    return None


# --------------------------------------------------------------------------- #
# Runtime helpers
# --------------------------------------------------------------------------- #
def query_ready(metrics_addr: str, *, timeout: float = 3.0) -> tuple[Optional[int], Optional[int]]:
    """Return ``(http_status, readyConnections)`` from the /ready endpoint.
    ``(None, None)`` if the metrics port is not reachable (process likely dead).
    """
    url = f"http://{metrics_addr}/ready"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode(errors="replace")
            ready = _ready_connections(body)
            return resp.status, ready
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        return exc.code, _ready_connections(body)
    except urllib.error.URLError:
        return None, None


def query_ha_connections(metrics_addr: str, *, timeout: float = 3.0) -> Optional[int]:
    url = f"http://{metrics_addr}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return parse_ha_connections(resp.read().decode(errors="replace"))
    except (urllib.error.URLError, urllib.error.HTTPError):
        return None


def _ready_connections(body: str) -> Optional[int]:
    try:
        return json.loads(body).get("readyConnections")
    except (json.JSONDecodeError, AttributeError, ValueError):
        return None


class ConnectorManager:
    """Start/stop a ``cloudflared`` connector and remember its metrics address."""

    def __init__(self, binary: str = "cloudflared"):
        self.binary = binary
        self.proc: Optional[subprocess.Popen] = None
        self.metrics_addr: Optional[str] = None
        self.quick_url: Optional[str] = None

    def run_token(self, token: str, metrics_addr: str = "127.0.0.1:20241") -> None:
        """Start a remotely-managed connector with a run token."""
        self.metrics_addr = metrics_addr
        self.proc = subprocess.Popen(
            [self.binary, "tunnel", "--no-autoupdate", "--metrics", metrics_addr,
             "--loglevel", "info", "--log-format", "json", "run", "--token", token],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )

    def run_quick(self, port: int, scheme: str = "http", read_lines: int = 40) -> Optional[str]:
        """Start a Quick Tunnel to ``localhost:port`` and return its public URL
        (the API-less demo fallback). Reads stderr until the URL appears."""
        self.proc = subprocess.Popen(
            [self.binary, "tunnel", "--no-autoupdate", "--url",
             f"{scheme}://localhost:{port}"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        assert self.proc.stdout is not None
        for _ in range(read_lines):
            line = self.proc.stdout.readline()
            if not line:
                break
            url = parse_quick_tunnel_url(line)
            if url:
                self.quick_url = url
                return url
        return None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None
