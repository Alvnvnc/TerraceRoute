"""A tiny, dependency-free Ollama chat client for constrained JSON planning.

The agent never lets the model touch Cloudflare; the model only emits a typed
:class:`~agent.brain.schemas.Plan`. This module is the narrow bridge to a *local*
Ollama server (so the Cloudflare token never leaves the machine — the whole point
of running on the AMD GPU).

Design choices, per plan.md section 6:

* **Constrain format, not the answer.** We pass ``PLAN_JSON_SCHEMA`` as Ollama's
  ``format`` so the output is guaranteed syntactically valid JSON, but the schema
  leads with a free-text ``reasoning`` field so the model reasons before it
  commits to the constrained decision fields (avoids the "constraint tax").
* **Fight omission** (the #1 small-model failure: prose instead of a decision).
  The system prompt carries in-distribution few-shot examples; on a parse miss we
  retry once with a terse reminder.
* **Deterministic.** ``temperature=0`` and a fixed seed so evals are repeatable.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from .schemas import PLAN_JSON_SCHEMA, Plan, parse_plan

DEFAULT_ENDPOINT = "http://localhost:11434"

# Few-shot examples live in the system prompt: in-distribution, mixed ID/EN, and
# every operation appears at least once so the model never has to guess the op
# vocabulary. Kept short — small models copy structure from what they just saw.
SYSTEM_PROMPT = """You are the planning brain of a local infrastructure agent for \
Cloudflare Tunnels. Convert the user's request into ONE typed action. Do not \
explain in prose; put any thinking in the "reasoning" field and then fill the \
decision fields exactly.

Operations:
- expose: make a local service reachable at a public hostname (needs hostname + port).
- unexpose: remove a public hostname / tear down its tunnel (needs hostname).
- status: report whether a tunnel/hostname is up (read-only).
- diagnose: investigate why something is failing (read-only).
- heal: restart/repair a broken tunnel or origin.

Rules:
- Pick exactly one op. If no port is given for expose, use 0 (the caller will ask).
- hostname is the public name (e.g. media.example.com), never the local address.
- service_scheme is http unless the user clearly means https to the local origin.
- If the request is ambiguous or dangerous, still emit your single best-guess op;
  a separate safety gate decides whether to act. Never invent extra hostnames.

Examples:
User: Expose my Jellyfin on media.example.com, it's on port 8096
{"reasoning":"expose a local service at a hostname","op":"expose","hostname":"media.example.com","port":8096,"service_scheme":"http"}
User: tolong buka grafana saya di stats.example.com port 3000 pakai https
{"reasoning":"expose local https origin","op":"expose","hostname":"stats.example.com","port":3000,"service_scheme":"https"}
User: is media.example.com up right now?
{"reasoning":"read-only status check","op":"status","hostname":"media.example.com","port":0,"service_scheme":"http"}
User: kenapa app.example.com error terus?
{"reasoning":"investigate a failing hostname","op":"diagnose","hostname":"app.example.com","port":0,"service_scheme":"http"}
User: the tunnel for git.example.com died, bring it back
{"reasoning":"repair a broken tunnel","op":"heal","hostname":"git.example.com","port":0,"service_scheme":"http"}
User: take down old.example.com, I don't need it anymore
{"reasoning":"remove a public hostname","op":"unexpose","hostname":"old.example.com","port":0,"service_scheme":"http"}"""

_RETRY_REMINDER = (
    "Respond with ONLY the JSON object for the single best action. "
    "No prose, no markdown fence."
)


@dataclass
class LLMResult:
    plan: Optional[Plan]
    raw: str
    model: str
    latency_s: float
    eval_count: int = 0          # tokens generated (from Ollama response)
    eval_duration_ns: int = 0
    retried: bool = False

    @property
    def tokens_per_s(self) -> float:
        if self.eval_duration_ns <= 0:
            return 0.0
        return self.eval_count / (self.eval_duration_ns / 1e9)


class OllamaClient:
    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        timeout: float = 120.0,
        num_ctx: int = 4096,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.num_ctx = num_ctx

    def plan(self, model: str, user_text: str, *, constrain: bool = True) -> LLMResult:
        """Ask ``model`` for a single typed Plan. Retries once on omission."""
        result = self._chat(model, user_text, constrain=constrain)
        if result.plan is None:
            # Omission recovery: remind and retry once.
            retry = self._chat(
                model, user_text, constrain=constrain, reminder=_RETRY_REMINDER
            )
            retry.retried = True
            if retry.plan is not None or True:
                return retry
        return result

    def _chat(
        self,
        model: str,
        user_text: str,
        *,
        constrain: bool,
        reminder: str = "",
    ) -> LLMResult:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if reminder:
            messages.append({"role": "system", "content": reminder})
        messages.append({"role": "user", "content": user_text})

        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0,
                "seed": 42,
                "num_ctx": self.num_ctx,
            },
        }
        if constrain:
            body["format"] = PLAN_JSON_SCHEMA

        t0 = time.time()
        payload = self._post("/api/chat", body)
        latency = time.time() - t0

        raw = (payload.get("message") or {}).get("content", "") if payload else ""
        return LLMResult(
            plan=parse_plan(raw),
            raw=raw,
            model=model,
            latency_s=latency,
            eval_count=payload.get("eval_count", 0) if payload else 0,
            eval_duration_ns=payload.get("eval_duration", 0) if payload else 0,
        )

    def list_models(self) -> list[str]:
        payload = self._get("/api/tags")
        return [m["name"] for m in (payload.get("models") or [])]

    # ---- transport --------------------------------------------------------- #
    def _post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(
            self.endpoint + path,
            data=json.dumps(body).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        return self._send(req)

    def _get(self, path: str) -> dict:
        req = urllib.request.Request(self.endpoint + path, method="GET")
        return self._send(req)

    def _send(self, req: urllib.request.Request) -> dict:
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise LLMError(f"ollama HTTP {exc.code}: {exc.read().decode(errors='replace')[:200]}")
        except urllib.error.URLError as exc:
            raise LLMError(f"ollama unreachable at {self.endpoint}: {exc.reason}")
        except (json.JSONDecodeError, ValueError) as exc:
            raise LLMError(f"ollama bad JSON: {exc}")


class LLMError(Exception):
    pass
