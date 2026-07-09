"""The typed plan the local models emit, and how to normalize it.

The plan is what the LLM produces *instead of* touching Cloudflare. It is a
flat JSON object (objects / enums / required only) so llama.cpp's JSON-Schema
-> GBNF converter can constrain it reliably (no ``oneOf``/``$ref``/``if-then``).

Anti "constraint tax": the schema puts a free-text ``reasoning`` field first so
the model can think before committing to the constrained decision fields.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional

# Operations the agent understands. NL maps onto exactly one of these.
OPERATIONS = ("expose", "unexpose", "status", "diagnose", "heal")

# JSON Schema passed verbatim as Ollama's ``format`` field for constrained
# decoding. Kept flat on purpose (see module docstring).
PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief free-text reasoning before deciding. Think here.",
        },
        "op": {
            "type": "string",
            "enum": list(OPERATIONS),
            "description": "The single operation to perform.",
        },
        "hostname": {
            "type": "string",
            "description": "Public hostname, e.g. media.example.com. "
                           "Empty for status/diagnose/heal with no specific host.",
        },
        "port": {
            "type": "integer",
            "description": "Local origin port for 'expose' (e.g. 8096). 0 if N/A.",
        },
        "service_scheme": {
            "type": "string",
            "enum": ["http", "https"],
            "description": "Scheme of the local origin service. Default http.",
        },
    },
    "required": ["reasoning", "op", "hostname", "port", "service_scheme"],
}


@dataclass
class Plan:
    op: str
    hostname: str = ""
    port: int = 0
    service_scheme: str = "http"
    reasoning: str = ""

    @property
    def valid(self) -> bool:
        return self.op in OPERATIONS

    def origin_service(self) -> str:
        """The ``service`` value for a cloudflared ingress rule."""
        return f"{self.service_scheme}://localhost:{self.port}"


_HOST_RE = re.compile(r"[a-z0-9]([-a-z0-9.]*[a-z0-9])?", re.IGNORECASE)


def parse_plan(raw: str | dict) -> Optional[Plan]:
    """Parse a model's JSON output into a :class:`Plan`.

    Returns ``None`` when the output is not usable (the caller treats a missing
    plan as maximum disagreement / omission, never as silent success).
    """
    data: Any
    if isinstance(raw, dict):
        data = raw
    else:
        data = _extract_json_object(raw)
        if data is None:
            return None
    if not isinstance(data, dict):
        return None

    op = str(data.get("op", "")).strip().lower()
    if op not in OPERATIONS:
        return None

    hostname = _clean_hostname(data.get("hostname", ""))
    port = _coerce_port(data.get("port", 0))
    scheme = str(data.get("service_scheme", "http")).strip().lower()
    if scheme not in ("http", "https"):
        scheme = "http"

    return Plan(
        op=op,
        hostname=hostname,
        port=port,
        service_scheme=scheme,
        reasoning=str(data.get("reasoning", "")).strip(),
    )


def normalize_for_compare(plan: Optional[Plan]) -> dict[str, Any]:
    """Reduce a plan to the decision-relevant fields used for disagreement.

    ``reasoning`` is deliberately excluded: two models may reason in different
    words yet agree on the action. Only the action matters for the gate.
    """
    if plan is None:
        return {}
    fields: dict[str, Any] = {"op": plan.op}
    # Only compare arguments that matter for the chosen op.
    if plan.op in ("expose", "unexpose"):
        fields["hostname"] = plan.hostname.lower()
    if plan.op == "expose":
        fields["port"] = plan.port
        fields["service_scheme"] = plan.service_scheme
    return fields


def _extract_json_object(text: str) -> Optional[dict]:
    text = text.strip()
    # Strip a ```json ... ``` fence if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to the first balanced {...} span.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None


def _clean_hostname(value: Any) -> str:
    s = str(value).strip().lower()
    if not s:
        return ""
    s = s.replace("https://", "").replace("http://", "").split("/")[0]
    m = _HOST_RE.match(s)
    return m.group(0) if m else ""


def _coerce_port(value: Any) -> int:
    try:
        p = int(value)
    except (TypeError, ValueError):
        m = re.search(r"\d{2,5}", str(value))
        p = int(m.group(0)) if m else 0
    return p if 0 <= p <= 65535 else 0
