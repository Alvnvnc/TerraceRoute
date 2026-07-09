"""TerraceGate — a local-first infrastructure agent for self-hosters.

Natural language in, verified Cloudflare Tunnel + DNS out; a deterministic
self-healing watchdog; and an uncertainty gate that decides when *not* to act.

The LLM never touches the Cloudflare API directly. It only emits a typed plan
that maps onto the deterministic typed tools in ``agent.tools``. Diagnosis of
failures is pure rules (``agent.heal``); the safety gate is a cross-model
disagreement + blast-radius decision (``agent.brain``).
"""

__version__ = "0.1.0"
