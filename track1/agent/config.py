"""Runtime configuration — read PURELY from the environment (harness rule).

Do not hardcode secrets or model IDs. The harness injects FIREWORKS_* and
ALLOWED_MODELS at scoring time. Everything else has a safe default for local dev.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


@dataclass
class Config:
    # --- I/O (harness paths) ---
    input_path: str = field(default_factory=lambda: _env("INPUT_PATH", "/input/tasks.json"))
    output_path: str = field(default_factory=lambda: _env("OUTPUT_PATH", "/output/results.json"))

    # --- Fireworks (injected by the harness at scoring time) ---
    fireworks_api_key: str = field(default_factory=lambda: _env("FIREWORKS_API_KEY"))
    fireworks_base_url: str = field(
        default_factory=lambda: _env("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    )
    # Comma-separated, read at runtime. Empty = zero-API mode (local only).
    allowed_models_raw: str = field(default_factory=lambda: _env("ALLOWED_MODELS", ""))

    # --- Local model ---
    # Backend: "ollama" (dev) or "llamacpp" (prod, in-process GGUF).
    local_backend: str = field(default_factory=lambda: _env("LOCAL_BACKEND", "ollama"))
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    local_model: str = field(default_factory=lambda: _env("LOCAL_MODEL", "qwen2.5:3b-instruct"))
    llamacpp_model_path: str = field(default_factory=lambda: _env("LLAMACPP_MODEL_PATH", "/models/local.gguf"))
    llamacpp_threads: int = field(default_factory=lambda: _env_int("LLAMACPP_THREADS", 2))
    llamacpp_ctx: int = field(default_factory=lambda: _env_int("LLAMACPP_CTX", 4096))

    # --- Routing / escalation policy ---
    # 0 = zero-API (local only); 1 = escalate only on verify-failed/empty;
    # 2 = + unverifiable math & factual; 3 = + sentiment & logical;
    # 4 = everything unverifiable. Rung order is data-driven (docs/escalation-math.md).
    # This is the main knob tuned via the leaderboard submission ladder.
    escalation_level: int = field(default_factory=lambda: _env_int("ESCALATION_LEVEL", 0))

    # --- Time budget (rule: 10 minutes total) ---
    total_budget_s: float = field(default_factory=lambda: _env_float("TOTAL_BUDGET_S", 540.0))
    watchdog_s: float = field(default_factory=lambda: _env_float("WATCHDOG_S", 510.0))

    # --- Token caps (save ranking; input + output both count) ---
    local_max_tokens: int = field(default_factory=lambda: _env_int("LOCAL_MAX_TOKENS", 512))
    remote_max_tokens: int = field(default_factory=lambda: _env_int("REMOTE_MAX_TOKENS", 512))
    # Sent to Fireworks as reasoning_effort ("" = never send). Measured 2026-07-11:
    # all current serverless models think by default; "none" collapses answers to
    # 2-7 completion tokens on glm/deepseek/kimi. Models that reject the param are
    # auto-detected (400) and called without it.
    reasoning_effort: str = field(default_factory=lambda: _env("REASONING_EFFORT", "none"))
    remote_timeout_s: float = field(default_factory=lambda: _env_float("REMOTE_TIMEOUT_S", 60.0))
    # Worker threads for parallel remote escalation (I/O-bound; no CPU contention with llama.cpp).
    remote_concurrency: int = field(default_factory=lambda: _env_int("REMOTE_CONCURRENCY", 4))

    @property
    def allowed_models(self) -> list[str]:
        return [m.strip() for m in self.allowed_models_raw.split(",") if m.strip()]

    @property
    def can_remote(self) -> bool:
        """Remote credentials are available (the harness ALWAYS injects them at scoring)."""
        return bool(self.fireworks_api_key) and bool(self.allowed_models)

    @property
    def can_escalate(self) -> bool:
        return self.escalation_level > 0 and self.can_remote


config = Config()
