"""Konfigurasi runtime — dibaca MURNI dari environment (aturan harness).

Jangan hardcode nilai rahasia / model IDs. Harness menginjeksi FIREWORKS_* dan
ALLOWED_MODELS saat evaluasi. Sisanya punya default aman untuk dev lokal.
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
    # --- I/O (path harness) ---
    input_path: str = field(default_factory=lambda: _env("INPUT_PATH", "/input/tasks.json"))
    output_path: str = field(default_factory=lambda: _env("OUTPUT_PATH", "/output/results.json"))

    # --- Fireworks (diinjeksi harness saat scoring) ---
    fireworks_api_key: str = field(default_factory=lambda: _env("FIREWORKS_API_KEY"))
    fireworks_base_url: str = field(
        default_factory=lambda: _env("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
    )
    # Comma-separated, dibaca runtime. Kosong = zero-API mode (lokal saja).
    allowed_models_raw: str = field(default_factory=lambda: _env("ALLOWED_MODELS", ""))

    # --- Model lokal ---
    # Backend: "ollama" (dev) atau "llamacpp" (prod, in-process GGUF).
    local_backend: str = field(default_factory=lambda: _env("LOCAL_BACKEND", "ollama"))
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    local_model: str = field(default_factory=lambda: _env("LOCAL_MODEL", "qwen2.5:3b-instruct"))
    llamacpp_model_path: str = field(default_factory=lambda: _env("LLAMACPP_MODEL_PATH", "/models/local.gguf"))
    llamacpp_threads: int = field(default_factory=lambda: _env_int("LLAMACPP_THREADS", 2))
    llamacpp_ctx: int = field(default_factory=lambda: _env_int("LLAMACPP_CTX", 4096))

    # --- Kebijakan routing / eskalasi ---
    # 0 = zero-API (lokal saja); 1 = escalate hanya verifikasi-gagal; 2 = + kategori jebakan;
    # 3 = agresif (escalate saat ragu). Knob utama yang di-tune lewat leaderboard.
    escalation_level: int = field(default_factory=lambda: _env_int("ESCALATION_LEVEL", 1))

    # --- Anggaran waktu (aturan: total 10 menit) ---
    total_budget_s: float = field(default_factory=lambda: _env_float("TOTAL_BUDGET_S", 540.0))
    watchdog_s: float = field(default_factory=lambda: _env_float("WATCHDOG_S", 510.0))

    # --- Batas token (hemat ranking; input+output dihitung) ---
    local_max_tokens: int = field(default_factory=lambda: _env_int("LOCAL_MAX_TOKENS", 512))
    remote_max_tokens: int = field(default_factory=lambda: _env_int("REMOTE_MAX_TOKENS", 512))
    remote_timeout_s: float = field(default_factory=lambda: _env_float("REMOTE_TIMEOUT_S", 60.0))

    @property
    def allowed_models(self) -> list[str]:
        return [m.strip() for m in self.allowed_models_raw.split(",") if m.strip()]

    @property
    def can_escalate(self) -> bool:
        return self.escalation_level > 0 and bool(self.fireworks_api_key) and bool(self.allowed_models)


config = Config()
