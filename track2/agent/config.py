"""Runtime configuration with restricted self-hosted defaults.

Track 2 injects no credentials. A Fireworks Gemma vision deployment can take over
generator calls when the primary backend is unavailable; checker calls never use
that generator fallback.
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
    work_dir: str = field(default_factory=lambda: _env("WORK_DIR", "/tmp/clips"))

    # --- Primary backend: restricted self-hosted Ollama proxy (OpenAI-compatible) ---
    # The token rides as a query parameter on the restricted public VLM proxy.
    vlm_base_url: str = field(default_factory=lambda: _env(
        "VLM_BASE_URL",
        "https://amd.alvnvnc.site/v1"))
    vlm_token: str = field(default_factory=lambda: _env("VLM_TOKEN", "terraceroute-track2-v1"))
    vlm_model: str = field(default_factory=lambda: _env("VLM_MODEL", "gemma3:12b"))
    checker_model: str = field(default_factory=lambda: _env("CHECKER_MODEL", "gemma3:12b"))

    # --- Fireworks Gemma vision fallback ---
    # FIREWORKS_GEMMA_MODEL must be an image-capable on-demand deployment ID, not a
    # base-model ID. The API key may be shared with Track 1 but is injected only at runtime.
    fireworks_api_key: str = field(default_factory=lambda: _env("FIREWORKS_API_KEY", ""))
    fireworks_base_url: str = field(default_factory=lambda: _env(
        "FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1"))
    fireworks_gemma_model: str = field(default_factory=lambda: _env("FIREWORKS_GEMMA_MODEL", ""))

    # --- Optional fallback backend (any OpenAI-compatible vision API) ---
    # FB_MODEL is used only for generator calls. Set FB_CHECKER_MODEL separately
    # when the checker needs a fallback from a different model family.
    fb_base_url: str = field(default_factory=lambda: _env("FB_BASE_URL", ""))
    fb_api_key: str = field(default_factory=lambda: _env("FB_API_KEY", ""))
    fb_model: str = field(default_factory=lambda: _env("FB_MODEL", ""))
    fb_checker_model: str = field(default_factory=lambda: _env("FB_CHECKER_MODEL", ""))

    # --- Ingest / adaptive temporal sampler ---
    ffmpeg_bin: str = field(default_factory=lambda: _env("FFMPEG_BIN", "ffmpeg"))
    frames: int = field(default_factory=lambda: _env_int("FRAMES", 6))  # uniform fallback
    frame_width: int = field(default_factory=lambda: _env_int("FRAME_WIDTH", 512))
    download_timeout_s: float = field(default_factory=lambda: _env_float("DOWNLOAD_TIMEOUT_S", 90.0))
    # Candidate pool size for the analysis pass (~2 thumbs/s, capped for long clips).
    sampler_candidates: int = field(default_factory=lambda: _env_int("SAMPLER_CANDIDATES", 96))
    # Simple clip → min_frames; many transitions/motion → up to max_frames.
    min_frames: int = field(default_factory=lambda: _env_int("MIN_FRAMES", 6))
    max_frames: int = field(default_factory=lambda: _env_int("MAX_FRAMES", 20))
    # Provider-safety caps (Fireworks: ≤30 images, <10MB request payload).
    max_images: int = field(default_factory=lambda: _env_int("MAX_IMAGES", 30))
    max_payload_mb: float = field(default_factory=lambda: _env_float("MAX_PAYLOAD_MB", 10.0))
    # Near-duplicate threshold: mean abs diff (0-255) between gray thumbnails.
    dedupe_mad: float = field(default_factory=lambda: _env_float("DEDUPE_MAD", 1.5))

    # --- Time budget (rule: 10 minutes total) ---
    watchdog_s: float = field(default_factory=lambda: _env_float("WATCHDOG_S", 510.0))
    request_timeout_s: float = field(default_factory=lambda: _env_float("REQUEST_TIMEOUT_S", 120.0))
    clip_workers: int = field(default_factory=lambda: _env_int("CLIP_WORKERS", 3))

    # --- Quality knobs ---
    # Verification (checker call + regeneration) is skipped when the remaining
    # per-clip budget is below this — the degradation ladder in plan.md §4.
    verify_min_budget_s: float = field(default_factory=lambda: _env_float("VERIFY_MIN_BUDGET_S", 45.0))
    # The temporal evidence graph is bigger than a prose description (timestamped JSON).
    evidence_max_tokens: int = field(default_factory=lambda: _env_int("EVIDENCE_MAX_TOKENS", 700))
    stylize_max_tokens: int = field(default_factory=lambda: _env_int("STYLIZE_MAX_TOKENS", 450))
    # Frames shown to the Qwen checker (spread over the clip) — enough to ground the
    # verdict without paying full perception cost twice.
    checker_frames: int = field(default_factory=lambda: _env_int("CHECKER_FRAMES", 6))

    @property
    def has_fallback(self) -> bool:
        return bool(self.fb_base_url) and bool(self.fb_model)

    @property
    def has_checker_fallback(self) -> bool:
        return bool(self.fb_base_url) and bool(self.fb_checker_model)

    @property
    def has_fireworks_gemma(self) -> bool:
        return bool(self.fireworks_api_key) and bool(self.fireworks_gemma_model)


config = Config()
