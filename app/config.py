"""Konfigurasi terpusat, dibaca dari environment / .env."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Local (Ollama)
    ollama_host: str = "http://localhost:11434"
    local_model: str = "qwen2.5:14b-instruct"
    embed_model: str = "bge-m3"
    ollama_logprobs: bool = True

    # Remote (Fireworks)
    fireworks_api_key: str = "changeme"
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"
    remote_model: str = "accounts/fireworks/models/qwen2p5-72b-instruct"

    # Router thresholds
    tau1: float = 0.20      # dari sweep: p_wrong(setuju)≈0.13 < 0.20 < p_wrong(tak setuju)≈0.33
    tau2: float = 0.65
    accuracy_floor: float = 0.90     # > akurasi lokal (0.80) supaya routing WAJIB escalate error
    self_consistency_k: int = 3      # sinyal sekunder (terbukti lemah); kecil utk hemat waktu
    sc_temperature: float = 0.7      # suhu sampling utk memancing ketidaksepakatan

    # Local self-check: sebelum escalate, coba perbaiki draft secara lokal (gratis).
    self_check: bool = True
    logprob_min_weight: float = 0.3   # bobot worst-token pada sinyal uncertainty

    # Evaluasi/kalibrasi: judge independen (beda keluarga dari local_model agar tak menilai diri sendiri)
    judge_model: str = "gemma3:12b"
    eval_tasks: str = "eval/sample_tasks_v3.jsonl"   # 143 task deterministik (v2=30 lama)

    # Cross-model check: verifier beda-keluarga menangkap confident-consistent errors yg
    # buta terhadap perplexity & self-consistency (temuan v2). Comma-separated.
    cross_model: bool = True
    verifier_models: str = "gemma3:12b"   # verifier beda-keluarga KUAT; 3b menambah noise (buang)

    # Cache
    cache_enabled: bool = True
    cache_sim_threshold: float = 0.95

    # Kalibrasi
    calibrator_path: str = "artifacts/calibrator.json"


settings = Settings()
