"""Model request/response (Pydantic v2)."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Mode(str, Enum):
    auto = "auto"
    local = "local"      # paksa lokal (debug/eval)
    remote = "remote"    # paksa remote (debug/eval)


class Route(str, Enum):
    local_only = "LOCAL_ONLY"
    local_self_check = "LOCAL_SELF_CHECK"       # draft diperbaiki lokal, remote dihindari
    local_draft_remote_patch = "LOCAL_DRAFT_REMOTE_PATCH"
    remote_compressed = "REMOTE_COMPRESSED"
    cache_hit = "CACHE_HIT"


class AnswerRequest(BaseModel):
    input: str = Field(..., min_length=1)
    mode: Mode = Mode.auto


class RouteDecision(BaseModel):
    route: Route
    uncertainty: float                 # sinyal mentah u in [0,1]
    p_wrong: float                     # setelah kalibrasi
    high_risk: bool
    signal: str                        # "logprob" | "self_consistency"
    reason: str


class AnswerResponse(BaseModel):
    route: Route
    p_wrong: float
    uncertainty: float
    signal: str
    local_model: str
    remote_model: Optional[str] = None
    remote_used: bool
    remote_tokens_in: int = 0
    remote_tokens_out: int = 0
    estimated_tokens_saved: int = 0
    compression_ratio: Optional[float] = None
    latency_ms: int
    answer: str
