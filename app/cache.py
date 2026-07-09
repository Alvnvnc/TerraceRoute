"""Cache presisi-tinggi berbasis embedding (opsional).

Sengaja pakai threshold tinggi (default 0.95) agar tidak wrong-hit — di env
dengan accuracy floor, cache hit yang salah lebih berbahaya daripada cache miss.
"""
from __future__ import annotations

import math
from typing import Optional

from .config import settings
from .ollama_client import OllamaClient


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class SemanticCache:
    def __init__(self, local: OllamaClient, threshold: Optional[float] = None):
        self.local = local
        self.threshold = threshold if threshold is not None else settings.cache_sim_threshold
        self._store: list[tuple[list[float], str]] = []

    async def get(self, text: str) -> Optional[str]:
        if not settings.cache_enabled:
            return None
        try:
            emb = await self.local.embed(text)
        except Exception:
            return None
        best_sim, best_ans = 0.0, None
        for e, ans in self._store:
            sim = _cosine(emb, e)
            if sim > best_sim:
                best_sim, best_ans = sim, ans
        return best_ans if best_sim >= self.threshold else None

    async def put(self, text: str, answer: str) -> None:
        if not settings.cache_enabled:
            return
        try:
            emb = await self.local.embed(text)
        except Exception:
            return
        self._store.append((emb, answer))
