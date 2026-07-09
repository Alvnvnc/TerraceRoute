"""Compress-before-remote dengan guardrail anti-expansion.

Versi MVP: prompt-based compression pakai model lokal (gratis). Untuk kualitas
lebih tinggi, ganti dengan LLMLingua-2 (encoder kecil) tanpa mengubah antarmuka.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .ollama_client import OllamaClient
from .tokens import estimate_tokens

_COMPRESS_SYSTEM = (
    "You compress task prompts. Output the shortest instruction that preserves ALL "
    "requirements, constraints, and specifics. Remove greetings, repetition, and noise. "
    "Return ONLY the compressed task, nothing else."
)

# Task yang rawan output-expansion saat dikompresi -> lewati kompresi.
_RISKY = re.compile(r"```|def |class |SELECT |json|regex|schema", re.IGNORECASE)


@dataclass
class Compression:
    text: str
    ratio: float
    applied: bool
    reason: str


async def compress(prompt: str, local: OllamaClient) -> Compression:
    orig_tok = estimate_tokens(prompt)

    if _RISKY.search(prompt):
        return Compression(prompt, 1.0, False, "risky_task_skip")
    if orig_tok < 60:
        return Compression(prompt, 1.0, False, "too_short")

    gen = await local.generate(prompt, system=_COMPRESS_SYSTEM, temperature=0.0,
                               want_logprobs=False)
    compressed = gen.text.strip()
    new_tok = estimate_tokens(compressed)

    # Guardrail: hanya pakai kalau benar-benar lebih pendek & tidak kosong.
    if not compressed or new_tok >= orig_tok:
        return Compression(prompt, 1.0, False, "no_savings")

    return Compression(compressed, new_tok / orig_tok, True, "ok")
