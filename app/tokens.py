"""Estimasi jumlah token yang cepat & tak butuh dependency berat.

Prioritas: pakai `usage` dari API kalau tersedia. Fungsi ini hanya untuk
estimasi lokal (mis. keputusan patch-vs-full, guardrail kompresi).
~4 karakter per token adalah heuristik kasar yang cukup untuk keputusan relatif.
"""
from __future__ import annotations


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Heuristik: campuran char/word. Cukup untuk perbandingan relatif.
    return max(1, round(len(text) / 4))
