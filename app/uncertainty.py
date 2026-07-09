"""Sinyal uncertainty lokal yang murah (bukan verbalized self-confidence).

- Perplexity dari logprob (Opsi A) — MURAH tapi TEMUAN v2: anti-korelasi dgn error
  (model confidently-wrong pada jebakan penalaran). corr(u_ppl, wrong) = -0.146.
- Self-consistency (Opsi B) — sample k jawaban, ukur ketidaksepakatan. Menangkap
  confident errors yang dilewatkan perplexity (ref: semantic entropy). Jadi sinyal UTAMA.

Self-consistency hybrid:
  * jawaban punya angka/pendek  -> clustering exact pada "answer key" (kuat utk penalaran)
  * jawaban terbuka/panjang     -> semantic disagreement via embedding cosine
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional


def uncertainty_from_logprob(
    avg_logprob: Optional[float],
    min_logprob: Optional[float] = None,
    min_weight: float = 0.3,
) -> Optional[float]:
    """Worst-token-aware perplexity uncertainty, di-clamp ke [0,1].

    u = 1 - exp( (1-w)*avg + w*min ). Dipakai sebagai FITUR SEKUNDER saja karena
    v2 menunjukkan perplexity anti-korelasi dgn kebenaran. Kembalikan None kalau
    logprob tak tersedia.
    """
    if avg_logprob is None:
        return None
    if min_logprob is None:
        blended = avg_logprob
    else:
        blended = (1.0 - min_weight) * avg_logprob + min_weight * min_logprob
    try:
        u = 1.0 - math.exp(blended)
    except OverflowError:
        return 1.0
    return max(0.0, min(1.0, u))


def _normalize(ans: str) -> str:
    """Normalisasi kasar untuk membandingkan jawaban pendek."""
    ans = ans.strip().lower()
    ans = re.sub(r"\s+", " ", ans)
    ans = re.sub(r"[^\w\s]", "", ans)
    return ans


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def answer_key(text: str) -> Optional[str]:
    """Ekstrak 'kunci jawaban' yang bisa di-cluster secara exact.

    Heuristik: jawaban akhir biasanya angka TERAKHIR di output; kalau tak ada angka
    tapi output pendek (<=4 kata) pakai teks ternormalisasi. Kembalikan None kalau
    tak salient (task terbuka) -> caller pakai embedding.
    """
    nums = _NUM.findall(text.replace(",", ""))
    if nums:
        v = nums[-1].rstrip(".")
        try:
            f = float(v)
            return str(int(f)) if f == int(f) else str(f)   # 8.0 == 8
        except ValueError:
            return v
    norm = _normalize(text)
    words = norm.split()
    if 0 < len(words) <= 2:            # jawaban satu-kata (Paris/Friday/XLIX); frasa panjang -> embedding
        return norm
    return None


_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "it", "its",
    "for", "and", "or", "as", "at", "be", "that", "this", "with", "will", "would", "so",
    "answer", "final", "result", "value", "number", "day", "would", "left", "there",
}


def _content_words(s: str) -> set[str]:
    return set(_normalize(s).split()) - _STOP


def _overlap(a: set[str], b: set[str]) -> float:
    """Overlap coefficient: |A∩B| / min(|A|,|B|). Tahan verbositas (subset -> 1.0)."""
    if not a or not b:
        return 1.0 if a == b else 0.0
    return len(a & b) / min(len(a), len(b))


def _mean_pairwise_overlap(sets: list[set[str]]) -> float:
    n = len(sets)
    if n < 2:
        return 1.0
    total = pairs = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            total += _overlap(sets[i], sets[j])
            pairs += 1
    return total / pairs if pairs else 1.0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _mean_pairwise_cosine(vectors: list[list[float]]) -> float:
    n = len(vectors)
    if n < 2:
        return 1.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _cosine(vectors[i], vectors[j])
            pairs += 1
    return total / pairs if pairs else 1.0


def self_consistency_uncertainty(
    samples: list[str],
    embeddings: Optional[list[list[float]]] = None,
) -> tuple[float, str]:
    """u = ketidaksepakatan antar sample. Kembalikan (u, metode).

    - answer_cluster : semua sample punya answer_key -> 1 - mayoritas/k (kuat utk penalaran)
    - semantic       : task terbuka -> 1 - rata2 cosine pairwise embedding
    - text_cluster   : fallback tanpa embedding
    """
    if not samples:
        return 1.0, "empty"
    keys = [answer_key(s) for s in samples]
    if all(k is not None for k in keys):
        counts = Counter(keys)
        majority = counts.most_common(1)[0][1]
        return 1.0 - majority / len(samples), "answer_cluster"
    if embeddings and len(embeddings) == len(samples):
        sim = _mean_pairwise_cosine(embeddings)
        return max(0.0, min(1.0, 1.0 - sim)), "semantic"
    # Non-numerik tanpa embedding: overlap kata-konten (tahan verbositas), bukan exact-match.
    sim = _mean_pairwise_overlap([_content_words(s) for s in samples])
    return max(0.0, min(1.0, 1.0 - sim)), "text_overlap"


def uncertainty_from_samples(samples: list[str]) -> float:
    """Kompat lama: disagreement clustering (tanpa embedding)."""
    return self_consistency_uncertainty(samples)[0]
