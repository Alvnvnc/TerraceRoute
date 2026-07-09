"""Grader hybrid untuk kalibrasi: exact-match deterministik + LLM-judge independen.

Motivasi (temuan kalibrasi v1): 14B menilai jawabannya sendiri terlalu murah hati
sehingga hampir semua task dilabeli "benar" -> kalibrator degenerate ("jangan escalate").
Solusi:
  - `exact`  : task berjawaban pasti (angka/kata) dinilai boundary-aware, tanpa LLM.
  - `judge`  : task terbuka dinilai model INDEPENDEN (settings.judge_model, mis. gemma3:12b),
               bukan local_model yang menilai dirinya sendiri.
"""
from __future__ import annotations

import re
from typing import Optional

from app.config import settings
from app.ollama_client import OllamaClient

# Client judge independen (lazy: dibuat sekali).
_judge: Optional[OllamaClient] = None


def judge_client() -> OllamaClient:
    global _judge
    if _judge is None:
        _judge = OllamaClient(model=settings.judge_model)
    return _judge


_NUMERIC = re.compile(r"^[$€£]?-?\d[\d,]*\.?\d*%?$")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _match_one(answer: str, draft: str) -> bool:
    """Cari `answer` di dalam `draft` secara boundary-aware (hindari 54 cocok dgn 154)."""
    a = answer.strip()
    d = draft.lower()
    if _NUMERIC.match(a):
        # Ambil inti angka (buang $,%,koma) lalu cocokkan dgn batas non-digit.
        core = re.sub(r"[,$€£%\s]", "", a).lstrip("-")
        if not core:
            return False
        # Izinkan trailing nol desimal: 8 cocok dgn 8, 8.0, 8.00 ; tapi bukan 80 atau 18.
        pat = re.escape(core)
        if "." not in core:
            pat += r"(?:\.0+)?"
        # Boundary: tolak digit di kiri/kanan & titik-desimal (titik+digit) di kanan,
        # tapi IZINKAN titik akhir kalimat. Mis. "0.05." cocok, "0.056" tidak.
        return re.search(rf"(?<![\d.]){pat}(?!\d)(?!\.\d)", d) is not None
    # Non-numerik: word-boundary containment, case-insensitive.
    return re.search(rf"(?<![a-z0-9]){re.escape(a.lower())}(?![a-z0-9])", d) is not None


def grade_exact(draft: str, answers: list[str]) -> bool:
    """Benar bila salah satu jawaban yang diterima muncul di draft (boundary-aware)."""
    return any(_match_one(a, draft) for a in answers if a)


async def judge_llm(task: str, draft: str, reference: str) -> Optional[bool]:
    """Judge independen (gemma). Kembalikan None bila output ambigu (biar caller fallback)."""
    if not reference:
        return True
    prompt = (
        f"You are a strict grader. Judge ONLY whether the candidate answer is correct "
        f"and equivalent in meaning to the reference for this task.\n\n"
        f"Task:\n{task}\n\nReference answer:\n{reference}\n\n"
        f"Candidate answer:\n{draft}\n\nReply with ONLY 'YES' or 'NO'."
    )
    g = await judge_client().generate(prompt, temperature=0.0, want_logprobs=False, num_predict=8)
    ans = g.text.strip().upper()
    if ans.startswith("YES"):
        return True
    if ans.startswith("NO"):
        return False
    return None


async def grade(task: dict, draft: str) -> bool:
    """Dispatch grader hybrid berdasar field `grader` di task."""
    kind = task.get("grader", "judge")
    if kind == "exact":
        return grade_exact(draft, task.get("answer", []))
    # judge (independen) dgn fallback exact bila ambigu & ada `answer`.
    verdict = await judge_llm(task["input"], draft, task.get("reference", ""))
    if verdict is None:
        return grade_exact(draft, task.get("answer", []))
    return verdict
