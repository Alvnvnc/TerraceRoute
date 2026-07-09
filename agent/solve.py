"""Terrace solve: per-task, keluar sedini mungkin dengan token remote minimum.

Alur: klasifikasi → short-circuit deterministik → solve lokal → verifikasi gratis →
escalate HANYA bila kebijakan mengizinkan & keyakinan lokal kurang.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from . import classify as C
from .config import config
from .fireworks import Fireworks
from .local_llm import LocalLLM
from . import verify as V

# System prompt ringkas per kategori — jawaban to-the-point (hemat token bila di-escalate,
# dan bantu judge menilai "expected intent" tanpa bertele-tele).
_SYS = {
    C.FACTUAL: "Answer the question directly and correctly in 1-3 sentences. No preamble.",
    C.SENTIMENT: "Classify the sentiment (positive, negative, or neutral) and give a one-line justification.",
    C.SUMMARISATION: "Summarise following the exact length/format constraint in the task. Output only the summary.",
    C.NER: "Extract the named entities and label each with its type (person, organization, location, date). Be concise.",
    C.LOGICAL: "Solve the logic puzzle. Reason briefly, then state the final answer on the last line as 'Answer: ...'.",
    C.MATH: "Solve step by step, then state the final answer on the last line as 'Answer: ...'.",
    C.CODE_GEN: "Write correct, well-structured code that satisfies the spec. Output only the code.",
    C.CODE_DEBUG: "Identify the bug and output the corrected code. Output only the corrected code.",
}
_DEFAULT_SYS = "Answer correctly and concisely. No preamble."

# Prompt REMOTE terpisah: output token remote DIHITUNG ke skor → paksa sesingkat mungkin
# (jawaban final saja, tanpa penalaran). Untuk judge 'expected intent', jawaban final
# yang benar sudah cukup; penalaran panjang hanya membakar token.
_SYS_REMOTE = {
    C.FACTUAL: "Answer in one short sentence. No preamble, no explanation.",
    C.SENTIMENT: "Reply with exactly one word: positive, negative, or neutral.",
    C.SUMMARISATION: "Output only the summary, obeying the task's length/format constraint. Nothing else.",
    C.NER: "List entities as 'type: value', one per line. Nothing else.",
    C.LOGICAL: "Give only the final answer, as briefly as possible. No reasoning.",
    C.MATH: "Give only the final numeric answer. No units unless required, no explanation.",
    C.CODE_GEN: "Output only the code. No explanation, no markdown fences.",
    C.CODE_DEBUG: "Output only the corrected code. No explanation, no markdown fences.",
}
# Batas token output remote per kategori (hemat ranking).
_REMOTE_MAXTOK = {
    C.SENTIMENT: 8, C.MATH: 32, C.FACTUAL: 96, C.LOGICAL: 64,
    C.NER: 128, C.SUMMARISATION: 160, C.CODE_GEN: 384, C.CODE_DEBUG: 384,
}

# Rung eskalasi per level utk status 'unverifiable' (level 4 = semua kategori).
# Diturunkan dari efisiensi marginal terukur, bukan intuisi — lihat eval/escalation-math.md.
_LEVEL_CATEGORIES = {
    2: {C.MATH, C.FACTUAL},
    3: {C.MATH, C.FACTUAL, C.SENTIMENT, C.LOGICAL},
}

_ANSWER_LINE = re.compile(r"answer\s*[:=]\s*(.+?)\s*$", re.I | re.M)


def _last_answer_line(raw: str) -> Optional[str]:
    """Ambil isi baris 'Answer: ...' terakhir bila ada."""
    matches = _ANSWER_LINE.findall(raw or "")
    return matches[-1].strip() if matches else None


# Coercion format GRATIS (nol token): ubah "salah format" jadi jawaban bersih sebelum
# dinilai judge. Preamble ("Sure, here's...") melanggar constraint eksplisit (mis.
# "exactly one sentence") dan menodai jawaban ringkas.
_PREAMBLE = re.compile(
    r"^(sure|okay|ok|certainly|of course|here(?:'s| is| are)\b[^:\n]*|the answer is)[,:.!—-]?\s*",
    re.I)


def _coerce(category: str, raw: str) -> str:
    """Bersihkan jawaban per kategori; selalu kembalikan teks terbaik yang ada."""
    text = (raw or "").strip()
    if not text:
        return text
    if category in (C.CODE_GEN, C.CODE_DEBUG):
        code = V.extract_code_block(text)
        return code or text
    stripped = text
    for _ in range(3):  # preamble bisa bertumpuk: "Sure, here's the answer: ..."
        nxt = _PREAMBLE.sub("", stripped, count=1).strip()
        if nxt == stripped:
            break
        stripped = nxt
    if category == C.LOGICAL:
        ans = _last_answer_line(stripped)
        if ans:
            return ans
    return stripped or text


_RE_ANSWER = re.compile(r"ANSWER\s*:\s*(.+)", re.I)
_RE_PYTHON = re.compile(r"PYTHON\s*:\s*(.*)", re.I | re.S)


def _parse_answer_python(raw: str) -> tuple[Optional[str], Optional[str]]:
    """Parse format terdelimitasi 'ANSWER: ...' + 'PYTHON: <kode>' (robust thd newline)."""
    if not raw:
        return None, None
    ans = None
    m = _RE_ANSWER.search(raw)
    if m:
        ans = m.group(1).splitlines()[0].strip()
    code = None
    m2 = _RE_PYTHON.search(raw)
    if m2:
        from .verify import extract_code_block
        body = m2.group(1)
        code = extract_code_block(body) or body.strip()
    return ans, code


@dataclass
class TaskResult:
    task_id: str
    answer: str
    category: str = ""
    route: str = "local"          # local | local_verified | remote | deterministic | fallback
    remote_tokens: int = 0
    remote_model: str = ""
    needs_escalation: bool = False  # kebijakan bilang escalate; eksekusinya deferred (run.py)
    meta: dict = field(default_factory=dict)


class Solver:
    def __init__(self, local: Optional[LocalLLM] = None, remote: Optional[Fireworks] = None):
        self.local = local or LocalLLM()
        self.remote = remote or Fireworks()

    # ---------- utilitas escalate ----------
    def escalate(self, prompt: str, tr: TaskResult) -> bool:
        """Eksekusi eskalasi remote utk task bertanda needs_escalation.

        Dipanggil run.py — boleh dari worker thread (stateless per-call; tiap tr
        berbeda). True bila berhasil & tr diperbarui; gagal = jawaban lokal bertahan.
        """
        return self._escalate(prompt, tr.category, tr)

    def _escalate(self, prompt: str, category: str, tr: TaskResult) -> bool:
        """Coba jawab via remote; True bila berhasil & tr diperbarui.

        Guard berbasis kredensial (bukan level): kebijakan level sudah ditegakkan
        _decide; caller lain (tail-to-remote run.py) sah memakai remote di level 0.
        """
        if not config.can_remote:
            return False
        prefer = "code" if category in (C.CODE_GEN, C.CODE_DEBUG) else None
        model = self.remote.pick_model(prefer=prefer)
        res = self.remote.generate(prompt, system=_SYS_REMOTE.get(category, _DEFAULT_SYS),
                                   model=model, temperature=0.0,
                                   max_tokens=_REMOTE_MAXTOK.get(category, config.remote_max_tokens))
        if res is None or not res.text:
            return False
        answer = _coerce(category, res.text)
        if not answer:
            return False
        tr.answer = answer
        tr.route = "remote"
        tr.remote_tokens = res.total_tokens
        tr.remote_model = res.model
        return True

    def _decide(self, category: str, status: str) -> bool:
        """Kebijakan eskalasi MURNI (tanpa eksekusi) berbasis ESCALATION_LEVEL (0..4).

        Eksekusinya deferred: solve() hanya menandai needs_escalation; run.py yang
        menjalankan solver.escalate() — paralel di thread pool, overlap dgn kerja
        lokal task berikutnya (latensi remote tak lagi memblokir antrean lokal).

        status:
          verified     — konfirmasi deterministik INDEPENDEN lulus → jangan escalate
          verify_failed— cek independen GAGAL (mis. kode error) → sinyal presisi tinggi
          empty        — model lokal tak menghasilkan apa-apa
          unverifiable — tak ada cek independen (word-problem math, logical, factual, dst)

        Urutan kategori per level BERBASIS DATA (eval v4 + Monte Carlo,
        eval/escalation-math.md): efisiensi marginal ΔAcc/token = math ≈ factual
        >> sentiment ≈ logical >> sisanya (lokal ~100%, escalate = rugi murni).
        Catatan: factual menggantikan logical di rung pertama karena halusinasi
        factual TAK terlihat verifikasi lokal, sedangkan error logical sebagian
        sudah tertangkap jalur trap-routing classify.

        Catatan kalibrasi: kesepakatan model-dengan-dirinya (NL vs python buatan model)
        BUKAN 'verified' — model kecil confidently-consistent wrong. Hanya cek yang benar2
        independen (eksekusi kode, hitung huruf, aritmetika yang KITA parse) = verified.
        """
        lvl = config.escalation_level
        if status == "verified":
            return False
        if status in ("empty", "verify_failed"):
            escalate = lvl >= 1
        else:  # unverifiable
            if lvl >= 4:
                escalate = True
            elif lvl >= 2:
                allowed = _LEVEL_CATEGORIES[min(lvl, 3)]
                escalate = category in allowed
            else:
                escalate = False
        return escalate

    def _local(self, prompt: str, category: str, max_tokens: Optional[int] = None) -> str:
        return self.local.chat(prompt, system=_SYS.get(category, _DEFAULT_SYS),
                               max_tokens=max_tokens or config.local_max_tokens)

    # ---------- jalur per kategori ----------
    def _solve_math(self, prompt: str, tr: TaskResult) -> None:
        # SATU panggilan lokal. Format terdelimitasi (BUKAN JSON — kode multiline bikin
        # JSON invalid): ANSWER: <baris> lalu PYTHON: <kode>. python bantu hitung ulang
        # aritmetika (bukan verifikasi independen: model bisa salah-konsisten).
        raw = self._local(
            f"{prompt}\n\nSolve it. Then on a new line write 'Answer:' followed by the final "
            f"concise answer. Then on a new line write 'Python:' followed by python code that "
            f"prints only the numeric result.",
            C.MATH, max_tokens=config.local_max_tokens,
        )
        nl_answer, code = _parse_answer_python(raw)
        nl_answer = nl_answer or _last_answer_line(raw) or raw.strip()
        computed = None
        if code:
            ex = V.run_python(str(code))
            if ex.ok:
                nums = V.numbers_in(ex.stdout)
                if nums:
                    computed = nums[-1]

        # Jawaban lokal terbaik: angka hasil eksekusi python bila ada, else NL.
        tr.answer = f"{computed:g}" if computed is not None else str(nl_answer).strip()
        status = "empty" if not tr.answer else "unverifiable"  # math word-problem = trap
        tr.needs_escalation = self._decide(C.MATH, status)
        tr.route = "local"

    def _solve_code(self, prompt: str, category: str, tr: TaskResult) -> None:
        raw = self._local(prompt, category, max_tokens=config.local_max_tokens)
        code = V.extract_code_block(raw) or raw
        tr.answer = _coerce(category, raw)
        # Cek INDEPENDEN: kode harus benar2 di-load tanpa syntax error (eksekusi nyata).
        if not raw.strip():
            status = "empty"
        else:
            loadable = V.run_python(f"import ast\nast.parse({code!r})")
            status = "unverifiable" if loadable.ok else "verify_failed"
        tr.needs_escalation = self._decide(category, status)
        tr.route = "local"

    def _solve_generic(self, prompt: str, category: str, tr: TaskResult) -> None:
        tr.answer = _coerce(category, self._local(prompt, category))
        status = "empty" if not tr.answer else "unverifiable"
        tr.needs_escalation = self._decide(category, status)
        tr.route = "local"

    # ---------- entri ----------
    def solve(self, task_id: str, prompt: str) -> TaskResult:
        category = C.classify(prompt)
        tr = TaskResult(task_id=task_id, answer="", category=category)

        # Short-circuit deterministik: counting huruf (blind-spot universal LLM).
        cnt = V.try_count_letter(prompt)
        if cnt is not None:
            tr.answer = str(cnt)
            tr.route = "deterministic"
            return tr

        try:
            if category == C.MATH:
                self._solve_math(prompt, tr)
            elif category in (C.CODE_GEN, C.CODE_DEBUG):
                self._solve_code(prompt, category, tr)
            else:
                self._solve_generic(prompt, category, tr)
        except Exception as exc:  # noqa: BLE001
            tr.meta["error"] = str(exc)
            if not tr.answer:
                tr.answer = self._local(prompt, category) or ""
                tr.route = "fallback"
        return tr
