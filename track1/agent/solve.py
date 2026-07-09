"""Terrace solve: per task, exit as early as possible at minimum remote token cost.

Flow: classify → deterministic short-circuit → local solve → free verification →
escalate ONLY when the policy allows it and local confidence is lacking.
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

# Terse per-category system prompt — to-the-point answers (saves tokens when escalated,
# and helps the judge assess "expected intent" without rambling).
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

# Separate REMOTE prompt: remote output tokens COUNT toward the score → force it as short
# as possible (final answer only, no reasoning). For the judge's 'expected intent', a
# correct final answer is enough; long reasoning only burns tokens.
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
# Per-category remote output token caps (save ranking).
_REMOTE_MAXTOK = {
    C.SENTIMENT: 8, C.MATH: 32, C.FACTUAL: 96, C.LOGICAL: 64,
    C.NER: 128, C.SUMMARISATION: 160, C.CODE_GEN: 384, C.CODE_DEBUG: 384,
}

# Per-level escalation rungs for 'unverifiable' status (level 4 = all categories).
# Derived from measured marginal efficiency, not intuition — see docs/escalation-math.md.
_LEVEL_CATEGORIES = {
    2: {C.MATH, C.FACTUAL},
    3: {C.MATH, C.FACTUAL, C.SENTIMENT, C.LOGICAL},
}

_ANSWER_LINE = re.compile(r"answer\s*[:=]\s*(.+?)\s*$", re.I | re.M)


def _last_answer_line(raw: str) -> Optional[str]:
    """Return the content of the last 'Answer: ...' line, if any."""
    matches = _ANSWER_LINE.findall(raw or "")
    return matches[-1].strip() if matches else None


# FREE format coercion (zero tokens): turn "wrong format" into a clean answer before the
# judge sees it. A preamble ("Sure, here's...") violates explicit constraints (e.g.
# "exactly one sentence") and pollutes a terse answer.
_PREAMBLE = re.compile(
    r"^(sure|okay|ok|certainly|of course|here(?:'s| is| are)\b[^:\n]*|the answer is)[,:.!—-]?\s*",
    re.I)


def _coerce(category: str, raw: str) -> str:
    """Clean the answer per category; always return the best available text."""
    text = (raw or "").strip()
    if not text:
        return text
    if category in (C.CODE_GEN, C.CODE_DEBUG):
        code = V.extract_code_block(text)
        return code or text
    stripped = text
    for _ in range(3):  # preambles can stack: "Sure, here's the answer: ..."
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
    """Parse the delimited 'ANSWER: ...' + 'PYTHON: <code>' format (newline-robust)."""
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
    needs_escalation: bool = False  # policy says escalate; execution is deferred (run.py)
    meta: dict = field(default_factory=dict)


class Solver:
    def __init__(self, local: Optional[LocalLLM] = None, remote: Optional[Fireworks] = None):
        self.local = local or LocalLLM()
        self.remote = remote or Fireworks()

    # ---------- escalation utilities ----------
    def escalate(self, prompt: str, tr: TaskResult) -> bool:
        """Execute the remote escalation for a task flagged needs_escalation.

        Called by run.py — safe from a worker thread (stateless per call; each tr is
        distinct). True if it succeeded and tr was updated; failure keeps the local answer.
        """
        return self._escalate(prompt, tr.category, tr)

    def _escalate(self, prompt: str, category: str, tr: TaskResult) -> bool:
        """Try to answer via remote; True if it succeeded and tr was updated.

        Guard is credential-based (not level-based): the level policy is already enforced
        by _decide; other callers (run.py tail-to-remote) may legitimately use remote at level 0.
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
        """PURE escalation policy (no execution) based on ESCALATION_LEVEL (0..4).

        Execution is deferred: solve() only flags needs_escalation; run.py runs
        solver.escalate() — in parallel on a thread pool, overlapping with local work on
        the next task (remote latency no longer blocks the local queue).

        status:
          verified     — an INDEPENDENT deterministic check passed → do not escalate
          verify_failed— the independent check FAILED (e.g. code error) → high-precision signal
          empty        — the local model produced nothing
          unverifiable — no independent check exists (math word-problem, logical, factual, etc.)

        The per-level category order is DATA-DRIVEN (eval v4 + Monte Carlo,
        docs/escalation-math.md): marginal efficiency ΔAcc/token = math ≈ factual
        >> sentiment ≈ logical >> the rest (local ~100%, escalating them is a pure loss).
        Note: factual replaces logical on the first rung because factual hallucinations are
        invisible to local verification, whereas some logical errors are already caught by
        the classify trap-routing.

        Calibration note: the model agreeing with itself (NL vs the model's own Python) is
        NOT 'verified' — a small model is confidently-consistent wrong. Only a genuinely
        independent check (code execution, letter counting, arithmetic WE parse) is verified.
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

    # ---------- per-category paths ----------
    def _solve_math(self, prompt: str, tr: TaskResult) -> None:
        # ONE local call. Delimited format (NOT JSON — multiline code breaks JSON):
        # ANSWER: <line> then PYTHON: <code>. Python helps recompute the arithmetic
        # (not an independent check: the model can be consistently wrong).
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

        # Best local answer: the number from Python execution if present, else the NL answer.
        tr.answer = f"{computed:g}" if computed is not None else str(nl_answer).strip()
        status = "empty" if not tr.answer else "unverifiable"  # math word-problem = trap
        tr.needs_escalation = self._decide(C.MATH, status)
        tr.route = "local"

    def _solve_code(self, prompt: str, category: str, tr: TaskResult) -> None:
        raw = self._local(prompt, category, max_tokens=config.local_max_tokens)
        code = V.extract_code_block(raw) or raw
        tr.answer = _coerce(category, raw)
        # INDEPENDENT check: the code must actually load without a syntax error (real execution).
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

    # ---------- entry ----------
    def solve(self, task_id: str, prompt: str) -> TaskResult:
        category = C.classify(prompt)
        tr = TaskResult(task_id=task_id, answer="", category=category)

        # Deterministic short-circuit: letter counting (a universal LLM blind spot).
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
