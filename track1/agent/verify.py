"""FREE deterministic verification — raises local accuracy at zero remote tokens.

Idea: for machine-checkable categories (math, code, counting), do not trust the LLM
blindly. Compute/run it ourselves; a verified answer needs no escalation.
"""
from __future__ import annotations

import ast
import operator
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional

# --- Safe arithmetic (no free-form eval) ---
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_arith(expr: str) -> Optional[float]:
    """Evaluate a pure arithmetic expression; None if invalid/unsafe."""
    try:
        node = ast.parse(expr, mode="eval").body
        return _eval_node(node)
    except Exception:
        return None


def _eval_node(node) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


# --- Sandboxed Python execution (timeout + isolated subprocess) ---
@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str


def run_python(code: str, timeout: float = 5.0) -> ExecResult:
    """Run Python code in a separate subprocess with a hard timeout."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as f:
        f.write(code)
        f.flush()
        try:
            proc = subprocess.run(
                [sys.executable, "-I", f.name],
                capture_output=True, text=True, timeout=timeout,
            )
            return ExecResult(proc.returncode == 0, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired:
            return ExecResult(False, "", "timeout")
        except Exception as exc:  # noqa: BLE001
            return ExecResult(False, "", str(exc))


def extract_code_block(text: str) -> Optional[str]:
    """Take the code out of a markdown fence, or the whole text if it looks like code."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    if re.search(r"\bdef \w+\(|\bimport \b|\breturn\b", text):
        return text.strip()
    return None


# --- Counting (a universal LLM blind spot: count letters/words) ---
# Two patterns:
#   A. "...letter/character X ... in (the word) WORD"  (e.g. strawberry)
#   B. "...X['s] (are/appear) in WORD"                 (e.g. "how many r's in strawberry")
# The target WORD must be >=2 letters and the counted letter must be a lone isolated
# token → guards against false positives ("how many apples in the basket").
_COUNT_A = re.compile(
    r"(?:letter|character)\s+['\"]?([a-z])['\"]?.*?\bin\b(?:\s+the)?(?:\s+word)?\s+"
    r"['\"]?([a-z]{2,})['\"]?", re.I | re.S)
_COUNT_B = re.compile(
    r"\b([a-z])(?:['’]s|s)?\s+(?:are\s+|is\s+|appears?\s+|occur\w*\s+|there\s+)*"
    r"in\b(?:\s+the)?(?:\s+word)?\s+['\"]?([a-z]{2,})['\"]?", re.I | re.S)
_COUNT_TRIGGER = ("how many", "number of", "count ")
# Code prompts ("def f(lst): ... for x in lst", "write a function that returns how
# many times c appears in the string s") match the count patterns as false positives.
# solve() already gates by category; this guard also protects misclassified prompts.
_COUNT_CODE = re.compile(r"```|\bdef \w+\(|\bfunction\b|\breturn\b|=>|\bclass \w+", re.I)


def try_count_letter(prompt: str) -> Optional[int]:
    """Detect 'how many of letter X in WORD' → count deterministically (case-insensitive)."""
    low = prompt.lower()
    if not any(t in low for t in _COUNT_TRIGGER):
        return None
    if _COUNT_CODE.search(prompt):
        return None
    m = _COUNT_A.search(prompt) or _COUNT_B.search(prompt)
    if not m:
        return None
    letter, word = m.group(1).lower(), m.group(2).lower()
    return word.count(letter)


def numbers_in(text: str) -> list[float]:
    """All numbers in the text (to match an NL answer against a computed value)."""
    out = []
    for tok in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def answers_agree_numeric(computed: float, nl_answer: str, tol: float = 1e-6) -> bool:
    """True if the computed value appears among the numbers in a natural-language answer."""
    for n in numbers_in(nl_answer):
        if abs(n - computed) <= tol or (computed != 0 and abs(n - computed) / abs(computed) < 1e-4):
            return True
    return False
