"""Verifikasi deterministik GRATIS — menaikkan akurasi lokal tanpa token remote.

Ide: untuk kategori yang bisa dicek mesin (math, code, counting), jangan percaya
LLM buta. Hitung/jalankan sendiri; jawaban terverifikasi = tak perlu escalate.
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

# --- Aritmetika aman (tanpa eval bebas) ---
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def safe_arith(expr: str) -> Optional[float]:
    """Evaluasi ekspresi aritmetika murni; None bila tak valid/tak aman."""
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


# --- Eksekusi kode Python tersandbox (timeout + subprocess terisolasi) ---
@dataclass
class ExecResult:
    ok: bool
    stdout: str
    stderr: str


def run_python(code: str, timeout: float = 5.0) -> ExecResult:
    """Jalankan kode Python di subprocess terpisah dengan timeout keras."""
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
    """Ambil blok kode dari markdown fence, atau seluruh teks bila terlihat seperti kode."""
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S | re.I)
    if m:
        return m.group(1).strip()
    if re.search(r"\bdef \w+\(|\bimport \b|\breturn\b", text):
        return text.strip()
    return None


# --- Counting (blind-spot universal LLM: hitung huruf/kata) ---
# Dua pola:
#   A. "...letter/character X ... in (the word) WORD"  (mis. strawberry)
#   B. "...X['s] (are/appear) in WORD"                 (mis. "how many r's in strawberry")
# Target WORD wajib >=2 huruf & huruf yang dihitung wajib token tunggal terisolasi →
# menjaga dari false-positive ("how many apples in the basket").
_COUNT_A = re.compile(
    r"(?:letter|character)\s+['\"]?([a-z])['\"]?.*?\bin\b(?:\s+the)?(?:\s+word)?\s+"
    r"['\"]?([a-z]{2,})['\"]?", re.I | re.S)
_COUNT_B = re.compile(
    r"\b([a-z])(?:['’]s|s)?\s+(?:are\s+|is\s+|appears?\s+|occur\w*\s+|there\s+)*"
    r"in\b(?:\s+the)?(?:\s+word)?\s+['\"]?([a-z]{2,})['\"]?", re.I | re.S)
_COUNT_TRIGGER = ("how many", "number of", "count ")


def try_count_letter(prompt: str) -> Optional[int]:
    """Deteksi 'berapa banyak huruf X di WORD' → hitung deterministik (case-insensitive)."""
    low = prompt.lower()
    if not any(t in low for t in _COUNT_TRIGGER):
        return None
    m = _COUNT_A.search(prompt) or _COUNT_B.search(prompt)
    if not m:
        return None
    letter, word = m.group(1).lower(), m.group(2).lower()
    return word.count(letter)


def numbers_in(text: str) -> list[float]:
    """Semua angka dalam teks (untuk cocokkan jawaban NL vs hasil hitung)."""
    out = []
    for tok in re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", "")):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def answers_agree_numeric(computed: float, nl_answer: str, tol: float = 1e-6) -> bool:
    """True bila hasil hitung muncul di antara angka pada jawaban natural-language."""
    for n in numbers_in(nl_answer):
        if abs(n - computed) <= tol or (computed != 0 and abs(n - computed) / abs(computed) < 1e-4):
            return True
    return False
