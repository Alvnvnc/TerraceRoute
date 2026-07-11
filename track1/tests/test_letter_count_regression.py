"""Regression tests for the letter-count shortcut (eval v5 tasks 92 & 122).

The deterministic counter used to run before category dispatch, so code prompts
containing "for x in lst" (92) or "how many times the character c appears in
the string s" (122) were answered with a number instead of code. The fix is
two-layered: solve() gates the shortcut to non-code categories, and
verify.try_count_letter() itself refuses prompts with code markers.
"""
import os

os.environ.setdefault("ESCALATION_LEVEL", "0")

from agent import classify as C
from agent import verify as V
from agent.solve import Solver

# Verbatim from eval/tasks_v5.jsonl (ids 92 and 122).
TASK_92 = (
    "This function should return the number of items greater than 10 in a list "
    "but is buggy: def count_big(lst): return sum(1 for x in lst if x < 10). Fix it."
)
TASK_122 = (
    "Write a Python function named count_char(s, c) that returns how many times "
    "the character c appears in the string s."
)
FIX_92 = "def count_big(lst):\n    return sum(1 for x in lst if x > 10)"
GEN_122 = "def count_char(s, c):\n    return s.count(c)"


class StubLocal:
    """Local-LLM stand-in returning a canned answer (no network)."""

    def __init__(self, reply: str):
        self.reply = reply

    def chat(self, user, system=None, max_tokens=512, temperature=0.0):
        return self.reply


def _solve(prompt: str, reply: str):
    return Solver(local=StubLocal(reply), remote=object()).solve("t", prompt)


def test_task_92_reaches_code_solver():
    assert C.classify(TASK_92) == C.CODE_DEBUG
    tr = _solve(TASK_92, FIX_92)
    assert tr.route != "deterministic"
    assert "def count_big" in tr.answer


def test_task_122_reaches_code_solver():
    assert C.classify(TASK_122) == C.CODE_GEN
    tr = _solve(TASK_122, GEN_122)
    assert tr.route != "deterministic"
    assert "def count_char" in tr.answer


def test_counter_refuses_code_prompts():
    assert V.try_count_letter(TASK_92) is None
    assert V.try_count_letter(TASK_122) is None


def test_genuine_letter_count_still_deterministic():
    tr = _solve("How many r's are in the word strawberry?", "unused")
    assert tr.route == "deterministic"
    assert tr.answer == "3"


def test_counter_positive_cases():
    assert V.try_count_letter("How many r's are in the word strawberry?") == 3
    assert V.try_count_letter(
        "Count the number of occurrences of the letter 'e' in 'excellence'."
    ) == 4
