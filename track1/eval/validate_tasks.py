"""Offline (LLM-free) sanity checker for an eval task set.

The escalation ladder (docs/escalation-math.md) is only as trustworthy as the tasks it is
measured on. A single wrong `tests` block or malformed answer silently biases the per-category
accuracy — so run this BEFORE spending GPU time on eval.agent_eval.

Checks:
  1. schema        — every task has id/category/input/grader and the fields its grader needs
  2. ids           — contiguous 1..N, no duplicates
  3. duplicates    — no two tasks share the same prompt
  4. balance       — report the per-category / per-grader distribution
  5. pytests       — run the embedded `solution` against `tests`; the tests MUST pass, proving
                     the task is satisfiable and the asserts are correct (catches typo'd tests)
  6. exact answers — non-empty, and every answer string is matchable by the same _norm_match the
                     real grader uses (so a correct answer can actually be detected)

Exit code is non-zero if any hard check fails, so it can gate CI / a pre-submit script.

Run:  python -m eval.validate_tasks [--tasks eval/tasks_v5.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from agent import verify as V

# Reuse the EXACT grader matching logic the real eval uses, so "valid answer" here means
# "detectable by the real grader" — not a second, divergent definition.
from eval.agent_eval import _norm_match

REQUIRED_BY_GRADER = {
    "exact": ["answer"],
    "contains_all": ["answer"],
    "pytests": ["tests"],
    "judge": ["reference"],
}


def check(tasks: list[dict]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    # --- ids contiguous & unique ---
    ids = [t.get("id") for t in tasks]
    if ids != list(range(1, len(tasks) + 1)):
        errors.append(f"ids are not contiguous 1..{len(tasks)} (got {ids[:5]}...)")

    # --- duplicate prompts ---
    seen_prompts: dict[str, int] = {}
    for t in tasks:
        p = t.get("input", "").strip().lower()
        if p in seen_prompts:
            errors.append(f"[{t['id']}] duplicate prompt of task {seen_prompts[p]}: {p[:60]!r}")
        else:
            seen_prompts[p] = t.get("id")

    # --- per-task schema + grader-specific validation ---
    for t in tasks:
        tid = t.get("id", "?")
        for field in ("category", "input", "grader"):
            if not t.get(field):
                errors.append(f"[{tid}] missing '{field}'")
        grader = t.get("grader")
        if grader not in REQUIRED_BY_GRADER:
            errors.append(f"[{tid}] unknown grader {grader!r}")
            continue
        for field in REQUIRED_BY_GRADER[grader]:
            if not t.get(field):
                errors.append(f"[{tid}] grader '{grader}' needs non-empty '{field}'")

        if grader in ("exact", "contains_all"):
            for a in t.get("answer", []):
                if not str(a).strip():
                    errors.append(f"[{tid}] empty answer string")
                elif not _norm_match(str(a), str(a)):
                    # An answer must match itself under the real matcher, else it can NEVER be
                    # detected in a model output (e.g. punctuation-only / regex-hostile answer).
                    warnings.append(f"[{tid}] answer {a!r} is not self-matchable by the grader")

        if grader == "pytests":
            tests = t.get("tests", "")
            sol = t.get("solution")
            if not sol:
                warnings.append(f"[{tid}] pytests task has no reference `solution` — cannot prove "
                                f"the tests are satisfiable offline")
                continue
            res = V.run_python(sol + "\n\n" + tests, timeout=10.0)
            if not res.ok:
                errors.append(f"[{tid}] reference solution FAILS its own tests:\n"
                              f"        {res.stderr.strip().splitlines()[-1] if res.stderr.strip() else '(no stderr)'}")

    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(Path(__file__).parent / "tasks_v5.jsonl"))
    args = ap.parse_args()

    path = Path(args.tasks)
    if not path.exists():
        print(f"task file not found: {path} (run `python -m eval.gen_tasks_v5` first)", file=sys.stderr)
        return 2
    tasks = [json.loads(l) for l in open(path) if l.strip()]

    errors, warnings = check(tasks)

    cats = Counter(t.get("category") for t in tasks)
    graders = Counter(t.get("grader") for t in tasks)
    print(f"validating {len(tasks)} tasks in {path.name}")
    print(f"  categories: {dict(sorted(cats.items()))}")
    print(f"  graders   : {dict(sorted(graders.items()))}")
    n_code = sum(1 for t in tasks if t.get("grader") == "pytests" and t.get("solution"))
    print(f"  ran {n_code} reference solutions against their tests")

    for w in warnings:
        print(f"  WARN  {w}")
    for e in errors:
        print(f"  FAIL  {e}")

    if errors:
        print(f"\n{len(errors)} error(s), {len(warnings)} warning(s) — NOT OK")
        return 1
    print(f"\nOK — 0 errors, {len(warnings)} warning(s). Task set is well-formed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
