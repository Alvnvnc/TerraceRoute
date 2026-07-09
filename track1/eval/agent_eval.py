"""Evaluate the production agent/ pipeline against the 8-category task set (tasks_v4.jsonl).

Runs the Solver EXACTLY as in the container (agent/ code unchanged), with the Ollama
backend pointed at the remote AMD instance (qwen2.5:3b-instruct = the shipped model).
Grading:
  - exact / contains_all : deterministic, boundary-aware (no LLM)
  - pytests              : extract code from the answer, then EXECUTE + assert
  - judge                : gemma3:12b (independent, different family) vs a reference

Output: per-category accuracy table + router classification accuracy + a JSONL dump
(raw results for further analysis). This is the data that drives ESCALATION_LEVEL.

Run (env MUST be set before importing agent.config):
  OLLAMA_HOST=https://.../proxy/11434 python -m eval.agent_eval [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

# Dev env defaults → remote AMD (overridable from the shell).
os.environ.setdefault("LOCAL_BACKEND", "ollama")
os.environ.setdefault("LOCAL_MODEL", "qwen2.5:3b-instruct")
os.environ.setdefault("ESCALATION_LEVEL", "0")   # zero-API probe

from agent.http import post_json           # noqa: E402
from agent.solve import Solver             # noqa: E402
from agent.config import config            # noqa: E402
from agent import verify as V              # noqa: E402

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gemma3:12b")
TASKS_PATH = Path(__file__).parent / "tasks_v4.jsonl"
OUT_PATH = Path(__file__).parent / "agent_eval_results.jsonl"

_NUMERIC = re.compile(r"^[$€£]?-?\d[\d,]*\.?\d*%?$")


def _norm_match(answer: str, draft: str) -> bool:
    """Match `answer` in `draft` boundary-aware."""
    a = answer.strip()
    d = draft.lower()
    if _NUMERIC.match(a):
        core = re.sub(r"[,$€£%\s]", "", a).lstrip("-")
        if not core:
            return False
        pat = re.escape(core)
        if "." not in core:
            pat += r"(?:\.0+)?"
        return re.search(rf"(?<![\d.]){pat}(?!\d)(?!\.\d)", d) is not None
    return re.search(rf"(?<![a-z0-9]){re.escape(a.lower())}(?![a-z0-9])", d) is not None


def grade_exact(draft: str, answers: list[str]) -> bool:
    return any(_norm_match(a, draft) for a in answers if a)


def grade_contains_all(draft: str, answers: list[str]) -> bool:
    return all(_norm_match(a, draft) for a in answers if a)


def grade_pytests(draft: str, tests: str) -> bool:
    code = V.extract_code_block(draft)
    if not code:
        return False
    res = V.run_python(code + "\n\n" + tests, timeout=10.0)
    return res.ok


def grade_judge(task_prompt: str, draft: str, reference: str) -> bool:
    """Independent gemma3:12b judge via remote Ollama (retry lives in post_json)."""
    prompt = (
        "You are a strict grader. Judge ONLY whether the candidate answer is correct, "
        "equivalent in meaning to the reference, and obeys any explicit format/length "
        "constraint in the task.\n\n"
        f"Task:\n{task_prompt}\n\nReference answer:\n{reference}\n\n"
        f"Candidate answer:\n{draft}\n\nReply with ONLY 'YES' or 'NO'."
    )
    url = config.ollama_host.rstrip("/") + "/v1/chat/completions"
    try:
        data = post_json(url, {
            "model": JUDGE_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0, "max_tokens": 8, "stream": False,
        }, timeout=180)
        verdict = (data["choices"][0]["message"]["content"] or "").strip().upper()
        return verdict.startswith("YES")
    except Exception as exc:  # noqa: BLE001
        print(f"    [judge error: {exc}]", file=sys.stderr)
        return False


def grade(task: dict, draft: str) -> bool:
    kind = task["grader"]
    if kind == "exact":
        return grade_exact(draft, task.get("answer", []))
    if kind == "contains_all":
        return grade_contains_all(draft, task.get("answer", []))
    if kind == "pytests":
        return grade_pytests(draft, task.get("tests", ""))
    return grade_judge(task["input"], draft, task.get("reference", ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(TASKS_PATH))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in open(args.tasks) if l.strip()]
    if args.limit:
        tasks = tasks[: args.limit]

    print(f"agent eval: {len(tasks)} tasks | local={config.local_model} "
          f"@ {config.ollama_host} | judge={JUDGE_MODEL} | "
          f"ESCALATION_LEVEL={config.escalation_level}")

    solver = Solver()
    rows = []
    t_start = time.time()
    for t in tasks:
        t0 = time.time()
        # Retry-on-empty: the Jupyter proxy sometimes returns transient HTML → empty answer.
        # This is eval infra noise, not a prod failure mode (llama.cpp is in-process).
        for _ in range(4):
            tr = solver.solve(str(t["id"]), t["input"])
            if tr.answer.strip():
                break
            time.sleep(2)
        # Escalation is now deferred (parallel in run.py); in eval a sync call is fine.
        if tr.needs_escalation and config.can_escalate:
            solver.escalate(t["input"], tr)
        dt = time.time() - t0
        ok = grade(t, tr.answer)
        rows.append({
            "id": t["id"], "category": t["category"], "grader": t["grader"],
            "classified_as": tr.category, "route": tr.route, "correct": ok,
            "secs": round(dt, 2), "answer_chars": len(tr.answer),
            "remote_tokens": tr.remote_tokens, "remote_model": tr.remote_model,
            "prompt": t["input"], "answer": tr.answer,
        })
        mark = "ok " if ok else "WRONG"
        cls = "" if tr.category == t["category"] else f"  (classified: {tr.category})"
        print(f"  [{t['id']:>3}] {t['category']:<13} {mark} route={tr.route:<13} "
              f"{dt:5.1f}s{cls}")

    total_s = time.time() - t_start
    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ---- per-category summary ----
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r)
    n_ok = sum(r["correct"] for r in rows)
    n_cls = sum(r["classified_as"] == r["category"] for r in rows)

    print(f"\n{'category':<15}{'n':>4}{'acc':>8}{'classify':>10}{'avg s':>8}{'avg chars':>11}")
    for cat in sorted(by_cat):
        rs = by_cat[cat]
        acc = sum(r["correct"] for r in rs) / len(rs)
        cls = sum(r["classified_as"] == r["category"] for r in rs) / len(rs)
        avg_s = sum(r["secs"] for r in rs) / len(rs)
        avg_c = sum(r["answer_chars"] for r in rs) / len(rs)
        print(f"{cat:<15}{len(rs):>4}{acc:>8.2f}{cls:>10.2f}{avg_s:>8.1f}{avg_c:>11.0f}")
    print(f"{'TOTAL':<15}{len(rows):>4}{n_ok/len(rows):>8.2f}{n_cls/len(rows):>10.2f}"
          f"{total_s/len(rows):>8.1f}")
    n_remote = sum(r["route"] == "remote" for r in rows)
    tok_remote = sum(r["remote_tokens"] for r in rows)
    print(f"\nwall time {total_s:.0f}s | remote: {n_remote} tasks, {tok_remote} tokens "
          f"| raw results -> {OUT_PATH}")

    fails = [r for r in rows if not r["correct"]]
    if fails:
        print(f"\n{len(fails)} wrong:")
        for r in fails:
            print(f"  [{r['id']:>3}] {r['category']:<13} {r['prompt'][:70]}")
            print(f"        -> {r['answer'][:110].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
