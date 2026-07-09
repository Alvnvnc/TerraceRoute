"""TerraceRoute batch entrypoint.

Read /input/tasks.json → solve each task → write /output/results.json (exit 0).
Reliability first (zero = an infra failure):
  - progressive atomic write: every task is flushed immediately (crash/timeout → the
    partial JSON stays valid)
  - watchdog: at WATCHDOG_S, write whatever exists (remaining tasks stay non-crashing),
    then exit 0
  - EVERY task always has a {task_id, answer} entry → the schema is always valid
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any

from . import classify as C
from .config import config
from .solve import Solver, TaskResult

_lock = threading.Lock()
_results: dict[str, str] = {}     # task_id -> answer (ordered by insertion)
_order: list[str] = []


def _atomic_write(path: str) -> None:
    payload = [{"task_id": tid, "answer": _results.get(tid, "")} for tid in _order]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)


def _flush() -> None:
    with _lock:
        _atomic_write(config.output_path)


def _load_tasks() -> list[dict[str, Any]]:
    try:
        with open(config.input_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [t for t in data if isinstance(t, dict) and "task_id" in t]
    except Exception as exc:  # noqa: BLE001
        print(f"[run] failed to read input: {exc}", file=sys.stderr)
    return []


def _watchdog(start: float) -> None:
    """Force a write and clean exit when approaching the 10-minute limit."""
    while True:
        time.sleep(1.0)
        if time.perf_counter() - start >= config.watchdog_s:
            print("[run] watchdog: flush & exit", file=sys.stderr)
            _flush()
            os._exit(0)


def main() -> int:
    start = time.perf_counter()
    tasks = _load_tasks()

    # Pre-seed: every task gets an entry (empty answer) → schema is valid even before solving.
    for t in tasks:
        tid = str(t["task_id"])
        if tid not in _results:
            _results[tid] = ""
            _order.append(tid)
    _flush()  # write the skeleton early (just in case)

    if not tasks:
        return 0

    threading.Thread(target=_watchdog, args=(start,), daemon=True).start()

    solver = Solver()
    n = len(tasks)
    all_tr: list[TaskResult] = []

    # PREEMPTIVE throttle: the hidden task count is unknown — if the per-task budget is
    # clearly tight from the start, don't wait for the reactive throttle (the first tasks
    # would be wasteful). Rough calibration from 2 vCPU measurements: ~7.5s/task @512 output.
    per_task_s = config.watchdog_s / max(1, n)
    _before = config.local_max_tokens
    for budget_s, cap in ((8.0, 256), (5.5, 128), (4.0, 96)):
        if per_task_s < budget_s and config.local_max_tokens > cap:
            config.local_max_tokens = cap
    if config.local_max_tokens != _before:
        print(f"[run] preemptive throttle: {n} tasks, {per_task_s:.1f}s/task → "
              f"LOCAL_MAX_TOKENS={config.local_max_tokens}", file=sys.stderr)
    # PARALLEL remote escalation: local solve() only flags needs_escalation; the remote
    # call runs on a worker thread → Fireworks latency overlaps with local work on the next
    # task (the 2 vCPUs stay on llama.cpp; the remote thread only waits on I/O).
    executor = ThreadPoolExecutor(max_workers=config.remote_concurrency)
    pending: dict[Future, TaskResult] = {}

    def _harvest(futures) -> None:
        """Write finished escalations (a failure keeps the local answer)."""
        for fut in futures:
            tr = pending.pop(fut)
            try:
                ok = fut.result()
            except Exception as exc:  # noqa: BLE001
                ok = False
                print(f"[run] escalate {tr.task_id} error: {exc}", file=sys.stderr)
            if ok:
                with _lock:
                    _results[tr.task_id] = tr.answer
                _flush()
            print(f"[run] escalate {tr.task_id} [{tr.route}]", file=sys.stderr)

    cut_at = n       # index of the first task NOT processed locally
    local_end = n    # local queue boundary; the tail past it was already sent to remote
    for i, t in enumerate(tasks):
        if i >= local_end:
            break  # the rest was already handed to remote (early tail-to-remote)
        elapsed = time.perf_counter() - start
        # Stop early if the time budget is exhausted (the watchdog is the backstop).
        # If remote is possible, reserve 20s: the remaining queue goes to remote in parallel.
        reserve = 20.0 if config.can_remote else 0.0
        if elapsed >= config.watchdog_s - reserve:
            cut_at = i
            print(f"[run] local budget exhausted at task {i + 1}/{n}", file=sys.stderr)
            break
        # Adaptive budget: the hidden task count is unknown — if the projection
        # (avg/task × remaining) exceeds the remaining time, shrink LOCAL_MAX_TOKENS
        # (CPU generation is ~linear in output length). A shorter answer >> a blank one.
        if i > 2 and config.local_max_tokens > 96:
            avg = elapsed / i
            if avg * (n - i) > (config.watchdog_s - elapsed) * 0.95:
                config.local_max_tokens = max(96, config.local_max_tokens // 2)
                print(f"[run] throttle: projection over budget → LOCAL_MAX_TOKENS="
                      f"{config.local_max_tokens}", file=sys.stderr)
        # EARLY tail-to-remote: throttle has bottomed out but the projection still won't
        # fit → cut the local queue NOW and send its tail to remote (network-bound, runs in
        # parallel with local work — not crammed into the last 20 seconds). Gated on
        # credentials, NOT level: a blank answer fails the gate outright; tokens are only a
        # ranking penalty. On a normal-sized task set this path is never touched.
        if (config.can_remote and local_end == n and i > 2
                and config.local_max_tokens <= 96):
            avg = elapsed / i
            time_left = config.watchdog_s - 20.0 - elapsed
            if avg * (n - i) > time_left:
                keep = max(0, int(time_left / avg * 0.9))
                local_end = min(n, i + keep)
                print(f"[run] early tail-to-remote: local can fit ~{keep} more → "
                      f"{n - local_end} tasks sent to remote now", file=sys.stderr)
                for tt in tasks[local_end:]:
                    ttid = str(tt["task_id"])
                    tprompt = str(tt.get("prompt", ""))
                    ttr = TaskResult(task_id=ttid, answer="", category=C.classify(tprompt))
                    all_tr.append(ttr)
                    pending[executor.submit(solver.escalate, tprompt, ttr)] = ttr
        tid = str(t["task_id"])
        prompt = str(t.get("prompt", ""))
        route = "error"
        try:
            tr = solver.solve(tid, prompt)
            all_tr.append(tr)
            ans, route = tr.answer, tr.route
        except Exception as exc:  # noqa: BLE001
            print(f"[run] task {tid} error: {exc}", file=sys.stderr)
            tr, ans = None, ""
        with _lock:
            _results[tid] = ans
        _flush()  # progressive: local answer first; escalation overwrites it on success
        if tr is not None and tr.needs_escalation and config.can_escalate:
            pending[executor.submit(solver.escalate, prompt, tr)] = tr
        _harvest([f for f in list(pending) if f.done()])
        print(f"[run] {i + 1}/{n} {tid} [{route}]", file=sys.stderr)

    # Late tail-to-remote (backstop): the watchdog cut BEFORE early-tail fired (e.g. the
    # early tasks were suddenly slow) → send the remaining queue [cut_at, local_end).
    if cut_at < local_end and config.can_remote:
        print(f"[run] late tail-to-remote: {local_end - cut_at} tasks left → remote",
              file=sys.stderr)
        for t in tasks[cut_at:local_end]:
            tid = str(t["task_id"])
            prompt = str(t.get("prompt", ""))
            tr = TaskResult(task_id=tid, answer="", category=C.classify(prompt))
            all_tr.append(tr)
            pending[executor.submit(solver.escalate, prompt, tr)] = tr

    # Drain remaining escalations until close to the watchdog (leave margin for the final flush).
    while pending and time.perf_counter() - start < config.watchdog_s - 5.0:
        done, _ = wait(list(pending), timeout=1.0, return_when=FIRST_COMPLETED)
        _harvest(done)
    if pending:
        print(f"[run] {len(pending)} escalations unfinished — local answers used",
              file=sys.stderr)
    executor.shutdown(wait=False, cancel_futures=True)

    _flush()
    elapsed = time.perf_counter() - start
    remote_tokens = sum(tr.remote_tokens for tr in all_tr)
    routes: dict[str, int] = {}
    for tr in all_tr:
        routes[tr.route] = routes.get(tr.route, 0) + 1
    print(f"[run] DONE n={n} time={elapsed:.1f}s REMOTE_TOKENS={remote_tokens} "
          f"routes={routes}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
