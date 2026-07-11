"""Track 2 batch entrypoint — the Track 1 reliability layer, re-targeted.

Read /input/tasks.json → caption each clip → write /output/results.json (exit 0).
Zero-proofing (identical invariants to Track 1's run.py):
  - pre-seed: every task_id × every requested style gets a fallback caption BEFORE any
    work (missing style = 0 for the clip; malformed JSON = 0 overall)
  - progressive atomic write after every finished clip
  - watchdog: at WATCHDOG_S flush whatever exists and exit 0
Clips are processed in parallel (all stages are network-bound).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from . import styles as S
from . import vlm
from .config import config
from .pipeline import _LAST_RESORT, process_clip

_lock = threading.Lock()
_results: dict[str, dict[str, str]] = {}   # task_id -> {style: caption}
_order: list[str] = []


def _atomic_write(path: str) -> None:
    payload = [{"task_id": tid, "captions": _results.get(tid, {})} for tid in _order]
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


def _req_styles(task: dict[str, Any]) -> list[str]:
    raw = task.get("styles")
    if isinstance(raw, list) and raw:
        return [str(s) for s in raw]
    return list(S.ALL_STYLES)  # unspecified → all four (safe default)


def _watchdog(start: float) -> None:
    while True:
        time.sleep(1.0)
        if time.perf_counter() - start >= config.watchdog_s:
            print("[run] watchdog: flush & exit", file=sys.stderr)
            _flush()
            os._exit(0)


def main() -> int:
    start = time.perf_counter()
    tasks = _load_tasks()

    # Pre-seed: a valid, complete answer exists from second zero.
    for t in tasks:
        tid = str(t["task_id"])
        if tid not in _results:
            _results[tid] = {s: _LAST_RESORT.get(s, _LAST_RESORT[S.FORMAL])
                             for s in _req_styles(t)}
            _order.append(tid)
    _flush()

    if not tasks:
        return 0

    threading.Thread(target=_watchdog, args=(start,), daemon=True).start()
    vlm.warm_up()

    n = len(tasks)
    workers = min(config.clip_workers, n)
    # Soft per-clip budget: parallel lanes share the wall clock.
    budget_s = (config.watchdog_s - 20.0) * workers / n
    print(f"[run] {n} clips | workers={workers} | budget={budget_s:.0f}s/clip",
          file=sys.stderr)

    def _one(task: dict[str, Any]):
        tid = str(task["task_id"])
        url = str(task.get("video_url", ""))
        return process_clip(tid, url, _req_styles(task), budget_s)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, t): str(t["task_id"]) for t in tasks}
        for fut in as_completed(futs):
            tid = futs[fut]
            try:
                res = fut.result()
                with _lock:
                    _results[tid].update(res.captions)
                routes = ",".join(sorted(set(res.routes.values()))) or "none"
                print(f"[run] {tid} done [{routes}]", file=sys.stderr)
            except Exception as exc:  # noqa: BLE001 — pre-seeded fallback stays in place
                print(f"[run] {tid} error: {exc}", file=sys.stderr)
            done += 1
            _flush()

    _flush()
    print(f"[run] DONE n={n} clips={done} time={time.perf_counter() - start:.1f}s",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
