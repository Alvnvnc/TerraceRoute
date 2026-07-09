"""Entrypoint batch TerraceRoute.

Baca /input/tasks.json → selesaikan tiap task → tulis /output/results.json (exit 0).
Prioritas keandalan (nol = kegagalan infra):
  - progressive atomic write: tiap task langsung di-flush (crash/timeout → JSON parsial tetap valid)
  - watchdog: pada WATCHDOG_S, tulis apa yang ada (task tersisa diisi non-crash) lalu exit 0
  - SEMUA task selalu punya entri {task_id, answer} → schema selalu valid
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

from .config import config
from .solve import Solver

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
        print(f"[run] gagal baca input: {exc}", file=sys.stderr)
    return []


def _watchdog(start: float) -> None:
    """Paksa tulis & keluar bersih bila mendekati batas 10 menit."""
    while True:
        time.sleep(1.0)
        if time.perf_counter() - start >= config.watchdog_s:
            print("[run] watchdog: flush & exit", file=sys.stderr)
            _flush()
            os._exit(0)


def main() -> int:
    start = time.perf_counter()
    tasks = _load_tasks()

    # Pre-seed: setiap task punya entri (answer kosong) → schema valid walau belum diproses.
    for t in tasks:
        tid = str(t["task_id"])
        if tid not in _results:
            _results[tid] = ""
            _order.append(tid)
    _flush()  # tulis kerangka lebih awal (jaga-jaga)

    if not tasks:
        return 0

    threading.Thread(target=_watchdog, args=(start,), daemon=True).start()

    solver = Solver()
    n = len(tasks)
    remote_tokens = 0
    routes: dict[str, int] = {}
    for i, t in enumerate(tasks):
        # Hentikan lebih awal bila anggaran waktu habis (watchdog jadi cadangan).
        if time.perf_counter() - start >= config.watchdog_s:
            print(f"[run] anggaran waktu habis di task {i + 1}/{n}, sisanya dikosongkan",
                  file=sys.stderr)
            break
        tid = str(t["task_id"])
        prompt = str(t.get("prompt", ""))
        route = "error"
        try:
            tr = solver.solve(tid, prompt)
            ans, route = tr.answer, tr.route
            remote_tokens += tr.remote_tokens
        except Exception as exc:  # noqa: BLE001
            print(f"[run] task {tid} error: {exc}", file=sys.stderr)
            ans = ""
        routes[route] = routes.get(route, 0) + 1
        with _lock:
            _results[tid] = ans
        _flush()  # progressive
        print(f"[run] {i + 1}/{n} {tid} [{route}]", file=sys.stderr)

    _flush()
    elapsed = time.perf_counter() - start
    print(f"[run] SELESAI n={n} waktu={elapsed:.1f}s REMOTE_TOKENS={remote_tokens} "
          f"routes={routes}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
