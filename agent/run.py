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
    all_tr: list[TaskResult] = []

    # Throttle PREEMPTIF: jumlah task hidden tak diketahui — kalau anggaran per task
    # sudah jelas sempit dari awal, jangan tunggu throttle reaktif (task awal boros).
    # Kalibrasi kasar dari pengukuran 2 vCPU: ~7.5 dtk/task @512 output token.
    per_task_s = config.watchdog_s / max(1, n)
    _before = config.local_max_tokens
    for budget_s, cap in ((8.0, 256), (5.5, 128), (4.0, 96)):
        if per_task_s < budget_s and config.local_max_tokens > cap:
            config.local_max_tokens = cap
    if config.local_max_tokens != _before:
        print(f"[run] preemptive throttle: {n} task, {per_task_s:.1f}s/task → "
              f"LOCAL_MAX_TOKENS={config.local_max_tokens}", file=sys.stderr)
    # Eskalasi remote PARALEL: solve() lokal hanya menandai needs_escalation; call
    # remote jalan di worker thread → latensi Fireworks overlap dgn kerja lokal task
    # berikutnya (2 vCPU tetap fokus llama.cpp; thread remote cuma menunggu I/O).
    executor = ThreadPoolExecutor(max_workers=config.remote_concurrency)
    pending: dict[Future, TaskResult] = {}

    def _harvest(futures) -> None:
        """Tulis hasil eskalasi yang sudah selesai (gagal = jawaban lokal bertahan)."""
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

    cut_at = n       # index task pertama yang TIDAK sempat diproses lokal
    local_end = n    # batas antrean lokal; ekor di belakangnya sudah dikirim ke remote
    for i, t in enumerate(tasks):
        if i >= local_end:
            break  # sisanya sudah dilempar ke remote (early tail-to-remote)
        elapsed = time.perf_counter() - start
        # Hentikan lebih awal bila anggaran waktu habis (watchdog jadi cadangan).
        # Bila remote mungkin, sisakan 20s: sisa antrean dilempar ke remote paralel.
        reserve = 20.0 if config.can_remote else 0.0
        if elapsed >= config.watchdog_s - reserve:
            cut_at = i
            print(f"[run] anggaran lokal habis di task {i + 1}/{n}", file=sys.stderr)
            break
        # Anggaran adaptif (plan §3.2): jumlah task hidden tak diketahui — bila proyeksi
        # (rata2/task × sisa) melebihi sisa waktu, pangkas LOCAL_MAX_TOKENS (generasi CPU
        # ~linier thd output token). Jawaban lebih pendek >> jawaban kosong.
        if i > 2 and config.local_max_tokens > 96:
            avg = elapsed / i
            if avg * (n - i) > (config.watchdog_s - elapsed) * 0.95:
                config.local_max_tokens = max(96, config.local_max_tokens // 2)
                print(f"[run] throttle: proyeksi lewat budget → LOCAL_MAX_TOKENS="
                      f"{config.local_max_tokens}", file=sys.stderr)
        # EARLY tail-to-remote: throttle sudah mentok tapi proyeksi tetap tak muat →
        # potong antrean lokal SEKARANG dan kirim ekornya ke remote (network-bound,
        # jalan paralel dgn kerja lokal — bukan diserbu di 20 detik terakhir).
        # Berbasis kredensial, BUKAN level: jawaban kosong = gagal gate pasti;
        # token cuma penalti ranking. Di task set normal jalur ini tak tersentuh.
        if (config.can_remote and local_end == n and i > 2
                and config.local_max_tokens <= 96):
            avg = elapsed / i
            time_left = config.watchdog_s - 20.0 - elapsed
            if avg * (n - i) > time_left:
                keep = max(0, int(time_left / avg * 0.9))
                local_end = min(n, i + keep)
                print(f"[run] early tail-to-remote: lokal hanya muat ~{keep} lagi → "
                      f"{n - local_end} task dikirim remote sekarang", file=sys.stderr)
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
        _flush()  # progressive: jawaban lokal dulu; eskalasi menimpanya bila sukses
        if tr is not None and tr.needs_escalation and config.can_escalate:
            pending[executor.submit(solver.escalate, prompt, tr)] = tr
        _harvest([f for f in list(pending) if f.done()])
        print(f"[run] {i + 1}/{n} {tid} [{route}]", file=sys.stderr)

    # Late tail-to-remote (cadangan): watchdog memotong SEBELUM early-tail sempat
    # menyala (mis. task awal lambat mendadak) → sisa antrean [cut_at, local_end).
    if cut_at < local_end and config.can_remote:
        print(f"[run] late tail-to-remote: {local_end - cut_at} task sisa → remote",
              file=sys.stderr)
        for t in tasks[cut_at:local_end]:
            tid = str(t["task_id"])
            prompt = str(t.get("prompt", ""))
            tr = TaskResult(task_id=tid, answer="", category=C.classify(prompt))
            all_tr.append(tr)
            pending[executor.submit(solver.escalate, prompt, tr)] = tr

    # Drain sisa eskalasi sampai mendekati watchdog (sisa margin utk flush terakhir).
    while pending and time.perf_counter() - start < config.watchdog_s - 5.0:
        done, _ = wait(list(pending), timeout=1.0, return_when=FIRST_COMPLETED)
        _harvest(done)
    if pending:
        print(f"[run] {len(pending)} eskalasi tak selesai — jawaban lokal dipakai",
              file=sys.stderr)
    executor.shutdown(wait=False, cancel_futures=True)

    _flush()
    elapsed = time.perf_counter() - start
    remote_tokens = sum(tr.remote_tokens for tr in all_tr)
    routes: dict[str, int] = {}
    for tr in all_tr:
        routes[tr.route] = routes.get(tr.route, 0) + 1
    print(f"[run] SELESAI n={n} waktu={elapsed:.1f}s REMOTE_TOKENS={remote_tokens} "
          f"routes={routes}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
