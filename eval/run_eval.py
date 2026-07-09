"""Fit calibrator + tune threshold τ1 pada eval set berlabel.

Alur:
  1. Untuk tiap task: ambil draft lokal + uncertainty (reuse Router).
  2. Grade draft dengan grader HYBRID (exact-match deterministik + judge INDEPENDEN gemma)
     -> label local_wrong (0/1). Lihat eval/grader.py.
  3. Fit isotonic calibrator, simpan ke settings.calibrator_path.
  4. Sweep τ1: simulasikan akurasi & rata-rata remote token, pilih τ1 termurah s.t. acc >= floor.

Task set: settings.eval_tasks (default sample_tasks_v2.jsonl, sengaja lebih sulit
agar 14B benar-benar gagal di sebagian task -> kalibrator punya sinyal).

Jalankan:  python -m eval.run_eval  (Ollama harus hidup: local_model + judge_model)
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

from app.calibrator import Calibrator
from app.config import settings
from app.ollama_client import OllamaClient
from app.router import Router
from app.tokens import estimate_tokens
from app.uncertainty import (
    answer_key,
    self_consistency_uncertainty,
    uncertainty_from_logprob,
)
from eval.grader import grade

VERIFIERS = [OllamaClient(model=m.strip())
             for m in settings.verifier_models.split(",") if m.strip()]

TASKS = Path(settings.eval_tasks)
if not TASKS.is_absolute():
    TASKS = Path(__file__).parent.parent / TASKS


def load_tasks() -> list[dict]:
    with open(TASKS) as f:
        return [json.loads(line) for line in f if line.strip()]


async def collect(router: Router, tasks: list[dict]) -> list[dict]:
    """Ukur perplexity (gratis) vs cross-model (sinyal utama). Self-consistency dibuang
    (terbukti anti-korelasi) supaya cukup cepat utk eval set besar."""
    rows = []
    for t in tasks:
        text = t["input"]
        gen = await router.local.generate(text)               # draft temp rendah + logprobs
        u_ppl = uncertainty_from_logprob(gen.avg_logprob, gen.min_logprob,
                                         settings.logprob_min_weight)

        # Cross-model: jawaban VERBOSE (bernalar) dari primary (reuse draft) + verifier
        # beda-keluarga. Ini sinyal routing utama = yg dipakai router saat inference.
        model_answers = [gen.text]
        for v in VERIFIERS:
            va = await v.generate(text, temperature=0.0, want_logprobs=False, num_predict=768)
            model_answers.append(va.text)
        u_cross, method = self_consistency_uncertainty(model_answers)   # ketidaksepakatan antar-model

        correct = await grade(t, gen.text)
        wrong = 0 if correct else 1
        rows.append({
            # "u" = sinyal yg dipakai router saat inference (cross-model) -> calibrator dilatih atasnya
            "id": t["id"], "u": u_cross, "u_ppl": u_ppl,
            "u_cross": u_cross, "method": method, "wrong": wrong,
            "grader": t.get("grader", "judge"),
            "in_tokens": estimate_tokens(text), "draft": gen.text,
        })
        pp = f"{u_ppl:.2f}" if u_ppl is not None else " NA "
        print(f"  task {t['id']:>3}  u_ppl={pp} u_cross={u_cross:.2f}  "
              f"local_{'WRONG' if wrong else 'ok '}  [{t.get('grader','judge'):>5}]  ({method})")
    return rows


def _pointbiserial(us: list[float], ws: list[int]) -> Optional[float]:
    """Korelasi point-biserial antara sinyal u dan label wrong. None kalau tak terdefinisi."""
    pairs = [(u, w) for u, w in zip(us, ws) if u is not None]
    n = len(pairs)
    if n < 3:
        return None
    mu = sum(u for u, _ in pairs) / n
    mw = sum(w for _, w in pairs) / n
    su = (sum((u - mu) ** 2 for u, _ in pairs) / n) ** 0.5
    sw = (sum((w - mw) ** 2 for _, w in pairs) / n) ** 0.5
    if su == 0 or sw == 0:
        return None
    cov = sum((u - mu) * (w - mw) for u, w in pairs) / n
    return cov / (su * sw)


def sweep_tau(rows: list[dict], cal: Calibrator, floor: float) -> dict:
    """Simulasi: escalate jika p_wrong>=τ1 (remote diasumsikan benar).

    Remote token proxy = input tokens (setelah asumsi kompresi 0.6) + estimasi output.
    """
    n = len(rows)
    best = None
    grid = [i / 100 for i in range(5, 100, 5)]
    print("\n  τ1     acc    remoteRate  avgRemoteTok")
    for tau in grid:
        correct = 0
        remote_tok = 0
        remote_calls = 0
        for r in rows:
            p = cal.predict(r["u"])
            if p >= tau:  # escalate -> remote (diasumsikan benar)
                correct += 1
                remote_calls += 1
                remote_tok += int(r["in_tokens"] * 0.6) + 200  # proxy in(compressed)+out
            else:          # lokal
                correct += 1 - r["wrong"]
        acc = correct / n
        avg_rt = remote_tok / n
        rate = remote_calls / n
        flag = "  <= feasible" if acc >= floor else ""
        print(f"  {tau:.2f}   {acc:.3f}   {rate:.3f}      {avg_rt:7.1f}{flag}")
        if acc >= floor and (best is None or avg_rt < best["avg_remote_tokens"]):
            best = {"tau1": tau, "accuracy": acc, "remote_rate": rate,
                    "avg_remote_tokens": avg_rt}
    return best or {"tau1": None, "note": "no threshold meets accuracy floor"}


async def main():
    tasks = load_tasks()
    router = Router()
    print(f"Collecting draft+uncertainty for {len(tasks)} tasks "
          f"(model={settings.local_model})...")
    rows = await collect(router, tasks)

    n_wrong = sum(r["wrong"] for r in rows)
    ws = [r["wrong"] for r in rows]
    print(f"\nLocal fail rate: {n_wrong}/{len(rows)} = {n_wrong/len(rows):.1%} "
          f"(judge={settings.judge_model}) — butuh ~20-40% agar kalibrasi bermakna")

    # Bukti: sinyal mana yang benar-benar memprediksi error? (corr > 0 = makin ragu makin salah)
    print("\nKorelasi sinyal vs error (point-biserial, makin + makin baik):")
    for name, key in [("perplexity  ", "u_ppl"), ("cross-model ", "u_cross")]:
        r = _pointbiserial([row[key] for row in rows], ws)
        print(f"  {name} corr = {r:+.3f}" if r is not None else f"  {name} corr = NA")

    cal = Calibrator().fit([r["u"] for r in rows], [r["wrong"] for r in rows])
    cal.save(settings.calibrator_path)
    print(f"\nCalibrator fitted & saved -> {settings.calibrator_path}")

    best = sweep_tau(rows, cal, settings.accuracy_floor)
    print("\nRecommended threshold (accuracy floor "
          f"{settings.accuracy_floor}):\n  {json.dumps(best, indent=2)}")
    print("\nSet TAU1 di .env sesuai rekomendasi di atas, lalu jalankan ulang server.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(1)
