# Submissions Log — TerraceRoute Track 1

Leaderboard = oracle kalibrasi. Setiap submit ubah SATU variabel, catat status & skor di sini.
Rate limit 10/jam. Deadline 11 Juli 2026 (cek jam persis di Event Schedule lablab.ai).

## Knob utama
- `ESCALATION_LEVEL` (0..4, urutan rung BERBASIS DATA — eval/escalation-math.md):
  0=zero-API · 1=verify-fail+empty · 2=+math&factual · 3=+sentiment&logical · 4=semua unverifiable
- `REMOTE_CONCURRENCY` (default 4, eskalasi paralel) · `LOCAL_MAX_TOKENS` ·
  `REMOTE_MAX_TOKENS` per kategori (solve.py) · `ALLOWED_MODELS` (dari harness)

## Strategi probing (protokol walk-down, lihat eval/escalation-math.md §protokol)
Ekspektasi dari eval v4 + Monte Carlo: L0 acc ~0.85-0.91 (0 token) · L2 ~0.91 (~1.1-1.4k token)
· L3 (+~1k token) · L4 (mahal, hanya bila gate >~0.92).
1. **v0 = level 0 (zero-API).** Lolos gate → SELESAI & FREEZE (0 token tak terkalahkan).
2. Gagal → **level 3** (probabilitas lolos tinggi, token moderat) — JANGAN naik satu-satu;
   satu kegagalan hanya bilang "gate > acc(L0)", bukan seberapa jauh.
3. Lolos di lvl3 → walk-down: coba lvl2, lalu lvl1; berhenti di rung terendah yang masih
   lolos → resubmit rung itu → FREEZE. (Aman: konfigurasi lolos selalu bisa dikembalikan.)
4. Gagal juga di lvl4 → masalahnya bukan kebijakan eskalasi (akurasi remote / kategori kuat
   kami) — investigasi feedback, jangan naik-turun buta.
5. Fine-tuning pasca-lolos (bila perlu margin token): kecilkan REMOTE_MAX_TOKENS,
   pilih model remote non-reasoning yang outputnya pendek.

## Log

| # | waktu (WIB) | image tag | level | perubahan | status | skor/tokens | catatan |
|---|---|---|---|---|---|---|---|
| 0 | _isi_ | terraceroute:v0 | 0 | baseline zero-API | _isi_ | _isi_ | probe: lokal saja lolos gate? |
|   |   |   |   |   |   |   |   |

## Status harness (arti singkat)
PULL_ERROR=image tak publik/bukan amd64 · RUNTIME_ERROR=exit≠0 · TIMEOUT=>10min ·
OUTPUT_MISSING=tak tulis results.json · INVALID_RESULTS_SCHEMA=format salah ·
MODEL_VIOLATION=model di luar ALLOWED_MODELS · IMAGE_TOO_LARGE=>10GB ·
ACCURACY_GATE_FAILED=akurasi di bawah threshold (naikkan escalation) ·
ZERO_API_CALLS=(bukan error) 0 call remote = valid.
