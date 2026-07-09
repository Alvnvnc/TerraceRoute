# Submissions Log — TerraceRoute Track 1

Leaderboard = oracle kalibrasi. Setiap submit ubah SATU variabel, catat status & skor di sini.
Rate limit 10/jam. Deadline 11 Juli 2026 (cek jam persis di Event Schedule lablab.ai).

## Knob utama
- `ESCALATION_LEVEL`: 0=zero-API · 1=escalate verify-fail+empty · 2=+trap(math,logical) · 3=agresif
- `LOCAL_MODEL` / GGUF di image · `LOCAL_MAX_TOKENS` · `REMOTE_MAX_TOKENS` per kategori (solve.py)
- `ALLOWED_MODELS` (dari harness) → pilihan model remote

## Strategi probing
1. **v0 (level 0, zero-API):** apakah 3B lokal lolos accuracy gate? → status `ACCURACY_GATE_FAILED`
   atau skor (REMOTE_TOKENS=0 = skor token sempurna bila lolos).
2. Bila gagal gate: naikkan ke **level 1** (escalate hanya kode-gagal & lokal-kosong) → cek gate.
3. Masih gagal: **level 2** (escalate math+logical). Ini biasanya titik lolos gate.
4. Setelah lolos: turunkan token — kecilkan REMOTE_MAX_TOKENS, pilih model remote hemat,
   pindahkan kategori dari escalate→local bila akurasinya ternyata cukup.
5. Final: konfigurasi lolos-gate dgn REMOTE_TOKENS terendah + MARGIN akurasi.

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
