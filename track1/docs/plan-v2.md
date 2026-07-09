# TerraceRoute — Plan v2 (FINAL SPRINT, 9–11 Juli 2026)

> Menggantikan `plan.md`. Aturan kini **TERVERIFIKASI** dari Participant Guide resmi
> (PDF: https://drive.google.com/file/d/1UGpOZiGGGBqQhGQxX7g19QAA-Dq9hPKk/view — salinan lokal
> ada di scratchpad session). Deadline: **11 Juli 2026** (konfirmasi jam persisnya di tab
> Event Schedule lablab.ai / Discord — tampil di timezone lokal).

## 0. Aturan resmi (VERIFIED — sumber: Participant Guide)

- **I/O**: container baca `/input/tasks.json` (`[{"task_id","prompt"}]`), tulis
  `/output/results.json` (`[{"task_id","answer"}]`), exit code 0. Malformed JSON = skor nol.
- **Env dari harness**: `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL` (SEMUA call Fireworks wajib
  lewat URL ini — bypass = 0 token tercatat tapi tak dinilai), `ALLOWED_MODELS` (dibaca
  runtime, JANGAN hardcode; pelanggaran = MODEL_VIOLATION).
- **Grading environment: 4 GB RAM, 2 vCPU, TANPA GPU.** Kutipan: "2B–3B 4-bit quantized
  models are safe; 7B 4-bit fills the full RAM budget."
- **Runtime maksimum: 10 menit total** untuk seluruh task set.
- **Scoring**: (1) accuracy gate via LLM-Judge (di bawah threshold = keluar leaderboard);
  (2) yang lolos di-ranking **ascending by total tokens** yang tercatat proxy. **Input DAN
  output token dihitung** (system prompt kita ikut dihitung).
- **Token lokal = 0.** `ZERO_API_CALLS` adalah strategi valid dan sah.
- 8 kategori: factual, math reasoning, sentiment, summarisation, NER, code debugging,
  logical/deductive reasoning, code generation. Evaluasi pakai **unseen prompt variants** —
  dilarang hardcode/cache jawaban.
- Docker image publik, linux/amd64, **compressed ≤ 10 GB**. Rate limit **10 submission/jam**
  — resubmission diperbolehkan, feedback status otomatis (PULL_ERROR / RUNTIME_ERROR /
  TIMEOUT / OUTPUT_MISSING / INVALID_RESULTS_SCHEMA / MODEL_VIOLATION / IMAGE_TOO_LARGE /
  ACCURACY_GATE_FAILED).
- Model Fireworks terlihat di forum AMD (indikatif, tetap baca env): minimax-m3,
  kimi-k2p7-code, gemma-4-31b-it, gemma-4-26b-a4b-it, gemma-4-31b-it-nvfp4.

## 1. Apa artinya bagi kode yang ada (kejujuran brutal)

| Aset lama | Status |
|---|---|
| FastAPI server (`app/main.py`, endpoint `/answer`) | **Tidak kompatibel dengan harness.** Harness menjalankan container batch: file-in → file-out → exit. Perlu entrypoint runner baru; logika router bisa direuse sebagai library. |
| qwen2.5:14b + gemma3:12b (Ollama) | **Mati.** 14b/12b tidak muat & tidak jalan di 4 GB / 2 vCPU CPU-only. Ganti model 1.5B–3B Q4 via llama.cpp. |
| Temuan cross-model disagreement (corr +0.487) | **Insight tetap valid** (sinyal internal buta pada confident-wrong), tapi implementasi 2 model resident berat di RAM+waktu. Dipakai selektif (lihat §3). |
| Draft-patch escalation | **Sebagian besar dibuang.** Input token DIHITUNG — mengirim draft menambah input. Untuk jawaban pendek, query minimal langsung lebih murah. |
| Kalibrator isotonic + τ1 | Konsep dipertahankan sebagai "escalation knob", tapi kalibrasi ulang WAJIB di model kecil + kalibrasi sesungguhnya = feedback leaderboard (§4). |
| Semantic cache | **Buang** (aturan melarang cache jawaban; risiko diskualifikasi > manfaat). |
| eval/gen_tasks.py + grader hybrid | **Reuse & perluas** ke 8 kategori resmi (tambah sentiment, NER, summarisation dgn length-constraint). |

## 2. Cara berpikir juri (leaderboard otomatis)

Juri = mesin. Tidak ada poin "arsitektur elegan". Fungsi menang persis dua langkah:
1. `accuracy ≥ threshold` (nilai threshold TIDAK dipublikasi → harus diprobe), lalu
2. `min(total_fireworks_tokens)` — 0 token adalah skor sempurna.

Konsekuensi strategis:
- **Leaderboard adalah oracle kalibrasi.** 10 submission/jam × ~2 hari = puluhan probe.
  Submit varian dengan tingkat eskalasi berbeda → `ACCURACY_GATE_FAILED` vs skor = data.
  Ini kalibrasi sesungguhnya, bukan eval lokal.
- **Skenario menang ideal = lolos gate dengan ~0 token.** Semua yang lolos gate dengan
  0 token kemungkinan seri di puncak → cadangan pembeda: pastikan lolos gate dengan margin.
- **Setiap token input dihemat = ranking.** System prompt minimal, instruksi "answer
  concisely", `max_tokens` ketat, tanpa few-shot kecuali terbukti perlu.
- **Reliabilitas = nilai.** TIMEOUT/OUTPUT_MISSING/INVALID_SCHEMA = nol. Wajib: progressive
  write + watchdog (§3.5), selalu jawab SEMUA task (jawaban tebakan > kosong, karena gate
  adalah rata-rata akurasi).

## 3. Arsitektur v2 (batch pipeline, CPU-first)

```
entrypoint.py (batch, TANPA server)
 1. Baca /input/tasks.json  → validasi
 2. KLASIFIKASI kategori per task (regex/keyword + fallback tiny-LLM) — murah, lokal
 3. Bagi antrean:
    a. REMOTE-bound (async, parallel, mulai DULUAN — jangan tunggu lokal)
    b. LOCAL-bound  (serial di llama.cpp, 2 vCPU)
 4. LOCAL solve dgn verifikasi deterministik GRATIS:
    - math      → LLM ekstrak ekspresi → eval Python; cocok → yakin; beda → escalate
    - code gen  → tulis fungsi → EKSEKUSI + self-test dari spek; lulus → yakin; gagal → escalate
    - code debug→ jalankan sebelum/sesudah; perbaikan terverifikasi → yakin
    - sentiment/NER/summarisation/factual → 3B biasanya cukup; format ketat
    - logical/deductive puzzle → ZONA JEBAKAN (temuan kalibrasi: model kecil confident-wrong,
      sinyal internal tak bisa deteksi) → default ESCALATE (atau cross-check bila budget ada)
 5. ESCALATE = prompt minimal ke model ALLOWED_MODELS tercerdas-per-kategori
    (BUKAN termurah-per-dolar — ranking dihitung per TOKEN, bukan dolar; yang penting
    akurasi tinggi + output pendek; hindari model yang menulis reasoning panjang)
 6. Tulis /output/results.json PROGRESIF (flush tiap task) + watchdog global ~8.5 menit
    → sisa task diisi jawaban terbaik-yang-ada → exit 0
```

Konfigurasi kunci: `ESCALATION_LEVEL` (0=zero-API … 3=agresif) — satu knob yang di-tune
lewat leaderboard, dipetakan ke: kategori mana yang boleh escalate + ambang keraguan lokal.

### 3.1 Model lokal (bundle di image, llama.cpp / llama-cpp-python)
- Primary: **Qwen2.5-3B-Instruct Q4_K_M** (~1.9 GB) — muat nyaman di 4 GB.
- Opsional (eksperimen E2): tambah **gemma-2-2b-it Q4** (~1.7 GB) untuk cross-check
  HANYA kategori rawan — keputusan berdasar ukur RAM+kecepatan nyata di limit
  `docker run --cpus=2 --memory=4g`, bukan asumsi. Kalau tok/s tak cukup utk 10 menit → buang.
- Ukur throughput: target seluruh task set selesai < 8 menit di 2 vCPU. Kalau tidak,
  turunkan ke Qwen2.5-1.5B utk kategori mudah.

### 3.2 Anggaran waktu adaptif
`per_task_budget = (510s − elapsed) / tasks_tersisa`; task lokal yang melewati budget →
potong (greedy, max_tokens kecil) atau escalate bila level mengizinkan.

## 4. Jadwal eksekusi (sisa ±48 jam)

**Hari ini (9 Jul, malam) — "BISA SUBMIT DULU, PINTAR BELAKANGAN"**
1. Entrypoint batch + Dockerfile baru (python:slim + llama.cpp + gguf 3B; jauh di bawah 10 GB).
2. Uji lokal PERSIS seperti harness: `docker run --cpus=2 --memory=4g -v input:/input -v
   output:/output -e FIREWORKS_*` pakai practice tasks dari guide.
3. Push ke GHCR (public) → **submit v0 malam ini juga** walau bodoh — memvalidasi
   PULL/RUNTIME/SCHEMA lebih awal, selagi slot murah. v0 = local-only (ZERO_API_CALLS)
   → sekaligus probe pertama: apakah 3B lokal lolos gate?
4. Perluas gen_tasks ke 8 kategori resmi (ikuti gaya practice tasks) — eval lokal cermin.

**Besok (10 Jul) — "KALIBRASI VIA LEADERBOARD"**
5. Implement verifikasi deterministik (math eval, code exec sandbox) → naikkan akurasi lokal gratis.
6. Ladder submission: v0 (zero) → kalau `ACCURACY_GATE_FAILED`, naikkan ESCALATION_LEVEL
   bertahap (mulai kategori logical + math-gagal-verify) → cari titik LOLOS gate dengan token minimum.
   Catat semua hasil di `submissions-log.md` (image tag, level, status, skor).
7. Setelah lolos gate: pangkas token — system prompt minimal, max_tokens per kategori,
   pilih model remote per kategori berdasarkan verbositas terukur (tes kecil via proxy).

**11 Jul (hari deadline) — "MARGIN & FREEZE"**
8. Pilih kandidat final = konfigurasi lolos-gate dgn token terendah + margin akurasi
   (jangan submit yang lolos mepet — evaluation set bisa bergeser dari probe kita).
9. Re-submit final ≥ 3 jam sebelum deadline (buffer utk PULL_ERROR mendadak). FREEZE.

## 5. Risiko utama & mitigasi
- **Threshold gate tak diketahui** → mitigasi: ladder probing (§4.6) + margin di pilihan final.
- **10 menit di 2 vCPU sangat ketat** → ukur dini (hari ini), model 1.5B fallback, remote
  async paralel, watchdog progressive-write.
- **LLM-Judge menilai "expected intent"** → jawaban ringkas TAPI lengkap; patuhi constraint
  eksplisit (mis. "exactly one sentence"); jangan over-terse sampai kehilangan poin judge.
- **Jumlah task hidden tak diketahui** → semua budget dihitung dinamis dari len(tasks).
- **Fireworks error/latency saat scoring** → retry + backoff singkat, timeout per call,
  fallback ke jawaban lokal (jawaban lokal salah masih lebih baik dari kosong).
- **Blind-spot universal (strawberry-class)** → justru INI selesai dgn tool lokal:
  hitung huruf/kata pakai Python, bukan LLM. Deteksi pola task counting → jalur deterministik.

## 6. Definisi selesai (checklist submit final)
- [ ] Image publik, linux/amd64, < 10 GB compressed, tag eksplisit
- [ ] Baca env harness murni dari environment (tanpa .env di image)
- [ ] Practice tasks → results.json valid schema, exit 0, < 10 menit @ 2 vCPU/4 GB
- [ ] Semua task_id terjawab (tak ada yang kosong) dalam segala kondisi (watchdog teruji
      dengan simulasi hang Fireworks)
- [ ] Hanya model dari ALLOWED_MODELS runtime yang dipanggil, semua via FIREWORKS_BASE_URL
- [ ] submissions-log.md terisi; konfigurasi final = lolos gate + token minimum + margin
