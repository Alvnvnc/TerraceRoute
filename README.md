# TerraceRoute — Token-Efficient Routing Agent (AMD ACT II, Track 1)

Agent **batch** yang menyelesaikan task natural-language dengan biaya token Fireworks
**seminimal mungkin** (lokal = gratis), tanpa jatuh di bawah accuracy gate.

> Aturan resmi terverifikasi ada di [`participant-guide.pdf`](participant-guide.pdf).
> Rencana & strategi: [`plan-v2.md`](plan-v2.md). Log kalibrasi: [`submissions-log.md`](submissions-log.md).

## Kontrak harness (yang dinilai)

- Baca `/input/tasks.json` `[{task_id,prompt}]` → tulis `/output/results.json` `[{task_id,answer}]`, exit 0.
- Env diinjeksi: `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL` (SEMUA call remote lewat sini),
  `ALLOWED_MODELS` (dibaca runtime).
- Scoring env: **4 GB RAM, 2 vCPU, CPU-only, 10 menit**. Token lokal = 0; ranking = total token
  Fireworks (input+output) menaik. Accuracy gate via LLM-Judge; unseen prompt variants.

## Cara kerja (terrace routing)

Tiap task turun "teras" cek yang makin mahal, keluar sedini mungkin:

```
klasifikasi kategori (regex, gratis)
  ├─ counting huruf ("berapa 'r' di strawberry")  → hitung Python (DETERMINISTIK, 0 token)
  ├─ math    → solve lokal + Python hitung-ulang; word-problem = zona jebakan
  ├─ code    → solve lokal + eksekusi cek syntax (verifikasi INDEPENDEN)
  └─ lainnya → solve lokal (3B)
        ↓ kebijakan eskalasi (knob ESCALATION_LEVEL)
   verify-fail / lokal-kosong → escalate (lvl≥1)
   kategori jebakan (math,logical) → escalate (lvl≥2)
   apa pun yg tak terverifikasi   → escalate (lvl≥3)
        ↓
   remote: prompt TERSE (jawaban final saja) + cap max_tokens per kategori → hemat token
```

**Insight kunci** (dari kalibrasi sebelumnya): sinyal keyakinan internal model kecil
(perplexity, self-consistency, "python buatan model sendiri") **anti-korelasi** dengan
kebenaran — model *confidently-consistent wrong*. Maka yang dihitung "verified" hanyalah cek
**benar-benar independen** (eksekusi kode, hitung huruf), bukan model menyetujui dirinya.
Selebihnya di-escalate sesuai anggaran gate.

## Struktur

```
agent/
  run.py        # entrypoint batch: I/O, watchdog, progressive atomic write, ringkas token
  solve.py      # terrace per-kategori + kebijakan eskalasi terpadu
  classify.py   # klasifikasi 8 kategori (regex, gratis)
  verify.py     # verifikasi deterministik: aritmetika, eksekusi Python, counting
  local_llm.py  # model lokal: ollama (dev) / llama-cpp in-process (prod)
  fireworks.py  # escalation client (baca ALLOWED_MODELS runtime, semua via BASE_URL)
  config.py     # semua dari env; http.py = urllib (tanpa dependency jaringan)
Dockerfile.agent          # multi-stage, CPU-only, ~1.84 GB compressed
scripts/build_and_push.sh # build linux/amd64 + push registri publik
eval/practice_tasks.json  # 9 practice tasks dari guide (bukan set penilaian)
```

## Menjalankan

### Dev (host Ollama, cepat)
```bash
ollama pull qwen2.5:3b-instruct
INPUT_PATH=eval/practice_tasks.json OUTPUT_PATH=/tmp/results.json \
LOCAL_BACKEND=ollama LOCAL_MODEL=qwen2.5:3b-instruct ESCALATION_LEVEL=0 \
python3 -m agent.run
```

### Container (persis seperti scoring)
```bash
docker build -f Dockerfile.agent -t terraceroute:test .
mkdir -p in out && cp eval/practice_tasks.json in/tasks.json
docker run --rm --cpus=2 --memory=4g -v $PWD/in:/input:ro -v $PWD/out:/output terraceroute:test
```
Escalation live: tambah `-e FIREWORKS_API_KEY=... -e FIREWORKS_BASE_URL=... -e ALLOWED_MODELS=...`
dan `-e ESCALATION_LEVEL=2`.

### Submit
```bash
REGISTRY=ghcr.io/USERNAME ./scripts/build_and_push.sh v0     # push publik
# paste ghcr.io/USERNAME/terraceroute:v0 ke form lablab.ai
```
Lalu kalibrasi lewat leaderboard — lihat [`submissions-log.md`](submissions-log.md).

## Status tervalidasi (2026-07-10)

- Container build & jalan di **2 vCPU / 4 GB CPU-only**; schema selalu valid; exit 0.
- Zero-API (level 0), escalation (level 2), input malformed (→ `[]`, exit 0) semua teruji.
- Counting deterministik (strawberry→3), math extraction, terse remote (134 tok/2 escalate) OK.
- **Catatan waktu:** ~7–8 dtk/task CPU-only → set besar (>~60 task) bisa kena watchdog 510s;
  lever: kecilkan `LOCAL_MAX_TOKENS`, model lebih kecil utk kategori mudah, atau escalate lebih.
- **Belum:** `ALLOWED_MODELS` resmi launch-day (baca runtime) & kalibrasi threshold via leaderboard.
