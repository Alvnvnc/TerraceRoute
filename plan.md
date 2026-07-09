Kita sebaiknya buat project ini sebagai **Track 1: Hybrid Token-Efficient Routing Agent**, bukan “benchmark GPU/CPU”. Track 1 memang meminta agent yang me-routing task antara model lokal dan remote Fireworks untuk menekan token, dan semua submission wajib **containerized**. ([LabLab][1])

## Nama project

**TerraceRoute**
Tagline:

> A token-efficient hybrid routing agent that solves easy tasks locally and escalates only difficult requests to Fireworks AI.

## Strategi utama

Target kita bukan bikin model paling besar. Target kita bikin **router yang pintar menghindari panggilan remote**.

Karena di Track 1, nilai utamanya adalah:

```text id="4xv3zn"
akurasi tetap lolos
+
remote token serendah mungkin
```

Jadi strategi kita:

```text id="gg5o4d"
Easy task      → jawab lokal pakai Ollama/Qwen
Medium task    → jawab lokal + self-check
Hard task      → compress prompt → Fireworks AI
Uncertain task → Fireworks AI
```

Ini cocok dengan aturan hackathon karena ACT II memakai AMD Developer Cloud, ROCm, dan Fireworks AI API credits, dan semua submission harus dalam container. ([LabLab][1])

---

# MVP kita

MVP-nya jangan terlalu besar. MVP yang cukup untuk submit:

## MVP: **Local-first routing agent**

Input user masuk, sistem memutuskan:

```text id="5i76v2"
LOCAL_ONLY
REMOTE_ONLY
LOCAL_THEN_REMOTE
```

Lalu sistem mencatat:

```text id="3engpn"
- route decision
- confidence
- local model used
- remote model used or skipped
- estimated local tokens
- estimated remote tokens
- latency
- final answer
```

## Komponen MVP

```text id="fc9zrt"
1. FastAPI server
2. Ollama local backend
3. Fireworks remote backend
4. Router classifier
5. Prompt compressor
6. Token/latency logger
7. Simple evaluation script
8. Dockerfile + docker-compose
9. README + demo video
```

---

# Model yang kita pakai

Untuk development di PC Anda:

```text id="hhxzhr"
Local strong model:
qwen2.5:14b-instruct

Optional local comparator:
gemma3:12b

Embedding, kalau butuh:
bge-m3:latest
```

Tapi untuk submission yang lebih aman:

```text id="edtvge"
Default portable model:
qwen2.5:3b-instruct-q4_K_M

Strong local mode:
qwen2.5:14b-instruct

Remote:
Fireworks AI
```

Kenapa begitu?

Karena PC Anda kuat, tapi environment final/leaderboard belum tentu punya resource sebesar Quadro RTX 8000 Anda. Jadi di demo lokal boleh pakai `qwen2.5:14b-instruct`, tapi container harus tetap bisa jalan dengan model kecil.

---

# Arsitektur

```text id="1qslrq"
User Request
    ↓
Input Cleaner
    ↓
Router Classifier
    ↓
┌───────────────────────────────┐
│ EASY                          │
│ → Ollama Qwen local answer    │
└───────────────────────────────┘

┌───────────────────────────────┐
│ MEDIUM                        │
│ → Ollama answer               │
│ → local self-check            │
│ → if low confidence, remote   │
└───────────────────────────────┘

┌───────────────────────────────┐
│ HARD                          │
│ → prompt compressor           │
│ → Fireworks AI                │
└───────────────────────────────┘

    ↓
Final Answer
    ↓
Metrics Logger
```

---

# Folder repo

```text id="b08k5l"
terrace-route/
  app/
    main.py
    router.py
    ollama_client.py
    fireworks_client.py
    compressor.py
    evaluator.py
    tokenizer.py
    schemas.py
    metrics.py

  eval/
    sample_tasks.jsonl
    run_eval.py

  scripts/
    start_local.sh
    test_router.sh

  Dockerfile
  docker-compose.yml
  requirements.txt
  .env.example
  README.md
```

---

# Logic routing MVP

Jangan pakai LLM untuk semua keputusan. Kita buat hybrid:

## Step 1: rule-based pre-router

Langsung lokal kalau request sederhana:

```text id="etk5kq"
- definisi sederhana
- ringkasan pendek
- rewrite kalimat
- klasifikasi mudah
- pertanyaan umum
- format JSON sederhana
```

Langsung remote kalau request sulit:

```text id="3whpze"
- matematika kompleks
- coding kompleks
- multi-step reasoning
- legal/medical/financial
- butuh akurasi tinggi
- konteks panjang
- instruksi ambigu
```

## Step 2: local classifier

Kalau rule belum yakin, tanya Qwen lokal:

```text id="7x88aw"
Classify this task as EASY, MEDIUM, or HARD.
Return JSON only:
{
  "difficulty": "...",
  "confidence": 0.0,
  "reason": "..."
}
```

## Step 3: threshold

```text id="bq9328"
EASY + confidence >= 0.75
→ local answer

MEDIUM + confidence >= 0.65
→ local answer + self-check

HARD or confidence < 0.65
→ compress + Fireworks
```

---

# Prompt compressor

Ini selling point penting.

Sebelum call Fireworks, jangan kirim prompt mentah. Kirim versi ringkas:

```text id="gf4mmy"
Original user request:
...

Compress this into the shortest instruction that preserves all requirements.
Remove greetings, repetition, irrelevant context, and noise.
Return only the compressed task.
```

Contoh:

Input panjang:

```text id="iamx0y"
halo kak, saya sedang bikin project hackathon, bisa bantu jelaskan secara detail dan mungkin kasih contoh apa itu docker, saya masih pemula dan ingin tahu kenapa dipakai dalam deployment?
```

Compressed remote prompt:

```text id="3yxmm8"
Explain Docker for beginner deployment use. Include why it is used and one simple example.
```

Ini langsung menurunkan token remote.

---

# Endpoint MVP

Minimal cukup punya 3 endpoint:

```text id="2yhjsa"
POST /route
POST /answer
GET /metrics
```

Contoh request:

```json id="h9j36x"
{
  "input": "Explain Docker in two sentences.",
  "mode": "auto"
}
```

Contoh response:

```json id="htnj4c"
{
  "route": "LOCAL_ONLY",
  "difficulty": "EASY",
  "confidence": 0.86,
  "local_model": "qwen2.5:14b-instruct",
  "remote_used": false,
  "estimated_remote_tokens": 0,
  "latency_ms": 1240,
  "answer": "Docker is..."
}
```

---

# Evaluation sederhana

Kita buat dataset kecil sendiri dulu:

```text id="mtnx93"
eval/sample_tasks.jsonl
```

Isi 30–50 task:

```json id="aaz8cs"
{"id":1,"input":"Explain Docker in two sentences.","expected_route":"LOCAL_ONLY"}
{"id":2,"input":"Write a secure FastAPI JWT middleware with refresh token rotation.","expected_route":"REMOTE_ONLY"}
{"id":3,"input":"Summarize this log and identify likely cause: ...","expected_route":"LOCAL_THEN_REMOTE"}
```

Metric internal:

```text id="yr1c19"
- local hit rate
- remote call rate
- estimated remote tokens
- average latency
- failed JSON rate
- route accuracy
```

Target MVP:

```text id="fki3ks"
remote call rate < 40%
JSON parsing success > 90%
router latency < 3s
semua endpoint jalan via Docker
```

---

# Rencana kerja

## Sekarang–5 Juli 2026: build MVP lokal

Fokus:

```text id="811yu7"
- FastAPI app
- Ollama client
- Fireworks client dummy/real
- router.py
- metrics logger
- Dockerfile
```

Gunakan Quadro RTX 8000 + Ollama Anda sekarang. Tidak usah setup ROCm dulu.

Command kondisi model:

```bash id="3du27h"
ollama stop gemma3:12b
ollama run qwen2.5:14b-instruct
```

## 6 Juli 2026: launch day hackathon

Begitu detail final keluar, cek:

```text id="1zb89k"
- model Fireworks yang boleh dipakai
- format input/output leaderboard
- scoring token
- container requirements
```

Lalu sesuaikan adapter, bukan rewrite total.

## 7–9 Juli 2026: optimasi token

Fokus ke hal yang bikin skor naik:

```text id="pe1xik"
- improve routing threshold
- prompt compression
- cache repeated requests
- local self-check
- fallback only when necessary
```

## 10 Juli 2026: polish submission

Siapkan:

```text id="vbrr65"
- README
- demo video ≤5 menit
- diagram arsitektur
- contoh token saving
- Docker run command
- Devpost/lablab description
```

## 11 Juli 2026: final submit

Submission window ACT II live page menunjukkan build dimulai 6 Juli 2026 dan kredit AMD Developer Cloud disediakan untuk peserta. ([LabLab][2]) Jangan menunggu setup sempurna; submit versi stabil lebih baik daripada versi besar tapi error.

---

# Fitur yang jangan dibuat dulu

Jangan dulu:

```text id="f5evw3"
- fine-tuning
- multi-agent kompleks
- UI dashboard besar
- RAG besar
- ROCm setup rumit
- training model sendiri
- benchmark panjang
```

Itu semua bisa makan waktu. Untuk Track 1, yang paling penting adalah router.

---

# Demo story

Demo video kita harus sederhana:

## Scene 1: Easy task

Input:

```text id="o88plm"
Explain Docker in two sentences.
```

Output:

```text id="sne4yn"
Route: LOCAL_ONLY
Remote tokens: 0
```

## Scene 2: Medium task

Input:

```text id="wwqqhb"
Analyze this backend error log and suggest likely cause.
```

Output:

```text id="eio6y4"
Route: LOCAL_THEN_VERIFY
Remote tokens: 0 or small
```

## Scene 3: Hard task

Input:

```text id="7tgvpk"
Design secure multi-tenant RBAC architecture with dynamic permissions and audit logs.
```

Output:

```text id="vym3xu"
Route: REMOTE_AFTER_COMPRESSION
Original tokens: 900
Compressed tokens: 220
```

Ini menunjukkan nilai project dengan jelas.

---

# Pitch singkat

Gunakan ini untuk deskripsi:

```text id="gzlmpu"
TerraceRoute is a hybrid token-efficient routing agent that reduces expensive remote model usage by solving easy tasks locally, verifying medium tasks with a local model, and escalating only difficult requests to Fireworks AI after prompt compression. It tracks routing decisions, latency, and remote token usage in a containerized FastAPI service.
```

Versi Indonesia:

```text id="49o5mi"
TerraceRoute adalah agent routing hemat token yang menyelesaikan task mudah secara lokal, memverifikasi task sedang dengan model lokal, dan hanya mengeskalasi task sulit ke Fireworks AI setelah prompt dikompresi. Sistem ini mencatat keputusan routing, latency, dan estimasi token remote dalam service FastAPI yang containerized.
```

---

# Kesimpulan strategi kita

MVP final:

```text id="8pxpql"
FastAPI + Ollama Qwen + Fireworks fallback + routing classifier + prompt compressor + token logger + Docker
```

Model untuk development:

```text id="d2eag0"
qwen2.5:14b-instruct
```

Model portable untuk submission:

```text id="rwe9tq"
qwen2.5:3b-instruct-q4_K_M
```

Yang harus kita kejar bukan “model paling besar”, tapi:

> **remote token serendah mungkin, akurasi tetap aman, container bisa jalan tanpa drama.**

[1]: https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii?utm_source=chatgpt.com "AMD Developer Hackathon: ACT II"
[2]: https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii/live?utm_source=chatgpt.com "AMD Developer Hackathon: ACT II"
