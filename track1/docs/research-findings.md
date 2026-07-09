# Research Findings — TerraceRoute / AMD ACT II Track 1

> Status verifikasi: **UNVERIFIED**. Fase search + ekstraksi klaim (deep-research) berhasil
> (23 sumber, 110 klaim, 25 klaim kunci). Fase verifikasi adversarial 3-vote **gagal total**
> karena session limit akun, bukan karena klaim ditolak. Artinya: perlakukan temuan di bawah
> sebagai **lead berkualitas tinggi yang belum di-cross-check**, terutama aturan hackathon.
> Tanggal riset: 2026-07-03.

---

## 0. Peringatan paling penting dulu

Beberapa klaim "aturan hackathon" datang dari extractor yang membaca halaman lablab.ai. Klaim-klaim
ini **saling agak bertentangan** dan belum diverifikasi, jadi WAJIB dikonfirmasi manual dari halaman
resmi / Discord saat kickoff (6 Juli 2026, 15:00 UTC):

- **Lingkungan scoring mungkin CPU-only, no GPU** (1 klaim, `.../act-ii/live`). Kalau benar, rencana
  "local model utama = qwen2.5:14b via ROCm/GPU" **salah untuk environment yang dinilai** — model
  lokal harus viable di CPU. Ini bertabrakan dengan asumsi di `plan.md`/`review.md`.
- **Scoring = threshold-constrained cost minimization**, BUKAN weighted tradeoff bebas. Formulasinya
  kemungkinan: *"minimalkan remote token, dengan syarat akurasi ≥ ambang α"* — bukan
  `min Loss + λ·tokens`. Ini mengubah objective di `formula.md`.
- **Local compute = gratis untuk scoring.** Hanya remote (Fireworks) token yang dihitung. Jadi
  drafting lokal, verifikasi lokal, kompresi lokal, caching → **zero penalty**. Ini insentif kuat
  untuk memindahkan sebanyak mungkin kerja ke sisi lokal.
- Halaman lablab.ai **tidak menyebut Ollama / qwen2.5** secara eksplisit, dan Event Schedule masih
  "To be announced". Model remote mungkin dibatasi ke ~2 model AMD-hardware di Fireworks.
- Prize pool (klaim): $10k ($5k/$3k/$2k). Build window 5 hari: kickoff Sen 6 Juli 15:00 UTC →
  deadline Sab 11 Juli 15:00 UTC. Semua submission containerized.

**Aksi:** jangan hard-commit ke GPU 14B sebelum konfirmasi environment. Bangun agar model lokal
bisa di-swap (env var) antara model kecil CPU-friendly dan model besar GPU.

---

## 1. State-of-the-art (2024–2026) yang relevan

**Routing & cascade**
- RouteLLM (2406.18665): router preference-trained hemat biaya s/d ~3.66×; generalisasi zero-shot ke
  pasangan model baru tanpa retrain. → Router open-source bisa dipakai ulang untuk pasangan
  qwen-lokal vs Fireworks-remote.
- MixLLM: 97.25% kualitas GPT-4 dengan 24.18% biaya. R2-Reasoner: hemat 84.46% API cost.
- AutoMix: cascade black-box dgn self-verification skor pada output model kecil untuk decide escalate.

**Uncertainty / calibration (ini kunci pembeda)**
- **UCCI (2605.18796): kalibrasi adalah pembedanya.** Memetakan uncertainty token-level → probabilitas
  error per-query via **isotonic regression** memangkas biaya 31% pada micro-F1 0.91, dan menurunkan
  expected calibration error 0.12 → 0.03. **Threshold sederhana pada error-prob terkalibrasi terbukti
  cost-optimal** untuk cascade small-vs-large (di 3 asumsi eksplisit). Artinya: **tidak butuh router
  kompleks — satu threshold terkalibrasi cukup, dan itu mengalahkan** entropy-thresholding mentah
  (biaya 11% lebih tinggi), conformal (+5%), dan cascade ala FrugalGPT (+8%).
- **Probe-based / perplexity uncertainty >> verbalized self-confidence** (2503.10657 / RouterEval
  area). Small model menyamai LLM pada ~20% query paling percaya-diri. → **Minta Qwen "rate confidence
  0–1" (seperti di plan.md) adalah pendekatan LEMAH.** Pakai logprob/perplexity atau probe terlatih.

**Prompt compression**
- LLMLingua: s/d 20× kompresi, hanya ~1.5 poin turun di GSM8K/BBH pada kompresi maksimal.
- LLMLingua-2 (2403.12968): kompresi 2–5× via **binary token classification** pakai encoder kecil
  (XLM-RoBERTa-large 355M atau mBERT 110M) — **ringan, jalan lokal**. Retensi akurasi ~99% di
  MeetingBank QA (86.92 vs 87.75 EM).
- **CAVEAT PENTING:** kompresi agresif (r=0.3) bisa memicu **output-token EXPANSION** (mis. ~56×
  di MBPP untuk DeepSeek-Chat) — net token bisa NAIK, tergantung benchmark & provider. Compress
  ≠ selalu hemat.

**Speculative / draft-verify hybrid**
- Response-level speculative via proxy layer (2606.22840): draft model murah bikin full response,
  verify model mahal accept/enhance/di-skip oleh complexity router. **Hemat 45.8% biaya vs Claude
  Opus, draft-use rate 88.8%**, tanpa akses logit / shared vocab. **Latency 1.83× lebih cepat DAN
  kualitas naik (100% vs 95% pass)** karena path "skip" (cheap-only) mendominasi. → Validasi kuat
  untuk **local-draft-remote-patch**.

**Semantic caching**
- Hit-rate produksi nyata **20–45%**, bukan 95%. Angka "95%" itu *match accuracy* (kebenaran hit),
  bukan fraksi query yang hit. GPTCache adalah implementasi standar.

---

## 2. Yang sudah GENERIK (bukan pembeda)

- **Router "kalau sulit → remote, kalau mudah → lokal"** biasa. Paper evaluasi adil (2602.11877):
  banyak router modern **gagal reliably mengalahkan baseline sederhana**; kebanyakan metode collapse
  ke performa mirip.
- **Verbalized self-confidence** ("Classify EASY/MEDIUM/HARD, confidence 0.0") — persis yang ada di
  plan.md Step 2. Ini sinyal lemah & tidak terkalibrasi.
- **Weighted-sum risk score heuristik** (`R(x)=0.30D+0.25U+0.15L+...` di formula.md) — terlihat ilmiah
  tapi bobotnya asal; ini justru kategori "router yang sering tak kalahkan baseline".
- **Entropy/logprob thresholding mentah** tanpa kalibrasi — baseline generik yang dikalahkan UCCI.
- **Semantic cache fuzzy** yang dijual "95% hit" — over-sold; risiko wrong-hit malah menurunkan
  akurasi (bahaya di scoring yang punya accuracy floor).

**Peringatan tuning:** ranking router **sangat sensitif terhadap threshold** di rentang call-rate
20–40% — tuning ke satu operating point bisa tidak generalisasi. Jangan overfit threshold ke eval set
kecilmu.

---

## 3. Evaluasi 5 kandidat ide (grounded)

| Ide | Verdict | Catatan |
|---|---|---|
| **Risk-calibrated router** | ✅ **Andalan — TAPI ubah caranya** | Jangan weighted-sum heuristik. Kalibrasi sinyal uncertainty murah (perplexity/logprob **atau** self-consistency agreement) → isotonic/Platt → P(local salah). Threshold tunggal terkalibrasi, di-set tepat di accuracy floor. Ini yang terbukti cost-optimal & mengalahkan baseline. |
| **Token-ROI routing** | ⚠️ Sekunder | Elegan untuk narasi, tapi kalau scoring = threshold-constrained (bukan weighted), ROI/λ jadi kurang pas. Pakai sebagai tie-breaker/framing, bukan mesin keputusan utama. |
| **Compress-before-remote** | ✅ dengan rem | LLMLingua-2 encoder kecil lokal = gratis & ~99% retensi. TAPI wajib guard: ukur token sebelum/sesudah, **batalkan kompresi kalau output malah membengkak** (fenomena expansion). |
| **Local-draft-remote-patch** | ✅✅ **Pembeda terkuat & terbukti** | 2606.22840: 45.8% hemat, kualitas naik, tanpa logit. Kunci: remote dipaksa hanya emit **patch/diff/verdict pendek**, meng-cap remote OUTPUT token (bagian mahal). |
| **Similarity cache (embeddings)** | ⚠️ Nice-to-have, hati-hati | Hit nyata 20–45%. Untuk env dgn accuracy floor, pakai **threshold sim tinggi (mis. >0.95) atau exact-match** agar tak wrong-hit. Prioritas rendah. |

---

## 4. Rekomendasi terobosan (realistis 5 hari, 1 dev)

**"Calibrated Risk Router + Budgeted Draft-Patch Escalation"** — gabungan yang grounded & pembeda:

1. **Sinyal lokal murah, bukan self-report.** Ambil uncertainty dari logprob/perplexity Qwen
   (butuh Ollama **v0.12.11+** — versi lama TIDAK expose logprobs; ini gotcha nyata dari tracker
   Ollama #2415). Kalau logprobs bermasalah, fallback ke **self-consistency**: sample 3 jawaban
   pendek, ukur agreement.
2. **Kalibrator kecil.** Saat kickoff, ambil sampel task berlabel, fit isotonic/Platt:
   uncertainty → P(local salah). Simpan sebagai artefak.
3. **Threshold di accuracy floor.** Escalate ke remote hanya jika P(local salah) melebihi ambang yang
   dipilih supaya akurasi agregat pas di atas α. (Objective sebenarnya: min remote token s.t. acc≥α.)
4. **Escalation = draft-patch, bukan full re-answer.** Kirim task terkompresi + draft lokal ke
   Fireworks, minta *"perbaiki hanya jika salah, kembalikan koreksi minimal"* → cap remote output.
5. **Guardrails:** kompresi dibatalkan jika bikin token naik; cache hanya high-precision.
6. **Dashboard `/metrics`:** tampilkan remote-call rate, remote tokens, tokens saved, acc — bukti
   "ekonomi token" untuk demo.

**Novelty claim aman:** "Calibration-first hybrid router: memetakan uncertainty lokal ke probabilitas
error terkalibrasi untuk keputusan escalate hemat-token, dengan remote dibatasi ke mode patch."

---

## Sumber utama
- RouteLLM 2406.18665 · MixLLM/R2/BEST-Route/AutoMix survey 2603.04445 · **UCCI kalibrasi 2605.18796**
- LLMLingua-2 2403.12968 · **Draft-verify proxy 2606.22840** · Uncertainty routing 2503.10657
- Fair router eval 2602.11877 · Ollama logprobs issue #2415 · GPTCache · lablab.ai act-ii (+/live)
