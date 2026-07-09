# TerraceRoute — Calibration-First Hybrid Router (Formula & Metode)

> Revisi 2026-07-03. Menggantikan pendekatan **weighted-sum risk score** yang lama.
> Alasan: literatur terbaru (UCCI 2605.18796, fair-router-eval 2602.11877, uncertainty-routing
> 2503.10657) menunjukkan weighted-sum heuristik & verbalized self-confidence adalah **baseline
> generik yang sering tidak mengalahkan baseline sederhana**. Pembeda sesungguhnya = **kalibrasi**.
> Lihat `research-findings.md` untuk sumber & status verifikasi (klaim belum di-cross-check).

---

## 0. Objective yang benar

Scoring Track 1 (klaim, belum terverifikasi) = **threshold-constrained cost minimization**, bukan
weighted tradeoff. Local compute = gratis; hanya remote (Fireworks) token yang dihitung.

```math
\min_{\pi}\; \mathbb{E}_{x\sim\mathcal{D}}\big[\,T_{remote}(x,\pi)\,\big]
\quad\text{s.t.}\quad
\mathbb{E}_{x\sim\mathcal{D}}\big[\,\mathrm{Acc}(y_\pi(x),y^*(x))\,\big] \ge \alpha
```

Konsekuensi desain:
- Pindahkan **sebanyak mungkin** kerja ke lokal (gratis). Remote hanya saat perlu untuk jaga akurasi.
- Keputusan intinya **satu pertanyaan**: *"Apakah jawaban lokal cukup akurat, atau harus escalate?"*
  Bukan skor kesulitan yang mewah.

---

## 1. Inti: probabilitas error lokal yang TERKALIBRASI

Alih-alih skor risiko heuristik `R(x)=0.30D+0.25U+...`, kita estimasi **satu besaran yang punya
makna probabilistik**:

```math
p_{\text{wrong}}(x) \;=\; \Pr[\,y_{\text{local}}(x)\neq y^*(x)\,]
```

### 1a. Sinyal uncertainty mentah (murah, lokal)

Pilih salah satu (bukan verbalized self-confidence — itu lemah):

**Opsi A — Perplexity / logprob** (butuh Ollama **v0.12.11+**; versi lama tidak expose logprobs):

```math
\bar{\ell}(x) = \frac{1}{n}\sum_{i=1}^{n}\log p(t_i \mid t_{<i}),
\qquad
u_{\text{ppl}}(x) = 1 - e^{\,\bar{\ell}(x)} \in [0,1]
```

`u` tinggi = model ragu (logprob rata-rata rendah).

**Opsi B — Self-consistency** (fallback jika logprobs tidak tersedia): sample `k` jawaban pendek,
ukur ketidaksepakatan.

```math
u_{\text{sc}}(x) = 1 - \frac{\text{ukuran cluster mayoritas}}{k}
```

### 1b. Kalibrasi (INI pembedanya)

Petakan sinyal mentah `u` → probabilitas error nyata via **isotonic regression** `g`, di-fit pada
set berlabel kecil yang dikumpulkan saat kickoff:

```math
p_{\text{wrong}}(x) = g\big(u(x)\big),\qquad
g = \arg\min_{h\ \text{monoton}} \sum_{i}\big(h(u_i) - \mathbb{1}[\text{local salah}_i]\big)^2
```

Efek: Expected Calibration Error turun drastis (di UCCI 0.12 → 0.03), sehingga threshold pada
`p_wrong` benar-benar berarti "sekian persen salah", bukan angka arbitrer.

### 1c. TEMUAN EMPIRIS (2026-07-04) — sinyal internal GAGAL, cross-model yang bekerja

Kalibrasi nyata pada eval set sulit (30 task, grader hybrid exact-match + judge independen
gemma3:12b; qwen2.5:14b gagal ~20–23%) mengungkap: **sinyal uncertainty internal anti-korelasi
dengan error.** Korelasi point-biserial `corr(u, wrong)` (makin + makin baik):

| sinyal | corr | catatan |
|---|---|---|
| perplexity (`u_ppl`) | **−0.09 … −0.16** | model **confidently wrong** pada jebakan penalaran |
| self-consistency (`u_sc`) | **−0.12 … −0.21** | error **konsisten** (salah sama tiap sampel) → deteksi variansi tak menangkap bias |
| **cross-model (`u_cross`)** | **+0.03 … +0.30** | satu-satunya yang non-negatif; verifier beda-keluarga |

**Sebab:** perplexity & self-consistency memakai model yang **sama**; ketika model salah dengan
yakin DAN konsisten (mis. hitung huruf, jebakan logika), tak ada sinyal internal. Yang menangkapnya
adalah **ketidaksepakatan dengan model keluarga berbeda** (gemma3:12b) — arsitektur berbeda membuat
kesalahan berbeda. Contoh: qwen bilang 3 killers→4, snail→7 (salah); gemma benar → tidak setuju → escalate.

**Detail penting yang terbukti dari eksperimen:**
- **Verifier harus KUAT & beda keluarga.** Menambah qwen2.5:3b (lemah, sekeluarga) *menurunkan* corr
  (+0.30 → +0.12): ia ikut tak setuju walau primary benar (noise). Pakai gemma3:12b saja.
- **Verifier harus VERBOSE (bernalar).** Memaksa jawaban terse merusak akurasi verifier & menghapus
  sinyal (+0.30 → +0.03). Bandingkan jawaban penuh, jangan paksa singkat.
- **Blind-spot universal tetap ada:** task di mana SEMUA model salah-dan-setuju (mis. hitung 'r' pada
  "strawberry") tak bisa diroute oleh disagreement — butuh tool/eksekusi kode.
- **n=30 terlalu kecil:** estimasi corr sangat bervariasi antar-run + label bergeser (nondeterminisme
  temp=0 GPU). Threshold dari data ini **provisional**; butuh eval set jauh lebih besar (100–300) untuk
  kalibrasi tepercaya.

**Konsekuensi arsitektur (arah pemenang):** sinyal routing utama = `cross_model_uncertainty` =
ketidaksepakatan antar-model (`self_consistency_uncertainty` diterapkan pada jawaban antar-model).
Dengan 1 verifier → sinyal biner: **setuju → LOCAL (gratis); tak setuju → escalate.** Bonus: banyak
kasus "qwen salah" = "gemma benar" → bisa dikoreksi **gratis** oleh ensemble lokal sebelum bayar remote.

---

## 2. Kebijakan routing: satu threshold + dua lane escalate

Karena "threshold pada error-prob terkalibrasi terbukti **cost-optimal** untuk cascade small-vs-large"
(UCCI), kita TIDAK butuh router kompleks. Cukup:

```math
\text{route}(x)=
\begin{cases}
\textbf{LOCAL\_ONLY}, & p_{\text{wrong}}(x) < \tau_1\\[4pt]
\textbf{LOCAL\_DRAFT + REMOTE\_PATCH}, & \tau_1 \le p_{\text{wrong}}(x) < \tau_2\\[4pt]
\textbf{REMOTE\_COMPRESSED (full)}, & p_{\text{wrong}}(x) \ge \tau_2
\end{cases}
```

`τ₁` dipilih sebagai ambang terbesar yang masih menjaga akurasi agregat ≥ α (lihat §5).
`τ₂` memisahkan "koreksi kecil cukup" vs "butuh jawaban remote penuh".

Rule keras tambahan (high-risk domain): jika flag `H(x)=1` (legal/medis/finansial/keamanan) dan
`p_wrong ≥ τ₁·0.5`, langsung escalate — asimetri severity.

---

## 3. Escalation hemat: Draft-Patch, bukan re-answer

Bukti (2606.22840): draft lokal + remote yang hanya *memperbaiki* → hemat 45.8% biaya, kualitas
malah naik, tanpa akses logit. Kuncinya **membatasi remote OUTPUT token** (bagian termahal).

Prompt patch ke Fireworks:

```text
Task (compressed): {x'}
Draft answer from a local model: {draft}
If the draft is correct, reply exactly: OK
Otherwise return ONLY the minimal corrected answer. Do not explain.
```

Token remote pada mode patch:

```math
T_{remote}^{patch} = T_{in}(x') + T_{in}(\text{draft}) + T_{out}^{patch},
\qquad T_{out}^{patch} \ll T_{out}^{full}
```

Pakai patch-mode hanya jika diperkirakan lebih murah dari full-remote:

```math
T_{in}(x') + T_{in}(\text{draft}) + \widehat{T_{out}^{patch}} \;<\; T_{in}(x') + \widehat{T_{out}^{full}}
```

---

## 4. Compress-before-remote (dengan REM keselamatan)

Encoder kecil (LLMLingua-2, mBERT 110M) atau prompt-based compression lokal (gratis). TAPI kompresi
agresif bisa memicu **output-token expansion** (sampai ~56× di sebagian benchmark). Maka:

```math
r_c(x) = \frac{T(x')}{T(x)},\qquad \text{target } 0.4 \le r_c \le 0.7
```

**Guardrail:** batalkan kompresi jika `T(x') \ge T(x)` (tidak menghemat) atau jika deteksi task
kode/format ketat (rawan expansion) → kirim prompt asli.

---

## 5. Tuning threshold (di eval set kecil, saat kickoff)

Untuk tiap kandidat `τ₁`, hitung di dataset berlabel:

```math
\mathrm{RemoteRate}(\tau_1)=\tfrac{1}{N}\sum_i \mathbb{1}[\text{escalate}_i],\quad
\mathrm{AvgRemoteTok}(\tau_1)=\tfrac{1}{N}\sum_i T_{remote}(x_i),\quad
\mathrm{Acc}(\tau_1)
```

Pilih:

```math
\tau_1^{*}=\arg\min_{\tau_1}\ \mathrm{AvgRemoteTok}(\tau_1)
\quad\text{s.t.}\quad \mathrm{Acc}(\tau_1)\ge \alpha
```

⚠️ Ranking router **sangat sensitif terhadap threshold** di call-rate 20–40% (2602.11877). Jangan
overfit ke eval set kecil: pakai margin (mis. target `α + 0.03`) dan validasi silang sederhana.

---

## 6. Cache (opsional, presisi tinggi)

Hit-rate nyata semantic cache hanya 20–45% (bukan 95%). Untuk env dengan accuracy floor, wrong-hit
berbahaya. Maka:

```math
\text{gunakan cache jika } \mathrm{sim}(x, x_{\text{cache}}) > 0.95 \ \text{(bukan 0.90)}
```

Prioritas rendah — implement setelah router + draft-patch stabil.

---

## 7. Pseudocode

```python
def answer(x):
    if cache_hit(x, thresh=0.95):
        return cache[x]                      # gratis, presisi tinggi

    draft, u = local_generate_with_uncertainty(x)   # 1 pass, ambil logprob/perplexity
    p = calibrator.predict(u)                        # isotonic: u -> P(local salah)

    if p < TAU1 and not high_risk(x):
        return draft                          # LOCAL_ONLY, remote token = 0

    x_c = compress(x)                          # dgn guardrail anti-expansion
    if p < TAU2:
        return remote_patch(x_c, draft)        # remote hanya koreksi minimal
    return remote_full(x_c)                    # remote jawaban penuh
```

---

## 8. Yang di-drop dari versi lama & alasannya

| Lama (generik) | Baru (grounded) |
|---|---|
| `R(x)=0.30D+0.25U+0.15L+...` weighted-sum | `p_wrong = g(u)` terkalibrasi (isotonic) |
| Qwen "rate confidence 0.0–1.0" | Perplexity/logprob **atau** self-consistency |
| Bobot heuristik di-tuning manual | Satu threshold di accuracy floor (cost-optimal) |
| Remote menjawab ulang penuh | Draft-patch (cap output token) |
| Cache sim > 0.90 | Cache sim > 0.95 / exact (hindari wrong-hit) |

**Novelty claim aman:** *"Calibration-first hybrid router — memetakan uncertainty lokal ke
probabilitas error terkalibrasi untuk keputusan escalate hemat-token, dengan remote dibatasi ke mode
patch."*
