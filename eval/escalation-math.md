# Matematika keputusan eskalasi — 10 Juli 2026

Reproduksi: `python3 eval/escalation_math.py` (Monte Carlo 20k sampel).
Data dasar: eval v4 (66 task, grader ketat) — lihat `findings-v4.md`.

## Model

- Akurasi lokal per kategori `p_c ~ Beta(benar+1, salah+1)` (posterior, jujur thd n kecil:
  8/8 dihitung ~0.9, bukan 1.0 — makanya E[acc] L0 di tabel 0.834, di bawah point estimate 0.91).
- Akurasi remote (Gemma-31B-class) `q ~ Beta(37,3)` (mean 0.925, asumsi — belum terukur).
- Hidden set: 64 task, 8/kategori (asumsi seimbang).
- Escalate = jawaban lokal DIGANTI remote → ada dua arah: memperbaiki error lokal
  ((1−p)·q) **dan merusak jawaban yang sudah benar** (p·(1−q)).
  `ΔAcc/task = (1−p)q − p(1−q)`.

## Hasil utama

### P(lolos gate) per kebijakan (grader ketat)

| kebijakan | E[acc] | E[token] | P≥.75 | P≥.80 | P≥.85 | P≥.90 |
|---|---:|---:|---:|---:|---:|---:|
| L0 zero-API | .834 | 0 | .93 | .71 | .41 | .14 |
| P1 math saja | .856 | ~650 | .97 | .83 | .56 | .23 |
| P2 math+logical (lvl2 LAMA) | .872 | ~1340 | .99 | .90 | .68 | .32 |
| **P3 math+factual (lvl2 BARU)** | **.878** | **~1380** | .99 | .91 | .71 | .37 |
| P4 +logical | .894 | ~2070 | 1.00 | .95 | .81 | .50 |
| P5 +sentiment (lvl3 BARU) | .910 | ~2510 | 1.00 | .98 | .89 | .63 |
| P6 semua (lvl4) | .925 | ~7880 | 1.00 | .97 | .90 | .74 |

**Temuan 1: P3 mendominasi P2** — akurasi lebih tinggi dengan token setara. Level-2 lama
(math+logical, warisan konsep "kategori trap") bukan frontier. Penyebab: error factual =
halusinasi yang TAK terlihat verifikasi lokal (2/6 error kami), sedangkan error logical
sebagian sudah tertangkap trap-routing classify (logical lokal naik ke 7/8 setelah patch).

### Efisiensi marginal per kategori (ΔAcc per 1000 token)

| kategori | p lokal | ΔAcc/task | tok/task | efisiensi |
|---|---:|---:|---:|---:|
| sentiment | .800 | +.125 | 55 | 2.27‰ * |
| math | .750 | +.175 | 81 | 2.16‰ |
| factual | .750 | +.175 | 92 | 1.90‰ |
| logical | .800 | +.125 | 86 | 1.45‰ |
| summarisation | .875 | +.050 | 169 | 0.30‰ |
| ner | .900 | +.025 | 94 | 0.27‰ |
| code_debug | .900 | +.025 | 194 | 0.13‰ |
| code_gen | .900 | +.025 | 214 | 0.12‰ |

\* sentiment tampak paling efisien HANYA karena outputnya 3 token; tapi satu-satunya error
sentiment kami adalah kasus "mixed" yang defensible (judge longgar mungkin menerimanya) —
sinyal dari 1 datapoint, jangan dipercaya berlebihan. Makanya rung-nya di level 3, bukan 2.

**Temuan 2 (negatif, penting): "verify-then-fix" MATI secara matematis.** Ide: kirim jawaban
lokal ke remote utk dicek YES/NO (output 1 token), regenerate hanya bila NO. Karena INPUT
dihitung, biaya verifikasi = prompt+jawaban+instruksi ≈ biaya menjawab langsung; hasil:
lebih mahal di SEMUA kategori (−25 s/d −39 token/task). Jangan dibangun.

**Temuan 3: eskalasi kategori yang sudah ~100% lokal = rugi murni.** ΔAcc ≈ +0.025/task
(hanya shrinkage prior) dengan token termahal (code 194–214/task) — dan berisiko merusak
jawaban benar (p·(1−q)). Level 4 hanya masuk akal bila gate > ~0.92.

### Sensitivitas (urutan kebijakan TIDAK berubah di semua skenario)

- Judge longgar (2 error defensible diterima): L0 P≥.80 naik .71→.83; P3 → .95.
- Remote lebih lemah (q mean .85): semua turun, tapi P3 tetap ≥ P2 dan urutan rung sama.

## Ladder yang diimplement (solve.py `_LEVEL_CATEGORIES`)

| ESCALATION_LEVEL | escalate |
|---|---|
| 0 | tidak pernah (zero-API) |
| 1 | hanya `empty` / `verify_failed` (sinyal presisi tinggi, hampir gratis) |
| 2 | + `unverifiable` di {math, factual} |
| 3 | + {sentiment, logical} |
| 4 | semua `unverifiable` |

## Protokol leaderboard (oracle 1-bit, 10 submit/jam)

Ranking = token ascending di antara yang lolos gate, dan hanya konfigurasi terakhir yang
berlaku → strategi optimal = cari **rung terendah yang lolos**:

1. Submit **L0**. Lolos → SELESAI (0 token tak terkalahkan; freeze).
2. Gagal → submit **lvl 3** (P5, prob lolos tinggi, token masih moderat).
   - Lolos → turun satu rung (lvl 2); lolos lagi → coba lvl 1; gagal → naik balik ke
     rung terakhir yang lolos, freeze. (Walk-down aman: konfigurasi lolos selalu bisa
     di-resubmit.)
   - Gagal → lvl 4. Gagal juga → masalahnya akurasi remote/kategori kuat kami, bukan
     kebijakan — investigasi feedback, jangan naik-turun buta.
3. Setiap probe dicatat di `submissions-log.md` (tag image, level, status, skor) —
   tiap hasil memperketat interval estimasi T (gate).

Ekspektasi jumlah probe sampai konvergen: ≤ 5 — jauh di bawah rate limit.
