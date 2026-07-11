# TerraceRoute — Agen Perutean Hemat Token

**AMD Developer Hackathon (ACT II) · Track 1**

Sebuah agen **batch** yang menyelesaikan tugas natural-language pada 8 kategori resmi dengan
menggunakan **token API Fireworks seminimal mungkin** — inferensi lokal tidak dikenai biaya,
sehingga strategi terbaik adalah menjawab secara lokal setiap kali kebenarannya dapat
*dibuktikan*, dan melakukan eskalasi ke model remote hanya ketika kebijakan berbasis data
menyatakan hal itu sepadan.

> 🇬🇧 English version: [`README.md`](README.md)

## Kontrak Harness (yang dinilai)

- Membaca `/input/tasks.json` (`[{task_id, prompt}]`) → menulis `/output/results.json`
  (`[{task_id, answer}]`), lalu keluar dengan kode `0`. Keluaran yang rusak bernilai nol.
- Environment yang diinjeksi: `FIREWORKS_API_KEY`, `FIREWORKS_BASE_URL` (setiap panggilan
  remote wajib melalui URL ini), dan `ALLOWED_MODELS` (dibaca saat runtime — tidak pernah
  ditulis permanen di kode).
- Lingkungan penilaian: **RAM 4 GB, 2 vCPU, tanpa GPU, anggaran total 10 menit**.
- Penilaian: terdapat gerbang akurasi (LLM-Judge, dengan varian prompt yang tak terlihat
  sebelumnya); peserta yang lolos kemudian **diurutkan berdasarkan total token Fireworks
  menaik** (input + output). **Token lokal bernilai nol**, sehingga peserta zero-API yang
  lolos gerbang merupakan optimum teoretis.

## Cara Kerja — Terrace Routing

Setiap tugas menuruni "teras" pemeriksaan yang makin mahal dan keluar sedini mungkin:

```
klasifikasi kategori (regex, gratis)
  ├─ penghitungan huruf ("berapa 'r' pada strawberry"; hanya prompt non-code) → hitung di Python  (DETERMINISTIK, 0 token)
  ├─ math    → selesaikan lokal + hitung ulang dengan Python
  ├─ code    → selesaikan lokal + cek sintaks `ast.parse` di subprocess  (menangkap kode yang gagal dimuat)
  └─ lainnya → selesaikan lokal (model 3B)
        ↓ kebijakan eskalasi (knob ESCALATION_LEVEL, 0..4)
   verifikasi gagal / jawaban lokal kosong → eskalasi (level ≥ 1)
   tak terverifikasi + kategori pada rung ini → eskalasi (level 2/3)
        ↓
   remote: prompt ringkas (hanya jawaban final) + batas max_tokens per kategori → token minimal
```

**Wawasan kunci (dari kalibrasi).** Sinyal keyakinan *internal* model kecil — perpleksitas,
konsistensi-diri, bahkan "kode Python yang ia tulis untuk memeriksa dirinya sendiri" —
**berkorelasi negatif** dengan kebenaran: model kerap *salah dengan penuh keyakinan dan
konsisten*. Karena itu, satu-satunya yang dianggap **terverifikasi** adalah pemeriksaan yang
benar-benar **independen** (cek sintaks pada kode yang dihasilkan, menghitung huruf,
aritmetika yang kita urai sendiri), bukan model yang menyetujui dirinya sendiri. Selebihnya dieskalasi sesuai anggaran
token yang diizinkan oleh gerbang akurasi.

## Tangga Eskalasi (berbasis data)

Urutan rung bukan berdasar intuisi, melainkan hasil model keputusan Monte-Carlo atas akurasi
per-kategori yang terukur (lihat [`docs/escalation-math.md`](docs/escalation-math.md)).
Efisiensi marginal (ΔAkurasi per token) mengurutkan **math ≈ factual > sentiment ≈ logical ≫**
kategori yang sudah dikuasai model lokal (NER, ringkasan, kode — mengeskalasinya justru rugi
murni).

| `ESCALATION_LEVEL` | mengeskalasi |
|---|---|
| 0 | tidak pernah (zero-API) |
| 1 | hanya `empty` / `verify-failed` (sinyal presisi tinggi, nyaris gratis) |
| 2 | + `math` & `factual` yang tak terverifikasi |
| 3 | + `sentiment` & `logical` |
| 4 | + semua yang tak terverifikasi |

Satu gagasan yang ditolak dan perlu dicatat: **"verify-then-fix"** (meminta remote menilai
jawaban lokal YA/TIDAK, lalu meregenerasi hanya bila TIDAK) ternyata *merugikan secara neto di
semua kategori* — karena token input ikut dihitung, biaya pemeriksaannya kira-kira sama dengan
menjawab langsung. Perhitungannya ada pada dokumen di atas.

## Rekayasa Keandalan

Kegagalan infrastruktur bernilai nol, sehingga runner dibuat defensif:

- **Penulisan atomik progresif** — setiap tugas langsung ditulis; crash atau timeout tetap
  meninggalkan `results.json` parsial yang valid.
- **Watchdog** — menulis dan keluar dengan bersih menjelang batas 10 menit.
- **Throttle preemptif + adaptif** — memangkas `LOCAL_MAX_TOKENS` bila proyeksi waktu jalan
  akan melewati anggaran (waktu generasi CPU ~linear terhadap panjang keluaran).
- **Eskalasi paralel** — panggilan remote berjalan pada thread pool, menumpangtindihkan
  latensi jaringannya dengan pekerjaan lokal tugas berikutnya (2 vCPU tetap fokus pada
  llama.cpp).
- **Tail-to-remote awal/akhir** — bila antrean lokal tak dapat selesai tepat waktu, ekornya
  dikirim ke model remote alih-alih dibiarkan kosong. Dikontrol oleh *kredensial*, bukan
  `ESCALATION_LEVEL`: jawaban kosong pasti gagal gerbang, sedangkan token hanya penalti
  peringkat. Pada set tugas berukuran normal, jalur ini tak pernah tersentuh (tetap 0 token).

## Tata Letak Repositori

```
agent/
  run.py        entrypoint: I/O batch, watchdog, penulisan progresif, eskalasi paralel, throttle
  solve.py      terrace per-kategori + keputusan eskalasi terpadu + coercion format gratis
  classify.py   klasifikasi 8 kategori (regex, gratis)
  verify.py     verifikasi deterministik: aritmetika, eksekusi Python tersandbox, penghitungan
  local_llm.py  model lokal: ollama (dev) / llama-cpp in-process (prod)
  fireworks.py  klien eskalasi (membaca ALLOWED_MODELS saat runtime, semua via BASE_URL)
  config.py     seluruh pengaturan dari env; http.py = urllib (tanpa dependensi jaringan)
Dockerfile      multi-stage, tanpa GPU, model menyatu di image (~1,8 GB terkompresi)
requirements.txt
scripts/build_and_push.sh   build linux/amd64 + push ke registry publik
eval/           harness evaluasi yang reprodusibel (lihat di bawah)
docs/           escalation-math.md — seluruh desain dalam satu dokumen (matematika keputusan, ladder, hasil terukur)
```

## Menjalankan

### Dev (Ollama di host — iterasi cepat)

```bash
cp .env.example .env            # lalu sunting seperlunya
pip install -r requirements.txt
ollama pull qwen2.5:3b-instruct
INPUT_PATH=eval/practice_tasks.json OUTPUT_PATH=/tmp/results.json \
  LOCAL_BACKEND=ollama python -m agent.run
```

### Prod (Docker — mencerminkan lingkungan penilaian)

```bash
docker build -t terraceroute:latest .
docker run --rm --cpus=2 --memory=4g --network none \
  -v "$PWD/eval":/input:ro -v /tmp/out:/output \
  terraceroute:latest        # jalan zero-API, luring
```

Image menyertakan model GGUF sehingga tidak memerlukan jaringan saat penilaian. Build dan push
image `linux/amd64` publik dengan `REGISTRY=ghcr.io/<user> ./scripts/build_and_push.sh v1`.

## Evaluasi

`eval/` mencerminkan 8 kategori resmi dengan penilaian deterministik (kode dinilai dengan
**mengeksekusinya terhadap assert**, fakta dengan pencocokan sadar-batas, ringkasan oleh juri
independen gemma-3-12B):

```bash
python -m eval.gen_tasks_v5                        # regenerasi eval/tasks_v5.jsonl (124 tugas, ~16/kategori)
python -m eval.validate_tasks --tasks eval/tasks_v5.jsonl   # tanpa-LLM: buktikan set well-formed
OLLAMA_HOST=<host> python -m eval.agent_eval --tasks eval/tasks_v5.jsonl   # jalankan solver produksi
python -m eval.escalation_math --results eval/agent_eval_results.jsonl     # hitung ulang ladder dari data
```

`validate_tasks` menjalankan solusi referensi tiap tugas kode terhadap assert-nya sendiri, jadi test
yang salah ketik tak bisa diam-diam membiaskan estimasi akurasi. `escalation_math --results` menghitung
ulang seluruh ladder kebijakan dari run *terukur* (tanpa flag → pakai angka v4 bawaan). Set v5 yang lebih
besar dibuat karena v4 (66 tugas) terlalu kecil untuk mengestimasi rate per-kategori secara andal.

Diukur pada mesin evaluasi AMD Radeon (v4): **akurasi zero-API ≈ 0,85–0,91**, level-2 ≈ 0,91 dengan
~1,1–1,4 ribu token remote atas 66 tugas. Hasil terukur lengkap ada di
[`docs/escalation-math.md`](docs/escalation-math.md) §11.

## Protokol Submission / Papan Peringkat

Papan peringkat adalah oracle kalibrasi yang sesungguhnya (10 submission/jam). Karena
pemeringkatan menaik berdasarkan token di antara peserta yang lolos, langkah optimalnya adalah
mencari **rung terendah yang masih lolos**:

1. Submit **level 0** (zero-API). Lolos → selesai, bekukan (0 token tak terkalahkan).
2. Gagal → lompat ke **level 3** (probabilitas lolos tinggi, token moderat); lalu turun
   bertahap ke rung termurah yang masih lolos dan bekukan di sana.

Protokol probing lengkap ada di [`docs/escalation-math.md`](docs/escalation-math.md) §8.

## Dokumen

- [`docs/escalation-math.md`](docs/escalation-math.md) — seluruh desain dalam satu dokumen: teori
  keputusan eskalasi, urutan rung berbasis data, protokol probing papan peringkat, dan hasil
  evaluasi terukur (§11). Berbahasa Inggris.
