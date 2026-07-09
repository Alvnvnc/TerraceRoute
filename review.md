Untuk **hackathon Track 1**, dari model yang Anda punya, saya sarankan pakai kombinasi ini:

```text id="dy6b6n"
Primary local model:
qwen2.5:14b-instruct

Fallback / eksperimen:
gemma3:12b

Lightweight classifier:
qwen2.5:3b-instruct-q4_K_M

Embedding, kalau butuh RAG:
bge-m3:latest atau nomic-embed-text-v2-moe
```

## Rekomendasi utama saya

Pakai:

```text id="kbdchj"
qwen2.5:14b-instruct
```

sebagai **local model utama**.

Alasannya: Track 1 menilai agent berdasarkan **token count dan output accuracy**. Agent harus memutuskan kapan cukup pakai local model dan kapan perlu memanggil remote model via Fireworks AI. Local model/token lokal dihitung **0 terhadap final score**, jadi model lokal yang cukup pintar sangat berharga. ([LabLab][1])

Jadi arsitektur paling aman untuk Anda:

```text id="mil8z2"
User task
  ↓
qwen2.5:3b-instruct-q4_K_M
  → classify: EASY / MEDIUM / HARD
  ↓
Kalau EASY:
  qwen2.5:14b-instruct jawab lokal
  ↓
Kalau MEDIUM:
  qwen2.5:14b-instruct jawab + self-check
  ↓
Kalau HARD / confidence rendah:
  compress prompt
  ↓
  Fireworks AI
```

## Kenapa bukan Gemma dulu?

`gemma3:12b` bagus, dan lablab.ai memang menyebut Gemma sebagai partner technology yang bisa dipakai dalam workflow hackathon. ([LabLab][1])

Tapi untuk setup Anda sekarang, saya akan jadikan Gemma sebagai **pembanding**, bukan model utama.

Pakai Gemma untuk:

```text id="o10jfd"
- compare hasil Qwen vs Gemma
- local verifier
- demo "multi-local model router"
```

Tapi jangan jalankan Gemma dan Qwen 14B bersamaan terus, karena VRAM Anda sudah penuh. Lebih baik satu model utama dulu.

## Pilihan final paling praktis

Untuk development cepat:

```bash id="5rv2db"
ollama stop gemma3:12b
```

Lalu pakai:

```text id="gklghx"
qwen2.5:14b-instruct
```

Untuk router ringan, Anda bisa load:

```text id="wmstgd"
qwen2.5:3b-instruct-q4_K_M
```

Tapi kalau VRAM mepet, tidak perlu 3B classifier. Cukup gunakan **rule-based classifier + qwen2.5:14b-instruct**.

## Yang paling cocok untuk submission

Karena semua submission harus **containerized**, dan final scoring berjalan di standardized environment, jangan buat app yang hanya bisa jalan di PC Quadro Anda. Lablab.ai juga menulis bahwa final scoring memakai environment standar, jadi local model harus cukup kecil agar bisa jalan dalam constraint tersebut. ([LabLab][1])

Maka pilihan paling aman untuk submission:

```text id="s7g3zu"
Default local model:
qwen2.5:3b-instruct-q4_K_M

Optional stronger local model:
qwen2.5:14b-instruct

Remote:
Fireworks AI model yang diumumkan saat launch
```

## Kesimpulan

Untuk **bangun dan demo di PC Anda**:

```text id="exk3l1"
Pakai qwen2.5:14b-instruct
```

Untuk **submission yang aman dan portable**:

```text id="snjhrm"
Pakai qwen2.5:3b-instruct-q4_K_M sebagai default local router,
qwen2.5:14b-instruct sebagai optional strong local backend,
Fireworks AI sebagai fallback hard task.
```

Strategi terbaiknya bukan memilih model terbesar, tapi membuat **router yang tahu kapan tidak perlu memanggil Fireworks**. Itu paling sesuai dengan Track 1.

[1]: https://lablab.ai/ai-hackathons/amd-developer-hackathon-act-ii "AMD Developer Hackathon: ACT II AI Hackathon | lablab.ai"
