# AMD Developer Hackathon (ACT II) — Kumpulan Submission

Monorepo yang memuat submission saya untuk AMD Developer Hackathon (ACT II).
Setiap track berada pada folder mandiri dengan README, kode, dan dokumentasinya sendiri.

> 🇬🇧 English version: [`README.md`](README.md)

## Daftar Track

| Track | Folder | Judul | Status |
|-------|--------|-------|--------|
| 1 | [`track1/`](track1/) | **TerraceRoute** — agen perutean hemat token | ✅ Selesai |
| 2 | [`track2/`](track2/) | — | ⏳ Placeholder |
| 3 | [`track3/`](track3/) | Unicorn — infrastruktur agen swakelola di AMD | 📝 Perencanaan |

## Track 1 — TerraceRoute (sorotan)

Agen batch yang menjawab tugas natural-language pada 8 kategori dengan menggunakan
**token API Fireworks seminimal mungkin** — inferensi lokal tidak dikenai biaya, dan
papan peringkat mengurutkan peserta yang lolos berdasarkan jumlah token menaik. Model
lokal 3B yang dipadukan dengan **verifikasi deterministik independen** (eksekusi kode,
penghitungan huruf, aritmetika) mencapai **akurasi ~85–91% pada nol token API**, dan
melakukan eskalasi ke model remote hanya ketika kebijakan berbasis data menilai kenaikan
akurasinya sepadan dengan biaya token.

Lihat [`track1/README.id.md`](track1/README.id.md) untuk arsitektur lengkap, evaluasi,
serta panduan build dan menjalankannya.

## Tata Letak

```
README.md        ← halaman utama repo (Inggris)
README.id.md     ← Anda di sini (halaman utama Indonesia)
track1/          ← Submission Track 1 (selesai)
track2/          ← Track 2 (placeholder)
track3/          ← Track 3 (placeholder)
```
