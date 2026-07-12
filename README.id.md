# TerraceRoute - Submission AMD Hackathon

Track 1 dan Track 2 berada dalam satu repository, tetapi dibangun dan disubmit sebagai
container terpisah. Dengan demikian setiap image penilaian tetap minimal dan tidak
bergantung pada deteksi schema. Container terpadu di root hanya untuk pengujian lokal.

> 🇬🇧 English version: [`README.md`](README.md)

## Daftar Track

| Track | Folder | Judul | Status |
|-------|--------|-------|--------|
| 1 | [`track1/`](track1/) | **TerraceRoute** — agen perutean hemat token | ✅ Selesai |
| 2 | [`track2/`](track2/) | Video captioning dalam empat gaya wajib | ✅ Diimplementasikan |
| 3 | [`track3/`](track3/) | Unicorn — infrastruktur agen swakelola di AMD | 📝 Perencanaan |

## Container Event

Build setiap image dari direktori track masing-masing:

```bash
docker build --platform linux/amd64 -t terraceroute-track1:test track1
docker build --platform linux/amd64 -t terraceroute-track2:test track2
```

Publikasikan keduanya dengan tag unik yang tidak ditimpa:

```bash
REGISTRY=ghcr.io/alvnvnc ./scripts/build_and_push.sh 1 track1-v2
REGISTRY=ghcr.io/alvnvnc ./scripts/build_and_push.sh 2 track2-v2
```

Masukkan referensi yang sesuai pada form masing-masing track tanpa `http://` atau
`https://`:

```text
Track 1: ghcr.io/alvnvnc/terraceroute:track1-v2
Track 2: ghcr.io/alvnvnc/terraceroute:track2-v2
```

Kedua image harus publik dan memiliki manifest `linux/amd64`. Jangan menimpa tag setelah
disubmit; gunakan tag baru agar grader tidak memakai manifest cache lama.

## Container Terpadu Lokal

Root image tetap dapat dibangun untuk pengujian integrasi kedua runner:

```bash
docker build -t terraceroute:unified-test .
```

Jangan gunakan image yang lebih besar ini sebagai referensi event utama kecuali event
secara eksplisit meminta satu image untuk beberapa track.

## Track 1 - TerraceRoute

Agen batch yang menjawab tugas natural-language pada 8 kategori dengan menggunakan
**token API Fireworks seminimal mungkin** — inferensi lokal tidak dikenai biaya, dan
papan peringkat mengurutkan peserta yang lolos berdasarkan jumlah token menaik. Model
lokal 3B yang dipadukan dengan deterministic tools serta pemeriksaan sintaks mencapai
**akurasi ~85–91% pada nol token API**, dan
melakukan eskalasi ke model remote hanya ketika kebijakan berbasis data menilai kenaikan
akurasinya sepadan dengan biaya token.

Lihat [`track1/README.id.md`](track1/README.id.md) untuk arsitektur lengkap, evaluasi,
serta panduan build dan menjalankannya.

## Tata Letak

```
entrypoint.py     ← dispatcher lokal schema Track 1/Track 2
Dockerfile       ← image integrasi terpadu opsional
requirements.txt ← dependency runtime terpadu
track1/          ← agen teks dan Dockerfile penilaian Track 1
track2/          ← agen video dan Dockerfile penilaian Track 2
track3/          ← proyek Unicorn Track terpisah
```
