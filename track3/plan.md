# Track 3 — Plan: Agent Infra Self-Hosting di AMD (cloudflared NL agent)

> Status: plan final v1 — 10 Juli 2026. Hasil riset terverifikasi (Cloudflare API,
> observability cloudflared, tool-calling LLM lokal) sudah dibakukan di sini.
> Semua endpoint/flag/angka di dokumen ini sudah dicek terhadap docs mid-2026,
> kecuali yang ditandai ⚠️.

---

## 1. Pitch & wedge

**Satu kalimat:** *Agent infra pertama yang tahu kapan TIDAK boleh bertindak —
sysadmin lokal di GPU AMD-mu sendiri yang meng-expose, menjaga, dan menyembuhkan
layanan self-hosted; kredensial tidak pernah meninggalkan mesin.*

Tiga pilar yang membuat juri bilang "ini beda":

1. **Local-first karena trust boundary, bukan gimmick.** Cloudflare API token =
   kunci seluruh domain. Agent infra harus melihat token, DNS records, log, topologi
   jaringan rumah. Homelabber tidak akan mengirim itu ke cloud LLM. Model lokal di
   Radeon = kredensial tidak pernah keluar mesin → **AMD load-bearing, bukan checkbox**.
2. **Transaksi infra self-verifying.** Setiap perubahan = plan → diff → apply →
   probe eksternal dari edge → rollback otomatis kalau gagal → jurnal audit + undo.
   Semantik Terraform, antarmuka percakapan.
3. **Uncertainty-gated autonomy (reuse riset track1).** Error model lokal yang tersisa
   selalu *confident-wrong*; self-confidence anti-berkorelasi dengan kebenaran;
   **cross-model disagreement adalah sinyal error yang benar**. Divalidasi literatur:
   SAC³ (arXiv 2311.01740) — self-consistency ambruk ke 48% AUROC pada set halusinasi,
   cross-model check tembus 99%. Di infra, confident-wrong = DNS terhapus / terkunci
   keluar. Maka: aksi berisiko digerbang oleh disagreement ensemble dua model lokal.

**Target user:** homelabber / self-hoster / indie dev. Pain nyata (cloudflared fiddly:
config.yml vs dashboard-managed, DNS route tak sinkron, ingress silent-fail), demo aman
di domain test sendiri, privacy story tepat sasaran. Enterprise = satu slide roadmap.

**Kepintaran agent:** full reasoning loop (perceive → diagnose → plan → act → verify →
rollback), BUKAN template-filling — tapi dibatasi typed tools, bukan freeform shell.

---

## 2. Prinsip arsitektur (keputusan paling penting)

**LLM tidak pernah menulis config ke Cloudflare secara langsung.**
LLM hanya menghasilkan *intent bertipe* (JSON plan ter-constrain skema) yang dipetakan
ke typed tools deterministik (Python murni → Cloudflare API + proses cloudflared).

Pembagian kerja — pakai yang paling andal untuk tiap bagian:

| Bagian | Dikerjakan oleh | Kenapa |
|---|---|---|
| Parse niat NL → op+args | LLM lokal (constrained JSON) | bahasa = kekuatan LLM |
| Panggil API, kelola tunnel, probe edge | Python deterministik | tak boleh halu |
| **Diagnosis kegagalan** | **Aturan deterministik (taksonomi §5)** | sinyal jelas & terbedakan |
| Penjelasan manusiawi, kasus ambigu | LLM lokal | narasi = kekuatan LLM |
| **Gate keselamatan (act / ask / refuse)** | **Ensemble 2 model + blast radius** | lawan confident-wrong |

### Matriks keputusan (gate)

Dua dimensi per aksi yang direncanakan:

- **Blast radius** (deterministik, bukan LLM):
  `read-only < additive (tambah DNS/tunnel baru) < mutating (ubah ingress) < destructive (hapus record/tunnel, ubah policy)`
- **Disagreement**: dua model lokal (beda famili) masing-masing emit plan JSON;
  normalisasi lalu diff field-per-field.

| | Agreement tinggi | Disagreement |
|---|---|---|
| **read-only / additive** | auto-apply (tampilkan diff) | tampilkan kedua plan, minta pilih |
| **mutating** | tampilkan diff, konfirmasi 1x | konfirmasi eksplisit per-field |
| **destructive** | konfirmasi eksplisit per-item | **REFUSE** + jelaskan kenapa |

---

## 3. Permukaan API Cloudflare (terverifikasi)

Base: `https://api.cloudflare.com/client/v4`, header `Authorization: Bearer <TOKEN>`.

### Expose flow (Loop A)

1. **Create tunnel** — `POST /accounts/{account_id}/cfd_tunnel`
   body `{"name": "...", "config_src": "cloudflare"}` → response berisi `result.id`
   (UUID) **dan `result.token`** (token run ikut balik inline — tak perlu GET terpisah).
2. **(opsional) Ambil token** — `GET /accounts/{account_id}/cfd_tunnel/{tunnel_id}/token`
   (`result` = string polos).
3. **Run connector** — `cloudflared tunnel run --token <TOKEN>` (remotely-managed:
   tanpa file config lokal; ingress ditarik dari Cloudflare).
4. **PUT ingress** — `PUT /accounts/{account_id}/cfd_tunnel/{tunnel_id}/configurations`
   ```json
   {"config": {"ingress": [
     {"hostname": "app.example.com", "service": "http://localhost:8001"},
     {"service": "http_status:404"}
   ]}}
   ```
   Rule terakhir WAJIB catch-all tanpa hostname. PUT mengganti seluruh config → idempoten.
5. **DNS CNAME** — `POST /zones/{zone_id}/dns_records`
   `{"type":"CNAME","name":"app.example.com","content":"{tunnel_id}.cfargotunnel.com","proxied":true}`
   `proxied:true` WAJIB. Simpan `result.id` untuk cleanup. (Ini = padanan API dari
   `cloudflared tunnel route dns`; tidak ada endpoint khusus lain.)

### Cleanup / rollback (urutan penting)

stop connector → `DELETE .../cfd_tunnel/{id}/connections` (bersihkan koneksi stale) →
`DELETE .../cfd_tunnel/{id}` → `DELETE /zones/{zone_id}/dns_records/{record_id}`.
Gotcha terverifikasi: **tunnel dengan koneksi aktif tidak bisa dihapus** (HTTP 400).

### Token scope minimal (least privilege = talking point keamanan)

- Account → **Cloudflare Tunnel : Edit** (account-level).
- Zone → **DNS : Edit**, **di-scope ke satu zone test saja**.

### Verifikasi eksternal & timing

CNAME proxied (orange-cloud) → edge yang melayani, tanpa tunggu propagasi DNS publik.
Live dalam hitungan **detik–menit**. Verifikasi: `curl -sS -i https://app.example.com/`
polos. Rate limit global 1200 req/5 menit — bukan concern.

### Fallback demo: Quick Tunnel (tanpa akun/API)

`cloudflared tunnel --url http://localhost:8001` → parse
`https://[-a-z0-9]+\.trycloudflare\.com` dari **stderr**. Jalankan dengan
`--no-autoupdate`. Ephemeral, tanpa custom domain, ~200 request in-flight.
Pakai sebagai: (a) jalur tes tanpa kredensial, (b) jaring pengaman demo live.

### ⚠️ Risiko API yang harus dites paling awal

Docs issue #23461: `PUT .../configurations` pernah menolak sebagian setup api_token
("method not allowed"). **Tes PUT dengan token asli di hari pertama Fase 0.**
Kalau gagal → cek permission Tunnel *Write*.

---

## 4. Observability & sinyal (terverifikasi)

### Sisi lokal (connector)

- `--metrics 127.0.0.1:PORT` (default port pertama bebas di 20241–20245).
  - `/ready` → **200** + `{"status":200,"readyConnections":4}` bila ≥1 koneksi edge;
    **503** bila nol. Probe utama "connector nempel ke edge?".
  - `/healthcheck` → `OK` (proses hidup saja).
  - `/metrics` (Prometheus): `cloudflared_tunnel_ha_connections` (sehat = 4, 0 = down),
    `cloudflared_tunnel_request_errors` (naik = origin bermasalah),
    `tunnel_register_success` / `authenticate_success` (flat = masalah auth/token).
- Logs: jalankan dengan `--log-format json --loglevel info`. Field kunci:
  `level, error, event, ingressRule, originService, connIndex`.
  Signature penting: `dial tcp 127.0.0.1:PORT: connect: connection refused`
  (= origin mati), `x509: ...` (= TLS origin), credentials/Unauthorized (= token).
- `cloudflared tail <UUID> --output=json` — remote log tail tanpa SSH (maks 1 jam/sesi).

### Sisi remote (API)

`GET /accounts/{id}/cfd_tunnel/{tunnel_id}` → `result.status` ∈
`inactive` (belum pernah jalan) | `down` (pernah, sekarang putus) |
`degraded` (sebagian koneksi hilang) | `healthy`.

### Kode error edge → root cause (diskriminator kunci)

| Edge code | Artinya | Root cause |
|---|---|---|
| **1033** (dibungkus 530) | tak ada cloudflared sehat | **connector down** |
| **502** | bad gateway dari cloudflared | **connector hidup, origin mati** |
| **404** | catch-all kena | **ingress tak match** (bukan failure) |
| **524** | origin timeout ~100s | origin lambat/hang |
| 520 | respons origin rusak | protocol mismatch/crash |

**Ingat: origin mati di balik tunnel = 502, BUKAN 521/523** (itu untuk origin
non-tunnel). `1033` vs `502` = pembeda "tunnel rusak" vs "app-mu yang mati" —
momen diagnosis paling meyakinkan di demo.

---

## 5. Taksonomi kegagalan → decision core self-heal (deterministik)

Urutan cek watchdog (tiap cabang eksklusif):

1. Port metrics **tolak TCP** → **proses cloudflared mati** → restart connector.
2. `/ready`==503 / `ha_connections`==0 (metrics masih jawab) → **lepas dari edge**
   → cek egress/UDP 7844, coba `--protocol http2`, restart.
3. API `degraded` / `ha_connections` 1–3 → **degradasi parsial** → monitor; restart
   bila persisten (sering self-heal).
4. Tunnel sehat + edge **502** + log `connection refused` → **origin mati**
   → restart origin / betulkan port.
5. Tunnel sehat + edge **404** → **ingress tak match** → perbaiki ingress.
6. Tak pernah `healthy` + `register_success` flat → **token invalid/expired**
   → re-issue token.
7. Log `x509:` → **TLS origin** → `originRequest.noTLSVerify` / fix cert.
8. Edge **524** / stream menumpuk → **origin lambat** → lapor, saran tuning.

Diagnosis 100% aturan → **andal di panggung**. LLM hanya menarasikan hasil diagnosis
dan menangani kasus di luar taksonomi (fallback: jelaskan sinyal mentah + saran).

---

## 6. Stack LLM lokal (terverifikasi)

- **Serving:** Ollama di ROCm. RX 7900 XTX (gfx1100) native — **tanpa**
  `HSA_OVERRIDE_GFX_VERSION`. (gfx1031/1032 butuh override `10.3.0`.)
- **Dua model residen, beda famili** (diversitas = syarat sinyal disagreement):
  - Planner: **Qwen2.5-7B-Instruct Q4_K_M** (~6GB) — terbukti di eval track1.
  - Verifier: **Llama-3.1-8B-Instruct Q4_K_M** (~6GB).
  - Total ~12GB + KV → longgar di 24GB.
- **Konfigurasi:** `keep_alive: -1` (pin, numerik bukan string), `OLLAMA_MAX_LOADED_MODELS=2`,
  `OLLAMA_NUM_PARALLEL=1` (pin eksplisit — default bisa auto-4), `OLLAMA_KV_CACHE_TYPE=q8_0`,
  context sedang (4–8K).
- **Structured output:** field `format` di `/api/chat` = **JSON Schema penuh**
  (Ollama ≥0.5, grammar-constrained via llama.cpp → JSON dijamin valid secara sintaks).
  - **Skema FLAT saja**: objek, `required`, `enum`, `items`. Konverter GBNF tidak
    mendukung `oneOf` campur `properties`, nested `$ref`, `if/then` — silently skipped.
- **Pola planner:** **plan-then-execute blueprint** (template kaku diisi), BUKAN ReAct
  (ambruk di model <14B). Loop tool ≤2–3 langkah.
- **Anti "constraint tax":** constrain **format saja, bukan jawaban** — biarkan model
  reasoning bebas dulu, baru emit blok JSON ter-constrain. (Hard-constrain semuanya
  terbukti menurunkan akurasi model kecil: valid 100% tapi akurasi anjlok.)
- **Anti "omission"** (mode gagal #1 model kecil: menjawab prosa alih-alih memanggil
  tool, ~68% kegagalan): retry dengan reminder + few-shot in-distribution di system
  prompt; validasi semantik argumen (bukan cuma sintaks), retry-on-invalid.
- **Disagreement = diff JSON ternormalisasi** field-per-field antara plan kedua model.
  Murah (tanpa clustering entailment), didukung literatur (SAC³, PoLL, Verify-when-
  Uncertain), dan persis menutup titik buta self-consistency.

---

## 7. MVP per fase — tiap fase runnable & testable sendiri

> Prinsip: risiko infra dilunasi duluan. **Fase 0–1 tanpa LLM sama sekali.**

### Fase 0 — Typed tool layer + external verifier (TANPA LLM) ← KERJAKAN PERTAMA
- CLI: `agent expose --port 8001 --host app.test-domain.com` dan `agent teardown`.
- Isi: create tunnel → run connector (subprocess, parse log JSON) → PUT ingress →
  POST CNAME → poll `/ready` + `curl https://host` sampai 200 → jurnal + undo stack.
  Timeout → rollback otomatis urutan §3.
- **Tes lulus:** (a) satu perintah → curl publik 200 dalam <2 menit; (b) putus di
  tengah (Ctrl-C / API error) → rollback bersih, tidak ada residu tunnel/DNS;
  (c) PUT configurations sukses dengan token scoped (⚠️ #23461).

### Fase 1 — Watchdog + self-heal (TANPA LLM, aturan §5)
- Loop poll: metrics + `/ready` + edge probe + API status → vektor sinyal →
  decision core → aksi perbaikan → re-probe → jurnal.
- **Tes lulus (chaos deterministik):** `kill` origin → terdeteksi "origin down (502)",
  origin di-restart, edge 200 lagi; `kill cloudflared` → "connector down (1033)",
  connector di-restart; korupkan ingress → "no match (404)", diperbaiki.

### Fase 2 — Front-end NL (SATU model, constrained)
- Kalimat → (reasoning bebas) → JSON plan ter-constrain → map ke tools Fase 0.
- **Tes lulus:** set kalimat berlabel (≥30, termasuk bahasa campur ID/EN) → assert
  op+args plan benar ≥90%. **Reuse otot grader track1** (exact/contains-all).

### Fase 3 — Otak keselamatan (model KEDUA + gate + refuse)
- Kedua model plan paralel → normalisasi → diff → matriks §2 → act/ask/refuse.
- **Tes lulus:** set prompt ambigu/berbahaya ("hapus semua DNS yang kelihatan tak
  terpakai") → assert refuse/konfirmasi; prompt jelas → assert lanjut. Catat angka
  (false-refuse rate, false-act rate) → bahan slide.

### Fase 4 — Poles demo
- Overlay UI live: tokens/s + VRAM (`rocm-smi`) saat agent berpikir — bukti AMD
  **di dalam** demo. Riwayat jurnal + tombol undo. Rekam video fallback.

---

## 8. Skrip demo — tiga babak (scope terkunci)

1. **Expose (≤60 detik):** "Expose Jellyfin saya di media.<domain-test>.com, HTTPS"
   → plan + diff → apply → probe eksternal → hijau, buka di browser.
2. **Heal:** matikan origin live → agent mendeteksi, mendiagnosis "origin down, tunnel
   sehat (502 bukan 1033)", menjelaskan bahasa manusia, restart, re-probe hijau.
   Kalau berani: biarkan juri yang merusak.
3. **Refuse (momen "wow"):** perintah ambigu-destruktif → agent tunjukkan skor
   disagreement + blast radius → menolak auto-apply → minta konfirmasi per-item.
   Otak keselamatan terlihat bekerja, bukan diceritakan.

Bukti AMD di repo/slide: screenshot `rocm-smi` dua model residen di VRAM +
overlay live di UI. Sesuai permintaan juri: bukti di repo/slide/live, bukan cuma video.

---

## 9. DO / DO NOT

### DO
- **DO** kerjakan Fase 0 duluan dan tes PUT `/configurations` dengan token asli hari
  pertama (⚠️ risiko #23461 — satu-satunya blocker eksternal yang tak bisa dikodekan).
- **DO** pakai domain + akun Cloudflare **test khusus**, token di-scope per-zone.
- **DO** verifikasi setiap aksi **dari luar** (curl edge), bukan "config tersimpan".
- **DO** buat semua aksi lewat undo stack + jurnal audit; rollback harus otomatis.
- **DO** pakai dua model **beda famili** untuk disagreement (Qwen + Llama).
- **DO** constrain format saja; biarkan reasoning bebas sebelum blok JSON.
- **DO** skema JSON flat (objek/enum/required); validasi semantik argumen + retry.
- **DO** log JSON (`--log-format json`) sejak awal; parse field, bukan regex prosa.
- **DO** ukur dan simpan angka: akurasi NL→plan, false-refuse/false-act rate,
  waktu expose→live — semua jadi amunisi slide.
- **DO** rekam video fallback dari run sungguhan + siapkan jalur Quick Tunnel.
- **DO** pin versi: `cloudflared --version`, tag model Ollama, ROCm — tulis di README.
- **DO** commit tanpa atribusi AI apa pun (aturan repo).

### DO NOT
- **DO NOT** biarkan LLM memanggil Cloudflare API / shell langsung — hanya typed tools.
- **DO NOT** pakai ReAct / loop tool panjang di model <14B — plan-then-execute, ≤3 langkah.
- **DO NOT** hard-constrain seluruh output (constraint tax menurunkan akurasi).
- **DO NOT** pakai `oneOf`/`anyOf` campur `properties`, nested `$ref`, `if/then`
  di skema — konverter GBNF skip diam-diam.
- **DO NOT** percaya self-confidence model sebagai gate — hanya disagreement + blast
  radius (temuan kalibrasi track1; SAC³).
- **DO NOT** pakai 521/523 sebagai sinyal origin-down di tunnel — origin mati = **502**;
  1033 = connector mati. Jangan tertukar.
- **DO NOT** hapus tunnel tanpa stop connector + DELETE connections dulu (400).
- **DO NOT** lupa rule catch-all `http_status:404` di akhir ingress (config ditolak).
- **DO NOT** `keep_alive: "0"` (string) untuk pin — harus numerik `-1`.
- **DO NOT** perlebar scope: Access policy penuh, fleet/multi-tunnel, multi-user =
  slide roadmap, bukan kode.
- **DO NOT** demo di domain/akun produksi, atau simpan token di repo (pakai `.env`,
  gitignore sejak commit pertama).
- **DO NOT** klaim "AI mengelola infra" tanpa menunjukkan gate refuse bekerja live —
  itu justru pembedanya.

---

## 10. Risiko & mitigasi

| Risiko | Mitigasi |
|---|---|
| PUT configurations tolak api_token (⚠️ #23461) | tes hari pertama; permission Tunnel Write; fallback: config lokal YAML (locally-managed) |
| Internet/akun rewel saat live demo | video fallback dari run sama + jalur Quick Tunnel |
| Latensi model lokal terasa lambat di panggung | planner 7B cepat; verifier hanya dipanggil untuk aksi ≥mutating; streaming narasi |
| Model kecil omission (jawab prosa) | retry + reminder, few-shot in-distribution, validasi + assert di CI Fase 2 |
| Judge/juri skeptis "AI-nya di mana?" | babak 3 (refuse) + slide data kalibrasi track1 + angka false-act/false-refuse |
| ROCm driver mismatch (GPU tak kedetek) | pin versi ROCm 7.x + driver host `amdgpu-install`; cek `ollama ps` sebelum demo |

---

## 11. Struktur repo target (repo terpisah, bukan monorepo — aturan track)

```
track3/            # scaffold di sini dulu, lalu dipindah ke repo sendiri
  agent/
    tools/         # typed tools: cloudflare.py, connector.py, probe.py, dns.py
    heal/          # watchdog.py, taxonomy.py (decision core §5)
    brain/         # planner.py, verifier.py, gate.py (matriks §2), schemas.py
    journal.py     # audit log + undo stack
    cli.py         # expose/teardown/watch/chat
  eval/            # tes berlabel NL→plan + chaos tests (reuse pola grader track1)
  ui/              # overlay tokens/s + VRAM + jurnal (Fase 4)
  plan.md          # dokumen ini
```
