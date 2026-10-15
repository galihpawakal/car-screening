# AI Car Screening Jabodetabek

Aplikasi screening mobil bekas yang mencari listing pembanding nyata melalui Gemini Grounding with Google Search. Tidak ada fallback ke CSV dummy, model harga, atau angka pasar buatan.

## Fitur

- Pencarian grounded, ekstraksi Pydantic, pembersihan deterministik, dan analisa akhir dalam empat tahap terpisah.
- Rentang harga dari P25, median, dan P75 listing tervalidasi.
- Skor 0-100 berbobot 40/20/20/20, red flag aturan dan AI yang dideduplikasi, serta alasan komponen.
- Tabel sumber yang bisa dibuka, tanggal ambil, distribusi harga, KM/tahun, negosiasi estimasi, confidence, dan checklist inspeksi.
- Cache SQLite TTL 24 jam, batas analisa harian, dan log panggilan grounding/token.

## Menjalankan Lokal

Prasyarat: Python 3.11+ dan Gemini API key dengan akses Gemini API.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env
```

Isi `GEMINI_API_KEY` secara lokal di `.env`, atau atur lewat Streamlit secrets. Jangan commit `.env` atau `secrets.toml`. `GEMINI_MODEL` harus mendukung Google Search; contoh template menggunakan `gemini-3.8-flash`, yang tercantum pada dokumentasi model grounding Google. `DAILY_ANALYSIS_LIMIT` mengatur kuota lokal per hari.

```bash
streamlit run app.py
```

Buka `http://localhost:8501`.

Untuk Streamlit Community Cloud, atur secrets:

```toml
GEMINI_API_KEY = "isi-melalui-dashboard-secrets"
GEMINI_MODEL = "gemini-3.8-flash"
DAILY_ANALYSIS_LIMIT = "10"
```

## Privasi dan Data

- Web search diperlakukan sebagai data tidak tepercaya; instruksi yang ditemukan di halaman diabaikan.
- Teks hasil search hanya diproses sementara untuk ekstraksi. SQLite menyimpan fakta hasil normalisasi, URL sumber, kueri grounding, tanggal ambil, alasan penolakan, statistik, kuota, dan token; deskripsi penjual/foto tidak disimpan di cache.
- Listing tanpa harga, model yang tidak cocok, nilai tidak masuk akal, URL yang tidak ada di grounding citations, serta duplikat ditolak beserta alasannya.
- Kurang dari lima pembanding memunculkan peringatan keras dan confidence Rendah. Tanpa listing valid, harga tidak diestimasi.
- Database cache berada di `data/analysis_cache.sqlite3` dan di-ignore Git. Cache kedaluwarsa setelah 24 jam.
- Pastikan penggunaan Google Search dan penayangan sumber mematuhi ketentuan Google serta situs sumber listing.

## Struktur Alur

```text
app.py -> src/ui.py -> src/pipeline.py
  1. src/search_provider.py  Gemini Google Search grounding
  2. src/extract.py          Ekstraksi JSON berskema Pydantic
  3. src/clean.py            Normalisasi, validasi, deduplikasi
      src/stats.py            P25/median/P75, posisi, KM/tahun
      src/cache.py            SQLite TTL, kuota, log token
  4. src/analyze.py          Kesimpulan terstruktur tanpa tools
      src/rules.py            Red flag berbasis aturan
      src/scoring.py          Skor dan confidence deterministik
```

## Konfigurasi

| Variabel | Wajib | Keterangan |
| --- | --- | --- |
| `GEMINI_API_KEY` | Ya | Dari environment atau Streamlit secrets; jangan simpan di repo. |
| `GEMINI_MODEL` | Ya | Harus model yang mendukung Google Search grounding. |
| `DAILY_ANALYSIS_LIMIT` | Tidak | Batas harian lokal; default 10. |

Model dipilih dari `GEMINI_MODEL`, tanpa default tersembunyi. Aplikasi menolak nama model yang tidak tercantum pada daftar dukungan Search provider.

## Tes

Semua tes tidak memanggil jaringan; respons Gemini diganti fixture/mock.

```bash
python -m pytest -q
```

## Referensi Resmi

- [Google Search grounding](https://ai.google.dev/gemini-api/docs/google-search)
- [Structured outputs](https://ai.google.dev/gemini-api/docs/structured-output)
- [Daftar model Gemini](https://ai.google.dev/gemini-api/docs/models/gemini)
- [SDK resmi google-genai](https://github.com/googleapis/python-genai)

## Disclaimer

Estimasi memakai harga listing dari web, bukan harga transaksi atau penilaian kondisi mobil. Selisih transaksi 3-10% adalah asumsi umum untuk negosiasi, bukan data transaksi terverifikasi. Selalu cek dokumen, lakukan inspeksi fisik, dan gunakan bengkel independen sebelum membeli.