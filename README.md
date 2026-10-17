# AI Car Screening Jabodetabek

Aplikasi skrining dan penilaian kewajaran harga mobil bekas berbasis AI dan data pasar aktual di wilayah Jabodetabek. Memadukan **pencarian katalog marketplace langsung** (bebas kuota AI) dan **Google Search Grounding (Gemini AI)** resmi untuk menemukan listing pembanding nyata yang tervalidasi. 

> [!NOTE]
> Sistem ini **100% menggunakan data listing nyata**. Tidak ada fallback ke data CSV tiruan (dummy), model prediksi buatan, maupun angka pasar rekayasa.

---

## 🚀 Fitur Utama

### 1. Dual-Engine Pencarian Pasar
* **Pencarian Langsung Marketplace (Default)**: Merayapi katalog resmi marketplace terkemuka di Indonesia (Carmudi, Mobil123, Carsome, OLX, Oto) secara otomatis dengan filter tahun, varian, dan transmisi tanpa memakan kuota Gemini AI.
* **Google Search Grounding (Gemini AI)**: Pencarian multi-round cerdas via SDK `google-genai` resmi dengan grounding Google Search untuk melacak listing terbaru di web.
* **Parallel & Resilient Fallback**: Bila grounding AI menemui limit kuota atau timeout, sistem otomatis melakukan fallback mulus ke pencarian katalog marketplace langsung.

### 2. Tiered Screening & Model Kembar (*Twin Models*)
* **Tier 1 (Bobot 1.0)**: Varian sama persis dan tahun yang identik.
* **Tier 2 (Bobot 0.8)**: Model sama dengan rentang tahun wajar (±1-2 tahun).
* **Tier 3 (Bobot 0.6)**: Model kembar sekelas (e.g. Avanza ↔ Xenia, Rush ↔ Terios, Agya ↔ Ayla, Calya ↔ Sigra).
* **Akuntansi Listing Transparan**: Melacak seluruh riwayat listing yang dibuang beserta alasan spesifik (varian berbeda, tahun di luar rentang, lokasi non-Jabodetabek, transmisi berbeda, atau harga tidak wajar).

### 3. Analisis Harga & Kewajaran Negosiasi
* **Distribusi Statistik Nyata**: Menghitung kuartil P25, Median, dan P75, serta persentil posisi harga unit terhadap pasar.
* **Deteksi Harga Anomali**: Jika selisih harga >50% di bawah/di atas pasar, sistem secara otomatis mengidentifikasi kemungkinan salah ketik, harga jebakan DP kredit, atau penipuan, serta mematikan saran tawar agar pembeli tidak tertipu.
* **Saran Negosiasi 3 Tahap**: Memberikan taksiran *Tawar Awal*, *Target Realistis*, dan *Batas Maksimal* berdasarkan koreksi realistis harga listing ke harga transaksi (3-10%).
* **Analisis Odometer (KM/Tahun)**: Membandingkan jarak tempuh unit terhadap rata-rata pembanding seumuran untuk mendeteksi potensi pemutaran odometer (*rollback*).

### 4. Empat Tab Antarmuka Pengguna (UI)
* **🔍 Tab 1: Analisa Unit**: Form input spesifikasi mobil, visualisasi sebaran harga (Vega-Lite), checklist inspeksi fisik & dokumen, deteksi red flag, serta tombol *Salin Ringkasan* dan *Unduh TXT*.
* **📋 Tab 2: Riwayat Analisa**: Seluruh riwayat screening tersimpan rapi di SQLite lokal dengan rincian skor dan kesimpulan.
* **🏪 Tab 3: Database Listing Pasar**: Database terintegrasi berisi seluruh listing marketplace yang pernah diambil, dilengkapi pencarian dan fitur ekspor CSV.
* **🩺 Tab 4: Diagnostik Lingkungan**: Panel pemantauan sistem yang menampilkan hash Git, waktu server (UTC/WIB), versi Python & paket, model AI & jumlah API key, statistik database SQLite, serta uji konektivitas ringan ke domain marketplace untuk mendeteksi blokir jaringan.

### 5. Keamanan & Nol Kebocoran Kunci (*Zero Secret Leakage*)
* Nilai API key, prefix, dan suffix tidak pernah ditampilkan di antarmuka web, file diagnostik, maupun log sistem.
* Seluruh error dan traceback secara otomatis disaring melalui fungsi *masking*.
* Database SQLite disimpan lokal di `data/analysis_cache.sqlite3` dan diabaikan dari pelacakan Git.

---

## 📁 Struktur Repositori

```text
car-screening/
├── app.py                      # Titik masuk utama aplikasi Streamlit
├── requirements.txt            # Dependensi Python terkunci (pinned)
├── .env.example                # Template konfigurasi environment variables
├── README.md                   # Dokumentasi lengkap proyek
├── data/                       # Direktori database SQLite lokal (gitignored)
│   └── analysis_cache.sqlite3
├── src/                        # Kode sumber modular aplikasi
│   ├── config.py               # Parameter mobil, bobot skor, ambang batas, daftar domain
│   ├── pipeline.py             # Orchestrator alur screening 4 tahap
│   ├── search_provider.py      # Provider Gemini AI, Search Grounding, dan rotasi API key
│   ├── marketplace_config.py   # Konfigurasi slug katalog dan pemetaan model kembar
│   ├── marketplace_fetcher.py  # Scraper katalog marketplace berkecepatan tinggi
│   ├── page_reader.py          # Pembaca halaman detail, kepatuhan robots.txt, parser JSON-LD
│   ├── detail_parser.py        # Parser detail spesifik per domain marketplace
│   ├── url_classifier.py       # Klasifikasi tautan (katalog vs halaman detail)
│   ├── url_resolve.py          # Resolusi Google grounding redirect URL
│   ├── tiered_screening.py     # Logika validasi bertingkat (Tier 1-3) & model kembar
│   ├── scoring.py              # Algoritma skor kewajaran harga, KM, dan keyakinan pasar
│   ├── stats.py                # Perhitungan P25, Median, P75, dan persentil pasar
│   ├── rules.py                # Pemeriksaan red flag berbasis aturan deterministik
│   ├── clean.py                # Pembersihan, validasi, dan deduplikasi listing
│   ├── extract.py              # Skema ekstraksi data listing berbasis Pydantic
│   ├── analyze.py              # Analisis akhir, checklist inspeksi, dan saran negosiasi
│   ├── cache.py                # SQLite cache manager, TTL 24 jam, kuota harian, log token
│   ├── diagnostics.py          # Diagnostik lingkungan, versi sistem, uji konektivitas domain
│   └── ui.py                   # Antarmuka Streamlit multi-tab dan visualisasi
└── tests/                      # Rangkaian pengujian otomatis (Unit & E2E Tests)
    ├── test_diagnostics.py     # Tes diagnostik sistem & perlindungan privasi API key
    ├── test_live_pipeline.py   # Tes pipeline end-to-end 4 tahap & simulasi fallback
    ├── test_tahap3_fixes.py    # Tes verifikasi perbaikan tahap 3
    ├── test_tahap4_fixes.py    # Tes verifikasi perbaikan tahap 4
    └── ...
```

---

## 🛠️ Panduan Instalasi & Menjalankan Lokal

### Prasyarat
* Python **3.11** atau **3.12**
* Google Gemini API Key (dapat diperoleh via [Google AI Studio](https://aistudio.google.com/))

### Langkah Menjalankan

1. **Clone repositori dan siapkan Virtual Environment**:
   ```bash
   git clone https://github.com/galihpawakal/car-screening.git
   cd car-screening

   python -m venv venv
   # Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   # macOS / Linux:
   source venv/bin/activate
   ```

2. **Pasang Dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Atur Variabel Lingkungan**:
   Salin `.env.example` menjadi `.env`:
   ```bash
   # Windows:
   copy .env.example .env
   # Linux / macOS:
   cp .env.example .env
   ```

   Buka `.env` dan lengkapi konfigurasinya:
   ```env
   # Masukkan satu atau beberapa API key (dipisahkan koma untuk rotasi otomatis)
   GEMINI_API_KEY=AIzaSy...
   GEMINI_API_KEYS=AIzaSy...,AQ.Ab...

   # Model Gemini yang mendukung Search Grounding
   GEMINI_MODEL=gemini-3.8-flash
   GEMINI_FALLBACK_MODELS=gemini-3.6-flash,gemini-3.5-flash-lite

   # Batas analisa harian lokal (mencegah lonjakan kuota)
   DAILY_ANALYSIS_LIMIT=20
   ```

4. **Jalankan Aplikasi Streamlit**:
   ```bash
   streamlit run app.py
   ```
   Buka peramban Anda di `http://localhost:8501`.

---

## ☁️ Panduan Deploy ke Streamlit Community Cloud

Saat mendeploy repositori ini ke **[Streamlit Community Cloud](https://share.streamlit.io/)**:

1. Pilih repositori, cabang `main`, dan set *Main file path* ke `app.py`.
2. Pada bagian **Advanced settings** -> **Secrets**, masukkan konfigurasi dalam format TOML:
   ```toml
   GEMINI_API_KEY = "kunci-gemini-anda"
   GEMINI_MODEL = "gemini-3.8-flash"
   GEMINI_FALLBACK_MODELS = "gemini-3.6-flash,gemini-3.5-flash-lite"
   DAILY_ANALYSIS_LIMIT = "20"
   ```
3. Tekan **Deploy**. Aplikasi akan otomatis membaca dependensi dari `requirements.txt` tanpa memerlukan paket apt Linux tambahan.

---

## 🧪 Pengujian Otomatis (*Testing*)

Seluruh unit test dirancang mandiri tanpa ketergantungan koneksi jaringan luar saat pengujian (respons grounding dan HTML marketplace menggunakan mock/fixture):

Jalankan seluruh tes pytest:
```powershell
python -m pytest
```

Jalankan pengujian panel diagnostik dan keamanan API key:
```powershell
python -m pytest tests/test_diagnostics.py
```

Jalankan linting kode:
```powershell
ruff check --select F821,F841 src/ tests/
```

---

## ⚖️ Disclaimer

* Estimasi harga dan saran negosiasi dihitung berdasarkan data listing web aktif, bukan data riil transaksi faktual notaris/kantor Samsat.
* Asumsi diskon 3-10% adalah kebiasaan tawar-menawar umum di pasar mobil bekas Indonesia.
* **Wajib** lakukan inspeksi fisik, cek nomor rangka & mesin, serta verifikasi keabsahan STNK/BPKB di Samsat sebelum melakukan transaksi uang muka (DP) atau pelunasan.