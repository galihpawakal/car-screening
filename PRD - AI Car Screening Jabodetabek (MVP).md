# PRD: AI Car Screening Jabodetabek (MVP)

Versi 0.1 | Status: Draft | Tujuan: prototipe sederhana untuk uji coba

## 1. Ringkasan

AI Car Screening adalah aplikasi web sederhana yang membantu calon pembeli menilai apakah sebuah listing mobil bekas di Jabodetabek layak dibeli. Pengguna memasukkan data satu mobil (harga, merek, model, tahun, kilometer, lokasi, deskripsi penjual). Sistem lalu memberi estimasi harga wajar, skor screening, red flag, dan ringkasan dari AI.

MVP ini sengaja dibuat kecil: input manual, dataset awal kecil, satu model harga sederhana, dan satu pemanggilan LLM untuk membaca deskripsi.

## 2. Masalah

- Pembeli sulit tahu apakah harga listing wajar karena harga di marketplace bervariasi.
- Deskripsi penjual panjang, tidak seragam, dan kadang menyembunyikan informasi penting (pajak mati, bekas banjir, KM tidak wajar).
- Membandingkan banyak listing secara manual memakan waktu.

## 3. Tujuan dan Non-Tujuan

**Tujuan MVP**

1. Pengguna mendapat estimasi rentang harga wajar dalam waktu kurang dari 10 detik.
2. Pengguna mendapat skor screening 0-100 beserta alasan yang mudah dipahami.
3. Deskripsi penjual diringkas dan diubah menjadi poin terstruktur oleh AI.
4. Membuktikan bahwa alur end-to-end (input, model, AI, hasil) berjalan.

**Non-Tujuan (belum dikerjakan)**

- Scraping otomatis dari marketplace
- Akun pengguna, login, dan pembayaran
- Analisis foto mobil
- Cakupan di luar Jabodetabek
- Mobil baru, motor, dan kendaraan niaga

## 4. Target Pengguna

| Persona | Kebutuhan |
| --- | --- |
| Pembeli pribadi | Memastikan harga wajar dan menghindari mobil bermasalah |
| Penjual pribadi | Mengetahui kisaran harga jual yang realistis |
| Pemilik showroom kecil (opsional) | Cek cepat harga pasar sebelum membeli unit |

## 5. User Stories

1. Sebagai pembeli, saya ingin memasukkan data mobil agar tahu harga wajarnya.
2. Sebagai pembeli, saya ingin melihat apakah harga listing terlalu murah, wajar, atau terlalu mahal.
3. Sebagai pembeli, saya ingin AI menandai hal mencurigakan di deskripsi penjual.
4. Sebagai pembeli, saya ingin melihat beberapa mobil pembanding di wilayah yang sama.
5. Sebagai pengguna, saya ingin hasil screening yang bisa disalin atau dibagikan.

## 6. Fitur MVP

### 6.1 Form input listing

| Field | Tipe | Wajib |
| --- | --- | --- |
| Merek | Dropdown | Ya |
| Model | Dropdown (bergantung merek) | Ya |
| Varian/tipe | Teks atau dropdown | Ya |
| Jenis bodi | MPV, SUV, Sedan, Hatchback, dll | Ya |
| Tahun | Angka | Ya |
| Transmisi | AT / MT | Ya |
| Kilometer | Angka | Ya |
| Harga listing (Rp) | Angka | Ya |
| Lokasi | Jakarta, Bogor, Depok, Tangerang, Bekasi | Ya |
| Plat nomor | B, F, D, T, lainnya | Tidak |
| Deskripsi penjual | Teks panjang | Tidak |

### 6.2 Estimasi harga wajar

- Keluaran: harga rendah, tengah, dan tinggi (rentang), plus jumlah data pembanding yang dipakai.
- Selisih harga listing terhadap harga tengah ditampilkan dalam persen.
- Label: Murah mencurigakan (lebih dari 20% di bawah), Bagus (5-20% di bawah), Wajar (dalam +/-5%), Mahal (lebih dari 5% di atas).

### 6.3 Analisis deskripsi dengan AI (LLM)

LLM membaca deskripsi dan menghasilkan JSON terstruktur:

- pajak\_hidup, service\_record, tangan\_pertama, klaim\_bebas\_banjir, kondisi\_mesin, kelengkapan\_surat
- ringkasan (maksimal 3 kalimat)
- red\_flag (daftar teks singkat)
- pertanyaan\_untuk\_penjual (3-5 pertanyaan yang sebaiknya ditanyakan pembeli)

### 6.4 Deteksi red flag berbasis aturan

- Harga lebih dari 25% di bawah estimasi
- KM per tahun di bawah 3.000 atau di atas 30.000
- Tahun di luar rentang wajar untuk model tersebut
- Deskripsi kosong atau sangat pendek
- Kata kunci: "BU", "butuh uang", "DP dulu", "transfer dulu", "tanpa surat"

### 6.5 Skor screening (0-100)

| Komponen | Bobot |
| --- | --- |
| Kewajaran harga | 40% |
| Kewajaran KM terhadap umur | 20% |
| Kelengkapan informasi | 20% |
| Risiko dari deskripsi (red flag AI dan aturan) | 20% |

Hasil dibagi tiga kategori: 75-100 Layak dipertimbangkan, 50-74 Perlu cek lebih lanjut, di bawah 50 Berisiko tinggi. Setiap skor wajib disertai alasan dalam bahasa sederhana.

### 6.6 Mobil pembanding

Tampilkan 5 listing paling mirip dari dataset (model sama, tahun +/-2, wilayah sama jika tersedia) dengan harga dan KM.

## 7. Alur Pengguna

1. Buka aplikasi dan isi form listing.
2. Tekan tombol Analisa.
3. Sistem menghitung estimasi harga, menjalankan aturan red flag, dan memanggil LLM.
4. Halaman hasil menampilkan: skor, label harga, rentang harga wajar, ringkasan AI, daftar red flag, pertanyaan untuk penjual, dan 5 pembanding.
5. Pengguna menyalin hasil atau mengisi listing lain.

## 8. Data

**Dataset awal (MVP)**

- Target 500-1.500 listing untuk 8-10 model populer (misal Avanza, Xenia, Brio, Mobilio, Rush, Innova, Fortuner, Jazz, Agya, Ayla).
- Sumber: input manual dari marketplace, atau data yang diperoleh secara sah. Periksa ketentuan penggunaan tiap situs sebelum mengambil data otomatis.
- Format: file CSV dengan kolom sesuai form di atas ditambah tanggal pengambilan data.

**Pembersihan minimum**

- Hapus duplikat, harga tidak wajar, dan data tanpa tahun atau KM
- Samakan penulisan merek, model, dan varian

## 9. Spesifikasi AI

| Komponen | Pendekatan MVP |
| --- | --- |
| Model harga | LightGBM atau XGBoost, dengan quantile (10%, 50%, 90%) untuk rentang. Cadangan: regresi sederhana jika data terlalu sedikit |
| Fitur | model, varian, tahun, KM, transmisi, lokasi, jenis bodi |
| Ekstraksi deskripsi | Satu pemanggilan LLM dengan prompt yang memaksa keluaran JSON valid |
| Penanganan gagal | Jika LLM error, tetap tampilkan hasil harga dan aturan tanpa ringkasan AI |

Catatan: hasil adalah estimasi, bukan penilaian pasti. Layar hasil harus menampilkan disclaimer dan tanggal pembaruan data.

## 10. Arsitektur Teknis (Sederhana)

| Lapisan | Pilihan |
| --- | --- |
| Antarmuka | Streamlit (tercepat) atau Next.js |
| Backend | FastAPI (Python) |
| Data | CSV, lalu SQLite atau PostgreSQL |
| Model | scikit-learn dan LightGBM, disimpan sebagai file model |
| LLM | Claude API |

Alur: Form, FastAPI, (model harga + aturan + LLM), halaman hasil.

## 11. Metrik Keberhasilan

| Metrik | Target MVP |
| --- | --- |
| Error estimasi harga (MAPE) | Di bawah 15% untuk model populer |
| Waktu respons | Di bawah 10 detik |
| JSON LLM valid | Di atas 95% |
| Kepuasan penguji (5-10 orang) | Hasil dinilai berguna oleh minimal 70% |

## 12. Risiko dan Mitigasi

| Risiko | Mitigasi |
| --- | --- |
| Data terlalu sedikit sehingga model tidak akurat | Fokus pada model populer, tampilkan jumlah pembanding dan tingkat keyakinan |
| Harga listing berbeda dari harga transaksi | Beri catatan bahwa listing umumnya 3-10% di atas harga jadi |
| LLM salah membaca deskripsi | Tampilkan ringkasan sebagai bantuan, bukan kepastian |
| Pelanggaran ketentuan situs sumber data | Gunakan input manual atau sumber yang sah |
| Pengguna menganggap hasil sebagai jaminan | Tampilkan disclaimer dan saran cek fisik serta bengkel |

## 13. Rencana Kerja (2 Minggu)

| Hari | Pekerjaan |
| --- | --- |
| 1-3 | Finalisasi skema, kumpulkan dan bersihkan dataset awal |
| 4-6 | Latih model harga, evaluasi MAPE |
| 7-8 | Buat prompt dan ekstraksi deskripsi dengan LLM |
| 9-10 | Aturan red flag dan skor screening |
| 11-12 | Bangun UI form dan halaman hasil |
| 13-14 | Uji dengan 5-10 pengguna, perbaikan, dokumentasi |

## 14. Pertanyaan Terbuka

1. Platform awal: Streamlit (cepat) atau Next.js (lebih rapi)?
2. Dari mana data awal diperoleh?
3. Model mobil apa yang diprioritaskan dulu?
4. Apakah perlu fitur tempel link listing di versi berikutnya?
