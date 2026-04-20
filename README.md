# Credit Intelligence — Sistem Penentu Kelayakan Kredit
**UTS Soft Computing — Universitas Padjadjaran 2026**

> Perbandingan tiga pendekatan sistem cerdas dalam memprediksi skor kelayakan kredit: Manual FIS (Mamdani), GA-FIS (Evolutionary Tuning), dan ANN (Backpropagation).

**Disusun oleh:**
| Nama | NIM |
|---|---|
| Senia Nur Hasanah | 140810230021 |
| Audrey Shaina Tjandra | 140810230026 |
| Siti Nailah Eko Putri Alawiyah | 140810230059 |

Program Studi S-1 Teknik Informatika — FMIPA Universitas Padjadjaran

---

## Deskripsi Proyek

Proyek ini membandingkan tiga metode Soft Computing untuk memprediksi **skor kelayakan kredit** (skala 0–100) berdasarkan 5 fitur input: pendapatan bulanan, jumlah pinjaman, lama bekerja, riwayat kredit, dan rasio hutang.

| Tahap | Metode | Deskripsi |
|---|---|---|
| 1 | **Manual FIS** | Fuzzy Inference System Mamdani yang dibangun menggunakan logika/intuisi kelompok |
| 2 | **GA-FIS** | Parameter Membership Function dari Tahap 1 dioptimasi menggunakan Algoritma Genetika |
| 3 | **ANN** | Artificial Neural Network (5→32→16→1) yang dilatih end-to-end dari data |

### Hasil Performa (Test Set, n=100)

| Metode | RMSE ↓ | MAE ↓ | R² ↑ |
|---|---|---|---|
| Manual FIS | 11.1644 | 9.0259 | 0.6191 |
| GA-FIS | 10.8852 | 8.7774 | 0.6380 |
| **ANN** | **9.1175** | **7.2333** | **0.7460** |

ANN unggul di seluruh metrik dengan peningkatan RMSE sebesar **18.3%** dibanding Manual FIS.

---

## Struktur Repositori

```
.
├── app_streamlit.py       # Aplikasi web interaktif (Streamlit)
├── dataset_kredit.csv     # Dataset sintetis 500 baris
├── fis_manual.pkl         # Parameter FIS Manual (hasil Tahap 1)
├── ga_params.pkl          # Parameter GA-FIS (hasil Tahap 2)
├── ann_model.pkl          # Bobot model ANN (hasil Tahap 3)
├── metrics.pkl            # Metrik evaluasi ketiga model
├── requirements.txt       # Daftar library yang dibutuhkan
└── README.md
```

> **Kode training lengkap** tersedia di notebook Google Colab (tidak disertakan di repo ini karena menggunakan `google.colab` untuk upload file).

---

## Persyaratan Sistem

- Python 3.8 atau lebih baru
- pip

Library yang dibutuhkan (lihat `requirements.txt`):

```
streamlit
numpy
pandas
matplotlib
scikit-learn
```

---

## Instalasi & Menjalankan Aplikasi

### 1. Clone repositori

```bash
git clone https://github.com/eoxxax/UTS-Softcom-Kelayakan-Kredit-230021-230026-230059.git
cd UTS-Softcom-Kelayakan-Kredit-230021-230026-230059
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan aplikasi

```bash
streamlit run app_streamlit.py
```

Aplikasi akan otomatis terbuka di browser pada `http://localhost:8501`.

> **Pastikan** file `fis_manual.pkl`, `ga_params.pkl`, `ann_model.pkl`, dan `metrics.pkl` berada di folder yang sama dengan `app_streamlit.py`.

---

## Panduan Penggunaan Aplikasi

Aplikasi memiliki **2 tab utama**:

### Tab 1 — Prediksi Debitur

Digunakan untuk menilai kelayakan kredit seorang calon debitur.

1. Isi input di sidebar kiri:
   - **Pendapatan Bulanan (Rp)** — rentang Rp 2.000.000 s.d. Rp 40.000.000
   - **Jumlah Pinjaman (Rp)** — rentang Rp 5.000.000 s.d. Rp 180.000.000
   - **Lama Bekerja (tahun)** — slider 0.5 s.d. 30 tahun
   - **Riwayat Kredit (0–100)** — slider skor riwayat kredit
   - **Rasio Hutang** — slider 0.05 s.d. 0.90
2. Klik tombol **"Analisis Kelayakan Kredit"**
3. Aplikasi menampilkan skor dan verdict dari ketiga metode:
   - `LAYAK` — skor ≥ 65
   - `PERTIMBANGKAN` — skor 40–64
   - `TIDAK LAYAK` — skor < 40
4. Ditampilkan juga **konsensus** dari ketiga metode sebagai rekomendasi akhir.

### Tab 2 — Performa Model

Menampilkan perbandingan metrik evaluasi (RMSE, MAE, R²) ketiga metode dalam bentuk tabel dan visualisasi.

---

## Detail Metode

### Tahap 1: Manual FIS (Mamdani)
- 5 variabel input, masing-masing 3 kelas linguistik (total 243 rules)
- Rules dibangkitkan otomatis via Cartesian product dengan sistem pembobotan
- Defuzzifikasi menggunakan metode centroid

### Tahap 2: GA-FIS
- Mengoptimasi 15 parameter titik tengah MF (3 label × 5 variabel)
- Population size: 30 | Generasi: 50 | Mutation rate: 0.15 | Crossover rate: 0.80
- Seleksi turnamen (ukuran 4) | Fitness: minimize RMSE pada train set

### Tahap 3: ANN Backpropagation
- Arsitektur: 5 → 32 → 16 → 1
- Aktivasi: ReLU (hidden), Sigmoid (output)
- Learning rate: 0.008 | Epochs: 300 | Batch size: 32
- Inisialisasi bobot: He initialization | Optimizer: SGD manual (NumPy)

---

## Dataset

Dataset sintetis (`dataset_kredit.csv`) berisi **500 baris** dengan 7 kolom:

| Kolom | Keterangan |
|---|---|
| `pendapatan_bulanan` | Pendapatan bulanan debitur (Rp) |
| `jumlah_pinjaman` | Jumlah pinjaman yang diminta (Rp) |
| `lama_bekerja_tahun` | Lama bekerja dalam tahun |
| `riwayat_kredit_skor` | Skor riwayat kredit (0–100) |
| `rasio_hutang` | Rasio hutang terhadap pendapatan (0–1) |
| `skor_kelayakan` | Target: skor kelayakan numerik (5.0–87.3) |
| `label_kelayakan` | Target: Layak / Pertimbangkan / Tidak Layak |

Split: 80% training (400 baris) / 20% testing (100 baris), stratified.

---

## Mata Kuliah

**Soft Computing** — UTS Genap 2025/2026  
Dosen: Dr. Ir. Intan Nurma Yulita, M.T  
Fakultas MIPA — Teknik Informatika
Universitas Padjadjaran

