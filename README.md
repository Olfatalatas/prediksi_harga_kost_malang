# 🏠 AI Estimasi Harga Kost Malang

Sistem prediksi harga sewa kost di Malang menggunakan Machine Learning dengan akurasi tinggi. Proyek ini mendemonstrasikan **end-to-end pipeline data science** dari data acquisition hingga deployment—sesuai standar industri untuk proyek ML production-ready.

---

## 📋 Daftar Isi

- [Overview](#-overview)
- [Fitur Utama](#-fitur-utama)
- [Arsitektur Proyek](#-arsitektur-proyek)
- [Stack Teknologi](#-stack-teknologi)
- [Requirement & Setup](#️-requirement--setup)
- [Panduan Penggunaan](#-panduan-penggunaan)
- [Pipeline Data Science](#-pipeline-data-science)
- [Model Performance](#-model-performance)
- [Design Decisions](#-design-decisions)
- [Reproducibility](#-reproducibility)
- [Etika & Legal](#-etika--legal)
- [Struktur Direktori](#-struktur-direktori)
- [Troubleshooting](#-troubleshooting)
- [Referensi](#-referensi)

---

## 📊 Overview

Proyek ini mengimplementasikan **predictive pricing model** untuk kost (akomodasi berpenghuni) di kota Malang, Jawa Timur. Dengan menganalisis **6.000+** data point dari [Mamikos](https://mamikos.com/), model dapat memprediksi harga sewa bulanan berdasarkan:

| Faktor | Deskripsi |
|--------|-----------|
| **Lokasi** | Kecamatan di Malang (one-hot encoded) |
| **Jenis Kost** | Putra / Putri / Campur |
| **Fasilitas** | AC, WiFi, Kamar Mandi Dalam, Kloset Duduk, Kasur, Akses 24 Jam (binary) |

**Deliverable:**

- ✅ Model terpilih: **Random Forest** (R² ≈ 0.95, MAE ≈ Rp 50k)
- ✅ Web interface interaktif berbasis **Streamlit**
- ✅ Laporan komparasi model & grafik evaluasi di `hasil_evaluasi/`
- ✅ Pipeline yang dapat dijalankan ulang (reproducible) dengan random seed tetap

---

## 🎯 Fitur Utama

| Modul | Fungsi |
|-------|--------|
| **Web Scraping** | Mengumpulkan data dari Mamikos; handle dynamic loading (tombol "Lihat lagi") dengan Selenium |
| **Data Cleaning** | Normalisasi harga, parsing fasilitas → binary features, standardisasi nama kecamatan |
| **EDA** | Distribusi harga, boxplot fasilitas vs harga, heatmap korelasi |
| **Training** | Linear Regression vs Random Forest; tuning RF dengan **Optuna** (15 trials, 3-fold CV) |
| **Web App** | Input lokasi + fasilitas → estimasi harga real-time (Streamlit) |

---

## 🏗️ Arsitektur Proyek

```
┌─────────────────────────────────────────────────────────────┐
│  DATA ACQUISITION                                            │
│  scrape_malang.py → Selenium + BeautifulSoup (Mamikos)       │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  PREPROCESSING                                               │
│  clean_data.py → Harga, Fasilitas, Daerah → ML-ready CSV     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  EDA                                                         │
│  eda_check.py → Distribusi, Fasilitas, Korelasi → PNG        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  TRAINING & EVALUATION                                       │
│  train_model.py → LR vs RF (Optuna) → pemenang + laporan     │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  SERVING                                                     │
│  app.py → Streamlit UI, load model_kost_terbaik.pkl          │
└─────────────────────────────────────────────────────────────┘
```

---

## 💻 Stack Teknologi

| Kategori | Library | Versi Min. |
|----------|---------|------------|
| Data & ML | pandas, numpy, scikit-learn, joblib | 1.3, 1.21, 0.24, 1.0 |
| Visualisasi | matplotlib, seaborn | 3.4, 0.11 |
| Web App | streamlit, altair | 1.24, &lt;5 |
| Scraping | selenium, beautifulsoup4, webdriver-manager | 4.0, 4.9, 3.8 |
| Tuning | optuna | 2.10 |

---

## ⚙️ Requirement & Setup

### Prasyarat

- **Python** 3.8+
- **Chrome** (untuk scraping; driver diatur otomatis via `webdriver-manager`)

### Instalasi

```bash
# Clone
git clone https://github.com/Olfatalatas/prediksi_harga_kost_malang.git
cd prediksi_harga_kost_malang  # atau projek_kost_malang

# Virtual environment (disarankan)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verifikasi

```bash
python -c "import pandas, sklearn, streamlit, optuna; print('OK')"
```

### Working Directory

Jalankan semua script dari **root proyek** agar path ke `data/`, `hasil_eda/`, dan `hasil_evaluasi/` konsisten. Beberapa script punya fallback baca dari `data/` jika file tidak ada di root.

---

## 🚀 Panduan Penggunaan

### Skenario A: Dari Nol (Full Pipeline)

| Langkah | Perintah | Output (umum) |
|---------|----------|----------------|
| 1. Scrape | `python scrape_malang.py` | `data_kost_malang.csv` (root) — pindahkan ke `data/` jika ingin seragam |
| 2. Clean | `python clean_data.py` | `data_kost_malang_clean.csv` (root) |
| 3. EDA | `python eda_check.py` | `eda_1_distribusi_harga.png`, `eda_2_fasilitas_lengkap.png`, `eda_3_korelasi.png` |
| 4. Train | `python train_model.py` | `model_kost_terbaik.pkl`, `list_fitur.pkl`, `hasil_evaluasi/*` |
| 5. App | `streamlit run app.py` | http://localhost:8501 |

**Catatan:**  
- Scraping: 15–30 menit; pastikan Chrome terpasang.  
- Training: ~5–10 menit (Optuna 15 trials).  
- Jika CSV ada di `data/`, `train_model.py` akan memakai `data/data_kost_malang_clean.csv`.

### Skenario B: Data Sudah Ada

```bash
# Langsung training (baca dari data/data_kost_malang_clean.csv atau root)
python train_model.py

# Lalu jalankan app
streamlit run app.py
```

### Skenario C: Hanya Demo Web App

Jika model sudah ada (`model_kost_terbaik.pkl`, `list_fitur.pkl` di root):

```bash
streamlit run app.py
```

Atau gunakan deployment online (jika tersedia):

- 🌐 **[Streamlit Cloud](https://prediksihargakostmalang.streamlit.app/)**

---

## 📈 Pipeline Data Science

### 1. Data Acquisition (`scrape_malang.py`)

- **Sumber:** Mamikos, hasil pencarian kost Malang (bulanan, rentang harga lebar).
- **Teknik:** Selenium (dynamic content), klik berulang tombol "Lihat lagi" (max 300x), BeautifulSoup untuk parsing.
- **Kolom:** Nama Kost, Jenis Kost, Harga Mentah, Fasilitas (teks), Daerah, Lokasi.
- **Output:** `data_kost_malang.csv` (default di root; disarankan pindah ke `data/`).

### 2. Data Cleaning (`clean_data.py`)

- **Harga:** Regex → angka; drop baris dengan harga &lt; 100.000.
- **Fasilitas:** Hapus artefak (mis. star-glyph); binary encoding untuk: AC, WiFi, K. Mandi Dalam, Kloset Duduk, Kasur, Akses 24 Jam.
- **Daerah:** "Kecamatan X" → "X".
- **Output:** `data_kost_malang_clean.csv` (kolom: Nama Kost, Jenis Kost, Daerah_Clean, Harga_Angka, Fasilitas_*).

### 3. EDA (`eda_check.py`)

- Grafik 1: distribusi harga (histogram + KDE).
- Grafik 2: boxplot harga vs tiap kolom fasilitas (grid).
- Grafik 3: heatmap korelasi (numerik).
- Statistik deskriptif `Harga_Angka`.

File disimpan di **current working directory** (bisa dipindah ke `hasil_eda/` untuk rapi).

### 4. Model Training (`train_model.py`)

- **Data:** Hanya kecamatan dengan ≥10 sampel (untuk stabilitas one-hot).
- **Preprocessing:** `get_dummies` untuk Daerah_Clean dan Jenis Kost; target: `Harga_Angka`; drop kolom non-fitur (mis. Nama Kost).
- **Split:** 80% train, 20% test, `random_state=42`.
- **Model:**
  - **Linear Regression** (baseline).
  - **Random Forest:** hyperparameter di-tune dengan Optuna (n_estimators, max_depth, min_samples_split, min_samples_leaf); 3-fold CV, negatif MAE.
- **Seleksi:** Model dengan MAE lebih rendah menang; disimpan sebagai `model_kost_terbaik.pkl`; daftar fitur disimpan di `list_fitur.pkl`.
- **Evaluasi tambahan:** Kategori harga (Ekonomis &lt;850k, Standar 850k–1.5M, Eksklusif &gt;1.5M) → classification report + confusion matrix; grafik: actual vs predicted, residual, feature importance/koefisien, confusion matrix.
- **Output:** `hasil_evaluasi/laporan_komparasi_model.txt`, `hasil_evaluasi/Grafik_Random_Forest.png` (atau Linear_Regression).

### 5. Model Serving (`app.py`)

- Load `model_kost_terbaik.pkl` dan `list_fitur.pkl`.
- Form: pilih kecamatan (dari nama fitur `Daerah_Clean_*`), jenis kost, centang fasilitas.
- Input di-encode one-hot sesuai `list_fitur.pkl` → `model.predict()` → tampilkan estimasi harga (Rp).

---

## 📊 Model Performance

Metrik di bawah berdasarkan **laporan aktual** (dari run terakhir yang tercatat).

### Perbandingan Model

| Metrik | Linear Regression | Random Forest | Pemenang |
|--------|-------------------|---------------|----------|
| **MAE (Rp)** | 127.885 | 49.982 | RF |
| **R²** | 0.88 | 0.95 | RF |
| **MAPE (%)** | 13,64 | 5,02 | RF |

**Kesimpulan:** Random Forest dipilih sebagai model produksi (lebih akurat ~Rp 78k dalam MAE).

### Interpretasi Metrik

| Metrik | Arti |
|--------|------|
| **MAE** | Rata-rata selisih absolut prediksi vs harga asli (rupiah). |
| **R²** | Proporsi varians harga yang dijelaskan model. |
| **MAPE** | Rata-rata error dalam persen terhadap harga asli. |

### Klasifikasi Kategori Harga

Model juga dievaluasi pada bucket: **Ekonomis** (&lt;850k), **Standar** (850k–1.5M), **Eksklusif** (&gt;1.5M). Laporan precision/recall/F1 ada di `hasil_evaluasi/laporan_komparasi_model.txt`.

---

## 🧠 Design Decisions

- **Mengapa Random Forest vs Linear Regression?**  
  Non-linearitas hubungan fitur–harga (lokasi, fasilitas) lebih tertangkap RF; tuning dengan Optuna meningkatkan generalisasi. LR tetap berguna sebagai baseline yang interpretable.

- **Mengapa filter kecamatan &lt;10 sampel?**  
  One-hot untuk daerah dengan sedikit data menyebabkan fitur sparse dan risiko overfitting; menghapusnya meningkatkan stabilitas tanpa kehilangan banyak data.

- **Mengapa 6 fasilitas itu?**  
  Fasilitas tersebut konsisten muncul di sumber dan cukup informatif untuk prediksi harga; kolom lain bisa ditambah jika skema data diperluas.

- **Mengapa Streamlit?**  
  Cepat untuk prototipe dan demo; cocok untuk portfolio dan deployment di Streamlit Cloud tanpa backend terpisah.

---

## 🔁 Reproducibility

- **Random state:** `random_state=42` di `train_test_split` dan Random Forest.
- **Versi:** Gunakan `requirements.txt` dan (opsional) `pip freeze` untuk lock versi.
- **Data:** Simpan sample atau metadata (jumlah baris, tanggal scrape) jika ingin melaporkan hasil yang persis sama.
- **Optuna:** Jumlah trial (15) dan CV (3-fold) tetap; hasil bisa sedikit berbeda antar run karena proses stokastik RF/Optuna jika seed tidak di-set di level Optuna.

---

## ⚖️ Etika & Legal

- **Scraping:** Data diambil dari situs pihak ketiga. Pastikan sesuai **Terms of Use** dan kebijakan robot situs tersebut; gunakan rate limiting dan user-agent yang wajar.
- **Tujuan:** Proyek ini untuk edukasi dan portfolio; prediksi harga tidak menggantikan penilaian pasar atau nasihat profesional.
- **Data pribadi:** Tidak menyimpan data pribadi pemilik kost; hanya informasi listing yang umum ditampilkan di situs.

---

## 📁 Struktur Direktori

```
projek_kost_malang/
├── README.md
├── requirements.txt
├── app.py                  # Streamlit web app
├── scrape_malang.py        # Scraping Mamikos
├── clean_data.py           # Preprocessing
├── eda_check.py            # EDA & visualisasi
├── train_model.py          # Training & evaluasi
├── model_kost_terbaik.pkl  # Model terpilih (generated)
├── list_fitur.pkl          # Daftar kolom fitur (generated)
├── data/
│   ├── data_kost_malang.csv
│   └── data_kost_malang_clean.csv
├── hasil_eda/              # PNG dari EDA (jika disimpan di sini)
└── hasil_evaluasi/
    ├── laporan_komparasi_model.txt
    └── Grafik_Random_Forest.png
```

---

## 🔍 Troubleshooting

| Masalah | Solusi |
|--------|--------|
| Module not found | `pip install -r requirements.txt --upgrade`; pastikan venv aktif. |
| Scraping timeout / gagal | Cek koneksi; kurangi `MAX_CLICKS` di `scrape_malang.py`; perhatikan pembatasan akses dari situs. |
| Model/file tidak ditemukan | Jalankan `python train_model.py` dari root; pastikan `model_kost_terbaik.pkl` dan `list_fitur.pkl` ada di root (atau sesuaikan path di `app.py`). |
| Streamlit cache aneh | Hapus cache: `rm -r ~/.streamlit` (Linux/macOS) atau hapus folder `.streamlit` di user (Windows); jalankan ulang `streamlit run app.py`. |
| File CSV tidak ketemu | `clean_data.py` baca dari root `data_kost_malang.csv`; `train_model.py` coba root lalu `data/data_kost_malang_clean.csv`. Simpan CSV di salah satu lokasi itu atau sesuaikan path di script. |

---

## 📚 Referensi

- [Selenium](https://www.selenium.dev/documentation/)
- [BeautifulSoup](https://beautiful-soup-4.readthedocs.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Optuna](https://optuna.readthedocs.io/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

## 📝 License & Kontak

- **Python:** 3.8+
- **Kontak:** [GitHub – Olfatalatas](https://github.com/Olfatalatas)
- **Last Updated:** Februari 2026

---

## ✅ Checklist Sebelum Production

- [ ] Semua dependency terpasang (`pip install -r requirements.txt`)
- [ ] Data mentah/clean tersedia dan path benar
- [ ] EDA dijalankan (opsional, untuk insight)
- [ ] Model sudah di-train dan file `.pkl` ada
- [ ] Web app diuji lokal (`streamlit run app.py`)
- [ ] ToS/legal scraping sudah dicek jika data dipakai ulang
- [ ] README dan laporan evaluasi terdokumentasi

---

**Happy Coding! 🚀**
