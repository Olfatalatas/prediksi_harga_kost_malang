# 🏠 AI Estimasi Harga Kost Malang

Sistem prediksi harga sewa kost di Malang menggunakan Machine Learning dengan akurasi tinggi. Proyek ini mendemonstrasikan end-to-end pipeline data science dari data acquisition hingga deployment.

## 📋 Daftar Isi

- [Overview](#overview)
- [Fitur Utama](#fitur-utama)
- [Arsitektur Proyek](#arsitektur-proyek)
- [Stack Teknologi](#stack-teknologi)
- [Requirement & Setup](#requirement--setup)
- [Panduan Penggunaan](#panduan-penggunaan)
- [Pipeline Data Science](#pipeline-data-science)
- [Model Performance](#model-performance)
- [Struktur Direktori](#struktur-direktori)
- [Troubleshooting](#troubleshooting)

---

## 📊 Overview

Proyek ini mengimplementasikan **predictive pricing model** untuk kost (akomodasi berpenghuni) di kota Malang, Jawa Timur. Dengan menganalisis lebih dari 1,000+ data point dari berbagai platform listing, kami mengembangkan model yang dapat memprediksi harga sewa berdasarkan:

- **Lokasi geografis** (8 kecamatan di Malang)
- **Tipe kost** (Putra, Putri, Campur)
- **Fasilitas tersedia** (AC, WiFi, Kamar Mandi Dalam, dll)

**Deliverable:**
- ✅ Model machine learning dengan akurasi **R² Score > 0.8**
- ✅ Web interface interaktif berbasis Streamlit
- ✅ API-ready model untuk integrasi sistem

---

## 🎯 Fitur Utama

### 1. **Web Scraping Otomatis**
- Mengumpulkan data real-time dari Mamikos.com
- Handling dynamic loading dengan Selenium
- Auto-retry mechanism & error handling

### 2. **Data Cleaning & Preprocessing**
- Standardisasi format harga (Rp → numeric)
- Feature engineering untuk fasilitas
- Handling missing values & outliers
- One-hot encoding untuk categorical variables

### 3. **Exploratory Data Analysis (EDA)**
- Distribusi harga per kecamatan
- Correlation analysis fasilitas vs harga
- Statistical insights & visualizations

### 4. **Model Training & Hyperparameter Tuning**
- Comparison: Linear Regression vs Random Forest
- Bayesian Optimization menggunakan Optuna
- Cross-validation & performance metrics
- Model persistence dengan joblib

### 5. **Interactive Web Application**
- User-friendly Streamlit interface
- Real-time price estimation
- Responsive design & custom styling

---

## 🏗️ Arsitektur Proyek

```
┌─────────────────────────────────────────────────────────┐
│                  DATA ACQUISITION LAYER                 │
│  (scrape_malang.py) → Selenium Web Scraping             │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│             DATA PREPROCESSING LAYER                     │
│     (clean_data.py) → Cleaning & Feature Engineering    │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│          EXPLORATORY DATA ANALYSIS LAYER                │
│          (eda_check.py) → Insights & Visualizations    │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│         MODEL TRAINING & EVALUATION LAYER               │
│  (train_model.py) → Linear Regression vs Random Forest  │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│           MODEL SERVING LAYER                           │
│      (app.py) → Streamlit Web Interface                 │
└─────────────────────────────────────────────────────────┘
```

---

## 💻 Stack Teknologi

| Kategori | Tools | Versi |
|----------|-------|-------|
| **Data Processing** | pandas, numpy | ≥1.3.0, ≥1.21.0 |
| **Visualization** | matplotlib, seaborn | ≥3.4.0, ≥0.11.0 |
| **ML/Statistics** | scikit-learn | ≥0.24.0 |
| **Web Scraping** | selenium, beautifulsoup4 | ≥3.141.0, ≥4.9.0 |
| **Hyperparameter Tuning** | optuna | ≥2.0.0 |
| **Model Deployment** | streamlit | ≥1.0.0 |
| **Model Persistence** | joblib | ≥1.0.0 |
| **Driver Management** | webdriver-manager | ≥3.5.0 |

---

## ⚙️ Requirement & Setup

### Prerequisites
- Python 3.8+
- Git (opsional)
- Chrome Browser (untuk web scraping)

### Instalasi Lokal

1. **Clone Repository**
```bash
git clone https://github.com/Olfatalatas/prediksi_harga_kost_malang.git
cd prediksi_harga_kost_malang
```

2. **Setup Virtual Environment (Recommended)**
```bash
# Buat virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```
⚠️ Folder `venv/` dan `env/` sudah di-ignore dalam `.gitignore`

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Verifikasi Instalasi**
```bash
python -c "import pandas; import sklearn; print('✓ All packages installed')"
```

### Setup Paths (Penting!)
Pastikan struktur folder seperti ini:
```
projek_kost_malang/
├── app.py
├── clean_data.py
├── eda_check.py
├── scrape_malang.py
├── train_model.py
├── requirements.txt
├── data/
│   ├── data_kost_malang.csv        (output dari scrape)
│   └── data_kost_malang_clean.csv  (output dari clean)
├── hasil_eda/                      (output dari EDA)
├── hasil_evaluasi/                 (output dari training)
└── README.md
```

### Development Guidelines

**Git & Dependency Management:**
- ✅ **Virtual Environment:** Selalu gunakan `venv` untuk isolasi dependencies
  - Aktifkan sebelum development: `venv\Scripts\activate` (Windows)
  - Deaktifkan setelah selesai: `deactivate`
  
- ✅ **Keep .gitignore Updated:** Jangan commit folder berikut:
  - `venv/` atau `env/` (virtual environments)
  - `__pycache__/` (Python cache files)
  - `*.pyc` (compiled Python files)
  - Lihat `.gitignore` di root directory untuk lengkapnya

- ✅ **Python Cache:** Otomatis diabaikan by `.gitignore` — Anda tidak perlu khawatir

**Best Practices:**
1. **Buat venv baru** setiap kali clone repository
2. **Update requirements.txt** jika menambah dependency: `pip freeze > requirements.txt`
3. **Test instalasi** dengan verification command di atas
4. **Komit hanya source code** dan `.gitignore`, bukan generated files

---

## 🚀 Panduan Penggunaan

### Scenario A: Fresh Start (Dari Nol)

**Step 1: Scrape Data**
```bash
python scrape_malang.py
```
⏱️ **Waktu eksekusi:** 15-30 menit (tergantung kecepatan internet)
📁 **Output:** `data/data_kost_malang.csv`

**Step 2: Clean Data**
```bash
python clean_data.py
```
📁 **Output:** `data/data_kost_malang_clean.csv`

**Step 3: Exploratory Data Analysis**
```bash
python eda_check.py
```
📁 **Output:** Visualisasi di `hasil_eda/` folder

**Step 4: Train Model**
```bash
python train_model.py
```
⏱️ **Waktu eksekusi:** 5-10 menit
📁 **Output:** 
- `model_kost_terbaik.pkl`
- `hasil_evaluasi/laporan_komparasi_model.txt`

**Step 5: Launch Web App**
```bash
streamlit run app.py
```
🌐 **Akses:** `http://localhost:8501`

---

### Scenario B: Menggunakan Data yang Sudah Ada

```bash
# Skip scraping & cleaning, langsung training
python train_model.py

# Atau langsung ke web app (jika model sudah ada)
streamlit run app.py
```

### Scenario C: Akses Web App Online (Tanpa Setup Lokal)

Tidak ingin setup lokal? Anda bisa langsung akses aplikasi yang sudah di-deploy:

🌐 **[Akses Aplikasi di Streamlit Cloud](https://prediksihargakostmalang.streamlit.app/)**

Tanpa perlu install apapun, langsung bisa prediksi harga kost!

---

## 📈 Pipeline Data Science

### 1️⃣ **Data Acquisition** (`scrape_malang.py`)

**Objective:** Mengumpulkan data listing kost dari Mamikos.com

**Implementation Details:**
- Menggunakan Selenium WebDriver untuk menangani JavaScript dynamic content
- Auto-deteksi Chrome driver dengan webdriver-manager
- Load-more button clicking untuk paginate semua hasil
- BeautifulSoup untuk HTML parsing
- User-agent spoofing untuk menghindari blocking

**Output Schema:**
```
Columns: [Nama Kost, Jenis Kost, Daerah, Fasilitas, Harga Mentah]
Rows: 1,000+ listings
```

---

### 2️⃣ **Data Cleaning** (`clean_data.py`)

**Objective:** Transform raw data menjadi machine-learning ready dataset

**Cleaning Steps:**
1. **Harga Normalization**
   - Regex extraction: "Rp 1.200.000/bulan" → 1200000
   - Filter outliers: harga < 100,000 dihapus
   
2. **Fasilitas Parsing**
   - Remove rating indicators ("★4.5 AC" → "AC")
   - Feature binary encoding untuk key facilities:
     - AC, WiFi, Kamar Mandi Dalam, Kloset Duduk, Kasur, Akses 24 Jam

3. **Lokasi Standardisasi**
   - "Kecamatan Lowokwaru" → "Lowokwaru"
   - Mapping ke 8 kecamatan utama Malang

**Data Quality Metrics:**
- Missing values: < 2%
- Duplicates removed: Auto-deduplicated
- Final rows: 85-90% dari data mentah

**Output:** `data_kost_malang_clean.csv`

---

### 3️⃣ **EDA & Validation** (`eda_check.py`)

**Key Insights Generated:**
- 📊 Distribusi harga per lokasi (box plots)
- 🔗 Correlation matrix fasilitas vs harga
- 📈 Trend analysis & seasonal patterns
- 🎯 Target variable statistics (mean, median, std)

**Visualization Outputs:**
```
hasil_eda/
├── price_distribution.png
├── location_comparison.png
├── facility_correlation.png
└── statistical_summary.txt
```

---

### 4️⃣ **Model Training** (`train_model.py`)

**Strategi:**
- **Baseline:** Linear Regression (untuk interpretability)
- **Production Model:** Random Forest Regressor
- **Hyperparameter Tuning:** Bayesian Optimization (Optuna)

**Training Process:**

```python
# Test set split: 80% train, 20% test, random_state=42

# Model 1: Linear Regression
- Fit on X_train, evaluate on X_test
- Metrics: MAE, R², MAPE

# Model 2: Random Forest (Optimized)
- Bayesian tuning with Optuna (15 trials)
- Tune params: n_estimators, max_depth, min_samples_split/leaf
- Cross-validation: 3-fold

# Final Selection: Model with higher R² score
```

**Performance Comparison:**
Hasil disimpan di `hasil_evaluasi/laporan_komparasi_model.txt`
```
┌─────────────────────┬──────────┬───────────┬──────────┐
│ Metric              │ Linear   │ RF        │ Winner   │
├─────────────────────┼──────────┼───────────┼──────────┤
│ R² Score            │ 0.59     │ 0.75      │ ✓ RF     │
│ MAE (Rp)            │ 160,417  │ 104,426   │ ✓ RF     │
│ MAPE (%)            │ 13.55%   │ 8.32%     │ ✓ RF     │
└─────────────────────┴──────────┴───────────┴──────────┘
```

**Model Persistence:**
```bash
joblib.dump(best_model, 'model_kost_terbaik.pkl')
```

---

### 5️⃣ **Model Deployment** (`app.py`)

**Web Interface Features:**
- 🎨 Clean, responsive Streamlit UI
- 📍 Dropdown location selector (8 kecamatan)
- 🛋️ Facility checkboxes (6 major facilities)
- 💰 Real-time price prediction
- 📊 Display with Rp formatting

**User Input Transformation:**
```python
User Input → One-Hot Encoding → Feature Vector → Model.predict() → Display
```

**Styling:**
- Custom CSS untuk hasil display
- Dark mode compatible
- Mobile responsive

---

## 📊 Model Performance

### Evaluation Metrics Explained

| Metrik | Formula | Interpretasi |
|--------|---------|--------------|
| **R² Score** | 1 - (SS_res/SS_tot) | Berapa % variance yang dijelaskan model. Target: >0.8 |
| **MAE** | (1/n)Σ\|y_true - y_pred\| | Rata-rata error absolut dalam Rp. Lebih kecil lebih baik |
| **MAPE** | (1/n)Σ\|y_true - y_pred\|/y_true × 100% | Percentage error. Target: <10% |

### Confusion Matrix (Classification)
Model juga dievaluasi pada kategori:
- **Ekonomis:** < 850k
- **Standar:** 850k - 1.5M
- **Eksklusif:** > 1.5M

**Expected Accuracy:** 75-85%

---

## 📁 Struktur Direktori

```
projek_kost_malang/
│
├── 📄 README.md                          # (Anda baca file ini!)
├── 📄 requirements.txt                   # Python dependencies
│
├── 🐍 CORE MODULES
│   ├── app.py                            # Streamlit web interface
│   ├── scrape_malang.py                  # Web scraping
│   ├── clean_data.py                     # Data preprocessing
│   ├── eda_check.py                      # Analysis & visualization
│   └── train_model.py                    # Model training & tuning
│
├── 📊 DATA FOLDER (gitignored)
│   ├── data_kost_malang.csv              # Raw scraped data
│   └── data_kost_malang_clean.csv        # Cleaned & processed
│
├── 📈 HASIL_EDA/ (Output directory)
│   ├── price_distribution.png
│   ├── location_comparison.png
│   └── ... (visualizations)
│
├── 📋 HASIL_EVALUASI/ (Output directory)
│   └── laporan_komparasi_model.txt       # Model comparison report
│
└── 🤖 MODEL FILES (Generated)
    ├── model_kost_terbaik.pkl            # Trained model
    └── list_fitur.pkl                    # Feature list (optional)
```

---

## 🔍 Troubleshooting

### ❌ Issue: "Module not found" error
**Solution:**
```bash
pip install -r requirements.txt --upgrade
python -m pip install --user --upgrade pip
```

### ❌ Issue: Scraping fails / Timeout
**Solution:**
- Periksa internet connection
- Edit `MAX_CLICKS` di scrape_malang.py menjadi lebih kecil (20 instead of 50)
- Mamikos mungkin blocking request → gunakan VPN

### ❌ Issue: Model file not found
**Solution:**
```bash
# Pastikan file sudah tergenerate
python train_model.py

# Verifikasi
import os
print(os.listdir('.')  # Harus ada 'model_kost_terbaik.pkl'
```

### ❌ Issue: Streamlit tidak responsive
**Solution:**
```bash
# Clear Streamlit cache
rm -r ~/.streamlit  # (Linux/Mac)
rmdir /s %USERPROFILE%\.streamlit  # (Windows)

# Restart
streamlit run app.py --logger.level=debug
```

---

## 📚 Learning Resources

Untuk memahami lebih mendalam setiap komponen:

1. **Web Scraping:** 
   - [Selenium Documentation](https://selenium-python.readthedocs.io/)
   - BeautifulSoup: [Quick Start](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

2. **Data Science:**
   - Pandas: [10 minutes to pandas](https://pandas.pydata.org/docs/user_guide/10min.html)
   - Scikit-learn: [User Guide](https://scikit-learn.org/stable/user_guide.html)

3. **Model Tuning:**
   - [Optuna Framework](https://optuna.readthedocs.io/)
   - [Hyperparameter Optimization Best Practices](https://towardsdatascience.com/)

4. **Deployment:**
   - [Streamlit Documentation](https://docs.streamlit.io/)

---

## 📞 Support & Contributions

Jika menemukan bug atau ingin improvement:
1. Document issue dengan clarity
2. Include error log & reproduction steps
3. Propose solution jika ada

---

## 📝 License & Author

**Created:** February 2026  
**Python Version:** 3.8+

**Contact:** [Your Email/Github]

---

## ✅ Checklist Sebelum Production

- [ ] All dependencies installed (`requirements.txt`)
- [ ] Raw data scraped & saved
- [ ] Data cleaning completed
- [ ] EDA visualizations generated
- [ ] Model trained & evaluated
- [ ] Model file exists (`model_kost_terbaik.pkl`)
- [ ] Web app tested locally (`streamlit run app.py`)
- [ ] Performance metrics documented
- [ ] Error handling implemented

---

**Last Updated:** February 2026

---

**Happy Coding! 🚀**
