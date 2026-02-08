import streamlit as st
import pandas as pd
import joblib

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
st.set_page_config(
    page_title="Prediksi Harga Kost Malang",
    page_icon="🏠",
    layout="centered"
)

# Sedikit CSS agar tampilan angka harganya besar dan jelas
st.markdown("""
    <style>
    /* Mengatur gaya teks harga besar */
    .big-font {
        font-size: 30px !important;
        font-weight: bold;
        color: #2e7d32 !important; /* Hijau tua */
    }
    
    /* Mengatur kotak hasil */
    .result-box {
        background-color: #f0f2f6; /* Latar abu-abu muda */
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        border: 1px solid #e0e0e0;
        margin-top: 20px;
        
        /* --- PERBAIKAN UTAMA DI SINI --- */
        color: #333333 !important; /* Paksa teks jadi HITAM, abaikan dark mode */
    }

    /* Memaksa judul kecil (h3) di dalam kotak ikut hitam */
    .result-box h3 {
        color: #333333 !important;
        margin-bottom: 15px;
        font-size: 18px;
    }

    /* Memaksa teks footnote (p) jadi abu-abu tua */
    .result-box p {
        color: #666666 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD MODEL & FITUR
# ==============================================================================
@st.cache_resource
def load_assets():
    try:
# 1. Load Model (Ini aman)
        model_loaded = joblib.load('model_kost_terbaik.pkl')
        
        # 2. Load Fitur
        fitur_loaded = [
            'Fasilitas_AC', 
            'Fasilitas_WiFi', 
            'Fasilitas_K_Mandi_Dalam', 
            'Fasilitas_Kloset_Duduk', 
            'Fasilitas_Kasur', 
            'Fasilitas_Akses_24_Jam', 
            'Daerah_Clean_Karang Ploso', 
            'Daerah_Clean_Kedungkandang', 
            'Daerah_Clean_Klojen', 
            'Daerah_Clean_Lowokwaru', 
            'Daerah_Clean_Pakis', 
            'Daerah_Clean_Sukun', 
            'Jenis Kost_Putra', 
            'Jenis Kost_Putri'
        ]
        
        return model_loaded, fitur_loaded
    except Exception as e:
        st.error(f"Gagal memuat model. Error: {e}")
        return None, None

model, fitur_model = load_assets()

# ==============================================================================
# 3. TAMPILAN UTAMA (USER INTERFACE)
# ==============================================================================
st.title("🏠 AI Estimasi Harga Kost Malang")
st.write("Masukkan spesifikasi kost untuk mendapatkan prediksi harga wajar.")

if model is not None:
    # --- BAGIAN INPUT ---
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📍 Lokasi & Jenis")
            
            # 1. Ambil daftar kecamatan dari list_fitur.pkl secara otomatis
            # Mencari kolom yang diawali 'Daerah_Clean_' lalu membuang prefix-nya
            opsi_lokasi = [f.replace('Daerah_Clean_', '') for f in fitur_model if 'Daerah_Clean_' in f]
            lokasi = st.selectbox("Pilih Kecamatan", sorted(opsi_lokasi))
            
            # 2. Input Jenis Kost
            jenis = st.radio("Jenis Kost", ["Putra", "Putri", "Campur"], horizontal=True)

        with col2:
            st.subheader("🛋️ Fasilitas")
            ac = st.checkbox("AC")
            km_dalam = st.checkbox("Kamar Mandi Dalam")
            wifi = st.checkbox("WiFi")
            kasur = st.checkbox("Kasur / Isian")
            kloset = st.checkbox("Kloset Duduk")
            akses_24 = st.checkbox("Akses 24 Jam")

        # Tombol Submit
        submitted = st.form_submit_button("💰 Hitung Harga")

    # --- BAGIAN LOGIKA PREDIKSI ---
    if submitted:
        # 1. Siapkan satu baris data kosong dengan angka 0 semua
        # Kolomnya HARUS sama persis dengan 'list_fitur.pkl'
        input_data = pd.DataFrame(0, index=[0], columns=fitur_model)
        
        # 2. Isi data Fasilitas (Jika dicentang jadi 1, jika tidak 0)
        # Nama kolom ini harus sama dengan yang ada di CSV Training Anda
        input_data['Fasilitas_AC'] = 1 if ac else 0
        input_data['Fasilitas_WiFi'] = 1 if wifi else 0
        input_data['Fasilitas_K_Mandi_Dalam'] = 1 if km_dalam else 0
        input_data['Fasilitas_Kloset_Duduk'] = 1 if kloset else 0
        input_data['Fasilitas_Kasur'] = 1 if kasur else 0
        input_data['Fasilitas_Akses_24_Jam'] = 1 if akses_24 else 0
        
        # 3. Isi data Lokasi (One-Hot Encoding Manual)
        # Kita cari kolom 'Daerah_Clean_PILIHANUSER' lalu set jadi 1
        nama_kolom_lokasi = f"Daerah_Clean_{lokasi}"
        if nama_kolom_lokasi in input_data.columns:
            input_data[nama_kolom_lokasi] = 1
            
        # 4. Isi data Jenis Kost
        nama_kolom_jenis = f"Jenis Kost_{jenis}"
        if nama_kolom_jenis in input_data.columns:
            input_data[nama_kolom_jenis] = 1

        # 5. Lakukan Prediksi
        try:
            prediksi_harga = model.predict(input_data)[0]
            
            # Tampilkan Hasil
            st.markdown("---")
            st.markdown(f"""
                <div class="result-box">
                    <h3>Estimasi Harga Sewa per Bulan:</h3>
                    <p class="big-font">Rp {int(prediksi_harga):,}</p>
                    <p><i>*Harga ini adalah estimasi AI berdasarkan data pasar saat ini.</i></p>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Terjadi error saat prediksi: {e}")