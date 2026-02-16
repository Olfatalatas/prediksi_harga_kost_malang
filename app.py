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

# Custom CSS
st.markdown("""
    <style>
    .big-font { font-size: 30px !important; font-weight: bold; color: #2e7d32 !important; }
    .result-box { background-color: #f0f2f6; padding: 25px; border-radius: 15px; text-align: center; border: 1px solid #e0e0e0; margin-top: 20px; color: #333333 !important; }
    .result-box h3 { color: #333333 !important; margin-bottom: 15px; font-size: 18px; }
    .result-box p { color: #666666 !important; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. LOAD MODEL & FITUR
# ==============================================================================
# Kita ganti cache_resource jadi cache_data dan tambahkan ttl=0 supaya dia reload kalau filenya berubah
@st.cache_resource(ttl=0) 
def load_assets():
    try:
        model_loaded = joblib.load('model_kost_terbaik.pkl')
        
        fitur_loaded = joblib.load('list_fitur.pkl')
        
        return model_loaded, fitur_loaded
    except Exception as e:
        return None, None

model, fitur_model = load_assets()

# ==============================================================================
# 3. TAMPILAN UTAMA
# ==============================================================================
st.title("🏠 AI Estimasi Harga Kost Malang")

if model is None:
    st.error("⚠️ File model/fitur tidak ditemukan. Jalankan training dulu!")
else:
    # --- FORM INPUT ---
    with st.form("form_prediksi"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📍 Lokasi & Jenis")
            
            # 1. Logika Pengambilan Lokasi
            # Mengambil semua fitur yang mengandung kata 'Daerah_Clean_'
            opsi_lokasi = [f.replace('Daerah_Clean_', '') for f in fitur_model if 'Daerah_Clean_' in f]
            lokasi = st.selectbox("Pilih Kecamatan", sorted(opsi_lokasi))
            
            jenis = st.radio("Jenis Kost", ["Putra", "Putri", "Campur"], horizontal=True)

        with col2:
            st.subheader("🛋️ Fasilitas")
            ac = st.checkbox("AC")
            km_dalam = st.checkbox("Kamar Mandi Dalam")
            wifi = st.checkbox("WiFi")
            kasur = st.checkbox("Kasur / Isian")
            kloset = st.checkbox("Kloset Duduk")
            akses_24 = st.checkbox("Akses 24 Jam")

        submitted = st.form_submit_button("💰 Hitung Harga")

    # --- LOGIKA PREDIKSI ---
    if submitted:
        input_data = pd.DataFrame(0, index=[0], columns=fitur_model)
        
        input_data['Fasilitas_AC'] = 1 if ac else 0
        input_data['Fasilitas_WiFi'] = 1 if wifi else 0
        input_data['Fasilitas_K_Mandi_Dalam'] = 1 if km_dalam else 0
        input_data['Fasilitas_Kloset_Duduk'] = 1 if kloset else 0
        input_data['Fasilitas_Kasur'] = 1 if kasur else 0
        input_data['Fasilitas_Akses_24_Jam'] = 1 if akses_24 else 0
        
        nama_kolom_lokasi = f"Daerah_Clean_{lokasi}"
        if nama_kolom_lokasi in input_data.columns:
            input_data[nama_kolom_lokasi] = 1
            
        nama_kolom_jenis = f"Jenis Kost_{jenis}"
        if nama_kolom_jenis in input_data.columns:
            input_data[nama_kolom_jenis] = 1

        try:
            prediksi_harga = model.predict(input_data)[0]
            st.markdown(f"""
                <div class="result-box">
                    <h3>Estimasi Harga Sewa per Bulan:</h3>
                    <p class="big-font">Rp {int(prediksi_harga):,}</p>
                </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {e}")