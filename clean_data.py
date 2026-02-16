import pandas as pd
import re

def run_cleaning():
    print("=== MEMULAI PEMBERSIHAN DATA ===")

    # 1. Load Data Mentah
    # Pastikan nama file sesuai dengan hasil scrape Anda
    try:
        df = pd.read_csv('data_kost_malang.csv')
        print(f"Data awal dimuat: {len(df)} baris.")
    except FileNotFoundError:
        print("Error: File 'data_kost_malang.csv' tidak ditemukan.")
        return

    # ====================================================================
    # TAHAP 1: BERSIHKAN HARGA (Hapus Rp, Titik, spasi)
    # ====================================================================
    def clean_harga(text):
        # Hapus semua karakter yang BUKAN angka
        clean_text = re.sub(r'[^0-9]', '', str(text))
        return int(clean_text) if clean_text else 0

    df['Harga_Angka'] = df['Harga Mentah'].apply(clean_harga)
    
    # Filter: Hapus data yang harganya 0 atau aneh (di bawah 100rb)
    df = df[df['Harga_Angka'] > 100000]

    # ====================================================================
    # TAHAP 2: BERSIHKAN TEXT FASILITAS (Hapus star-glyph)
    # ====================================================================
    def clean_fasilitas_text(text):
        text = str(text)
        # Hapus 'star-glyph' dan angka rating di belakangnya
        text = re.sub(r'star-glyph.*', '', text)
        return text.strip()

    df['Fasilitas_Clean'] = df['Fasilitas'].apply(clean_fasilitas_text)

    # ====================================================================
    # TAHAP 3: FEATURE ENGINEERING (Pecah Fasilitas jadi Kolom 0/1)
    # ====================================================================
    # Ini langkah paling penting buat AI. Kita ubah teks jadi angka.
    
    # Daftar fasilitas kunci yang mempengaruhi harga
    fitur_kunci = ['AC', 'WiFi', 'K. Mandi Dalam', 'Kloset Duduk', 'Kasur', 'Akses 24 Jam']

    for fitur in fitur_kunci:
        # Nama kolom baru, misal: "Fasilitas_AC"
        col_name = f"Fasilitas_{fitur.replace(' ', '_').replace('.', '')}"
        
        # Isi 1 jika ada kata kuncinya, 0 jika tidak ada
        df[col_name] = df['Fasilitas_Clean'].apply(lambda x: 1 if fitur.lower() in x.lower() else 0)

    # ====================================================================
    # TAHAP 4: STANDARDISASI DAERAH
    # ====================================================================
    # Mengubah "Kecamatan Lowokwaru" menjadi "Lowokwaru" saja agar seragam
    def clean_daerah(text):
        text = str(text)
        return text.replace('Kecamatan ', '').strip()

    df['Daerah_Clean'] = df['Daerah'].apply(clean_daerah)

    # ====================================================================
    # TAHAP 5: SIMPAN HASIL
    # ====================================================================
    
    # Kita pilih kolom-kolom yang sudah bersih saja untuk disimpan
    kolom_final = [
        'Nama Kost', 
        'Jenis Kost', 
        'Daerah_Clean', 
        'Harga_Angka'
    ]
    # Masukkan semua kolom fasilitas baru secara otomatis
    kolom_fasilitas = [col for col in df.columns if 'Fasilitas_' in col and col != 'Fasilitas_Clean']
    kolom_final.extend(kolom_fasilitas)

    df_clean = df[kolom_final]
    
    # Simpan ke CSV baru
    output_file = 'data_kost_malang_clean.csv'
    df_clean.to_csv(output_file, index=False)

    print("\n=== CONTOH HASIL PEMBERSIHAN (5 DATA PERTAMA) ===")
    print(df_clean.head().to_string())
    print("\n===============================================")
    print(f"SUKSES! Data bersih tersimpan di '{output_file}'")
    print(f"Total Data Bersih: {len(df_clean)} baris.")

if __name__ == "__main__":
    run_cleaning()