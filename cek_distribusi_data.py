import pandas as pd

def cek_sebaran_data():
    print("=== CEK JUMLAH DATA PER KECAMATAN ===")
    
    try:
        # Load data
        df = pd.read_csv('data/data_kost_malang_clean.csv')
        
        # Hitung jumlah kemunculan setiap kecamatan
        sebaran = df['Daerah_Clean'].value_counts()
        
        # Tampilkan hasilnya
        print("\nJumlah Data per Kecamatan:")
        print(sebaran)
        
        print(f"\nTotal Seluruh Data: {len(df)}")
        
        # Cek apakah ada yang terlalu sedikit (kurang dari 10 data)
        sedikit = sebaran[sebaran < 10]
        if not sedikit.empty:
            print("\n⚠️ PERINGATAN: Kecamatan dengan data terlalu sedikit (Potensi Kurang Akurat):")
            print(sedikit)
        else:
            print("\n✅ Semua kecamatan memiliki jumlah data yang cukup.")
            
    except FileNotFoundError:
        print("❌ File 'data_kost_malang_clean.csv' tidak ditemukan.")

if __name__ == "__main__":
    cek_sebaran_data()