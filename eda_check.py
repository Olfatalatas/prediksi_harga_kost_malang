import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def jalankan_eda():
    print("=== MEMULAI EKSPLORASI DATA (EDA) ===")
    
    # 1. Load Data Bersih
    try:
        df = pd.read_csv('data_kost_malang_clean.csv')
        print(f"Data dimuat: {len(df)} baris.")
    except FileNotFoundError:
        print("File clean belum ada. Jalankan clean_data.py dulu.")
        return

    # Set gaya visualisasi
    sns.set_theme(style="whitegrid")

    # ====================================================================
    # GRAFIK 1: DISTRIBUSI HARGA (Cek apakah ada harga outlier)
    # ====================================================================
    print("1. Membuat grafik distribusi harga...")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Harga_Angka'], kde=True, color='blue')
    plt.title('Sebaran Harga Kost di Malang')
    plt.xlabel('Harga (Rupiah)')
    plt.ylabel('Jumlah Kost')
    
    # Format angka sumbu X agar tidak muncul notasi ilmiah (1e6)
    plt.ticklabel_format(style='plain', axis='x')
    
    # Simpan ke file gambar
    plt.savefig('eda_1_distribusi_harga.png')
    print("   -> Disimpan sebagai 'eda_1_distribusi_harga.png'")
    plt.close() # Tutup memori gambar

    # ====================================================================
    # GRAFIK 2: PERBANDINGAN FASILITAS UTAMA (Boxplot)
    # ====================================================================
    print("2. Membuat perbandingan harga fasilitas...")
    
    # Kita cek pengaruh AC dan Kamar Mandi Dalam
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot AC
    sns.boxplot(ax=axes[0], x='Fasilitas_AC', y='Harga_Angka', data=df)
    axes[0].set_title('Harga Kost: Non-AC (0) vs AC (1)')
    
    # Plot KM Dalam
    # Perhatikan nama kolom sesuaikan dengan hasil cleaning Anda (biasanya ada underscore)
    # Cek nama kolom dulu
    col_km = [c for c in df.columns if 'Mandi' in c][0] # Cari kolom yang ada kata 'Mandi'
    
    sns.boxplot(ax=axes[1], x=col_km, y='Harga_Angka', data=df)
    axes[1].set_title(f'Harga Kost: KM Luar (0) vs {col_km} (1)')

    plt.savefig('eda_2_fasilitas.png')
    print("   -> Disimpan sebagai 'eda_2_fasilitas.png'")
    plt.close()

    # ====================================================================
    # GRAFIK 3: CEK KORELASI (Heatmap)
    # ====================================================================
    print("3. Membuat Heatmap Korelasi...")
    plt.figure(figsize=(10, 8))
    
    # Pilih hanya kolom angka untuk korelasi
    kolom_angka = df.select_dtypes(include=['float64', 'int64'])
    korelasi = kolom_angka.corr()
    
    sns.heatmap(korelasi, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Seberapa Kuat Hubungan Antar Fitur?')
    
    plt.savefig('eda_3_korelasi.png')
    print("   -> Disimpan sebagai 'eda_3_korelasi.png'")
    plt.close()

    # ====================================================================
    # STATISTIK DESKRIPTIF (Print Angka)
    # ====================================================================
    print("\n=== RANGKUMAN STATISTIK ===")
    print(df['Harga_Angka'].describe().apply(lambda x: format(x, 'f')))

if __name__ == "__main__":
    jalankan_eda()