import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

def jalankan_eda():
    print("=== MEMULAI EKSPLORASI DATA (EDA) ===")
    
    # 1. Load Data Bersih
    try:
        df = pd.read_csv('data_kost_malang_clean.csv')
        print(f"Data dimuat: {len(df)} baris.")
    except FileNotFoundError:
        print("File clean belum ada. Jalankan clean_data.py dulu.")
        return

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
    plt.ticklabel_format(style='plain', axis='x')
    
    # Simpan ke file gambar
    plt.savefig('eda_1_distribusi_harga.png')
    print("   -> Disimpan sebagai 'eda_1_distribusi_harga.png'")
    plt.close()

    # ====================================================================
    # GRAFIK 2: PERBANDINGAN SEMUA FASILITAS (Boxplot Grid)
    # ====================================================================
    print("2. Membuat perbandingan harga SEMUA fasilitas...")
    
    # 1. Cari otomatis kolom yang mengandung kata 'Fasilitas' atau 'Mandi'
    daftar_fasilitas = [col for col in df.columns if 'Fasilitas' in col or 'Mandi' in col]

    if len(daftar_fasilitas) > 0:
        # 2. Hitung ukuran grid
        n_cols = 3
        n_rows = math.ceil(len(daftar_fasilitas) / n_cols)
        
        # Buat Canvas besar
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten()

        # 3. Looping membuat Boxplot
        for i, col in enumerate(daftar_fasilitas):
            sns.boxplot(x=df[col], y=df['Harga_Angka'], ax=axes[i], palette='Set2')
            
            # Percantik
            axes[i].set_title(f'Harga vs {col}', fontweight='bold')
            axes[i].set_xlabel('Status (0=Tidak, 1=Ada)')
            axes[i].set_ylabel('Harga Sewa')
            
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig('eda_2_fasilitas_lengkap.png')
        print(f"   -> Disimpan sebagai 'eda_2_fasilitas_lengkap.png' ({len(daftar_fasilitas)} Fasilitas)")
        plt.close()
    else:
        print("   [WARNING] Tidak ditemukan kolom fasilitas. Pastikan nama kolom mengandung kata 'Fasilitas'.")

    # ====================================================================
    # GRAFIK 3: CEK KORELASI (Heatmap)
    # ====================================================================
    print("3. Membuat Heatmap Korelasi...")
    plt.figure(figsize=(12, 10))
    
    kolom_angka = df.select_dtypes(include=['float64', 'int64'])
    korelasi = kolom_angka.corr()
    
    sns.heatmap(korelasi, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Seberapa Kuat Hubungan Antar Fitur?')
    plt.tight_layout()
    
    plt.savefig('eda_3_korelasi.png')
    print("   -> Disimpan sebagai 'eda_3_korelasi.png'")
    plt.close()

    # ====================================================================
    # STATISTIK DESKRIPTIF
    # ====================================================================
    print("\n=== RANGKUMAN STATISTIK ===")
    print(df['Harga_Angka'].describe().apply(lambda x: format(x, 'f')))

if __name__ == "__main__":
    jalankan_eda()