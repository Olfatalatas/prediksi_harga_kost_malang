import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def scrape_kost_malang():
    print("=== MEMULAI ROBOT SCRAPING ===")
    
    # 1. Setup Chrome Driver
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless") 
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    
    # 2. Target URL
    url = "https://mamikos.com/cari/malang-kota-malang-jawa-timur-indonesia/all/bulanan/0-15000000/191?keyword=malang&suggestion_type=search&rent=2&sort=price,-&price=10000-20000000&singgahsini=0" 
    print(f"Sedang membuka: {url}...")
    driver.get(url)
    
    # =========================================================================
    # 3. LOGIKA BARU: KLIK TOMBOL "LIHAT LAGI"
    # =========================================================================
    print("Sedang memuat data... Robot akan mencoba mengklik tombol 'Lihat lagi' beberapa kali.")
    
    # Tentukan mau berapa kali klik tombol "Lihat Lagi"?
    # Semakin banyak, semakin lama, tapi data semakin banyak.
    # Coba 5 kali dulu untuk tes. Kalau lancar, ganti jadi 20 atau 50.
    MAX_CLICKS = 50
    
    for i in range(MAX_CLICKS):
        try:
            # Scroll ke bawah dulu agar tombolnya terlihat
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2) 

            # Cari tombol yang mengandung kata "Lihat" (Lihat lagi / Lihat lebih banyak)
            # Kita pakai XPATH karena lebih sakti mencari teks
            tombol_load_more = driver.find_element(By.XPATH, "//*[contains(text(), 'Lihat')]")
            
            # KLIK TOMBOLNYA
            # Kita pakai JavaScript Click agar lebih ampuh (tembus jika ada iklan yang menutupi)
            driver.execute_script("arguments[0].click();", tombol_load_more)
            
            print(f"Berhasil klik 'Lihat lagi' ke-{i+1}...")
            time.sleep(3) # Tunggu data baru loading (JANGAN DIHAPUS, PENTING!)
            
        except Exception as e:
            # Jika tombol tidak ditemukan (berarti data sudah habis), berhenti loop
            print("Tombol 'Lihat lagi' tidak ditemukan atau data sudah habis. Berhenti loading.")
            break
    
    # =========================================================================

    # 4. Ambil Source Code HTML (Setelah semua klik selesai)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    
    # 5. Ekstrak Data
    # NOTE: Pastikan class ini masih valid saat Anda jalankan.
    kost_cards = soup.find_all('div', class_='kost-rc') 
    
    data_kost = []
    print(f"\nSelesai loading! Ditemukan total {len(kost_cards)} slot kartu kost.")
    print("Sedang mengekstrak data...")

    for card in kost_cards:
        try:
            # Mengambil Nama
            nama_elem = card.find('span', class_='rc-info__name bg-c-text bg-c-text--body-4')
            nama = nama_elem.text.strip() if nama_elem else "Tanpa Nama"
            
            # Mengambil Harga
            harga_elem = card.find('span', class_='rc-price__text bg-c-text bg-c-text--body-1')
            harga = harga_elem.text.strip() if harga_elem else "0"
            
            # Mengambil Fasilitas
            fasilitas_elem = card.find('div', class_='kost-rc__facilities')
            fasilitas = fasilitas_elem.text.strip() if fasilitas_elem else "Tidak info"

            # Mengambil Daerah
            daerah_elem = card.find('span', class_='rc-info__location bg-c-text bg-c-text--body-3')
            daerah = daerah_elem.text.strip() if daerah_elem else "Tidak Diketahui"

            # Mengambil Jenis Kost (Logika Anda sudah benar)
            text_kartu = card.text.lower()
            jenis_kost = "Tidak Diketahui"
            if "putri" in text_kartu:
                jenis_kost = "Putri"
            elif "putra" in text_kartu:
                jenis_kost = "Putra"
            elif "campur" in text_kartu:
                jenis_kost = "Campur"

            data_kost.append({
                'Nama Kost': nama,
                'Jenis Kost': jenis_kost,
                'Harga Mentah': harga,
                'Fasilitas': fasilitas,
                'Daerah': daerah,
                'Lokasi': 'Malang',
            })
            
        except Exception as e:
            continue

    # 6. Tutup Browser
    driver.quit()
    
    # 7. Simpan ke CSV
    if len(data_kost) > 0:
        df = pd.DataFrame(data_kost)
        df.to_csv('data_kost_malang.csv', index=False)
        print("\n==================================================")
        print(f"SUKSES! {len(data_kost)} data berhasil disimpan.")
        print("File tersimpan di: data_kost_malang.csv")
        print("==================================================")
        print(df.head())
    else:
        print("\nGAGAL: Tidak ada data yang tertangkap.")

if __name__ == "__main__":
    scrape_kost_malang()