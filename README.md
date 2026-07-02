## 🚀 Fitur Utama
* **Real-time Pothole Detection:** Mendeteksi lubang di jalan raya melalui input video/kamera menggunakan model AI (`best.pt`).
* **Database Integration:** Menyimpan riwayat log lubang yang terdeteksi ke dalam basis data MySQL/PostgreSQL.
* **Interactive Dashboard:** Menampilkan statistik, grafik, dan riwayat keamanan jalan secara visual kepada pengguna.

## 🛠️ Teknologi & Library yang Digunakan
* **Python** (Bahasa Pemrograman Utama)
* **OpenCV** & **PyTorch / Ultralytics** (Pengolahan Citra & Object Detection)
* **SQL / Basis Data** (Penyimpanan Log Deteksi)
* **Streamlit** *(Sesuaikan dengan library dashboard yang kamu pakai)*

## 📦 Cara Instalasi & Menjalankan
1. Clone repositori ini atau download file ZIP-nya.
2. Import database `db_pothole.sql` yang ada di dalam folder `SQL_Database` ke server SQL lokal kamu (XAMPP/pgAdmin).
3. Install library yang dibutuhkan yang tertera pada folder `requirements`.
4. Jalankan dashboard utama dengan perintah:
   ```bash
   python -m streamlit run dashboard.py
