-- =====================================================================
--  SMART ROAD GUARD - DATABASE SCHEMA (db_pothole.sql)
--  Tujuan : Skema relasional ternormalisasi (>= 3NF) untuk sistem
--           deteksi lubang jalan berbasis YOLOv8 + OpenCV.
--  Mesin  : MySQL 8.0+
--
--  Alasan Normalisasi (3NF):
--  - tabel_kamera, tabel_cuaca, tabel_jalan masing-masing menyimpan
--    atribut yang HANYA bergantung pada Primary Key-nya sendiri
--    (tidak ada partial/transitive dependency).
--  - tabel_deteksi adalah tabel transaksi/fakta yang HANYA menyimpan
--    Foreign Key + atribut yang murni milik event deteksi itu sendiri
--    (akurasi, lat/long, waktu, level kerusakan). Tidak ada duplikasi
--    nama kamera/cuaca/jalan di tabel ini -> menghindari redundansi.
-- =====================================================================

CREATE DATABASE IF NOT EXISTS uas_smart_road_guard
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE uas_smart_road_guard;

-- ---------------------------------------------------------------------
-- 1. TABEL_KAMERA
--    Menyimpan referensi posisi/orientasi kamera pengambil video.
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS tabel_kamera;
CREATE TABLE tabel_kamera (
    id_kamera        INT AUTO_INCREMENT PRIMARY KEY,
    nama_kamera      VARCHAR(100)  NOT NULL,                 -- contoh: 'Dashcam Depan'
    posisi_kamera    VARCHAR(150)  NOT NULL,                 -- contoh: 'Tinggi 1.2m, Sudut 0 derajat'
    tipe_kamera      VARCHAR(50)   DEFAULT 'Dashcam',         -- contoh: 'Dashcam' / 'Drone' / 'CCTV'
    created_at       TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uq_kamera (nama_kamera, posisi_kamera)         -- cegah duplikasi data master
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 2. TABEL_CUACA
--    Master kondisi cuaca saat perekaman video.
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS tabel_cuaca;
CREATE TABLE tabel_cuaca (
    id_cuaca         INT AUTO_INCREMENT PRIMARY KEY,
    kondisi_cuaca    VARCHAR(50) NOT NULL UNIQUE              -- 'Cerah','Hujan Gerimis','Hujan Lebat','Berawan','Malam'
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 3. TABEL_JALAN
--    Master ruas jalan yang dipantau, termasuk status perbaikan
--    (atribut ini valid 3NF karena status memang melekat pada jalan,
--     bukan pada event deteksi individual).
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS tabel_jalan;
CREATE TABLE tabel_jalan (
    id_jalan          INT AUTO_INCREMENT PRIMARY KEY,
    nama_jalan        VARCHAR(150) NOT NULL,
    wilayah_kota      VARCHAR(100) NOT NULL,
    titik_awal_lat    DECIMAL(10,7) NULL,                    -- titik referensi awal ruas jalan (untuk simulasi koordinat)
    titik_awal_long   DECIMAL(10,7) NULL,
    titik_akhir_lat   DECIMAL(10,7) NULL,                    -- titik referensi akhir ruas jalan
    titik_akhir_long  DECIMAL(10,7) NULL,
    status_perbaikan  ENUM('Belum Diperbaiki','Sedang Diperbaiki','Selesai Dikerjakan')
                       NOT NULL DEFAULT 'Belum Diperbaiki',
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uq_jalan (nama_jalan, wilayah_kota)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- 4. TABEL_DETEKSI (Tabel Transaksi / Fakta)
--    Setiap baris = satu event deteksi lubang jalan oleh YOLOv8.
--    Relasi many-to-one ke kamera, cuaca, dan jalan via Foreign Key.
-- ---------------------------------------------------------------------
DROP TABLE IF EXISTS tabel_deteksi;
CREATE TABLE tabel_deteksi (
    id_deteksi        BIGINT AUTO_INCREMENT PRIMARY KEY,
    id_kamera         INT NOT NULL,
    id_cuaca          INT NOT NULL,
    id_jalan          INT NOT NULL,
    level_kerusakan   ENUM('Low','Medium','High') NOT NULL,
    akurasi           DECIMAL(5,4) NOT NULL,                  -- confidence score YOLO, 0.0000 - 1.0000
    latitude          DECIMAL(10,7) NOT NULL,
    longitude         DECIMAL(10,7) NOT NULL,
    bbox_x1           INT NULL,                               -- koordinat bounding box (opsional, untuk audit/debug)
    bbox_y1           INT NULL,
    bbox_x2           INT NULL,
    bbox_y2           INT NULL,
    waktu_deteksi     DATETIME NOT NULL,
    frame_path         VARCHAR(255) NULL,                     -- path screenshot frame (opsional, untuk bukti visual)

    CONSTRAINT fk_deteksi_kamera
        FOREIGN KEY (id_kamera) REFERENCES tabel_kamera(id_kamera)
        ON UPDATE CASCADE ON DELETE RESTRICT,

    CONSTRAINT fk_deteksi_cuaca
        FOREIGN KEY (id_cuaca) REFERENCES tabel_cuaca(id_cuaca)
        ON UPDATE CASCADE ON DELETE RESTRICT,

    CONSTRAINT fk_deteksi_jalan
        FOREIGN KEY (id_jalan) REFERENCES tabel_jalan(id_jalan)
        ON UPDATE CASCADE ON DELETE RESTRICT,

    INDEX idx_waktu (waktu_deteksi),
    INDEX idx_level (level_kerusakan)
) ENGINE=InnoDB;

-- ---------------------------------------------------------------------
-- DATA AWAL (SEED) - opsional, mempermudah testing dashboard
-- ---------------------------------------------------------------------
INSERT INTO tabel_cuaca (kondisi_cuaca) VALUES
    ('Cerah'), ('Berawan'), ('Hujan Gerimis'), ('Hujan Lebat'), ('Malam');

INSERT INTO tabel_kamera (nama_kamera, posisi_kamera, tipe_kamera) VALUES
    ('Dashcam Depan', 'Tinggi 1.2m, Sudut 0 derajat', 'Dashcam'),
    ('Dashcam Belakang', 'Tinggi 1.1m, Sudut 180 derajat', 'Dashcam'),
    ('Drone Survey', 'Tinggi 15m, Sudut Nadir 90 derajat', 'Drone');

INSERT INTO tabel_jalan
    (nama_jalan, wilayah_kota, titik_awal_lat, titik_awal_long, titik_akhir_lat, titik_akhir_long)
VALUES
    ('Jalan Ulu Linjing', 'Marga Tiga', -5.2164219, 105.4943466, -5.2174219, 105.4953466);

-- ---------------------------------------------------------------------
-- VIEW BANTUAN (opsional) - mempermudah query JOIN di dashboard/laporan
-- ---------------------------------------------------------------------
CREATE OR REPLACE VIEW v_laporan_deteksi AS
SELECT
    d.id_deteksi,
    j.nama_jalan,
    j.wilayah_kota,
    j.status_perbaikan,
    k.nama_kamera,
    k.posisi_kamera,
    c.kondisi_cuaca,
    d.level_kerusakan,
    d.akurasi,
    d.latitude,
    d.longitude,
    d.waktu_deteksi
FROM tabel_deteksi d
JOIN tabel_kamera k ON d.id_kamera = k.id_kamera
JOIN tabel_cuaca  c ON d.id_cuaca  = c.id_cuaca
JOIN tabel_jalan  j ON d.id_jalan  = j.id_jalan;
