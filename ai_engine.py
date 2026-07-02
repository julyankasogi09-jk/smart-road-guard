# =====================================================================
#  SMART ROAD GUARD - AI ENGINE (ai_engine.py)
#  Peran   : Vision Layer
#  Fungsi  : - Membaca video (dashcam/drone) dan menjalankan YOLOv8
#              untuk mendeteksi lubang jalan (pothole) secara real-time.
#            - Mengekstrak metadata kontekstual (kamera, cuaca, waktu).
#            - Mensimulasikan koordinat GPS yang bergerak dinamis
#              sepanjang ruas jalan yang dipantau.
#            - Menyimpan setiap hasil deteksi ke database MySQL.
#            - Menulis frame teranotasi (bounding box merah) ke disk
#              agar dapat ditampilkan secara "live" oleh dashboard.py
#              (komunikasi antar-proses lewat shared file, lihat README).
#
#  Cara pakai cepat:
#       python ai_engine.py
# =====================================================================

import os
import cv2
import time
import random
import datetime
import mysql.connector
from mysql.connector import Error as MySQLError
from ultralytics import YOLO

# =====================================================================
# 1. KONFIGURASI GLOBAL
#    Ubah bagian ini sesuai kebutuhan video / lingkungan Anda.
# =====================================================================
CONFIG = {
    # --- Sumber video: path file, atau 0 untuk webcam ---
    "VIDEO_SOURCE": "vid_jalan_raya.mp4",

    # --- Model YOLOv8 hasil training (deteksi pothole) ---
    "MODEL_PATH": "best.pt",

    # --- Ambang batas confidence minimum agar deteksi disimpan ---
    "CONF_THRESHOLD": 0.60,

    # --- Lewati N frame agar pemrosesan tidak terlalu berat (real-time) ---
    "FRAME_SKIP": 2,

    # --- Tampilkan jendela OpenCV langsung di komputer (True/False) ---
    "SHOW_WINDOW": True,

    # --- Lokasi file frame "live" yang dibaca oleh dashboard Streamlit ---
    "LATEST_FRAME_PATH": "shared/latest_frame.jpg",

    # ---------------------------------------------------------------
    # METADATA KONTEKSTUAL (Bagian 1 spesifikasi: Posisi Kamera & Cuaca)
    # Pada implementasi nyata, nilai ini bisa diisi otomatis dari
    # nama file video / form input saat upload di dashboard.
    # ---------------------------------------------------------------
    "CAMERA_NAME": "Dashcam Depan",
    "CAMERA_POSITION": "Tinggi 1.2m, Sudut 0 derajat",
    "CAMERA_TYPE": "Dashcam",
    "WEATHER_CONDITION": "Cerah",          # contoh lain: 'Hujan Gerimis', 'Berawan', 'Malam'

    # --- Ruas jalan yang sedang dipantau (harus ada di tabel_jalan) ---
    "ROAD_NAME": "Jalan Ulu Linjing",
    "ROAD_REGION": "Marga Tiga",

    # --- Titik koordinat awal & akhir ruas jalan (untuk simulasi GPS) ---
    "ROAD_START_COORD": (-5.2164219, 105.4943466),
    "ROAD_END_COORD":   (-5.2174219, 105.4953466),

    # --- Konfigurasi koneksi MySQL ---
    "DB_CONFIG": {
        "host": "127.0.0.1",
        "port": 3306,
        "user": "root",
        "password": "",
        "database": "uas_smart_road_guard",
    },
}


# =====================================================================
# 2. GEO SIMULATOR
#    Catatan tugas: "Karena video direkam manual, buat fungsi helper
#    yang mengonversi frame/timestamp menjadi simulasi koordinat
#    Google Maps yang bergerak dinamis di sepanjang ruas jalan."
# =====================================================================
class GeoSimulator:
    """
    Mensimulasikan posisi GPS kendaraan yang bergerak linear dari
    titik awal ke titik akhir ruas jalan, berdasarkan progres video
    (frame saat ini / total frame video).

    Sedikit noise (jitter) ditambahkan agar titik tidak terlihat
    "kaku" mengikuti satu garis lurus sempurna, menyerupai noise GPS
    sungguhan di lapangan.
    """

    def __init__(self, start_coord, end_coord, total_frames, jitter=0.00006):
        self.start_lat, self.start_lon = start_coord
        self.end_lat, self.end_lon = end_coord
        # Hindari pembagian dengan nol jika metadata video tidak terbaca
        self.total_frames = max(total_frames, 1)
        self.jitter = jitter

    def get_coordinate(self, current_frame: int):
        """
        Mengembalikan tuple (latitude, longitude) untuk frame tertentu.
        Progress dibuat 'pulang-pergi' (ping-pong) menggunakan modulo,
        sehingga simulasi tetap valid walau video diputar berulang
        atau lebih panjang dari estimasi total_frames.
        """
        cycle = self.total_frames * 2
        pos_in_cycle = current_frame % cycle
        progress = pos_in_cycle / self.total_frames
        if progress > 1:
            progress = 2 - progress  # fase "pulang" (mundur)

        lat = self.start_lat + (self.end_lat - self.start_lat) * progress
        lon = self.start_lon + (self.end_lon - self.start_lon) * progress

        # Tambahkan jitter kecil agar realistis
        lat += random.uniform(-self.jitter, self.jitter)
        lon += random.uniform(-self.jitter, self.jitter)
        return round(lat, 7), round(lon, 7)


# =====================================================================
# 3. KLASIFIKASI TINGKAT KERUSAKAN
#    Menggabungkan confidence score YOLO dan luas area bounding box
#    relatif terhadap frame, sebagai proxy ukuran lubang jalan.
# =====================================================================
def classify_severity(confidence: float, box_area: float, frame_area: float) -> str:
    """
    Aturan sederhana (dapat dikalibrasi ulang sesuai dataset Anda):
      - High   : lubang besar (>4% luas frame) ATAU confidence sangat tinggi
      - Medium : lubang berukuran sedang
      - Low    : lubang kecil / confidence pas-pasan di atas threshold
    """
    area_ratio = box_area / frame_area if frame_area > 0 else 0

    if area_ratio > 0.04 or confidence > 0.85:
        return "High"
    elif area_ratio > 0.015 or confidence > 0.65:
        return "Medium"
    else:
        return "Low"


# =====================================================================
# 4. DATA LAYER HELPER - Koneksi & Insert ke MySQL
#    Menggunakan pola "get_or_create" agar tabel master (kamera, cuaca,
#    jalan) tidak terduplikasi (memenuhi normalisasi / menghindari
#    redundansi data).
# =====================================================================
class DBConnector:
    def __init__(self, db_config: dict):
        self.db_config = db_config
        self.conn = None
        self._connect()

    def _connect(self):
        try:
            self.conn = mysql.connector.connect(**self.db_config)
            print("[DB] Koneksi MySQL berhasil dibuka.")
        except MySQLError as e:
            print(f"[DB] GAGAL konek ke MySQL: {e}")
            self.conn = None

    def _ensure_connection(self):
        if self.conn is None or not self.conn.is_connected():
            self._connect()

    def get_or_create_kamera(self, nama_kamera, posisi_kamera, tipe_kamera="Dashcam") -> int:
        self._ensure_connection()
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id_kamera FROM tabel_kamera WHERE nama_kamera=%s AND posisi_kamera=%s",
            (nama_kamera, posisi_kamera),
        )
        row = cur.fetchone()
        if row:
            cur.close()
            return row[0]
        cur.execute(
            "INSERT INTO tabel_kamera (nama_kamera, posisi_kamera, tipe_kamera) VALUES (%s,%s,%s)",
            (nama_kamera, posisi_kamera, tipe_kamera),
        )
        self.conn.commit()
        new_id = cur.lastrowid
        cur.close()
        return new_id

    def get_or_create_cuaca(self, kondisi_cuaca) -> int:
        self._ensure_connection()
        cur = self.conn.cursor()
        cur.execute("SELECT id_cuaca FROM tabel_cuaca WHERE kondisi_cuaca=%s", (kondisi_cuaca,))
        row = cur.fetchone()
        if row:
            cur.close()
            return row[0]
        cur.execute("INSERT INTO tabel_cuaca (kondisi_cuaca) VALUES (%s)", (kondisi_cuaca,))
        self.conn.commit()
        new_id = cur.lastrowid
        cur.close()
        return new_id

    def get_or_create_jalan(self, nama_jalan, wilayah_kota, start_coord=None, end_coord=None) -> int:
        self._ensure_connection()
        cur = self.conn.cursor()
        cur.execute(
            "SELECT id_jalan FROM tabel_jalan WHERE nama_jalan=%s AND wilayah_kota=%s",
            (nama_jalan, wilayah_kota),
        )
        row = cur.fetchone()
        if row:
            cur.close()
            return row[0]

        sl, so = (start_coord or (None, None))
        el, eo = (end_coord or (None, None))
        cur.execute(
            """INSERT INTO tabel_jalan
               (nama_jalan, wilayah_kota, titik_awal_lat, titik_awal_long,
                titik_akhir_lat, titik_akhir_long)
               VALUES (%s,%s,%s,%s,%s,%s)""",
            (nama_jalan, wilayah_kota, sl, so, el, eo),
        )
        self.conn.commit()
        new_id = cur.lastrowid
        cur.close()
        return new_id

    def insert_deteksi(self, id_kamera, id_cuaca, id_jalan, level_kerusakan,
                        akurasi, lat, lon, bbox, waktu_deteksi, frame_path=None):
        self._ensure_connection()
        cur = self.conn.cursor()
        x1, y1, x2, y2 = bbox
        cur.execute(
            """INSERT INTO tabel_deteksi
               (id_kamera, id_cuaca, id_jalan, level_kerusakan, akurasi,
                latitude, longitude, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                waktu_deteksi, frame_path)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
            (id_kamera, id_cuaca, id_jalan, level_kerusakan, akurasi,
             lat, lon, x1, y1, x2, y2, waktu_deteksi, frame_path),
        )
        self.conn.commit()
        cur.close()

    def close(self):
        if self.conn and self.conn.is_connected():
            self.conn.close()
            print("[DB] Koneksi MySQL ditutup.")


# =====================================================================
# 5. ENGINE UTAMA - menggabungkan Vision Layer + Data Layer
# =====================================================================
class PotholeDetectionEngine:
    def __init__(self, config: dict):
        self.cfg = config

        print("[ENGINE] Memuat model YOLOv8 ...")
        self.model = YOLO(self.cfg["MODEL_PATH"])

        print("[ENGINE] Membuka koneksi database ...")
        self.db = DBConnector(self.cfg["DB_CONFIG"])

        # Daftarkan / ambil id master data sekali di awal (efisien)
        self.id_kamera = self.db.get_or_create_kamera(
            self.cfg["CAMERA_NAME"], self.cfg["CAMERA_POSITION"], self.cfg["CAMERA_TYPE"]
        )
        self.id_cuaca = self.db.get_or_create_cuaca(self.cfg["WEATHER_CONDITION"])
        self.id_jalan = self.db.get_or_create_jalan(
            self.cfg["ROAD_NAME"], self.cfg["ROAD_REGION"],
            self.cfg["ROAD_START_COORD"], self.cfg["ROAD_END_COORD"],
        )

        print("[ENGINE] Membuka video source ...")
        self.cap = cv2.VideoCapture(self.cfg["VIDEO_SOURCE"])
        if not self.cap.isOpened():
            raise RuntimeError(f"Tidak dapat membuka video: {self.cfg['VIDEO_SOURCE']}")

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 300
        self.geo = GeoSimulator(self.cfg["ROAD_START_COORD"], self.cfg["ROAD_END_COORD"], total_frames)

        # Pastikan folder untuk shared frame ada
        os.makedirs(os.path.dirname(self.cfg["LATEST_FRAME_PATH"]), exist_ok=True)

    # -----------------------------------------------------------------
    def _draw_detection(self, frame, box, label, conf):
        """Menggambar bounding box MERAH sesuai requirement dashboard."""
        x1, y1, x2, y2 = box
        color = (0, 0, 255)  # BGR -> merah
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = f"{label} ({conf:.2f})"
        cv2.putText(frame, text, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # -----------------------------------------------------------------
    def run(self):
        frame_idx = 0
        print("[ENGINE] Mulai memproses video. Tekan 'q' pada jendela video untuk berhenti.")

        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("[ENGINE] Video selesai / tidak ada frame baru.")
                    break

                frame_idx += 1
                if frame_idx % self.cfg["FRAME_SKIP"] != 0:
                    continue  # lewati frame ini demi performa real-time

                # --- Ekstraksi metadata waktu (Bagian 1 spesifikasi) ---
                # waktu_deteksi: waktu nyata saat frame diproses (real-time clock)
                waktu_deteksi = datetime.datetime.now()

                frame_h, frame_w = frame.shape[:2]
                frame_area = frame_w * frame_h

                # --- Inference YOLOv8 ---
                results = self.model.predict(
                    frame, conf=self.cfg["CONF_THRESHOLD"], verbose=False
                )[0]

                lat, lon = self.geo.get_coordinate(frame_idx)

                for box in results.boxes:
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    box_area = max(x2 - x1, 0) * max(y2 - y1, 0)

                    level = classify_severity(conf, box_area, frame_area)
                    self._draw_detection(frame, (x1, y1, x2, y2), level, conf)

                    # --- Simpan ke database (Data Layer) ---
                    self.db.insert_deteksi(
                        id_kamera=self.id_kamera,
                        id_cuaca=self.id_cuaca,
                        id_jalan=self.id_jalan,
                        level_kerusakan=level,
                        akurasi=round(conf, 4),
                        lat=lat,
                        lon=lon,
                        bbox=(x1, y1, x2, y2),
                        waktu_deteksi=waktu_deteksi,
                    )

                # --- Overlay info metadata di frame (transparansi proses) ---
                info = (f"{self.cfg['CAMERA_NAME']} | {self.cfg['WEATHER_CONDITION']} | "
                        f"{waktu_deteksi.strftime('%Y-%m-%d %H:%M:%S')}")
                cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 1, cv2.LINE_AA)

                # --- Tulis frame "live" untuk dibaca dashboard Streamlit ---
                cv2.imwrite(self.cfg["LATEST_FRAME_PATH"], frame)

                if self.cfg["SHOW_WINDOW"]:
                    cv2.imshow("Smart Road Guard - YOLOv8 Live Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[ENGINE] Dihentikan oleh user (tombol 'q').")
                        break

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.db.close()
            print("[ENGINE] Selesai. Semua resource ditutup dengan rapi.")


# =====================================================================
# 6. ENTRY POINT
# =====================================================================
if __name__ == "__main__":
    engine = PotholeDetectionEngine(CONFIG)
    engine.run()
