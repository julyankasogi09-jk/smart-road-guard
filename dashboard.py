"""
=============================================================
  Smart Road Guard – Dashboard Streamlit
  dashboard.py

  Fitur:
    • Live AI Monitoring (YOLOv8 + OpenCV langsung di dashboard)
    • Peta Folium real-time dengan marker level kerusakan
    • Grafik fluktuasi deteksi (Plotly)
    • Filter Jalan / Cuaca / Kamera
    • Manajemen status perbaikan jalan
    • Ekspor laporan CSV / ringkasan otomatis

  Jalankan dengan:
       streamlit run dashboard.py
=============================================================
"""

import io
import datetime
import pandas as pd
import streamlit as st
import folium
import plotly.express as px
import mysql.connector
from streamlit_folium import st_folium
import cv2
import time
import random
from ultralytics import YOLO

# ─────────────────────────────────────────────
# FUNGSI HELPER (inline, tanpa import ai_engine)
# ─────────────────────────────────────────────

def classify_damage_level(conf: float) -> str:
    """Klasifikasi level kerusakan dari confidence score YOLO."""
    if conf >= 0.85:
        return "High"
    elif conf >= 0.65:
        return "Medium"
    else:
        return "Low"


def get_simulated_coordinates(frame_num: int, total_frames: int):
    """
    Simulasi koordinat GPS yang bergerak ping-pong sepanjang ruas jalan.
    Koordinat sesuai ruas Jalan Ulu Linjing, Marga Tiga.
    """
    start_lat, start_lon = -5.2164219, 105.4943466
    end_lat,   end_lon   = -5.2174219, 105.4953466
    cycle    = max(total_frames, 1) * 2
    pos      = frame_num % cycle
    progress = pos / max(total_frames, 1)
    if progress > 1:
        progress = 2 - progress
    lat = start_lat + (end_lat - start_lat) * progress + random.uniform(-0.00006, 0.00006)
    lon = start_lon + (end_lon - start_lon) * progress + random.uniform(-0.00006, 0.00006)
    return round(lat, 7), round(lon, 7)


# ─────────────────────────────────────────────
# KONFIGURASI DATABASE
# ─────────────────────────────────────────────

DB_CONFIG = {
    "host":     "127.0.0.1",
    "port":     3306,
    "user":     "root",
    "password": "",
    "database": "uas_smart_road_guard",
}

# File video sumber deteksi (ada di folder smart_road_guard)
VIDEO_SOURCE = "vid_jalan_raya.mp4"


# ─────────────────────────────────────────────
# HELPER: KONEKSI & QUERY
# ─────────────────────────────────────────────

@st.cache_resource
def get_connection():
    """Koneksi MySQL yang di-cache Streamlit agar tidak reconnect setiap render."""
    return mysql.connector.connect(**DB_CONFIG)


def query_df(sql: str, params=None) -> pd.DataFrame:
    """Eksekusi SQL SELECT dan kembalikan sebagai DataFrame."""
    try:
        conn = get_connection()
        if not conn.is_connected():
            conn.reconnect()
        return pd.read_sql(sql, conn, params=params)
    except Exception as e:
        st.error(f"[DB Error] {e}")
        return pd.DataFrame()


def execute_update(sql: str, params: tuple) -> bool:
    """Eksekusi SQL UPDATE/INSERT. Kembalikan True jika berhasil."""
    try:
        conn = get_connection()
        if not conn.is_connected():
            conn.reconnect()
        cursor = conn.cursor()
        cursor.execute(sql, params)
        conn.commit()
        cursor.close()
        return True
    except Exception as e:
        st.error(f"[DB Update Error] {e}")
        return False


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_master_data():
    """Muat data master untuk pilihan filter."""
    jalan  = query_df("SELECT id_jalan,  nama_jalan    FROM tabel_jalan")
    cuaca  = query_df("SELECT id_cuaca,  kondisi_cuaca FROM tabel_cuaca")
    kamera = query_df("SELECT id_kamera, nama_kamera   FROM tabel_kamera")
    return jalan, cuaca, kamera


def load_deteksi(
    id_jalan:  int | None = None,
    id_cuaca:  int | None = None,
    id_kamera: int | None = None,
) -> pd.DataFrame:
    """
    Muat data deteksi dengan JOIN ke tabel master.
    status_perbaikan diambil dari tabel_jalan (bukan tabel_deteksi).
    """
    conditions = []
    params     = []

    if id_jalan:
        conditions.append("d.id_jalan  = %s")
        params.append(id_jalan)
    if id_cuaca:
        conditions.append("d.id_cuaca  = %s")
        params.append(id_cuaca)
    if id_kamera:
        conditions.append("d.id_kamera = %s")
        params.append(id_kamera)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    sql = f"""
        SELECT
            d.id_deteksi,
            d.id_jalan,
            k.nama_kamera,
            k.posisi_kamera,
            c.kondisi_cuaca,
            j.nama_jalan,
            j.wilayah_kota,
            d.level_kerusakan,
            d.akurasi,
            d.latitude,
            d.longitude,
            d.waktu_deteksi,
            j.status_perbaikan
        FROM tabel_deteksi d
        JOIN tabel_kamera  k ON d.id_kamera = k.id_kamera
        JOIN tabel_cuaca   c ON d.id_cuaca  = c.id_cuaca
        JOIN tabel_jalan   j ON d.id_jalan  = j.id_jalan
        {where}
        ORDER BY d.waktu_deteksi DESC
    """
    return query_df(sql, params if params else None)


# ─────────────────────────────────────────────
# KOMPONEN PETA FOLIUM
# ─────────────────────────────────────────────

MARKER_COLOR = {
    "High":   "red",
    "Medium": "orange",
    "Low":    "green",
}


def build_map(df: pd.DataFrame) -> folium.Map:
    """
    Membuat peta Folium dengan marker berwarna sesuai level_kerusakan.
    Pusat peta di koordinat rata-rata deteksi, atau default Jalan Ulu Linjing.
    """
    if df.empty:
        center = [-5.2169219, 105.4948466]   # default: Jalan Ulu Linjing
    else:
        center = [df["latitude"].mean(), df["longitude"].mean()]

    m = folium.Map(location=center, zoom_start=16, tiles="OpenStreetMap")

    for _, row in df.iterrows():
        color  = MARKER_COLOR.get(row["level_kerusakan"], "blue")
        popup_html = f"""
        <b>Jalan:</b> {row['nama_jalan']}<br>
        <b>Level:</b> {row['level_kerusakan']}<br>
        <b>Akurasi:</b> {row['akurasi']:.1%}<br>
        <b>Cuaca:</b> {row['kondisi_cuaca']}<br>
        <b>Waktu:</b> {row['waktu_deteksi']}<br>
        <b>Status:</b> {row['status_perbaikan']}
        """
        folium.Marker(
            location=[row["latitude"], row["longitude"]],
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row['level_kerusakan']} – {row['nama_jalan']}",
            icon=folium.Icon(color=color, icon="exclamation-sign"),
        ).add_to(m)

    return m


# ─────────────────────────────────────────────
# KOMPONEN GRAFIK
# ─────────────────────────────────────────────

def chart_fluktuasi(df: pd.DataFrame):
    """Grafik jumlah deteksi per hari, diwarnai per level kerusakan."""
    if df.empty:
        st.info("Belum ada data untuk ditampilkan.")
        return

    df_copy = df.copy()
    df_copy["tanggal"] = pd.to_datetime(df_copy["waktu_deteksi"]).dt.date

    grouped = (
        df_copy.groupby(["tanggal", "level_kerusakan"])
               .size()
               .reset_index(name="jumlah")
    )

    color_seq = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}
    fig = px.bar(
        grouped,
        x="tanggal", y="jumlah", color="level_kerusakan",
        color_discrete_map=color_seq,
        title="Fluktuasi Deteksi Kerusakan per Hari",
        labels={"tanggal": "Tanggal", "jumlah": "Jumlah Deteksi",
                "level_kerusakan": "Level"},
    )
    fig.update_layout(bargap=0.2, legend_title_text="Level Kerusakan")
    st.plotly_chart(fig, use_container_width=True)


def chart_pie_level(df: pd.DataFrame):
    """Pie chart distribusi level kerusakan."""
    if df.empty:
        return
    counts = df["level_kerusakan"].value_counts().reset_index()
    counts.columns = ["Level", "Jumlah"]
    color_seq = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}
    fig = px.pie(counts, names="Level", values="Jumlah",
                 color="Level", color_discrete_map=color_seq,
                 title="Distribusi Level Kerusakan")
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# FITUR: MANAJEMEN STATUS PERBAIKAN
# ─────────────────────────────────────────────

def section_management(df: pd.DataFrame):
    """
    Form untuk mengubah status perbaikan ruas jalan.
    Update dilakukan ke tabel_jalan (bukan tabel_deteksi).
    """
    st.subheader("🔧 Manajemen Status Perbaikan Jalan")

    if df.empty:
        st.warning("Tidak ada data deteksi yang tersedia.")
        return

    col1, col2 = st.columns(2)

    with col1:
        id_pilih = st.selectbox(
            "Pilih ID Deteksi",
            options=df["id_deteksi"].tolist(),
            format_func=lambda x: (
                f"#{x} – "
                + df.loc[df["id_deteksi"] == x, "nama_jalan"].values[0]
                + " ["
                + df.loc[df["id_deteksi"] == x, "level_kerusakan"].values[0]
                + "]"
            ),
        )

    with col2:
        status_baru = st.selectbox(
            "Status Perbaikan Baru",
            options=["Belum Diperbaiki", "Sedang Diperbaiki", "Selesai Dikerjakan"],
        )

    if st.button("💾 Simpan Perubahan Status", type="primary"):
        # status_perbaikan ada di tabel_jalan, update via id_jalan
        id_jalan_det = int(df.loc[df["id_deteksi"] == id_pilih, "id_jalan"].values[0])
        ok = execute_update(
            "UPDATE tabel_jalan SET status_perbaikan = %s WHERE id_jalan = %s",
            (status_baru, id_jalan_det),
        )
        if ok:
            st.success(f"✅ Status jalan untuk deteksi #{id_pilih} diperbarui menjadi '{status_baru}'")
            st.rerun()


# ─────────────────────────────────────────────
# FITUR: EKSPOR LAPORAN
# ─────────────────────────────────────────────

def generate_summary_text(df: pd.DataFrame) -> str:
    """
    Membuat teks ringkasan kondisi jalan otomatis dari data deteksi.
    """
    if df.empty:
        return "Tidak ada data deteksi yang tersedia untuk dilaporkan."

    lines = ["=" * 60,
             "  LAPORAN KONDISI JALAN – Smart Road Guard",
             f"  Dibuat: {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}",
             "=" * 60, ""]

    for jalan, grp in df.groupby("nama_jalan"):
        high_count   = (grp["level_kerusakan"] == "High").sum()
        medium_count = (grp["level_kerusakan"] == "Medium").sum()
        low_count    = (grp["level_kerusakan"] == "Low").sum()
        avg_akurasi  = grp["akurasi"].mean()
        cuaca_umum   = grp["kondisi_cuaca"].mode()[0]
        wilayah      = grp["wilayah_kota"].iloc[0]

        if high_count >= 3:
            status_warning = "⚠️  PERLU PERBAIKAN SEGERA"
        elif high_count >= 1:
            status_warning = "⚠️  Prioritas Menengah"
        else:
            status_warning = "✅  Kondisi Dapat Ditoleransi"

        lines.append(f"📍 {jalan} ({wilayah})")
        lines.append(f"   Kondisi Cuaca Dominan : {cuaca_umum}")
        lines.append(f"   Total Lubang          : {len(grp)}")
        lines.append(f"   High / Medium / Low   : {high_count} / {medium_count} / {low_count}")
        lines.append(f"   Rata-rata Akurasi     : {avg_akurasi:.1%}")
        lines.append(f"   Status                : {status_warning}")
        if high_count > 0:
            lines.append(
                f"\n   → {jalan} memerlukan perbaikan segera karena memiliki "
                f"{high_count} titik lubang tingkat High yang dideteksi pada cuaca {cuaca_umum}."
            )
        lines.append("")

    return "\n".join(lines)


def section_export(df: pd.DataFrame):
    """Tombol unduh CSV dan ringkasan teks laporan."""
    st.subheader("📄 Ekspor Laporan Kondisi Jalan")

    col1, col2 = st.columns(2)

    with col1:
        if not df.empty:
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Unduh CSV",
                data=csv_bytes,
                file_name=f"laporan_pothole_{datetime.date.today()}.csv",
                mime="text/csv",
            )

    with col2:
        summary = generate_summary_text(df)
        st.download_button(
            "⬇️ Unduh Ringkasan (.txt)",
            data=summary.encode("utf-8"),
            file_name=f"ringkasan_pothole_{datetime.date.today()}.txt",
            mime="text/plain",
        )

    with st.expander("📋 Lihat Ringkasan Otomatis"):
        st.code(summary, language=None)


# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Smart Road Guard",
        page_icon="🛣️",
        layout="wide",
    )

    # ── Header ───────────────────────────────────────────────────
    st.title("🛣️ Smart Road Guard – Command Center")
    st.markdown("Sistem Deteksi & Manajemen Kerusakan Jalan Berbasis AI")
    st.divider()

    # ── Muat Data Master ─────────────────────────────────────────
    df_jalan, df_cuaca, df_kamera = load_master_data()

    # ── Sidebar Filter ───────────────────────────────────────────
    with st.sidebar:
        st.header("🔍 Filter Data")

        if df_jalan.empty or df_cuaca.empty or df_kamera.empty:
            st.error("Gagal memuat data master dari database. Periksa koneksi MySQL.")
            st.stop()

        jalan_opts  = {"Semua": None} | dict(zip(df_jalan["nama_jalan"],  df_jalan["id_jalan"]))
        cuaca_opts  = {"Semua": None} | dict(zip(df_cuaca["kondisi_cuaca"], df_cuaca["id_cuaca"]))
        kamera_opts = {"Semua": None} | dict(zip(df_kamera["nama_kamera"], df_kamera["id_kamera"]))

        sel_jalan  = st.selectbox("Nama Jalan",    list(jalan_opts.keys()))
        sel_cuaca  = st.selectbox("Kondisi Cuaca", list(cuaca_opts.keys()))
        sel_kamera = st.selectbox("Posisi Kamera", list(kamera_opts.keys()))

        id_jalan  = jalan_opts[sel_jalan]
        id_cuaca  = cuaca_opts[sel_cuaca]
        id_kamera = kamera_opts[sel_kamera]

        st.divider()
        st.caption("Legenda Marker Peta")
        st.markdown("🔴 High &nbsp; 🟠 Medium &nbsp; 🟢 Low")

        auto_refresh = st.checkbox("Auto-refresh (30 dtk)", value=False)
        if auto_refresh:
            time.sleep(30)
            st.rerun()

    # ── Muat Data Deteksi ─────────────────────────────────────────
    df = load_deteksi(id_jalan, id_cuaca, id_kamera)

    # ── KPI Cards ────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    total     = len(df)
    avg_akura = df["akurasi"].mean() if not df.empty else 0
    high_cnt  = (df["level_kerusakan"] == "High").sum()  if not df.empty else 0
    belum_cnt = (df["status_perbaikan"] == "Belum Diperbaiki").sum() if not df.empty else 0

    col1.metric("🕳️ Total Lubang",      total)
    col2.metric("🎯 Rata-rata Akurasi", f"{avg_akura:.1%}")
    col3.metric("🔴 Kerusakan High",    high_cnt)
    col4.metric("🔧 Belum Diperbaiki",  belum_cnt)

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────
    tab_live, tab_map, tab_chart, tab_table, tab_manage, tab_export = st.tabs([
        "🎥 Live AI Monitoring",
        "🗺️ Peta Real-time",
        "📊 Grafik Fluktuasi",
        "📋 Tabel Data",
        "🔧 Manajemen Jalan",
        "📄 Laporan & Ekspor",
    ])

    # ── Tab Live AI ───────────────────────────────────────────────
    with tab_live:
        st.subheader("🎥 Live Monitoring & AI Command Center")
        st.markdown("Deteksi kerusakan jalan secara real-time menggunakan YOLOv8 terintegrasi dengan dashboard analitik.")

        col_video, col_stats = st.columns([5, 4])

        with col_video:
            st.markdown("##### Camera Stream (YOLOv8 Deteksi)")
            placeholder_video = st.empty()
            placeholder_video.info("Klik tombol '▶️ Mulai Deteksi AI' di bawah untuk memulai visualisasi deteksi real-time.")

            c_btn1, c_btn2, c_btn3 = st.columns(3)
            with c_btn1:
                run_btn   = st.button("▶️ Mulai Deteksi AI", key="run_ai",   type="primary")
            with c_btn2:
                reset_btn = st.button("🔄 Reset Data Simulasi", key="reset_db")
            with c_btn3:
                stop_btn  = st.button("⏹️ Hentikan",          key="stop_ai")

        with col_stats:
            st.markdown("##### Real-time Analytics")
            placeholder_metrics = st.empty()
            placeholder_chart   = st.empty()
            placeholder_recent  = st.empty()

            # Tampilkan metrik awal (id_jalan=1 = Jalan Ulu Linjing)
            df_init    = load_deteksi(id_jalan=1)
            total_init = len(df_init)
            avg_init   = df_init["akurasi"].mean() if not df_init.empty else 0
            high_init  = (df_init["level_kerusakan"] == "High").sum() if not df_init.empty else 0

            with placeholder_metrics.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("🕳️ Total Lubang", total_init)
                m2.metric("🎯 Akurasi Avg",  f"{avg_init:.1%}")
                m3.metric("🔴 Level High",   high_init)

            if not df_init.empty:
                counts_init = df_init["level_kerusakan"].value_counts().reset_index()
                counts_init.columns = ["Level", "Jumlah"]
                color_seq = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}
                fig_init = px.bar(
                    counts_init, x="Level", y="Jumlah", color="Level",
                    color_discrete_map=color_seq,
                    title="Distribusi Level Kerusakan (Jalan Ulu Linjing)",
                    height=260,
                )
                fig_init.update_layout(showlegend=False)
                placeholder_chart.plotly_chart(fig_init, use_container_width=True)
            else:
                placeholder_chart.info("Belum ada data deteksi.")

        # ── Reset Database ──
        if reset_btn:
            try:
                execute_update("DELETE FROM tabel_deteksi", ())
                execute_update("ALTER TABLE tabel_deteksi AUTO_INCREMENT = 1", ())
                # Seed data sesuai skema tabel_deteksi (tanpa kolom status_perbaikan)
                seeds = [
                    (1, 1, 1, "High",   0.9120, -5.2164219, 105.4943466, "2025-06-01 08:30:00"),
                    (1, 1, 1, "Medium", 0.6750, -5.2165000, 105.4945000, "2025-06-01 08:30:15"),
                    (1, 3, 1, "High",   0.8830, -5.2167000, 105.4947000, "2025-06-02 14:15:00"),
                    (2, 2, 1, "Low",    0.4950, -5.2169000, 105.4949000, "2025-06-03 10:00:00"),
                    (3, 5, 1, "Medium", 0.7200, -5.2171000, 105.4951000, "2025-06-04 20:00:00"),
                ]
                sql = """
                    INSERT INTO tabel_deteksi
                        (id_kamera, id_cuaca, id_jalan, level_kerusakan, akurasi,
                         latitude, longitude, waktu_deteksi)
                    VALUES
                        (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                for s in seeds:
                    execute_update(sql, s)
                st.success("✅ Database berhasil direset dengan data simulasi!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Gagal mereset database: {e}")

        # ── Mulai Deteksi AI ──
        if run_btn:
            try:
                model = YOLO("best.pt")
            except Exception as e:
                st.error(f"Gagal memuat model YOLOv8 (best.pt): {e}")
                st.stop()

            cap = cv2.VideoCapture(VIDEO_SOURCE)
            if not cap.isOpened():
                st.error(f"Gagal membuka file video: {VIDEO_SOURCE}")
                st.stop()

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            _id_kamera = 1
            _id_cuaca  = 1
            _id_jalan  = 1
            frame_num  = 0

            color_map = {
                "High":   (0,   0, 255),
                "Medium": (0, 165, 255),
                "Low":    (0, 255,   0),
            }

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                is_detection_frame   = (frame_num % 2 == 0)
                detections_this_frame = []

                if is_detection_frame:
                    results = model(frame, verbose=False)
                    for result in results:
                        for box in result.boxes:
                            conf = float(box.conf[0])
                            if conf < 0.40:
                                continue

                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            level = classify_damage_level(conf)
                            lat, lon = get_simulated_coordinates(frame_num, total_frames)
                            waktu = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                            execute_update(
                                """
                                INSERT INTO tabel_deteksi
                                    (id_kamera, id_cuaca, id_jalan, level_kerusakan,
                                     akurasi, latitude, longitude, waktu_deteksi)
                                VALUES
                                    (%s, %s, %s, %s, %s, %s, %s, %s)
                                """,
                                (_id_kamera, _id_cuaca, _id_jalan, level,
                                 round(conf, 4), lat, lon, waktu),
                            )

                            detections_this_frame.append({
                                "bbox": (x1, y1, x2, y2),
                                "level": level,
                                "confidence": conf,
                            })

                # Gambar bounding box pada frame
                for det in detections_this_frame:
                    x1, y1, x2, y2 = det["bbox"]
                    level = det["level"]
                    conf  = det["confidence"]
                    color = color_map.get(level, (255, 255, 255))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{level} | {conf:.0%}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                info_text = f"Frame: {frame_num}/{total_frames} | Deteksi Aktif"
                cv2.putText(frame, info_text, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                placeholder_video.image(frame, channels="BGR", use_container_width=True)

                # Update stats & charts saat ada deteksi
                if is_detection_frame and len(detections_this_frame) > 0:
                    df_realtime = load_deteksi(id_jalan=1)
                    if not df_realtime.empty:
                        total_rt = len(df_realtime)
                        avg_rt   = df_realtime["akurasi"].mean()
                        high_rt  = (df_realtime["level_kerusakan"] == "High").sum()

                        with placeholder_metrics.container():
                            m1, m2, m3 = st.columns(3)
                            m1.metric("🕳️ Total Lubang", total_rt)
                            m2.metric("🎯 Akurasi Avg",  f"{avg_rt:.1%}")
                            m3.metric("🔴 Level High",   high_rt)

                        counts_rt = df_realtime["level_kerusakan"].value_counts().reset_index()
                        counts_rt.columns = ["Level", "Jumlah"]
                        color_seq = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#27ae60"}
                        fig_rt = px.bar(
                            counts_rt, x="Level", y="Jumlah", color="Level",
                            color_discrete_map=color_seq,
                            title="Distribusi Level Kerusakan (Live)",
                            height=260,
                        )
                        fig_rt.update_layout(showlegend=False)
                        placeholder_chart.plotly_chart(fig_rt, use_container_width=True)

                        with placeholder_recent.container():
                            st.markdown("**5 Deteksi Terakhir:**")
                            st.dataframe(
                                df_realtime[["waktu_deteksi", "level_kerusakan",
                                             "akurasi", "latitude", "longitude"]].head(5),
                                use_container_width=True,
                                height=150,
                            )

                time.sleep(0.01)

            cap.release()
            st.success("✅ Simulasi deteksi video selesai!")

    # ── Tab Peta ──────────────────────────────────────────────────
    with tab_map:
        st.subheader("Peta Titik Kerusakan Jalan")
        m = build_map(df)
        st_folium(m, width="100%", height=520)

    # ── Tab Grafik ─────────────────────────────────────────────────
    with tab_chart:
        st.subheader("Analisis Kerusakan Jalan")
        c1, c2 = st.columns([2, 1])
        with c1:
            chart_fluktuasi(df)
        with c2:
            chart_pie_level(df)

    # ── Tab Tabel ──────────────────────────────────────────────────
    with tab_table:
        st.subheader("Data Deteksi Lengkap")
        if not df.empty:
            def highlight_level(row):
                color = {"High": "#ffd6d6", "Medium": "#fff4cc", "Low": "#d6ffd6"}.get(
                    row["level_kerusakan"], ""
                )
                return [f"background-color: {color}"] * len(row)

            st.dataframe(
                df.style.apply(highlight_level, axis=1),
                use_container_width=True,
                height=400,
            )
        else:
            st.info("Tidak ada data deteksi yang sesuai filter.")

    # ── Tab Manajemen ──────────────────────────────────────────────
    with tab_manage:
        section_management(df)

    # ── Tab Ekspor ─────────────────────────────────────────────────
    with tab_export:
        section_export(df)


if __name__ == "__main__":
    main()
