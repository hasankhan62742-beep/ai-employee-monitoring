#!/usr/bin/env python3
"""AI Employee Monitoring — Pro Web App (Streamlit).

Modern stack: Streamlit + Plotly + YOLOv8s + ByteTrack + OpenCV.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Deploy (free): Streamlit Community Cloud — New app → repo
``hasankhan62742-beep/ai-employee-monitoring`` → branch ``main`` →
main file ``app.py``. Note: free Cloud runs on CPU, so keep test
videos short (30–60 s). For GPU speed, use Google Colab (see README).

Flow: upload a video → set names / thresholds → Process →
watch the annotated video, KPI cards, charts, alerts and downloads.
"""
import os
import shutil
import subprocess
import tempfile
import urllib.request

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import Config
from src.report import (
    find_alerts,
    productivity_by_desk,
    productivity_by_employee,
    save_excel_report,
)

# NOTE: src.engine (YOLOv8 / torch / OpenCV) is imported LAZILY inside
# the Process handler only. Importing it at the top made the whole app
# hang on Streamlit Cloud's free CPU tier — the page stayed on the
# "Running..." spinner for minutes with no UI at all, because torch
# takes that long to load. With lazy import, the title, sidebar and
# upload form render instantly; the heavy import happens only when
# the user actually presses Process.

SAMPLE_VIDEO_URL = (
    "https://videos.pexels.com/video-files/7686621/"
    "7686621-sd_640_360_24fps.mp4"
)
STATE_COLORS = {
    "WORKING": "#00C853",
    "PHONE USAGE": "#FF6D00",
    "ABSENT": "#D50000",
}

st.set_page_config(
    page_title="AI Employee Monitoring Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #0e1117; }
    h1, h2, h3 { color: #ffffff; }
    .stMetric { background: #1a1d27; border-radius: 12px; padding: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 AI Employee Monitoring — Pro")
st.caption(
    "YOLOv8s + ByteTrack · WORKING / ABSENT / PHONE USAGE · "
    "Productivity Score · Smart Alerts · Face Blur · Excel Report"
)
st.markdown("---")


def convert_to_h264(src_path, dst_path):
    """Convert OpenCV mp4v output to browser-playable H264. Returns True/False."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", src_path,
                "-vcodec", "libx264", "-pix_fmt", "yuv420p", dst_path,
            ],
            check=True, capture_output=True, timeout=300,
        )
        return os.path.exists(dst_path)
    except Exception:
        return False


# ---------------- Sidebar: input + settings ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded = st.file_uploader(
        "Upload office video", type=["mp4", "mov", "avi", "mkv"]
    )
    use_sample = st.checkbox("Use sample office video (no file needed)")
    model_name = st.selectbox(
        "YOLO model", ["yolov8s.pt", "yolov8n.pt", "yolov8m.pt"], index=0,
        help="s = best balance of speed + accuracy. n = fastest, m = most accurate.",
    )
    conf = st.slider("Detection confidence", 0.2, 0.6, 0.35, 0.05)
    max_seconds = st.number_input(
        "Process first N seconds (0 = full video)", 0, 3600, 60, step=10
    )
    face_blur = st.toggle("Face blur (privacy)", value=True)
    phone_alert_sec = st.number_input("Phone alert after (sec)", 1, 30, 3)
    absent_alert_sec = st.number_input("Absent alert after (sec)", 1, 60, 8)

    st.subheader("👥 Employee names (by track ID)")
    names = {}
    for track_id, default in [(1, "Ahmed"), (2, "Sara"), (3, "Ali"), (4, "Usman")]:
        names[track_id] = st.text_input(f"ID {track_id}", value=default)

    consent = st.checkbox(
        "I have consent to monitor the people in this video, "
        "and face blur stays ON unless I have a clear reason to turn it off."
    )
    process_btn = st.button("▶️ Process video", type="primary", use_container_width=True)

# ---------------- Process ----------------
if process_btn:
    if not consent:
        st.error("Please tick the consent box first — monitoring needs consent.")
        st.stop()
    if uploaded is None and not use_sample:
        st.error("Please upload a video, or tick 'Use sample office video'.")
        st.stop()

    workdir = tempfile.mkdtemp(prefix="empmon_")
    input_path = os.path.join(workdir, "input.mp4")
    if use_sample and uploaded is None:
        with st.spinner("Downloading sample office video…"):
            urllib.request.urlretrieve(SAMPLE_VIDEO_URL, input_path)
    else:
        with open(input_path, "wb") as f:
            f.write(uploaded.getbuffer())

    config = Config(
        model_name=model_name,
        conf_threshold=float(conf),
        max_seconds=int(max_seconds) if max_seconds else 9999,
        phone_alert_sec=int(phone_alert_sec),
        absent_alert_sec=int(absent_alert_sec),
        face_blur=bool(face_blur),
        employee_names={int(k): v for k, v in names.items() if v.strip()},
    )
    output_video = os.path.join(workdir, "annotated.mp4")
    log_csv = os.path.join(workdir, "log.csv")
    report_xlsx = os.path.join(workdir, "report.xlsx")

    with st.spinner("Loading YOLOv8s + ByteTrack (first run takes a while on free CPU)…"):
        # Lazy import: torch / ultralytics load here, not at app startup.
        from src.engine import run_monitoring

        df = run_monitoring(input_path, output_video, log_csv, config)
        save_excel_report(df, report_xlsx)

    st.session_state["df"] = df
    st.session_state["output_video"] = output_video
    st.session_state["log_csv"] = log_csv
    st.session_state["report_xlsx"] = report_xlsx
    st.session_state["workdir"] = workdir
    st.success(f"Done! {len(df)} log rows — {df['time_sec'].max() + 1:.0f} seconds × 3 desks.")

# ---------------- Dashboard ----------------
df = st.session_state.get("df")
if df is None:
    st.info(
        "👆 Upload a video in the sidebar (or use the sample video), "
        "tick consent, and press **Process video**."
    )
    st.markdown(
        "### What you get\n"
        "- **Annotated video** — green WORKING, orange PHONE USAGE, red live alerts, face blur\n"
        "- **KPI cards + charts** — Working / Phone / Absent %, productivity score per desk & employee\n"
        "- **Smart alerts** — phone ≥ 3 s, absent ≥ 8 s\n"
        "- **Downloads** — annotated video, per-second CSV, Excel report (Summary / Log / Alerts)\n\n"
        "Verified demo (Colab T4, 18 s sample): 63.2% Working, 3.5% Phone, "
        "33.3% Absent · Desk 3 scored 86.8 (A+ Excellent) · 1 alert."
    )
    st.stop()

total = len(df)
working = (df["state"] == "WORKING").sum()
phone = (df["state"] == "PHONE USAGE").sum()
absent = (df["state"] == "ABSENT").sum()
summary = productivity_by_desk(df)
by_employee = productivity_by_employee(df)
alerts = find_alerts(df)
avg_score = round(summary["Productivity_Score"].mean(), 1) if len(summary) else 0.0

# KPI cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🟢 Working", f"{working / total * 100:.1f}%", f"{working} sec")
c2.metric("🟠 Phone Usage", f"{phone / total * 100:.1f}%", f"{phone} sec")
c3.metric("🔴 Absent", f"{absent / total * 100:.1f}%", f"{absent} sec")
c4.metric("⭐ Avg Score", f"{avg_score}/100")
c5.metric("🚨 Alerts", str(len(alerts)))

# Video (H264 for browser playback)
video_path = st.session_state["output_video"]
h264_path = video_path.replace(".mp4", "_h264.mp4")
if not os.path.exists(h264_path):
    convert_to_h264(video_path, h264_path)
play_path = h264_path if os.path.exists(h264_path) else video_path
st.subheader("🎬 Annotated video")
st.video(play_path)

tab_overview, tab_employee, tab_timeline, tab_alerts, tab_downloads = st.tabs(
    ["📊 Overview", "👥 By Employee", "⏱️ Timeline", "🚨 Alerts", "⬇️ Downloads"]
)

with tab_overview:
    col_a, col_b = st.columns(2)
    with col_a:
        pie = px.pie(
            values=[working, phone, absent],
            names=["Working", "Phone Usage", "Absent"],
            color=["Working", "Phone Usage", "Absent"],
            color_discrete_map=STATE_COLORS,
            title="Overall time distribution",
        )
        st.plotly_chart(pie, use_container_width=True)
    with col_b:
        bar = px.bar(
            summary, x="Desk", y="Productivity_Score", color="Desk",
            title="Productivity score by desk", range_y=[0, 100],
        )
        st.plotly_chart(bar, use_container_width=True)
    st.subheader("Productivity by desk")
    st.dataframe(summary, use_container_width=True)

with tab_employee:
    if len(by_employee):
        st.dataframe(by_employee, use_container_width=True)
        bar2 = px.bar(
            by_employee, x="Employee", y="Score", color="Employee",
            title="Productivity score by employee", range_y=[0, 100],
        )
        st.plotly_chart(bar2, use_container_width=True)
        st.caption(
            "Employee scores cover only seconds when the person was visible. "
            "Absent seconds count in the desk score, not here."
        )
    else:
        st.info("No named employees in this video yet.")

with tab_timeline:
    fig = px.scatter(
        df, x="time_sec", y="desk", color="state",
        color_discrete_map=STATE_COLORS,
        title="Second-by-second timeline (green = Working, orange = Phone, red = Absent)",
        labels={"time_sec": "Time (seconds)", "desk": "Desk"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, use_container_width=True)

with tab_alerts:
    if len(alerts):
        st.dataframe(alerts, use_container_width=True)
    else:
        st.success("No major alerts — no long phone use or long absence detected.")

with tab_downloads:
    with open(play_path, "rb") as f:
        st.download_button(
            "⬇️ Annotated video (MP4)", f.read(),
            file_name="employee_monitoring_annotated.mp4",
            mime="video/mp4",
        )
    with open(st.session_state["log_csv"], "rb") as f:
        st.download_button(
            "⬇️ Per-second log (CSV)", f.read(),
            file_name="employee_monitoring_log.csv", mime="text/csv",
        )
    with open(st.session_state["report_xlsx"], "rb") as f:
        st.download_button(
            "⬇️ Excel report (XLSX)", f.read(),
            file_name="Employee_Monitoring_Pro_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

st.markdown("---")
st.caption(
    "Use only with consent. Tell people what is recorded, keep face blur ON "
    "unless you have a clear reason not to. · Built by Muhammad Abdul Qadeer, "
    "Qadeer Automations — abdulqadeerml.com"
)
