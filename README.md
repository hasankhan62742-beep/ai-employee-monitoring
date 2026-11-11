# AI Employee Monitoring — Computer Vision

Real-time employee activity monitoring with **YOLOv8 + ByteTrack**: detects people and phones, tracks each person with a stable ID, and classifies every desk, every second, as **WORKING / ABSENT / PHONE USAGE** — with productivity scores, smart alerts, face blur, and an Excel report.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8s-ByteTrack-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Author:** Muhammad Abdul Qadeer — AI/ML Engineer, Qadeer Automations
**Website:** https://abdulqadeerml.com · **LinkedIn:** https://linkedin.com/in/muhammad-abdul-qadeer-8262b237b · **GitHub:** https://github.com/hasankhan62742-beep

## What it does

| Feature | How it works |
|---|---|
| Person + phone detection | YOLOv8s, COCO classes `person` (0) and `cell phone` (67), conf 0.35 |
| Unique tracking | ByteTrack (`bytetrack.yaml`, `persist=True`) — one ID per person |
| 3 states | WORKING: person in desk zone · PHONE USAGE: phone overlaps person box · ABSENT: desk zone empty |
| Desk zones | Screen split into 3 vertical zones: Desk 1 / Desk 2 / Desk 3 |
| Productivity score | Per desk and per employee, 0–100, with grades A+ / A / B / C |
| Smart alerts | Phone ≥ 3 s continuous → HIGH · Absent ≥ 8 s continuous → MEDIUM, plus a live red banner on the video |
| Privacy | Face blur (Gaussian blur on the top 35% of each person box), ON by default |
| Names, not just IDs | Map track IDs to names in `src/config.py` — boxes show `Ahmed - WORKING` |
| Reports | Per-second CSV log + Excel workbook (Summary / Detailed_Log / Alerts) + annotated MP4 |

## Verified demo results

Tested in Google Colab (T4 GPU) on a free Pexels office video — 640×360, 23 FPS, 18 seconds, 421 frames.

- **Processing:** 421/421 frames in **17 s** (v1) and **6 s** (v2, warm cache, 60 it/s)
- **Log:** 57 rows = 19 seconds × 3 desks
- **Dashboard (desk score):** **63.2% WORKING** (36 s) · **3.5% PHONE USAGE** (2 s) · **33.3% ABSENT** (19 s)
- **Desk scores:** Desk 3 — 86.8, A+ Excellent (89.5% working) · Desk 2 — 65.8, A Good · Desk 1 — 31.6, C Needs Attention
- **Alerts:** 1 — Desk 1 ABSENT for 10 s (9–18 s, MEDIUM)
- **Employee scores (visible time only):** Ahmed 100.0 · Sara 100.0 · Usman 87.5 · Ali 86.4

> Honest note: employee scores cover only seconds when the person was visible. Absent seconds have no employee name, so they count in the desk score, not the employee score. The sample video is a meeting clip, so people walk in and out of frame and ByteTrack sometimes assigns a new ID (4 IDs seen for 2–3 people). On a fixed office CCTV view with people seated at desks, IDs are stable.

## Quickstart — Google Colab (recommended)

1. Open https://colab.research.google.com → New notebook → name it `AI_Employee_Monitoring`
2. **Runtime → Change runtime type → T4 GPU → Save**
3. Cell 1 — install and load the model:
   ```python
   !pip install -q ultralytics opencv-python-headless pandas matplotlib tqdm
   from ultralytics import YOLO
   model = YOLO("yolov8s.pt")
   ```
4. Cell 2 — download the sample office video (or upload your own):
   ```python
   !wget -q -O /content/sample_office.mp4 "https://videos.pexels.com/video-files/7686621/7686621-sd_640_360_24fps.mp4"
   VIDEO_PATH = "/content/sample_office.mp4"
   ```
5. Clone this repo in Colab and run the engine — or copy the cells from the notebook walkthrough in the project history. The production code is in `main.py` / `src/`.

## Quickstart — local

```bash
git clone https://github.com/hasankhan62742-beep/ai-employee-monitoring.git
cd ai-employee-monitoring
pip install -r requirements.txt

# Quick 60-second test
python main.py --video your_video.mp4 --max-seconds 60

# Full video, no face blur
python main.py --video your_video.mp4 --no-blur
```

Outputs go to `outputs/`:

```
outputs/
  annotated.mp4   # boxes, names, states, live alert banner
  log.csv         # time_sec, desk, employee_id, employee_name, state
  report.xlsx     # Summary / Detailed_Log / Alerts
```

Convert for browser playback (OpenCV writes `mp4v`, browsers want H264):

```bash
ffmpeg -y -i outputs/annotated.mp4 -vcodec libx264 -pix_fmt yuv420p outputs/annotated_h264.mp4
```

## Web App — Pro (Streamlit + Plotly)

Modern stack: **Streamlit + Plotly + YOLOv8s + ByteTrack**. Upload a video, set names and thresholds in the sidebar, press Process — the app shows the annotated video, KPI cards, interactive charts, timeline, alerts, and download buttons for video / CSV / Excel.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501 — upload a video (or tick *Use sample office video*), tick the consent box, and press **Process video**.

**Deploy free — Streamlit Community Cloud:**
1. Push this repo to GitHub (already at `hasankhan62742-beep/ai-employee-monitoring`)
2. https://share.streamlit.io → New app → repo `hasankhan62742-beep/ai-employee-monitoring`, branch `main`, main file `app.py` → Deploy
3. Free Cloud runs on CPU — keep test videos short (30–60 s). For GPU speed, use the Colab walkthrough above.

## Configuration

Edit `src/config.py` (or set names in the app sidebar):

```python
EMPLOYEE_NAMES = {1: "Ahmed", 2: "Sara", 6: "Ali", 8: "Usman"}
FACE_BLUR = True          # privacy on by default
PHONE_ALERT_SEC = 3       # continuous phone use before a HIGH alert
ABSENT_ALERT_SEC = 8      # continuous absence before a MEDIUM alert
```

## Project structure

```
ai-employee-monitoring/
  app.py                  # Streamlit Pro web app (upload → dashboard → downloads)
  main.py                 # CLI entry point
  requirements.txt
  .streamlit/config.toml  # dark modern theme
  src/
    config.py             # model, thresholds, names, face blur
    engine.py             # YOLOv8 + ByteTrack, states, drawing, logging
    report.py             # scores, alerts, Excel export
  examples/
    sample_log.csv        # sample per-second log (format reference)
  README.md
  LICENSE                 # MIT
```

## How the states are decided

1. YOLOv8 detects `person` and `cell phone` boxes each frame.
2. ByteTrack assigns a persistent ID to each person.
3. The person's centre-x places them in Desk 1 (< 0.33), Desk 2 (< 0.66) or Desk 3.
4. If a phone box overlaps the person box → **PHONE USAGE**. Else if a person is in the zone → **WORKING**. Else → **ABSENT**.
5. Once per second, one row per desk is written to the log.

## Limitations — read before deploying

- Desk zones are fixed vertical thirds; for a real office, calibrate them to the actual desk positions.
- Phone detection is the generic COCO phone class — a phone on a desk can briefly trigger PHONE USAGE. The 3-second continuous rule filters most false positives.
- Tracking IDs can change when a person leaves and re-enters the frame.
- This is a **video-file** system. Live CCTV/RTSP and multi-camera support are the next step, not in v1.0.
- **Use only with consent.** Employee monitoring has legal and workplace rules that vary by country and company. Get written consent, tell people what is recorded, and keep face blur ON unless you have a clear reason not to.

## Roadmap

- [x] Streamlit web app — upload a video, get the dashboard in one click
- [ ] RTSP / live camera input
- [ ] Calibratable desk zones (click to define)
- [ ] Pose-based states (sitting / standing / head-down)
- [ ] WhatsApp / email alert delivery

## License

MIT — see `LICENSE`. Built by Muhammad Abdul Qadeer, 2026.
