"""Core monitoring engine: YOLOv8 detection + ByteTrack tracking.

States per desk, logged once per second:
    WORKING     — a person is in the desk zone, no phone overlap
    PHONE USAGE — a detected cell phone overlaps the person box
    ABSENT      — no person in the desk zone

Phone detection uses the COCO ``cell phone`` class (67).
Person detection uses the COCO ``person`` class (0).
"""
import cv2
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from .config import Config


def _is_overlap(person_box, phone_box):
    """True if the phone box intersects the person box."""
    return not (
        phone_box[2] < person_box[0]
        or phone_box[0] > person_box[2]
        or phone_box[3] < person_box[1]
        or phone_box[1] > person_box[3]
    )


def _desk_for_center_x(cx_ratio):
    if cx_ratio < 0.33:
        return 1
    if cx_ratio < 0.66:
        return 2
    return 3


def run_monitoring(video_path, output_video, log_csv, config=None):
    """Process one video and write an annotated video + per-second log CSV.

    Returns the log DataFrame with columns:
    time_sec, desk, employee_id, employee_name, state
    """
    config = config or Config()
    model = YOLO(config.model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    logs = []
    phone_start = {}
    desk_last_seen = {1: 0.0, 2: 0.0, 3: 0.0}
    frame_idx = 0
    limit = min(total_frames, fps * config.max_seconds)
    pbar = tqdm(total=limit, desc="Processing")

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= limit:
            break
        cur_time = frame_idx / fps

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0, 67],
            conf=config.conf_threshold,
            verbose=False,
        )

        persons, phones = [], []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            ids = (
                boxes.id.int().tolist()
                if boxes.id is not None
                else [None] * len(boxes)
            )
            for box, cls, track_id in zip(
                boxes.xyxy.tolist(), boxes.cls.tolist(), ids
            ):
                x1, y1, x2, y2 = map(int, box)
                if int(cls) == 0:
                    persons.append({"id": track_id, "box": (x1, y1, x2, y2)})
                elif int(cls) == 67:
                    phones.append((x1, y1, x2, y2))

        desk_info = {1: None, 2: None, 3: None}
        for person in persons:
            x1, y1, x2, y2 = person["box"]
            desk = _desk_for_center_x((x1 + x2) / 2 / width)
            person["desk"] = desk
            person["phone"] = any(
                _is_overlap(person["box"], phone) for phone in phones
            )
            person["name"] = config.get_name(person["id"])
            desk_info[desk] = person
            desk_last_seen[desk] = cur_time

            if config.face_blur:
                face_h = int((y2 - y1) * 0.35)
                fx1, fy1 = max(0, x1), max(0, y1)
                fx2, fy2 = min(width, x2), min(height, y1 + face_h)
                if fy2 > fy1 and fx2 > fx1:
                    roi = frame[fy1:fy2, fx1:fx2]
                    frame[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(roi, (31, 31), 0)

        # Live alerts
        active_alerts = []
        for person in persons:
            if person["phone"]:
                if person["id"] not in phone_start:
                    phone_start[person["id"]] = cur_time
                elif cur_time - phone_start[person["id"]] >= config.phone_alert_sec:
                    active_alerts.append(
                        f"PHONE ALERT: {person['name']} "
                        f"{int(cur_time - phone_start[person['id']])}s"
                    )
            elif person["id"] in phone_start:
                del phone_start[person["id"]]
        for desk in (1, 2, 3):
            absent_for = cur_time - desk_last_seen[desk]
            if desk_info[desk] is None and absent_for >= config.absent_alert_sec:
                active_alerts.append(
                    f"ABSENT ALERT: Desk {desk} {int(absent_for)}s"
                )

        # Draw desk zones
        for x in (int(width * 0.33), int(width * 0.66)):
            cv2.line(frame, (x, 60), (x, height), (255, 255, 255), 1)
        for desk, lx in (
            (1, 20),
            (2, int(width * 0.33) + 20),
            (3, int(width * 0.66) + 20),
        ):
            cv2.putText(
                frame, f"DESK {desk}", (lx, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )

        working_n = sum(1 for v in desk_info.values() if v and not v["phone"])
        phone_n = sum(1 for v in desk_info.values() if v and v["phone"])
        absent_n = sum(1 for v in desk_info.values() if v is None)

        for person in persons:
            x1, y1, x2, y2 = person["box"]
            if person["phone"]:
                color = (0, 140, 255)
                label = f"{person['name']} - PHONE USAGE"
            else:
                color = (0, 200, 0)
                label = f"{person['name']} - WORKING"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2,
            )
        for (x1, y1, x2, y2) in phones:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)

        cv2.rectangle(frame, (0, 0), (width, 30), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"AI MONITORING PRO  |  Working: {working_n}  |  "
            f"Phone: {phone_n}  |  Absent: {absent_n}",
            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        if active_alerts:
            cv2.rectangle(frame, (0, 30), (width, 60), (0, 0, 200), -1)
            cv2.putText(
                frame, "ALERT: " + " | ".join(active_alerts[:2]), (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

        if frame_idx % fps == 0:
            second = frame_idx // fps
            for desk in (1, 2, 3):
                info = desk_info[desk]
                if info is None:
                    state, emp, name = "ABSENT", "", ""
                elif info["phone"]:
                    state, emp, name = "PHONE USAGE", str(info["id"]), info["name"]
                else:
                    state, emp, name = "WORKING", str(info["id"]), info["name"]
                logs.append([second, f"Desk {desk}", emp, name, state])

        out.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    df = pd.DataFrame(
        logs,
        columns=["time_sec", "desk", "employee_id", "employee_name", "state"],
    )
    df.to_csv(log_csv, index=False)
    return df
