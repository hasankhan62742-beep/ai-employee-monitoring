#!/usr/bin/env python3
"""AI Employee Monitoring — CLI entry point.

Usage:
    python main.py --video input.mp4
    python main.py --video input.mp4 --output outputs/ --max-seconds 60 --no-blur

Outputs (in --output):
    annotated.mp4   — video with boxes, names, states and live alerts
    log.csv         — one row per desk per second
    report.xlsx     — Summary + Detailed_Log + Alerts

Requires a T4 GPU in Colab for fast processing; CPU works but is slow.
See README.md for the full Colab walkthrough.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
# Heavy imports (cv2, ultralytics, pandas) are loaded inside main(),
# so `python main.py --help` works even before `pip install -r requirements.txt`.


def main():
    parser = argparse.ArgumentParser(description="AI Employee Monitoring")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output", default="outputs",
                        help="Output folder (default: outputs)")
    parser.add_argument("--model", default="yolov8s.pt",
                        help="YOLO weights (default: yolov8s.pt)")
    parser.add_argument("--max-seconds", type=int, default=9999,
                        help="Process only the first N seconds (default: all)")
    parser.add_argument("--no-blur", action="store_true",
                        help="Disable face blur")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Detection confidence (default: 0.35)")
    args = parser.parse_args()

    from src.engine import run_monitoring
    from src.report import (
        find_alerts,
        productivity_by_desk,
        productivity_by_employee,
        save_excel_report,
    )

    os.makedirs(args.output, exist_ok=True)
    output_video = os.path.join(args.output, "annotated.mp4")
    log_csv = os.path.join(args.output, "log.csv")
    report_xlsx = os.path.join(args.output, "report.xlsx")

    config = Config(
        model_name=args.model,
        conf_threshold=args.conf,
        max_seconds=args.max_seconds,
        face_blur=not args.no_blur,
    )

    print(f"Video: {args.video} | Model: {config.model_name} "
          f"| Face blur: {config.face_blur}")
    df = run_monitoring(args.video, output_video, log_csv, config)

    print(f"\nDone! Frames logged: {len(df)} rows")
    print(f"Annotated video: {output_video}")
    print(f"Log CSV: {log_csv}")

    summary = productivity_by_desk(df)
    print("\n=== PRODUCTIVITY BY DESK ===")
    print(summary.to_string(index=False))

    by_employee = productivity_by_employee(df)
    if len(by_employee):
        print("\n=== PRODUCTIVITY BY EMPLOYEE (visible time only) ===")
        print(by_employee.to_string(index=False))

    alerts = find_alerts(df)
    print(f"\n=== ALERTS: {len(alerts)} ===")
    print(alerts.to_string(index=False) if len(alerts) else "No major alerts")

    save_excel_report(df, report_xlsx)
    print(f"\nExcel report: {report_xlsx}")
    print("\nConvert to H264 for browser playback:")
    print(f"  ffmpeg -y -i {output_video} -vcodec libx264 "
          f"-pix_fmt yuv420p {args.output}/annotated_h264.mp4")


if __name__ == "__main__":
    main()
