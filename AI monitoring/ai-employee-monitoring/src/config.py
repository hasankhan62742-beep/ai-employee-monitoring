"""Configuration for AI Employee Monitoring.

Edit EMPLOYEE_NAMES to map ByteTrack IDs to real names.
Desk zones are three vertical zones: Desk 1 | Desk 2 | Desk 3.
"""
from dataclasses import dataclass, field


@dataclass
class Config:
    model_name: str = "yolov8s.pt"
    conf_threshold: float = 0.35
    max_seconds: int = 9999  # set to 60 for a quick test
    phone_alert_sec: int = 3
    absent_alert_sec: int = 8
    face_blur: bool = True
    employee_names: dict = field(default_factory=lambda: {
        1: "Ahmed",
        2: "Sara",
        6: "Ali",
        8: "Usman",
    })

    def get_name(self, track_id):
        if track_id is None:
            return "Unknown"
        return self.employee_names.get(int(track_id), f"ID {int(track_id)}")
