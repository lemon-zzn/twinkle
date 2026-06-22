from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DetectionStatus:
    is_detected: bool
    label: str


def detection_status(
    current_time_s: Optional[float] = None,
    last_detection_time_s: Optional[float] = None,
    hold_s: float = 1.0,
    latest_time_s: Optional[float] = None,
    detected_label: str = "WAVE DETECTED",
) -> DetectionStatus:
    if current_time_s is None:
        current_time_s = latest_time_s
    if current_time_s is None:
        raise ValueError("current_time_s is required")
    if last_detection_time_s is None:
        return DetectionStatus(is_detected=False, label="LISTENING")
    if current_time_s - last_detection_time_s <= hold_s:
        return DetectionStatus(is_detected=True, label=detected_label)
    return DetectionStatus(is_detected=False, label="LISTENING")
