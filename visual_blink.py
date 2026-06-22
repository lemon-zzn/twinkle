from dataclasses import dataclass
import math
from typing import Sequence, Tuple


EAR_LEFT_EYE_IDXS = (263, 387, 386, 362, 380, 374)
EAR_RIGHT_EYE_IDXS = (33, 160, 158, 133, 153, 144)


@dataclass
class VisualBlinkState:
    time_s: float
    face_present: bool
    left_ear: float
    right_ear: float
    mean_ear: float
    closed_frames: int
    is_closed: bool
    is_blink_event: bool
    blink_count: int


class VisualBlinkLabeler:
    def __init__(self, ear_threshold: float = 0.22, consecutive_frames: int = 3):
        if consecutive_frames < 1:
            raise ValueError("consecutive_frames must be >= 1")
        self.ear_threshold = float(ear_threshold)
        self.consecutive_frames = int(consecutive_frames)
        self.closed_frames = 0
        self.was_closed = False
        self.blink_count = 0

    def update(
        self,
        time_s: float,
        left_ear: float,
        right_ear: float,
        face_present: bool,
    ) -> VisualBlinkState:
        mean_ear = (float(left_ear) + float(right_ear)) / 2.0
        is_closed = bool(face_present and mean_ear < self.ear_threshold)
        if is_closed:
            self.closed_frames += 1
        else:
            blink_event = self.was_closed and self.closed_frames >= self.consecutive_frames
            if blink_event:
                self.blink_count += 1
            self.was_closed = False
            self.closed_frames = 0
            return VisualBlinkState(
                time_s=float(time_s),
                face_present=bool(face_present),
                left_ear=float(left_ear),
                right_ear=float(right_ear),
                mean_ear=mean_ear,
                closed_frames=0,
                is_closed=False,
                is_blink_event=blink_event,
                blink_count=self.blink_count,
            )

        self.was_closed = True
        return VisualBlinkState(
            time_s=float(time_s),
            face_present=bool(face_present),
            left_ear=float(left_ear),
            right_ear=float(right_ear),
            mean_ear=mean_ear,
            closed_frames=self.closed_frames,
            is_closed=True,
            is_blink_event=False,
            blink_count=self.blink_count,
        )


def calculate_ear(landmarks: Sequence, eye_idxs: Sequence[int]) -> float:
    if len(eye_idxs) != 6:
        raise ValueError("eye_idxs must contain exactly 6 landmark indices")
    p1, p2, p3, p4, p5, p6 = (_point_xy(landmarks[index]) for index in eye_idxs)
    vertical1 = _distance(p2, p6)
    vertical2 = _distance(p3, p5)
    horizontal = _distance(p1, p4)
    if horizontal < 1e-9:
        return 0.0
    return float((vertical1 + vertical2) / (2.0 * horizontal))


def _point_xy(point) -> Tuple[float, float]:
    if hasattr(point, "x") and hasattr(point, "y"):
        x_value = point.x
        y_value = point.y
        if callable(x_value):
            x_value = x_value()
        if callable(y_value):
            y_value = y_value()
        return float(x_value), float(y_value)
    return float(point[0]), float(point[1])


def _distance(first: Tuple[float, float], second: Tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])
