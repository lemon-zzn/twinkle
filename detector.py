from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np

from hp_acoustic_wave.config import DetectorConfig


@dataclass
class DetectionResult:
    is_event: bool
    event_id: int
    threshold: float
    baseline: float
    mad: float


class AdaptiveWaveDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config
        self.history: Deque[float] = deque(maxlen=config.history_size)
        self.event_count = 0
        self.last_event_time_s = -1e9

    def _stats(self):
        if not self.history:
            return 0.0, 0.0, self.config.min_energy
        values = np.asarray(list(self.history), dtype=np.float64)
        baseline = float(np.median(values))
        mad = float(np.median(np.abs(values - baseline)))
        robust_sigma = 1.4826 * mad
        threshold = max(self.config.min_energy, baseline + self.config.threshold_k * robust_sigma)
        return baseline, mad, threshold

    def update(self, time_s: float, motion_energy: float) -> DetectionResult:
        baseline, mad, threshold = self._stats()
        enough_history = len(self.history) >= self.config.min_history
        outside_refractory = (time_s - self.last_event_time_s) >= self.config.refractory_s
        above_threshold = bool(enough_history and motion_energy > threshold)
        is_event = bool(above_threshold and outside_refractory)

        if is_event:
            self.event_count += 1
            self.last_event_time_s = time_s

        in_baseline_freeze = (time_s - self.last_event_time_s) < self.config.baseline_freeze_s
        should_update_baseline = (not enough_history) or (not above_threshold and not in_baseline_freeze)
        if should_update_baseline:
            self.history.append(float(motion_energy))
        return DetectionResult(
            is_event=is_event,
            event_id=self.event_count,
            threshold=float(threshold),
            baseline=float(baseline),
            mad=float(mad),
        )
