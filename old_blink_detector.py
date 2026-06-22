"""Old blink detector algorithm extracted from commit 7d64a2f.

This is the simpler pre-FMCW-coherence implementation: pure phase trajectory +
local peak/segment gate. Kept here so we can A/B benchmark against the current
mutation. Classes are renamed with an ``Old`` prefix to avoid colliding with
the live ``TwinkleTwinkleBlinkDetector``/``BlinkListenerBlinkDetector``.

Source: ``git show 7d64a2f:hp_acoustic_wave/blink_detector.py`` (538 lines).
"""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from hp_acoustic_wave.dsp import ChunkFeature, unwrap_delta
from hp_acoustic_wave.blink_detector import (
    BlinkDetectionConfig,
    BlinkDetectionResult,
)


@dataclass
class _TwinkleGatePoint:
    time_s: float
    score: float
    threshold: float
    above_threshold: bool
    motion_energy: float
    sign_changes: int


class _OldRobustEventGate:
    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        self.history: Deque[float] = deque(maxlen=config.history_size)
        self.event_count = 0
        self.last_event_time_s = -1e9
        self.active = False

    def stats(self):
        if not self.history:
            return 0.0, 0.0, self.config.min_score
        values = np.asarray(list(self.history), dtype=np.float64)
        baseline = float(np.median(values))
        mad = float(np.median(np.abs(values - baseline)))
        robust_sigma = 1.4826 * mad
        threshold = max(self.config.min_score, baseline + self.config.threshold_k * robust_sigma)
        return baseline, mad, threshold

    def update(self, time_s: float, score: float, force_event: bool = False):
        baseline, mad, threshold = self.stats()
        enough_history = len(self.history) >= self.config.min_history
        outside_startup = time_s >= self.config.startup_ignore_s
        event_level = threshold
        if force_event and getattr(self.config, "absolute_score_floor", 0.0) > 0.0:
            event_level = min(event_level, self.config.absolute_score_floor)
        release_level = event_level * self.config.release_ratio
        if self.active and score <= release_level:
            self.active = False
        above_threshold = bool(enough_history and outside_startup and (score > threshold or force_event))
        outside_refractory = (time_s - self.last_event_time_s) >= self.config.refractory_s
        is_event = bool(above_threshold and outside_refractory and not self.active)

        if is_event:
            self.event_count += 1
            self.last_event_time_s = time_s
            self.active = True

        in_baseline_freeze = (time_s - self.last_event_time_s) < self.config.baseline_freeze_s
        warmup_outlier = bool(
            not enough_history
            and len(self.history) >= max(3, self.config.min_history // 3)
            and outside_startup
            and score > threshold
        )
        should_update_baseline = (
            (not enough_history and not warmup_outlier)
            or (enough_history and not above_threshold and not in_baseline_freeze and not self.active)
        )
        if should_update_baseline:
            self.history.append(float(score))

        return is_event, self.event_count, baseline, mad, threshold


class _OldTwinklePeakEventGate:
    """Local peak/segment gate for Twinkle-style phase trajectories.

    Verbatim copy from commit 7d64a2f. Kept simple — no FMCW coherence, no
    phase-pair library, no spatial spread. Just 3-point local peak + refractory
    + large-motion suppression.
    """

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        self.history: Deque[float] = deque(maxlen=config.history_size)
        self.points: Deque[_TwinkleGatePoint] = deque(maxlen=3)
        self.event_count = 0
        self.last_event_time_s = -1e9
        self.suppress_until_s = -1e9
        self.last_peak_point: Optional[_TwinkleGatePoint] = None
        self.last_gate_metrics: Dict[str, float] = {}

    def stats(self):
        if not self.history:
            return 0.0, 0.0, self.config.min_score
        values = np.asarray(list(self.history), dtype=np.float64)
        baseline = float(np.median(values))
        mad = float(np.median(np.abs(values - baseline)))
        robust_sigma = 1.4826 * mad
        threshold = max(self.config.min_score, baseline + self.config.threshold_k * robust_sigma)
        return baseline, mad, threshold

    def update(
        self,
        time_s: float,
        score: float,
        motion_energy: float,
        sign_changes: int,
    ):
        baseline, mad, threshold = self.stats()
        enough_history = len(self.history) >= self.config.min_history
        outside_startup = time_s >= self.config.startup_ignore_s
        above_threshold = bool(enough_history and outside_startup and score > threshold)
        suppressing_large_motion = bool(
            score >= self.config.twinkle_large_motion_score
            and motion_energy >= self.config.twinkle_large_motion_energy
        )
        if suppressing_large_motion:
            self.suppress_until_s = max(
                self.suppress_until_s,
                time_s + self.config.twinkle_large_motion_suppress_s,
            )

        point = _TwinkleGatePoint(
            time_s=float(time_s),
            score=float(score),
            threshold=float(threshold),
            above_threshold=above_threshold,
            motion_energy=float(motion_energy),
            sign_changes=int(sign_changes),
        )
        self.points.append(point)

        is_event = False
        candidate = None
        candidate_is_local_peak = False
        candidate_is_rising_edge = False
        if len(self.points) == 3:
            previous, middle, current = self.points
            candidate_is_local_peak = middle.score >= previous.score and middle.score > current.score
            candidate_is_rising_edge = (not previous.above_threshold) and middle.above_threshold
            if candidate_is_local_peak or candidate_is_rising_edge:
                candidate = middle

        if candidate is not None:
            score_ceiling_ok = (
                self.config.twinkle_max_peak_score <= 0.0
                or candidate.score <= self.config.twinkle_max_peak_score
            )
            candidate_ok = bool(
                candidate.above_threshold
                and candidate.score >= candidate.threshold * self.config.twinkle_peak_min_ratio
                and score_ceiling_ok
                and candidate.motion_energy <= self.config.twinkle_max_motion_energy
                and candidate.sign_changes <= self.config.twinkle_max_sign_changes
                and candidate.time_s >= self.suppress_until_s
                and (candidate.time_s - self.last_event_time_s) >= self.config.refractory_s
            )
            if candidate_ok:
                self.event_count += 1
                self.last_event_time_s = candidate.time_s
                self.last_peak_point = candidate
                is_event = True

        warmup_outlier = bool(
            not enough_history
            and len(self.history) >= max(3, self.config.min_history // 3)
            and outside_startup
            and score > threshold
        )
        should_update_baseline = (
            (not enough_history and not warmup_outlier)
            or (enough_history and not above_threshold)
        )
        if should_update_baseline:
            self.history.append(float(score))

        self.last_gate_metrics = {
            "twinkle_candidate_peak": float(candidate is not None),
            "twinkle_candidate_local_peak": float(candidate_is_local_peak),
            "twinkle_candidate_rising_edge": float(candidate_is_rising_edge),
            "twinkle_large_motion_suppressed": float(suppressing_large_motion),
            "twinkle_suppress_until_s": float(self.suppress_until_s),
            "twinkle_peak_time_s": float(candidate.time_s) if candidate is not None else float(time_s),
            "twinkle_peak_score": float(candidate.score) if candidate is not None else float(score),
            "twinkle_peak_threshold": float(candidate.threshold) if candidate is not None else float(threshold),
            "twinkle_peak_motion_energy": (
                float(candidate.motion_energy) if candidate is not None else float(motion_energy)
            ),
            "twinkle_peak_sign_changes": (
                float(candidate.sign_changes) if candidate is not None else float(sign_changes)
            ),
        }

        return is_event, self.event_count, baseline, mad, threshold


class OldBlinkListenerDetector:
    """BlinkListener-inspired I/Q viewing-position bump detector (7d64a2f version)."""

    method = "old_blinklistener"

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        self.gate = _OldRobustEventGate(config)
        self.baseline_i: Deque[float] = deque(maxlen=config.history_size)
        self.baseline_q: Deque[float] = deque(maxlen=config.history_size)
        self.baseline_amplitude: Deque[float] = deque(maxlen=config.history_size)
        self.amplitude_window: Deque[float] = deque(maxlen=config.short_window)
        self.projection_window: Deque[float] = deque(maxlen=config.short_window)
        self.last_ungated_coherence = 0.0

    def _center(self, feature: ChunkFeature):
        if not self.baseline_i:
            return feature.i_value, feature.q_value
        return float(np.median(self.baseline_i)), float(np.median(self.baseline_q))

    def _baseline_amplitude(self, feature: ChunkFeature) -> float:
        if not self.baseline_amplitude:
            return float(feature.amplitude)
        return float(np.median(self.baseline_amplitude))

    def _best_viewing_projection(self, feature: ChunkFeature, center_i: float, center_q: float) -> float:
        if len(self.baseline_i) < 3:
            return 0.0

        points_i = np.asarray(self.baseline_i, dtype=np.float64) - center_i
        points_q = np.asarray(self.baseline_q, dtype=np.float64) - center_q
        delta_i = float(feature.i_value - center_i)
        delta_q = float(feature.q_value - center_q)

        best = 0.0
        for angle in np.linspace(0.0, math.pi, num=12, endpoint=False):
            direction_i = math.cos(float(angle))
            direction_q = math.sin(float(angle))
            baseline_projection = points_i * direction_i + points_q * direction_q
            current_projection = delta_i * direction_i + delta_q * direction_q
            best = max(best, abs(float(current_projection - np.median(baseline_projection))))
        return float(best)

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        center_i, center_q = self._center(feature)
        baseline_amplitude = self._baseline_amplitude(feature)
        amplitude_bump = float(abs(feature.amplitude - baseline_amplitude))
        best_projection = self._best_viewing_projection(feature, center_i, center_q)
        gated_projection = min(best_projection, amplitude_bump * 2.0)
        phase_stable_projection = best_projection if abs(feature.phase_delta) <= 0.12 else gated_projection

        self.amplitude_window.append(amplitude_bump)
        self.projection_window.append(phase_stable_projection)

        if len(self.amplitude_window) >= 3:
            amplitude_range = float(max(self.amplitude_window) - min(self.amplitude_window))
            projection_range = float(max(self.projection_window) - min(self.projection_window))
        else:
            amplitude_range = amplitude_bump
            projection_range = phase_stable_projection
        raw_score = max(amplitude_bump, amplitude_range, phase_stable_projection, 0.75 * projection_range)
        score_scale = max(baseline_amplitude, 1e-4)
        score = raw_score / score_scale

        is_event, event_id, baseline, mad, threshold = self.gate.update(feature.time_s, score)

        in_baseline_freeze = (feature.time_s - self.gate.last_event_time_s) < self.config.baseline_freeze_s
        should_update_center = (not is_event) and (not in_baseline_freeze or len(self.baseline_i) < self.config.min_history)
        if should_update_center:
            self.baseline_i.append(float(feature.i_value))
            self.baseline_q.append(float(feature.q_value))
            self.baseline_amplitude.append(float(feature.amplitude))

        return BlinkDetectionResult(
            is_event=is_event,
            event_id=event_id,
            method=self.method,
            score=float(score),
            threshold=float(threshold),
            baseline=float(baseline),
            mad=float(mad),
            metrics={
                "amplitude_bump": amplitude_bump,
                "amplitude_range": amplitude_range,
                "viewing_projection": best_projection,
                "gated_projection": gated_projection,
                "phase_stable_projection": phase_stable_projection,
                "viewing_amplitude": phase_stable_projection,
                "viewing_range": projection_range,
                "raw_viewing_score": raw_score,
                "score_scale": score_scale,
                "relative_viewing_score": score,
                "center_i": float(center_i),
                "center_q": float(center_q),
                "baseline_amplitude": baseline_amplitude,
                "origin_amplitude": float(feature.amplitude),
            },
        )


class OldTwinkleDetector:
    """TwinkleTwinkle-inspired phase-pair trajectory detector (7d64a2f version).

    Uses baseband phase trajectory (not FMCW phase_pair_delta) and the simple
    3-point local-peak gate.
    """

    method = "old_twinkle"

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        if config.twinkle_peak_gate_enabled:
            self.gate = _OldTwinklePeakEventGate(config)
        else:
            self.gate = _OldRobustEventGate(config)
        candidate_windows = tuple(
            int(w) for w in getattr(config, "twinkle_candidate_windows", ()) if int(w) > 0
        )
        max_window = max((config.short_window,) + candidate_windows)
        self.unwrapped_phases: Deque[float] = deque(maxlen=config.history_size)
        self.phase_window: Deque[float] = deque(maxlen=max_window)
        self.phase_steps: Deque[float] = deque(maxlen=max_window)
        self.previous_phase = None
        self.current_unwrapped_phase = 0.0
        self.last_ungated_coherence = 0.0

    def _append_unwrapped_phase(self, feature: ChunkFeature):
        if self.previous_phase is None:
            self.current_unwrapped_phase = float(feature.phase)
            phase_step = 0.0
        else:
            phase_step = unwrap_delta(feature.phase, self.previous_phase)
            self.current_unwrapped_phase += phase_step
        self.previous_phase = float(feature.phase)
        self.unwrapped_phases.append(float(self.current_unwrapped_phase))
        self.phase_window.append(float(self.current_unwrapped_phase))
        self.phase_steps.append(float(phase_step))
        return float(self.current_unwrapped_phase), float(phase_step)

    def _trajectory_score(self, window_size: Optional[int] = None):
        if window_size is None:
            window_size = self.config.short_window
        window_size = max(1, int(window_size))
        if len(self.phase_window) < max(4, min(window_size, 4)):
            return 0.0, 0.0, 0.0, 0

        phases = np.asarray(list(self.phase_window)[-window_size:], dtype=np.float64)
        steps = np.asarray(list(self.phase_steps)[-window_size:], dtype=np.float64)
        trajectory_span = float(np.max(phases) - np.min(phases))
        if steps.size >= 2:
            acceleration = np.diff(steps)
            acceleration_rms = float(np.sqrt(np.mean(np.square(acceleration))))
        else:
            acceleration_rms = 0.0

        step_floor = max(0.0, self.config.phase_step_floor)
        signs: List[int] = []
        for step in steps:
            if abs(float(step)) < step_floor:
                continue
            signs.append(1 if step > 0.0 else -1)

        sign_changes = sum(1 for index in range(1, len(signs)) if signs[index] != signs[index - 1])
        if sign_changes == 0:
            if acceleration_rms >= step_floor:
                return float(max(trajectory_span, 2.0 * acceleration_rms)), trajectory_span, acceleration_rms, 0
            return 0.0, trajectory_span, acceleration_rms, 0

        reversal_score = max(trajectory_span, 2.0 * acceleration_rms)
        return float(reversal_score), trajectory_span, acceleration_rms, sign_changes

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        current_phase, phase_step = self._append_unwrapped_phase(feature)
        lag = max(1, int(self.config.phase_pair_lag))
        if len(self.unwrapped_phases) > lag:
            past_phase = list(self.unwrapped_phases)[-1 - lag]
            phase_pair_delta = float(abs(current_phase - past_phase))
        else:
            phase_pair_delta = 0.0

        score, trajectory_span, acceleration_rms, sign_changes = self._trajectory_score()

        force_event = bool(
            getattr(self.config, "absolute_score_floor", 0.0) > 0.0
            and score >= self.config.absolute_score_floor
        )
        if self.config.twinkle_peak_gate_enabled:
            is_event, event_id, baseline, mad, threshold = self.gate.update(
                feature.time_s,
                score,
                feature.motion_energy,
                sign_changes,
            )
            gate_metrics = self.gate.last_gate_metrics
        else:
            is_event, event_id, baseline, mad, threshold = self.gate.update(
                feature.time_s,
                score,
                force_event=force_event,
            )
            gate_metrics = {}
        metrics = {
            "phase_step": phase_step,
            "phase_pair_delta": phase_pair_delta,
            "trajectory_span": trajectory_span,
            "acceleration_rms": acceleration_rms,
            "sign_changes": float(sign_changes),
            "unwrapped_phase": current_phase,
        }
        metrics.update(gate_metrics)
        return BlinkDetectionResult(
            is_event=is_event,
            event_id=event_id,
            method=self.method,
            score=float(score),
            threshold=float(threshold),
            baseline=float(baseline),
            mad=float(mad),
            metrics=metrics,
        )
