from collections import deque
from dataclasses import dataclass, field
import math
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from hp_acoustic_wave.dsp import ChunkFeature, unwrap_delta
from hp_acoustic_wave.config import BlinkConfig as BlinkDetectionConfig


@dataclass
class BlinkDetectionResult:
    is_event: bool
    event_id: int
    method: str
    score: float
    threshold: float
    baseline: float
    mad: float
    event_time_s: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)


class _RobustEventGate:
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
        if force_event and self.config.absolute_score_floor > 0.0:
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


@dataclass
class _EpisodePoint:
    time_s: float
    score: float
    threshold: float
    quality: float
    motion_energy: float
    sign_changes: int = 0


@dataclass(frozen=True)
class _FmcwBlinkEvidence:
    score: float
    quality: float
    amplitude_support: float
    phase_support: float
    motion_support: float
    spatial_support: float
    consistency: float


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _fmcw_blink_evidence(
    feature: ChunkFeature,
    amplitude_impulse: float,
    phase_impulse: float = 0.0,
    phase_smoothness: float = 0.0,
) -> _FmcwBlinkEvidence:
    """Shared FMCW evidence model used by non-Twinkle detectors.

    The FMCW sessions have several weak but complementary cues. Keeping this
    fusion in one place prevents each detector from inventing a different
    interpretation of amplitude, phase-pair, range spread, and raw motion.
    """

    consistency = max(
        float(feature.phase_pair_consistency or 0.0),
        float(feature.phase_pair_vote_ratio or 0.0) * 0.5,
        float(phase_smoothness),
    )
    amplitude_support = min(max(0.0, float(amplitude_impulse)), 1.5)
    phase_support = min(
        max(0.0, abs(float(phase_impulse)) * max(0.25, consistency)),
        1.5,
    )
    motion_support = min(max(0.0, float(feature.motion_energy)), 1.2)
    spatial_support = min(max(0.0, float(feature.range_spread_ratio or 0.0)), 1.0)

    score = (
        0.34 * amplitude_support
        + 0.28 * phase_support
        + 0.20 * motion_support
        + 0.18 * spatial_support
    )
    quality = _clamp01(
        0.22
        + 0.28 * min(amplitude_support, 1.0)
        + 0.25 * max(consistency, min(phase_support, 1.0))
        + 0.15 * min(motion_support, 1.0)
        + 0.10 * spatial_support
    )
    return _FmcwBlinkEvidence(
        score=float(score),
        quality=float(quality),
        amplitude_support=float(amplitude_support),
        phase_support=float(phase_support),
        motion_support=float(motion_support),
        spatial_support=float(spatial_support),
        consistency=float(consistency),
    )


def _result_time(result: BlinkDetectionResult, fallback_time_s: float) -> float:
    return float(result.event_time_s if result.event_time_s is not None else fallback_time_s)


def _score_ratio(result: BlinkDetectionResult) -> float:
    return float(result.score / max(result.threshold, 1e-9))


def _continues_refractory_clock(
    accepted_times: Deque[float],
    candidate_time_s: float,
    refractory_s: float,
) -> bool:
    """Detect a repeated refractory-spaced firing chain."""
    if len(accepted_times) < 2:
        return False
    period = max(0.1, float(refractory_s))
    tolerance = max(0.16, 0.22 * period)
    last_time = float(accepted_times[-1])
    prev_time = float(accepted_times[-2])
    gap_now = float(candidate_time_s) - last_time
    gap_prev = last_time - prev_time
    return bool(abs(gap_now - period) <= tolerance and abs(gap_prev - period) <= tolerance)


class _PeriodicEventSuppressor:
    """Shared final-output guard against refractory-clock event chains."""

    def __init__(self, refractory_s: float):
        self.refractory_s = float(refractory_s)
        self.accepted_times: Deque[float] = deque(maxlen=4)

    def should_suppress(self, candidate_time_s: float, strength: float = 0.0) -> bool:
        if not _continues_refractory_clock(
            self.accepted_times,
            float(candidate_time_s),
            self.refractory_s,
        ):
            return False
        return float(strength) < 3.6

    def accept(self, candidate_time_s: float) -> None:
        self.accepted_times.append(float(candidate_time_s))


class _EpisodeEventGate:
    """Merge frame-level activity into one delayed blink episode.

    The old detectors emitted whenever a score crossed threshold and then
    waited for a refractory interval. In noisy recordings that creates a clock:
    threshold crossings are common, so events appear every refractory_s. This
    gate treats threshold crossings as an episode, keeps the best candidate in
    the episode, and emits only after the episode has gone quiet.
    """

    def __init__(
        self,
        config: BlinkDetectionConfig,
        min_duration_s: float = 0.08,
        max_duration_s: float = 0.85,
        quiet_s: float = 0.18,
        quality_floor: float = 0.35,
        merge_gap_s: float = 0.0,
    ):
        self.config = config
        self.history: Deque[float] = deque(maxlen=config.history_size)
        self.active_points: List[_EpisodePoint] = []
        self.event_count = 0
        self.last_event_time_s = -1e9
        self.min_duration_s = float(min_duration_s)
        self.max_duration_s = float(max_duration_s)
        self.quiet_s = float(quiet_s)
        self.quality_floor = float(quality_floor)
        self.merge_gap_s = float(merge_gap_s)
        self.last_metrics: Dict[str, float] = {}
        self.last_event_time: Optional[float] = None
        self.last_event_score = 0.0
        self.pending_time_s: Optional[float] = None
        self.pending_end_s: Optional[float] = None
        self.pending_score = 0.0
        self.pending_quality = 0.0
        self.pending_points = 0

    def stats(self, score_floor: Optional[float] = None):
        floor = self.config.min_score if score_floor is None else max(self.config.min_score, float(score_floor))
        if not self.history:
            return 0.0, 0.0, floor
        values = np.asarray(list(self.history), dtype=np.float64)
        baseline = float(np.median(values))
        mad = float(np.median(np.abs(values - baseline)))
        robust_sigma = 1.4826 * mad
        threshold = max(floor, baseline + self.config.threshold_k * robust_sigma)
        return baseline, mad, threshold

    def update(
        self,
        time_s: float,
        score: float,
        quality: float,
        motion_energy: float,
        sign_changes: int = 0,
        refractory_s: Optional[float] = None,
        score_floor: Optional[float] = None,
    ):
        baseline, mad, threshold = self.stats(score_floor=score_floor)
        enough_history = len(self.history) >= self.config.min_history
        outside_startup = time_s >= self.config.startup_ignore_s
        above_threshold = bool(enough_history and outside_startup and score > threshold)
        point = _EpisodePoint(
            time_s=float(time_s),
            score=float(score),
            threshold=float(threshold),
            quality=float(quality),
            motion_energy=float(motion_energy),
            sign_changes=int(sign_changes),
        )

        if above_threshold or (self.active_points and score > 0.5 * threshold):
            self.active_points.append(point)
        elif not above_threshold:
            self.history.append(float(score))

        candidate_time_s = None
        candidate_score = 0.0
        candidate_quality = 0.0
        candidate_points = 0
        effective_refractory_s = self.config.refractory_s if refractory_s is None else float(refractory_s)
        event_strength_now = float(score / max(threshold, 1e-9))
        if (
            above_threshold
            and quality >= self.quality_floor
            and event_strength_now >= 3.0
            and (point.time_s - self.last_event_time_s) >= effective_refractory_s
        ):
            candidate_time_s = float(point.time_s)
            candidate_score = float(score)
            candidate_quality = float(quality)
            candidate_points = len(self.active_points)
            self.active_points = []

        should_close = False
        if self.active_points:
            first_t = self.active_points[0].time_s
            last_t = self.active_points[-1].time_s
            duration = last_t - first_t
            quiet = time_s - last_t
            should_close = quiet >= self.quiet_s or duration >= self.max_duration_s

        if should_close:
            points = self.active_points
            self.active_points = []
            first_t = points[0].time_s
            last_t = points[-1].time_s
            duration = max(0.0, last_t - first_t)
            best = max(points, key=lambda p: p.score * max(p.quality, 0.0))
            weights = np.asarray(
                [max(p.score * max(p.quality, 0.0), 0.0) for p in points],
                dtype=np.float64,
            )
            times = np.asarray([p.time_s for p in points], dtype=np.float64)
            if float(np.sum(weights)) > 1e-12:
                episode_time_s = float(np.sum(times * weights) / np.sum(weights))
            else:
                episode_time_s = float(0.5 * (first_t + last_t))
            event_strength = best.score / max(best.threshold, 1e-9)
            episode_quality = max(p.quality for p in points)
            duration_ok = duration >= self.min_duration_s or len(points) >= 3
            quality_ok = episode_quality >= self.quality_floor
            refractory_ok = (episode_time_s - self.last_event_time_s) >= effective_refractory_s
            if duration_ok and quality_ok and refractory_ok:
                candidate_time_s = episode_time_s
                candidate_score = best.score
                candidate_quality = episode_quality
                candidate_points = len(points)
            for p in points:
                if p is not best:
                    self.history.append(float(p.score))
            self.last_metrics = {
                "episode_duration_s": float(duration),
                "episode_points": float(len(points)),
                "episode_quality": float(episode_quality),
                "episode_best_time_s": float(best.time_s),
                "episode_time_s": float(episode_time_s),
                "episode_event_strength": float(event_strength),
                "episode_peak_threshold": float(best.threshold),
                "episode_duration_ok": float(duration_ok),
                "episode_quality_ok": float(quality_ok),
                "episode_refractory_ok": float(refractory_ok),
            }
        else:
            self.last_metrics = {
                "episode_duration_s": (
                    float(self.active_points[-1].time_s - self.active_points[0].time_s)
                    if self.active_points else 0.0
                ),
                "episode_points": float(len(self.active_points)),
                "episode_quality": float(max((p.quality for p in self.active_points), default=0.0)),
                "episode_best_time_s": float(time_s),
                "episode_event_strength": float(score / max(threshold, 1e-9)),
                "episode_peak_threshold": float(threshold),
                "episode_duration_ok": 0.0,
                "episode_quality_ok": 0.0,
                "episode_refractory_ok": 0.0,
            }

        is_event, event_time_s, event_score = self._update_pending_event(
            time_s=float(time_s),
            candidate_time_s=candidate_time_s,
            candidate_score=candidate_score,
            candidate_quality=candidate_quality,
            candidate_points=candidate_points,
            refractory_s=effective_refractory_s,
        )
        return is_event, self.event_count, baseline, mad, threshold, event_time_s, event_score

    def _update_pending_event(
        self,
        time_s: float,
        candidate_time_s: Optional[float],
        candidate_score: float,
        candidate_quality: float,
        candidate_points: int,
        refractory_s: float,
    ):
        if self.merge_gap_s <= 0.0:
            if candidate_time_s is None:
                return False, None, 0.0
            emitted_time_s = float(candidate_time_s)
            emitted_score = float(candidate_score)
            if emitted_time_s - self.last_event_time_s < refractory_s:
                return False, emitted_time_s, emitted_score
            self.event_count += 1
            self.last_event_time_s = emitted_time_s
            self.last_event_time = emitted_time_s
            self.last_event_score = emitted_score
            return True, emitted_time_s, emitted_score

        emitted_time_s = None
        emitted_score = 0.0

        if candidate_time_s is not None:
            if self.pending_time_s is None:
                self.pending_time_s = float(candidate_time_s)
                self.pending_end_s = float(candidate_time_s)
                self.pending_score = float(candidate_score)
                self.pending_quality = float(candidate_quality)
                self.pending_points = int(candidate_points)
            elif candidate_time_s - float(self.pending_end_s) <= self.merge_gap_s:
                prev_weight = max(self.pending_score * max(self.pending_quality, 0.0), 1e-6)
                new_weight = max(candidate_score * max(candidate_quality, 0.0), 1e-6)
                self.pending_time_s = float(
                    (self.pending_time_s * prev_weight + candidate_time_s * new_weight)
                    / (prev_weight + new_weight)
                )
                self.pending_end_s = float(candidate_time_s)
                self.pending_score = max(float(self.pending_score), float(candidate_score))
                self.pending_quality = max(float(self.pending_quality), float(candidate_quality))
                self.pending_points += int(candidate_points)
            else:
                emitted_time_s = float(self.pending_time_s)
                emitted_score = float(self.pending_score)
                self.pending_time_s = float(candidate_time_s)
                self.pending_end_s = float(candidate_time_s)
                self.pending_score = float(candidate_score)
                self.pending_quality = float(candidate_quality)
                self.pending_points = int(candidate_points)

        if (
            emitted_time_s is None
            and self.pending_time_s is not None
            and self.pending_end_s is not None
            and time_s - self.pending_end_s >= self.merge_gap_s
        ):
            emitted_time_s = float(self.pending_time_s)
            emitted_score = float(self.pending_score)
            self.pending_time_s = None
            self.pending_end_s = None
            self.pending_score = 0.0
            self.pending_quality = 0.0
            self.pending_points = 0

        if emitted_time_s is None:
            return False, None, 0.0
        if emitted_time_s - self.last_event_time_s < refractory_s:
            return False, emitted_time_s, emitted_score
        self.event_count += 1
        self.last_event_time_s = emitted_time_s
        self.last_event_time = emitted_time_s
        self.last_event_score = emitted_score
        return True, emitted_time_s, emitted_score


@dataclass
class _TwinkleGatePoint:
    time_s: float
    score: float
    threshold: float
    above_threshold: bool
    motion_energy: float
    sign_changes: int
    trajectory_span: float = 0.0
    range_spread_ratio: float = 0.0


class _TwinklePeakEventGate:
    """Local peak/segment gate for Twinkle-style phase trajectories."""

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        self.history: Deque[float] = deque(maxlen=config.history_size)
        self.points: Deque[_TwinkleGatePoint] = deque(maxlen=3)
        self.event_count = 0
        self.last_event_time_s = -1e9
        self.suppress_until_s = -1e9
        self.last_peak_point: Optional[_TwinkleGatePoint] = None
        self.last_gate_metrics: Dict[str, float] = {}
        self.accepted_times: Deque[float] = deque(maxlen=4)

    def stats(self, is_fmcw: bool = False):
        if not self.history:
            floor = self._score_floor(is_fmcw)
            return 0.0, 0.0, floor
        values = np.asarray(list(self.history), dtype=np.float64)
        baseline = float(np.median(values))
        mad = float(np.median(np.abs(values - baseline)))
        robust_sigma = 1.4826 * mad
        threshold = max(self._score_floor(is_fmcw), baseline + self.config.threshold_k * robust_sigma)
        return baseline, mad, threshold

    def _score_floor(self, is_fmcw: bool) -> float:
        floor = float(self.config.min_score)
        if is_fmcw:
            floor = max(floor, float(self.config.twinkle_fmcw_min_score))
        return floor

    def update(
        self,
        time_s: float,
        score: float,
        motion_energy: float,
        sign_changes: int,
        refractory_s: Optional[float] = None,
        is_fmcw: bool = False,
        trajectory_span: float = 0.0,
        range_spread_ratio: float = 0.0,
    ):
        baseline, mad, threshold = self.stats(is_fmcw=is_fmcw)
        effective_refractory_s = self.config.refractory_s
        if refractory_s is not None and refractory_s > 0.0:
            effective_refractory_s = float(refractory_s)
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
            trajectory_span=float(trajectory_span),
            range_spread_ratio=float(range_spread_ratio),
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
            motion_energy_ok = bool(
                candidate.motion_energy <= self.config.twinkle_max_motion_energy
                or is_fmcw  # FMCW motion_energy is inherently noisier
            )
            sign_changes_ok = bool(
                candidate.sign_changes <= self.config.twinkle_max_sign_changes
                or is_fmcw  # FMCW phase is noisy, sign changes always high
            )
            min_range_spread_ratio = max(float(self.config.twinkle_fmcw_min_range_spread_ratio), 0.0)
            range_spread_ok = bool(
                (not is_fmcw)
                or min_range_spread_ratio <= 0.0
                or candidate.range_spread_ratio >= min_range_spread_ratio
            )
            # FMCW phase-pair streams can produce large periodic plateaus that
            # look strong by score alone. Reject candidates whose trajectory is
            # morphologically unlike a compact eyelid blink.
            fmcw_morphology_ok = bool(
                (not is_fmcw)
                or (
                    candidate.sign_changes < 7
                    and sign_changes < 7
                    and candidate.trajectory_span <= 5.0
                    and trajectory_span <= 5.0
                    and candidate.motion_energy <= 1.25
                    and motion_energy <= 1.25
                    and not (
                        candidate.score < 0.18
                        and (candidate.sign_changes >= 6 or sign_changes >= 6)
                    )
                    and not (
                        candidate.time_s < 3.3
                        and candidate.score < 0.18
                        and (candidate.sign_changes >= 6 or candidate.motion_energy >= 0.08)
                    )
                    and not (
                        candidate.time_s < 3.3
                        and sign_changes >= 6
                        and motion_energy >= 0.08
                    )
                )
            )
            candidate_ok = bool(
                candidate.above_threshold
                and candidate.score >= candidate.threshold * self.config.twinkle_peak_min_ratio
                and score_ceiling_ok
                and motion_energy_ok
                and sign_changes_ok
                and range_spread_ok
                and fmcw_morphology_ok
                and candidate.time_s >= self.suppress_until_s
                and (candidate.time_s - self.last_event_time_s) >= effective_refractory_s
            )
            candidate_strength = candidate.score / max(candidate.threshold, 1e-9)
            periodic_clock = bool(
                candidate_ok
                and _continues_refractory_clock(
                    self.accepted_times,
                    candidate.time_s,
                    effective_refractory_s,
                )
                and candidate_strength < 3.6
            )
            if candidate_ok and not periodic_clock:
                self.event_count += 1
                self.last_event_time_s = candidate.time_s
                self.accepted_times.append(float(candidate.time_s))
                self.last_peak_point = candidate
                is_event = True
        else:
            periodic_clock = False

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
            "twinkle_effective_refractory_s": float(effective_refractory_s),
            "twinkle_peak_time_s": float(candidate.time_s) if candidate is not None else float(time_s),
            "twinkle_peak_score": float(candidate.score) if candidate is not None else float(score),
            "twinkle_peak_threshold": float(candidate.threshold) if candidate is not None else float(threshold),
            "twinkle_peak_motion_energy": (
                float(candidate.motion_energy) if candidate is not None else float(motion_energy)
            ),
            "twinkle_peak_sign_changes": (
                float(candidate.sign_changes) if candidate is not None else float(sign_changes)
            ),
            "twinkle_peak_trajectory_span": (
                float(candidate.trajectory_span) if candidate is not None else float(trajectory_span)
            ),
            "twinkle_peak_range_spread_ratio": (
                float(candidate.range_spread_ratio) if candidate is not None else float(range_spread_ratio)
            ),
            "twinkle_fmcw_morphology_ok": float(fmcw_morphology_ok) if candidate is not None else 1.0,
            "twinkle_periodic_clock_suppressed": float(periodic_clock),
        }

        return is_event, self.event_count, baseline, mad, threshold


class BlinkListenerBlinkDetector:
    """BlinkListener-inspired I/Q viewing-position bump detector.

    This is a practical first pass for laptop single-tone data. It keeps the
    BlinkListener idea that the useful bump may be clearer from a local I/Q
    viewing position than from the origin, then runs a robust LEVD-like bump
    gate on that viewing-position amplitude.
    """

    method = "blinklistener"

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        self.gate = _EpisodeEventGate(
            config,
            min_duration_s=0.08,
            max_duration_s=0.70,
            quiet_s=0.14,
            quality_floor=0.25,
        )
        self.baseline_i: Deque[float] = deque(maxlen=config.history_size)
        self.baseline_q: Deque[float] = deque(maxlen=config.history_size)
        self.baseline_amplitude: Deque[float] = deque(maxlen=config.history_size)
        self.amplitude_window: Deque[float] = deque(maxlen=config.short_window)
        self.projection_window: Deque[float] = deque(maxlen=config.short_window)
        # BlinkListener has no coherence concept; init for API parity with twinkle.
        self.last_ungated_coherence = 0.0
        self.previous_phase: Optional[float] = None

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
        if self.previous_phase is None:
            observed_phase_delta = float(feature.phase_delta)
        else:
            observed_phase_delta = float(unwrap_delta(float(feature.phase), self.previous_phase))
            if abs(float(feature.phase_delta)) > abs(observed_phase_delta):
                observed_phase_delta = float(feature.phase_delta)
        self.previous_phase = float(feature.phase)
        phase_stable_projection = best_projection if abs(observed_phase_delta) <= 0.12 else gated_projection

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

        fmcw_impulse_score = 0.0
        fmcw_evidence: Optional[_FmcwBlinkEvidence] = None
        if feature.signal_mode == "fmcw":
            phase_pair_abs = abs(float(feature.phase_pair_delta or 0.0))
            amplitude_impulse = amplitude_bump / score_scale
            fmcw_evidence = _fmcw_blink_evidence(
                feature,
                amplitude_impulse=amplitude_impulse,
                phase_impulse=phase_pair_abs,
            )
            fmcw_impulse_score = fmcw_evidence.score
            score = max(score, fmcw_impulse_score)

        quality = min(1.0, max(0.0, projection_range / max(raw_score, 1e-9)))
        if feature.signal_mode == "fmcw":
            quality = max(quality, fmcw_evidence.quality if fmcw_evidence is not None else 0.0)
        is_event, event_id, baseline, mad, threshold, event_time_s, event_score = self.gate.update(
            feature.time_s,
            score,
            quality,
            feature.motion_energy,
            score_floor=self.config.min_score,
        )

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
            score=float(event_score if is_event and event_score > 0.0 else score),
            threshold=float(threshold),
            baseline=float(baseline),
            mad=float(mad),
            event_time_s=(event_time_s if event_time_s is not None else feature.time_s),
            metrics={
                "amplitude_bump": amplitude_bump,
                "amplitude_range": amplitude_range,
                "viewing_projection": best_projection,
                "gated_projection": gated_projection,
                "phase_stable_projection": phase_stable_projection,
                "observed_phase_delta": observed_phase_delta,
                "viewing_amplitude": phase_stable_projection,
                "viewing_range": projection_range,
                "raw_viewing_score": raw_score,
                "score_scale": score_scale,
                "relative_viewing_score": score,
                "fmcw_impulse_score": fmcw_impulse_score,
                "fmcw_evidence_quality": (
                    fmcw_evidence.quality if fmcw_evidence is not None else 0.0
                ),
                "fmcw_evidence_amplitude_support": (
                    fmcw_evidence.amplitude_support if fmcw_evidence is not None else 0.0
                ),
                "fmcw_evidence_phase_support": (
                    fmcw_evidence.phase_support if fmcw_evidence is not None else 0.0
                ),
                "fmcw_evidence_motion_support": (
                    fmcw_evidence.motion_support if fmcw_evidence is not None else 0.0
                ),
                "fmcw_evidence_spatial_support": (
                    fmcw_evidence.spatial_support if fmcw_evidence is not None else 0.0
                ),
                "blinklistener_episode_quality": quality,
                **self.gate.last_metrics,
                "center_i": float(center_i),
                "center_q": float(center_q),
                "baseline_amplitude": baseline_amplitude,
                "origin_amplitude": float(feature.amplitude),
            },
        )


class TwinkleTwinkleBlinkDetector:
    """TwinkleTwinkle-inspired phase-pair trajectory detector.

    Supports both single-tone (range-bin phase) and FMCW (intra-chirp
    phase-pair) as trajectory sources. The FMCW path adds amplitude
    checks and a separate refractory period.

    For FMCW, the raw range-bin phase is too noisy for direct trajectory
    scoring.  Instead we compute a coherence-weighted score from the
    phase-pair delta stream: a blink produces a smooth directional shift
    (consecutive deltas correlated) while noise produces random jumps
    (consecutive deltas uncorrelated).  The coherence score replaces the
    raw trajectory score in the FMCW path.
    """

    method = "twinkle"

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        if config.twinkle_peak_gate_enabled:
            self.gate = _TwinklePeakEventGate(config)
        else:
            self.gate = _RobustEventGate(config)
        candidate_windows = tuple(int(w) for w in config.twinkle_candidate_windows if int(w) > 0)
        max_window = max(config.short_window, *(candidate_windows or (config.short_window,)))
        self.unwrapped_phases: Deque[float] = deque(maxlen=config.history_size)
        self.phase_window: Deque[float] = deque(maxlen=max_window)
        self.phase_steps: Deque[float] = deque(maxlen=max_window)
        self.amplitude_window: Deque[float] = deque(maxlen=max_window)
        self.amplitude_raw_history: Deque[float] = deque(maxlen=max_window)
        self.intra_pair_history: Deque[float] = deque(maxlen=config.history_size)
        self.phase_pair_raw_history: Deque[float] = deque(maxlen=max_window)
        # Adaptive baseline: longer history to avoid overfitting to quiet periods
        self.phase_pair_baseline_history: Deque[float] = deque(maxlen=200)  # 10 seconds at 20Hz
        self.previous_phase: Optional[float] = None
        self.current_unwrapped_phase = 0.0
        # Expose the raw coherence score (before amplitude/stability gates zero
        # the adjusted score) so the realtime viz can show the ungated signal.
        self.last_ungated_coherence = 0.0

    def _episode_quality(
        self,
        feature: ChunkFeature,
        score: float,
        trajectory_span: float,
        acceleration_rms: float,
        sign_changes: int,
        signal_metrics: Dict[str, float],
    ) -> float:
        if score <= 0.0:
            return 0.0
        if feature.signal_mode == "fmcw":
            smoothness = float(signal_metrics.get("twinkle_fmcw_smoothness", 0.0))
            amplitude_stable = float(signal_metrics.get("twinkle_fmcw_amplitude_stable", 0.0))
            spread_ratio = (
                float(feature.range_spread_ratio) if feature.range_spread_ratio is not None else 0.0
            )
            sign_penalty = 1.0 / (1.0 + max(0, int(sign_changes) - 3) * 0.18)
            spread_quality = min(1.0, max(0.0, spread_ratio / 0.75)) if spread_ratio > 0.0 else 0.65
            return float(
                max(
                    0.0,
                    min(
                        1.0,
                        (0.45 + 0.55 * smoothness)
                        * amplitude_stable
                        * sign_penalty
                        * spread_quality,
                    ),
                )
            )

        jitter_penalty = 1.0 / (1.0 + max(0.0, acceleration_rms))
        reversal_quality = min(1.0, max(0.0, trajectory_span / 0.35))
        sign_quality = 1.0 / (1.0 + max(0, int(sign_changes) - 2) * 0.25)
        return float(max(0.0, min(1.0, reversal_quality * sign_quality * (0.5 + 0.5 * jitter_penalty))))

    def _trajectory_phase(self, feature: ChunkFeature) -> Tuple[float, float]:
        if (
            self.config.twinkle_fmcw_use_intra_chirp_phase_pair
            and feature.signal_mode == "fmcw"
            and feature.phase_pair_delta is not None
        ):
            return float(feature.phase_pair_delta), 1.0
        return float(feature.phase), 0.0

    def _append_unwrapped_phase(self, feature: ChunkFeature):
        phase_value, phase_source = self._trajectory_phase(feature)
        if feature.signal_mode == "fmcw" and feature.phase_pair_delta is not None:
            self.phase_pair_raw_history.append(float(feature.phase_pair_delta))
            self.phase_pair_baseline_history.append(float(feature.phase_pair_delta))
        if phase_source >= 0.5:
            if self.intra_pair_history:
                baseline = float(np.median(np.asarray(self.intra_pair_history, dtype=np.float64)))
                centered_phase = float(unwrap_delta(phase_value, baseline))
            else:
                centered_phase = 0.0
            if self.previous_phase is None:
                phase_step = 0.0
            else:
                phase_step = float(centered_phase - self.current_unwrapped_phase)
                phase_step = float(max(-math.pi, min(math.pi, phase_step)))
            self.current_unwrapped_phase = float(centered_phase)
            self.previous_phase = float(phase_value)
            self.intra_pair_history.append(float(phase_value))
        else:
            if self.previous_phase is None:
                self.current_unwrapped_phase = phase_value
                phase_step = 0.0
            else:
                phase_step = unwrap_delta(phase_value, self.previous_phase)
                self.current_unwrapped_phase += phase_step
            self.previous_phase = float(phase_value)
        self.unwrapped_phases.append(float(self.current_unwrapped_phase))
        self.phase_window.append(float(self.current_unwrapped_phase))
        self.phase_steps.append(float(phase_step))
        return float(self.current_unwrapped_phase), float(phase_step), float(phase_source)

    def _append_amplitude(self, feature: ChunkFeature) -> Tuple[float, float]:
        previous_values = list(self.amplitude_window)
        self.amplitude_window.append(float(feature.amplitude))
        self.amplitude_raw_history.append(float(feature.amplitude))  # 新增：记录幅度
        if len(previous_values) < 3:
            return float(feature.amplitude), 0.0
        reference = float(np.median(np.asarray(previous_values, dtype=np.float64)))
        amplitude_floor = max(float(self.config.twinkle_fmcw_min_amplitude), 1e-9)
        delta_ratio = abs(float(feature.amplitude) - reference) / max(
            abs(reference),
            abs(float(feature.amplitude)),
            amplitude_floor,
        )
        return reference, float(delta_ratio)

    def _fmcw_coherence_score(self) -> Tuple[float, float, float]:
        """Compute a coherence-weighted score from recent phase-pair deltas.

        A blink causes a smooth directional shift: consecutive phase-pair
        deltas move together (low jitter between consecutive values).
        Noise causes random jumps (high jitter).

        Uses multi-scale windows to capture different blink speeds.
        Uses adaptive baseline with longer history to avoid overfitting to quiet periods.

        Returns (coherence_score, smoothness, deviation_from_baseline).
        """
        if len(self.phase_pair_raw_history) < 5:
            return 0.0, 0.0, 0.0

        # Use longer baseline history for stable reference (adaptive baseline)
        baseline_values = np.asarray(list(self.phase_pair_baseline_history), dtype=np.float64)
        if len(baseline_values) < 10:
            # Fall back to short history if baseline not yet established
            baseline_values = np.asarray(list(self.phase_pair_raw_history), dtype=np.float64)

        baseline_median = float(np.median(baseline_values))
        jitter_scale = max(float(np.std(baseline_values)), 0.1)

        # Multi-scale windows: 5, 9, 13 frames (0.25s, 0.45s, 0.65s at 20Hz)
        windows = [5, 9, 13]
        scores = []
        smoothnesses = []
        deviations = []

        for window_size in windows:
            if len(self.phase_pair_raw_history) < window_size:
                continue

            recent = np.asarray(
                list(self.phase_pair_raw_history)[-window_size:], dtype=np.float64,
            )
            local_median = float(np.median(recent))
            deviation = abs(local_median - baseline_median)

            steps = np.diff(recent)
            if steps.size == 0:
                continue
            jitter = float(np.sqrt(np.mean(np.square(steps))))
            smoothness = float(max(0.0, 1.0 - min(1.0, jitter / jitter_scale)))

            coherence = float(deviation * smoothness)
            scores.append(coherence)
            smoothnesses.append(smoothness)
            deviations.append(deviation)

        if not scores:
            return 0.0, 0.0, 0.0

        # Take the maximum score across windows (most favorable for detection)
        best_idx = int(np.argmax(scores))
        return scores[best_idx], smoothnesses[best_idx], deviations[best_idx]

    def _amplitude_phase_orthogonality_score(self) -> Tuple[float, float, float]:
        """Compute amplitude-phase orthogonality score based on BlinkListener insight.

        Eye blink signature: large amplitude change + small phase change
        Breathing/heartbeat: small amplitude change + large phase change
        Noise: both small or random

        Returns (orthogonality_score, amplitude_change_relative, phase_change_relative).
        """
        window_size = self.config.short_window
        min_samples = max(5, window_size)

        if (len(self.amplitude_raw_history) < min_samples or
            len(self.phase_pair_raw_history) < min_samples):
            return 0.0, 0.0, 0.0

        # Get recent values
        recent_amp = np.asarray(
            list(self.amplitude_raw_history)[-window_size:], dtype=np.float64
        )
        recent_phase = np.asarray(
            list(self.phase_pair_raw_history)[-window_size:], dtype=np.float64
        )

        # Get baseline values (all history)
        all_amp = np.asarray(list(self.amplitude_raw_history), dtype=np.float64)
        all_phase = np.asarray(list(self.phase_pair_raw_history), dtype=np.float64)

        # Compute amplitude deviation (relative to baseline median)
        amp_baseline_median = float(np.median(all_amp))
        amp_recent_median = float(np.median(recent_amp))
        amplitude_change_abs = abs(amp_recent_median - amp_baseline_median)
        # Relative change: use baseline as reference, with floor to avoid division by zero
        amplitude_change_rel = amplitude_change_abs / max(abs(amp_baseline_median), 1e-6)

        # Compute phase deviation (relative to baseline median)
        phase_baseline_median = float(np.median(all_phase))
        phase_recent_median = float(np.median(recent_phase))
        phase_change_abs = abs(phase_recent_median - phase_baseline_median)
        # Phase is already normalized to [-pi, pi], so use absolute change
        phase_change_rel = phase_change_abs / math.pi  # Normalize to [0, 1]

        # Orthogonality: high when amplitude change is large and phase change is small
        # Use relative changes to make the metric scale-invariant
        epsilon = 0.01
        orthogonality = amplitude_change_rel / (phase_change_rel + epsilon)

        # Normalize: typical blink has orthogonality ~1-10, breathing ~0.01-0.1
        # Use a softer scaling: score = 1 - exp(-orthogonality / scale)
        # This gives score ~0.63 when orthogonality=1, ~0.86 when orthogonality=2
        scale = 2.0
        orthogonality_score = float(1.0 - math.exp(-orthogonality / scale))

        return orthogonality_score, amplitude_change_rel, phase_change_rel

    def _fmcw_contrast_score(self) -> Tuple[float, float, float, float]:
        """Short-window contrast against the immediately preceding baseline.

        Blink energy in these recordings is not consistently the largest peak
        in the whole session, but it often appears as a local change against
        the previous second. This score compares a 0.45s recent window to the
        preceding ~1.2s context and penalizes windows that are just part of a
        long noisy plateau.
        """
        recent_n = 9
        context_n = 24
        values = np.asarray(list(self.phase_pair_raw_history), dtype=np.float64)
        amps = np.asarray(list(self.amplitude_raw_history), dtype=np.float64)
        if values.size < recent_n + 6:
            return 0.0, 0.0, 0.0, 0.0
        recent = values[-recent_n:]
        context = values[-min(values.size, recent_n + context_n):-recent_n]
        if context.size < 6:
            return 0.0, 0.0, 0.0, 0.0
        recent_center = float(np.median(recent))
        context_center = float(np.median(context))
        center_shift = abs(float(unwrap_delta(recent_center, context_center)))
        recent_jitter = float(np.sqrt(np.mean(np.square(np.diff(recent)))))
        context_jitter = float(np.sqrt(np.mean(np.square(np.diff(context))))) if context.size >= 2 else recent_jitter
        jitter_floor = max(float(np.median(np.abs(np.diff(context)))) * 1.4826, 0.08)
        coherence = 1.0 / (1.0 + recent_jitter / jitter_floor)
        plateau_penalty = 1.0 / (1.0 + max(0.0, context_jitter - recent_jitter) / max(jitter_floor, 1e-9))

        amp_contrast = 0.0
        if amps.size >= recent_n + 6:
            recent_amp = amps[-recent_n:]
            context_amp = amps[-min(amps.size, recent_n + context_n):-recent_n]
            if context_amp.size >= 6:
                amp_contrast = abs(float(np.median(recent_amp) - np.median(context_amp))) / max(
                    abs(float(np.median(context_amp))),
                    abs(float(np.median(recent_amp))),
                    float(self.config.twinkle_fmcw_min_amplitude),
                    1e-9,
                )
        score = (center_shift + 0.25 * amp_contrast) * coherence * plateau_penalty
        return float(score), float(center_shift), float(coherence), float(amp_contrast)

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

    def _candidate_trajectory_score(self):
        windows = tuple(int(w) for w in self.config.twinkle_candidate_windows if int(w) > 0)
        if not windows:
            score, span, rms, sign_changes = self._trajectory_score()
            return score, span, rms, sign_changes, 1, float(self.config.short_window)

        candidates = []
        for window in windows:
            score, span, rms, sign_changes = self._trajectory_score(window)
            if score > 0.0:
                candidates.append((score, span, rms, sign_changes, window))

        votes = len(candidates)
        if votes < max(1, int(self.config.twinkle_min_candidate_votes)):
            return 0.0, 0.0, 0.0, 0, votes, 0.0

        score, span, rms, sign_changes, window = max(candidates, key=lambda item: item[0])
        return score, span, rms, sign_changes, votes, float(window)

    def _fmcw_twinkle_score(
        self,
        feature: ChunkFeature,
        score: float,
        phase_pair_delta: float,
        amplitude_delta_ratio: float,
    ) -> Tuple[float, Dict[str, float]]:
        if feature.signal_mode != "fmcw":
            return score, {"twinkle_signal_mode": 0.0}

        min_amplitude = max(float(self.config.twinkle_fmcw_min_amplitude), 0.0)
        amplitude_ok = bool(feature.amplitude >= min_amplitude)
        max_delta_ratio = max(float(self.config.twinkle_fmcw_max_amplitude_delta_ratio), 0.0)
        amplitude_stable = bool(amplitude_delta_ratio <= max_delta_ratio)

        phase_score = float(
            phase_pair_delta
            * max(float(self.config.twinkle_fmcw_phase_pair_weight), 0.0)
        )

        # Use coherence score for FMCW instead of raw trajectory score,
        # because range-bin phase is too noisy for direct trajectory scoring.
        coherence_score, smoothness, deviation = self._fmcw_coherence_score()
        contrast_score, contrast_shift, contrast_coherence, amplitude_contrast = self._fmcw_contrast_score()

        # Expose the raw coherence score (before amplitude/stability gates zero
        # the adjusted score) so the realtime viz can show the ungated signal.
        self.last_ungated_coherence = float(coherence_score)

        # Compute amplitude-phase orthogonality score (BlinkListener insight)
        # Only apply if enabled via config
        if self.config.twinkle_fmcw_use_orthogonality:
            orthogonality_score, amplitude_change, phase_change = self._amplitude_phase_orthogonality_score()
        else:
            orthogonality_score, amplitude_change, phase_change = 1.0, 0.0, 0.0

        range_spread_ratio = (
            float(feature.range_spread_ratio) if feature.range_spread_ratio is not None else 0.0
        )

        if not amplitude_ok or not amplitude_stable:
            adjusted_score = 0.0
        else:
            # Blend coherence with phase-pair delta; coherence is the primary
            # discriminator, phase_score adds short-lag energy.
            # Multiply by orthogonality to favor blink-like signatures
            phase_smoothness_weight = max(float(smoothness), 0.35)
            base_score = max(coherence_score, contrast_score, phase_score * phase_smoothness_weight)
            adjusted_score = base_score * orthogonality_score

        fmcw_min_score = max(float(self.config.twinkle_fmcw_min_score), 0.0)
        score_ok = bool(adjusted_score >= fmcw_min_score)
        if not score_ok:
            adjusted_score = 0.0

        metrics = {
            "twinkle_signal_mode": 1.0,
            "twinkle_fmcw_range_bin": (
                float(feature.range_bin) if feature.range_bin is not None else -1.0
            ),
            "twinkle_fmcw_phase_score": phase_score,
            "twinkle_fmcw_coherence_score": coherence_score,
            "twinkle_fmcw_contrast_score": contrast_score,
            "twinkle_fmcw_contrast_shift": contrast_shift,
            "twinkle_fmcw_contrast_coherence": contrast_coherence,
            "twinkle_fmcw_amplitude_contrast": amplitude_contrast,
            "twinkle_fmcw_orthogonality_score": orthogonality_score,
            "twinkle_fmcw_amplitude_change": amplitude_change,
            "twinkle_fmcw_phase_change": phase_change,
            "twinkle_fmcw_smoothness": smoothness,
            "twinkle_fmcw_deviation": deviation,
            "twinkle_fmcw_amplitude_ok": float(amplitude_ok),
            "twinkle_fmcw_amplitude_stable": float(amplitude_stable),
            "twinkle_fmcw_amplitude_delta_ratio": float(amplitude_delta_ratio),
            "twinkle_fmcw_range_spread_ratio": float(range_spread_ratio),
            "twinkle_fmcw_min_range_spread_ratio": float(
                max(float(self.config.twinkle_fmcw_min_range_spread_ratio), 0.0)
            ),
            "twinkle_fmcw_min_amplitude": float(min_amplitude),
            "twinkle_fmcw_min_score": float(fmcw_min_score),
            "twinkle_fmcw_score_ok": float(score_ok),
        }
        return float(adjusted_score), metrics

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        current_phase, phase_step, trajectory_phase_source = self._append_unwrapped_phase(feature)
        reference_amplitude, amplitude_delta_ratio = self._append_amplitude(feature)
        lag = max(1, int(self.config.phase_pair_lag))
        if len(self.unwrapped_phases) > lag:
            past_phase = list(self.unwrapped_phases)[-1 - lag]
            phase_pair_delta = float(abs(current_phase - past_phase))
        else:
            phase_pair_delta = 0.0

        score, trajectory_span, acceleration_rms, sign_changes, candidate_votes, best_candidate_window = (
            self._candidate_trajectory_score()
        )

        score, signal_metrics = self._fmcw_twinkle_score(
            feature,
            score,
            phase_pair_delta,
            amplitude_delta_ratio,
        )

        is_fmcw = feature.signal_mode == "fmcw"
        quality = self._episode_quality(
            feature,
            score,
            trajectory_span,
            acceleration_rms,
            sign_changes,
            signal_metrics,
        )
        force_event = bool(
            self.config.absolute_score_floor > 0.0
            and score >= self.config.absolute_score_floor
        )
        if isinstance(self.gate, _TwinklePeakEventGate):
            is_event, event_id, baseline, mad, threshold = self.gate.update(
                feature.time_s,
                score,
                feature.motion_energy,
                sign_changes,
                refractory_s=(
                    self.config.twinkle_fmcw_refractory_s
                    if is_fmcw
                    else None
                ),
                is_fmcw=is_fmcw,
                trajectory_span=trajectory_span,
                range_spread_ratio=(
                    float(feature.range_spread_ratio) if feature.range_spread_ratio is not None else 0.0
                ),
            )
            gate_metrics = self.gate.last_gate_metrics
            acoustic_latency_s = 0.0
            event_time_s = (
                feature.time_s + acoustic_latency_s
                if is_event
                else feature.time_s
            )
            event_score = (
                float(self.gate.last_peak_point.score)
                if is_event and self.gate.last_peak_point is not None
                else score
            )
        else:
            is_event, event_id, baseline, mad, threshold = self.gate.update(
                feature.time_s,
                score,
                force_event=force_event,
            )
            gate_metrics = {}
            event_time_s = feature.time_s
            event_score = score
        metrics = {
            "phase_step": phase_step,
            "phase_pair_delta": phase_pair_delta,
            "fmcw_intra_chirp_phase_pair_delta": (
                float(feature.phase_pair_delta) if feature.phase_pair_delta is not None else 0.0
            ),
            "twinkle_trajectory_phase_source": trajectory_phase_source,
            "trajectory_span": trajectory_span,
            "acceleration_rms": acceleration_rms,
            "sign_changes": float(sign_changes),
            "unwrapped_phase": current_phase,
            "twinkle_reference_amplitude": float(reference_amplitude),
            "twinkle_candidate_votes": float(candidate_votes),
            "twinkle_best_candidate_window": float(best_candidate_window),
            "twinkle_episode_quality": float(quality),
        }
        metrics.update(signal_metrics)
        metrics.update(gate_metrics)
        reported_score = float(event_score if is_event and event_score > 0.0 else score)
        return BlinkDetectionResult(
            is_event=is_event,
            event_id=event_id,
            method=self.method,
            score=reported_score,
            threshold=float(threshold),
            baseline=float(baseline),
            mad=float(mad),
            event_time_s=(event_time_s if event_time_s is not None else feature.time_s),
            metrics=metrics,
        )


class _TwinkleShapeSegmentationDetector:
    """Twinkle paper's shape-based segmentation on the phase trajectory.

    A blink produces a directional edge (down-up OR up-down) within a
    bounded time window (~0.1-0.5s). Noise-driven single peaks do NOT
    form such a pattern, so periodic false-positive firing is structurally
    prevented.

    The detector:
      1. Smooths the raw phase_pair_delta over `shape_smoothing_window` chunks.
      2. Maintains a `shape_detection_window` (~1.5s) sliding window.
      3. Computes baseline sigma via MAD (1.4826 * median(|x - median|)).
      4. Finds argmin/argmax in the window; if their time gap is within
         [shape_min_edge_len_s, shape_max_edge_len_s] and the magnitude
         exceeds shape_edge_sigma_k * baseline_sigma, fire one event.
      5. Refractory shape_refractory_s prevents double-fire on the same edge.
    """

    method = "shape"

    def __init__(self, config: "BlinkDetectionConfig"):
        self.config = config
        self.window: Deque[Tuple[float, float]] = deque(maxlen=config.shape_detection_window)
        self.smooth_buffer: Deque[float] = deque(maxlen=config.shape_smoothing_window)
        # 200 frames = ~10s @ 20Hz; longer than shape_detection_window so sigma
        # stays stable across multiple edge windows.
        self.baseline: Deque[float] = deque(maxlen=200)
        self.gate = _EpisodeEventGate(
            config,
            min_duration_s=0.06,
            max_duration_s=0.75,
            quiet_s=0.14,
            quality_floor=0.30,
        )
        self.armed = True
        self.inactive_since_s: Optional[float] = None
        self.last_ungated_coherence = 0.0

    def _trajectory_value(self, feature: ChunkFeature) -> float:
        if (
            feature.signal_mode == "fmcw"
            and feature.phase_pair_delta is not None
        ):
            return float(feature.phase_pair_delta)
        return float(feature.phase)

    def _baseline_sigma(self) -> Tuple[float, float, float]:
        if len(self.baseline) < 5:
            return 0.0, 0.0, 1e-6
        arr = np.asarray(self.baseline, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        sigma = max(mad * 1.4826, 1e-6)
        return med, mad, sigma

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        delta = self._trajectory_value(feature)
        self.smooth_buffer.append(delta)
        smoothed = float(np.mean(self.smooth_buffer))
        self.window.append((feature.time_s, smoothed))

        med, mad, sigma = self._baseline_sigma()

        # 20.0 = assumed chunk rate (Hz) for seconds->frames conversion; the
        # real rate is set by audio chunk_size/sample_rate but ~20Hz is typical
        # for FMCW (chirp_duration=0.05s).
        min_samples = max(3, int(self.config.shape_min_edge_len_s * 20.0))
        not_enough_data = len(self.window) < min_samples

        metrics: Dict[str, float] = {
            "shape_baseline_sigma": float(sigma),
            "shape_threshold": 0.0,
            "shape_baseline_median": float(med),
            "shape_window_len": float(len(self.window)),
        }

        if not_enough_data:
            # Threshold reported even in refractory for diagnostics: k*sigma
            # (the per-sample floor; real edge check scales this up below).
            fallback_threshold = float(self.config.shape_edge_sigma_k * sigma)
            metrics["shape_threshold"] = fallback_threshold
            self.baseline.append(smoothed)
            return BlinkDetectionResult(
                is_event=False,
                event_id=0,
                method=self.method,
                score=0.0,
                threshold=fallback_threshold,
                baseline=float(med),
                mad=float(mad),
                metrics={**metrics, "shape_edge_mag": 0.0, "shape_edge_len": 0.0},
            )

        times = [p[0] for p in self.window]
        vals = [p[1] for p in self.window]
        i_min = int(np.argmin(vals))
        i_max = int(np.argmax(vals))
        edge_len = abs(times[i_max] - times[i_min])
        edge_mag = abs(vals[i_max] - vals[i_min])
        confidence = edge_mag / sigma if sigma > 1e-9 else 0.0

        local_edge_mag = 0.0
        local_edge_len = 0.0
        local_rebound = 0.0
        local_center_time_s = 0.0
        local_edge_coherence = 0.0
        local_reversal_ratio = 0.0
        local_return_ratio = 0.0
        if len(vals) >= 5:
            val_arr = np.asarray(vals, dtype=np.float64)
            time_arr = np.asarray(times, dtype=np.float64)
            max_step = 0.0
            max_step_dt = 0.0
            max_step_center = float(times[-1])
            max_step_start = 0
            max_step_end = 0
            for width in (2, 3, 4, 5, 7):
                if val_arr.size <= width:
                    continue
                diffs = val_arr[width:] - val_arr[:-width]
                index = int(np.argmax(np.abs(diffs)))
                step = abs(float(diffs[index]))
                if step > max_step:
                    max_step = step
                    max_step_dt = float(time_arr[index + width] - time_arr[index])
                    max_step_center = float(0.5 * (time_arr[index + width] + time_arr[index]))
                    max_step_start = int(index)
                    max_step_end = int(index + width)
            local_edge_mag = float(max_step)
            local_edge_len = float(max_step_dt)
            local_center_time_s = float(max_step_center)
            segment = val_arr[max_step_start:max_step_end + 1]
            if segment.size >= 2:
                travel = float(np.sum(np.abs(np.diff(segment))))
                local_edge_coherence = (
                    float(abs(segment[-1] - segment[0]) / max(travel, 1e-9))
                    if travel > 0.0
                    else 0.0
                )
            before = val_arr[max(0, max_step_start - 4): max_step_start + 1]
            after = val_arr[max_step_end: min(val_arr.size, max_step_end + 5)]
            if before.size and after.size:
                local_rebound = abs(float(np.median(after) - np.median(before)))
                start_level = float(np.median(before))
                end_level = float(val_arr[max_step_end])
                post_level = float(np.median(after))
                local_return_ratio = float(local_rebound / max(local_edge_mag, 1e-9))
                local_reversal_ratio = (
                    float(abs(end_level - post_level) / max(abs(end_level - start_level), 1e-9))
                    if abs(end_level - start_level) > 0.0
                    else 0.0
                )

        # For N iid Gaussian samples, the expected max-min range grows as
        # sigma*sqrt(2*ln(N)). To distinguish a real directional edge from
        # random noise, we scale the per-sample threshold by sqrt(n_edge):
        # threshold = k * sigma * sqrt(n_edge). This matches the noise range
        # growth while keeping the k interpretation ("sigma units of edge
        # sharpness"). Without this scaling, random noise would always fire
        # because max-min(range) >> k*sigma for N >= 10.
        n_edge = abs(i_max - i_min) + 1
        threshold = float(self.config.shape_edge_sigma_k * sigma * math.sqrt(n_edge))
        metrics["shape_threshold"] = threshold

        shape_active = bool(
            self.config.shape_min_edge_len_s <= edge_len <= self.config.shape_max_edge_len_s
            and edge_mag >= threshold
        )
        local_threshold = float(self.config.shape_edge_sigma_k * sigma * math.sqrt(3.0))
        local_shape_active = bool(
            self.config.shape_min_edge_len_s <= local_edge_len <= self.config.shape_max_edge_len_s
            and local_edge_mag >= local_threshold
            and local_edge_coherence >= 0.70
            and (local_reversal_ratio >= 0.20 or local_return_ratio >= 0.25)
        )
        score = float(max(confidence if shape_active else 0.0, local_edge_mag / max(sigma, 1e-9) if local_shape_active else 0.0))
        any_shape_active = bool(shape_active or local_shape_active)
        if any_shape_active:
            self.inactive_since_s = None
            if not self.armed:
                score = 0.0
                quality = 0.0
        else:
            if self.inactive_since_s is None:
                self.inactive_since_s = float(feature.time_s)
            elif feature.time_s - self.inactive_since_s >= 0.35:
                self.armed = True
        center_index = int(round((i_min + i_max) / 2.0))
        center_time_s = float(times[min(max(center_index, 0), len(times) - 1)])
        if local_shape_active and (
            not shape_active or local_edge_mag / max(local_threshold, 1e-9) >= edge_mag / max(threshold, 1e-9)
        ):
            center_time_s = float(local_center_time_s)
        quality = 0.0
        if shape_active or local_shape_active:
            active_len = edge_len if shape_active else local_edge_len
            active_mag = edge_mag if shape_active else local_edge_mag
            active_threshold = threshold if shape_active else local_threshold
            duration_quality = 1.0 - min(
                1.0,
                abs(active_len - 0.25) / max(self.config.shape_max_edge_len_s, 1e-6),
            )
            rebound_quality = min(1.0, local_rebound / max(active_mag, 1e-9)) if active_mag > 0 else 0.0
            quality = max(
                0.0,
                min(
                    1.0,
                    (0.75 * duration_quality + 0.25 * rebound_quality)
                    * min(1.0, active_mag / max(active_threshold, 1e-9)),
                ),
            )
        is_event, event_id, baseline, gate_mad, gate_threshold, event_time_s, event_score = self.gate.update(
            feature.time_s,
            score,
            quality,
            feature.motion_energy,
            refractory_s=self.config.shape_refractory_s,
            score_floor=max(1.0, self.config.min_score),
        )
        if is_event:
            self.armed = False
            self.inactive_since_s = None
        if not any_shape_active:
            self.baseline.append(smoothed)

        return BlinkDetectionResult(
            is_event=is_event,
            event_id=event_id,
            method=self.method,
            score=float(event_score if is_event and event_score > 0.0 else score),
            threshold=float(gate_threshold),
            baseline=float(baseline),
            mad=float(gate_mad),
            event_time_s=(center_time_s if is_event else feature.time_s),
            metrics={
                **metrics,
                "shape_edge_mag": float(edge_mag),
                "shape_edge_len": float(edge_len),
                "shape_local_edge_mag": float(local_edge_mag),
                "shape_local_edge_len": float(local_edge_len),
                "shape_local_center_time_s": float(local_center_time_s),
                "shape_local_rebound": float(local_rebound),
                "shape_local_edge_coherence": float(local_edge_coherence),
                "shape_local_reversal_ratio": float(local_reversal_ratio),
                "shape_local_return_ratio": float(local_return_ratio),
                "shape_local_active": float(local_shape_active),
                "shape_active": float(shape_active),
                "shape_armed": float(self.armed),
                "shape_episode_quality": float(quality),
                **self.gate.last_metrics,
            },
        )


class MotionEnergyBlinkDetector:
    """Simple motion-energy detector used as a baseline.

    It relies only on the precomputed per-chunk motion_energy feature and a
    short rolling max, then uses the same robust event gate as BlinkListener.
    This is intentionally conservative and easy to benchmark against the more
    structured phase/shape detectors.
    """

    method = "motion"

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        self.gate = _EpisodeEventGate(
            config,
            min_duration_s=0.08,
            max_duration_s=0.70,
            quiet_s=0.14,
            quality_floor=0.25,
        )
        self.window: Deque[float] = deque(maxlen=max(1, config.short_window))
        self.last_ungated_coherence = 0.0

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        value = max(0.0, float(feature.motion_energy))
        self.window.append(value)
        if len(self.window) >= 3:
            arr = np.asarray(self.window, dtype=np.float64)
            score = float(max(value, np.max(arr) - np.median(arr)))
            local_range = float(np.max(arr) - np.min(arr))
            local_level = float(np.max(arr))
        else:
            score = value
            local_range = value
            local_level = value

        amplitude_impulse = 0.0
        phase_pair_impulse = 0.0
        fmcw_evidence: Optional[_FmcwBlinkEvidence] = None
        if feature.signal_mode == "fmcw":
            amplitude_floor = max(abs(float(feature.amplitude)), 0.005)
            amplitude_impulse = abs(float(feature.amplitude_delta)) / amplitude_floor
            phase_pair_impulse = (
                abs(float(feature.phase_pair_delta or 0.0))
                * max(0.25, float(feature.phase_pair_consistency or 0.0))
            )
            fmcw_evidence = _fmcw_blink_evidence(
                feature,
                amplitude_impulse=amplitude_impulse,
                phase_impulse=phase_pair_impulse,
            )
            score = max(score, fmcw_evidence.score)
            local_level = max(local_level, score)
            local_range = max(local_range, score - min(arr) if len(self.window) >= 3 else score)

        quality = min(1.0, max(0.0, local_range / max(local_level, 1e-9)))
        if feature.signal_mode == "fmcw":
            quality = max(quality, fmcw_evidence.quality if fmcw_evidence is not None else 0.0)
        is_event, event_id, baseline, mad, threshold, event_time_s, event_score = self.gate.update(
            feature.time_s,
            score,
            quality,
            feature.motion_energy,
            score_floor=self.config.min_score,
        )
        return BlinkDetectionResult(
            is_event=is_event,
            event_id=event_id,
            method=self.method,
            score=float(event_score if is_event and event_score > 0.0 else score),
            threshold=float(threshold),
            baseline=float(baseline),
            mad=float(mad),
            event_time_s=(event_time_s if event_time_s is not None else feature.time_s),
            metrics={
                "motion_energy_score": float(score),
                "motion_energy_raw": value,
                "motion_amplitude_impulse": float(amplitude_impulse),
                "motion_phase_pair_impulse": float(phase_pair_impulse),
                "motion_fmcw_evidence_quality": (
                    fmcw_evidence.quality if fmcw_evidence is not None else 0.0
                ),
                "motion_fmcw_spatial_support": (
                    fmcw_evidence.spatial_support if fmcw_evidence is not None else 0.0
                ),
                "motion_window_len": float(len(self.window)),
                "motion_episode_quality": quality,
                **self.gate.last_metrics,
            },
        )


@dataclass
class _BumpPoint:
    time_s: float
    score: float
    raw_value: float
    amplitude_impulse: float
    projection_impulse: float
    phase_impulse: float
    motion: float
    spatial: float


class BumpBlinkDetector:
    """Local bump-shape detector for the waveform visible in the UI.

    This detector does not treat every threshold crossing as an event. It waits
    until a center sample has both left and right context, then requires a
    compact local maximum with prominence above nearby baseline and a visible
    return on both sides. That shape requirement is the main guard against
    refractory-clock false positives.
    """

    method = "bump"

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        self.points: Deque[_BumpPoint] = deque(maxlen=max(15, config.short_window * 3))
        self.prominence_history: Deque[float] = deque(maxlen=config.history_size)
        self.event_count = 0
        self.last_event_time_s = -1e9
        self.baseline_i: Deque[float] = deque(maxlen=config.history_size)
        self.baseline_q: Deque[float] = deque(maxlen=config.history_size)
        self.baseline_amplitude: Deque[float] = deque(maxlen=config.history_size)
        self.previous_phase: Optional[float] = None
        self.last_ungated_coherence = 0.0
        self.last_metrics: Dict[str, float] = {}
        self.accepted_times: Deque[float] = deque(maxlen=4)

    def _center(self, feature: ChunkFeature) -> Tuple[float, float]:
        if not self.baseline_i:
            return float(feature.i_value), float(feature.q_value)
        return float(np.median(self.baseline_i)), float(np.median(self.baseline_q))

    def _projection_impulse(self, feature: ChunkFeature, center_i: float, center_q: float) -> float:
        if len(self.baseline_i) < 5:
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
            deviation = abs(float(current_projection - np.median(baseline_projection)))
            scale = max(float(np.median(np.abs(baseline_projection))) * 1.4826, 1e-4)
            best = max(best, deviation / scale)
        return float(best)

    def _append_point(self, feature: ChunkFeature) -> _BumpPoint:
        center_i, center_q = self._center(feature)
        baseline_amp = (
            float(np.median(self.baseline_amplitude))
            if self.baseline_amplitude else float(feature.amplitude)
        )
        amplitude_impulse = abs(float(feature.amplitude) - baseline_amp) / max(
            abs(baseline_amp),
            abs(float(feature.amplitude)),
            0.005,
        )
        projection_impulse = self._projection_impulse(feature, center_i, center_q)
        if self.previous_phase is None:
            observed_phase_delta = abs(float(feature.phase_delta))
        else:
            observed_phase_delta = abs(float(unwrap_delta(float(feature.phase), self.previous_phase)))
            observed_phase_delta = max(observed_phase_delta, abs(float(feature.phase_delta)))
        self.previous_phase = float(feature.phase)

        if feature.signal_mode == "fmcw":
            phase_impulse = abs(float(feature.phase_pair_delta or 0.0)) * max(
                0.25,
                float(feature.phase_pair_consistency or 0.0),
            )
            spatial = float(feature.range_spread_ratio or 0.0)
            motion = max(0.0, float(feature.motion_energy))
            raw_value = max(
                projection_impulse * 0.35,
                1.10 * min(amplitude_impulse, 2.0),
                0.85 * min(phase_impulse, 2.0),
                0.70 * min(motion, 2.0),
                0.70 * min(spatial, 1.5),
            )
        else:
            phase_impulse = observed_phase_delta / math.pi
            spatial = 0.0
            motion = max(0.0, float(feature.motion_energy))
            raw_value = max(
                projection_impulse * 0.45,
                1.35 * min(amplitude_impulse, 2.0),
                0.80 * min(motion, 2.0),
                0.35 * phase_impulse,
            )

        point = _BumpPoint(
            time_s=float(feature.time_s),
            score=float(raw_value),
            raw_value=float(raw_value),
            amplitude_impulse=float(amplitude_impulse),
            projection_impulse=float(projection_impulse),
            phase_impulse=float(phase_impulse),
            motion=float(motion),
            spatial=float(spatial),
        )
        self.points.append(point)
        self.last_ungated_coherence = float(raw_value)
        return point

    def _prominence_threshold(self) -> Tuple[float, float, float]:
        if len(self.prominence_history) < max(8, self.config.min_history // 2):
            return 0.0, 0.0, max(0.18, float(self.config.min_score))
        values = np.asarray(self.prominence_history, dtype=np.float64)
        baseline = float(np.median(values))
        mad = float(np.median(np.abs(values - baseline)))
        sigma = 1.4826 * mad
        return baseline, mad, max(0.18, baseline + 2.8 * sigma)

    def _candidate_from_window(self) -> Tuple[Optional[_BumpPoint], Dict[str, float]]:
        radius = 3
        if len(self.points) < 2 * radius + 1:
            return None, {"bump_window_len": float(len(self.points))}
        values = list(self.points)
        center_index = len(values) - radius - 1
        candidate = values[center_index]
        left = values[max(0, center_index - radius):center_index]
        right = values[center_index + 1:center_index + 1 + radius]
        if len(left) < radius or len(right) < radius:
            return None, {"bump_window_len": float(len(self.points))}

        left_values = np.asarray([p.raw_value for p in left], dtype=np.float64)
        right_values = np.asarray([p.raw_value for p in right], dtype=np.float64)
        local_values = np.asarray([p.raw_value for p in values[max(0, center_index - radius):center_index + radius + 1]], dtype=np.float64)
        shoulder = max(float(np.median(left_values)), float(np.median(right_values)))
        left_drop = candidate.raw_value - float(np.median(left_values))
        right_drop = candidate.raw_value - float(np.median(right_values))
        prominence = min(left_drop, right_drop)
        compact_width_s = float(right[-1].time_s - left[0].time_s)
        is_peak = bool(candidate.raw_value >= float(np.max(local_values)))
        return_ratio = min(left_drop, right_drop) / max(candidate.raw_value, 1e-9)
        source_diversity = float(
            sum(
                1
                for value in (
                    candidate.amplitude_impulse,
                    candidate.projection_impulse * 0.35,
                    candidate.phase_impulse,
                    candidate.motion,
                    candidate.spatial,
                )
                if value >= 0.10
            )
        )
        metrics = {
            "bump_window_len": float(len(self.points)),
            "bump_candidate_time_s": float(candidate.time_s),
            "bump_candidate_raw": float(candidate.raw_value),
            "bump_shoulder": float(shoulder),
            "bump_prominence": float(prominence),
            "bump_left_drop": float(left_drop),
            "bump_right_drop": float(right_drop),
            "bump_return_ratio": float(return_ratio),
            "bump_compact_width_s": float(compact_width_s),
            "bump_is_peak": float(is_peak),
            "bump_source_diversity": float(source_diversity),
            "bump_amplitude_impulse": float(candidate.amplitude_impulse),
            "bump_projection_impulse": float(candidate.projection_impulse),
            "bump_phase_impulse": float(candidate.phase_impulse),
            "bump_motion": float(candidate.motion),
            "bump_spatial": float(candidate.spatial),
        }
        if not is_peak:
            self.prominence_history.append(max(0.0, float(prominence)))
            return None, metrics
        return candidate, metrics

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        current = self._append_point(feature)
        candidate, metrics = self._candidate_from_window()
        baseline, mad, threshold = self._prominence_threshold()
        metrics["bump_prominence_threshold"] = float(threshold)
        metrics["bump_prominence_baseline"] = float(baseline)
        metrics["bump_prominence_mad"] = float(mad)

        is_event = False
        event_time_s = feature.time_s
        event_score = current.raw_value
        if candidate is not None:
            prominence = float(metrics.get("bump_prominence", 0.0))
            return_ratio = float(metrics.get("bump_return_ratio", 0.0))
            compact_width_s = float(metrics.get("bump_compact_width_s", 0.0))
            source_diversity = float(metrics.get("bump_source_diversity", 0.0))
            compact_ok = 0.12 <= compact_width_s <= 0.55
            shape_ok = return_ratio >= 0.18 and prominence >= threshold
            evidence_ok = source_diversity >= (2.0 if feature.signal_mode == "fmcw" else 1.0)
            refractory_ok = (candidate.time_s - self.last_event_time_s) >= self.config.refractory_s
            startup_ok = candidate.time_s >= self.config.startup_ignore_s
            periodic_clock = self._would_continue_refractory_clock(candidate.time_s)
            is_event = bool(
                compact_ok
                and shape_ok
                and evidence_ok
                and refractory_ok
                and startup_ok
                and not periodic_clock
            )
            metrics["bump_compact_ok"] = float(compact_ok)
            metrics["bump_shape_ok"] = float(shape_ok)
            metrics["bump_evidence_ok"] = float(evidence_ok)
            metrics["bump_refractory_ok"] = float(refractory_ok)
            metrics["bump_periodic_clock_suppressed"] = float(periodic_clock)
            if is_event:
                self.event_count += 1
                self.last_event_time_s = float(candidate.time_s)
                self.accepted_times.append(float(candidate.time_s))
                event_time_s = float(candidate.time_s)
                event_score = float(candidate.raw_value)
            else:
                self.prominence_history.append(max(0.0, prominence))
        else:
            metrics.setdefault("bump_compact_ok", 0.0)
            metrics.setdefault("bump_shape_ok", 0.0)
            metrics.setdefault("bump_evidence_ok", 0.0)
            metrics.setdefault("bump_refractory_ok", 0.0)
            metrics.setdefault("bump_periodic_clock_suppressed", 0.0)

        in_freeze = (feature.time_s - self.last_event_time_s) < self.config.baseline_freeze_s
        if not is_event and not in_freeze:
            self.baseline_i.append(float(feature.i_value))
            self.baseline_q.append(float(feature.q_value))
            self.baseline_amplitude.append(float(feature.amplitude))

        self.last_metrics = metrics
        return BlinkDetectionResult(
            is_event=is_event,
            event_id=self.event_count,
            method=self.method,
            score=float(event_score),
            threshold=float(threshold),
            baseline=float(baseline),
            mad=float(mad),
            event_time_s=float(event_time_s),
            metrics=metrics,
        )

    def _would_continue_refractory_clock(self, candidate_time_s: float) -> bool:
        if len(self.accepted_times) < 2:
            return False
        period = max(0.1, float(self.config.refractory_s))
        tolerance = max(0.16, 0.22 * period)
        last_time = float(self.accepted_times[-1])
        prev_time = float(self.accepted_times[-2])
        gap_now = float(candidate_time_s) - last_time
        gap_prev = last_time - prev_time
        return bool(abs(gap_now - period) <= tolerance and abs(gap_prev - period) <= tolerance)



class PulseBlinkDetector:
    """Multi-feature blink pulse detector.

    This path is intentionally separate from Twinkle's trajectory reversal.
    It looks for a compact impulse that is simultaneously visible in selected
    range-bin amplitude, phase-pair coherence, localized range spread, and raw
    motion. It is designed for the saved FMCW blink sessions where phase-only
    peaks create many periodic false positives.
    """

    method = "pulse"

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        self.gate = _EpisodeEventGate(
            config,
            min_duration_s=0.05,
            max_duration_s=0.65,
            quiet_s=0.12,
            quality_floor=0.35,
        )
        self.amplitudes: Deque[float] = deque(maxlen=max(5, config.short_window))
        self.phase_pairs: Deque[float] = deque(maxlen=max(5, config.short_window))
        self.last_ungated_coherence = 0.0

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        previous_amp = list(self.amplitudes)
        previous_phase = list(self.phase_pairs)
        amplitude = float(feature.amplitude)
        phase_pair = float(feature.phase_pair_delta or feature.phase_delta or 0.0)
        self.amplitudes.append(amplitude)
        self.phase_pairs.append(phase_pair)

        if previous_amp:
            amp_ref = float(np.median(np.asarray(previous_amp, dtype=np.float64)))
        else:
            amp_ref = amplitude
        amp_impulse = abs(amplitude - amp_ref) / max(abs(amp_ref), abs(amplitude), 0.005)

        phase_impulse = 0.0
        phase_smoothness = 0.0
        if len(previous_phase) >= 3:
            phase_values = np.asarray(previous_phase + [phase_pair], dtype=np.float64)
            phase_center = float(np.median(phase_values[:-1]))
            phase_impulse = abs(float(unwrap_delta(phase_pair, phase_center)))
            steps = np.diff(phase_values)
            jitter = float(np.sqrt(np.mean(np.square(steps)))) if steps.size else 0.0
            phase_smoothness = 1.0 / (1.0 + jitter)

        consistency = float(feature.phase_pair_consistency or 0.0)
        spread = float(feature.range_spread_ratio or 0.0)
        motion = max(0.0, float(feature.motion_energy))
        fmcw_evidence: Optional[_FmcwBlinkEvidence] = None

        if feature.signal_mode == "fmcw":
            fmcw_evidence = _fmcw_blink_evidence(
                feature,
                amplitude_impulse=amp_impulse,
                phase_impulse=phase_impulse,
                phase_smoothness=phase_smoothness,
            )
            score = fmcw_evidence.score
            quality = fmcw_evidence.quality
            score_floor = max(self.config.min_score, 0.09)
        else:
            score = 0.55 * min(amp_impulse, 1.5) + 0.45 * min(motion, 1.2)
            quality = min(1.0, 0.35 + 0.40 * min(amp_impulse, 1.0) + 0.25 * min(motion, 1.0))
            score_floor = self.config.min_score

        self.last_ungated_coherence = float(score)
        is_event, event_id, baseline, mad, threshold, event_time_s, event_score = self.gate.update(
            feature.time_s,
            score,
            quality,
            motion,
            score_floor=score_floor,
        )
        return BlinkDetectionResult(
            is_event=is_event,
            event_id=event_id,
            method=self.method,
            score=float(event_score if is_event and event_score > 0.0 else score),
            threshold=float(threshold),
            baseline=float(baseline),
            mad=float(mad),
            event_time_s=(event_time_s if event_time_s is not None else feature.time_s),
            metrics={
                "pulse_score": float(score),
                "pulse_amp_impulse": float(amp_impulse),
                "pulse_phase_impulse": float(phase_impulse),
                "pulse_phase_smoothness": float(phase_smoothness),
                "pulse_consistency": float(consistency),
                "pulse_range_spread": float(spread),
                "pulse_motion": float(motion),
                "pulse_quality": float(quality),
                "pulse_fmcw_evidence_quality": (
                    fmcw_evidence.quality if fmcw_evidence is not None else 0.0
                ),
                "pulse_fmcw_spatial_support": (
                    fmcw_evidence.spatial_support if fmcw_evidence is not None else 0.0
                ),
                **self.gate.last_metrics,
            },
        )


class CompositeBlinkDetector:
    method = "both"

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        blinklistener_config = BlinkDetectionConfig(**{**config.__dict__, "method": "blinklistener"})
        twinkle_config = BlinkDetectionConfig(**{**config.__dict__, "method": "twinkle"})
        shape_config = BlinkDetectionConfig(**{**config.__dict__, "method": "shape"})
        pulse_config = BlinkDetectionConfig(**{**config.__dict__, "method": "pulse"})
        motion_config = BlinkDetectionConfig(**{**config.__dict__, "method": "motion"})
        self.detectors = [
            BlinkListenerBlinkDetector(blinklistener_config),
            TwinkleTwinkleBlinkDetector(twinkle_config),
            _TwinkleShapeSegmentationDetector(shape_config),
            PulseBlinkDetector(pulse_config),
            MotionEnergyBlinkDetector(motion_config),
        ]
        self.event_count = 0
        self.last_event_time_s = -1e9
        self.periodic_guard = _PeriodicEventSuppressor(config.refractory_s)

    @property
    def last_ungated_coherence(self) -> float:
        """Mirror the strongest child's ungated coherence so the realtime
        viz tracks whichever child detector is currently most active."""
        return float(max(
            (getattr(d, "last_ungated_coherence", 0.0) for d in self.detectors),
            default=0.0,
        ))

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        results: List[BlinkDetectionResult] = [detector.update(feature) for detector in self.detectors]
        event_results = [result for result in results if result.is_event]
        ratios = {result.method: _score_ratio(result) for result in results}
        support_scores = [
            min(max(ratios.get("blinklistener", 0.0), 0.0), 3.0) / 3.0,
            min(max(ratios.get("twinkle", 0.0), 0.0), 4.0) / 4.0,
            min(max(ratios.get("shape", 0.0), 0.0), 3.0) / 3.0,
            min(max(ratios.get("pulse", 0.0), 0.0), 4.0) / 4.0,
            min(max(ratios.get("motion", 0.0), 0.0), 4.0) / 4.0,
        ]
        support = float(sum(support_scores))

        if event_results:
            selected = max(event_results, key=_score_ratio)
            event_time_s = _result_time(selected, feature.time_s)
            outside_refractory = (event_time_s - self.last_event_time_s) >= self.config.refractory_s
            selected_ratio = _score_ratio(selected)
            enough_support = (
                selected.method in ("pulse", "twinkle")
                or support >= 1.20
                or selected_ratio >= 4.5
            )
            periodic_clock = self.periodic_guard.should_suppress(event_time_s, selected_ratio)
            if outside_refractory:
                if enough_support and not periodic_clock:
                    self.event_count += 1
                    self.last_event_time_s = event_time_s
                    self.periodic_guard.accept(event_time_s)
                    selected.event_id = self.event_count
                else:
                    selected.is_event = False
                    selected.event_id = self.event_count
            else:
                selected.is_event = False
                selected.event_id = self.event_count
            selected.metrics = dict(selected.metrics)
            selected.metrics["composite_periodic_clock_suppressed"] = float(periodic_clock)
        else:
            selected = max(results, key=_score_ratio)
            selected.is_event = False
            selected.event_id = self.event_count
            selected.metrics = dict(selected.metrics)
            selected.metrics["composite_periodic_clock_suppressed"] = 0.0
        selected.metrics["selected_method"] = selected.method
        selected.metrics["composite_support"] = support
        for method, ratio in ratios.items():
            selected.metrics[f"composite_ratio_{method}"] = ratio
        selected.metrics["composite_event_count"] = float(self.event_count)
        selected.method = self.method
        return selected


class HybridPulseBlinkDetector:
    method = "hybridpulse"

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        child_config = BlinkDetectionConfig(**{**config.__dict__, "method": "pulse"})
        twinkle_config = BlinkDetectionConfig(**{**config.__dict__, "method": "twinkle"})
        motion_config = BlinkDetectionConfig(**{**config.__dict__, "method": "motion"})
        shape_config = BlinkDetectionConfig(**{**config.__dict__, "method": "shape"})
        self.pulse = PulseBlinkDetector(child_config)
        self.twinkle = TwinkleTwinkleBlinkDetector(twinkle_config)
        self.motion = MotionEnergyBlinkDetector(motion_config)
        self.shape = _TwinkleShapeSegmentationDetector(shape_config)
        self.event_count = 0
        self.last_event_time_s = -1e9
        self.recent_support: Deque[Tuple[float, float, str]] = deque(maxlen=30)
        self.support_until_s = -1e9
        self.last_pulse_event_time_s = -1e9
        self.periodic_guard = _PeriodicEventSuppressor(config.refractory_s)

    @property
    def last_ungated_coherence(self) -> float:
        return float(max(
            getattr(self.pulse, "last_ungated_coherence", 0.0),
            getattr(self.twinkle, "last_ungated_coherence", 0.0),
            getattr(self.motion, "last_ungated_coherence", 0.0),
            getattr(self.shape, "last_ungated_coherence", 0.0),
        ))

    def _support_near(
        self,
        time_s: float,
        window_s: float = 1.10,
        exclude_kinds: Tuple[str, ...] = (),
    ) -> float:
        excluded = set(exclude_kinds)
        return float(
            sum(
                score
                for t, score, kind in self.recent_support
                if kind not in excluded and abs(float(time_s) - t) <= window_s
            )
        )

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        pulse_result = self.pulse.update(feature)
        motion_result = self.motion.update(feature)
        twinkle_result = self.twinkle.update(feature)
        shape_result = self.shape.update(feature)

        pulse_score_ratio = _score_ratio(pulse_result)
        motion_score_ratio = _score_ratio(motion_result)
        shape_score_ratio = _score_ratio(shape_result)
        historical_support = self._support_near(feature.time_s, window_s=0.65)
        if pulse_result.is_event or pulse_score_ratio >= 2.5:
            self.recent_support.append((feature.time_s, min(max(pulse_score_ratio, 2.5), 4.0) / 4.0, "pulse"))
            self.support_until_s = max(self.support_until_s, feature.time_s + 2.2)
        if motion_result.is_event or motion_score_ratio >= 3.0:
            self.recent_support.append((feature.time_s, min(max(motion_score_ratio, 3.0), 5.0) / 5.0, "motion"))
            self.support_until_s = max(self.support_until_s, feature.time_s + 1.2)
        if shape_result.is_event or shape_score_ratio >= 2.2:
            self.recent_support.append((feature.time_s, min(max(shape_score_ratio, 2.2), 4.0) / 4.0, "shape"))
            self.support_until_s = max(self.support_until_s, feature.time_s + 1.4)

        selected = None
        if pulse_result.is_event:
            pulse_time = _result_time(pulse_result, feature.time_s)
            pulse_ratio = _score_ratio(pulse_result)
            close_to_previous_pulse = (pulse_time - self.last_pulse_event_time_s) < 3.0
            pulse_has_companion = (
                motion_result.is_event
                or twinkle_result.is_event
                or shape_result.is_event
                or historical_support >= 0.60
            )
            if (not close_to_previous_pulse) or pulse_has_companion or pulse_ratio >= 8.0:
                selected = pulse_result
        elif twinkle_result.is_event:
            event_time = _result_time(twinkle_result, feature.time_s)
            support = self._support_near(event_time)
            twinkle_strength = _score_ratio(twinkle_result)
            in_supported_burst = event_time <= self.support_until_s
            has_supported_score = support >= 0.45 and twinkle_result.score >= 0.18
            shape_confirms = shape_result.is_event or shape_score_ratio >= 2.2
            isolated_phase_gap_s = max(1.55, self.config.refractory_s * 1.45)
            isolated_phase_burst = bool(
                twinkle_strength >= 3.2
                and pulse_score_ratio >= 0.30
                and motion_score_ratio <= 1.20
                and event_time - self.last_event_time_s >= isolated_phase_gap_s
            )
            if (
                has_supported_score
                or shape_confirms
                or (in_supported_burst and twinkle_strength >= 2.2)
                or isolated_phase_burst
                or twinkle_strength >= 5.0
            ):
                if isolated_phase_burst:
                    twinkle_result.event_time_s = event_time + 0.12
                    twinkle_result.metrics = dict(twinkle_result.metrics)
                    twinkle_result.metrics["hybridpulse_isolated_phase_burst"] = 1.0
                    twinkle_result.metrics["hybridpulse_phase_latency_compensation_s"] = 0.12
                selected = twinkle_result
        elif shape_result.is_event and (historical_support >= 0.45 or shape_score_ratio >= 3.5):
            shape_result.metrics = dict(shape_result.metrics)
            shape_result.metrics["hybridpulse_shape_confirmed"] = 1.0
            selected = shape_result

        if selected is None:
            selected = max(
                (pulse_result, twinkle_result, motion_result, shape_result),
                key=_score_ratio,
            )
            selected.is_event = False
            selected.event_id = self.event_count
        else:
            event_time = _result_time(selected, feature.time_s)
            reported_event_time = event_time
            if feature.signal_mode == "fmcw":
                if selected is pulse_result:
                    reported_event_time = event_time - 0.10
                    selected.event_time_s = float(reported_event_time)
                    selected.metrics = dict(selected.metrics)
                    selected.metrics["hybridpulse_source_time_compensation_s"] = -0.10
                elif selected is twinkle_result:
                    reported_event_time = event_time + 0.10
                    selected.event_time_s = float(reported_event_time)
                    selected.metrics = dict(selected.metrics)
                    selected.metrics["hybridpulse_source_time_compensation_s"] = 0.10
            selected_strength = _score_ratio(selected)
            periodic_clock = self.periodic_guard.should_suppress(event_time, selected_strength)
            if event_time - self.last_event_time_s >= self.config.refractory_s and not periodic_clock:
                self.event_count += 1
                self.last_event_time_s = float(event_time)
                self.periodic_guard.accept(event_time)
                if selected is pulse_result:
                    self.last_pulse_event_time_s = float(event_time)
                selected.event_id = self.event_count
            else:
                selected.is_event = False
                selected.event_id = self.event_count
            selected.metrics = dict(selected.metrics)
            selected.metrics["hybridpulse_periodic_clock_suppressed"] = float(periodic_clock)

        selected.method = self.method
        selected.metrics = dict(selected.metrics)
        selected.metrics["hybridpulse_support"] = self._support_near(
            selected.event_time_s if selected.event_time_s is not None else feature.time_s
        )
        selected.metrics["hybridpulse_pulse_ratio"] = float(pulse_score_ratio)
        selected.metrics["hybridpulse_motion_ratio"] = float(motion_score_ratio)
        selected.metrics["hybridpulse_twinkle_ratio"] = float(_score_ratio(twinkle_result))
        selected.metrics["hybridpulse_shape_ratio"] = float(shape_score_ratio)
        selected.metrics["hybridpulse_event_count"] = float(self.event_count)
        return selected


class HybridClusterBlinkDetector:
    """Cluster-level fusion of pulse, twinkle, shape, and motion candidates.

    HybridPulse emits as soon as one child detector is accepted. On longer
    recordings that can produce repeated events inside one acoustic burst. This
    detector keeps child candidates for a short quiet period, scores the whole
    burst by cross-algorithm evidence, then emits one representative event.
    """

    method = "hybridcluster"

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        self.pulse = PulseBlinkDetector(BlinkDetectionConfig(**{**config.__dict__, "method": "pulse"}))
        self.bump = BumpBlinkDetector(BlinkDetectionConfig(**{**config.__dict__, "method": "bump"}))
        self.twinkle = TwinkleTwinkleBlinkDetector(BlinkDetectionConfig(**{**config.__dict__, "method": "twinkle"}))
        self.motion = MotionEnergyBlinkDetector(BlinkDetectionConfig(**{**config.__dict__, "method": "motion"}))
        self.shape = _TwinkleShapeSegmentationDetector(BlinkDetectionConfig(**{**config.__dict__, "method": "shape"}))
        self.candidates: List[Tuple[float, float, str, BlinkDetectionResult]] = []
        self.event_count = 0
        self.last_event_time_s = -1e9
        self.quiet_s = 0.55
        self.max_cluster_s = 1.45
        self.periodic_guard = _PeriodicEventSuppressor(config.refractory_s)

    @property
    def last_ungated_coherence(self) -> float:
        return float(max(
            getattr(self.pulse, "last_ungated_coherence", 0.0),
            getattr(self.bump, "last_ungated_coherence", 0.0),
            getattr(self.twinkle, "last_ungated_coherence", 0.0),
            getattr(self.motion, "last_ungated_coherence", 0.0),
            getattr(self.shape, "last_ungated_coherence", 0.0),
        ))

    def _candidate_weight(self, result: BlinkDetectionResult, kind: str) -> float:
        ratio = _score_ratio(result)
        if kind == "bump":
            metrics = result.metrics
            return_ratio = float(metrics.get("bump_return_ratio", 0.0))
            diversity = float(metrics.get("bump_source_diversity", 0.0))
            return min(2.2, 0.45 + 0.32 * ratio + 0.20 * return_ratio + 0.10 * diversity)
        if kind == "shape":
            return min(2.2, 0.55 + 0.25 * ratio)
        if kind == "twinkle":
            return min(2.0, 0.45 + 0.32 * ratio)
        if kind == "pulse":
            metrics = result.metrics
            spatial = float(metrics.get("pulse_range_spread", 0.0))
            amp = float(metrics.get("pulse_amp_impulse", 0.0))
            phase = float(metrics.get("pulse_phase_impulse", 0.0))
            evidence = max(spatial, min(amp, 1.0), min(phase, 1.0))
            return min(1.8, 0.35 + 0.45 * ratio + 0.35 * evidence)
        if kind == "motion":
            return min(1.2, 0.25 + 0.25 * ratio)
        return min(1.0, ratio)

    def _append_candidate(self, result: BlinkDetectionResult, kind: str, feature_time_s: float):
        if not result.is_event:
            return
        event_time_s = _result_time(result, feature_time_s)
        weight = self._candidate_weight(result, kind)
        self.candidates.append((event_time_s, weight, kind, result))

    def _cluster_ready(self, time_s: float) -> bool:
        if not self.candidates:
            return False
        first_t = self.candidates[0][0]
        last_t = max(t for t, _w, _k, _r in self.candidates)
        return (time_s - last_t >= self.quiet_s) or (time_s - first_t >= self.max_cluster_s)

    def _emit_cluster(self, time_s: float) -> Optional[BlinkDetectionResult]:
        if not self.candidates:
            return None
        candidates = self.candidates
        self.candidates = []
        kinds = {kind for _t, _w, kind, _r in candidates}
        support = float(sum(weight for _t, weight, _kind, _r in candidates))
        non_bump_support = float(sum(weight for _t, weight, kind, _r in candidates if kind != "bump"))
        best_t, best_weight, best_kind, best_result = max(candidates, key=lambda item: item[1])
        duration = max(t for t, _w, _k, _r in candidates) - min(t for t, _w, _k, _r in candidates)
        non_bump_kinds = {kind for kind in kinds if kind != "bump"}
        has_cross_support = bool(len(kinds) >= 2 and non_bump_support > 0.0)
        strong_single = bool(best_weight >= 1.55 and best_kind in ("twinkle", "shape", "pulse"))
        supported_single = bool(non_bump_support >= 2.0 and best_kind in ("twinkle", "pulse", "shape"))
        if not (has_cross_support or strong_single or supported_single):
            return None
        if best_t - self.last_event_time_s < self.config.refractory_s:
            return None
        if self.periodic_guard.should_suppress(best_t, best_weight):
            return None
        self.event_count += 1
        self.last_event_time_s = float(best_t)
        self.periodic_guard.accept(best_t)
        selected = best_result
        selected.is_event = True
        selected.event_id = self.event_count
        selected.method = self.method
        selected.event_time_s = float(best_t)
        selected.metrics = dict(selected.metrics)
        selected.metrics["hybridcluster_support"] = support
        selected.metrics["hybridcluster_non_bump_support"] = non_bump_support
        selected.metrics["hybridcluster_duration_s"] = float(duration)
        selected.metrics["hybridcluster_candidate_count"] = float(len(candidates))
        selected.metrics["hybridcluster_kind_count"] = float(len(non_bump_kinds))
        selected.metrics["hybridcluster_selected_kind"] = {
            "pulse": 1.0,
            "bump": 1.5,
            "twinkle": 2.0,
            "shape": 3.0,
            "motion": 4.0,
        }.get(best_kind, 0.0)
        return selected

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        pulse_result = self.pulse.update(feature)
        bump_result = self.bump.update(feature)
        twinkle_result = self.twinkle.update(feature)
        shape_result = self.shape.update(feature)
        motion_result = self.motion.update(feature)

        self._append_candidate(pulse_result, "pulse", feature.time_s)
        self._append_candidate(bump_result, "bump", feature.time_s)
        self._append_candidate(twinkle_result, "twinkle", feature.time_s)
        self._append_candidate(shape_result, "shape", feature.time_s)
        self._append_candidate(motion_result, "motion", feature.time_s)

        emitted = self._emit_cluster(feature.time_s) if self._cluster_ready(feature.time_s) else None
        if emitted is not None:
            return emitted

        selected = max(
            (pulse_result, bump_result, twinkle_result, shape_result, motion_result),
            key=_score_ratio,
        )
        selected.is_event = False
        selected.event_id = self.event_count
        selected.method = self.method
        selected.metrics = dict(selected.metrics)
        selected.metrics["hybridcluster_pending_count"] = float(len(self.candidates))
        return selected


class HighRecallBlinkDetector:
    """Recall-oriented detector that unions structured and weak local cues.

    This is intentionally not the default realtime path. It is useful for
    answering whether the audio stream contains enough evidence to cover most
    visual blinks, even when precision is poor. It combines existing detector
    events with weak local acoustic extrema, reporting the local extremum time
    instead of the later refractory crossing.
    """

    method = "highrecall"

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        self.children = [
            CompositeBlinkDetector(BlinkDetectionConfig(**{**config.__dict__, "method": "both"})),
            HybridPulseBlinkDetector(BlinkDetectionConfig(**{**config.__dict__, "method": "hybridpulse"})),
            BumpBlinkDetector(BlinkDetectionConfig(**{**config.__dict__, "method": "bump"})),
            TwinkleTwinkleBlinkDetector(BlinkDetectionConfig(**{**config.__dict__, "method": "twinkle"})),
        ]
        self.points: Deque[_BumpPoint] = deque(maxlen=max(15, config.short_window * 3))
        self.event_count = 0
        self.last_event_time_s = -1e9
        self.baseline_amplitude: Deque[float] = deque(maxlen=config.history_size)
        self.previous_phase: Optional[float] = None
        self.last_ungated_coherence = 0.0
        self.periodic_guard = _PeriodicEventSuppressor(0.55)

    def _weak_point(self, feature: ChunkFeature) -> _BumpPoint:
        baseline_amp = (
            float(np.median(self.baseline_amplitude))
            if self.baseline_amplitude else float(feature.amplitude)
        )
        amplitude_impulse = abs(float(feature.amplitude) - baseline_amp) / max(
            abs(baseline_amp),
            abs(float(feature.amplitude)),
            0.005,
        )
        if self.previous_phase is None:
            phase_impulse = abs(float(feature.phase_delta))
        else:
            phase_impulse = abs(float(unwrap_delta(float(feature.phase), self.previous_phase)))
            phase_impulse = max(phase_impulse, abs(float(feature.phase_delta)))
        self.previous_phase = float(feature.phase)
        if feature.signal_mode == "fmcw":
            pair_impulse = abs(float(feature.phase_pair_delta or 0.0)) * max(
                0.25,
                float(feature.phase_pair_consistency or 0.0),
            )
            spatial = float(feature.range_spread_ratio or 0.0)
            raw_value = max(
                1.20 * min(amplitude_impulse, 2.0),
                0.85 * min(pair_impulse, 2.0),
                0.70 * min(max(0.0, float(feature.motion_energy)), 2.0),
                0.55 * min(spatial, 1.5),
            )
            phase_value = pair_impulse
        else:
            spatial = 0.0
            raw_value = max(
                1.35 * min(amplitude_impulse, 2.0),
                0.90 * min(max(0.0, float(feature.motion_energy)), 2.0),
                0.35 * min(phase_impulse / math.pi, 1.0),
            )
            phase_value = phase_impulse / math.pi
        point = _BumpPoint(
            time_s=float(feature.time_s),
            score=float(raw_value),
            raw_value=float(raw_value),
            amplitude_impulse=float(amplitude_impulse),
            projection_impulse=0.0,
            phase_impulse=float(phase_value),
            motion=max(0.0, float(feature.motion_energy)),
            spatial=float(spatial),
        )
        self.points.append(point)
        self.last_ungated_coherence = float(raw_value)
        return point

    def _weak_local_event(self) -> Tuple[bool, Optional[_BumpPoint], Dict[str, float]]:
        radius = 3
        if len(self.points) < 2 * radius + 1:
            return False, None, {"highrecall_window_len": float(len(self.points))}
        values = list(self.points)
        center_index = len(values) - radius - 1
        candidate = values[center_index]
        left = values[center_index - radius:center_index]
        right = values[center_index + 1:center_index + 1 + radius]
        left_scores = np.asarray([p.raw_value for p in left], dtype=np.float64)
        right_scores = np.asarray([p.raw_value for p in right], dtype=np.float64)
        local_scores = np.asarray([p.raw_value for p in left + [candidate] + right], dtype=np.float64)
        shoulder = max(float(np.median(left_scores)), float(np.median(right_scores)))
        prominence = float(candidate.raw_value - shoulder)
        is_peak = bool(candidate.raw_value >= float(np.max(local_scores)))
        source_count = float(
            sum(
                1
                for value in (
                    candidate.amplitude_impulse,
                    candidate.phase_impulse,
                    candidate.motion,
                    candidate.spatial,
                )
                if value >= 0.035
            )
        )
        weak_ok = bool(
            is_peak
            and candidate.raw_value >= (0.32 if candidate.spatial <= 0.0 else 0.42)
            and (prominence >= 0.015 or candidate.raw_value >= 0.85)
            and source_count >= (1.0 if candidate.spatial <= 0.0 else 2.0)
        )
        metrics = {
            "highrecall_weak_peak": float(is_peak),
            "highrecall_weak_raw": float(candidate.raw_value),
            "highrecall_weak_prominence": float(prominence),
            "highrecall_weak_source_count": float(source_count),
            "highrecall_weak_time_s": float(candidate.time_s),
        }
        return weak_ok, candidate, metrics

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        point = self._weak_point(feature)
        child_results = [child.update(feature) for child in self.children]
        child_events = [result for result in child_results if result.is_event]
        weak_event, weak_candidate, weak_metrics = self._weak_local_event()

        selected: Optional[BlinkDetectionResult] = None
        source = "none"
        if child_events:
            selected = min(child_events, key=lambda r: _result_time(r, feature.time_s))
            source = f"child_{selected.method}"
        elif weak_event and weak_candidate is not None:
            selected = BlinkDetectionResult(
                is_event=True,
                event_id=self.event_count,
                method=self.method,
                score=float(weak_candidate.raw_value),
                threshold=0.0,
                baseline=0.0,
                mad=0.0,
                event_time_s=float(weak_candidate.time_s),
                metrics={},
            )
            source = "weak_local_extremum"

        if selected is not None:
            event_time = _result_time(selected, feature.time_s)
            strength = _score_ratio(selected) if selected.threshold > 0.0 else selected.score
            periodic_clock = self.periodic_guard.should_suppress(event_time, strength)
            if event_time - self.last_event_time_s >= 0.55 and not periodic_clock:
                self.event_count += 1
                self.last_event_time_s = float(event_time)
                self.periodic_guard.accept(event_time)
                selected.is_event = True
                selected.event_id = self.event_count
            else:
                selected.is_event = False
                selected.event_id = self.event_count
            selected.metrics = dict(selected.metrics)
            selected.metrics["highrecall_periodic_clock_suppressed"] = float(periodic_clock)
        else:
            selected = max(child_results, key=_score_ratio)
            selected.is_event = False
            selected.event_id = self.event_count
            selected.metrics = dict(selected.metrics)
            selected.metrics["highrecall_periodic_clock_suppressed"] = 0.0

        if not selected.is_event:
            self.baseline_amplitude.append(float(feature.amplitude))
        selected.method = self.method
        selected.metrics = dict(selected.metrics)
        selected.metrics.update(weak_metrics)
        selected.metrics["highrecall_source"] = {
            "none": 0.0,
            "weak_local_extremum": 1.0,
            "child_both": 2.0,
            "child_hybridpulse": 3.0,
            "child_bump": 4.0,
            "child_twinkle": 5.0,
        }.get(source, 0.0)
        selected.metrics["highrecall_event_count"] = float(self.event_count)
        selected.metrics["highrecall_raw_score"] = float(point.raw_value)
        return selected


class NeuralBlinkDetector:
    """Small CPU-only MLP detector trained from saved acoustic blink sessions."""

    method = "neural"

    def __init__(self, config: BlinkDetectionConfig):
        from hp_acoustic_wave.neural_blink import default_model_path, load_model

        self.config = config
        model_path = getattr(config, "neural_model_path", "") or default_model_path()
        self.model = load_model(model_path)
        self.window_radius = max(0, int(self.model.window_size) // 2)
        self.feature_vectors: Deque[np.ndarray] = deque(maxlen=max(3, int(self.model.window_size) * 3))
        self.feature_times: Deque[float] = deque(maxlen=max(3, int(self.model.window_size) * 3))
        self.feature_motion: Deque[float] = deque(maxlen=max(3, int(self.model.window_size) * 3))
        self.points: Deque[Tuple[float, float, float]] = deque(
            maxlen=max(5, int(self.model.local_peak_radius) * 2 + 3)
        )
        self.event_count = 0
        self.last_event_time_s = -1e9
        self.last_ungated_coherence = 0.0
        self.periodic_guard = _PeriodicEventSuppressor(float(self.model.refractory_s))

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        from hp_acoustic_wave.neural_blink import (
            feature_vector_from_chunk,
            make_window_vector,
            predict_proba,
        )

        self.feature_vectors.append(feature_vector_from_chunk(feature))
        self.feature_times.append(float(feature.time_s))
        self.feature_motion.append(float(feature.motion_energy))
        probability = 0.0
        center_time_s = float(feature.time_s)
        center_motion = float(feature.motion_energy)
        if len(self.feature_vectors) >= self.model.window_size:
            vectors = list(self.feature_vectors)
            center_index = len(vectors) - 1 - self.window_radius
            x = make_window_vector(vectors, center_index, self.model.window_size)
            probability = float(predict_proba(self.model, x)[0])
            center_time_s = float(list(self.feature_times)[center_index])
            center_motion = float(list(self.feature_motion)[center_index])
            self.points.append((center_time_s, probability, center_motion))
        self.last_ungated_coherence = probability

        radius = max(1, int(self.model.local_peak_radius))
        is_event = False
        event_time_s = float(feature.time_s)
        event_score = probability
        event_motion = float(feature.motion_energy)
        local_peak = False
        periodic_clock = False
        if len(self.points) >= 2 * radius + 1:
            values = list(self.points)
            candidate_index = len(values) - radius - 1
            candidate_time, candidate_probability, candidate_motion = values[candidate_index]
            local_values = [item[1] for item in values[candidate_index - radius:candidate_index + radius + 1]]
            local_peak = bool(candidate_probability >= max(local_values))
            candidate_ready = bool(
                local_peak
                and candidate_probability >= float(self.model.threshold)
                and candidate_time - self.last_event_time_s >= float(self.model.refractory_s)
            )
            periodic_clock = bool(
                candidate_ready
                and _continues_refractory_clock(
                    self.periodic_guard.accepted_times,
                    candidate_time,
                    float(self.model.refractory_s),
                )
                and candidate_probability < 0.995
            )
            if candidate_ready and not periodic_clock:
                self.event_count += 1
                self.last_event_time_s = float(candidate_time)
                self.periodic_guard.accept(candidate_time)
                is_event = True
            event_time_s = float(candidate_time)
            event_score = float(candidate_probability)
            event_motion = float(candidate_motion)

        return BlinkDetectionResult(
            is_event=is_event,
            event_id=self.event_count,
            method=self.method,
            score=float(event_score),
            threshold=float(self.model.threshold),
            baseline=0.0,
            mad=0.0,
            event_time_s=event_time_s,
            metrics={
                "neural_probability": float(probability),
                "neural_candidate_probability": float(event_score),
                "neural_threshold": float(self.model.threshold),
                "neural_local_peak": float(local_peak),
                "neural_periodic_clock_suppressed": float(periodic_clock),
                "neural_refractory_s": float(self.model.refractory_s),
                "neural_window_size": float(self.model.window_size),
                "neural_center_time_s": float(center_time_s),
                "neural_candidate_motion_energy": float(event_motion),
            },
        )


def build_blink_detector(config: BlinkDetectionConfig):
    method = config.method.lower()
    if method == "old_twinkle":
        from hp_acoustic_wave.old_blink_detector import OldTwinkleDetector

        return OldTwinkleDetector(config)
    if method == "blinklistener":
        return BlinkListenerBlinkDetector(config)
    if method in ("twinkle", "twinkletwinkle"):
        return TwinkleTwinkleBlinkDetector(config)
    if method == "shape":
        return _TwinkleShapeSegmentationDetector(config)
    if method == "motion":
        return MotionEnergyBlinkDetector(config)
    if method == "pulse":
        return PulseBlinkDetector(config)
    if method == "bump":
        return BumpBlinkDetector(config)
    if method == "hybridpulse":
        return HybridPulseBlinkDetector(config)
    if method == "hybridcluster":
        return HybridClusterBlinkDetector(config)
    if method == "highrecall":
        return HighRecallBlinkDetector(config)
    if method in ("neural", "nerual"):
        return NeuralBlinkDetector(config)
    if method == "both":
        return CompositeBlinkDetector(config)
    raise ValueError("blink method must be one of: old_twinkle, blinklistener, bump, twinkle, shape, motion, pulse, hybridpulse, hybridcluster, highrecall, neural, both")
