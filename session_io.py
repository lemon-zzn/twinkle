import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import wave

import numpy as np


FEATURE_FIELDS = [
    "time_s",
    "sample_index",
    "i",
    "q",
    "amplitude",
    "amplitude_delta",
    "phase",
    "phase_delta",
    "phase_pair_delta",
    "phase_pair_vote_ratio",
    "phase_pair_consistency",
    "phase_pair_candidate_count",
    "phase_pair_deltas",
    "motion_energy",
    "rms",
    "peak_abs",
    "baseline",
    "mad",
    "threshold",
    "detector_method",
    "blink_score",
    "blink_threshold",
    "blink_baseline",
    "blink_mad",
    "blinklistener_viewing_amplitude",
    "blinklistener_viewing_range",
    "blinklistener_raw_viewing_score",
    "blinklistener_relative_viewing_score",
    "blinklistener_fmcw_impulse_score",
    "blinklistener_center_i",
    "blinklistener_center_q",
    "twinkle_phase_pair_delta",
    "twinkle_trajectory_span",
    "twinkle_trajectory_rms",
    "twinkle_peak_score",
    "twinkle_peak_threshold",
    "twinkle_peak_motion_energy",
    "twinkle_peak_sign_changes",
    "twinkle_peak_trajectory_span",
    "twinkle_candidate_local_peak",
    "twinkle_candidate_rising_edge",
    "twinkle_large_motion_suppressed",
    "twinkle_fmcw_morphology_ok",
    "twinkle_fmcw_range_bin",
    "twinkle_fmcw_phase_score",
    "twinkle_fmcw_amplitude_ok",
    "twinkle_fmcw_amplitude_stable",
    "twinkle_fmcw_amplitude_delta_ratio",
    "twinkle_effective_refractory_s",
    "motion_amplitude_impulse",
    "motion_phase_pair_impulse",
    "shape_local_edge_mag",
    "shape_local_edge_len",
    "shape_local_center_time_s",
    "shape_local_rebound",
    "shape_local_active",
    "pulse_score",
    "pulse_amp_impulse",
    "pulse_phase_impulse",
    "pulse_phase_smoothness",
    "pulse_consistency",
    "pulse_range_spread",
    "pulse_motion",
    "pulse_quality",
    "bump_candidate_raw",
    "bump_prominence",
    "bump_return_ratio",
    "bump_source_diversity",
    "bump_periodic_clock_suppressed",
    "highrecall_source",
    "highrecall_weak_raw",
    "highrecall_weak_prominence",
    "highrecall_weak_source_count",
    "highrecall_event_count",
    "hybridpulse_support",
    "hybridpulse_event_count",
    "visual_face_present",
    "visual_left_ear",
    "visual_right_ear",
    "visual_mean_ear",
    "visual_is_closed",
    "visual_blink_event",
    "visual_blink_count",
    "signal_mode",
    "range_bin",
    "range_distance_m",
    "range_spread_bins",
    "range_spread_ratio",
    "range_dominance_ratio",
    "is_event",
    "event_id",
]

EVENT_FIELDS = [
    "event_id",
    "time_s",
    "label",
    "method",
    "score",
    "motion_energy",
    "threshold",
]

MARKER_FIELDS = [
    "time_s",
    "label",
    "key",
    "event_id",
    "amplitude",
    "phase",
    "motion_energy",
    "signal_mode",
    "range_bin",
    "range_distance_m",
]

VISUAL_LABEL_FIELDS = [
    "time_s",
    "face_present",
    "left_ear",
    "right_ear",
    "mean_ear",
    "is_closed",
    "closed_frames",
    "is_blink_event",
    "blink_count",
]


def create_session_dir(root: str, prefix: str = "hp_wave") -> Path:
    root_path = Path(root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = root_path / f"{prefix}_{timestamp}"
    suffix = 1
    while session_dir.exists():
        session_dir = root_path / f"{prefix}_{timestamp}_{suffix}"
        suffix += 1
    session_dir.mkdir(parents=True, exist_ok=False)
    return session_dir


class SessionWriter:
    def __init__(self, session_dir: Path, sample_rate: int):
        self.session_dir = Path(session_dir)
        self.sample_rate = sample_rate
        self._wav: Optional[wave.Wave_write] = None
        self._features_handle = None
        self._events_handle = None
        self._markers_handle = None
        self._visual_labels_handle = None
        self._features_writer = None
        self._events_writer = None
        self._markers_writer = None
        self._visual_labels_writer = None

    def open(self) -> None:
        self._wav = wave.open(str(self.session_dir / "audio.wav"), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(self.sample_rate)

        self._features_handle = open(self.session_dir / "features.csv", "w", newline="", encoding="utf-8")
        self._features_writer = csv.DictWriter(self._features_handle, fieldnames=FEATURE_FIELDS)
        self._features_writer.writeheader()

        self._events_handle = open(self.session_dir / "events.csv", "w", newline="", encoding="utf-8")
        self._events_writer = csv.DictWriter(self._events_handle, fieldnames=EVENT_FIELDS)
        self._events_writer.writeheader()

        self._markers_handle = open(self.session_dir / "manual_markers.csv", "w", newline="", encoding="utf-8")
        self._markers_writer = csv.DictWriter(self._markers_handle, fieldnames=MARKER_FIELDS)
        self._markers_writer.writeheader()

        self._visual_labels_handle = open(self.session_dir / "visual_labels.csv", "w", newline="", encoding="utf-8")
        self._visual_labels_writer = csv.DictWriter(self._visual_labels_handle, fieldnames=VISUAL_LABEL_FIELDS)
        self._visual_labels_writer.writeheader()

    def write_audio(self, samples: np.ndarray) -> None:
        if self._wav is None:
            raise RuntimeError("SessionWriter is not open")
        mono = np.asarray(samples, dtype=np.float32).reshape(-1)
        clipped = np.clip(mono, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype("<i2")
        self._wav.writeframes(pcm16.tobytes())

    def write_feature(self, row: Dict[str, float]) -> None:
        if self._features_writer is None:
            raise RuntimeError("SessionWriter is not open")
        self._features_writer.writerow(row)

    def write_event(
        self,
        event_id: int,
        time_s: float,
        motion_energy: float,
        threshold: float,
        label: str = "wave",
        method: str = "wave",
        score: Optional[float] = None,
    ) -> None:
        if self._events_writer is None:
            raise RuntimeError("SessionWriter is not open")
        self._events_writer.writerow(
            {
                "event_id": event_id,
                "time_s": f"{time_s:.6f}",
                "label": label,
                "method": method,
                "score": "" if score is None else f"{score:.9f}",
                "motion_energy": f"{motion_energy:.9f}",
                "threshold": f"{threshold:.9f}",
            }
        )

    def write_manual_marker(
        self,
        time_s: float,
        label: str = "manual_wave",
        key: str = "m",
        feature_snapshot: Optional[Dict] = None,
        event_id: Optional[int] = None,
    ) -> None:
        if self._markers_writer is None:
            raise RuntimeError("SessionWriter is not open")
        snapshot = feature_snapshot or {}
        self._markers_writer.writerow(
            {
                "time_s": f"{time_s:.6f}",
                "label": label,
                "key": key,
                "event_id": "" if event_id is None else event_id,
                "amplitude": _format_optional_float(snapshot.get("amplitude")),
                "phase": _format_optional_float(snapshot.get("phase")),
                "motion_energy": _format_optional_float(snapshot.get("motion_energy")),
                "signal_mode": snapshot.get("signal_mode", ""),
                "range_bin": "" if snapshot.get("range_bin") is None else snapshot.get("range_bin"),
                "range_distance_m": _format_optional_float(snapshot.get("range_distance_m")),
            }
        )

    def write_visual_label(
        self,
        time_s: float,
        face_present: bool,
        left_ear: float,
        right_ear: float,
        mean_ear: float,
        is_closed: bool,
        closed_frames: int,
        is_blink_event: bool,
        blink_count: int,
    ) -> None:
        if self._visual_labels_writer is None:
            raise RuntimeError("SessionWriter is not open")
        self._visual_labels_writer.writerow(
            {
                "time_s": f"{float(time_s):.6f}",
                "face_present": int(bool(face_present)),
                "left_ear": f"{float(left_ear):.9f}",
                "right_ear": f"{float(right_ear):.9f}",
                "mean_ear": f"{float(mean_ear):.9f}",
                "is_closed": int(bool(is_closed)),
                "closed_frames": int(closed_frames),
                "is_blink_event": int(bool(is_blink_event)),
                "blink_count": int(blink_count),
            }
        )

    def write_metadata(self, payload: Dict) -> None:
        with open(self.session_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def flush(self) -> None:
        for handle in (
            self._features_handle,
            self._events_handle,
            self._markers_handle,
            self._visual_labels_handle,
        ):
            if handle is not None:
                handle.flush()

    def close(self) -> None:
        if self._wav is not None:
            self._wav.close()
            self._wav = None
        for attr in ("_features_handle", "_events_handle", "_markers_handle", "_visual_labels_handle"):
            handle = getattr(self, attr)
            if handle is not None:
                handle.close()
                setattr(self, attr, None)


def _format_optional_float(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.9f}"
