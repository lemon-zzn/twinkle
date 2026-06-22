import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List
import wave

import numpy as np

from hp_acoustic_wave.blink_detector import BlinkDetectionConfig, build_blink_detector
from hp_acoustic_wave.dsp import (
    ChunkFeature,
    FmcwBackgroundSubtractor,
    extract_chunk_feature,
    extract_fmcw_chunk_feature,
    generate_fmcw_chirp,
)


@dataclass(frozen=True)
class AcousticEvent:
    time_s: float
    event_id: int = 0
    method: str = ""
    score: float = 0.0
    motion_energy: float = 0.0
    threshold: float = 0.0
    label: str = ""


@dataclass(frozen=True)
class ManualMarker:
    time_s: float
    label: str


@dataclass(frozen=True)
class BenchmarkSummary:
    events: int
    blink_hits: int
    blink_markers: int
    blink_misses: int
    large_motion_hits: int
    large_motion_markers: int
    large_motion_misses: int
    unexplained_events: int
    nonblink_events: int
    balanced_score: float
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0


@dataclass(frozen=True)
class BenchmarkRun:
    session: str
    source: str
    truth: str
    events: List[AcousticEvent]
    markers: List[ManualMarker]
    summary: BenchmarkSummary


def benchmark_session(
    session_dir: Path,
    config: BlinkDetectionConfig,
    source: str = "features",
    sample_rate: int = 48000,
    tone_hz: float = 18500.0,
    chunk_size: int = 1024,
    tukey_alpha: float = 0.0,
    truth: str = "manual",
) -> BenchmarkRun:
    session_path = Path(session_dir)
    resolved_truth = _resolve_truth(session_path, truth)
    if resolved_truth == "visual":
        markers = load_visual_blink_markers(session_path)
    else:
        markers = load_manual_markers(session_path / "manual_markers.csv")

    if source == "features":
        events = replay_feature_rows(load_feature_rows(session_path / "features.csv"), config)
    elif source == "audio":
        audio_config = _load_audio_config(session_path)
        signal_mode = str(audio_config.get("signal_mode", "tone"))
        emission = str(audio_config.get("fmcw_emission", "linear"))
        if signal_mode != "fmcw":
            emission = "cw_single"
        events = replay_feature_rows(
            reprocess_audio_feature_rows(
                session_path / "audio.wav",
                sample_rate=int(audio_config.get("sample_rate", sample_rate)),
                tone_hz=float(audio_config.get("tone_hz", tone_hz)),
                chunk_size=int(audio_config.get("chunk_size", chunk_size)),
                tukey_alpha=tukey_alpha,
                signal_mode=signal_mode,
                fmcw_freq_low=float(audio_config.get("fmcw_freq_low", 17_000.0)),
                fmcw_freq_high=float(audio_config.get("fmcw_freq_high", 23_000.0)),
                fmcw_chirp_duration=float(audio_config.get("fmcw_chirp_duration", 0.05)),
                fmcw_range_bin=int(audio_config.get("fmcw_range_bin", 15)),
                fmcw_lowpass_cutoff=float(audio_config.get("fmcw_lowpass_cutoff", 5_000.0)),
                fmcw_motion_amplitude_floor=float(audio_config.get("fmcw_motion_amplitude_floor", 0.02)),
                fmcw_output_amplitude=float(audio_config.get("output_amplitude", 0.2)),
                fmcw_emission=emission,
            ),
            config,
        )
    elif source == "events":
        events = _acoustic_events_only(load_events(session_path / "events.csv"))
    else:
        raise ValueError("source must be one of: features, audio, events")

    if resolved_truth == "visual":
        summary = score_events(events, markers, match_before_s=0.8, match_after_s=0.8)
    else:
        summary = score_events(events, markers)
    return BenchmarkRun(
        session=str(session_path),
        source=source,
        truth=resolved_truth,
        events=events,
        markers=markers,
        summary=summary,
    )


def load_manual_markers(path: Path) -> List[ManualMarker]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [
            ManualMarker(time_s=float(row["time_s"]), label=row["label"])
            for row in csv.DictReader(handle)
            if row.get("time_s") and row.get("label")
        ]


def load_events(path: Path) -> List[AcousticEvent]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return [_event_from_row(row) for row in csv.DictReader(handle) if row.get("time_s")]


def load_visual_blink_markers(session_dir: Path) -> List[ManualMarker]:
    session_path = Path(session_dir)
    events_path = session_path / "events.csv"
    if events_path.exists():
        events = load_events(events_path)
        markers = [
            ManualMarker(time_s=event.time_s, label="blink")
            for event in events
            if event.label == "visual_blink"
        ]
        if markers:
            return markers

    labels_path = session_path / "visual_labels.csv"
    if not labels_path.exists():
        return []
    with labels_path.open(newline="", encoding="utf-8") as handle:
        return [
            ManualMarker(time_s=float(row["time_s"]), label="blink")
            for row in csv.DictReader(handle)
            if row.get("time_s") and row.get("is_blink_event") == "1"
        ]


def load_feature_rows(path: Path) -> List[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reprocess_audio_feature_rows(
    audio_path: Path,
    sample_rate: int,
    tone_hz: float,
    chunk_size: int,
    tukey_alpha: float = 0.0,
    signal_mode: str = "tone",
    fmcw_freq_low: float = 17_000.0,
    fmcw_freq_high: float = 23_000.0,
    fmcw_chirp_duration: float = 0.05,
    fmcw_range_bin: int = 15,
    fmcw_lowpass_cutoff: float = 5_000.0,
    fmcw_motion_amplitude_floor: float = 0.02,
    fmcw_output_amplitude: float = 0.2,
    fmcw_emission: str = "linear",
) -> List[dict]:
    samples, wav_sample_rate = _read_wav_mono_float(audio_path)
    if wav_sample_rate != sample_rate:
        raise ValueError(f"audio sample rate {wav_sample_rate} does not match requested {sample_rate}")

    rows = []
    previous = None
    fmcw_background_subtractor = FmcwBackgroundSubtractor() if signal_mode == "fmcw" else None
    for start_sample in range(0, samples.size, int(chunk_size)):
        chunk = samples[start_sample : start_sample + int(chunk_size)]
        if chunk.size == 0:
            continue
        if signal_mode == "fmcw":
            tx = generate_fmcw_chirp(
                num_samples=chunk.size,
                sample_rate=sample_rate,
                freq_low=fmcw_freq_low,
                freq_high=fmcw_freq_high,
                chirp_duration=fmcw_chirp_duration,
                start_sample=start_sample,
                amplitude=fmcw_output_amplitude,
                emission=fmcw_emission,
            )
            feature = extract_fmcw_chunk_feature(
                samples=chunk,
                tx_samples=tx,
                sample_rate=sample_rate,
                freq_low=fmcw_freq_low,
                freq_high=fmcw_freq_high,
                chirp_duration=fmcw_chirp_duration,
                range_bin=fmcw_range_bin,
                start_sample=start_sample,
                previous=previous,
                lowpass_cutoff=fmcw_lowpass_cutoff,
                motion_amplitude_floor=fmcw_motion_amplitude_floor,
                background_subtractor=fmcw_background_subtractor,
                emission=fmcw_emission,
            )
        else:
            feature = extract_chunk_feature(
                samples=chunk,
                sample_rate=sample_rate,
                tone_hz=tone_hz,
                start_sample=start_sample,
                previous=previous,
                tukey_alpha=tukey_alpha,
            )
        previous = feature
        rows.append(_feature_to_row(feature))
    return rows


def _load_audio_config(session_path: Path) -> dict:
    metadata_path = Path(session_path) / "metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    config = metadata.get("config", {})
    audio_config = config.get("audio", {})
    return audio_config if isinstance(audio_config, dict) else {}


def replay_feature_rows(
    rows: Iterable[dict],
    config: BlinkDetectionConfig,
) -> List[AcousticEvent]:
    detector = build_blink_detector(config)
    events: List[AcousticEvent] = []
    for row in rows:
        feature = _feature_from_row(row)
        result = detector.update(feature)
        if result.is_event:
            event_time_s = result.event_time_s if result.event_time_s is not None else feature.time_s
            events.append(
                AcousticEvent(
                    time_s=event_time_s,
                    event_id=result.event_id,
                    method=result.method,
                    score=result.score,
                    motion_energy=feature.motion_energy,
                    threshold=result.threshold,
                    label="blink_candidate",
                )
            )
    return events


def _resolve_truth(session_path: Path, truth: str) -> str:
    truth = truth.lower()
    if truth not in ("manual", "visual", "auto"):
        raise ValueError("truth must be one of: manual, visual, auto")
    if truth != "auto":
        return truth
    if load_visual_blink_markers(session_path):
        return "visual"
    return "manual"


def _acoustic_events_only(events: Iterable[AcousticEvent]) -> List[AcousticEvent]:
    return [event for event in events if event.label != "visual_blink"]


def score_events(
    events: Iterable[AcousticEvent],
    markers: Iterable[ManualMarker],
    match_before_s: float = 0.0,
    match_after_s: float = 0.5,
    unexplained_penalty: float = 0.55,
    large_motion_penalty: float = 0.9,
    one_to_one: bool = True,
) -> BenchmarkSummary:
    event_list = list(events)
    marker_list = list(markers)

    blink_markers = [m for m in marker_list if m.label == "blink"]
    large_motion_markers = [m for m in marker_list if m.label == "large_motion"]

    if one_to_one:
        # 1:1 greedy nearest-neighbour pairing between events and blink markers.
        tp_event_indices = _pair_event_indices(event_list, blink_markers, match_before_s, match_after_s)
        tp = len(tp_event_indices)

        # Events labeled "large_motion" are correctly classified non-blinks (not FP).
        # Also count events matching large_motion markers (manual annotations).
        large_motion_label_indices = {
            i for i, e in enumerate(event_list)
            if i not in tp_event_indices and e.label == "large_motion"
        }
        large_motion_marker_indices = [
            i for i, e in enumerate(event_list)
            if i not in tp_event_indices
            and i not in large_motion_label_indices
            and any(_event_matches_marker(e, m, match_before_s, match_after_s) for m in large_motion_markers)
        ]
        large_hits_event_idx = large_motion_label_indices | set(large_motion_marker_indices)
        large_motion_hits = len(large_hits_event_idx)
        consumed = tp_event_indices | large_hits_event_idx

        # Under 1:1 pairing, every unconsumed event is FP (even if it matches a marker
        # but was not selected as the nearest pair).
        fp_indices = [i for i, e in enumerate(event_list) if i not in consumed]
        unexplained_events = len(fp_indices)
        fp = unexplained_events
    else:
        # Legacy per-marker ANY-event semantics: a marker is "hit" if ANY event
        # falls in its window; multiple events on the same marker are NOT FP.
        tp = _count_hit_markers(event_list, blink_markers, match_before_s, match_after_s)
        tp_event_indices = set()  # no per-event consumption in legacy mode

        # FP only for events that match NO marker at all (blink or large_motion).
        all_nonblink_markers = blink_markers + large_motion_markers
        fp_indices = [
            i for i, e in enumerate(event_list)
            if e.label != "large_motion"
            and not _matches_any_marker(e, all_nonblink_markers, match_before_s, match_after_s)
        ]
        unexplained_events = len(fp_indices)
        fp = unexplained_events

        # large_motion: events labeled or matching large_motion markers
        large_motion_label_indices = {
            i for i, e in enumerate(event_list) if e.label == "large_motion"
        }
        large_motion_marker_indices = {
            i for i, e in enumerate(event_list)
            if i not in large_motion_label_indices
            and any(_event_matches_marker(e, m, match_before_s, match_after_s) for m in large_motion_markers)
        }
        large_hits_event_idx = large_motion_label_indices | large_motion_marker_indices
        large_motion_hits = len(large_hits_event_idx)

    fn = max(0, len(blink_markers) - tp)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    balanced_score = tp - unexplained_penalty * unexplained_events - large_motion_penalty * large_motion_hits

    return BenchmarkSummary(
        events=len(event_list),
        blink_hits=tp,
        blink_markers=len(blink_markers),
        blink_misses=len(blink_markers) - tp,
        large_motion_hits=large_motion_hits,
        large_motion_markers=len(large_motion_markers),
        large_motion_misses=max(0, len(large_motion_markers) - large_motion_hits),
        unexplained_events=unexplained_events,
        nonblink_events=large_motion_hits + unexplained_events,
        balanced_score=balanced_score,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _pair_event_indices(events, blink_markers, match_before_s, match_after_s):
    """Return the set of event indices chosen as TP under 1:1 greedy nearest pairing."""
    candidates = []
    for mi, m in enumerate(blink_markers):
        for ei, e in enumerate(events):
            if _event_matches_marker(e, m, match_before_s, match_after_s):
                dt = abs(e.time_s - m.time_s)
                candidates.append((dt, ei, mi))
    candidates.sort()
    used_events, used_markers, chosen = set(), set(), set()
    for dt, ei, mi in candidates:
        if ei in used_events or mi in used_markers:
            continue
        used_events.add(ei)
        used_markers.add(mi)
        chosen.add(ei)
    return chosen


def _count_hit_markers(
    events: List[AcousticEvent],
    markers: List[ManualMarker],
    match_before_s: float,
    match_after_s: float,
) -> int:
    return sum(
        1
        for marker in markers
        if any(_event_matches_marker(event, marker, match_before_s, match_after_s) for event in events)
    )


def _matches_any_marker(
    event: AcousticEvent,
    markers: List[ManualMarker],
    match_before_s: float,
    match_after_s: float,
) -> bool:
    return any(_event_matches_marker(event, marker, match_before_s, match_after_s) for marker in markers)


def _event_matches_marker(
    event: AcousticEvent,
    marker: ManualMarker,
    match_before_s: float,
    match_after_s: float,
) -> bool:
    return marker.time_s - match_before_s <= event.time_s <= marker.time_s + match_after_s


def _event_from_row(row: dict) -> AcousticEvent:
    return AcousticEvent(
        time_s=float(row["time_s"]),
        event_id=int(row.get("event_id") or 0),
        method=row.get("method", ""),
        score=_optional_float(row.get("score")),
        motion_energy=_optional_float(row.get("motion_energy")),
        threshold=_optional_float(row.get("threshold")),
        label=row.get("label", ""),
    )


def _feature_from_row(row: dict) -> ChunkFeature:
    return ChunkFeature(
        time_s=float(row["time_s"]),
        sample_index=int(float(row.get("sample_index") or 0)),
        i_value=_optional_float(row.get("i")),
        q_value=_optional_float(row.get("q")),
        amplitude=_optional_float(row.get("amplitude")),
        amplitude_delta=_optional_float(row.get("amplitude_delta")),
        phase=_optional_float(row.get("phase")),
        phase_delta=_optional_float(row.get("phase_delta")),
        motion_energy=_optional_float(row.get("motion_energy")),
        rms=_optional_float(row.get("rms")),
        peak_abs=_optional_float(row.get("peak_abs")),
        signal_mode=row.get("signal_mode") or "tone",
        range_bin=_optional_int(row.get("range_bin")),
        range_distance_m=(
            _optional_float(row.get("range_distance_m"))
            if row.get("range_distance_m") not in (None, "")
            else None
        ),
        phase_pair_delta=(
            _optional_float(row.get("phase_pair_delta"))
            if row.get("phase_pair_delta") not in (None, "")
            else None
        ),
        phase_pair_vote_ratio=(
            _optional_float(row.get("phase_pair_vote_ratio"))
            if row.get("phase_pair_vote_ratio") not in (None, "")
            else None
        ),
        phase_pair_consistency=(
            _optional_float(row.get("phase_pair_consistency"))
            if row.get("phase_pair_consistency") not in (None, "")
            else None
        ),
        phase_pair_candidate_count=_optional_int(row.get("phase_pair_candidate_count")),
        phase_pair_deltas=_optional_float_tuple(row.get("phase_pair_deltas")),
        range_spread_bins=_optional_int(row.get("range_spread_bins")),
        range_spread_ratio=(
            _optional_float(row.get("range_spread_ratio"))
            if row.get("range_spread_ratio") not in (None, "")
            else None
        ),
        range_dominance_ratio=(
            _optional_float(row.get("range_dominance_ratio"))
            if row.get("range_dominance_ratio") not in (None, "")
            else None
        ),
    )


def _optional_float(value) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _optional_int(value):
    if value is None or value == "":
        return None
    return int(float(value))


def _optional_float_tuple(value):
    if value is None or value == "":
        return None
    items = []
    for token in str(value).split(";"):
        token = token.strip()
        if not token:
            continue
        try:
            items.append(float(token))
        except ValueError:
            continue
    return tuple(items)


def _read_wav_mono_float(path: Path):
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if sample_width != 2:
        raise ValueError("only 16-bit PCM wav files are supported")
    array = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        array = array.reshape(-1, channels).mean(axis=1, dtype=np.float32)
    return array.reshape(-1), sample_rate


def _feature_to_row(feature: ChunkFeature) -> dict:
    return {
        "time_s": f"{feature.time_s:.6f}",
        "sample_index": str(feature.sample_index),
        "i": f"{feature.i_value:.9f}",
        "q": f"{feature.q_value:.9f}",
        "amplitude": f"{feature.amplitude:.9f}",
        "amplitude_delta": f"{feature.amplitude_delta:.9f}",
        "phase": f"{feature.phase:.9f}",
        "phase_delta": f"{feature.phase_delta:.9f}",
        "phase_pair_delta": "" if feature.phase_pair_delta is None else f"{feature.phase_pair_delta:.9f}",
        "phase_pair_vote_ratio": (
            "" if feature.phase_pair_vote_ratio is None else f"{feature.phase_pair_vote_ratio:.9f}"
        ),
        "phase_pair_consistency": (
            "" if feature.phase_pair_consistency is None else f"{feature.phase_pair_consistency:.9f}"
        ),
        "phase_pair_candidate_count": (
            "" if feature.phase_pair_candidate_count is None else str(int(feature.phase_pair_candidate_count))
        ),
        "phase_pair_deltas": (
            ""
            if feature.phase_pair_deltas is None
            else ";".join(
                "nan" if not np.isfinite(float(value)) else f"{float(value):.9f}"
                for value in feature.phase_pair_deltas
            )
        ),
        "motion_energy": f"{feature.motion_energy:.9f}",
        "rms": f"{feature.rms:.9f}",
        "peak_abs": f"{feature.peak_abs:.9f}",
        "signal_mode": feature.signal_mode,
        "range_bin": "" if feature.range_bin is None else str(feature.range_bin),
        "range_distance_m": "" if feature.range_distance_m is None else f"{feature.range_distance_m:.9f}",
        "range_spread_bins": "" if feature.range_spread_bins is None else str(feature.range_spread_bins),
        "range_spread_ratio": (
            "" if feature.range_spread_ratio is None else f"{feature.range_spread_ratio:.9f}"
        ),
        "range_dominance_ratio": (
            "" if feature.range_dominance_ratio is None else f"{feature.range_dominance_ratio:.9f}"
        ),
    }
