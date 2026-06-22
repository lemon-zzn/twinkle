from dataclasses import dataclass
from typing import Iterable, Tuple

from hp_acoustic_wave.benchmark import AcousticEvent, BenchmarkSummary


@dataclass(frozen=True)
class BinCalibrationScore:
    range_bin: int
    summary: BenchmarkSummary
    score: float = None
    prior_score: float = 0.0

    @property
    def ranking_score(self) -> float:
        if self.score is not None:
            return float(self.score)
        return float(self.summary.balanced_score)


@dataclass(frozen=True)
class BinSelectionResult:
    selected_bins: Tuple[int, ...]
    bin_weights: Tuple[float, ...]
    primary_bin: int
    bin_scores: Tuple[BinCalibrationScore, ...]


def parse_range_bins(value: str) -> Tuple[int, ...]:
    text = str(value or "").strip()
    if not text:
        raise ValueError("range bin list cannot be empty")
    if ":" in text:
        start_text, end_text = text.split(":", 1)
        start = int(start_text.strip())
        end = int(end_text.strip())
        step = 1 if end >= start else -1
        return tuple(range(start, end + step, step))
    return tuple(int(part.strip()) for part in text.split(",") if part.strip())


def compute_fmcw_prior_scores(mean_amplitudes, band_radius: int = 4):
    if not mean_amplitudes:
        return {}
    radius = max(0, int(band_radius))
    peak_bin = max(mean_amplitudes, key=lambda item: float(mean_amplitudes[item]))
    peak_amplitude = max(float(mean_amplitudes[peak_bin]), 1e-12)
    scores = {}
    for range_bin, amplitude in mean_amplitudes.items():
        distance = abs(int(range_bin) - int(peak_bin))
        if distance > radius:
            scores[int(range_bin)] = 0.0
            continue
        distance_weight = 1.0 if radius == 0 else 1.0 - (float(distance) / float(radius + 1))
        amplitude_weight = max(0.0, min(1.0, float(amplitude) / peak_amplitude))
        scores[int(range_bin)] = round(float(distance_weight * amplitude_weight), 6)
    scores[int(peak_bin)] = 1.0
    return scores


def compute_fmcw_response_band_prior_scores(
    mean_amplitudes,
    calibration_scores,
    band_radius: int = 4,
    amplitude_weight: float = 0.35,
):
    if not mean_amplitudes or not calibration_scores:
        return {}
    radius = max(0, int(band_radius))
    bins = sorted(int(value) for value in mean_amplitudes)
    max_amplitude = max((float(mean_amplitudes[item]) for item in mean_amplitudes), default=1e-12)
    max_amplitude = max(max_amplitude, 1e-12)

    def band_strength(center_bin):
        total = 0.0
        for range_bin in bins:
            distance = abs(int(range_bin) - int(center_bin))
            if distance > radius:
                continue
            distance_weight = 1.0 if radius == 0 else 1.0 - (float(distance) / float(radius + 1))
            response = max(0.0, float(calibration_scores.get(range_bin, 0.0)))
            amplitude = max(0.0, float(mean_amplitudes.get(range_bin, 0.0))) / max_amplitude
            total += distance_weight * (response + max(0.0, float(amplitude_weight)) * amplitude)
        return total

    center_bin = max(bins, key=lambda item: (band_strength(item), -abs(item - 15)))
    scores = {}
    for range_bin in bins:
        distance = abs(int(range_bin) - int(center_bin))
        if distance > radius:
            scores[int(range_bin)] = 0.0
            continue
        distance_weight = 1.0 if radius == 0 else 1.0 - (float(distance) / float(radius + 1))
        scores[int(range_bin)] = round(float(distance_weight), 6)
    scores[int(center_bin)] = 1.0
    return scores


def select_top_bins(
    scores: Iterable[BinCalibrationScore],
    top_k: int,
    preferred_bin: int = 15,
) -> BinSelectionResult:
    preferred = int(preferred_bin)
    ordered = tuple(
        sorted(
            scores,
            key=lambda item: (
                item.ranking_score,
                item.summary.blink_hits,
                -item.summary.unexplained_events,
                -abs(int(item.range_bin) - preferred),
            ),
            reverse=True,
        )
    )
    if not ordered:
        raise ValueError("at least one bin score is required")
    selected = ordered[: max(1, int(top_k))]
    max_score = max(abs(selected[0].ranking_score), 1e-9)
    weights = tuple(round(max(0.0, item.ranking_score) / max_score, 3) for item in selected)
    return BinSelectionResult(
        selected_bins=tuple(item.range_bin for item in selected),
        bin_weights=weights,
        primary_bin=int(selected[0].range_bin),
        bin_scores=ordered,
    )


def calibration_end_time(
    markers: Iterable,
    seconds: float,
    max_seconds: float,
    min_blinks: int,
) -> float:
    base_end = max(0.0, float(seconds))
    hard_end = max(base_end, float(max_seconds))
    required = max(0, int(min_blinks))
    blink_times = sorted(
        float(marker.time_s)
        for marker in markers
        if getattr(marker, "label", "") == "blink" and float(marker.time_s) <= hard_end
    )
    if required <= 0:
        return base_end
    if sum(1 for time_s in blink_times if time_s <= base_end) >= required:
        return base_end
    if len(blink_times) >= required:
        return max(base_end, float(blink_times[required - 1]))
    return hard_end


def merge_weighted_bin_events(
    events_by_bin,
    bin_weights,
    merge_window_s: float,
    min_votes: int,
) -> Tuple[AcousticEvent, ...]:
    weighted = []
    for range_bin, events in events_by_bin.items():
        weight = float(bin_weights.get(range_bin, 1.0))
        for event in events:
            weighted.append((float(event.time_s), int(range_bin), weight, event))
    weighted.sort(key=lambda item: item[0])

    merged = []
    event_id = 0
    index = 0
    window_s = max(0.0, float(merge_window_s))
    required_votes = max(1, int(min_votes))
    while index < len(weighted):
        group = [weighted[index]]
        group_start = weighted[index][0]
        index += 1
        while index < len(weighted) and weighted[index][0] <= group_start + window_s:
            group.append(weighted[index])
            index += 1
        vote_count = len({item[1] for item in group})
        if vote_count < required_votes:
            continue
        best_time, _best_bin, best_weight, best_event = max(
            group,
            key=lambda item: (item[3].score * item[2], item[3].score),
        )
        event_id += 1
        merged.append(
            AcousticEvent(
                time_s=float(best_time),
                event_id=event_id,
                method="twinkle_multibin",
                score=float(best_event.score) * float(best_weight),
                motion_energy=float(best_event.motion_energy),
                threshold=float(best_event.threshold),
                label=best_event.label,
            )
        )
    return tuple(merged)


def filter_events_by_motion_spread(
    events: Iterable[AcousticEvent],
    events_by_bin,
    window_s: float,
    spread_bins: int,
) -> Tuple[Tuple[AcousticEvent, ...], Tuple[AcousticEvent, ...]]:
    required_spread = max(0, int(spread_bins))
    if required_spread <= 0:
        event_tuple = tuple(events)
        return event_tuple, tuple()
    window = max(0.0, float(window_s))
    kept = []
    rejected = []
    for event in events:
        active_bins = set()
        event_time = float(event.time_s)
        for range_bin, bin_events in events_by_bin.items():
            if any(abs(float(bin_event.time_s) - event_time) <= window for bin_event in bin_events):
                active_bins.add(int(range_bin))
        if len(active_bins) >= required_spread:
            rejected.append(event)
        else:
            kept.append(event)
    return tuple(kept), tuple(rejected)
