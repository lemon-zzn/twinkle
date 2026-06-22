from hp_acoustic_wave.benchmark import AcousticEvent, BenchmarkSummary
from hp_acoustic_wave.fmcw_bin_calibration import (
    BinCalibrationScore,
    calibration_end_time,
    compute_fmcw_prior_scores,
    compute_fmcw_response_band_prior_scores,
    filter_events_by_motion_spread,
    merge_weighted_bin_events,
    parse_range_bins,
    select_top_bins,
)


def _summary(blink_hits, blink_markers, unexplained_events, balanced_score):
    return BenchmarkSummary(
        events=blink_hits + unexplained_events,
        blink_hits=blink_hits,
        blink_markers=blink_markers,
        blink_misses=blink_markers - blink_hits,
        large_motion_hits=0,
        large_motion_markers=0,
        large_motion_misses=0,
        unexplained_events=unexplained_events,
        nonblink_events=unexplained_events,
        balanced_score=balanced_score,
    )


def test_parse_range_bins_accepts_colon_ranges_and_csv():
    assert parse_range_bins("8:12") == (8, 9, 10, 11, 12)
    assert parse_range_bins("24, 23, 26") == (24, 23, 26)
    assert parse_range_bins("15") == (15,)


def test_select_top_bins_orders_by_score_and_returns_normalized_weights():
    scores = [
        BinCalibrationScore(range_bin=15, summary=_summary(25, 72, 4, 22.8)),
        BinCalibrationScore(range_bin=24, summary=_summary(71, 72, 7, 67.15)),
        BinCalibrationScore(range_bin=9, summary=_summary(69, 72, 9, 64.05)),
    ]

    selected = select_top_bins(scores, top_k=2)

    assert selected.selected_bins == (24, 9)
    assert selected.primary_bin == 24
    assert selected.bin_weights == (1.0, 0.954)


def test_select_top_bins_breaks_ties_toward_preferred_bin():
    scores = [
        BinCalibrationScore(range_bin=10, summary=_summary(12, 12, 7, 8.15)),
        BinCalibrationScore(range_bin=15, summary=_summary(12, 12, 7, 8.15)),
        BinCalibrationScore(range_bin=19, summary=_summary(12, 12, 7, 8.15)),
    ]

    selected = select_top_bins(scores, top_k=1, preferred_bin=15)

    assert selected.selected_bins == (15,)


def test_compute_fmcw_prior_scores_prefers_bins_near_strong_reflection_band():
    mean_amplitudes = {
        12: 0.04,
        24: 0.041,
        25: 0.052,
        26: 0.054,
        27: 0.062,
    }

    priors = compute_fmcw_prior_scores(mean_amplitudes, band_radius=4)

    assert priors[27] == 1.0
    assert priors[24] > 0.0
    assert priors[12] == 0.0


def test_response_band_prior_uses_blink_response_before_static_peak():
    mean_amplitudes = {
        10: 0.051,
        11: 0.042,
        12: 0.042,
        19: 0.120,
        24: 0.041,
        25: 0.052,
        26: 0.054,
        27: 0.062,
    }
    calibration_scores = {
        10: 2.45,
        11: 2.90,
        12: 3.45,
        19: 0.0,
        24: 3.45,
        25: 2.45,
        26: 3.45,
        27: 3.45,
    }

    priors = compute_fmcw_response_band_prior_scores(
        mean_amplitudes,
        calibration_scores,
        band_radius=4,
    )

    assert priors[24] > priors[12]
    assert priors[27] > 0.0
    assert priors[19] == 0.0


def test_calibration_end_extends_until_minimum_visual_blinks():
    markers = [
        type("Marker", (), {"time_s": 3.0, "label": "blink"})(),
        type("Marker", (), {"time_s": 12.0, "label": "blink"})(),
        type("Marker", (), {"time_s": 18.0, "label": "blink"})(),
        type("Marker", (), {"time_s": 30.0, "label": "blink"})(),
    ]

    assert calibration_end_time(markers, seconds=10.0, max_seconds=25.0, min_blinks=3) == 18.0


def test_calibration_end_stops_at_base_window_when_enough_blinks_exist():
    markers = [
        type("Marker", (), {"time_s": 2.0, "label": "blink"})(),
        type("Marker", (), {"time_s": 4.0, "label": "blink"})(),
        type("Marker", (), {"time_s": 8.0, "label": "blink"})(),
    ]

    assert calibration_end_time(markers, seconds=10.0, max_seconds=25.0, min_blinks=3) == 10.0


def test_merge_weighted_bin_events_coalesces_nearby_bin_votes():
    events_by_bin = {
        24: [
            AcousticEvent(time_s=1.00, event_id=1, method="twinkle", score=0.9, label="blink_candidate"),
            AcousticEvent(time_s=3.00, event_id=2, method="twinkle", score=0.5, label="blink_candidate"),
        ],
        23: [
            AcousticEvent(time_s=1.12, event_id=1, method="twinkle", score=0.7, label="blink_candidate"),
        ],
    }

    merged = merge_weighted_bin_events(
        events_by_bin,
        bin_weights={24: 1.0, 23: 0.8},
        merge_window_s=0.25,
        min_votes=1,
    )

    assert [round(event.time_s, 2) for event in merged] == [1.0, 3.0]
    assert merged[0].method == "twinkle_multibin"
    assert merged[0].event_id == 1


def test_filter_events_by_motion_spread_rejects_global_range_motion():
    selected_events = (
        AcousticEvent(time_s=1.00, event_id=1, method="twinkle", score=0.9, label="blink_candidate"),
        AcousticEvent(time_s=3.00, event_id=2, method="twinkle", score=0.5, label="blink_candidate"),
    )
    events_by_bin = {
        14: [
            AcousticEvent(time_s=1.00, event_id=1, method="twinkle", score=0.9, label="blink_candidate"),
            AcousticEvent(time_s=3.00, event_id=2, method="twinkle", score=0.5, label="blink_candidate"),
        ],
        15: [AcousticEvent(time_s=1.05, event_id=1, method="twinkle", score=0.7, label="blink_candidate")],
        16: [AcousticEvent(time_s=1.08, event_id=1, method="twinkle", score=0.6, label="blink_candidate")],
        17: [AcousticEvent(time_s=1.12, event_id=1, method="twinkle", score=0.6, label="blink_candidate")],
    }

    kept, rejected = filter_events_by_motion_spread(
        selected_events,
        events_by_bin,
        window_s=0.15,
        spread_bins=4,
    )

    assert [round(event.time_s, 2) for event in kept] == [3.0]
    assert [round(event.time_s, 2) for event in rejected] == [1.0]
