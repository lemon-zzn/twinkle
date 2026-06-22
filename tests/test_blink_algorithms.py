import math
from typing import Optional

import numpy as np

from hp_acoustic_wave.blink_detector import (
    BlinkDetectionConfig,
    BlinkListenerBlinkDetector,
    CompositeBlinkDetector,
    MotionEnergyBlinkDetector,
    NeuralBlinkDetector,
    TwinkleTwinkleBlinkDetector,
    _TwinklePeakEventGate,
    build_blink_detector,
)
from hp_acoustic_wave.dsp import ChunkFeature, extract_chunk_feature


def _feature(
    index: int,
    i_value: float = 0.0,
    q_value: float = 0.0,
    phase: float = 0.0,
    amplitude: Optional[float] = None,
    motion_energy: float = 0.0,
    signal_mode: str = "tone",
    range_bin: Optional[int] = None,
    phase_pair_delta: Optional[float] = None,
    phase_pair_vote_ratio: Optional[float] = None,
    phase_pair_consistency: Optional[float] = None,
    phase_pair_candidate_count: Optional[int] = None,
    phase_pair_deltas: Optional[tuple] = None,
    range_spread_bins: Optional[int] = None,
    range_spread_ratio: Optional[float] = None,
    range_dominance_ratio: Optional[float] = None,
) -> ChunkFeature:
    if amplitude is None:
        amplitude = math.hypot(i_value, q_value)
    return ChunkFeature(
        time_s=index * 0.05,
        sample_index=index,
        i_value=i_value,
        q_value=q_value,
        amplitude=amplitude,
        amplitude_delta=0.0,
        phase=phase,
        phase_delta=0.0,
        motion_energy=motion_energy,
        rms=0.0,
        peak_abs=0.0,
        signal_mode=signal_mode,
        range_bin=range_bin,
        phase_pair_delta=phase_pair_delta,
        phase_pair_vote_ratio=phase_pair_vote_ratio,
        phase_pair_consistency=phase_pair_consistency,
        phase_pair_candidate_count=phase_pair_candidate_count,
        phase_pair_deltas=phase_pair_deltas,
        range_spread_bins=range_spread_bins,
        range_spread_ratio=range_spread_ratio,
        range_dominance_ratio=range_dominance_ratio,
    )


def _config(**overrides) -> BlinkDetectionConfig:
    values = {
        "history_size": 80,
        "min_history": 8,
        "threshold_k": 3.0,
        "min_score": 0.01,
        "refractory_s": 0.4,
        "baseline_freeze_s": 0.35,
        "short_window": 7,
        "phase_pair_lag": 3,
    }
    values.update(overrides)
    return BlinkDetectionConfig(**values)


def test_default_fmcw_twinkle_gate_values_follow_visual_truth_tuning():
    config = BlinkDetectionConfig()

    assert config.twinkle_fmcw_min_amplitude == 0.005
    assert config.twinkle_fmcw_min_score == 0.09
    assert config.twinkle_fmcw_min_range_spread_ratio == 0.0
    assert config.twinkle_fmcw_refractory_s == 1.3
    assert config.twinkle_fmcw_use_intra_chirp_phase_pair is False


def test_twinkle_fmcw_range_spread_gate_rejects_narrow_candidates():
    config = _config(
        min_history=3,
        min_score=0.1,
        twinkle_peak_min_ratio=1.0,
        twinkle_fmcw_min_range_spread_ratio=0.5,
    )
    gate = _TwinklePeakEventGate(config)

    for index in range(3):
        gate.update(index * 0.05, score=0.02, motion_energy=0.0, sign_changes=0, is_fmcw=True)

    gate.update(0.20, score=0.02, motion_energy=0.0, sign_changes=0, is_fmcw=True)
    gate.update(
        0.25,
        score=0.2,
        motion_energy=0.0,
        sign_changes=0,
        is_fmcw=True,
        range_spread_ratio=0.2,
    )
    is_event, *_ = gate.update(0.30, score=0.02, motion_energy=0.0, sign_changes=0, is_fmcw=True)

    assert is_event is False


def test_extract_chunk_feature_downmixes_stereo_instead_of_discarding_channel():
    sample_rate = 48_000
    tone_hz = 18_000
    samples = np.zeros((256, 2), dtype=np.float32)
    samples[:, 1] = np.sin(2.0 * np.pi * tone_hz * np.arange(256) / sample_rate)

    feature = extract_chunk_feature(samples, sample_rate, tone_hz, 0, None)

    assert feature.rms > 0.2
    assert feature.amplitude > 0.05


def test_extract_chunk_feature_optional_tukey_window_reduces_edge_transient():
    sample_rate = 48_000
    tone_hz = 18_000
    samples = np.zeros(256, dtype=np.float32)
    samples[0] = 1.0

    rectangular = extract_chunk_feature(samples, sample_rate, tone_hz, 0, None)
    windowed = extract_chunk_feature(samples, sample_rate, tone_hz, 0, None, tukey_alpha=1.0)

    assert rectangular.amplitude > 0.0
    assert windowed.amplitude < rectangular.amplitude * 0.1


def test_motion_energy_prioritizes_amplitude_change_over_phase_jump():
    sample_rate = 48_000
    tone_hz = 18_000
    samples = np.sin(2.0 * np.pi * tone_hz * np.arange(256) / sample_rate).astype(np.float32)
    previous = extract_chunk_feature(samples, sample_rate, tone_hz, 0, None)
    phase_jump = extract_chunk_feature(
        np.sin(2.0 * np.pi * tone_hz * np.arange(256) / sample_rate + math.pi / 2).astype(np.float32),
        sample_rate,
        tone_hz,
        0,
        previous,
    )
    amp_change = extract_chunk_feature(
        np.float32(1.45) * samples,
        sample_rate,
        tone_hz,
        0,
        previous,
    )

    assert amp_change.motion_energy > phase_jump.motion_energy


def test_blinklistener_detects_iq_bump_after_quiet_baseline():
    detector = BlinkListenerBlinkDetector(_config())
    events = []

    for index in range(12):
        events.append(detector.update(_feature(index, i_value=1.0, q_value=0.05)).is_event)
    for index, i_value in enumerate([1.04, 1.16, 1.32, 1.15, 1.03], start=12):
        events.append(detector.update(_feature(index, i_value=i_value, q_value=0.05)).is_event)

    assert any(events[12:])


def test_blinklistener_ignores_pure_phase_jump_when_amplitude_is_stable():
    detector = BlinkListenerBlinkDetector(_config(min_score=0.02))
    events = []

    for index in range(12):
        phase = 0.01 * index
        events.append(
            detector.update(
                _feature(index, i_value=math.cos(phase), q_value=math.sin(phase), phase=phase)
            ).is_event
        )
    for index, phase in enumerate([0.9, 1.4, 2.0, 2.5], start=12):
        events.append(
            detector.update(
                _feature(index, i_value=math.cos(phase), q_value=math.sin(phase), phase=phase)
            ).is_event
        )

    assert not any(events[12:])


def test_twinkle_proxy_detects_reversal_trajectory_not_monotonic_drift():
    drift_detector = TwinkleTwinkleBlinkDetector(_config(startup_ignore_s=0.55))
    blink_detector = TwinkleTwinkleBlinkDetector(_config(startup_ignore_s=0.55))

    drift_events = [
        drift_detector.update(_feature(index, phase=index * 0.03, amplitude=1.0)).is_event
        for index in range(24)
    ]
    blink_phases = [index * 0.01 for index in range(12)] + [0.20, 0.35, 0.53, 0.36, 0.18, 0.08]
    blink_events = [
        blink_detector.update(_feature(index, phase=phase, amplitude=1.0)).is_event
        for index, phase in enumerate(blink_phases)
    ]

    assert not any(drift_events[12:])
    assert any(blink_events[12:])


def test_twinkle_fmcw_detects_stable_range_bin_phase_reversal():
    detector = TwinkleTwinkleBlinkDetector(
        _config(
            min_score=0.015,
            twinkle_fmcw_min_amplitude=0.05,
            twinkle_fmcw_phase_pair_weight=0.8,
            twinkle_fmcw_use_intra_chirp_phase_pair=True,
        )
    )
    events = []
    results = []

    for index in range(12):
        ppd = index * 0.004
        results.append(
            detector.update(
                _feature(
                    index,
                    phase=ppd,
                    phase_pair_delta=ppd,
                    amplitude=0.8,
                    signal_mode="fmcw",
                    range_bin=15,
                )
            )
        )
        events.append(results[-1].is_event)

    blink_phases = [0.08, 0.20, 0.38, 0.56, 0.35, 0.14, 0.04]
    for index, phase in enumerate(blink_phases, start=12):
        results.append(
            detector.update(
                _feature(
                    index,
                    phase=phase,
                    phase_pair_delta=phase,
                    amplitude=0.82,
                    motion_energy=0.04,
                    signal_mode="fmcw",
                    range_bin=15,
                )
            )
        )
        events.append(results[-1].is_event)

    assert any(events[12:])
    assert results[-1].metrics["twinkle_signal_mode"] == 1.0
    assert results[-1].metrics["twinkle_fmcw_range_bin"] == 15.0


def test_twinkle_fmcw_uses_intra_chirp_phase_pair_as_trajectory():
    detector = TwinkleTwinkleBlinkDetector(
        _config(
            min_score=0.015,
            twinkle_fmcw_min_amplitude=0.05,
            twinkle_fmcw_phase_pair_weight=0.8,
            twinkle_fmcw_use_intra_chirp_phase_pair=True,
        )
    )
    events = []
    results = []

    for index in range(12):
        result = detector.update(
            _feature(
                index,
                phase=0.0,
                phase_pair_delta=index * 0.004,
                amplitude=0.8,
                signal_mode="fmcw",
                range_bin=15,
            )
        )
        events.append(result.is_event)

    blink_pair_phases = [0.08, 0.20, 0.38, 0.56, 0.35, 0.14, 0.04]
    for index, phase_pair_delta in enumerate(blink_pair_phases, start=12):
        result = detector.update(
            _feature(
                index,
                phase=0.0,
                phase_pair_delta=phase_pair_delta,
                amplitude=0.82,
                motion_energy=0.04,
                signal_mode="fmcw",
                range_bin=15,
            )
        )
        results.append(result)
        events.append(result.is_event)

    assert any(events[12:])
    assert results[-1].metrics["twinkle_trajectory_phase_source"] == 1.0


def test_twinkle_fmcw_rejects_low_amplitude_range_bin_phase_noise():
    detector = TwinkleTwinkleBlinkDetector(
        _config(
            min_score=0.015,
            twinkle_fmcw_min_amplitude=0.05,
            twinkle_fmcw_phase_pair_weight=0.8,
        )
    )
    events = []

    for index in range(12):
        events.append(
            detector.update(
                _feature(
                    index,
                    phase=index * 0.004,
                    amplitude=1e-4,
                    signal_mode="fmcw",
                    range_bin=15,
                )
            ).is_event
        )

    noisy_phases = [0.08, 0.20, 0.38, 0.56, 0.35, 0.14, 0.04]
    results = []
    for index, phase in enumerate(noisy_phases, start=12):
        results.append(
            detector.update(
                _feature(
                    index,
                    phase=phase,
                    amplitude=1e-4,
                    motion_energy=0.0,
                    signal_mode="fmcw",
                    range_bin=15,
                )
            )
        )
        events.append(results[-1].is_event)

    assert not any(events[12:])
    assert results[-1].metrics["twinkle_fmcw_amplitude_ok"] == 0.0


def test_twinkle_fmcw_default_amplitude_guard_blocks_low_snr_random_phase_walk():
    detector = TwinkleTwinkleBlinkDetector(
        _config(
            min_score=0.006,
            threshold_k=2.5,
            twinkle_candidate_windows=(5, 7, 9, 11),
            twinkle_min_candidate_votes=2,
            twinkle_fmcw_min_score=0.07,
            startup_ignore_s=0.0,
        )
    )

    rng = np.random.default_rng(7)
    phase = 0.0
    event_count = 0

    for index in range(300):
        phase += float(rng.uniform(-1.3, 1.3))
        phase = math.atan2(math.sin(phase), math.cos(phase))
        amplitude = max(2e-4, float(rng.normal(1.8e-3, 8e-4)))
        result = detector.update(
            _feature(
                index,
                phase=phase,
                amplitude=amplitude,
                motion_energy=0.08,
                signal_mode="fmcw",
                range_bin=15,
            )
        )
        if result.is_event:
            event_count += 1

    assert event_count == 0


def test_twinkle_fmcw_rejects_small_phase_profile_below_fmcw_score_floor():
    detector = TwinkleTwinkleBlinkDetector(
        _config(
            min_score=0.006,
            refractory_s=0.4,
            twinkle_fmcw_min_score=0.09,
            twinkle_fmcw_min_amplitude=0.02,
        )
    )
    events = []

    for index in range(12):
        events.append(
            detector.update(
                _feature(
                    index,
                    phase=index * 0.002,
                    amplitude=0.08,
                    signal_mode="fmcw",
                    range_bin=15,
                )
            ).is_event
        )

    small_profile = [0.02, 0.04, 0.06, 0.04, 0.02, 0.01]
    results = []
    for index, phase in enumerate(small_profile, start=12):
        results.append(
            detector.update(
                _feature(
                    index,
                    phase=phase,
                    amplitude=0.08,
                    signal_mode="fmcw",
                    range_bin=15,
                )
            )
        )
        events.append(results[-1].is_event)

    assert not any(events[12:])
    assert results[-1].metrics["twinkle_fmcw_score_ok"] == 0.0


def test_twinkle_fmcw_uses_fmcw_specific_refractory_for_duplicate_peaks():
    detector = TwinkleTwinkleBlinkDetector(
        _config(
            min_score=0.006,
            refractory_s=0.4,
            twinkle_fmcw_min_score=0.02,
            twinkle_fmcw_refractory_s=1.7,
        )
    )
    events = []

    for index in range(12):
        detector.update(
            _feature(
                index,
                phase=index * 0.002,
                phase_pair_delta=index * 0.002,
                amplitude=0.08,
                signal_mode="fmcw",
                range_bin=15,
            )
        )

    profiles = [
        [0.08, 0.18, 0.30, 0.18, 0.08],
        [0.08, 0.20, 0.34, 0.20, 0.08],
    ]
    start = 12
    for profile in profiles:
        for offset, phase in enumerate(profile):
            result = detector.update(
                _feature(
                    start + offset,
                    phase=phase,
                    phase_pair_delta=phase,
                    amplitude=0.08,
                    signal_mode="fmcw",
                    range_bin=15,
                )
            )
            if result.is_event:
                events.append(result)
        start += len(profile) + 8
        for index in range(start - 8, start):
            detector.update(
                _feature(
                    index,
                    phase=0.02,
                    phase_pair_delta=0.02,
                    amplitude=0.08,
                    signal_mode="fmcw",
                    range_bin=15,
                )
            )

    assert len(events) == 1
    assert events[0].metrics["twinkle_effective_refractory_s"] == 1.7


def test_twinkle_peak_gate_rejects_compact_segment_with_large_motion_metrics():
    gate = _TwinklePeakEventGate(_config(min_score=0.05, min_history=8))

    for index in range(10):
        gate.update(index * 0.05, score=0.01, motion_energy=0.0, sign_changes=0)

    events = []
    segment_scores = [0.30, 1.80, 2.40, 1.10, 0.03]
    for offset, score in enumerate(segment_scores, start=10):
        is_event, *_ = gate.update(
            offset * 0.05,
            score=score,
            motion_energy=0.70,
            sign_changes=5,
            refractory_s=0.4,
        )
        events.append(is_event)

    assert not any(events)


def test_twinkle_candidate_windows_require_configured_votes():
    detector = TwinkleTwinkleBlinkDetector(
        _config(
            twinkle_candidate_windows=(5, 7, 9),
            twinkle_min_candidate_votes=2,
        )
    )

    for index in range(12):
        detector.update(_feature(index, phase=index * 0.004, amplitude=1.0))

    result = None
    for index, phase in enumerate([0.08, 0.20, 0.38, 0.56, 0.35, 0.14, 0.04], start=12):
        result = detector.update(_feature(index, phase=phase, amplitude=1.0))

    assert result is not None
    assert result.metrics["twinkle_candidate_votes"] >= 2.0
    assert result.metrics["twinkle_best_candidate_window"] in {5.0, 7.0, 9.0}


def test_composite_detector_applies_shared_refractory_across_child_detectors():
    detector = CompositeBlinkDetector(_config(method="both", min_score=0.01))
    event_ids = []

    for index in range(12):
        result = detector.update(_feature(index, i_value=1.0, q_value=0.0, phase=index * 0.01))
        event_ids.append(result.event_id)
    for index, (i_value, phase) in enumerate(
        [(1.05, 0.12), (1.28, 0.30), (1.45, 0.55), (1.18, 0.32), (1.02, 0.10)],
        start=12,
    ):
        result = detector.update(_feature(index, i_value=i_value, q_value=0.0, phase=phase))
        event_ids.append(result.event_id)

    assert max(event_ids) == 1


def test_motion_energy_detector_fires_on_motion_spike():
    detector = MotionEnergyBlinkDetector(
        _config(method="motion", min_history=3, history_size=20, min_score=0.05, threshold_k=2.0)
    )
    for index in range(6):
        detector.update(_feature(index, motion_energy=0.01))

    result = detector.update(_feature(20, motion_energy=0.5))

    assert result.is_event
    assert result.method == "motion"
    assert result.metrics["motion_energy_raw"] == 0.5


def test_build_blink_detector_accepts_motion_method():
    detector = build_blink_detector(_config(method="motion"))

    assert isinstance(detector, MotionEnergyBlinkDetector)


def test_build_blink_detector_accepts_old_twinkle_method():
    detector = build_blink_detector(_config(method="old_twinkle"))

    assert detector.method == "old_twinkle"


def test_build_blink_detector_accepts_highrecall_method():
    detector = build_blink_detector(_config(method="highrecall"))

    assert detector.method == "highrecall"


def test_build_blink_detector_accepts_neural_method(tmp_path):
    from hp_acoustic_wave.neural_blink import FEATURE_NAMES, NeuralBlinkModel, save_model

    input_size = len(FEATURE_NAMES) * 3
    model_path = tmp_path / "neural.npz"
    save_model(
        model_path,
        NeuralBlinkModel(
            window_size=3,
            feature_names=FEATURE_NAMES,
            mean=np.zeros((input_size,), dtype=np.float32),
            scale=np.ones((input_size,), dtype=np.float32),
            w1=np.zeros((input_size, 2), dtype=np.float32),
            b1=np.zeros((2,), dtype=np.float32),
            w2=np.zeros((2, 1), dtype=np.float32),
            b2=np.zeros((1,), dtype=np.float32),
            threshold=0.5,
            refractory_s=0.75,
            local_peak_radius=1,
        ),
    )

    detector = build_blink_detector(_config(method="neural", neural_model_path=str(model_path)))

    assert isinstance(detector, NeuralBlinkDetector)


def test_build_blink_detector_accepts_nerual_alias(tmp_path):
    from hp_acoustic_wave.neural_blink import FEATURE_NAMES, NeuralBlinkModel, save_model

    input_size = len(FEATURE_NAMES) * 3
    model_path = tmp_path / "neural.npz"
    save_model(
        model_path,
        NeuralBlinkModel(
            window_size=3,
            feature_names=FEATURE_NAMES,
            mean=np.zeros((input_size,), dtype=np.float32),
            scale=np.ones((input_size,), dtype=np.float32),
            w1=np.zeros((input_size, 2), dtype=np.float32),
            b1=np.zeros((2,), dtype=np.float32),
            w2=np.zeros((2, 1), dtype=np.float32),
            b2=np.zeros((1,), dtype=np.float32),
            threshold=0.5,
            refractory_s=0.75,
            local_peak_radius=1,
        ),
    )

    detector = build_blink_detector(_config(method="nerual", neural_model_path=str(model_path)))

    assert detector.method == "neural"
