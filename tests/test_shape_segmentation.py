"""Unit tests for _TwinkleShapeSegmentationDetector.

Shape-based segmentation detects a down-up (or up-down) edge pattern on the
phase trajectory within time bounds [min_edge_len, max_edge_len]. A single
noise peak does NOT form such a pattern, so periodic noise-driven firing is
structurally prevented.
"""
import numpy as np
import pytest

from hp_acoustic_wave.blink_detector import (
    BlinkDetectionConfig,
    _TwinkleShapeSegmentationDetector,
)
from hp_acoustic_wave.dsp import ChunkFeature


def make_feature(time_s, phase_value, signal_mode="fmcw"):
    """Build a minimal ChunkFeature. Shape detector only reads time_s,
    signal_mode, phase, phase_pair_delta."""
    return ChunkFeature(
        time_s=time_s,
        sample_index=int(time_s * 48000),
        i_value=0.0,
        q_value=0.0,
        amplitude=0.1,
        amplitude_delta=0.0,
        phase=float(phase_value),
        phase_delta=0.0,
        motion_energy=0.0,
        rms=0.1,
        peak_abs=0.1,
        signal_mode=signal_mode,
        range_bin=15,
        range_distance_m=0.4,
        phase_pair_delta=float(phase_value),
    )


def _run_series(detector, values, dt=0.05, start_t=0.0):
    events = []
    t = start_t
    for v in values:
        r = detector.update(make_feature(t, v))
        if r.is_event:
            events.append(t)
        t += dt
    return events


def test_shape_detects_down_up_edge_within_blink_duration():
    """A clear down-then-up edge spanning ~0.2s should trigger exactly one blink."""
    config = BlinkDetectionConfig(method="shape")
    det = _TwinkleShapeSegmentationDetector(config)
    # 1s baseline (tiny noise)
    rng = np.random.default_rng(0)
    baseline = list(rng.normal(0.0, 1e-4, 20))
    _run_series(det, baseline)
    # down-up edge from -0.3 to +0.3 over 5 frames (0.25s)
    edge = [-0.3, -0.2, 0.0, 0.2, 0.3, 0.0, 0.0, 0.0]
    events = _run_series(det, edge, start_t=1.0)
    assert len(events) >= 1, "expected at least one blink on a down-up edge"
    assert len(events) <= 2, "refractory should prevent double-fire on the same edge"


def test_shape_ignores_single_peak_noise():
    """Stationary random noise with no coherent edge pattern should not
    produce many false events (refractory bounds the count)."""
    config = BlinkDetectionConfig(method="shape")
    det = _TwinkleShapeSegmentationDetector(config)
    rng = np.random.default_rng(42)
    events = _run_series(det, list(rng.normal(0.0, 0.01, 200)))
    assert len(events) <= 2, f"random noise produced {len(events)} false positives"


def test_shape_refractory_prevents_double_fire():
    """A single clean blink should fire exactly once even if the edge is
    sharp enough to look like two peaks."""
    config = BlinkDetectionConfig(method="shape")
    det = _TwinkleShapeSegmentationDetector(config)
    # baseline
    _run_series(det, [0.0] * 20)
    # one V-shape blink over 0.25s
    events = _run_series(det, [-0.5, -0.4, 0.0, 0.4, 0.5, 0.0, 0.0], start_t=1.0)
    assert len(events) == 1, f"expected 1 blink, got {len(events)}"


def test_shape_supports_up_down_direction():
    """A real blink may produce an up-then-down edge. Both directions are valid."""
    config = BlinkDetectionConfig(method="shape")
    det = _TwinkleShapeSegmentationDetector(config)
    _run_series(det, [0.0] * 20)
    events = _run_series(det, [0.5, 0.4, 0.0, -0.4, -0.5, 0.0, 0.0], start_t=1.0)
    assert len(events) >= 1, "up-down edge should also trigger"


def test_shape_below_baseline_sigma_does_not_fire():
    """If edge magnitude is below shape_edge_sigma_k * baseline sigma, no fire."""
    # baseline sigma is set high by feeding noise first; k=10 demands a large edge
    det2 = _TwinkleShapeSegmentationDetector(
        BlinkDetectionConfig(method="shape", shape_edge_sigma_k=10.0)
    )
    rng = np.random.default_rng(1)
    _run_series(det2, list(rng.normal(0.0, 0.01, 30)))  # baseline sigma ~0.01
    # now a small edge of 0.05: 0.05/0.01 = 5 sigma < 10 → no fire
    events = _run_series(det2, [0.05, 0.0, -0.05, 0.0], start_t=1.5)
    assert len(events) == 0, f"sub-threshold edge fired {len(events)} times"
