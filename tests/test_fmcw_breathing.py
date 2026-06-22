from pathlib import Path

import numpy as np

from hp_acoustic_wave import fmcw_breathing as fmcw


def test_background_subtract_removes_static_component():
    static = np.array([10.0, 20.0, 30.0])
    dynamic = np.array(
        [
            [0.0, 1.0, -1.0],
            [2.0, -1.0, -1.0],
            [-2.0, 0.0, 2.0],
        ]
    )
    spectra = static + dynamic

    subtracted = fmcw.background_subtract(spectra)

    np.testing.assert_allclose(subtracted.mean(axis=0), np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(subtracted, dynamic - dynamic.mean(axis=0), atol=1e-12)


def test_idx_to_distance_uses_fmcw_slope_and_round_trip():
    cfg = fmcw.FmcwConfig(
        sample_rate=48_000,
        freq_low=17_000,
        freq_high=23_000,
        chirp_duration=0.05,
        sound_speed=343.0,
    )

    distance = fmcw.idx_to_distance(10, cfg)

    expected_delta_f = 10 * cfg.sample_rate / cfg.samples_per_chirp
    expected = expected_delta_f * cfg.sound_speed / cfg.chirp_slope / 2.0
    assert distance == expected


def test_process_breathing_npz_extracts_expected_range_bin_phase():
    result = fmcw.process_npz(
        "refer/FMCW/breathing_1_rangebin=18.npz",
        range_bin=18,
    )

    assert result.range_bin == 18
    assert result.unwrap_phase.shape == (200,)
    assert np.isfinite(result.unwrap_phase).all()
    assert np.ptp(result.unwrap_phase) > 1.0
    assert result.phase_by_bin.shape[1] == fmcw.FmcwConfig().samples_per_chirp // 2 + 1


def test_smooth_phase_trace_keeps_length_and_constant_edges():
    values = np.array([3.0, 3.0, 3.0, 9.0, 9.0, 9.0])

    smoothed = fmcw.smooth_phase_trace(values, window=3)

    assert smoothed.shape == values.shape
    assert smoothed[0] == values[0]
    assert smoothed[-1] == values[-1]
    assert smoothed[2] > values[2]
    assert smoothed[3] < values[3]


def test_save_outputs_includes_clean_breath_plot(tmp_path: Path):
    result = fmcw.process_npz(
        "refer/FMCW/live/live_20260617_101435.npz",
        range_bin=15,
    )

    written = fmcw.save_outputs(result, tmp_path, "live_sample_rangebin_15")

    assert tmp_path.joinpath("live_sample_rangebin_15_breath_rangebin15_clean.png").exists()
    assert any(path.name.endswith("_breath_rangebin15_clean.png") for path in written)
