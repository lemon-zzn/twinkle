import numpy as np
from unittest.mock import patch

from hp_acoustic_wave.app import RealtimeHandWaveApp, resolve_camera_backend_name
from hp_acoustic_wave.config import AppConfig, AudioConfig, CameraConfig
from hp_acoustic_wave.dsp import (
    calculate_fmcw_motion_energy,
    calculate_fmcw_spatial_spread,
    extract_fmcw_chunk_feature,
    FmcwBackgroundSubtractor,
    generate_fmcw_chirp,
    idx_to_fmcw_distance,
)
from hp_acoustic_wave.run_hp_wave_detector import build_app_config, list_cameras, parse_args


def test_generate_fmcw_chirp_resets_each_chirp_period():
    sample_rate = 48_000
    chirp_duration = 0.05
    samples_per_chirp = int(sample_rate * chirp_duration)

    first = generate_fmcw_chirp(
        samples_per_chirp,
        sample_rate=sample_rate,
        freq_low=17_000.0,
        freq_high=23_000.0,
        chirp_duration=chirp_duration,
        start_sample=0,
        amplitude=0.2,
    )
    second = generate_fmcw_chirp(
        samples_per_chirp,
        sample_rate=sample_rate,
        freq_low=17_000.0,
        freq_high=23_000.0,
        chirp_duration=chirp_duration,
        start_sample=samples_per_chirp,
        amplitude=0.2,
    )

    assert first.shape == (samples_per_chirp,)
    assert np.max(np.abs(first)) <= 0.200001
    np.testing.assert_allclose(first, second, atol=1e-6)


def test_fmcw_motion_energy_ignores_near_zero_amplitude_jitter():
    energy = calculate_fmcw_motion_energy(
        amplitude=0.00015,
        previous_amplitude=0.00010,
        phase_delta=0.0,
    )

    assert energy < 0.01


def test_calculate_fmcw_spatial_spread_prefers_localized_energy():
    localized_bins = np.zeros(64, dtype=np.complex128)
    localized_bins[15] = 1.0 + 0.0j
    localized_bins[14] = 0.15 + 0.0j
    localized_bins[16] = 0.12 + 0.0j
    local_active, local_spread, local_dominance = calculate_fmcw_spatial_spread(localized_bins, 15)

    diffuse_bins = np.zeros(64, dtype=np.complex128)
    for index in range(9, 22):
        diffuse_bins[index] = (0.25 + 0.0j)
    diffuse_bins[15] = 0.30 + 0.0j
    diffuse_active, diffuse_spread, diffuse_dominance = calculate_fmcw_spatial_spread(diffuse_bins, 15)

    assert local_active < diffuse_active
    assert local_spread < diffuse_spread
    assert local_dominance > diffuse_dominance


def test_extract_fmcw_chunk_feature_returns_range_bin_phase_feature():
    sample_rate = 48_000
    chirp_duration = 0.05
    samples_per_chirp = int(sample_rate * chirp_duration)
    tx = generate_fmcw_chirp(
        samples_per_chirp,
        sample_rate=sample_rate,
        freq_low=17_000.0,
        freq_high=23_000.0,
        chirp_duration=chirp_duration,
        start_sample=0,
        amplitude=0.2,
    )
    rx = np.roll(tx, 40) * 0.4

    first = extract_fmcw_chunk_feature(
        samples=rx,
        tx_samples=tx,
        sample_rate=sample_rate,
        freq_low=17_000.0,
        freq_high=23_000.0,
        chirp_duration=chirp_duration,
        range_bin=15,
        start_sample=0,
        previous=None,
    )
    second = extract_fmcw_chunk_feature(
        samples=rx * 1.4,
        tx_samples=tx,
        sample_rate=sample_rate,
        freq_low=17_000.0,
        freq_high=23_000.0,
        chirp_duration=chirp_duration,
        range_bin=15,
        start_sample=samples_per_chirp,
        previous=first,
    )

    assert first.signal_mode == "fmcw"
    assert first.range_bin == 15
    assert first.range_distance_m == idx_to_fmcw_distance(15, sample_rate, 17_000.0, 23_000.0, chirp_duration)
    assert np.isfinite(first.phase)
    assert first.phase_pair_delta is not None
    assert np.isfinite(first.phase_pair_delta)
    assert first.phase_pair_vote_ratio is not None
    assert 0.0 <= first.phase_pair_vote_ratio <= 1.0
    assert first.phase_pair_consistency is not None
    assert 0.0 <= first.phase_pair_consistency <= 1.0
    assert first.phase_pair_candidate_count is not None
    assert first.phase_pair_candidate_count >= 2
    assert first.amplitude > 0.0
    assert second.motion_energy > 0.0


def test_extract_fmcw_chunk_feature_peak_bin_matches_known_echo_delay():
    sample_rate = 48_000
    chirp_duration = 0.05
    samples_per_chirp = int(sample_rate * chirp_duration)
    freq_low = 17_000.0
    freq_high = 23_000.0
    amplitude = 0.2
    start_sample = samples_per_chirp * 3

    for expected_bin in (5, 10, 15, 20, 25, 30):
        expected_distance = idx_to_fmcw_distance(
            expected_bin,
            sample_rate,
            freq_low,
            freq_high,
            chirp_duration,
        )
        echo_delay_samples = int(round((2.0 * expected_distance / 343.0) * sample_rate))
        tx_long = generate_fmcw_chirp(
            start_sample + samples_per_chirp,
            sample_rate=sample_rate,
            freq_low=freq_low,
            freq_high=freq_high,
            chirp_duration=chirp_duration,
            start_sample=0,
            amplitude=amplitude,
        )
        tx_chunk = generate_fmcw_chirp(
            samples_per_chirp,
            sample_rate=sample_rate,
            freq_low=freq_low,
            freq_high=freq_high,
            chirp_duration=chirp_duration,
            start_sample=start_sample,
            amplitude=amplitude,
        )
        rx_chunk = tx_long[
            start_sample - echo_delay_samples : start_sample - echo_delay_samples + samples_per_chirp
        ]

        amplitudes = [
            extract_fmcw_chunk_feature(
                samples=rx_chunk,
                tx_samples=tx_chunk,
                sample_rate=sample_rate,
                freq_low=freq_low,
                freq_high=freq_high,
                chirp_duration=chirp_duration,
                range_bin=range_bin,
                start_sample=start_sample,
                previous=None,
            ).amplitude
            for range_bin in range(1, 36)
        ]

        detected_bin = int(np.argmax(amplitudes)) + 1
        assert detected_bin == expected_bin


def test_fmcw_background_subtraction_removes_static_range_bin_echo():
    sample_rate = 48_000
    chirp_duration = 0.05
    samples_per_chirp = int(sample_rate * chirp_duration)
    freq_low = 17_000.0
    freq_high = 23_000.0
    amplitude = 0.2
    range_bin = 15
    start_sample = samples_per_chirp * 3
    echo_delay_samples = 120
    tx_long = generate_fmcw_chirp(
        start_sample + samples_per_chirp,
        sample_rate=sample_rate,
        freq_low=freq_low,
        freq_high=freq_high,
        chirp_duration=chirp_duration,
        start_sample=0,
        amplitude=amplitude,
    )
    tx_chunk = generate_fmcw_chirp(
        samples_per_chirp,
        sample_rate=sample_rate,
        freq_low=freq_low,
        freq_high=freq_high,
        chirp_duration=chirp_duration,
        start_sample=start_sample,
        amplitude=amplitude,
    )
    rx_chunk = tx_long[start_sample - echo_delay_samples : start_sample - echo_delay_samples + samples_per_chirp]

    raw_feature = extract_fmcw_chunk_feature(
        samples=rx_chunk,
        tx_samples=tx_chunk,
        sample_rate=sample_rate,
        freq_low=freq_low,
        freq_high=freq_high,
        chirp_duration=chirp_duration,
        range_bin=range_bin,
        start_sample=start_sample,
        previous=None,
    )
    background = FmcwBackgroundSubtractor()
    first_subtracted = extract_fmcw_chunk_feature(
        samples=rx_chunk,
        tx_samples=tx_chunk,
        sample_rate=sample_rate,
        freq_low=freq_low,
        freq_high=freq_high,
        chirp_duration=chirp_duration,
        range_bin=range_bin,
        start_sample=start_sample,
        previous=None,
        background_subtractor=background,
    )
    second_subtracted = extract_fmcw_chunk_feature(
        samples=rx_chunk,
        tx_samples=tx_chunk,
        sample_rate=sample_rate,
        freq_low=freq_low,
        freq_high=freq_high,
        chirp_duration=chirp_duration,
        range_bin=range_bin,
        start_sample=start_sample + samples_per_chirp,
        previous=first_subtracted,
        background_subtractor=background,
    )

    assert raw_feature.amplitude > 1.0
    assert second_subtracted.amplitude < raw_feature.amplitude * 0.01


def test_parse_args_accepts_fmcw_wave_mode():
    args = parse_args(
        [
            "--signal-mode",
            "fmcw",
            "--fmcw-range-bin",
            "15",
            "--fmcw-freq-low",
            "17000",
            "--fmcw-freq-high",
            "23000",
            "--fmcw-motion-amplitude-floor",
            "0.02",
        ]
    )

    assert args.signal_mode == "fmcw"
    assert args.fmcw_range_bin == 15
    assert args.fmcw_freq_low == 17_000.0
    assert args.fmcw_freq_high == 23_000.0
    assert args.fmcw_motion_amplitude_floor == 0.02


def test_parse_args_accepts_fmcw_twinkle_blink_tuning():
    args = parse_args(
        [
            "--mode",
            "blink",
            "--blink-method",
            "twinkle",
            "--signal-mode",
            "fmcw",
            "--fmcw-range-bin",
            "15",
            "--blink-twinkle-fmcw-min-amplitude",
            "0.05",
            "--blink-twinkle-fmcw-phase-pair-weight",
            "0.8",
            "--blink-twinkle-fmcw-max-amplitude-delta-ratio",
            "0.35",
            "--blink-twinkle-fmcw-min-range-spread-ratio",
            "0.46",
            "--blink-twinkle-fmcw-min-score",
            "0.09",
            "--blink-twinkle-fmcw-refractory",
            "1.7",
            "--blink-twinkle-fmcw-use-intra-chirp-phase-pair",
            "--blink-candidate-windows",
            "5,7,9,11",
            "--blink-min-candidate-votes",
            "2",
        ]
    )

    assert args.mode == "blink"
    assert args.blink_method == "twinkle"
    assert args.signal_mode == "fmcw"
    assert args.fmcw_range_bin == 15
    assert args.blink_twinkle_fmcw_min_amplitude == 0.05
    assert args.blink_twinkle_fmcw_phase_pair_weight == 0.8
    assert args.blink_twinkle_fmcw_max_amplitude_delta_ratio == 0.35
    assert args.blink_twinkle_fmcw_min_range_spread_ratio == 0.46
    assert args.blink_twinkle_fmcw_min_score == 0.09
    assert args.blink_twinkle_fmcw_refractory == 1.7
    assert args.blink_twinkle_fmcw_use_intra_chirp_phase_pair is True
    assert args.blink_candidate_windows == "5,7,9,11"
    assert args.blink_min_candidate_votes == 2


def test_camera_backend_auto_uses_avfoundation_on_macos():
    with patch("hp_acoustic_wave.app.platform.system", return_value="Darwin"):
        assert resolve_camera_backend_name("auto") == "avfoundation"


def test_list_cameras_uses_resolved_backend(capsys):
    class FakeCamera:
        def __init__(self, opened):
            self.opened = opened

        def isOpened(self):
            return self.opened

        def read(self):
            return True, np.zeros((3, 4, 3), dtype=np.uint8)

        def release(self):
            pass

    class FakeCv2:
        CAP_AVFOUNDATION = 1200

        def __init__(self):
            self.calls = []

        def VideoCapture(self, index, backend=None):
            self.calls.append((index, backend))
            return FakeCamera(opened=index == 0)

    fake_cv2 = FakeCv2()
    with patch("hp_acoustic_wave.app.platform.system", return_value="Darwin"):
        list_cameras(fake_cv2, "auto", 2)

    assert fake_cv2.calls == [(0, FakeCv2.CAP_AVFOUNDATION), (1, FakeCv2.CAP_AVFOUNDATION)]
    captured = capsys.readouterr()
    assert "Camera probe backend: avfoundation" in captured.out
    assert "[0] ok 4x3" in captured.out


def test_build_app_config_respects_intra_chirp_cli_flag():
    args = parse_args(
        [
            "--mode",
            "blink",
            "--blink-method",
            "twinkle",
            "--signal-mode",
            "fmcw",
            "--blink-twinkle-fmcw-use-intra-chirp-phase-pair",
        ]
    )

    config = build_app_config(args)

    assert config.blink.twinkle_fmcw_use_intra_chirp_phase_pair is True


def test_parse_args_uses_visual_truth_tuned_fmcw_twinkle_defaults():
    args = parse_args(["--mode", "blink", "--blink-method", "twinkle", "--signal-mode", "fmcw"])

    assert args.blink_candidate_windows == "0"
    assert args.blink_min_candidate_votes == 1
    assert args.blink_twinkle_fmcw_min_amplitude == 0.005
    assert args.blink_twinkle_fmcw_min_score == 0.09
    assert args.blink_twinkle_fmcw_min_range_spread_ratio == 0.0
    assert args.blink_twinkle_fmcw_refractory == 1.3
    assert args.blink_twinkle_fmcw_use_intra_chirp_phase_pair is False


def test_app_audio_callback_outputs_fmcw_chirp_when_configured():
    cfg = AppConfig(
        audio=AudioConfig(
            signal_mode="fmcw",
            output_amplitude=0.2,
            chunk_size=2400,
            fmcw_freq_low=17_000.0,
            fmcw_freq_high=23_000.0,
            fmcw_chirp_duration=0.05,
            fmcw_range_bin=15,
        ),
        camera=CameraConfig(enabled=False),
    )
    app = RealtimeHandWaveApp(cfg)
    indata = np.zeros((2400, 1), dtype=np.float32)
    outdata = np.zeros((2400, 1), dtype=np.float32)

    app._audio_callback(indata, outdata, 2400, None, "")

    expected = generate_fmcw_chirp(
        2400,
        sample_rate=cfg.audio.sample_rate,
        freq_low=cfg.audio.fmcw_freq_low,
        freq_high=cfg.audio.fmcw_freq_high,
        chirp_duration=cfg.audio.fmcw_chirp_duration,
        start_sample=0,
        amplitude=cfg.audio.output_amplitude,
    )
    np.testing.assert_allclose(outdata[:, 0], expected, atol=1e-6)
