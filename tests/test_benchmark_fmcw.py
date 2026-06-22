import math
import wave

import numpy as np

from hp_acoustic_wave.benchmark import replay_feature_rows, reprocess_audio_feature_rows
from hp_acoustic_wave.blink_detector import BlinkDetectionConfig
from hp_acoustic_wave.dsp import generate_fmcw_chirp
from hp_acoustic_wave.benchmark_hp_blink import (
    build_config as build_benchmark_config,
    parse_args as parse_benchmark_args,
)


def test_replay_feature_rows_preserves_fmcw_signal_mode_for_twinkle_gates():
    config = BlinkDetectionConfig(
        method="twinkle",
        min_history=8,
        history_size=40,
        threshold_k=2.0,
        min_score=0.006,
        refractory_s=0.05,
        phase_step_floor=0.005,
        twinkle_peak_gate_enabled=True,
        twinkle_max_peak_score=2.0,
        twinkle_max_motion_energy=10.0,
        twinkle_max_sign_changes=4,
        twinkle_large_motion_score=10.0,
        twinkle_fmcw_min_score=0.30,
    )
    phases = [0.0] * 10 + [0.08, 0.16, 0.08, 0.0]
    rows = [
        {
            "time_s": f"{index * 0.02:.6f}",
            "sample_index": str(index),
            "i": f"{math.cos(phase):.9f}",
            "q": f"{math.sin(phase):.9f}",
            "amplitude": "1.000000000",
            "amplitude_delta": "0.000000000",
            "phase": f"{phase:.9f}",
            "phase_delta": "0.000000000",
            "motion_energy": "0.000000000",
            "rms": "0.100000000",
            "peak_abs": "0.200000000",
            "signal_mode": "fmcw",
            "range_bin": "15",
            "range_distance_m": "0.428750000",
        }
        for index, phase in enumerate(phases)
    ]

    events = replay_feature_rows(rows, config)

    assert events == []


def test_reprocess_audio_feature_rows_can_replay_fmcw_sessions(tmp_path):
    sample_rate = 48_000
    chunk_size = 2_400
    chirp = generate_fmcw_chirp(
        chunk_size,
        sample_rate=sample_rate,
        freq_low=17_000.0,
        freq_high=23_000.0,
        chirp_duration=0.05,
        start_sample=0,
        amplitude=0.2,
    )
    samples = np.roll(chirp, 40).astype(np.float32)
    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())

    rows = reprocess_audio_feature_rows(
        audio_path,
        sample_rate=sample_rate,
        tone_hz=18_500.0,
        chunk_size=chunk_size,
        signal_mode="fmcw",
        fmcw_freq_low=17_000.0,
        fmcw_freq_high=23_000.0,
        fmcw_chirp_duration=0.05,
        fmcw_range_bin=15,
        fmcw_lowpass_cutoff=5_000.0,
    )

    assert rows[0]["signal_mode"] == "fmcw"
    assert rows[0]["range_bin"] == "15"
    assert rows[0]["phase_pair_delta"] != ""


def test_benchmark_cli_respects_intra_chirp_phase_pair_flag():
    args = parse_benchmark_args(
        [
            "--blink-twinkle-fmcw-use-intra-chirp-phase-pair",
        ]
    )

    config = build_benchmark_config(args)

    assert config.twinkle_fmcw_use_intra_chirp_phase_pair is True
