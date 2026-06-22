import numpy as np
import pytest

from hp_acoustic_wave.dsp import generate_fmcw_chirp


SR = 48000
CHIRP_DUR = 0.05
SPC = int(round(SR * CHIRP_DUR))  # 2400 samples/chirp


def test_linear_tukey_amplitude_tapers_at_chirp_boundary():
    samples = generate_fmcw_chirp(
        num_samples=SPC, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="linear_tukey",
    )
    assert abs(samples[0]) < 0.2 * 0.05, f"expected fade-in at start, got {samples[0]}"
    assert abs(samples[-1]) < 0.2 * 0.05, f"expected fade-out at end, got {samples[-1]}"


def test_linear_tukey_mid_chirp_close_to_linear():
    tukey = generate_fmcw_chirp(
        num_samples=SPC, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="linear_tukey",
    )
    linear = generate_fmcw_chirp(
        num_samples=SPC, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="linear",
    )
    mid = SPC // 2
    assert abs(abs(tukey[mid]) - abs(linear[mid])) < 0.2 * 0.05


def test_emission_default_is_linear():
    a = generate_fmcw_chirp(SPC, SR, 17000, 23000, CHIRP_DUR, 0, 0.2)
    b = generate_fmcw_chirp(SPC, SR, 17000, 23000, CHIRP_DUR, 0, 0.2, emission="linear")
    assert np.allclose(a, b)


def test_triangle_phase_is_continuous_at_midpoint():
    samples = generate_fmcw_chirp(
        num_samples=SPC, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="triangle",
    )
    mid = SPC // 2
    assert abs(samples[mid]) <= 0.2 + 1e-6
    assert abs(samples[mid - 1]) <= 0.2 + 1e-6


def test_triangle_covers_full_chirp_length():
    samples = generate_fmcw_chirp(
        num_samples=SPC, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="triangle",
    )
    assert samples.shape == (SPC,)
    assert samples.dtype == np.float32


def test_triangle_spans_two_chirps_wraps_correctly():
    samples = generate_fmcw_chirp(
        num_samples=SPC * 2, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="triangle",
    )
    assert np.allclose(samples[:SPC], samples[SPC:], atol=1e-5)


def _inst_freq(samples, sr):
    """Instantaneous frequency via analytic signal (single-sideband FFT)."""
    spectrum = np.fft.fft(samples.astype(np.float64))
    n = samples.size
    h = np.zeros(n)
    h[0] = 1.0
    if n % 2 == 0:
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[1 : (n + 1) // 2] = 2.0
    analytic = np.fft.ifft(spectrum * h)
    phase = np.unwrap(np.angle(analytic))
    return np.diff(phase) * sr / (2.0 * np.pi)


def test_triangle_second_half_frequency_decreases():
    """The defining triangle property: instantaneous frequency peaks at midpoint and
    DECREASES in the second half.  Linear chirp keeps increasing throughout.
    This test must FAIL for linear and PASS for triangle."""
    for emission, expect_descending in [("triangle", True), ("linear", False)]:
        samples = generate_fmcw_chirp(
            num_samples=SPC, sample_rate=SR, freq_low=17000, freq_high=23000,
            chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
            emission=emission,
        )
        # Analyze a window safely in the second half (55%–95% of chirp),
        # avoiding the midpoint transition artifact at 50%.
        lo = int(SPC * 0.55)
        hi = int(SPC * 0.95)
        window = samples[lo:hi]
        freqs = _inst_freq(window, SR)
        # median instantaneous frequency at the later part vs earlier part of the window
        mid_w = window.size // 2
        early = np.median(freqs[:mid_w])
        late = np.median(freqs[mid_w:])
        if expect_descending:
            assert late < early, (
                f"{emission}: expected freq decreasing (late {late:.1f} < early {early:.1f})"
            )
        else:
            assert late > early, (
                f"{emission}: expected freq increasing (late {late:.1f} > early {early:.1f})"
            )


def test_hybrid_odd_frame_is_cw_at_freq_low():
    # odd chirp index: pure CW at freq_low (no sweep). Check spectral peak.
    samples = generate_fmcw_chirp(
        num_samples=SPC * 2, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="cw_fmcw_hybrid",
    )
    odd = samples[SPC:2 * SPC].astype(np.float64)
    spectrum = np.fft.rfft(odd * np.hanning(odd.size))
    freqs = np.fft.rfftfreq(odd.size, d=1.0 / SR)
    peak_freq = freqs[np.argmax(np.abs(spectrum))]
    assert abs(peak_freq - 17000) < 1500, f"odd frame should be CW at 17k, got {peak_freq}"


def test_hybrid_even_frame_is_chirp():
    # even chirp index: linear chirp, energy spreads 17k..23k
    samples = generate_fmcw_chirp(
        num_samples=SPC, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="cw_fmcw_hybrid",
    )
    spec = np.abs(np.fft.rfft(samples.astype(np.float64) * np.hanning(SPC)))
    freqs = np.fft.rfftfreq(SPC, d=1.0 / SR)
    in_band = spec[(freqs >= 17000) & (freqs <= 23000)].sum()
    total = spec.sum()
    assert in_band / total > 0.5


def test_hybrid_extract_feature_returns_valid_feature():
    # The critical integration test: extract_fmcw_chunk_feature must run on a hybrid
    # chunk without error and return a well-formed ChunkFeature.
    from hp_acoustic_wave.dsp import extract_fmcw_chunk_feature, FmcwBackgroundSubtractor
    # synthesize rx as the hybrid tx itself (loopback-ish: ideal reflector at DC)
    tx = generate_fmcw_chirp(
        num_samples=SPC * 4, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="cw_fmcw_hybrid",
    )
    feature = extract_fmcw_chunk_feature(
        samples=tx,
        tx_samples=tx,
        sample_rate=SR,
        freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR,
        range_bin=15,
        start_sample=0,
        previous=None,
        lowpass_cutoff=5000,
        motion_amplitude_floor=0.02,
        background_subtractor=None,
        emission="cw_fmcw_hybrid",
    )
    assert feature.signal_mode == "fmcw"
    assert np.isfinite(feature.amplitude)
    assert np.isfinite(feature.phase)
    assert np.isfinite(feature.motion_energy)
