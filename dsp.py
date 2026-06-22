from dataclasses import dataclass
import math
from typing import Optional, Tuple

import numpy as np


@dataclass
class ChunkFeature:
    time_s: float
    sample_index: int
    i_value: float
    q_value: float
    amplitude: float
    amplitude_delta: float
    phase: float
    phase_delta: float
    motion_energy: float
    rms: float
    peak_abs: float
    signal_mode: str = "tone"
    range_bin: Optional[int] = None
    range_distance_m: Optional[float] = None
    phase_pair_delta: Optional[float] = None
    phase_pair_vote_ratio: Optional[float] = None
    phase_pair_consistency: Optional[float] = None
    phase_pair_candidate_count: Optional[int] = None
    phase_pair_deltas: Optional[Tuple[float, ...]] = None
    range_spread_bins: Optional[int] = None
    range_spread_ratio: Optional[float] = None
    range_dominance_ratio: Optional[float] = None


def generate_tone(
    num_samples: int,
    sample_rate: int,
    frequency_hz: float,
    start_sample: int,
    amplitude: float,
) -> np.ndarray:
    indices = np.arange(start_sample, start_sample + num_samples, dtype=np.float64)
    phase = 2.0 * math.pi * frequency_hz * indices / float(sample_rate)
    tone = amplitude * np.sin(phase)
    return tone.astype(np.float32)


def generate_fmcw_chirp(
    num_samples: int,
    sample_rate: int,
    freq_low: float,
    freq_high: float,
    chirp_duration: float,
    start_sample: int,
    amplitude: float,
    emission: str = "linear",
    tukey_alpha: float = 0.2,
) -> np.ndarray:
    samples_per_chirp = int(round(sample_rate * chirp_duration))
    if samples_per_chirp <= 0:
        raise ValueError("chirp_duration is too small")
    phase = _fmcw_chirp_phase(
        num_samples, sample_rate, freq_low, freq_high, chirp_duration, start_sample, emission=emission
    )
    wave = np.cos(phase)
    if emission == "linear_tukey":
        wave = wave * _tukey_envelope_per_chirp(num_samples, samples_per_chirp, tukey_alpha)
    elif emission in ("linear", "cw_fmcw_hybrid", "triangle"):
        pass
    else:
        raise ValueError(f"unknown fmcw emission: {emission}")
    return (float(amplitude) * wave).astype(np.float32)


def _tukey_envelope_per_chirp(num_samples: int, samples_per_chirp: int, alpha: float) -> np.ndarray:
    one = _tukey_window(samples_per_chirp, alpha)
    n_full = num_samples // samples_per_chirp
    tail = num_samples - n_full * samples_per_chirp
    envelope = np.tile(one, n_full) if n_full > 0 else np.asarray([], dtype=np.float64)
    if tail > 0:
        envelope = np.concatenate([envelope, one[:tail]])
    return envelope


def unwrap_delta(current_phase: float, previous_phase: float) -> float:
    delta = current_phase - previous_phase
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


def _as_float32_mono(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32)
    if array.ndim == 2:
        if array.shape[1] == 0:
            return np.asarray([], dtype=np.float32)
        array = np.mean(array, axis=1, dtype=np.float32)
    return array.reshape(-1)


def _fmcw_chirp_phase(
    num_samples: int,
    sample_rate: int,
    freq_low: float,
    freq_high: float,
    chirp_duration: float,
    start_sample: int,
    emission: str = "linear",
) -> np.ndarray:
    samples_per_chirp = int(round(sample_rate * chirp_duration))
    if samples_per_chirp <= 0:
        raise ValueError("chirp_duration is too small")
    indices = np.arange(start_sample, start_sample + num_samples, dtype=np.int64)
    chirp_indices = np.mod(indices, samples_per_chirp).astype(np.float64)
    t = chirp_indices / float(sample_rate)
    slope = (float(freq_high) - float(freq_low)) / float(chirp_duration)

    if emission == "triangle":
        half = samples_per_chirp / 2.0
        t_half = chirp_duration / 2.0
        is_down = chirp_indices >= half
        t_local = np.where(is_down, t - t_half, t)
        phase_up = 2.0 * math.pi * (float(freq_low) * t + 0.5 * slope * t * t)
        phase_at_half = 2.0 * math.pi * (
            float(freq_low) * t_half + 0.5 * slope * t_half * t_half
        )
        phase_down = phase_at_half + 2.0 * math.pi * (
            float(freq_high) * t_local - 0.5 * slope * t_local * t_local
        )
        return np.where(is_down, phase_down, phase_up)

    if emission == "cw_fmcw_hybrid":
        # even chirp frames -> linear chirp; odd chirp frames -> pure CW at freq_low
        chirp_index = (indices // samples_per_chirp).astype(np.int64)
        is_cw_frame = (chirp_index % 2 == 1)
        cw_phase = 2.0 * math.pi * float(freq_low) * t
        chirp_phase_lin = 2.0 * math.pi * (float(freq_low) * t + 0.5 * slope * t * t)
        return np.where(is_cw_frame, cw_phase, chirp_phase_lin)

    return 2.0 * math.pi * (float(freq_low) * t + 0.5 * slope * t * t)


def _lowpass_complex(samples: np.ndarray, sample_rate: int, cutoff_hz: float) -> np.ndarray:
    spectrum = np.fft.fft(samples)
    freqs = np.fft.fftfreq(samples.size, d=1.0 / float(sample_rate))
    spectrum[np.abs(freqs) > float(cutoff_hz)] = 0
    return np.fft.ifft(spectrum)


class FmcwBackgroundSubtractor:
    def __init__(self, alpha: float = 0.02, window: int = 0):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.window = int(window)  # EXP-C: 0 = EMA mode; >0 = sliding-window mean
        self.background: Optional[np.ndarray] = None
        self.frames = 0
        # EXP-C: circular buffer of recent complex_bins frames
        self._history: list = []

    def apply(self, complex_bins: np.ndarray) -> np.ndarray:
        bins = np.asarray(complex_bins, dtype=np.complex128)
        if self.window > 0:
            # EXP-C: sliding-window mean background
            self._history.append(bins.copy())
            if len(self._history) > self.window:
                self._history.pop(0)
            if len(self._history) <= 1:
                self.background = bins.copy()
                self.frames = 1
                return np.zeros_like(bins)
            stack = np.asarray(self._history)
            self.background = np.mean(stack, axis=0)
            self.frames = len(self._history)
            return bins - self.background

        if self.background is None or self.background.shape != bins.shape:
            self.background = bins.copy()
            self.frames = 1
            return np.zeros_like(bins)

        corrected = bins - self.background
        self.background = (1.0 - self.alpha) * self.background + self.alpha * bins
        self.frames += 1
        return corrected


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    target = 0.5 * float(np.sum(sorted_weights))
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(max(index, 0), sorted_values.size - 1)
    return float(sorted_values[index])


def _phase_pair_features(
    baseband: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float], int, Tuple[float, ...]]:
    if baseband.size < 8:
        return None, None, None, 0, tuple()

    pair_fractions = (
        (0.15, 0.85),
        (0.20, 0.80),
        (0.25, 0.75),
        (0.30, 0.70),
        (0.35, 0.65),
        (0.40, 0.60),
    )
    deltas = []
    weights = []
    pair_delta_bank = []
    for first_frac, second_frac in pair_fractions:
        first_index = int(round((baseband.size - 1) * float(first_frac)))
        second_index = int(round((baseband.size - 1) * float(second_frac)))
        if first_index >= second_index:
            pair_delta_bank.append(float("nan"))
            continue
        first = complex(baseband[first_index])
        second = complex(baseband[second_index])
        magnitude = min(abs(first), abs(second))
        if magnitude < 1e-12:
            pair_delta_bank.append(float("nan"))
            continue
        delta = unwrap_delta(
            math.atan2(second.imag, second.real),
            math.atan2(first.imag, first.real),
        )
        deltas.append(float(delta))
        weights.append(float(magnitude))
        pair_delta_bank.append(float(delta))

    if not deltas:
        return None, None, None, 0, tuple(pair_delta_bank)

    values = np.asarray(deltas, dtype=np.float64)
    weight_array = np.maximum(np.asarray(weights, dtype=np.float64), 1e-9)
    center = _weighted_median(values, weight_array)
    relative_offsets = np.asarray(
        [abs(unwrap_delta(float(value), center)) for value in values],
        dtype=np.float64,
    )
    agreement_threshold = 0.35
    vote_ratio = float(np.sum(weight_array[relative_offsets <= agreement_threshold]) / np.sum(weight_array))
    dispersion = float(np.sqrt(np.average(np.square(relative_offsets), weights=weight_array)))
    consistency = float(max(0.0, 1.0 - min(1.0, dispersion / math.pi)))
    return float(center), vote_ratio, consistency, int(values.size), tuple(pair_delta_bank)


def calculate_fmcw_spatial_spread(
    complex_bins: np.ndarray,
    range_bin: int,
    radius_bins: int = 6,
    active_threshold_ratio: float = 0.35,
) -> Tuple[int, float, float]:
    magnitudes = np.abs(np.asarray(complex_bins, dtype=np.complex128))
    if magnitudes.size == 0:
        return 0, 0.0, 0.0

    center_bin = int(range_bin)
    lo = max(0, center_bin - max(0, int(radius_bins)))
    hi = min(int(magnitudes.size) - 1, center_bin + max(0, int(radius_bins)))
    band = magnitudes[lo : hi + 1]
    if band.size == 0:
        return 0, 0.0, 0.0

    center_index = min(max(center_bin - lo, 0), int(band.size) - 1)
    center_amplitude = float(band[center_index])
    peak_amplitude = max(float(np.max(band)), 1e-12)
    threshold = max(1e-12, peak_amplitude * max(0.0, float(active_threshold_ratio)))
    active_bins = int(np.sum(band >= threshold))
    spread_ratio = float(active_bins / float(band.size))
    dominance_ratio = float(center_amplitude / max(float(np.sum(band)), 1e-12))
    return active_bins, spread_ratio, dominance_ratio


def extract_chunk_feature(
    samples: np.ndarray,
    sample_rate: int,
    tone_hz: float,
    start_sample: int,
    previous: Optional[ChunkFeature],
    tukey_alpha: float = 0.0,
) -> ChunkFeature:
    mono = _as_float32_mono(samples)
    if mono.size == 0:
        raise ValueError("samples must not be empty")

    analysis_samples = mono.astype(np.float64)
    if tukey_alpha > 0.0:
        analysis_samples = analysis_samples * _tukey_window(mono.size, tukey_alpha)

    indices = np.arange(start_sample, start_sample + mono.size, dtype=np.float64)
    mixer = np.exp(-1j * 2.0 * math.pi * tone_hz * indices / float(sample_rate))
    baseband = analysis_samples * mixer
    complex_mean = np.mean(baseband)

    i_value = float(np.real(complex_mean))
    q_value = float(np.imag(complex_mean))
    amplitude = float(abs(complex_mean))
    phase = float(math.atan2(q_value, i_value))
    rms = float(np.sqrt(np.mean(np.square(mono.astype(np.float64)))))
    peak_abs = float(np.max(np.abs(mono)))

    if previous is None:
        amplitude_delta = 0.0
        phase_delta = 0.0
    else:
        amplitude_delta = amplitude - previous.amplitude
        phase_delta = unwrap_delta(phase, previous.phase)

    reference_amplitude = previous.amplitude if previous is not None else amplitude
    relative_amp_delta = abs(amplitude_delta) / max(reference_amplitude, amplitude, 1e-3)
    phase_assist = 0.15 * min(math.pi, abs(phase_delta)) / math.pi
    motion_energy = float(min(5.0, relative_amp_delta) + phase_assist)

    return ChunkFeature(
        time_s=float(start_sample) / float(sample_rate),
        sample_index=int(start_sample),
        i_value=i_value,
        q_value=q_value,
        amplitude=amplitude,
        amplitude_delta=float(amplitude_delta),
        phase=phase,
        phase_delta=float(phase_delta),
        motion_energy=motion_energy,
        rms=rms,
        peak_abs=peak_abs,
    )


def extract_fmcw_chunk_feature(
    samples: np.ndarray,
    tx_samples: np.ndarray,
    sample_rate: int,
    freq_low: float,
    freq_high: float,
    chirp_duration: float,
    range_bin: int,
    start_sample: int,
    previous: Optional[ChunkFeature],
    lowpass_cutoff: float = 5_000.0,
    motion_amplitude_floor: float = 0.02,
    background_subtractor: Optional[FmcwBackgroundSubtractor] = None,
    emission: str = "linear",
) -> ChunkFeature:
    rx = _as_float32_mono(samples)
    tx = _as_float32_mono(tx_samples)
    if rx.size == 0 or tx.size == 0:
        raise ValueError("samples and tx_samples must not be empty")
    usable = min(rx.size, tx.size)
    rx64 = rx[:usable].astype(np.float64)
    tx64 = tx[:usable].astype(np.float64)
    # Hybrid-only: frame mask. True = chirp frame (range-bin path), False = CW frame.
    frame_mask = np.ones(usable, dtype=bool)
    if emission == "cw_fmcw_hybrid":
        spc = int(round(sample_rate * chirp_duration))
        if spc > 0:
            sample_indices = np.arange(start_sample, start_sample + usable)
            chirp_index = sample_indices // spc
            frame_mask = (chirp_index % 2 == 0)
    mixed = rx64 * tx64
    if emission == "cw_fmcw_hybrid":
        mixed = mixed * frame_mask
    spectrum = np.fft.rfft(mixed)
    freqs = np.fft.rfftfreq(usable, d=1.0 / float(sample_rate))
    spectrum[freqs > float(lowpass_cutoff)] = 0
    lowpassed = np.fft.irfft(spectrum, n=usable)
    chirp_phase = _fmcw_chirp_phase(
        usable,
        sample_rate=sample_rate,
        freq_low=freq_low,
        freq_high=freq_high,
        chirp_duration=chirp_duration,
        start_sample=start_sample,
        emission=emission,
    )
    complex_reference = np.exp(-1j * chirp_phase)
    cw_mask = (~frame_mask).astype(np.float64) if emission == "cw_fmcw_hybrid" else np.ones(usable)
    complex_baseband = _lowpass_complex((rx64 * complex_reference) * cw_mask, sample_rate, lowpass_cutoff)
    (
        intra_chirp_phase_pair_delta,
        intra_chirp_phase_pair_vote_ratio,
        intra_chirp_phase_pair_consistency,
        intra_chirp_phase_pair_candidate_count,
        intra_chirp_phase_pair_deltas,
    ) = _phase_pair_features(complex_baseband)
    complex_bins = np.fft.rfft(lowpassed)
    if background_subtractor is not None:
        complex_bins = background_subtractor.apply(complex_bins)
    if not 0 <= int(range_bin) < complex_bins.shape[0]:
        raise ValueError(f"range_bin {range_bin} outside 0..{complex_bins.shape[0] - 1}")
    range_spread_bins, range_spread_ratio, range_dominance_ratio = calculate_fmcw_spatial_spread(
        complex_bins,
        int(range_bin),
    )
    value = complex_bins[int(range_bin)]
    i_value = float(np.real(value))
    q_value = float(np.imag(value))
    amplitude = float(abs(value))
    phase = float(math.atan2(q_value, i_value))
    rms = float(np.sqrt(np.mean(np.square(rx64))))
    peak_abs = float(np.max(np.abs(rx64)))

    if previous is None:
        amplitude_delta = 0.0
        phase_delta = 0.0
    else:
        amplitude_delta = amplitude - previous.amplitude
        phase_delta = unwrap_delta(phase, previous.phase)

    reference_amplitude = previous.amplitude if previous is not None else amplitude
    motion_energy = calculate_fmcw_motion_energy(
        amplitude=amplitude,
        previous_amplitude=reference_amplitude,
        phase_delta=phase_delta,
        amplitude_floor=motion_amplitude_floor,
    )

    return ChunkFeature(
        time_s=float(start_sample) / float(sample_rate),
        sample_index=int(start_sample),
        i_value=i_value,
        q_value=q_value,
        amplitude=amplitude,
        amplitude_delta=float(amplitude_delta),
        phase=phase,
        phase_delta=float(phase_delta),
        motion_energy=motion_energy,
        rms=rms,
        peak_abs=peak_abs,
        signal_mode="fmcw",
        range_bin=int(range_bin),
        range_distance_m=idx_to_fmcw_distance(
            int(range_bin),
            sample_rate=sample_rate,
            freq_low=freq_low,
            freq_high=freq_high,
            chirp_duration=chirp_duration,
        ),
        phase_pair_delta=intra_chirp_phase_pair_delta,
        phase_pair_vote_ratio=intra_chirp_phase_pair_vote_ratio,
        phase_pair_consistency=intra_chirp_phase_pair_consistency,
        phase_pair_candidate_count=intra_chirp_phase_pair_candidate_count,
        phase_pair_deltas=intra_chirp_phase_pair_deltas,
        range_spread_bins=range_spread_bins,
        range_spread_ratio=range_spread_ratio,
        range_dominance_ratio=range_dominance_ratio,
    )


def idx_to_fmcw_distance(
    idx: int,
    sample_rate: int,
    freq_low: float,
    freq_high: float,
    chirp_duration: float,
    sound_speed: float = 343.0,
) -> float:
    samples_per_chirp = int(round(sample_rate * chirp_duration))
    frequency_resolution = float(sample_rate) / float(samples_per_chirp)
    chirp_slope = (float(freq_high) - float(freq_low)) / float(chirp_duration)
    delta_f = float(idx) * frequency_resolution
    return float((delta_f * float(sound_speed) / chirp_slope) / 2.0)


def calculate_fmcw_motion_energy(
    amplitude: float,
    previous_amplitude: float,
    phase_delta: float,
    amplitude_floor: float = 0.02,
) -> float:
    relative_amp_delta = abs(float(amplitude) - float(previous_amplitude)) / max(
        abs(float(previous_amplitude)),
        abs(float(amplitude)),
        float(amplitude_floor),
    )
    phase_motion = min(math.pi, abs(float(phase_delta))) / math.pi
    return float(min(5.0, relative_amp_delta) + phase_motion)


def _tukey_window(length: int, alpha: float) -> np.ndarray:
    if length <= 0:
        return np.asarray([], dtype=np.float64)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return np.ones(length, dtype=np.float64)
    if length == 1:
        return np.ones(1, dtype=np.float64)
    positions = np.linspace(0.0, 1.0, length, dtype=np.float64)
    window = np.ones(length, dtype=np.float64)
    if alpha >= 1.0:
        return 0.5 - 0.5 * np.cos(2.0 * math.pi * positions)

    first = positions < alpha / 2.0
    last = positions >= 1.0 - alpha / 2.0
    window[first] = 0.5 * (1.0 + np.cos(2.0 * math.pi / alpha * (positions[first] - alpha / 2.0)))
    window[last] = 0.5 * (1.0 + np.cos(2.0 * math.pi / alpha * (positions[last] - 1.0 + alpha / 2.0)))
    return window
