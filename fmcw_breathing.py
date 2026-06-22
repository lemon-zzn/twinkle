"""FMCW acoustic ranging and breathing monitor for CIS3990 Lab 2.

This module is intentionally script-friendly: it does not require Jupyter or
SciPy, so it can run on the HP Windows laptop with the packages already present
in this workspace.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    from .dsp import generate_fmcw_chirp as generate_fmcw_samples
except ImportError:  # pragma: no cover - used when run as a script from this folder
    from dsp import generate_fmcw_chirp as generate_fmcw_samples


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "refer" / "FMCW"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "refer" / "FMCW" / "outputs"
FMCW_EMISSIONS = ("linear", "linear_tukey", "cw_fmcw_hybrid", "triangle")


@dataclass(frozen=True)
class FmcwConfig:
    sample_rate: int = 48_000
    freq_low: float = 17_000.0
    freq_high: float = 23_000.0
    chirp_duration: float = 0.05
    total_duration: float = 10.0
    lowpass_cutoff: float = 5_000.0
    sound_speed: float = 343.0
    emission: str = "linear"

    @property
    def samples_per_chirp(self) -> int:
        return int(round(self.sample_rate * self.chirp_duration))

    @property
    def chirp_slope(self) -> float:
        return (self.freq_high - self.freq_low) / self.chirp_duration

    @property
    def chirps_per_second(self) -> float:
        return 1.0 / self.chirp_duration

    @property
    def frequency_resolution(self) -> float:
        return self.sample_rate / self.samples_per_chirp


@dataclass(frozen=True)
class BreathInterval:
    start_s: float
    end_s: float
    duration_s: float


@dataclass(frozen=True)
class BreathDetection:
    breathing_present: bool
    phase_peak_to_peak_rad: float
    motion_threshold: float
    active_fraction: float
    intervals: Tuple[BreathInterval, ...]


@dataclass
class FmcwResult:
    source_path: Optional[Path]
    config: FmcwConfig
    range_bin: int
    tx_segments: np.ndarray
    rx_segments: np.ndarray
    mixed_lowpass: np.ndarray
    mixed_subtracted: np.ndarray
    complex_by_bin: np.ndarray
    amplitude_by_bin: np.ndarray
    phase_by_bin: np.ndarray
    unwrap_phase: np.ndarray
    peak_bins: np.ndarray
    detection: BreathDetection

    @property
    def timestamps(self) -> np.ndarray:
        return np.arange(self.unwrap_phase.shape[0]) * self.config.chirp_duration


def smooth_sound(samples: np.ndarray, window_length: int = 60) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.size <= 1:
        return samples.astype(np.float32, copy=True)
    if samples.size <= window_length:
        window = np.hanning(samples.size)
    else:
        half = window_length // 2
        taper = np.hanning(window_length)
        middle = np.ones(samples.size - window_length)
        window = np.concatenate([taper[:half], middle, taper[half:]])
    return (samples * window).astype(np.float32)


def generate_chirp(config: FmcwConfig = FmcwConfig()) -> np.ndarray:
    """Generate one FMCW chirp using the configured emission variant."""
    return generate_fmcw_samples(
        num_samples=config.samples_per_chirp,
        sample_rate=config.sample_rate,
        freq_low=config.freq_low,
        freq_high=config.freq_high,
        chirp_duration=config.chirp_duration,
        start_sample=0,
        amplitude=1.0,
        emission=config.emission,
    )


def generate_chirp_train(config: FmcwConfig = FmcwConfig()) -> np.ndarray:
    num_samples = int(round(config.total_duration * config.sample_rate))
    return generate_fmcw_samples(
        num_samples=num_samples,
        sample_rate=config.sample_rate,
        freq_low=config.freq_low,
        freq_high=config.freq_high,
        chirp_duration=config.chirp_duration,
        start_sample=0,
        amplitude=1.0,
        emission=config.emission,
    )


def play_and_record_chirp(
    config: FmcwConfig = FmcwConfig(),
    amplitude: float = 1.0,
    device: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Play an FMCW chirp train and record microphone input.

    This is optional for the assignment because the provided .npz files are
    enough to process breathing offline. It remains useful for live experiments.
    """
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover - depends on local hardware
        raise RuntimeError("sounddevice is required for live recording") from exc

    tx = amplitude * generate_chirp_train(config)
    sd.default.samplerate = config.sample_rate
    sd.default.channels = 1
    rx = sd.playrec(tx.astype(np.float32), samplerate=config.sample_rate, device=device)
    sd.wait()
    return tx.astype(np.float32), np.asarray(rx, dtype=np.float32)


def segment_chirps(
    tx: np.ndarray,
    rx: np.ndarray,
    config: FmcwConfig = FmcwConfig(),
    drop_seconds: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    tx_1d = np.asarray(tx).squeeze()
    rx_1d = np.asarray(rx).squeeze()
    if tx_1d.ndim != 1 or rx_1d.ndim != 1:
        raise ValueError("tx and rx must be one-dimensional after squeezing")

    samples_per_chirp = config.samples_per_chirp
    usable = min(tx_1d.size, rx_1d.size)
    num_chirps = usable // samples_per_chirp
    drop_chirps = int(round(drop_seconds / config.chirp_duration))
    if num_chirps <= drop_chirps:
        raise ValueError("not enough samples for one full chirp after dropping startup")

    usable = num_chirps * samples_per_chirp
    tx_segments = tx_1d[:usable].reshape(num_chirps, samples_per_chirp)
    rx_segments = rx_1d[:usable].reshape(num_chirps, samples_per_chirp)
    if drop_chirps:
        tx_segments = tx_segments[drop_chirps:]
        rx_segments = rx_segments[drop_chirps:]
    return tx_segments.astype(np.float64), rx_segments.astype(np.float64)


def fft_lowpass_segments(
    segments: np.ndarray,
    config: FmcwConfig = FmcwConfig(),
) -> np.ndarray:
    spectra = np.fft.rfft(segments, axis=1)
    freqs = np.fft.rfftfreq(config.samples_per_chirp, d=1.0 / config.sample_rate)
    spectra[:, freqs > config.lowpass_cutoff] = 0
    return np.fft.irfft(spectra, n=config.samples_per_chirp, axis=1)


def mix_and_lowpass(
    tx_segments: np.ndarray,
    rx_segments: np.ndarray,
    config: FmcwConfig = FmcwConfig(),
) -> np.ndarray:
    if tx_segments.shape != rx_segments.shape:
        raise ValueError("tx_segments and rx_segments must have the same shape")
    mixed = tx_segments * rx_segments
    return fft_lowpass_segments(mixed, config)


def background_subtract(values: np.ndarray) -> np.ndarray:
    """Subtract the mean chirp response from every chirp."""
    array = np.asarray(values)
    return array - np.mean(array, axis=0, keepdims=True)


def idx_to_distance(idx: np.ndarray, config: FmcwConfig = FmcwConfig()) -> np.ndarray:
    """Convert beat-frequency FFT bin index to round-trip-corrected distance."""
    delta_f = np.asarray(idx) * config.frequency_resolution
    return (delta_f * config.sound_speed / config.chirp_slope) / 2.0


def parse_range_bin(path: Path) -> Optional[int]:
    match = re.search(r"rangebin=(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def resolve_input_path(path: Path) -> Path:
    path = Path(path)
    if path.exists() or path.is_absolute():
        return path
    package_relative = Path(__file__).resolve().parent / path
    if package_relative.exists():
        return package_relative
    return path


def choose_range_bin(amplitude_by_bin: np.ndarray, search: Tuple[int, int] = (5, 80)) -> int:
    start, stop = search
    stop = min(stop, amplitude_by_bin.shape[1])
    if start >= stop:
        raise ValueError("empty range-bin search window")
    scores = np.mean(amplitude_by_bin[:, start:stop], axis=0)
    return int(start + np.argmax(scores))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def smooth_phase_trace(values: np.ndarray, window: int = 13) -> np.ndarray:
    """Smooth a phase trace without pulling the start/end toward zero."""
    phase = np.asarray(values, dtype=np.float64)
    if phase.size <= 1 or window <= 1:
        return phase.copy()
    window = min(int(window), phase.size)
    if window % 2 == 0:
        window = max(1, window - 1)
    if window <= 1:
        return phase.copy()
    pad = window // 2
    padded = np.pad(phase, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def robust_mad(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return 1.4826 * mad


def intervals_from_mask(mask: np.ndarray, chirp_duration: float) -> Tuple[BreathInterval, ...]:
    intervals: List[BreathInterval] = []
    start: Optional[int] = None
    for i, active in enumerate(mask):
        if active and start is None:
            start = i
        elif not active and start is not None:
            end = i
            intervals.append(
                BreathInterval(
                    start_s=start * chirp_duration,
                    end_s=end * chirp_duration,
                    duration_s=(end - start) * chirp_duration,
                )
            )
            start = None
    if start is not None:
        end = mask.size
        intervals.append(
            BreathInterval(
                start_s=start * chirp_duration,
                end_s=end * chirp_duration,
                duration_s=(end - start) * chirp_duration,
            )
        )
    return tuple(intervals)


def detect_breathing(
    unwrap_phase: np.ndarray,
    config: FmcwConfig = FmcwConfig(),
    smooth_window: int = 7,
) -> BreathDetection:
    smoothed = moving_average(np.asarray(unwrap_phase, dtype=np.float64), smooth_window)
    phase_ptp = float(np.ptp(smoothed))
    motion = np.abs(np.diff(smoothed, prepend=smoothed[0]))
    threshold = max(0.03, float(np.median(motion) + 2.5 * robust_mad(motion)))
    active = moving_average((motion > threshold).astype(float), 5) >= 0.4
    intervals = tuple(
        interval
        for interval in intervals_from_mask(active, config.chirp_duration)
        if interval.duration_s >= 0.15
    )
    active_fraction = float(np.mean(active))
    breathing_present = phase_ptp > 0.8 and (active_fraction > 0.05 or len(intervals) > 0)
    return BreathDetection(
        breathing_present=breathing_present,
        phase_peak_to_peak_rad=phase_ptp,
        motion_threshold=threshold,
        active_fraction=active_fraction,
        intervals=intervals,
    )


def process_arrays(
    tx: np.ndarray,
    rx: np.ndarray,
    config: FmcwConfig = FmcwConfig(),
    range_bin: Optional[int] = None,
    source_path: Optional[Path] = None,
    drop_seconds: float = 0.0,
) -> FmcwResult:
    tx_segments, rx_segments = segment_chirps(tx, rx, config, drop_seconds=drop_seconds)
    mixed_lowpass = mix_and_lowpass(tx_segments, rx_segments, config)
    mixed_subtracted = background_subtract(mixed_lowpass)
    complex_by_bin = np.fft.rfft(mixed_subtracted, axis=1)
    amplitude_by_bin = np.abs(complex_by_bin)
    phase_by_bin = np.angle(complex_by_bin)

    if range_bin is None:
        range_bin = choose_range_bin(amplitude_by_bin)
    if not 0 <= range_bin < phase_by_bin.shape[1]:
        raise ValueError(f"range_bin {range_bin} outside 0..{phase_by_bin.shape[1] - 1}")

    unwrap_phase = np.unwrap(phase_by_bin[:, range_bin])
    peak_bins = np.argmax(amplitude_by_bin, axis=1)
    detection = detect_breathing(unwrap_phase, config)

    return FmcwResult(
        source_path=source_path,
        config=config,
        range_bin=int(range_bin),
        tx_segments=tx_segments,
        rx_segments=rx_segments,
        mixed_lowpass=mixed_lowpass,
        mixed_subtracted=mixed_subtracted,
        complex_by_bin=complex_by_bin,
        amplitude_by_bin=amplitude_by_bin,
        phase_by_bin=phase_by_bin,
        unwrap_phase=unwrap_phase,
        peak_bins=peak_bins,
        detection=detection,
    )


def process_npz(
    path: Path,
    config: FmcwConfig = FmcwConfig(),
    range_bin: Optional[int] = None,
    drop_seconds: float = 0.0,
) -> FmcwResult:
    path = resolve_input_path(path)
    data = np.load(path)
    if "tx" not in data or "rx" not in data:
        raise ValueError(f"{path} must contain 'tx' and 'rx' arrays")
    if range_bin is None:
        range_bin = parse_range_bin(path)
    return process_arrays(
        data["tx"],
        data["rx"],
        config=config,
        range_bin=range_bin,
        source_path=path,
        drop_seconds=drop_seconds,
    )


def plot_breath_monitoring(result: FmcwResult, output_path: Path, phase_sign: float = 1.0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4.8))
    plt.plot(result.timestamps, phase_sign * result.unwrap_phase, linewidth=1.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Unwrapped phase (rad)")
    plt.title(f"Breath Monitoring, range_bin={result.range_bin}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_clean_breath_monitoring(
    result: FmcwResult,
    output_path: Path,
    smooth_window: int = 13,
    phase_sign: float = 1.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = phase_sign * (result.unwrap_phase - np.median(result.unwrap_phase[: min(10, result.unwrap_phase.size)]))
    smoothed = smooth_phase_trace(raw, smooth_window)
    plt.figure(figsize=(10, 4.8))
    plt.plot(result.timestamps, raw, color="#8bb8dd", linewidth=1.0, alpha=0.55, label="raw phase")
    plt.plot(
        result.timestamps,
        smoothed,
        color="#1f77b4",
        linewidth=2.6,
        label=f"{smooth_window * result.config.chirp_duration:.2f} s moving average",
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Centered unwrapped phase (rad)")
    plt.title(f"Clean Breath Monitoring, range_bin={result.range_bin}")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_range_spectrogram(result: FmcwResult, output_path: Path, max_bin: int = 80) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    max_bin = min(max_bin, result.amplitude_by_bin.shape[1])
    distances = idx_to_distance(np.arange(max_bin), result.config)
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(
        result.timestamps,
        distances,
        result.amplitude_by_bin[:, :max_bin].T,
        shading="auto",
    )
    plt.axhline(idx_to_distance(result.range_bin, result.config), color="white", linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("Background-subtracted FMCW range spectrum")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_range_bin_search(
    result: FmcwResult,
    output_path: Path,
    bins: Optional[Iterable[int]] = None,
    phase_sign: float = 1.0,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if bins is None:
        start = max(0, result.range_bin - 5)
        stop = min(result.phase_by_bin.shape[1], result.range_bin + 6)
        bins = range(start, stop)
    plt.figure(figsize=(10, 5.5))
    for bin_index in bins:
        phase = phase_sign * np.unwrap(result.phase_by_bin[:, bin_index])
        phase = phase - np.median(phase)
        linewidth = 2.2 if bin_index == result.range_bin else 0.9
        alpha = 1.0 if bin_index == result.range_bin else 0.42
        plt.plot(result.timestamps, phase, label=str(bin_index), linewidth=linewidth, alpha=alpha)
    plt.xlabel("Time (s)")
    plt.ylabel("Centered unwrapped phase (rad)")
    plt.title("Breath Monitoring range-bin search")
    plt.grid(True, alpha=0.2)
    plt.legend(title="range bin", ncol=4, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def result_summary(result: FmcwResult) -> dict:
    detection = result.detection
    return {
        "source": str(result.source_path) if result.source_path else None,
        "num_chirps": int(result.unwrap_phase.shape[0]),
        "range_bin": result.range_bin,
        "range_bin_distance_m": float(idx_to_distance(result.range_bin, result.config)),
        "breathing_present": detection.breathing_present,
        "phase_peak_to_peak_rad": detection.phase_peak_to_peak_rad,
        "active_fraction": detection.active_fraction,
        "motion_threshold": detection.motion_threshold,
        "active_intervals": [asdict(interval) for interval in detection.intervals],
    }


def save_outputs(
    result: FmcwResult,
    output_dir: Path,
    prefix: str,
    phase_sign: float = 1.0,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    breath_path = output_dir / f"{prefix}_breath_rangebin{result.range_bin}.png"
    clean_breath_path = output_dir / f"{prefix}_breath_rangebin{result.range_bin}_clean.png"
    spectrum_path = output_dir / f"{prefix}_range_spectrum.png"
    search_path = output_dir / f"{prefix}_rangebin_search.png"
    summary_path = output_dir / f"{prefix}_summary.json"

    plot_breath_monitoring(result, breath_path, phase_sign=phase_sign)
    plot_clean_breath_monitoring(result, clean_breath_path, phase_sign=phase_sign)
    plot_range_spectrogram(result, spectrum_path)
    plot_range_bin_search(result, search_path, phase_sign=phase_sign)
    summary_path.write_text(json.dumps(result_summary(result), indent=2), encoding="utf-8")
    return [breath_path, clean_breath_path, spectrum_path, search_path, summary_path]


def default_inputs() -> List[Path]:
    return sorted(DEFAULT_DATA_DIR.glob("breathing_*_rangebin=*.npz"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process FMCW acoustic breathing data and save lab plots."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input .npz files containing tx and rx. Defaults to refer/FMCW/breathing_*.npz.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for PNG plots and JSON summaries.",
    )
    parser.add_argument(
        "--range-bin",
        type=int,
        default=None,
        help="Override range bin. By default it is parsed from the filename or auto-selected.",
    )
    parser.add_argument("--drop-seconds", type=float, default=0.0)
    parser.add_argument(
        "--fmcw-emission",
        choices=FMCW_EMISSIONS,
        default="linear",
        help="Transmit waveform variant used for generated live/replay TX metadata.",
    )
    parser.add_argument(
        "--invert-phase",
        action="store_true",
        help="Invert displayed phase polarity. Useful when inhale/exhale direction is reversed.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    inputs = args.inputs or default_inputs()
    if not inputs:
        raise SystemExit(f"No input files found under {DEFAULT_DATA_DIR}")

    for input_path in inputs:
        config = FmcwConfig(emission=args.fmcw_emission)
        result = process_npz(
            input_path,
            config=config,
            range_bin=args.range_bin,
            drop_seconds=args.drop_seconds,
        )
        prefix = Path(input_path).stem.replace("=", "_")
        phase_sign = -1.0 if args.invert_phase else 1.0
        written = save_outputs(result, args.output_dir, prefix, phase_sign=phase_sign)
        summary = result_summary(result)
        print(json.dumps(summary, indent=2))
        print("wrote:")
        for path in written:
            print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
