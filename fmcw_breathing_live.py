"""Live FMCW acoustic breathing monitor.

Run from this directory on Windows:

    python fmcw_breathing_live.py --duration 10 --range-bin 15

The script plays a repeated 17-23 kHz FMCW chirp, records the microphone at the
same time, saves the live recording as .npz, then reuses fmcw_breathing.py to
produce the same breathing plots as the offline assignment script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    from . import fmcw_breathing as fmcw
except ImportError:  # pragma: no cover - used when run as a script from this folder
    import fmcw_breathing as fmcw


DEFAULT_LIVE_DIR = Path(__file__).resolve().parent / "refer" / "FMCW" / "live"


@dataclass
class LiveRecording:
    tx: np.ndarray
    rx: np.ndarray


def enable_interactive_matplotlib_backend() -> None:
    backend = matplotlib.get_backend().lower()
    if "agg" not in backend:
        return
    for candidate in ("TkAgg", "QtAgg", "Qt5Agg"):
        try:
            matplotlib.use(candidate, force=True)
            return
        except Exception:
            continue


class LivePhasePlotter:
    def __init__(self, smooth_window: int = 13):
        enable_interactive_matplotlib_backend()
        plt.ion()
        self.smooth_window = smooth_window
        self.figure, self.axis = plt.subplots(figsize=(10, 4.8))
        (self.raw_line,) = self.axis.plot([], [], color="#8bb8dd", linewidth=1.0, alpha=0.55, label="raw phase")
        (self.smooth_line,) = self.axis.plot([], [], color="#1f77b4", linewidth=2.5, label="smoothed phase")
        self.axis.set_xlabel("Time (s)")
        self.axis.set_ylabel("Centered unwrapped phase (rad)")
        self.axis.grid(True, alpha=0.25)
        self.axis.legend(loc="best")
        self.figure.tight_layout()
        self.figure.show()

    def update(self, result: fmcw.FmcwResult) -> None:
        phase = result.unwrap_phase
        center_count = min(10, phase.size)
        centered = phase - np.median(phase[:center_count])
        smoothed = fmcw.smooth_phase_trace(centered, self.smooth_window)
        self.raw_line.set_data(result.timestamps, centered)
        self.smooth_line.set_data(result.timestamps, smoothed)
        self.axis.set_title(f"Live Breath Monitoring, range_bin={result.range_bin}")
        self.axis.relim()
        self.axis.autoscale_view()
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)

    def close(self) -> None:
        plt.ioff()


def sounddevice_device_arg(
    input_device: Optional[int],
    output_device: Optional[int],
) -> Optional[Tuple[Optional[int], Optional[int]]]:
    if input_device is None and output_device is None:
        return None
    return (input_device, output_device)


def import_sounddevice():
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError(
            "sounddevice is required for live FMCW recording. Install it with "
            "`pip install sounddevice` if it is missing."
        ) from exc
    return sd


def safe_console_text(text: object, encoding: Optional[str] = None) -> str:
    encoding = encoding or getattr(sys.stdout, "encoding", None) or "utf-8"
    return str(text).encode(encoding, errors="replace").decode(encoding, errors="replace")


def list_audio_devices(sd_module=None) -> None:
    sd = sd_module or import_sounddevice()
    print(safe_console_text(f"Default device [input, output]: {sd.default.device}"))
    print()
    print(safe_console_text(sd.query_devices()))


def print_start_signal() -> None:
    print("可以开始了：FMCW 即将播放并录音，请保持姿势稳定并开始呼吸动作。", flush=True)


def record_live_chirp(
    config: fmcw.FmcwConfig,
    amplitude: float = 0.2,
    input_device: Optional[int] = None,
    output_device: Optional[int] = None,
    sd_module=None,
) -> LiveRecording:
    sd = sd_module or import_sounddevice()
    tx = amplitude * fmcw.generate_chirp_train(config)
    device = sounddevice_device_arg(input_device, output_device)

    sd.default.samplerate = config.sample_rate
    sd.default.channels = 1
    rx = sd.playrec(
        tx.astype(np.float32),
        samplerate=config.sample_rate,
        channels=1,
        device=device,
        blocking=True,
    )
    wait = getattr(sd, "wait", None)
    if callable(wait):
        wait()
    return LiveRecording(tx=tx.astype(np.float32), rx=np.asarray(rx, dtype=np.float32))


def stream_live_chirp(
    config: fmcw.FmcwConfig,
    amplitude: float = 0.2,
    input_device: Optional[int] = None,
    output_device: Optional[int] = None,
    blocksize: Optional[int] = None,
    range_bin: Optional[int] = None,
    live_plotter=None,
    plot_interval_s: float = 0.1,
    drop_seconds: float = 0.0,
    sd_module=None,
) -> LiveRecording:
    sd = sd_module or import_sounddevice()
    blocksize = blocksize or config.samples_per_chirp
    tx = amplitude * fmcw.generate_chirp_train(config)
    rx_chunks: List[np.ndarray] = []
    tx_chunks: List[np.ndarray] = []
    playback_index = 0
    total_samples = tx.shape[0]
    device = sounddevice_device_arg(input_device, output_device)

    def callback(indata, outdata, frames, callback_time, status):
        nonlocal playback_index
        end = min(playback_index + frames, total_samples)
        chunk = np.zeros(frames, dtype=np.float32)
        available = end - playback_index
        if available > 0:
            chunk[:available] = tx[playback_index:end]
        outdata[:, 0] = chunk
        rx_chunks.append(np.asarray(indata[:, 0], dtype=np.float32).copy())
        tx_chunks.append(chunk.copy())
        playback_index += frames

    stream = sd.Stream(
        samplerate=config.sample_rate,
        blocksize=blocksize,
        channels=1,
        dtype="float32",
        callback=callback,
        device=device,
    )
    last_plot_update = 0.0
    with stream:
        deadline = time.monotonic() + config.total_duration + 1.0
        while playback_index < total_samples and time.monotonic() < deadline:
            now = time.monotonic()
            if live_plotter is not None and now - last_plot_update >= plot_interval_s:
                update_live_phase_plotter(
                    live_plotter,
                    tx_chunks,
                    rx_chunks,
                    config,
                    range_bin=range_bin,
                    drop_seconds=drop_seconds,
                )
                last_plot_update = now
            time.sleep(0.01)

    if not rx_chunks:
        raise RuntimeError("no audio was captured from the input stream")
    if live_plotter is not None:
        update_live_phase_plotter(
            live_plotter,
            tx_chunks,
            rx_chunks,
            config,
            range_bin=range_bin,
            drop_seconds=drop_seconds,
        )
    recorded_rx = np.concatenate(rx_chunks).reshape(-1, 1)[:total_samples]
    recorded_tx = np.concatenate(tx_chunks)[:total_samples]
    if recorded_rx.shape[0] < total_samples:
        recorded_rx = np.pad(recorded_rx[:, 0], (0, total_samples - recorded_rx.shape[0])).reshape(-1, 1)
    if recorded_tx.shape[0] < total_samples:
        recorded_tx = np.pad(recorded_tx, (0, total_samples - recorded_tx.shape[0]))
    return LiveRecording(tx=recorded_tx.astype(np.float32), rx=recorded_rx.astype(np.float32))


def update_live_phase_plotter(
    live_plotter,
    tx_chunks: Sequence[np.ndarray],
    rx_chunks: Sequence[np.ndarray],
    config: fmcw.FmcwConfig,
    range_bin: Optional[int] = None,
    drop_seconds: float = 0.0,
) -> bool:
    if not tx_chunks or not rx_chunks:
        return False
    samples_per_chirp = config.samples_per_chirp
    tx = np.concatenate(list(tx_chunks)).astype(np.float32)
    rx = np.concatenate(list(rx_chunks)).astype(np.float32).reshape(-1, 1)
    usable = min(tx.shape[0], rx.shape[0])
    num_chirps = usable // samples_per_chirp
    drop_chirps = int(round(drop_seconds / config.chirp_duration))
    if num_chirps <= max(drop_chirps, 1):
        return False
    usable = num_chirps * samples_per_chirp
    try:
        result = fmcw.process_arrays(
            tx[:usable],
            rx[:usable],
            config=config,
            range_bin=range_bin,
            drop_seconds=drop_seconds,
        )
    except ValueError:
        return False
    live_plotter.update(result)
    return True


def save_live_recording_npz(
    recording: LiveRecording,
    output_dir: Path = DEFAULT_LIVE_DIR,
    stem: Optional[str] = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if stem is None:
        stem = "live_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{stem}.npz"
    np.savez(path, tx=recording.tx, rx=recording.rx)
    return path


def can_show_interactive_plots() -> bool:
    return "agg" not in matplotlib.get_backend().lower()


def show_result_plots(result: fmcw.FmcwResult, phase_sign: float = 1.0) -> None:
    plt.figure(figsize=(10, 4.8))
    plt.plot(result.timestamps, phase_sign * result.unwrap_phase, linewidth=1.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Unwrapped phase (rad)")
    plt.title(f"Breath Monitoring, range_bin={result.range_bin}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    max_bin = min(80, result.amplitude_by_bin.shape[1])
    distances = fmcw.idx_to_distance(np.arange(max_bin), result.config)
    plt.figure(figsize=(10, 5))
    plt.pcolormesh(
        result.timestamps,
        distances,
        result.amplitude_by_bin[:, :max_bin].T,
        shading="auto",
    )
    plt.axhline(fmcw.idx_to_distance(result.range_bin, result.config), color="white", linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("Background-subtracted FMCW range spectrum")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()


def build_config(args: argparse.Namespace) -> fmcw.FmcwConfig:
    return fmcw.FmcwConfig(
        sample_rate=args.sample_rate,
        freq_low=args.freq_low,
        freq_high=args.freq_high,
        chirp_duration=args.chirp_duration,
        total_duration=args.duration,
        lowpass_cutoff=args.lowpass_cutoff,
        sound_speed=args.sound_speed,
        emission=args.fmcw_emission,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record live FMCW breathing data and plot it.")
    parser.add_argument("--list-devices", action="store_true", help="Print sounddevice devices and exit.")
    parser.add_argument("--duration", type=float, default=10.0, help="Recording duration in seconds.")
    parser.add_argument("--range-bin", type=int, default=None, help="Range bin to plot, e.g. 15 near 40 cm.")
    parser.add_argument("--amplitude", type=float, default=0.2, help="Playback amplitude, 0.0 to 1.0.")
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--freq-low", type=float, default=17_000.0)
    parser.add_argument("--freq-high", type=float, default=23_000.0)
    parser.add_argument("--chirp-duration", type=float, default=0.05)
    parser.add_argument("--lowpass-cutoff", type=float, default=5_000.0)
    parser.add_argument("--sound-speed", type=float, default=343.0)
    parser.add_argument(
        "--fmcw-emission",
        choices=fmcw.FMCW_EMISSIONS,
        default="linear",
        help="Transmit waveform variant: linear, linear_tukey, cw_fmcw_hybrid, or triangle.",
    )
    parser.add_argument("--drop-seconds", type=float, default=0.0)
    parser.add_argument(
        "--blocksize",
        type=int,
        default=None,
        help="Audio callback block size. Defaults to one chirp.",
    )
    parser.add_argument(
        "--playrec",
        action="store_true",
        help="Use sounddevice.playrec instead of the callback stream.",
    )
    parser.add_argument(
        "--live-plot",
        action="store_true",
        help="Show the selected range-bin phase curve while callback recording is running.",
    )
    parser.add_argument(
        "--live-plot-interval",
        type=float,
        default=0.1,
        help="Seconds between live plot refreshes.",
    )
    parser.add_argument("--recording-dir", type=Path, default=DEFAULT_LIVE_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=fmcw.DEFAULT_OUTPUT_DIR,
        help="Directory for output plots and summary JSON.",
    )
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening a window.")
    parser.add_argument(
        "--invert-phase",
        action="store_true",
        help="Invert displayed phase polarity. Useful when inhale/exhale direction is reversed.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.list_devices:
        list_audio_devices()
        return 0

    if not 0.0 < args.amplitude <= 1.0:
        raise SystemExit("--amplitude must be in the range (0.0, 1.0]")

    config = build_config(args)
    print("Recording live FMCW chirps...")
    print(
        json.dumps(
            {
                "duration_s": config.total_duration,
                "sample_rate": config.sample_rate,
                "freq_low": config.freq_low,
                "freq_high": config.freq_high,
                "chirp_duration": config.chirp_duration,
                "fmcw_emission": config.emission,
                "range_bin": args.range_bin,
                "amplitude": args.amplitude,
            },
            indent=2,
        )
    )
    print_start_signal()

    if args.playrec and args.live_plot:
        raise SystemExit("--live-plot requires the callback stream; remove --playrec.")
    live_plotter = LivePhasePlotter() if args.live_plot else None
    try:
        if args.playrec:
            recording = record_live_chirp(
                config,
                amplitude=args.amplitude,
                input_device=args.input_device,
                output_device=args.output_device,
            )
        else:
            recording = stream_live_chirp(
                config,
                amplitude=args.amplitude,
                input_device=args.input_device,
                output_device=args.output_device,
                blocksize=args.blocksize,
                range_bin=args.range_bin,
                live_plotter=live_plotter,
                plot_interval_s=args.live_plot_interval,
                drop_seconds=args.drop_seconds,
            )
    except Exception as exc:
        raise SystemExit(
            f"Live recording failed: {exc}\n"
            "Try `python fmcw_breathing_live.py --list-devices`, then pass "
            "`--input-device N --output-device M`."
        ) from exc

    recording_path = save_live_recording_npz(recording, args.recording_dir)
    result = fmcw.process_arrays(
        recording.tx,
        recording.rx,
        config=config,
        range_bin=args.range_bin,
        source_path=recording_path,
        drop_seconds=args.drop_seconds,
    )
    prefix = recording_path.stem + f"_rangebin_{result.range_bin}"
    phase_sign = -1.0 if args.invert_phase else 1.0
    written = fmcw.save_outputs(result, args.output_dir, prefix, phase_sign=phase_sign)

    print(json.dumps(fmcw.result_summary(result), indent=2))
    print("recording:")
    print(f"  {recording_path}")
    print("wrote:")
    for path in written:
        print(f"  {path}")

    if not args.no_show:
        if can_show_interactive_plots():
            show_result_plots(result, phase_sign=phase_sign)
        else:
            print("Plot window skipped: current Matplotlib backend is non-interactive. Use --no-show to suppress this message.")
    if live_plotter is not None:
        live_plotter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
