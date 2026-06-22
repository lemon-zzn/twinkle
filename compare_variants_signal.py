"""Compare the 5 emission variants at the signal level.

Loads each session's audio.wav, reconstructs the TX chirp for that variant, and
plots:
  1. TX spectrogram (verifies the chirp shape was emitted)
  2. RX range-bin magnitude and phase over time, with visual blinks overlaid
  3. Per-variant coherence score vs time around one known blink

Output: docs/variant_comparison_<timestamp>.png
"""
from __future__ import annotations
import io
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hp_acoustic_wave.dsp import generate_fmcw_chirp, generate_tone, extract_fmcw_chunk_feature, extract_chunk_feature, FmcwBackgroundSubtractor

SESSIONS = [
    ("linear",         "hp_blink_20260618_162711", "linear"),
    ("linear_v2",     "hp_blink_20260618_172844", "linear"),
    ("linear_tukey",   "hp_blink_20260618_173242", "linear_tukey"),
    ("triangle",       "hp_blink_20260618_173057", "triangle"),
    ("cw_fmcw_hybrid", "hp_blink_20260618_173428", "cw_fmcw_hybrid"),
    ("cw_single",      "hp_blink_20260618_174829", "cw_single"),
]

SAMPLE_RATE = 48_000
FREQ_LOW = 17_000
FREQ_HIGH = 23_000
CHIRP_DURATION = 0.05
RANGE_BIN = 15
TONE_HZ = 18_500
CHUNK = 2400  # one chirp worth of samples at 48k


def load_mono_wav(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        ch = w.getnchannels()
        sw = w.getsampwidth()
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
    arr /= float(np.iinfo(dtype).max)
    return arr


def load_visual_blinks(session: Path) -> list:
    p = session / "visual_labels.csv"
    if not p.exists():
        return []
    out = []
    with io.open(p, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",")
            if len(parts) >= 8 and parts[7] == "1":
                out.append(float(parts[0]))
    return out


def tx_for(emission: str) -> np.ndarray:
    if emission == "cw_single":
        return generate_tone(CHUNK * 4, SAMPLE_RATE, TONE_HZ, 0, 0.2)
    return generate_fmcw_chirp(
        CHUNK * 4, SAMPLE_RATE, FREQ_LOW, FREQ_HIGH, CHIRP_DURATION,
        start_sample=0, amplitude=0.2, emission=emission, tukey_alpha=0.2,
    )


def main():
    out_dir = REPO / "hp_acoustic_wave" / "docs" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(SESSIONS), 3, figsize=(18, 4 * len(SESSIONS)))
    if len(SESSIONS) == 1:
        axes = axes[np.newaxis, :]

    for row, (label, sess_name, emission) in enumerate(SESSIONS):
        sess = REPO / "hp_acoustic_wave" / "sessions" / sess_name
        wav = load_mono_wav(sess / "audio.wav")
        blinks = load_visual_blinks(sess)

        # TX spectrogram
        tx = tx_for(emission)
        nfft = 4096
        sp = np.fft.rfft(tx[:nfft] * np.hanning(nfft))
        freqs = np.fft.rfftfreq(nfft, 1.0 / SAMPLE_RATE)
        ax = axes[row, 0]
        ax.plot(freqs / 1000.0, 20 * np.log10(np.abs(sp) + 1e-9))
        ax.set_xlim(14, 26)
        ax.set_title(f"TX spectrum: {label} ({emission})")
        ax.set_xlabel("kHz")
        ax.set_ylabel("dB")
        ax.grid(alpha=0.3)

        # Range-bin magnitude / phase over time
        n_frames = min(len(wav) // CHUNK, 800)
        times = []
        amps = []
        phases = []
        prev = None
        bg = FmcwBackgroundSubtractor(alpha=0.02) if emission != "cw_single" else None
        for i in range(n_frames):
            s = i * CHUNK
            chunk = wav[s:s + CHUNK]
            if emission == "cw_single":
                f = extract_chunk_feature(chunk, SAMPLE_RATE, TONE_HZ, s, prev)
                times.append(f.time_s)
                amps.append(f.amplitude)
                phases.append(f.phase)
            else:
                tx_chunk = tx_for(emission)[s:s + CHUNK] if False else generate_fmcw_chirp(
                    CHUNK, SAMPLE_RATE, FREQ_LOW, FREQ_HIGH, CHIRP_DURATION,
                    start_sample=s, amplitude=0.2, emission=emission, tukey_alpha=0.2,
                )
                f = extract_fmcw_chunk_feature(
                    chunk, tx_chunk, SAMPLE_RATE, FREQ_LOW, FREQ_HIGH,
                    CHIRP_DURATION, RANGE_BIN, s, prev, emission=emission,
                    background_subtractor=bg,
                )
                times.append(f.time_s)
                amps.append(f.amplitude)
                phases.append(f.phase)
            prev = f
        times = np.asarray(times)
        amps = np.asarray(amps)
        phases = np.unwrap(phases)

        ax = axes[row, 1]
        ax.plot(times, amps, lw=0.8, label="range-bin magnitude")
        ax2 = ax.twinx()
        ax2.plot(times, phases, lw=0.6, color="C1", alpha=0.6, label="unwrapped phase")
        for b in blinks:
            ax.axvline(b, color="red", lw=0.5, alpha=0.5)
        ax.set_title(f"Range-bin {RANGE_BIN}: {label} (red=visual blink)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("magnitude")
        ax2.set_ylabel("phase (rad)")
        ax.grid(alpha=0.3)

        # Phase delta histogram (blink vs non-blink)
        dphase = np.diff(phases)
        blink_mask = np.zeros(len(dphase), dtype=bool)
        for b in blinks:
            idx = np.searchsorted(times, b)
            if 0 <= idx < len(dphase):
                blink_mask[max(0, idx - 2):idx + 3] = True
        ax = axes[row, 2]
        bins = np.linspace(-np.pi, np.pi, 41)
        ax.hist(dphase[~blink_mask], bins=bins, alpha=0.5, density=True, label="non-blink")
        ax.hist(dphase[blink_mask], bins=bins, alpha=0.6, density=True, label="blink window")
        ax.set_title(f"phase-step distribution: {label}")
        ax.set_xlabel("phase step (rad)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = out_dir / "variant_comparison.png"
    plt.savefig(out, dpi=110)
    print("saved:", out)


if __name__ == "__main__":
    main()
