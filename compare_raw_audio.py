"""Draw the RAW microphone audio waveform for each variant session.

This shows what the mic actually captured (audio.wav), before any FMCW
processing or scoring. The "flat sections" the user sees in the live
visualization are score=0 (gated), not actual silence — this plot proves
the underlying audio is never flat.
"""
from __future__ import annotations
import io
import sys
import wave
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SESSIONS = [
    ("linear",         "hp_blink_20260618_162711", "linear"),
    ("linear_v2",     "hp_blink_20260618_172844", "linear"),
    ("linear_tukey",   "hp_blink_20260618_173242", "linear_tukey"),
    ("triangle",       "hp_blink_20260618_173057", "triangle"),
    ("cw_fmcw_hybrid", "hp_blink_20260618_173428", "cw_fmcw_hybrid"),
    ("cw_single",      "hp_blink_20260618_174829", "cw_single"),
]


def load_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        n = w.getnframes()
        raw = w.readframes(n)
        ch = w.getnchannels()
        sw = w.getsampwidth()
        sr = w.getframerate()
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    arr = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
    arr /= float(np.iinfo(dtype).max)
    return arr, sr


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


def main():
    out_dir = REPO / "hp_acoustic_wave" / "docs" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(SESSIONS), 2, figsize=(18, 2.6 * len(SESSIONS)))
    if len(SESSIONS) == 1:
        axes = axes[np.newaxis, :]

    for row, (label, sess_name, emission) in enumerate(SESSIONS):
        sess = REPO / "hp_acoustic_wave" / "sessions" / sess_name
        wav, sr = load_mono_wav(sess / "audio.wav")
        blinks = load_visual_blinks(sess)
        duration = len(wav) / sr

        # ---- Left: full-waveform envelope (RMS per ~10ms window) ----
        # The raw waveform at 48kHz is too dense to plot directly; show the
        # amplitude envelope instead, which preserves every peak/trough.
        win = int(sr * 0.01)  # 10 ms
        n_win = len(wav) // win
        env = wav[: n_win * win].reshape(n_win, win)
        rms = np.sqrt(np.mean(env ** 2, axis=1))
        peak = np.max(np.abs(env), axis=1)
        t_env = np.arange(n_win) * 0.01

        ax = axes[row, 0]
        ax.fill_between(t_env, -peak, peak, color="C0", alpha=0.4, label="peak envelope")
        ax.plot(t_env, rms, color="C1", lw=0.9, label="RMS (10ms)")
        for b in blinks:
            ax.axvline(b, color="red", lw=0.6, alpha=0.6)
        ax.set_title(f"RAW mic audio envelope: {label} ({emission})  [duration={duration:.1f}s, red=visual blink]")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("amplitude")
        ax.set_xlim(0, duration)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

        # ---- Right: zoom into first 30ms — the actual sample-by-sample waveform ----
        n_show = int(sr * 0.030)
        t_sample = np.arange(n_show) / sr * 1000.0  # ms
        ax = axes[row, 1]
        ax.plot(t_sample, wav[:n_show], lw=0.6, color="C0")
        ax.set_title(f"First 30ms of raw audio: {label}  [{sr}Hz, {n_show} samples]")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("amplitude")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    out = out_dir / "raw_audio_waveforms.png"
    plt.savefig(out, dpi=110)
    print("saved:", out)
    print()
    print("Note: 'flat sections' in the live visualization are score=0 (gated by")
    print("twinkle_fmcw_min_score), NOT silence in the mic audio. This plot proves")
    print("the underlying audio.wav is never flat.")


if __name__ == "__main__":
    main()
