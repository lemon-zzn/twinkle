"""Diagnose periodic firing root cause.

Read-only investigation: does the periodic firing of `twinkle` on 174829
(cw_single tone) and `blinklistener` on 214655 (FMCW linear) come from
structure in the raw data (audio.wav / FMCW TX artifact) or is it
introduced by the detector DSP?

Approach:
  1. For each problem session + a control, load `features.csv`
     (`phase_pair_delta` for FMCW, `phase` for tone), restrict to quiet
     chunks (no visual blink within +/-0.5s), and FFT the trajectory.
     Peaks at ~1Hz indicate refractory coupling; ~20Hz indicate chunk/TX
     artifact; anything else is detector-internal.
  2. For 214655, also load `audio.wav` and compute the FMCW baseband
     demodulated signal at range-bin 15 directly from raw audio. FFT that
     to see whether periodicity already exists pre-detector.
  3. Cross-variant comparison: 6 baseline-doc sessions, FFT each.

Outputs (under hp_acoustic_wave/docs/diagnostics/):
  - periodic_rootcause_174829.png
  - periodic_rootcause_214655.png
  - periodic_rootcause_variant_comparison.png
"""
from __future__ import annotations
import csv
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SESSIONS_DIR = REPO / "hp_acoustic_wave" / "sessions"
OUT_DIR = REPO / "hp_acoustic_wave" / "docs" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cross-variant comparison sessions (from baseline doc)
BASELINE_VARIANTS = [
    ("linear",         "hp_blink_20260618_162711", "linear"),
    ("linear_v2",      "hp_blink_20260618_172844", "linear"),
    ("linear_tukey",   "hp_blink_20260618_173242", "linear_tukey"),
    ("triangle",       "hp_blink_20260618_173057", "triangle"),
    ("cw_fmcw_hybrid", "hp_blink_20260618_173428", "cw_fmcw_hybrid"),
    ("cw_single",      "hp_blink_20260618_174829", "cw_single"),
]

# 20 Hz chunk rate (chirp_duration=0.05s).
CHUNK_RATE = 20.0
CHUNK_DT = 1.0 / CHUNK_RATE
# Margin around visual blinks considered "not quiet"
QUIET_MARGIN_S = 0.5


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------

def load_visual_blink_times(session: Path):
    p = session / "visual_labels.csv"
    if not p.exists():
        return []
    out = []
    with io.open(p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_blink_event") == "1":
                try:
                    out.append(float(row["time_s"]))
                except (KeyError, ValueError):
                    pass
    return np.asarray(out)


def load_feature_column(session: Path, column: str):
    """Return (times, values) for `column` from features.csv."""
    p = session / "features.csv"
    if not p.exists():
        return np.asarray([]), np.asarray([])
    times = []
    vals = []
    with io.open(p, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if column not in reader.fieldnames:
            return np.asarray([]), np.asarray([])
        for row in reader:
            try:
                t = float(row["time_s"])
                v = float(row[column])
            except (KeyError, ValueError):
                continue
            times.append(t)
            vals.append(v)
    return np.asarray(times), np.asarray(vals)


def load_signal_mode(session: Path):
    p = session / "metadata.json"
    if not p.exists():
        return None
    with io.open(p, encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("config", {}).get("audio", {}).get("signal_mode"), \
           meta.get("config", {}).get("audio", {}).get("fmcw_emission")


def load_mono_wav(path: Path):
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


# --------------------------------------------------------------------------
# FFT helpers
# --------------------------------------------------------------------------

def fft_traj(values, dt):
    """Return (freqs, mag_db) of one-sided power spectrum, dB scale."""
    x = np.asarray(values, dtype=np.float64)
    # remove mean
    x = x - x.mean()
    # Hann window
    w = np.hanning(len(x))
    win = x * w
    sp = np.fft.rfft(win)
    mag = np.abs(sp) * 2.0 / max(np.sum(w), 1e-9)
    freqs = np.fft.rfftfreq(len(win), dt)
    mag_db = 20.0 * np.log10(mag + 1e-9)
    return freqs, mag_db


def quiet_mask(times, blink_times, margin=QUIET_MARGIN_S):
    """Boolean mask of chunks further than `margin` from any blink."""
    if len(blink_times) == 0:
        return np.ones(len(times), dtype=bool)
    mask = np.ones(len(times), dtype=bool)
    for b in blink_times:
        mask &= np.abs(times - b) > margin
    return mask


def list_top_peaks(freqs, mag_db, top_n=6, min_sep_hz=0.3):
    """Return list of (freq_hz, mag_db) sorted by mag, separated by >=min_sep_hz."""
    # exclude DC bucket
    out = []
    mag = mag_db.copy()
    freqs = freqs.copy()
    # only consider >0.2Hz and < Nyquist
    valid = (freqs > 0.2) & (freqs < (1.0 / (2 * CHUNK_DT)) * 0.95)
    cand = np.where(valid)[0]
    # iterative peak pick
    used = []
    order = cand[np.argsort(-mag[cand])]
    for idx in order:
        f = freqs[idx]
        if any(abs(f - fu) < min_sep_hz for fu in used):
            continue
        used.append(f)
        out.append((f, mag[idx]))
        if len(out) >= top_n:
            break
    return out


# --------------------------------------------------------------------------
# Q1: features.csv FFT analysis
# --------------------------------------------------------------------------

def plot_session_phase_fft(session_name, column, tag, ax_phase, ax_fft):
    sess = SESSIONS_DIR / session_name
    times, vals = load_feature_column(sess, column)
    blinks = load_visual_blink_times(sess)
    if len(times) == 0:
        print(f"[{session_name}] no `{column}` data")
        return None, None, None
    mask = quiet_mask(times, blinks)
    quiet_times = times[mask]
    quiet_vals = vals[mask]
    # also plot full trajectory
    ax_phase.plot(times, vals, lw=0.5, alpha=0.4, color="C7", label="full")
    ax_phase.plot(quiet_times, quiet_vals, lw=0.7, color="C0", label="quiet")
    for b in blinks:
        ax_phase.axvline(b, color="red", lw=0.4, alpha=0.4)
    ax_phase.set_title(f"{tag}: {column} trajectory (red=visual blink)")
    ax_phase.set_xlabel("time (s)")
    ax_phase.set_ylabel(column)
    ax_phase.grid(alpha=0.3)
    ax_phase.legend(fontsize=8, loc="upper right")

    if len(quiet_vals) < 16:
        print(f"[{session_name}] too few quiet samples ({len(quiet_vals)})")
        return None, None, None
    freqs, mag_db = fft_traj(quiet_vals, CHUNK_DT)
    ax_fft.plot(freqs, mag_db, lw=0.8)
    ax_fft.set_xlabel("Hz")
    ax_fft.set_ylabel("magnitude (dB)")
    ax_fft.set_title(f"{tag}: FFT of quiet {column} ({len(quiet_vals)} chunks)")
    ax_fft.set_xlim(0, 10)
    ax_fft.grid(alpha=0.3)
    peaks = list_top_peaks(freqs, mag_db, top_n=6)
    # annotate refractory (1/refractory_s) and chunk rate
    for f, m in peaks[:4]:
        ax_fft.annotate(f"{f:.2f}Hz\n{m:.1f}dB", xy=(f, m),
                        xytext=(5, 5), textcoords="offset points", fontsize=7)
    return freqs, mag_db, peaks


def plot_session_174829():
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    freqs, mag, peaks = plot_session_phase_fft(
        "hp_blink_20260618_174829", "phase", "174829 cw_single (tone)",
        axes[0], axes[1])
    plt.tight_layout()
    out = OUT_DIR / "periodic_rootcause_174829.png"
    fig.savefig(out, dpi=120)
    print("saved:", out)
    return peaks


# --------------------------------------------------------------------------
# Q2: 214655 - raw audio vs post-FMCW
# --------------------------------------------------------------------------

def fmcw_baseband_from_audio(audio, sr, n_range_bin=15, chirp_dur=0.05,
                             f_lo=17000.0, f_hi=23000.0, emission="linear",
                             tukey_alpha=0.2):
    """Reproduce FMCW demod from raw audio:
    for each chirp chunk, mix RX with TX chirp, lowpass, FFT, take bin phase.
    Returns (times, phases, amps) at chunk rate ~20Hz."""
    # lazy import to avoid pulling if missing
    from hp_acoustic_wave.dsp import generate_fmcw_chirp
    chunk = int(sr * chirp_dur)
    n_frames = len(audio) // chunk
    # Reference TX (linear) — generate enough to cover session
    tx = generate_fmcw_chirp(
        chunk * n_frames, sr, f_lo, f_hi, chirp_dur,
        start_sample=0, amplitude=0.2, emission=emission, tukey_alpha=tukey_alpha,
    )
    times = np.zeros(n_frames)
    phases = np.zeros(n_frames)
    amps = np.zeros(n_frames)
    prev_phase = 0.0
    # simple lowpass at 5kHz
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, 5000.0 / (sr / 2.0), btype="low", output="sos")
    for i in range(n_frames):
        s = i * chunk
        rx = audio[s:s + chunk]
        tx_chunk = tx[s:s + chunk]
        # mix
        mixed = rx * np.conj(tx_chunk + 0j) if np.iscomplexobj(tx) else rx * tx_chunk
        # if tx is real (it is), demod needs analytic representation:
        # mix rx (real) * exp(-j*phi(t))  -> we need tx analytic
        pass
    # NOTE: the dsp module generates real-valued TX chirps. Real TX -> analytic
    # via hilbert. Use simpler approach: build analytic TX with scipy.hilbert.
    from scipy.signal import hilbert
    tx_analytic = hilbert(tx)
    for i in range(n_frames):
        s = i * chunk
        rx = audio[s:s + chunk]
        mix = rx * np.conj(tx_analytic[s:s + chunk])
        # lowpass
        bb_i = sosfiltfilt(sos, mix.real)
        bb_q = sosfiltfilt(sos, mix.imag)
        # FFT, take range bin
        sp = np.fft.fft(bb_i + 1j * bb_q)
        bin_idx = min(n_range_bin, len(sp) - 1)
        amps[i] = np.abs(sp[bin_idx])
        ph = np.angle(sp[bin_idx])
        # unwrap relative to previous
        if i == 0:
            prev_phase = ph
        d = np.angle(np.exp(1j * (ph - prev_phase)))
        phases[i] = prev_phase + d
        prev_phase = phases[i]
        times[i] = s / sr
    return times, phases, amps


def plot_session_214655():
    fig, axes = plt.subplots(4, 1, figsize=(12, 11))
    sess_name = "hp_blink_20260617_214655"
    sess = SESSIONS_DIR / sess_name
    # row 0: features.csv phase_pair_delta (post-FMCW detector input)
    times_pd, vals_pd = load_feature_column(sess, "phase_pair_delta")
    blinks = load_visual_blink_times(sess)
    mask = quiet_mask(times_pd, blinks)
    axes[0].plot(times_pd, vals_pd, lw=0.5, alpha=0.4, color="C7")
    axes[0].plot(times_pd[mask], vals_pd[mask], lw=0.7, color="C0",
                 label="quiet phase_pair_delta")
    for b in blinks:
        axes[0].axvline(b, color="red", lw=0.4, alpha=0.4)
    axes[0].set_title("214655 (FMCW): post-FMCW phase_pair_delta")
    axes[0].set_xlabel("time (s)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # row 1: FFT of quiet phase_pair_delta
    quiet_pd = vals_pd[mask]
    if len(quiet_pd) >= 16:
        f1, m1 = fft_traj(quiet_pd, CHUNK_DT)
        axes[1].plot(f1, m1, lw=0.8, color="C2")
        axes[1].set_xlim(0, 10)
        peaks_pd = list_top_peaks(f1, m1, top_n=6)
        for f, m in peaks_pd[:4]:
            axes[1].annotate(f"{f:.2f}Hz", xy=(f, m), xytext=(5, 5),
                             textcoords="offset points", fontsize=7)
        axes[1].set_title(f"FFT of quiet phase_pair_delta ({len(quiet_pd)} chunks)")
    else:
        peaks_pd = []
    axes[1].set_xlabel("Hz")
    axes[1].set_ylabel("dB")
    axes[1].grid(alpha=0.3)

    # row 2: raw audio FFT (full session, Welch)
    audio, sr = load_mono_wav(sess / "audio.wav")
    # Welch PSD restricted to 0-200Hz band to look for low-frequency modulation
    from scipy.signal import welch
    # take long FFT to resolve ~1Hz
    nper = 1 << int(np.log2(min(len(audio), sr * 8)))  # up to 8s windows
    f_audio, psd = welch(audio, sr, nperseg=nper)
    band = f_audio < 50
    axes[2].semilogy(f_audio[band], psd[band], lw=0.6)
    axes[2].set_title(f"Raw audio.wav Welch PSD (0-50Hz band, nperseg={nper})")
    axes[2].set_xlabel("Hz")
    axes[2].set_ylabel("PSD")
    axes[2].grid(alpha=0.3)
    # find peaks in 0.5-5 Hz band
    lowband = (f_audio > 0.5) & (f_audio < 5.0)
    if lowband.any():
        idxs = np.where(lowband)[0]
        order = idxs[np.argsort(-psd[idxs])]
        for i, k in enumerate(order[:4]):
            axes[2].annotate(f"{f_audio[k]:.2f}Hz", xy=(f_audio[k], psd[k]),
                             xytext=(5, 5), textcoords="offset points", fontsize=7)
        peaks_audio = [(f_audio[k], psd[k]) for k in order[:4]]
    else:
        peaks_audio = []

    # row 3: FMCW baseband phase from raw audio (range-bin 15)
    try:
        t_bb, p_bb, a_bb = fmcw_baseband_from_audio(
            audio, sr, n_range_bin=15, chirp_dur=0.05,
            f_lo=17000.0, f_hi=23000.0, emission="linear", tukey_alpha=0.0,
        )
        # Restrict to first ~2 min to keep FFT manageable
        n_keep = min(len(t_bb), int(120 * CHUNK_RATE))
        t_bb = t_bb[:n_keep]
        p_bb = p_bb[:n_keep]
        mask_bb = quiet_mask(t_bb, blinks)
        quiet_bb = p_bb[mask_bb]
        if len(quiet_bb) >= 16:
            f_bb, m_bb = fft_traj(quiet_bb, CHUNK_DT)
            axes[3].plot(f_bb, m_bb, lw=0.8, color="C3")
            axes[3].set_xlim(0, 10)
            peaks_bb = list_top_peaks(f_bb, m_bb, top_n=6)
            for f, m in peaks_bb[:4]:
                axes[3].annotate(f"{f:.2f}Hz", xy=(f, m), xytext=(5, 5),
                                 textcoords="offset points", fontsize=7)
            axes[3].set_title(f"Baseband range-bin 15 phase FFT (quiet, from raw audio.wav, {len(quiet_bb)} chunks)")
        else:
            peaks_bb = []
            axes[3].set_title("Baseband range-bin 15 phase FFT (insufficient quiet data)")
        axes[3].set_xlabel("Hz")
        axes[3].set_ylabel("dB")
        axes[3].grid(alpha=0.3)
    except Exception as e:
        peaks_bb = []
        axes[3].text(0.1, 0.5, f"baseband demod failed: {e}", transform=axes[3].transAxes)

    plt.tight_layout()
    out = OUT_DIR / "periodic_rootcause_214655.png"
    fig.savefig(out, dpi=120)
    print("saved:", out)
    return peaks_pd, peaks_audio, peaks_bb


# --------------------------------------------------------------------------
# Q3: cross-variant FFT
# --------------------------------------------------------------------------

def plot_variant_comparison():
    fig, axes = plt.subplots(len(BASELINE_VARIANTS), 2,
                             figsize=(13, 2.6 * len(BASELINE_VARIANTS)))
    if len(BASELINE_VARIANTS) == 1:
        axes = axes[np.newaxis, :]
    summary = []
    for row, (label, sess_name, emission) in enumerate(BASELINE_VARIANTS):
        sess = SESSIONS_DIR / sess_name
        signal_mode, _ = load_signal_mode(sess)
        if signal_mode == "tone":
            column = "phase"
        else:
            column = "phase_pair_delta"
        times, vals = load_feature_column(sess, column)
        blinks = load_visual_blink_times(sess)
        mask = quiet_mask(times, blinks)
        qvals = vals[mask]
        # left: trajectory
        ax = axes[row, 0]
        ax.plot(times, vals, lw=0.5, alpha=0.4, color="C7")
        ax.plot(times[mask], qvals, lw=0.7, color="C0", label="quiet")
        for b in blinks:
            ax.axvline(b, color="red", lw=0.4, alpha=0.4)
        ax.set_title(f"{label} ({emission}, signal={signal_mode}): {column}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(column)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

        ax = axes[row, 1]
        if len(qvals) >= 16:
            f, m = fft_traj(qvals, CHUNK_DT)
            ax.plot(f, m, lw=0.8)
            ax.set_xlim(0, 10)
            peaks = list_top_peaks(f, m, top_n=4)
            for pf, pm in peaks[:3]:
                ax.annotate(f"{pf:.2f}Hz", xy=(pf, pm), xytext=(5, 5),
                            textcoords="offset points", fontsize=7)
            summary.append((label, emission, len(qvals), peaks))
        else:
            summary.append((label, emission, len(qvals), []))
        ax.set_title(f"FFT of quiet {column} (n={len(qvals)})")
        ax.set_xlabel("Hz")
        ax.set_ylabel("dB")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "periodic_rootcause_variant_comparison.png"
    fig.savefig(out, dpi=120)
    print("saved:", out)
    return summary


def main():
    print("=== Q1: 174829 tone session ===")
    peaks_174829 = plot_session_174829()
    print("174829 top phase-FFT peaks (Hz, dB):")
    if peaks_174829:
        for f, m in peaks_174829:
            print(f"  {f:6.3f} Hz  {m:6.2f} dB")

    print()
    print("=== Q2: 214655 FMCW session ===")
    peaks_pd, peaks_audio, peaks_bb = plot_session_214655()
    print("214655 post-FMCW phase_pair_delta FFT peaks:")
    if peaks_pd:
        for f, m in peaks_pd:
            print(f"  {f:6.3f} Hz  {m:.2e}")
    print("214655 raw audio.wav Welch peaks (0.5-5Hz):")
    if peaks_audio:
        for f, p in peaks_audio:
            print(f"  {f:6.3f} Hz  PSD={p:.2e}")
    print("214655 baseband range-bin-15 phase FFT peaks:")
    if peaks_bb:
        for f, m in peaks_bb:
            print(f"  {f:6.3f} Hz  {m:6.2f} dB")

    print()
    print("=== Q3: cross-variant comparison ===")
    summary = plot_variant_comparison()
    for label, emission, n, peaks in summary:
        print(f"{label:18s} ({emission}) n_quiet={n:5d}  top peaks: " +
              ", ".join(f"{f:.2f}Hz" for f, _ in peaks[:3]))


if __name__ == "__main__":
    main()
