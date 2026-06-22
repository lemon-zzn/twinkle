"""Diagnostic dump: compare FMCW feature signatures at blink vs non-blink moments.

Reads audio.wav from each session, re-extracts DSP features with current code,
locates visual blink windows, and emits per-window statistics + a summary CSV.

Usage:
    python diagnostic_dump.py --sessions 184117 190655 214655 \
        --output docs/diagnostics/dump.csv
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hp_acoustic_wave.benchmark import _load_audio_config, reprocess_audio_feature_rows


SESSIONS_DIR = Path(__file__).resolve().parent / "sessions"
DOCS_DIR = Path(__file__).resolve().parent / "docs" / "diagnostics"


def load_visual_blink_times(session_dir: Path) -> List[float]:
    labels_path = session_dir / "visual_labels.csv"
    if not labels_path.exists():
        return []
    times: List[float] = []
    with labels_path.open(encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("is_blink_event") == "1":
                times.append(float(row["time_s"]))
    return times


def coherence_over_window(values: np.ndarray, window: int, baseline_median: float, jitter_scale: float) -> Tuple[float, float, float]:
    """Replicate _fmcw_coherence_score over a fixed window slice."""
    if values.size < window:
        return 0.0, 0.0, 0.0
    recent = values[-window:]
    local_median = float(np.median(recent))
    deviation = abs(local_median - baseline_median)
    steps = np.diff(recent)
    if steps.size == 0:
        return 0.0, 0.0, deviation
    jitter = float(np.sqrt(np.mean(np.square(steps))))
    smoothness = float(max(0.0, 1.0 - min(1.0, jitter / jitter_scale)))
    return float(deviation * smoothness), smoothness, deviation


def analyze_session(session_dir: Path, window_pre: float, window_post: float) -> Tuple[List[Dict], List[Dict]]:
    audio_cfg = _load_audio_config(session_dir)
    sample_rate = int(audio_cfg.get("sample_rate", 48000))
    chunk_size = int(audio_cfg.get("chunk_size", 1024))
    signal_mode = audio_cfg.get("signal_mode", "fmcw")
    fmcw_range_bin = int(audio_cfg.get("fmcw_range_bin", 15))
    fmcw_freq_low = float(audio_cfg.get("fmcw_freq_low", 17_000.0))
    fmcw_freq_high = float(audio_cfg.get("fmcw_freq_high", 23_000.0))
    fmcw_chirp_duration = float(audio_cfg.get("fmcw_chirp_duration", 0.05))
    fmcw_lowpass_cutoff = float(audio_cfg.get("fmcw_lowpass_cutoff", 5_000.0))
    fmcw_motion_amplitude_floor = float(audio_cfg.get("fmcw_motion_amplitude_floor", 0.02))
    output_amplitude = float(audio_cfg.get("output_amplitude", 0.2))

    if signal_mode != "fmcw":
        print(f"skip {session_dir.name}: signal_mode={signal_mode}", file=sys.stderr)
        return [], []

    rows = reprocess_audio_feature_rows(
        audio_path=session_dir / "audio.wav",
        sample_rate=sample_rate,
        tone_hz=float(audio_cfg.get("tone_hz", 18500.0)),
        chunk_size=chunk_size,
        signal_mode="fmcw",
        fmcw_freq_low=fmcw_freq_low,
        fmcw_freq_high=fmcw_freq_high,
        fmcw_chirp_duration=fmcw_chirp_duration,
        fmcw_range_bin=fmcw_range_bin,
        fmcw_lowpass_cutoff=fmcw_lowpass_cutoff,
        fmcw_motion_amplitude_floor=fmcw_motion_amplitude_floor,
        fmcw_output_amplitude=output_amplitude,
    )

    times = np.asarray([float(r["time_s"]) for r in rows], dtype=np.float64)
    ppd = np.asarray([float(r["phase_pair_delta"] or 0.0) for r in rows], dtype=np.float64)
    amp = np.asarray([float(r["amplitude"] or 0.0) for r in rows], dtype=np.float64)
    me = np.asarray([float(r["motion_energy"] or 0.0) for r in rows], dtype=np.float64)
    # Cross-chirp range-bin phase and its delta (alternative to intra-chirp phase-pair)
    rbp = np.asarray([float(r["phase"] or 0.0) for r in rows], dtype=np.float64)
    rbpd = np.asarray([float(r["phase_delta"] or 0.0) for r in rows], dtype=np.float64)

    # Global baseline from full session (mimics a long-history baseline)
    ppd_global_median = float(np.median(ppd))
    ppd_global_std = max(float(np.std(ppd)), 0.1)
    amp_global_median = float(np.median(amp))
    rbpd_global_median = float(np.median(rbpd))
    rbpd_global_std = max(float(np.std(rbpd)), 0.1)

    blink_times = load_visual_blink_times(session_dir)
    chunk_dt = float(times[1] - times[0]) if times.size > 1 else 0.05

    blink_windows: List[Dict] = []
    for bt in blink_times:
        lo = bt - window_pre
        hi = bt + window_post
        mask = (times >= lo) & (times <= hi)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        slice_ppd = ppd[idx]
        slice_amp = amp[idx]
        slice_me = me[idx]
        slice_rbpd = rbpd[idx]
        peak_ppd_idx = int(np.argmax(np.abs(slice_ppd)))
        # Multi-scale coherence on the slice using global baseline
        coh5, sm5, dev5 = coherence_over_window(slice_ppd, 5, ppd_global_median, ppd_global_std)
        coh9, sm9, dev9 = coherence_over_window(slice_ppd, 9, ppd_global_median, ppd_global_std)
        coh13, sm13, dev13 = coherence_over_window(slice_ppd, 13, ppd_global_median, ppd_global_std)
        # Cross-chirp range-bin phase_delta coherence (alternative signal)
        rb_coh9, rb_sm9, rb_dev9 = coherence_over_window(slice_rbpd, 9, rbpd_global_median, rbpd_global_std)
        amp_change_rel = abs(float(np.median(slice_amp)) - amp_global_median) / max(abs(amp_global_median), 1e-6)
        blink_windows.append({
            "session": session_dir.name,
            "blink_time_s": round(bt, 3),
            "n_frames": int(idx.size),
            "peak_abs_ppd": float(np.max(np.abs(slice_ppd))),
            "peak_ppd_value": float(slice_ppd[peak_ppd_idx]),
            "ppd_jitter_slice": float(np.sqrt(np.mean(np.square(np.diff(slice_ppd))))),
            "amp_change_rel": amp_change_rel,
            "motion_energy_max": float(np.max(slice_me)),
            "motion_energy_mean": float(np.mean(slice_me)),
            "coh5": coh5, "sm5": sm5, "dev5": dev5,
            "coh9": coh9, "sm9": sm9, "dev9": dev9,
            "coh13": coh13, "sm13": sm13, "dev13": dev13,
            "rb_coh9": rb_coh9, "rb_sm9": rb_sm9, "rb_dev9": rb_dev9,
            "global_ppd_median": ppd_global_median,
            "global_ppd_std": ppd_global_std,
        })

    # Sample non-blink windows: 10 random windows matching blink window length
    rng = np.random.default_rng(42)
    window_len_frames = int(round((window_pre + window_post) / chunk_dt))
    non_blink_windows: List[Dict] = []
    # Exclude ±2s around any blink
    exclude_mask = np.zeros(times.size, dtype=bool)
    for bt in blink_times:
        exclude_mask |= (times >= bt - 2.0) & (times <= bt + 2.0)
    candidate_starts = np.where(~exclude_mask)[0]
    candidate_starts = candidate_starts[candidate_starts + window_len_frames < times.size]
    if candidate_starts.size > 0:
        picks = rng.choice(candidate_starts, size=min(10, candidate_starts.size), replace=False)
        for start in picks:
            sl = slice(int(start), int(start) + window_len_frames)
            slice_ppd = ppd[sl]
            slice_amp = amp[sl]
            slice_me = me[sl]
            slice_rbpd = rbpd[sl]
            coh5, sm5, dev5 = coherence_over_window(slice_ppd, 5, ppd_global_median, ppd_global_std)
            coh9, sm9, dev9 = coherence_over_window(slice_ppd, 9, ppd_global_median, ppd_global_std)
            coh13, sm13, dev13 = coherence_over_window(slice_ppd, 13, ppd_global_median, ppd_global_std)
            rb_coh9, rb_sm9, rb_dev9 = coherence_over_window(slice_rbpd, 9, rbpd_global_median, rbpd_global_std)
            amp_change_rel = abs(float(np.median(slice_amp)) - amp_global_median) / max(abs(amp_global_median), 1e-6)
            non_blink_windows.append({
                "session": session_dir.name,
                "window_start_s": round(float(times[sl.start]), 3),
                "n_frames": int(slice_ppd.size),
                "peak_abs_ppd": float(np.max(np.abs(slice_ppd))) if slice_ppd.size else 0.0,
                "ppd_jitter_slice": float(np.sqrt(np.mean(np.square(np.diff(slice_ppd))))) if slice_ppd.size > 1 else 0.0,
                "amp_change_rel": amp_change_rel,
                "motion_energy_max": float(np.max(slice_me)) if slice_me.size else 0.0,
                "motion_energy_mean": float(np.mean(slice_me)) if slice_me.size else 0.0,
                "coh5": coh5, "sm5": sm5, "dev5": dev5,
                "coh9": coh9, "sm9": sm9, "dev9": dev9,
                "coh13": coh13, "sm13": sm13, "dev13": dev13,
                "rb_coh9": rb_coh9, "rb_sm9": rb_sm9, "rb_dev9": rb_dev9,
                "global_ppd_median": ppd_global_median,
                "global_ppd_std": ppd_global_std,
            })

    return blink_windows, non_blink_windows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dump blink vs non-blink feature signatures")
    parser.add_argument("--sessions", nargs="+", default=["184117", "190655", "214655"])
    parser.add_argument("--window-pre", type=float, default=0.4)
    parser.add_argument("--window-post", type=float, default=0.4)
    parser.add_argument("--output", default=str(DOCS_DIR / "dump.csv"))
    args = parser.parse_args(argv)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_blink: List[Dict] = []
    all_non: List[Dict] = []
    for s in args.sessions:
        session_dir = SESSIONS_DIR / f"hp_blink_20260617_{s}"
        if not session_dir.exists():
            print(f"missing session: {session_dir}", file=sys.stderr)
            continue
        bw, nbw = analyze_session(session_dir, args.window_pre, args.window_post)
        all_blink.extend(bw)
        all_non.extend(nbw)
        print(f"{s}: blink_windows={len(bw)}, non_blink_windows={len(nbw)}")

    # Write CSV
    blink_fields = ["session", "blink_time_s", "n_frames", "peak_abs_ppd", "peak_ppd_value",
              "ppd_jitter_slice", "amp_change_rel", "motion_energy_max", "motion_energy_mean",
              "coh5", "sm5", "dev5", "coh9", "sm9", "dev9", "coh13", "sm13", "dev13",
              "rb_coh9", "rb_sm9", "rb_dev9",
              "global_ppd_median", "global_ppd_std"]
    non_fields = ["session", "window_start_s", "n_frames", "peak_abs_ppd",
              "ppd_jitter_slice", "amp_change_rel", "motion_energy_max", "motion_energy_mean",
              "coh5", "sm5", "dev5", "coh9", "sm9", "dev9", "coh13", "sm13", "dev13",
              "rb_coh9", "rb_sm9", "rb_dev9",
              "global_ppd_median", "global_ppd_std"]

    def fmt(row, fields):
        out = {}
        for k in fields:
            v = row.get(k)
            if isinstance(v, float):
                out[k] = f"{v:.6f}"
            else:
                out[k] = v if v is not None else ""
        return out

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=blink_fields)
        writer.writeheader()
        for row in all_blink:
            writer.writerow(fmt(row, blink_fields))
        writer.writerow({})
        writer.writerow({"session": "# NON-BLINK WINDOWS"})
        writer = csv.DictWriter(handle, fieldnames=non_fields)
        for row in all_non:
            writer.writerow(fmt(row, non_fields))
    print(f"wrote {out_path}")

    # Summary
    def summarize(rows: List[Dict], label: str):
        if not rows:
            return
        print(f"\n=== {label} ({len(rows)} windows) ===")
        for key in ["peak_abs_ppd", "ppd_jitter_slice", "amp_change_rel", "motion_energy_max",
                    "coh9", "sm9", "dev9", "rb_coh9", "rb_sm9", "rb_dev9", "global_ppd_std"]:
            vals = np.asarray([float(r[key]) for r in rows])
            print(f"  {key:24s} median={np.median(vals):.4f} p10={np.percentile(vals,10):.4f} p90={np.percentile(vals,90):.4f}")
    summarize(all_blink, "BLINK windows")
    summarize(all_non, "NON-BLINK windows")


if __name__ == "__main__":
    main()
