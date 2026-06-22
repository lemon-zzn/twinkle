import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hp_acoustic_wave.benchmark import (
    _load_audio_config,
    _resolve_truth,
    benchmark_session,
    load_manual_markers,
    load_visual_blink_markers,
    replay_feature_rows,
    reprocess_audio_feature_rows,
    score_events,
)
from hp_acoustic_wave.blink_detector import BlinkDetectionConfig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Replay and score saved HP acoustic blink sessions")
    parser.add_argument(
        "--session",
        default=str(Path("sessions") / "hp_blink_20260612_193944"),
        help="Saved session directory containing features.csv and manual_markers.csv",
    )
    parser.add_argument("--source", choices=["features", "audio", "events"], default="features")
    parser.add_argument(
        "--fmcw-emission",
        choices=["linear", "linear_tukey", "cw_fmcw_hybrid", "triangle", "cw_single"],
        default=None,
        help="Override emission label (default: read from session meta.json)",
    )
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--frequency", type=float, default=18_500.0)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--tukey-alpha", type=float, default=0.0)
    parser.add_argument(
        "--blink-method",
        choices=["old_twinkle", "blinklistener", "bump", "twinkle", "shape", "motion", "pulse", "hybridpulse", "hybridcluster", "highrecall", "neural", "nerual", "both"],
        default="twinkle",
    )
    parser.add_argument(
        "--neural-model-path",
        default="",
        help="Path to a trained neural blink .npz model; default uses models/neural_blink_model.npz",
    )
    parser.add_argument("--blink-threshold-k", type=float, default=2.5)
    parser.add_argument("--blink-min-score", type=float, default=0.006)
    parser.add_argument("--blink-refractory", type=float, default=1.05)
    parser.add_argument("--blink-startup-ignore", type=float, default=0.0)
    parser.add_argument("--blink-short-window", type=int, default=9)
    parser.add_argument("--blink-candidate-windows", default="0")
    parser.add_argument("--blink-min-candidate-votes", type=int, default=1)
    parser.add_argument("--blink-phase-pair-lag", type=int, default=3)
    parser.add_argument("--blink-phase-step-floor", type=float, default=0.015)
    parser.add_argument("--blink-disable-twinkle-peak-gate", action="store_true")
    parser.add_argument("--blink-twinkle-peak-min-ratio", type=float, default=1.0)
    parser.add_argument("--blink-twinkle-max-peak-score", type=float, default=1.2)
    parser.add_argument("--blink-twinkle-max-motion-energy", type=float, default=0.22)
    parser.add_argument("--blink-twinkle-max-sign-changes", type=int, default=2)
    parser.add_argument("--blink-twinkle-large-motion-score", type=float, default=1.5)
    parser.add_argument("--blink-twinkle-large-motion-energy", type=float, default=0.18)
    parser.add_argument("--blink-twinkle-large-motion-suppress", type=float, default=0.75)
    parser.add_argument("--blink-twinkle-fmcw-min-amplitude", type=float, default=0.005)
    parser.add_argument("--blink-twinkle-fmcw-min-score", type=float, default=0.09)
    parser.add_argument("--blink-twinkle-fmcw-phase-pair-weight", type=float, default=0.5)
    parser.add_argument("--blink-twinkle-fmcw-max-amplitude-delta-ratio", type=float, default=0.45)
    parser.add_argument("--blink-twinkle-fmcw-min-range-spread-ratio", type=float, default=0.0)
    parser.add_argument("--blink-twinkle-fmcw-refractory", type=float, default=1.3)
    parser.add_argument("--blink-twinkle-fmcw-use-intra-chirp-phase-pair", action="store_true")
    parser.add_argument("--blink-twinkle-fmcw-use-orthogonality", action="store_true")
    # Shape-segmentation detector params
    parser.add_argument("--shape-smoothing-window", type=int, default=5)
    parser.add_argument("--shape-detection-window", type=int, default=30)
    parser.add_argument("--shape-min-edge-len-s", type=float, default=0.05)
    parser.add_argument("--shape-max-edge-len-s", type=float, default=0.5)
    parser.add_argument("--shape-edge-sigma-k", type=float, default=3.0)
    parser.add_argument("--shape-refractory-s", type=float, default=0.5)
    parser.add_argument(
        "--truth",
        choices=["auto", "manual", "visual"],
        default="auto",
        help="Ground truth source; auto prefers visual_blink events when present",
    )
    parser.add_argument("--write-events", default="", help="Optional path for replayed event CSV")
    parser.add_argument(
        "--check-periodic",
        action="store_true",
        help="Print anti-periodic cluster_ratio: fraction of event gaps within 0.2s of refractory_s.",
    )
    return parser.parse_args(argv)


def _parse_candidate_windows(value: str):
    windows = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        parsed = int(token)
        if parsed > 0:
            windows.append(parsed)
    return tuple(windows)


def check_periodic_cluster_ratio(event_times, refractory_s, tolerance=0.2):
    """Fraction of inter-event gaps within +/-tolerance of refractory_s.

    A high ratio (>0.5) indicates periodic noise-driven firing — the hallmark
    of the regression we're trying to eliminate. <0.3 is the target.
    """
    if len(event_times) < 3:
        return 0.0
    import numpy as np
    gaps = np.diff(sorted(event_times))
    clustered = float(sum(abs(float(g) - refractory_s) < tolerance for g in gaps))
    return clustered / float(len(gaps))


def _matches_any_marker(t, marker_times, before_s, after_s):
    return any((t - after_s) <= mt <= (t + before_s) for mt in marker_times)


def false_positive_events(events, blink_markers, large_motion_markers,
                          match_before_s=0.8, match_after_s=0.8):
    """Return events that don't match any blink or large_motion marker.

    These are the "static-period" events — false positives fired when the user
    was not blinking. Periodic clustering among *these* (not TPs) is the real
    regression signature per CLAUDE.md.
    """
    blink_ts = [m.time_s for m in blink_markers]
    motion_ts = [m.time_s for m in large_motion_markers]
    all_ts = blink_ts + motion_ts
    return [
        e for e in events
        if e.label != "large_motion"
        and not _matches_any_marker(e.time_s, all_ts, match_before_s, match_after_s)
    ]


def static_period_stats(events, blink_markers, large_motion_markers,
                        total_duration_s, refractory_s, tolerance=0.2,
                        match_before_s=0.8, match_after_s=0.8):
    """Anti-periodic metrics computed on FALSE POSITIVES only.

    Returns dict with:
      fp_count            - events in static periods (no nearby marker)
      fp_rate_per_min     - fp_count / total_duration_s * 60
      fp_cluster_ratio    - fraction of inter-FP gaps within tolerance of refractory_s
      fp_event_rate_ratio - fp_count vs random-uniform expectation under refractory
    """
    fp_events = false_positive_events(
        events, blink_markers, large_motion_markers, match_before_s, match_after_s
    )
    fp_times = [e.time_s for e in fp_events]
    if total_duration_s <= 0:
        fp_rate_per_min = 0.0
    else:
        fp_rate_per_min = len(fp_times) / total_duration_s * 60.0
    cluster_ratio = check_periodic_cluster_ratio(fp_times, refractory_s, tolerance)
    # How close is the observed FP rate to a uniform process limited by refractory?
    expected_uniform_per_min = 60.0 / refractory_s if refractory_s > 0 else 0.0
    if expected_uniform_per_min > 0:
        rate_ratio = fp_rate_per_min / expected_uniform_per_min
    else:
        rate_ratio = 0.0
    return {
        "fp_count": len(fp_times),
        "fp_rate_per_min": fp_rate_per_min,
        "fp_cluster_ratio": cluster_ratio,
        "fp_event_rate_ratio": rate_ratio,
    }


def build_config(args):
    return BlinkDetectionConfig(
        method=args.blink_method,
        threshold_k=args.blink_threshold_k,
        min_score=args.blink_min_score,
        refractory_s=args.blink_refractory,
        startup_ignore_s=args.blink_startup_ignore,
        short_window=args.blink_short_window,
        twinkle_candidate_windows=_parse_candidate_windows(args.blink_candidate_windows),
        twinkle_min_candidate_votes=args.blink_min_candidate_votes,
        phase_pair_lag=args.blink_phase_pair_lag,
        phase_step_floor=args.blink_phase_step_floor,
        twinkle_peak_gate_enabled=not args.blink_disable_twinkle_peak_gate,
        twinkle_peak_min_ratio=args.blink_twinkle_peak_min_ratio,
        twinkle_max_peak_score=args.blink_twinkle_max_peak_score,
        twinkle_max_motion_energy=args.blink_twinkle_max_motion_energy,
        twinkle_max_sign_changes=args.blink_twinkle_max_sign_changes,
        twinkle_large_motion_score=args.blink_twinkle_large_motion_score,
        twinkle_large_motion_energy=args.blink_twinkle_large_motion_energy,
        twinkle_large_motion_suppress_s=args.blink_twinkle_large_motion_suppress,
        twinkle_fmcw_min_amplitude=args.blink_twinkle_fmcw_min_amplitude,
        twinkle_fmcw_min_score=args.blink_twinkle_fmcw_min_score,
        twinkle_fmcw_phase_pair_weight=args.blink_twinkle_fmcw_phase_pair_weight,
        twinkle_fmcw_max_amplitude_delta_ratio=args.blink_twinkle_fmcw_max_amplitude_delta_ratio,
        twinkle_fmcw_min_range_spread_ratio=args.blink_twinkle_fmcw_min_range_spread_ratio,
        twinkle_fmcw_refractory_s=args.blink_twinkle_fmcw_refractory,
        twinkle_fmcw_use_intra_chirp_phase_pair=args.blink_twinkle_fmcw_use_intra_chirp_phase_pair,
        twinkle_fmcw_use_orthogonality=args.blink_twinkle_fmcw_use_orthogonality,
        shape_smoothing_window=args.shape_smoothing_window,
        shape_detection_window=args.shape_detection_window,
        shape_min_edge_len_s=args.shape_min_edge_len_s,
        shape_max_edge_len_s=args.shape_max_edge_len_s,
        shape_edge_sigma_k=args.shape_edge_sigma_k,
        shape_refractory_s=args.shape_refractory_s,
        neural_model_path=args.neural_model_path,
    )


def write_events(path, events):
    fieldnames = ["event_id", "time_s", "label", "method", "score", "motion_energy", "threshold"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "event_id": event.event_id,
                    "time_s": f"{event.time_s:.6f}",
                    "label": event.label,
                    "method": event.method,
                    "score": f"{event.score:.9f}",
                    "motion_energy": f"{event.motion_energy:.9f}",
                    "threshold": f"{event.threshold:.9f}",
                }
            )


def main():
    args = parse_args()
    run = benchmark_session(
        Path(args.session),
        build_config(args),
        source=args.source,
        sample_rate=args.sample_rate,
        tone_hz=args.frequency,
        chunk_size=args.chunk_size,
        tukey_alpha=args.tukey_alpha,
        truth=args.truth,
    )
    summary = asdict(run.summary)
    print(f"session: {run.session}")
    print(f"source: {run.source}")
    print(f"truth: {run.truth}")
    print(
        "events={events} blink_hits={blink_hits}/{blink_markers} "
        "large_motion_hits={large_motion_hits}/{large_motion_markers} "
        "unexplained_events={unexplained_events} nonblink_events={nonblink_events} "
        "tp={tp} fp={fp} fn={fn} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} "
        "balanced_score={balanced_score:.2f}".format(**summary)
    )
    if args.write_events:
        write_events(args.write_events, run.events)
        print(f"wrote_events: {args.write_events}")
    if args.check_periodic:
        config = build_config(args)
        refractory = getattr(config, "refractory_s", 1.05)
        all_times = [e.time_s for e in run.events]
        cluster_ratio_all = check_periodic_cluster_ratio(all_times, refractory)
        # FP-only (static-period) stats: this is the real "periodic false-positive"
        # signature — events fired while the user was NOT blinking.
        blink_markers = [m for m in run.markers if m.label == "blink"]
        large_motion_markers = [m for m in run.markers if m.label == "large_motion"]
        # Estimate total duration from features (last chunk time).
        total_duration_s = 0.0
        if all_times:
            total_duration_s = max(all_times[-1], 1.0)
        stats = static_period_stats(
            run.events, blink_markers, large_motion_markers,
            total_duration_s, refractory,
        )
        print(
            f"check_periodic: refractory={refractory:.2f}s "
            f"all_cluster_ratio={cluster_ratio_all:.3f} "
            f"fp_count={stats['fp_count']} fp_rate_per_min={stats['fp_rate_per_min']:.2f} "
            f"fp_cluster_ratio={stats['fp_cluster_ratio']:.3f} "
            f"fp_rate_ratio={stats['fp_event_rate_ratio']:.3f}"
        )
        # Verdict on the static-period FP rate (the real CLAUDE.md metric).
        if stats["fp_cluster_ratio"] > 0.5:
            print("FAIL: fp_cluster_ratio > 0.5 (periodic noise-driven firing in static periods)")
        elif stats["fp_cluster_ratio"] < 0.3 and stats["fp_count"] <= 20:
            print("PASS: static-period FP clustering low and count reasonable")
        else:
            print("BORDERLINE: check fp_count and fp_rate_per_min")


if __name__ == "__main__":
    main()
