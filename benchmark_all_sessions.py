import argparse
import csv
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hp_acoustic_wave.benchmark import (
    _load_audio_config,
    _resolve_truth,
    load_manual_markers,
    load_visual_blink_markers,
    replay_feature_rows,
    reprocess_audio_feature_rows,
    score_events,
)
from hp_acoustic_wave.blink_detector import BlinkDetectionConfig


BLINK_METHODS = (
    "old_twinkle",
    "blinklistener",
    "bump",
    "twinkle",
    "shape",
    "motion",
    "pulse",
    "hybridpulse",
    "hybridcluster",
    "highrecall",
    "neural",
    "both",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark all saved blink sessions against all detectors.")
    parser.add_argument("--session-root", default="sessions")
    parser.add_argument("--output", default="benchmark_all_sessions.csv")
    parser.add_argument("--truth", choices=["auto", "manual", "visual"], default="auto")
    parser.add_argument("--match-before", type=float, default=0.5)
    parser.add_argument("--match-after", type=float, default=0.5)
    return parser.parse_args(argv)


def _session_dirs(root: Path):
    return sorted(
        path
        for path in root.glob("hp_blink_*")
        if (path / "metadata.json").exists() and (path / "audio.wav").exists()
    )


def _metadata(session: Path):
    with (session / "metadata.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def _markers(session: Path, truth: str):
    resolved = _resolve_truth(session, truth)
    if resolved == "visual":
        return resolved, load_visual_blink_markers(session)
    return resolved, load_manual_markers(session / "manual_markers.csv")


def _feature_rows_from_audio(session: Path):
    audio_config = _load_audio_config(session)
    signal_mode = str(audio_config.get("signal_mode", "tone"))
    emission = str(audio_config.get("fmcw_emission", "linear"))
    if signal_mode != "fmcw":
        emission = "cw_single"
    rows = reprocess_audio_feature_rows(
        session / "audio.wav",
        sample_rate=int(audio_config.get("sample_rate", 48_000)),
        tone_hz=float(audio_config.get("tone_hz", 18_500.0)),
        chunk_size=int(audio_config.get("chunk_size", 1024)),
        signal_mode=signal_mode,
        fmcw_freq_low=float(audio_config.get("fmcw_freq_low", 17_000.0)),
        fmcw_freq_high=float(audio_config.get("fmcw_freq_high", 23_000.0)),
        fmcw_chirp_duration=float(audio_config.get("fmcw_chirp_duration", 0.05)),
        fmcw_range_bin=int(audio_config.get("fmcw_range_bin", 15)),
        fmcw_lowpass_cutoff=float(audio_config.get("fmcw_lowpass_cutoff", 5_000.0)),
        fmcw_motion_amplitude_floor=float(audio_config.get("fmcw_motion_amplitude_floor", 0.02)),
        fmcw_output_amplitude=float(audio_config.get("output_amplitude", 0.2)),
        fmcw_emission=emission,
    )
    return rows, signal_mode, emission


def _duration_s(rows):
    if not rows:
        return 0.0
    return max(float(rows[-1].get("time_s") or 0.0), 0.0)


def _config(method: str):
    return BlinkDetectionConfig(
        method=method,
        twinkle_fmcw_use_intra_chirp_phase_pair=True,
    )


def _row(
    session: Path,
    method: str,
    truth: str,
    markers,
    rows,
    signal_mode: str,
    emission: str,
    duration_s: float,
    match_before_s: float,
    match_after_s: float,
):
    events = replay_feature_rows(rows, _config(method))
    summary = score_events(events, markers, match_before_s=match_before_s, match_after_s=match_after_s)
    false_alarm_rate = summary.fp / (summary.tp + summary.fp) if (summary.tp + summary.fp) else 0.0
    fp_per_min = summary.fp / duration_s * 60.0 if duration_s > 0 else 0.0
    meta = _metadata(session)
    return {
        "session": session.name,
        "signal_mode": signal_mode,
        "emission": emission,
        "method": method,
        "truth": truth,
        "duration_s": f"{duration_s:.3f}",
        "markers": summary.blink_markers,
        "events": summary.events,
        "tp": summary.tp,
        "fp": summary.fp,
        "fn": summary.fn,
        "precision": f"{summary.precision:.6f}",
        "recall": f"{summary.recall:.6f}",
        "false_alarm_rate": f"{false_alarm_rate:.6f}",
        "fp_per_min": f"{fp_per_min:.6f}",
        "f1": f"{summary.f1:.6f}",
        "recorded_method": meta.get("blink_method", ""),
    }


def _aggregate(rows, keys):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        acc = groups.setdefault(
            key,
            {
                **{k: row[k] for k in keys},
                "sessions": 0,
                "markers": 0,
                "events": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "duration_s": 0.0,
            },
        )
        acc["sessions"] += 1
        for metric in ("markers", "events", "tp", "fp", "fn"):
            acc[metric] += int(row[metric])
        acc["duration_s"] += float(row["duration_s"])
    out = []
    for acc in groups.values():
        tp = acc["tp"]
        fp = acc["fp"]
        fn = acc["fn"]
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        false_alarm_rate = fp / (tp + fp) if (tp + fp) else 0.0
        fp_per_min = fp / acc["duration_s"] * 60.0 if acc["duration_s"] > 0 else 0.0
        out.append(
            {
                **acc,
                "precision": precision,
                "recall": recall,
                "false_alarm_rate": false_alarm_rate,
                "fp_per_min": fp_per_min,
                "f1": f1,
            }
        )
    return sorted(out, key=lambda item: tuple(str(item[k]) for k in keys))


def _print_table(title, rows, columns):
    print(title)
    print(",".join(columns))
    for row in rows:
        values = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                value = f"{value:.3f}"
            values.append(str(value))
        print(",".join(values))
    print()


def main(argv=None):
    args = parse_args(argv)
    sessions = _session_dirs(Path(args.session_root))
    all_rows = []
    for session in sessions:
        truth, markers = _markers(session, args.truth)
        rows, signal_mode, emission = _feature_rows_from_audio(session)
        duration_s = _duration_s(rows)
        for method in BLINK_METHODS:
            all_rows.append(
                _row(
                    session,
                    method,
                    truth,
                    markers,
                    rows,
                    signal_mode,
                    emission,
                    duration_s,
                    args.match_before,
                    args.match_after,
                )
            )

    output = Path(args.output)
    fieldnames = list(all_rows[0].keys()) if all_rows else []
    if fieldnames:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    by_method = _aggregate(all_rows, ("method",))
    by_emission_method = _aggregate(all_rows, ("emission", "method"))
    _print_table(
        "OVERALL_BY_METHOD",
        by_method,
        ("method", "sessions", "markers", "events", "tp", "fp", "fn", "recall", "false_alarm_rate", "fp_per_min", "f1"),
    )
    _print_table(
        "BY_EMISSION_AND_METHOD",
        by_emission_method,
        ("emission", "method", "sessions", "markers", "events", "tp", "fp", "fn", "recall", "false_alarm_rate", "fp_per_min", "f1"),
    )
    print(f"wrote_csv: {output}")


if __name__ == "__main__":
    main()
