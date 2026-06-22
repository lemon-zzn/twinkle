"""Reprocess + replay + score comparison across DSP preprocessing variants.

Measures TP/FP/balanced_score for the 3 reference sessions, so each
preprocessing experiment (Hamming window, DC suppression, etc.) can be
compared against baseline.
"""
import argparse
import sys
from pathlib import Path
from dataclasses import asdict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hp_acoustic_wave.benchmark import (
    _load_audio_config,
    _resolve_truth,
    benchmark_session,
)
from hp_acoustic_wave.blink_detector import BlinkDetectionConfig


DEFAULT_SESSIONS = ["184117", "190655", "214655"]


def default_config() -> BlinkDetectionConfig:
    return BlinkDetectionConfig(
        method="twinkle",
        twinkle_fmcw_use_intra_chirp_phase_pair=True,
        twinkle_fmcw_min_score=0.12,
    )


def run_session(session_id: str, args) -> dict:
    session_dir = Path(__file__).resolve().parent / "sessions" / f"hp_blink_20260617_{session_id}"
    if not session_dir.exists():
        return {"session": session_id, "error": f"missing {session_dir}"}
    config = default_config()
    run = benchmark_session(
        session_dir,
        config,
        source="audio",
        sample_rate=args.sample_rate,
        tone_hz=args.frequency,
        chunk_size=args.chunk_size,
        truth=args.truth,
    )
    s = asdict(run.summary)
    return {
        "session": session_id,
        "events": s["events"],
        "blink_hits": s["blink_hits"],
        "blink_markers": s["blink_markers"],
        "unexplained_events": s["unexplained_events"],
        "nonblink_events": s["nonblink_events"],
        "balanced_score": s["balanced_score"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark summary across sessions")
    parser.add_argument("--sessions", nargs="+", default=DEFAULT_SESSIONS)
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--frequency", type=float, default=18_500.0)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--truth", default="auto")
    args = parser.parse_args(argv)

    results = []
    for sid in args.sessions:
        r = run_session(sid, args)
        results.append(r)
        if "error" in r:
            print(f"{sid}: ERROR {r['error']}")
            continue
        print(
            f"{sid}: events={r['events']} "
            f"hits={r['blink_hits']}/{r['blink_markers']} "
            f"unexplained={r['unexplained_events']} "
            f"nonblink={r['nonblink_events']} "
            f"score={r['balanced_score']:.2f}"
        )


if __name__ == "__main__":
    main()
