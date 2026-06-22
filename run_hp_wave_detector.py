import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hp_acoustic_wave.app import RealtimeHandWaveApp, resolve_camera_backend_name
from hp_acoustic_wave.config import (
    AppConfig,
    AudioConfig,
    BlinkConfig,
    CameraConfig,
    DetectorConfig,
    VisualBlinkConfig,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Realtime HP laptop acoustic hand-wave detector")
    parser.add_argument("--mode", choices=["wave", "blink"], default="wave", help="Realtime detector mode")
    parser.add_argument(
        "--blink-method",
        choices=["old_twinkle", "blinklistener", "bump", "twinkle", "shape", "motion", "pulse", "hybridpulse", "hybridcluster", "highrecall", "neural", "nerual", "both"],
        default="twinkle",
        help="Blink detection method when --mode blink",
    )
    parser.add_argument(
        "--neural-model-path",
        default="",
        help="Path to a trained neural blink .npz model; default uses models/neural_blink_model.npz",
    )
    parser.add_argument("--session-root", default="sessions", help="Directory for saved run folders")
    parser.add_argument("--frequency", type=float, default=18500.0, help="Speaker tone in Hz")
    parser.add_argument("--signal-mode", choices=["tone", "fmcw"], default="tone", help="Transmit signal type")
    parser.add_argument("--sample-rate", type=int, default=48000, help="Audio sample rate")
    parser.add_argument("--chunk-size", type=int, default=1024, help="Audio chunk size")
    parser.add_argument("--amplitude", type=float, default=0.12, help="Output tone amplitude from 0.0 to 1.0")
    parser.add_argument("--input-device", type=int, default=None, help="sounddevice input device index")
    parser.add_argument("--output-device", type=int, default=None, help="sounddevice output device index")
    parser.add_argument("--fmcw-freq-low", type=float, default=17_000.0, help="FMCW chirp start frequency in Hz")
    parser.add_argument("--fmcw-freq-high", type=float, default=23_000.0, help="FMCW chirp end frequency in Hz")
    parser.add_argument("--fmcw-chirp-duration", type=float, default=0.05, help="FMCW chirp duration in seconds")
    parser.add_argument("--fmcw-lowpass-cutoff", type=float, default=5_000.0, help="FMCW mixer low-pass cutoff")
    parser.add_argument("--fmcw-range-bin", type=int, default=15, help="FMCW range bin used for wave detection")
    parser.add_argument(
        "--fmcw-motion-amplitude-floor",
        type=float,
        default=0.02,
        help="Amplitude floor for FMCW relative motion energy.",
    )
    parser.add_argument(
        "--fmcw-emission",
        choices=["linear", "linear_tukey", "cw_fmcw_hybrid", "triangle"],
        default="linear",
        help="FMCW transmit waveform (only used when --signal-mode fmcw)",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument(
        "--camera-backend",
        choices=["auto", "any", "avfoundation"],
        default="auto",
        help="OpenCV camera backend.",
    )
    parser.add_argument("--camera-width", type=int, default=1280, help="Requested camera width")
    parser.add_argument("--camera-height", type=int, default=720, help="Requested camera height")
    parser.add_argument("--camera-fps", type=float, default=30.0, help="Requested camera FPS")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Probe OpenCV camera indexes and exit",
    )
    parser.add_argument(
        "--camera-probe-count",
        type=int,
        default=5,
        help="Number of camera indexes to probe with --list-cameras",
    )
    parser.add_argument("--no-camera", action="store_true", help="Run acoustic detector without camera")
    parser.add_argument(
        "--no-visual-labels",
        action="store_true",
        help="Disable MediaPipe EAR visual blink ground-truth labels in blink mode",
    )
    parser.add_argument("--visual-ear-threshold", type=float, default=0.22, help="EAR threshold for visual labels")
    parser.add_argument(
        "--visual-consecutive-frames",
        type=int,
        default=3,
        help="Closed-eye frames required before a visual blink label",
    )
    parser.add_argument(
        "--visual-min-detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe face detection confidence for visual labels",
    )
    parser.add_argument(
        "--visual-min-tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe face tracking confidence for visual labels",
    )
    parser.add_argument("--threshold-k", type=float, default=8.0, help="MAD multiplier for event threshold")
    parser.add_argument("--min-energy", type=float, default=0.015, help="Minimum event threshold")
    parser.add_argument("--blink-threshold-k", type=float, default=2.5, help="MAD multiplier for blink threshold")
    parser.add_argument("--blink-min-score", type=float, default=0.006, help="Minimum blink candidate score")
    parser.add_argument(
        "--blink-absolute-score-floor",
        type=float,
        default=0.0,
        help="Optional absolute blink score trigger for subtle Twinkle candidates",
    )
    parser.add_argument(
        "--blink-phase-step-floor",
        type=float,
        default=0.015,
        help="Minimum unwrapped phase step used to count Twinkle trajectory direction changes",
    )
    parser.add_argument("--blink-refractory", type=float, default=1.05, help="Minimum seconds between blink events")
    parser.add_argument(
        "--blink-startup-ignore",
        type=float,
        default=0.0,
        help="Seconds to ignore blink auto-events while the acoustic baseline settles",
    )
    parser.add_argument(
        "--blink-release-ratio",
        type=float,
        default=0.4,
        help="Fraction of the event threshold that rearms blink detection after a candidate",
    )
    parser.add_argument(
        "--blink-disable-twinkle-peak-gate",
        action="store_true",
        help="Use the older threshold/release gate instead of local Twinkle peak gating",
    )
    parser.add_argument(
        "--blink-twinkle-peak-min-ratio",
        type=float,
        default=1.0,
        help="Minimum score/threshold ratio for local Twinkle peak candidates",
    )
    parser.add_argument(
        "--blink-twinkle-max-peak-score",
        type=float,
        default=1.2,
        help="Reject local Twinkle peaks above this score; <=0 disables the ceiling",
    )
    parser.add_argument(
        "--blink-twinkle-max-motion-energy",
        type=float,
        default=0.22,
        help="Reject local Twinkle peaks when the raw motion energy is above this value",
    )
    parser.add_argument(
        "--blink-twinkle-max-sign-changes",
        type=int,
        default=2,
        help="Reject local Twinkle peaks with more trajectory direction changes than this",
    )
    parser.add_argument(
        "--blink-twinkle-large-motion-score",
        type=float,
        default=1.5,
        help="Start temporary large-motion suppression above this Twinkle score",
    )
    parser.add_argument(
        "--blink-twinkle-large-motion-energy",
        type=float,
        default=0.18,
        help="Start temporary large-motion suppression above this raw motion energy",
    )
    parser.add_argument(
        "--blink-twinkle-large-motion-suppress",
        type=float,
        default=0.75,
        help="Seconds to suppress Twinkle peaks after a likely large-motion burst",
    )
    parser.add_argument(
        "--blink-twinkle-fmcw-min-amplitude",
        type=float,
        default=0.005,
        help="Minimum selected FMCW range-bin amplitude before Twinkle trusts phase",
    )
    parser.add_argument(
        "--blink-twinkle-fmcw-min-score",
        type=float,
        default=0.09,
        help="FMCW-specific Twinkle score floor before peak gating",
    )
    parser.add_argument(
        "--blink-twinkle-fmcw-phase-pair-weight",
        type=float,
        default=0.5,
        help="Weight applied to FMCW phase-pair delta in Twinkle blink scoring",
    )
    parser.add_argument(
        "--blink-twinkle-fmcw-max-amplitude-delta-ratio",
        type=float,
        default=0.45,
        help="Reject FMCW Twinkle candidates above this selected-bin amplitude delta ratio",
    )
    parser.add_argument(
        "--blink-twinkle-fmcw-min-range-spread-ratio",
        type=float,
        default=0.0,
        help="Reject FMCW Twinkle candidates whose range spread ratio is below this value; <=0 disables",
    )
    parser.add_argument(
        "--blink-twinkle-fmcw-refractory",
        type=float,
        default=1.3,
        help="Minimum seconds between FMCW Twinkle blink candidates",
    )
    parser.add_argument(
        "--blink-twinkle-fmcw-use-intra-chirp-phase-pair",
        action="store_true",
        help="Use the paper-style same-chirp phase-pair trajectory instead of selected range-bin phase",
    )
    parser.add_argument(
        "--blink-twinkle-fmcw-use-orthogonality",
        action="store_true",
        help="Enable amplitude-phase orthogonality scoring (BlinkListener insight)",
    )
    parser.add_argument(
        "--blink-candidate-windows",
        default="0",
        help="Comma-separated Twinkle candidate windows; 0 disables multi-window voting",
    )
    parser.add_argument(
        "--blink-min-candidate-votes",
        type=int,
        default=1,
        help="Minimum candidate-window votes when candidate windows are enabled",
    )
    parser.add_argument("--duration", type=float, default=None, help="Optional auto-stop duration in seconds")
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


def build_app_config(args):
    chunk_size = args.chunk_size
    if args.signal_mode == "fmcw":
        chunk_size = int(round(args.sample_rate * args.fmcw_chirp_duration))
    return AppConfig(
        audio=AudioConfig(
            sample_rate=args.sample_rate,
            tone_hz=args.frequency,
            chunk_size=chunk_size,
            output_amplitude=args.amplitude,
            input_device=args.input_device,
            output_device=args.output_device,
            signal_mode=args.signal_mode,
            fmcw_freq_low=args.fmcw_freq_low,
            fmcw_freq_high=args.fmcw_freq_high,
            fmcw_chirp_duration=args.fmcw_chirp_duration,
            fmcw_lowpass_cutoff=args.fmcw_lowpass_cutoff,
            fmcw_range_bin=args.fmcw_range_bin,
            fmcw_motion_amplitude_floor=args.fmcw_motion_amplitude_floor,
            fmcw_emission=args.fmcw_emission,
        ),
        detector=DetectorConfig(
            threshold_k=args.threshold_k,
            min_energy=args.min_energy,
        ),
        blink=BlinkConfig(
            method=args.blink_method,
            threshold_k=args.blink_threshold_k,
            min_score=args.blink_min_score,
            refractory_s=args.blink_refractory,
            absolute_score_floor=args.blink_absolute_score_floor,
            phase_step_floor=args.blink_phase_step_floor,
            startup_ignore_s=args.blink_startup_ignore,
            release_ratio=args.blink_release_ratio,
            twinkle_candidate_windows=_parse_candidate_windows(args.blink_candidate_windows),
            twinkle_min_candidate_votes=args.blink_min_candidate_votes,
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
            neural_model_path=args.neural_model_path,
        ),
        camera=CameraConfig(
            enabled=not args.no_camera,
            index=args.camera_index,
            backend=args.camera_backend,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        ),
        visual_blink=VisualBlinkConfig(
            enabled=not args.no_visual_labels,
            ear_threshold=args.visual_ear_threshold,
            consecutive_frames=args.visual_consecutive_frames,
            min_detection_confidence=args.visual_min_detection_confidence,
            min_tracking_confidence=args.visual_min_tracking_confidence,
        ),
        mode=args.mode,
        session_root=args.session_root,
        window_name="HP Acoustic Blink" if args.mode == "blink" else "HP Acoustic Hand Wave",
        max_duration_s=args.duration,
    )


def main():
    args = parse_args()
    if args.list_cameras:
        import cv2

        list_cameras(cv2, args.camera_backend, args.camera_probe_count)
        return
    config = build_app_config(args)
    app = RealtimeHandWaveApp(config)
    session_dir = app.run()
    print(f"Saved session: {session_dir}")


def list_cameras(cv2, backend: str, probe_count: int):
    resolved_backend = resolve_camera_backend_name(backend)
    print(f"Camera probe backend: {resolved_backend}")
    for index in range(max(0, int(probe_count))):
        if resolved_backend == "avfoundation":
            camera = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        else:
            camera = cv2.VideoCapture(index)
        opened = camera.isOpened()
        ok = False
        width = 0
        height = 0
        if opened:
            ok, frame = camera.read()
            if ok and frame is not None:
                height, width = frame.shape[:2]
        camera.release()
        status = "ok" if opened and ok else "unavailable"
        detail = f" {width}x{height}" if width and height else ""
        print(f"  [{index}] {status}{detail}")


if __name__ == "__main__":
    main()
