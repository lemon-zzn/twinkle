from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class AudioConfig:
    sample_rate: int = 48000
    tone_hz: float = 18500.0
    chunk_size: int = 1024
    output_amplitude: float = 0.12
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    signal_mode: str = "tone"
    fmcw_freq_low: float = 17_000.0
    fmcw_freq_high: float = 23_000.0
    fmcw_chirp_duration: float = 0.05
    fmcw_lowpass_cutoff: float = 5_000.0
    fmcw_range_bin: int = 15
    fmcw_motion_amplitude_floor: float = 0.02
    fmcw_emission: str = "linear"  # linear | linear_tukey | cw_fmcw_hybrid | triangle


@dataclass
class DetectorConfig:
    history_size: int = 120
    min_history: int = 20
    threshold_k: float = 8.0
    min_energy: float = 0.015
    refractory_s: float = 0.9
    baseline_freeze_s: float = 1.0
    detection_hold_s: float = 2.0


@dataclass
class BlinkConfig:
    method: str = "twinkle"
    history_size: int = 120
    min_history: int = 20
    threshold_k: float = 2.5
    min_score: float = 0.006
    refractory_s: float = 1.05
    baseline_freeze_s: float = 0.60
    short_window: int = 9
    twinkle_candidate_windows: Tuple[int, ...] = ()
    twinkle_min_candidate_votes: int = 1
    phase_pair_lag: int = 3
    phase_step_floor: float = 0.015
    absolute_score_floor: float = 0.0
    startup_ignore_s: float = 0.0
    release_ratio: float = 0.4
    twinkle_peak_gate_enabled: bool = True
    twinkle_peak_min_ratio: float = 1.0
    twinkle_max_peak_score: float = 1.2
    twinkle_max_motion_energy: float = 0.22
    twinkle_max_sign_changes: int = 2
    twinkle_large_motion_score: float = 1.5
    twinkle_large_motion_energy: float = 0.18
    twinkle_large_motion_suppress_s: float = 0.75
    # FMCW-specific
    twinkle_fmcw_min_amplitude: float = 0.005
    twinkle_fmcw_min_score: float = 0.09
    twinkle_fmcw_phase_pair_weight: float = 0.5
    twinkle_fmcw_max_amplitude_delta_ratio: float = 0.45
    twinkle_fmcw_min_range_spread_ratio: float = 0.0
    twinkle_fmcw_refractory_s: float = 1.3
    twinkle_fmcw_use_intra_chirp_phase_pair: bool = False
    twinkle_fmcw_use_orthogonality: bool = False  # 实验 1：幅度-相位正交特征
    # Shape-based segmentation detector (Twinkle paper pattern matching)
    shape_smoothing_window: int = 5
    shape_detection_window: int = 30
    shape_min_edge_len_s: float = 0.05
    shape_max_edge_len_s: float = 0.5
    shape_edge_sigma_k: float = 3.0
    shape_refractory_s: float = 0.5
    # Neural detector
    neural_model_path: str = ""


@dataclass
class CameraConfig:
    enabled: bool = True
    index: int = 0
    backend: str = "auto"
    width: int = 1280
    height: int = 720
    fps: float = 30.0


@dataclass
class VisualBlinkConfig:
    enabled: bool = True
    ear_threshold: float = 0.22
    consecutive_frames: int = 3
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    blink: BlinkConfig = field(default_factory=BlinkConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    visual_blink: VisualBlinkConfig = field(default_factory=VisualBlinkConfig)
    mode: str = "wave"
    session_root: str = "sessions"
    window_name: str = "HP Acoustic Hand Wave"
    max_duration_s: Optional[float] = None

    def to_metadata(self) -> Dict[str, Any]:
        audio_md = asdict(self.audio)
        if audio_md.get("signal_mode") != "fmcw":
            audio_md["fmcw_emission"] = "cw_single"
        return {
            "audio": audio_md,
            "detector": asdict(self.detector),
            "blink": asdict(self.blink),
            "camera": asdict(self.camera),
            "visual_blink": asdict(self.visual_blink),
            "mode": self.mode,
            "session_root": self.session_root,
            "window_name": self.window_name,
            "max_duration_s": self.max_duration_s,
        }
