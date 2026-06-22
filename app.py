import platform
import queue
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from hp_acoustic_wave.audio_devices import format_audio_device_selection, resolve_audio_device_selection
from hp_acoustic_wave.blink_detector import BlinkDetectionConfig, build_blink_detector
from hp_acoustic_wave.config import AppConfig
from hp_acoustic_wave.detector import AdaptiveWaveDetector
from hp_acoustic_wave.dsp import (
    ChunkFeature,
    FmcwBackgroundSubtractor,
    extract_chunk_feature,
    extract_fmcw_chunk_feature,
    generate_fmcw_chirp,
    generate_tone,
)
from hp_acoustic_wave.session_io import SessionWriter, create_session_dir
from hp_acoustic_wave.ui_state import detection_status
from hp_acoustic_wave.visual_blink import (
    EAR_LEFT_EYE_IDXS,
    EAR_RIGHT_EYE_IDXS,
    VisualBlinkLabeler,
    calculate_ear,
)


def resolve_camera_backend_name(backend: str) -> str:
    if backend == "auto":
        if platform.system() == "Darwin":
            return "avfoundation"
        return "any"
    return backend


class RealtimeHandWaveApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.audio_queue = queue.Queue(maxsize=64)
        self.playback_sample_index = 0
        self.previous_feature: Optional[ChunkFeature] = None
        self.detector = self._build_detector()
        self.energy_history = deque(maxlen=240)
        self.threshold_history = deque(maxlen=240)
        self.phase_trajectory_history = deque(maxlen=240)
        self.ungated_coherence_history = deque(maxlen=240)
        self.session_dir: Optional[Path] = None
        self.writer: Optional[SessionWriter] = None
        self.manual_marker_count = 0
        self.latest_event_id = 0
        self.latest_time_s = 0.0
        self.last_detection_time_s: Optional[float] = None
        self.last_detection_display_time_s: Optional[float] = None
        self.last_detection_energy = 0.0
        self.last_detection_method = "wave"
        self.last_detection_score = 0.0
        self.latest_score = 0.0
        self.latest_threshold = 0.0
        self.latest_detector_method = "wave"
        self.running = False
        self.video_writer = None
        self.camera = None
        self.camera_enabled = False
        self.camera_open_seconds: Optional[float] = None
        self.audio_device_selection = None
        self.face_mesh = None
        self.visual_labeler: Optional[VisualBlinkLabeler] = None
        self.visual_labeling_enabled = False
        self.visual_labeling_error: Optional[str] = None
        self.latest_visual_state = None
        self.latest_visual_blink_count = 0
        self.last_visual_blink_time_s: Optional[float] = None
        self.last_visual_blink_display_time_s: Optional[float] = None
        self.stream_started_monotonic: Optional[float] = None
        self.fmcw_background_subtractor = (
            FmcwBackgroundSubtractor()
            if self.config.audio.signal_mode == "fmcw"
            else None
        )

    def _build_detector(self):
        if self.config.mode == "blink":
            blink_config = BlinkDetectionConfig(**self.config.blink.__dict__)
            return build_blink_detector(blink_config)
        return AdaptiveWaveDetector(self.config.detector)

    def _audio_callback(self, indata, outdata, frames, callback_time, status):
        start_sample = self.playback_sample_index
        playback = self._generate_playback(frames, start_sample)
        outdata[:, 0] = playback
        recorded = indata[:, 0].copy()
        try:
            self.audio_queue.put_nowait((start_sample, recorded, str(status)))
        except queue.Full:
            pass
        self.playback_sample_index += frames

    def _generate_playback(self, frames: int, start_sample: int) -> np.ndarray:
        if self.config.audio.signal_mode == "fmcw":
            return generate_fmcw_chirp(
                num_samples=frames,
                sample_rate=self.config.audio.sample_rate,
                freq_low=self.config.audio.fmcw_freq_low,
                freq_high=self.config.audio.fmcw_freq_high,
                chirp_duration=self.config.audio.fmcw_chirp_duration,
                start_sample=start_sample,
                amplitude=self.config.audio.output_amplitude,
                emission=self.config.audio.fmcw_emission,
            )
        return generate_tone(
            num_samples=frames,
            sample_rate=self.config.audio.sample_rate,
            frequency_hz=self.config.audio.tone_hz,
            start_sample=start_sample,
            amplitude=self.config.audio.output_amplitude,
        )

    def _open_camera(self, cv2):
        if not self.config.camera.enabled:
            return
        started_at = time.monotonic()
        camera = self._create_video_capture(cv2)
        self.camera_open_seconds = time.monotonic() - started_at
        if not camera.isOpened():
            camera.release()
            self._print_camera_open_failure()
            return
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera.width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera.height)
        camera.set(cv2.CAP_PROP_FPS, self.config.camera.fps)
        self.camera = camera
        self.camera_enabled = True

    def _camera_backend_name(self) -> str:
        return resolve_camera_backend_name(self.config.camera.backend)

    def _create_video_capture(self, cv2):
        backend = self._camera_backend_name()
        if backend == "avfoundation":
            return cv2.VideoCapture(self.config.camera.index, cv2.CAP_AVFOUNDATION)
        return cv2.VideoCapture(self.config.camera.index)

    def _print_camera_open_failure(self):
        backend = self._camera_backend_name()
        message = (
            f"Camera unavailable: OpenCV could not open camera index {self.config.camera.index} "
            f"with backend {backend}."
        )
        if platform.system() == "Darwin":
            message += (
                " On macOS, grant Camera permission to the app running this script "
                "(Terminal, iTerm, VS Code, or Python) in System Settings > Privacy & Security > Camera, "
                "then restart that app. You can also try --camera-backend any or --camera-index N."
            )
        print(_safe_console_text(message), flush=True)

    def _open_video_writer(self, cv2):
        if not self.camera_enabled or self.session_dir is None:
            return
        width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.config.camera.width
        height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.config.camera.height
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        path = str(self.session_dir / "camera.mp4")
        writer = cv2.VideoWriter(path, fourcc, self.config.camera.fps, (width, height + 160))
        if writer.isOpened():
            self.video_writer = writer
        else:
            writer.release()

    def _open_visual_labeler(self):
        if (
            self.config.mode != "blink"
            or not self.config.camera.enabled
            or not self.camera_enabled
            or not self.config.visual_blink.enabled
        ):
            return
        try:
            import mediapipe as mp
        except ImportError:
            self.visual_labeling_error = "mediapipe is not installed"
            print(
                "Visual blink labels disabled: install mediapipe to enable EAR ground-truth labels.",
                flush=True,
            )
            return

        self.visual_labeler = VisualBlinkLabeler(
            ear_threshold=self.config.visual_blink.ear_threshold,
            consecutive_frames=self.config.visual_blink.consecutive_frames,
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.config.visual_blink.min_detection_confidence,
            min_tracking_confidence=self.config.visual_blink.min_tracking_confidence,
        )
        self.visual_labeling_enabled = True

    def _process_visual_frame(self, cv2, frame):
        if not self.visual_labeling_enabled or self.face_mesh is None or self.visual_labeler is None:
            return
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
        except Exception as exc:
            self.visual_labeling_enabled = False
            self.visual_labeling_error = f"visual label error: {exc}"
            print(f"Visual blink labels disabled: {exc}", flush=True)
            return

        face_present = bool(results.multi_face_landmarks)
        left_ear = 0.0
        right_ear = 0.0
        if face_present:
            landmarks = results.multi_face_landmarks[0].landmark
            left_ear = calculate_ear(landmarks, EAR_LEFT_EYE_IDXS)
            right_ear = calculate_ear(landmarks, EAR_RIGHT_EYE_IDXS)

        state = self.visual_labeler.update(
            time_s=self._elapsed_time_s(),
            left_ear=left_ear,
            right_ear=right_ear,
            face_present=face_present,
        )
        self.latest_visual_state = state
        self.latest_visual_blink_count = state.blink_count
        if self.writer is not None:
            self.writer.write_visual_label(
                time_s=state.time_s,
                face_present=state.face_present,
                left_ear=state.left_ear,
                right_ear=state.right_ear,
                mean_ear=state.mean_ear,
                is_closed=state.is_closed,
                closed_frames=state.closed_frames,
                is_blink_event=state.is_blink_event,
                blink_count=state.blink_count,
            )
            if state.is_blink_event:
                self.writer.write_event(
                    event_id=state.blink_count,
                    time_s=state.time_s,
                    motion_energy=self.previous_feature.motion_energy if self.previous_feature is not None else 0.0,
                    threshold=self.config.visual_blink.ear_threshold,
                    label="visual_blink",
                    method="mediapipe_ear",
                    score=state.mean_ear,
                )
        if state.is_blink_event:
            self.last_visual_blink_time_s = state.time_s
            self.last_visual_blink_display_time_s = time.monotonic()

    def _elapsed_time_s(self) -> float:
        if self.stream_started_monotonic is None:
            return float(self.latest_time_s)
        return float(time.monotonic() - self.stream_started_monotonic)

    def _process_audio_queue(self):
        processed = 0
        while True:
            try:
                start_sample, samples, status_text = self.audio_queue.get_nowait()
            except queue.Empty:
                break
            if self.writer is not None:
                self.writer.write_audio(samples)
            feature = self._extract_feature(samples, start_sample)
            self.previous_feature = feature
            self.latest_time_s = feature.time_s
            detection = self._update_detector(feature)
            plot_value = self.latest_score if self.config.mode == "blink" else feature.motion_energy
            self.energy_history.append(plot_value)
            self.threshold_history.append(self.latest_threshold)
            if self.config.mode == "blink":
                if (
                    feature.signal_mode == "fmcw"
                    and feature.phase_pair_delta is not None
                ):
                    phase_val = float(feature.phase_pair_delta)
                else:
                    phase_val = float(feature.phase)
                self.phase_trajectory_history.append(phase_val)
                self.ungated_coherence_history.append(
                    float(getattr(self.detector, "last_ungated_coherence", 0.0))
                )
            if detection.is_event:
                event_time_s = detection.event_time_s if detection.event_time_s is not None else feature.time_s
                self.latest_event_id = detection.event_id
                self.last_detection_time_s = event_time_s
                self.last_detection_display_time_s = time.monotonic()
                self.last_detection_energy = feature.motion_energy
                self.last_detection_score = self.latest_score
                self.last_detection_method = self.latest_detector_method
                if self.writer is not None:
                    self.writer.write_event(
                        event_id=detection.event_id,
                        time_s=event_time_s,
                        motion_energy=feature.motion_energy,
                        threshold=self.latest_threshold,
                        label="blink_candidate" if self.config.mode == "blink" else "wave",
                        method=self.latest_detector_method,
                        score=self.latest_score,
                    )
            if self.writer is not None:
                self.writer.write_feature(self._feature_row(feature, detection))
            processed += 1
        if processed and self.writer is not None:
            self.writer.flush()

    def _update_detector(self, feature: ChunkFeature):
        if self.config.mode == "blink":
            detection = self.detector.update(feature)
            self.latest_score = float(detection.score)
            self.latest_threshold = float(detection.threshold)
            self.latest_detector_method = detection.method
            return detection

        detection = self.detector.update(feature.time_s, feature.motion_energy)
        self.latest_score = float(feature.motion_energy)
        self.latest_threshold = float(detection.threshold)
        self.latest_detector_method = "fmcw_wave" if feature.signal_mode == "fmcw" else "wave"
        return detection

    def _extract_feature(self, samples: np.ndarray, start_sample: int) -> ChunkFeature:
        if self.config.audio.signal_mode == "fmcw":
            tx = self._generate_playback(len(samples), start_sample)
            return extract_fmcw_chunk_feature(
                samples=samples,
                tx_samples=tx,
                sample_rate=self.config.audio.sample_rate,
                freq_low=self.config.audio.fmcw_freq_low,
                freq_high=self.config.audio.fmcw_freq_high,
                chirp_duration=self.config.audio.fmcw_chirp_duration,
                range_bin=self.config.audio.fmcw_range_bin,
                start_sample=start_sample,
                previous=self.previous_feature,
                lowpass_cutoff=self.config.audio.fmcw_lowpass_cutoff,
                motion_amplitude_floor=self.config.audio.fmcw_motion_amplitude_floor,
                background_subtractor=self.fmcw_background_subtractor,
                emission=self.config.audio.fmcw_emission,
            )
        return extract_chunk_feature(
            samples=samples,
            sample_rate=self.config.audio.sample_rate,
            tone_hz=self.config.audio.tone_hz,
            start_sample=start_sample,
            previous=self.previous_feature,
        )

    def _feature_row(self, feature: ChunkFeature, detection):
        row = {
            "time_s": f"{feature.time_s:.6f}",
            "sample_index": feature.sample_index,
            "i": f"{feature.i_value:.9f}",
            "q": f"{feature.q_value:.9f}",
            "amplitude": f"{feature.amplitude:.9f}",
            "amplitude_delta": f"{feature.amplitude_delta:.9f}",
            "phase": f"{feature.phase:.9f}",
            "phase_delta": f"{feature.phase_delta:.9f}",
            "phase_pair_delta": _format_metric(feature.phase_pair_delta),
            "phase_pair_vote_ratio": _format_metric(feature.phase_pair_vote_ratio),
            "phase_pair_consistency": _format_metric(feature.phase_pair_consistency),
            "phase_pair_candidate_count": (
                "" if feature.phase_pair_candidate_count is None else int(feature.phase_pair_candidate_count)
            ),
            "phase_pair_deltas": _format_metric_tuple(feature.phase_pair_deltas),
            "motion_energy": f"{feature.motion_energy:.9f}",
            "rms": f"{feature.rms:.9f}",
            "peak_abs": f"{feature.peak_abs:.9f}",
            "baseline": f"{detection.baseline:.9f}",
            "mad": f"{detection.mad:.9f}",
            "threshold": f"{detection.threshold:.9f}",
            "detector_method": self.latest_detector_method,
            "blink_score": f"{self.latest_score:.9f}" if self.config.mode == "blink" else "",
            "blink_threshold": f"{self.latest_threshold:.9f}" if self.config.mode == "blink" else "",
            "blink_baseline": f"{detection.baseline:.9f}" if self.config.mode == "blink" else "",
            "blink_mad": f"{detection.mad:.9f}" if self.config.mode == "blink" else "",
            "is_event": int(detection.is_event),
            "event_id": detection.event_id,
            "signal_mode": feature.signal_mode,
            "range_bin": "" if feature.range_bin is None else feature.range_bin,
            "range_distance_m": _format_metric(feature.range_distance_m),
            "range_spread_bins": "" if feature.range_spread_bins is None else int(feature.range_spread_bins),
            "range_spread_ratio": _format_metric(feature.range_spread_ratio),
            "range_dominance_ratio": _format_metric(feature.range_dominance_ratio),
            "visual_face_present": _format_bool_metric(getattr(self.latest_visual_state, "face_present", None)),
            "visual_left_ear": _format_metric(getattr(self.latest_visual_state, "left_ear", None)),
            "visual_right_ear": _format_metric(getattr(self.latest_visual_state, "right_ear", None)),
            "visual_mean_ear": _format_metric(getattr(self.latest_visual_state, "mean_ear", None)),
            "visual_is_closed": _format_bool_metric(getattr(self.latest_visual_state, "is_closed", None)),
            "visual_blink_event": _format_bool_metric(getattr(self.latest_visual_state, "is_blink_event", None)),
            "visual_blink_count": "" if self.latest_visual_state is None else self.latest_visual_state.blink_count,
        }
        metrics = getattr(detection, "metrics", {}) or {}
        row.update(
            {
                "blinklistener_viewing_amplitude": _format_metric(metrics.get("viewing_amplitude")),
                "blinklistener_viewing_range": _format_metric(metrics.get("viewing_range")),
                "blinklistener_raw_viewing_score": _format_metric(metrics.get("raw_viewing_score")),
                "blinklistener_relative_viewing_score": _format_metric(metrics.get("relative_viewing_score")),
                "blinklistener_fmcw_impulse_score": _format_metric(metrics.get("fmcw_impulse_score")),
                "blinklistener_center_i": _format_metric(metrics.get("center_i")),
                "blinklistener_center_q": _format_metric(metrics.get("center_q")),
                "twinkle_phase_pair_delta": _format_metric(metrics.get("phase_pair_delta")),
                "twinkle_trajectory_span": _format_metric(metrics.get("trajectory_span")),
                "twinkle_trajectory_rms": _format_metric(
                    metrics.get("acceleration_rms", metrics.get("trajectory_rms"))
                ),
                "twinkle_peak_score": _format_metric(metrics.get("twinkle_peak_score")),
                "twinkle_peak_threshold": _format_metric(metrics.get("twinkle_peak_threshold")),
                "twinkle_peak_motion_energy": _format_metric(metrics.get("twinkle_peak_motion_energy")),
                "twinkle_peak_sign_changes": _format_metric(metrics.get("twinkle_peak_sign_changes")),
                "twinkle_peak_trajectory_span": _format_metric(metrics.get("twinkle_peak_trajectory_span")),
                "twinkle_candidate_local_peak": _format_metric(metrics.get("twinkle_candidate_local_peak")),
                "twinkle_candidate_rising_edge": _format_metric(metrics.get("twinkle_candidate_rising_edge")),
                "twinkle_large_motion_suppressed": _format_metric(metrics.get("twinkle_large_motion_suppressed")),
                "twinkle_fmcw_morphology_ok": _format_metric(metrics.get("twinkle_fmcw_morphology_ok")),
                "twinkle_fmcw_range_bin": _format_metric(metrics.get("twinkle_fmcw_range_bin")),
                "twinkle_fmcw_phase_score": _format_metric(metrics.get("twinkle_fmcw_phase_score")),
                "twinkle_fmcw_amplitude_ok": _format_metric(metrics.get("twinkle_fmcw_amplitude_ok")),
                "twinkle_fmcw_amplitude_stable": _format_metric(metrics.get("twinkle_fmcw_amplitude_stable")),
                "twinkle_fmcw_amplitude_delta_ratio": _format_metric(
                    metrics.get("twinkle_fmcw_amplitude_delta_ratio")
                ),
                "twinkle_effective_refractory_s": _format_metric(
                    metrics.get("twinkle_effective_refractory_s")
                ),
                "motion_amplitude_impulse": _format_metric(metrics.get("motion_amplitude_impulse")),
                "motion_phase_pair_impulse": _format_metric(metrics.get("motion_phase_pair_impulse")),
                "shape_local_edge_mag": _format_metric(metrics.get("shape_local_edge_mag")),
                "shape_local_edge_len": _format_metric(metrics.get("shape_local_edge_len")),
                "shape_local_center_time_s": _format_metric(metrics.get("shape_local_center_time_s")),
                "shape_local_rebound": _format_metric(metrics.get("shape_local_rebound")),
                "shape_local_active": _format_metric(metrics.get("shape_local_active")),
                "pulse_score": _format_metric(metrics.get("pulse_score")),
                "pulse_amp_impulse": _format_metric(metrics.get("pulse_amp_impulse")),
                "pulse_phase_impulse": _format_metric(metrics.get("pulse_phase_impulse")),
                "pulse_phase_smoothness": _format_metric(metrics.get("pulse_phase_smoothness")),
                "pulse_consistency": _format_metric(metrics.get("pulse_consistency")),
                "pulse_range_spread": _format_metric(metrics.get("pulse_range_spread")),
                "pulse_motion": _format_metric(metrics.get("pulse_motion")),
                "pulse_quality": _format_metric(metrics.get("pulse_quality")),
                "bump_candidate_raw": _format_metric(metrics.get("bump_candidate_raw")),
                "bump_prominence": _format_metric(metrics.get("bump_prominence")),
                "bump_return_ratio": _format_metric(metrics.get("bump_return_ratio")),
                "bump_source_diversity": _format_metric(metrics.get("bump_source_diversity")),
                "bump_periodic_clock_suppressed": _format_metric(
                    metrics.get("bump_periodic_clock_suppressed")
                ),
                "highrecall_source": _format_metric(metrics.get("highrecall_source")),
                "highrecall_weak_raw": _format_metric(metrics.get("highrecall_weak_raw")),
                "highrecall_weak_prominence": _format_metric(
                    metrics.get("highrecall_weak_prominence")
                ),
                "highrecall_weak_source_count": _format_metric(
                    metrics.get("highrecall_weak_source_count")
                ),
                "highrecall_event_count": _format_metric(metrics.get("highrecall_event_count")),
                "hybridpulse_support": _format_metric(metrics.get("hybridpulse_support")),
                "hybridpulse_event_count": _format_metric(metrics.get("hybridpulse_event_count")),
            }
        )
        return row

    def _blank_frame(self, cv2):
        frame = np.zeros((self.config.camera.height, self.config.camera.width, 3), dtype=np.uint8)
        cv2.putText(frame, "Camera unavailable", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 180, 255), 2)
        return frame

    def _draw_overlay(self, cv2, frame):
        height, width = frame.shape[:2]
        plot_h = 160
        canvas = np.zeros((height + plot_h, width, 3), dtype=np.uint8)
        canvas[:height, :width] = frame
        panel_y = height

        energy = self.energy_history[-1] if self.energy_history else 0.0
        threshold = self.threshold_history[-1] if self.threshold_history else self.config.detector.min_energy
        status = detection_status(
            current_time_s=time.monotonic(),
            last_detection_time_s=self.last_detection_display_time_s,
            hold_s=self._detection_display_hold_s(),
            detected_label="BLINK CANDIDATE" if self.config.mode == "blink" else "WAVE DETECTED",
        )
        status_color = (35, 35, 230) if status.is_detected else (35, 120, 35)
        text_color = (255, 255, 255)
        cv2.rectangle(canvas, (0, 0), (width, 58), status_color, -1)
        cv2.putText(canvas, status.label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.15, text_color, 3)
        if status.is_detected:
            if self.config.mode == "blink":
                detail = f"#{self.latest_event_id}  {self.last_detection_method}  S={self.last_detection_score:.4f}"
            else:
                detail = f"#{self.latest_event_id}  E={self.last_detection_energy:.2f}"
        else:
            if self.config.mode == "blink":
                detail = f"{self.latest_detector_method}  S={energy:.4f}  T={threshold:.4f}"
            else:
                detail = f"E={energy:.2f}  T={threshold:.2f}"
        cv2.putText(canvas, detail, (330, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)

        title = f"t={self.latest_time_s:6.2f}s  auto={self.latest_event_id}  manual={self.manual_marker_count}"
        cv2.putText(canvas, title, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        if self.config.mode == "blink":
            help_text = "b: blink marker   w: big motion   m: marker   q/Esc: quit"
        else:
            help_text = "m: manual marker   q/Esc: quit"
        cv2.putText(canvas, help_text, (20, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 2)
        visual_text = self._visual_status_text()
        if visual_text:
            cv2.putText(canvas, visual_text, (20, 154), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 235, 255), 2)

        # Dual-curve panel: top = phase trajectory (centered at 0),
        # bottom = ungated coherence score + threshold.
        left = 20
        right = width - 20
        panel_top = panel_y + 20
        panel_bottom = height + plot_h - 20
        mid_y = (panel_top + panel_bottom) // 2
        gap = 4
        top_top, top_bottom = panel_top, mid_y - gap
        bot_top, bot_bottom = mid_y + gap, panel_bottom

        if self.config.mode == "blink":
            # --- Top: phase trajectory (bipolar, centered at mid_top) ---
            phases = list(self.phase_trajectory_history)
            cv2.rectangle(canvas, (left, top_top), (right, top_bottom), (80, 80, 80), 1)
            cv2.putText(canvas, "phase trajectory", (left + 4, top_top + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 255), 1)
            if len(phases) >= 2:
                max_abs = max(abs(min(phases)), abs(max(phases)), 1e-6)
                center_y = (top_top + top_bottom) // 2
                span = (top_bottom - top_top) // 2 - 2
                points = []
                for idx, val in enumerate(phases):
                    x = int(left + idx * (right - left) / max(1, len(phases) - 1))
                    y = int(center_y - (val / max_abs) * span)
                    points.append((x, y))
                cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (0, 255, 0), 2)

            # --- Bottom: ungated coherence + threshold ---
            scores = list(self.ungated_coherence_history)
            thresholds = list(self.threshold_history)
            cv2.rectangle(canvas, (left, bot_top), (right, bot_bottom), (80, 80, 80), 1)
            cv2.putText(canvas, "ungated coherence", (left + 4, bot_top + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 255), 1)
            if len(scores) >= 2:
                all_vals = list(scores) + list(thresholds)
                max_value = max(max(all_vals), self.config.detector.min_energy, 1e-6)
                # threshold (orange dashed)
                if len(thresholds) >= 2:
                    pts = []
                    for idx, val in enumerate(thresholds):
                        x = int(left + idx * (right - left) / max(1, len(thresholds) - 1))
                        y = int(bot_bottom - min(val / max_value, 1.0) * (bot_bottom - bot_top - 4))
                        pts.append((x, y))
                    for i in range(0, len(pts) - 1, 2):
                        cv2.line(canvas, pts[i], pts[i + 1], (0, 180, 255), 2)
                # ungated coherence (blue)
                pts = []
                for idx, val in enumerate(scores):
                    x = int(left + idx * (right - left) / max(1, len(scores) - 1))
                    y = int(bot_bottom - min(val / max_value, 1.0) * (bot_bottom - bot_top - 4))
                    pts.append((x, y))
                cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, (255, 180, 0), 2)
        else:
            values = list(self.energy_history)
            thresholds = list(self.threshold_history)
            if values:
                max_value = max(max(values), max(thresholds) if thresholds else 0.0, self.config.detector.min_energy)
                max_value = max(max_value, 1e-6)
                cv2.rectangle(canvas, (left, panel_top), (right, panel_bottom), (80, 80, 80), 1)
                for series, color in ((values, (0, 255, 0)), (thresholds, (0, 180, 255))):
                    if len(series) < 2:
                        continue
                    points = []
                    for idx, val in enumerate(series):
                        x = int(left + idx * (right - left) / max(1, len(series) - 1))
                        y = int(panel_bottom - min(val / max_value, 1.0) * (panel_bottom - panel_top))
                        points.append((x, y))
                    cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2)
        return canvas

    def _detection_display_hold_s(self) -> float:
        if self.config.mode == "blink":
            return 0.55
        return float(self.config.detector.detection_hold_s)

    def _visual_status_text(self) -> str:
        if self.config.mode != "blink" or not self.config.visual_blink.enabled:
            return ""
        if self.visual_labeling_enabled and self.latest_visual_state is not None:
            state = self.latest_visual_state
            status = "closed" if state.is_closed else "open"
            event_age = (
                time.monotonic() - self.last_visual_blink_display_time_s
                if self.last_visual_blink_display_time_s is not None
                else 999.0
            )
            event_tag = " VISUAL BLINK" if event_age < 0.8 else ""
            return f"vision: {status} EAR={state.mean_ear:.3f} truth={state.blink_count}{event_tag}"
        if self.visual_labeling_error:
            return f"vision: unavailable ({self.visual_labeling_error})"
        if self.camera_enabled:
            return "vision: starting"
        return "vision: no camera"

    def _write_marker_for_key(self, key: int):
        marker_key = chr(key)
        if marker_key == "b":
            label = "blink"
        elif marker_key == "w":
            label = "large_motion"
        else:
            label = "manual"

        self.manual_marker_count += 1
        if self.writer is not None:
            self.writer.write_manual_marker(
                self.latest_time_s,
                label=label,
                key=marker_key,
                feature_snapshot=self._latest_feature_snapshot(),
                event_id=self.latest_event_id,
            )

    def _latest_feature_snapshot(self):
        if self.previous_feature is None:
            return {}
        return {
            "amplitude": self.previous_feature.amplitude,
            "phase": self.previous_feature.phase,
            "motion_energy": self.previous_feature.motion_energy,
            "signal_mode": self.previous_feature.signal_mode,
            "range_bin": self.previous_feature.range_bin,
            "range_distance_m": self.previous_feature.range_distance_m,
        }

    def _write_metadata(self, shutdown_reason: str):
        if self.writer is None or self.session_dir is None:
            return
        payload = {
            "config": self.config.to_metadata(),
            "session_dir": str(self.session_dir),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "audio_devices": self.audio_device_selection,
            "camera_enabled": self.camera_enabled,
            "camera_open_seconds": self.camera_open_seconds,
            "video_saved": self.video_writer is not None,
            "auto_event_count": self.latest_event_id,
            "manual_marker_count": self.manual_marker_count,
            "detector_mode": self.config.mode,
            "blink_method": self.config.blink.method if self.config.mode == "blink" else None,
            "visual_labeling_enabled": self.visual_labeling_enabled,
            "visual_labeling_error": self.visual_labeling_error,
            "visual_blink_count": self.latest_visual_blink_count,
            "shutdown_reason": shutdown_reason,
        }
        self.writer.write_metadata(payload)

    def run(self) -> Path:
        try:
            import cv2
            import sounddevice as sd
        except ImportError as exc:
            raise RuntimeError(
                "Missing dependency. Install with: python -m pip install -r requirements_hp_acoustic_wave.txt"
            ) from exc

        session_prefix = "hp_blink" if self.config.mode == "blink" else "hp_wave"
        self.session_dir = create_session_dir(self.config.session_root, prefix=session_prefix)
        self.writer = SessionWriter(self.session_dir, self.config.audio.sample_rate)
        self.writer.open()
        self.audio_device_selection = resolve_audio_device_selection(
            requested_input_device=self.config.audio.input_device,
            requested_output_device=self.config.audio.output_device,
            default_device=sd.default.device,
            devices=sd.query_devices(),
            hostapis=sd.query_hostapis(),
        )
        print(_safe_console_text(format_audio_device_selection(self.audio_device_selection)), flush=True)
        self._open_camera(cv2)
        self._open_visual_labeler()
        self._open_video_writer(cv2)

        self.running = True
        shutdown_reason = "completed"
        started_at = time.monotonic()
        device = None
        if self.config.audio.input_device is not None or self.config.audio.output_device is not None:
            device = (self.config.audio.input_device, self.config.audio.output_device)
        try:
            stream = sd.Stream(
                samplerate=self.config.audio.sample_rate,
                blocksize=self.config.audio.chunk_size,
                channels=1,
                dtype="float32",
                callback=self._audio_callback,
                device=device,
            )
            with stream:
                self.stream_started_monotonic = time.monotonic()
                cv2.namedWindow(self.config.window_name, cv2.WINDOW_NORMAL)
                while self.running:
                    self._process_audio_queue()
                    if self.camera_enabled:
                        ok, frame = self.camera.read()
                        if not ok:
                            frame = self._blank_frame(cv2)
                        else:
                            self._process_visual_frame(cv2, frame)
                    else:
                        frame = self._blank_frame(cv2)
                    canvas = self._draw_overlay(cv2, frame)
                    if self.video_writer is not None:
                        self.video_writer.write(canvas)
                    cv2.imshow(self.config.window_name, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        shutdown_reason = "user_quit"
                        self.running = False
                    elif key in (ord("m"), ord("b"), ord("w")):
                        self._write_marker_for_key(key)
                    if self.config.max_duration_s is not None and time.monotonic() - started_at >= self.config.max_duration_s:
                        shutdown_reason = "duration_elapsed"
                        self.running = False
                    time.sleep(0.005)
        except KeyboardInterrupt:
            shutdown_reason = "keyboard_interrupt"
        except Exception as exc:
            shutdown_reason = f"error: {exc}"
            raise
        finally:
            self._write_metadata(shutdown_reason)
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            if self.face_mesh is not None:
                self.face_mesh.close()
                self.face_mesh = None
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            if self.writer is not None:
                self.writer.close()
        return self.session_dir


def _format_metric(value) -> str:
    if value is None:
        return ""
    return f"{float(value):.9f}"


def _format_metric_tuple(values) -> str:
    if values is None:
        return ""
    return ";".join(
        "nan" if not np.isfinite(float(value)) else f"{float(value):.9f}"
        for value in values
    )


def _format_bool_metric(value) -> str:
    if value is None:
        return ""
    return str(int(bool(value)))


def _safe_console_text(text: str, encoding: Optional[str] = None) -> str:
    output_encoding = encoding or sys.stdout.encoding or "utf-8"
    return str(text).encode(output_encoding, errors="replace").decode(output_encoding, errors="replace")
