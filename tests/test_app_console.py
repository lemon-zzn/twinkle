from hp_acoustic_wave.app import RealtimeHandWaveApp, _safe_console_text
from hp_acoustic_wave.config import AppConfig, CameraConfig, DetectorConfig


def test_safe_console_text_replaces_characters_not_supported_by_target_encoding():
    text = "Realtek® microphone"

    safe = _safe_console_text(text, encoding="gbk")

    safe.encode("gbk")
    assert "Realtek" in safe


def test_blink_mode_uses_short_detection_display_hold():
    app = RealtimeHandWaveApp(
        AppConfig(
            camera=CameraConfig(enabled=False),
            detector=DetectorConfig(detection_hold_s=2.0),
            mode="blink",
        )
    )

    assert app._detection_display_hold_s() == 0.55


def test_wave_mode_keeps_configured_detection_display_hold():
    app = RealtimeHandWaveApp(
        AppConfig(
            camera=CameraConfig(enabled=False),
            detector=DetectorConfig(detection_hold_s=2.0),
            mode="wave",
        )
    )

    assert app._detection_display_hold_s() == 2.0
