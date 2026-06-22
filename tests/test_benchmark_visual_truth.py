from hp_acoustic_wave.benchmark import benchmark_session
from hp_acoustic_wave.blink_detector import BlinkDetectionConfig


def test_benchmark_session_scores_blink_candidates_against_visual_events(tmp_path):
    (tmp_path / "events.csv").write_text(
        "event_id,time_s,label,method,score,motion_energy,threshold\n"
        "1,1.000000,visual_blink,mediapipe_ear,0.310000000,0.000000000,0.220000000\n"
        "1,1.200000,blink_candidate,twinkle,0.300000000,0.020000000,0.090000000\n"
        "2,3.000000,visual_blink,mediapipe_ear,0.320000000,0.000000000,0.220000000\n"
        "2,5.000000,blink_candidate,twinkle,0.300000000,0.020000000,0.090000000\n",
        encoding="utf-8",
    )

    run = benchmark_session(
        tmp_path,
        BlinkDetectionConfig(method="twinkle"),
        source="events",
        truth="visual",
    )

    assert run.truth == "visual"
    assert run.summary.events == 2
    assert run.summary.blink_markers == 2
    assert run.summary.blink_hits == 1
    assert run.summary.unexplained_events == 1


def test_benchmark_session_auto_truth_prefers_visual_events(tmp_path):
    (tmp_path / "manual_markers.csv").write_text(
        "time_s,label,key,event_id,amplitude,phase,motion_energy\n"
        "9.000000,blink,b,0,,,,\n",
        encoding="utf-8",
    )
    (tmp_path / "events.csv").write_text(
        "event_id,time_s,label,method,score,motion_energy,threshold\n"
        "1,1.000000,visual_blink,mediapipe_ear,0.310000000,0.000000000,0.220000000\n"
        "1,1.100000,blink_candidate,twinkle,0.300000000,0.020000000,0.090000000\n",
        encoding="utf-8",
    )

    run = benchmark_session(
        tmp_path,
        BlinkDetectionConfig(method="twinkle"),
        source="events",
        truth="auto",
    )

    assert run.truth == "visual"
    assert run.summary.blink_markers == 1
    assert run.summary.blink_hits == 1
