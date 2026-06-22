import pytest

from hp_acoustic_wave.visual_blink import (
    EAR_LEFT_EYE_IDXS,
    EAR_RIGHT_EYE_IDXS,
    VisualBlinkLabeler,
    calculate_ear,
)


def test_visual_labeler_emits_one_blink_after_consecutive_closed_frames():
    labeler = VisualBlinkLabeler(ear_threshold=0.22, consecutive_frames=3)
    events = []

    for index, ear in enumerate([0.30, 0.19, 0.18, 0.17, 0.16, 0.31, 0.30]):
        state = labeler.update(
            time_s=index * 0.05,
            left_ear=ear,
            right_ear=ear,
            face_present=True,
        )
        events.append(state.is_blink_event)

    assert events == [False, False, False, False, False, True, False]
    assert labeler.blink_count == 1


def test_visual_labeler_rejects_single_closed_frame_noise():
    labeler = VisualBlinkLabeler(ear_threshold=0.22, consecutive_frames=3)

    states = [
        labeler.update(time_s=index * 0.05, left_ear=ear, right_ear=ear, face_present=True)
        for index, ear in enumerate([0.30, 0.18, 0.29, 0.31])
    ]

    assert not any(state.is_blink_event for state in states)
    assert labeler.blink_count == 0


def test_calculate_ear_uses_mediapipe_eye_indices():
    landmarks = [(0.0, 0.0)] * 478
    for eye_indices in (EAR_LEFT_EYE_IDXS, EAR_RIGHT_EYE_IDXS):
        p1, p2, p3, p4, p5, p6 = eye_indices
        mutable = list(landmarks)
        mutable[p1] = (0.0, 0.0)
        mutable[p4] = (2.0, 0.0)
        mutable[p2] = (0.5, 0.4)
        mutable[p6] = (0.5, -0.4)
        mutable[p3] = (1.5, 0.3)
        mutable[p5] = (1.5, -0.3)

        assert calculate_ear(mutable, eye_indices) == pytest.approx(0.35)
