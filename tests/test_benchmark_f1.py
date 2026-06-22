from hp_acoustic_wave.benchmark import AcousticEvent, ManualMarker, score_events


def _ev(t, label=""):
    return AcousticEvent(time_s=t, event_id=0, method="", score=0.0, motion_energy=0.0, threshold=0.0, label=label)


def _mk(t, label="blink"):
    return ManualMarker(time_s=t, label=label)


def test_one_to_one_nearest_pairing_excludes_second_event_in_window():
    # marker at 1.0s; two events within [1.0, 1.5]: only the nearest counts as TP,
    # the other is FP under 1:1 pairing.
    events = [_ev(1.10), _ev(1.45)]
    markers = [_mk(1.0)]
    s = score_events(events, markers, match_before_s=0.0, match_after_s=0.5, one_to_one=True)
    assert s.tp == 1
    assert s.fp == 1
    assert s.fn == 0


def test_after_only_window_rejects_event_before_marker():
    # event slightly before marker must NOT be a TP (before-window is 0)
    events = [_ev(0.95)]
    markers = [_mk(1.0)]
    s = score_events(events, markers, match_before_s=0.0, match_after_s=0.5, one_to_one=True)
    assert s.tp == 0
    assert s.fn == 1
    assert s.fp == 1


def test_f1_harmonic_mean():
    # 2 TP, 2 FP, 1 FN -> P=0.5, R=0.667, F1 = 2*.5*.667/(.5+.667)=0.571
    events = [_ev(1.10), _ev(2.10), _ev(5.0), _ev(5.1)]
    markers = [_mk(1.0), _mk(2.0), _mk(3.0)]
    s = score_events(events, markers, match_before_s=0.0, match_after_s=0.5, one_to_one=True)
    assert s.tp == 2
    assert s.fp == 2
    assert s.fn == 1
    assert abs(s.precision - 0.5) < 1e-9
    assert abs(s.recall - (2.0 / 3.0)) < 1e-9
    assert abs(s.f1 - (2 * 0.5 * (2.0 / 3.0) / (0.5 + 2.0 / 3.0))) < 1e-9


def test_large_motion_not_counted_as_fp():
    # event labeled large_motion is excluded from FP (correctly classified non-blink)
    events = [_ev(1.10), _ev(2.0, label="large_motion")]
    markers = [_mk(1.0)]
    s = score_events(events, markers, match_before_s=0.0, match_after_s=0.5, one_to_one=True)
    assert s.tp == 1
    assert s.fp == 0
    assert s.large_motion_hits == 1


def test_zero_tp_gives_zero_f1():
    events = [_ev(100.0)]
    markers = [_mk(1.0)]
    s = score_events(events, markers, match_before_s=0.0, match_after_s=0.5, one_to_one=True)
    assert s.f1 == 0.0
    assert s.precision == 0.0
    assert s.recall == 0.0


def test_legacy_mode_two_events_one_marker_both_not_fp():
    # Legacy (one_to_one=False): two events fall in one marker's window.
    # Both events are NOT FP (they match a marker), marker counted as hit once.
    events = [_ev(1.10), _ev(1.45)]
    markers = [_mk(1.0)]
    s = score_events(events, markers, match_before_s=0.0, match_after_s=0.5, one_to_one=False)
    assert s.tp == 1       # marker hit once (ANY-event semantics)
    assert s.fp == 0       # neither event is FP — both match the marker
    assert s.fn == 0       # the single marker was hit
    assert s.precision == 1.0
    assert s.recall == 1.0
    assert s.f1 == 1.0


def test_legacy_mode_fp_only_for_events_matching_no_marker():
    # Legacy mode: an event far from any marker IS FP; an event near a marker is NOT.
    events = [_ev(1.10), _ev(50.0)]  # first matches marker, second does not
    markers = [_mk(1.0)]
    s = score_events(events, markers, match_before_s=0.0, match_after_s=0.5, one_to_one=False)
    assert s.tp == 1
    assert s.fp == 1       # only the far event is FP
    assert s.fn == 0
