from hp_acoustic_wave.session_io import FEATURE_FIELDS, VISUAL_LABEL_FIELDS, SessionWriter


def test_session_writer_accepts_fmcw_twinkle_feature_columns(tmp_path):
    writer = SessionWriter(tmp_path, sample_rate=48_000)
    writer.open()
    try:
        row = {field: "" for field in FEATURE_FIELDS}
        row.update(
            {
                "twinkle_fmcw_range_bin": "15.000000000",
                "twinkle_fmcw_phase_score": "0.120000000",
                "twinkle_fmcw_amplitude_ok": "1.000000000",
                "twinkle_fmcw_amplitude_stable": "1.000000000",
                "twinkle_fmcw_amplitude_delta_ratio": "0.030000000",
                "twinkle_effective_refractory_s": "1.700000000",
                "phase_pair_delta": "0.123000000",
            }
        )

        writer.write_feature(row)
    finally:
        writer.close()


def test_session_writer_records_visual_blink_ground_truth_rows(tmp_path):
    writer = SessionWriter(tmp_path, sample_rate=48_000)
    writer.open()
    try:
        writer.write_visual_label(
            time_s=1.25,
            face_present=True,
            left_ear=0.18,
            right_ear=0.19,
            mean_ear=0.185,
            is_closed=True,
            closed_frames=3,
            is_blink_event=False,
            blink_count=0,
        )
    finally:
        writer.close()

    rows = (tmp_path / "visual_labels.csv").read_text(encoding="utf-8").splitlines()

    assert rows[0] == ",".join(VISUAL_LABEL_FIELDS)
    assert rows[1].startswith("1.250000,1,0.180000000,0.190000000,0.185000000,1,3,0,0")
