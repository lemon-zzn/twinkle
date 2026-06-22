from hp_acoustic_wave.benchmark import reprocess_audio_feature_rows


def test_reprocess_accepts_emission_kwarg(tmp_path):
    # generate a tiny rx wav (silence is fine; we only test the function signature path)
    import numpy as np
    import wave
    p = tmp_path / "rx.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(48000)
        w.writeframes(np.zeros(48000, dtype=np.int16).tobytes())
    rows = reprocess_audio_feature_rows(
        audio_path=p,
        sample_rate=48000,
        tone_hz=18500.0,
        chunk_size=1024,
        tukey_alpha=0.0,
        signal_mode="fmcw",
        fmcw_emission="linear_tukey",
    )
    assert len(rows) > 0
    assert rows[0]["signal_mode"] == "fmcw"
