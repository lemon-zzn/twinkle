from hp_acoustic_wave.config import AudioConfig, AppConfig


def test_default_emission_is_linear():
    assert AudioConfig().fmcw_emission == "linear"


def test_emission_flows_into_metadata():
    cfg = AppConfig(audio=AudioConfig(signal_mode="fmcw", fmcw_emission="linear_tukey"))
    md = cfg.to_metadata()
    assert md["audio"]["fmcw_emission"] == "linear_tukey"


def test_cw_mode_emission_field_unchanged_at_config_level():
    # field default is unchanged regardless of signal_mode; app.py writes cw_single at dump time
    cfg = AudioConfig(signal_mode="cw")
    assert cfg.fmcw_emission == "linear"


def test_cw_signal_mode_metadata_reports_cw_single():
    cfg = AppConfig(audio=AudioConfig(signal_mode="cw"))
    md = cfg.to_metadata()
    assert md["audio"]["fmcw_emission"] == "cw_single"


def test_fmcw_signal_mode_metadata_preserves_emission():
    cfg = AppConfig(audio=AudioConfig(signal_mode="fmcw", fmcw_emission="triangle"))
    md = cfg.to_metadata()
    assert md["audio"]["fmcw_emission"] == "triangle"
