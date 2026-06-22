from pathlib import Path

import numpy as np

from hp_acoustic_wave import fmcw_breathing as fmcw
from hp_acoustic_wave import fmcw_breathing_live as live


class FakeSoundDevice:
    def __init__(self):
        self.calls = []
        self.default = type("Default", (), {"samplerate": None, "channels": None})()

    def playrec(self, tx, samplerate=None, channels=None, device=None, blocking=None):
        self.calls.append(
            {
                "tx": np.asarray(tx),
                "samplerate": samplerate,
                "channels": channels,
                "device": device,
                "blocking": blocking,
            }
        )
        return np.asarray(tx, dtype=np.float32).reshape(-1, 1) * 0.5

    def wait(self):
        self.calls.append({"wait": True})


class FakeStream:
    def __init__(
        self,
        samplerate=None,
        blocksize=None,
        channels=None,
        dtype=None,
        callback=None,
        device=None,
        finished_callback=None,
    ):
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.dtype = dtype
        self.callback = callback
        self.device = device

    def __enter__(self):
        for start in range(0, 4 * self.blocksize, self.blocksize):
            indata = np.ones((self.blocksize, 1), dtype=np.float32) * 0.1
            outdata = np.zeros((self.blocksize, 1), dtype=np.float32)
            self.callback(indata, outdata, self.blocksize, None, "")
            assert np.max(np.abs(outdata)) > 0.0
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeStreamSoundDevice:
    Stream = FakeStream


class FakeLivePlotter:
    def __init__(self):
        self.updates = []
        self.closed = False

    def update(self, result):
        self.updates.append(result)

    def close(self):
        self.closed = True


def test_safe_console_text_replaces_unencodable_characters():
    assert live.safe_console_text("USB Audio \xae", encoding="ascii") == "USB Audio ?"


def test_print_start_signal_tells_user_recording_is_ready(capsys):
    live.print_start_signal()

    assert "可以开始了" in capsys.readouterr().out


def test_device_argument_is_tuple_only_when_a_device_is_requested():
    assert live.sounddevice_device_arg(None, None) is None
    assert live.sounddevice_device_arg(3, None) == (3, None)
    assert live.sounddevice_device_arg(None, 4) == (None, 4)
    assert live.sounddevice_device_arg(3, 4) == (3, 4)


def test_record_live_chirp_uses_sounddevice_and_returns_matching_shapes():
    cfg = fmcw.FmcwConfig(total_duration=0.1)
    fake_sd = FakeSoundDevice()

    recording = live.record_live_chirp(
        cfg,
        amplitude=0.25,
        input_device=1,
        output_device=2,
        sd_module=fake_sd,
    )

    assert recording.tx.shape == (cfg.samples_per_chirp * 2,)
    assert recording.rx.shape == (cfg.samples_per_chirp * 2, 1)
    assert fake_sd.calls[0]["samplerate"] == cfg.sample_rate
    assert fake_sd.calls[0]["channels"] == 1
    assert fake_sd.calls[0]["device"] == (1, 2)
    assert np.max(np.abs(recording.tx)) <= 0.25


def test_stream_live_chirp_records_callback_audio():
    cfg = fmcw.FmcwConfig(total_duration=0.2)

    recording = live.stream_live_chirp(
        cfg,
        amplitude=0.25,
        blocksize=cfg.samples_per_chirp,
        sd_module=FakeStreamSoundDevice(),
    )

    assert recording.tx.shape == (cfg.samples_per_chirp * 4,)
    assert recording.rx.shape == (cfg.samples_per_chirp * 4, 1)
    assert np.max(np.abs(recording.tx)) <= 0.25


def test_stream_live_chirp_updates_live_phase_plotter():
    cfg = fmcw.FmcwConfig(total_duration=0.2)
    plotter = FakeLivePlotter()

    recording = live.stream_live_chirp(
        cfg,
        amplitude=0.25,
        blocksize=cfg.samples_per_chirp,
        range_bin=2,
        live_plotter=plotter,
        plot_interval_s=0.0,
        sd_module=FakeStreamSoundDevice(),
    )

    assert recording.rx.shape == (cfg.samples_per_chirp * 4, 1)
    assert plotter.updates
    assert plotter.updates[-1].range_bin == 2
    assert plotter.updates[-1].unwrap_phase.shape == (4,)


def test_arg_parser_accepts_live_plot_flag():
    args = live.build_arg_parser().parse_args(["--live-plot", "--range-bin", "15"])

    assert args.live_plot is True
    assert args.range_bin == 15


def test_arg_parser_accepts_fmcw_emission_variant():
    args = live.build_arg_parser().parse_args(["--fmcw-emission", "linear_tukey"])

    assert args.fmcw_emission == "linear_tukey"


def test_breathing_chirp_train_preserves_hybrid_odd_cw_frame():
    cfg = fmcw.FmcwConfig(total_duration=0.1, emission="cw_fmcw_hybrid")

    samples = fmcw.generate_chirp_train(cfg)
    even = samples[: cfg.samples_per_chirp]
    odd = samples[cfg.samples_per_chirp : 2 * cfg.samples_per_chirp]

    assert not np.allclose(even, odd)


def test_save_live_recording_npz_round_trips(tmp_path: Path):
    recording = live.LiveRecording(
        tx=np.array([0.1, 0.2], dtype=np.float32),
        rx=np.array([[0.3], [0.4]], dtype=np.float32),
    )

    path = live.save_live_recording_npz(recording, tmp_path, "sample")

    data = np.load(path)
    np.testing.assert_allclose(data["tx"], recording.tx)
    np.testing.assert_allclose(data["rx"], recording.rx)
