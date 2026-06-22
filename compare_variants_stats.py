"""Print per-variant signal summary stats."""
from __future__ import annotations
import io
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from compare_variants_signal import (
    SESSIONS, SAMPLE_RATE, FREQ_LOW, FREQ_HIGH, CHIRP_DURATION,
    RANGE_BIN, TONE_HZ, CHUNK, load_mono_wav, load_visual_blinks, tx_for,
)
from hp_acoustic_wave.dsp import (
    extract_fmcw_chunk_feature, extract_chunk_feature, FmcwBackgroundSubtractor,
    generate_fmcw_chirp,
)


def main():
    print(f"{'variant':18s} {'n_blinks':>9s} {'amp_mean':>10s} {'amp_p2p':>9s} "
          f"{'dphase_q99':>11s} {'blink_dphase_med':>17s}")
    for label, sess_name, emission in SESSIONS:
        sess = REPO / "hp_acoustic_wave" / "sessions" / sess_name
        wav = load_mono_wav(sess / "audio.wav")
        blinks = load_visual_blinks(sess)

        n_frames = min(len(wav) // CHUNK, 800)
        times, amps, phases = [], [], []
        prev = None
        bg = FmcwBackgroundSubtractor(alpha=0.02) if emission != "cw_single" else None
        for i in range(n_frames):
            s = i * CHUNK
            chunk = wav[s:s + CHUNK]
            if emission == "cw_single":
                f = extract_chunk_feature(chunk, SAMPLE_RATE, TONE_HZ, s, prev)
            else:
                tx_chunk = generate_fmcw_chirp(
                    CHUNK, SAMPLE_RATE, FREQ_LOW, FREQ_HIGH, CHIRP_DURATION,
                    start_sample=s, amplitude=0.2, emission=emission, tukey_alpha=0.2,
                )
                f = extract_fmcw_chunk_feature(
                    chunk, tx_chunk, SAMPLE_RATE, FREQ_LOW, FREQ_HIGH,
                    CHIRP_DURATION, RANGE_BIN, s, prev, emission=emission,
                    background_subtractor=bg,
                )
            times.append(f.time_s)
            amps.append(f.amplitude)
            phases.append(f.phase)
            prev = f
        times = np.asarray(times)
        amps = np.asarray(amps)
        phases = np.unwrap(np.asarray(phases))
        dphase = np.diff(phases)

        blink_dphase = []
        for b in blinks:
            idx = np.searchsorted(times, b)
            lo = max(0, idx - 3)
            hi = min(len(dphase), idx + 4)
            blink_dphase.extend(dphase[lo:hi])
        blink_dphase_med = float(np.median(np.abs(blink_dphase))) if blink_dphase else float("nan")
        all_dphase_q99 = float(np.quantile(np.abs(dphase), 0.99))

        print(f"{label:18s} {len(blinks):>9d} {amps.mean():>10.4f} "
              f"{amps.max() - amps.min():>9.4f} {all_dphase_q99:>11.4f} "
              f"{blink_dphase_med:>17.4f}")


if __name__ == "__main__":
    main()
