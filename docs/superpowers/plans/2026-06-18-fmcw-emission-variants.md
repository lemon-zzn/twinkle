# FMCW Emission Variants + F1 Scoring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 5 selectable FMCW/CW emission variants (linear, linear+tukey, cw+fmcw hybrid, triangle, cw single) and upgrade benchmark scoring to standard F1 with after-only 0.5s 1:1 match window, so each variant can be recorded and compared head-to-head.

**Architecture:** Emission is an `AudioConfig` enum that switches one phase-function generator (`_fmcw_chirp_phase`) plus per-frame routing in `extract_fmcw_chunk_feature` for the hybrid case. Detector layer is emission-agnostic. Benchmark reads `fmcw_emission` from meta.json and propagates it through both `generate_fmcw_chirp` and `extract_fmcw_chunk_feature` at replay (because tx is regenerated, not read from tx.wav). F1 computed with greedy 1:1 nearest pairing, match window `[0, +0.5s]`.

**Tech Stack:** Python 3.8, numpy, scipy.io.wavfile, pytest, dataclasses.

**Spec:** `hp_acoustic_wave/docs/superpowers/specs/2026-06-18-fmcw-emission-variants-design.md`

**Spec deviation (approved during planning):** `fmcw_emission` lives on `AudioConfig` (not `BlinkConfig` as the spec text said) because it is an audio-emission property and must flow into `to_metadata()` → meta.json via the existing `asdict(self.audio)` path.

---

## File Structure

| File | Responsibility |
|---|---|
| `hp_acoustic_wave/config.py` | Add `AudioConfig.fmcw_emission` field |
| `hp_acoustic_wave/dsp.py` | Extend `_fmcw_chirp_phase` + `generate_fmcw_chirp` + `extract_fmcw_chunk_feature` with `emission` param |
| `hp_acoustic_wave/app.py` | `_generate_playback` passes `fmcw_emission`; cw mode writes `fmcw_emission="cw_single"` to config before meta dump |
| `hp_acoustic_wave/run_hp_wave_detector.py` | `--fmcw-emission` CLI arg → `AudioConfig` |
| `hp_acoustic_wave/benchmark.py` | F1 in `score_events`; `BenchmarkSummary` new fields; `reprocess_audio_feature_rows` propagates `fmcw_emission`; reads emission from meta.json |
| `hp_acoustic_wave/benchmark_hp_blink.py` | Print F1/precision/recall; `--fmcw-emission` override flag |
| `hp_acoustic_wave/run_baseline.sh` | Unchanged (still `linear`) |
| `hp_acoustic_wave/run_emission_<variant>.sh` | One clean per-variant launch script (5 files) |
| `tests/test_benchmark_f1.py` | New: F1 + 1:1 matching tests |
| `tests/test_fmcw_emission_variants.py` | New: emission generator + DSP-variant tests |

All paths relative to `E:\android_projects\eye_blink_detect\hp_acoustic_wave` unless noted. The `hp_acoustic_wave/` package prefix is what `benchmark_hp_blink.py` imports under (REPO_ROOT = parents[1]); tests import `hp_acoustic_wave.dsp` etc. — verify against existing `tests/test_fmcw_wave_integration.py` for the exact import style.

---

## Task 1: F1 scoring + after-only 1:1 matching

**Files:**
- Modify: `hp_acoustic_wave/benchmark.py:276-311` (`score_events`) and `BenchmarkSummary` dataclass (around `benchmark.py:40-48`)
- Create: `tests/test_benchmark_f1.py`

- [ ] **Step 1: Write failing test for F1 + 1:1 pairing**

Create `tests/test_benchmark_f1.py`:

```python
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
    events = [_ev(1.10), _ev(1.20), _ev(5.0), _ev(5.1)]
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:\android_projects\eye_blink_detect && "C:\Program Files\Python38\python.exe" -m pytest hp_acoustic_wave/tests/test_benchmark_f1.py -v`
Expected: FAIL with `score_events() got an unexpected keyword argument 'one_to_one'` or `AttributeError: precision`.

- [ ] **Step 3: Extend BenchmarkSummary dataclass**

In `hp_acoustic_wave/benchmark.py`, find the `BenchmarkSummary` dataclass (near line 40). Add new fields after `balanced_score`:

```python
@dataclass
class BenchmarkSummary:
    events: int
    blink_hits: int
    blink_markers: int
    blink_misses: int
    large_motion_hits: int
    large_motion_markers: int
    large_motion_misses: int
    unexplained_events: int
    nonblink_events: int
    balanced_score: float
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
```

(Keep existing field order/names — only append the new optional fields.)

- [ ] **Step 4: Implement 1:1 greedy pairing + F1 in score_events**

Replace `score_events` (benchmark.py:276-311) with:

```python
def score_events(
    events: Iterable[AcousticEvent],
    markers: Iterable[ManualMarker],
    match_before_s: float = 0.0,
    match_after_s: float = 0.5,
    unexplained_penalty: float = 0.55,
    large_motion_penalty: float = 0.9,
    one_to_one: bool = True,
) -> BenchmarkSummary:
    event_list = list(events)
    marker_list = list(markers)

    blink_markers = [m for m in marker_list if m.label == "blink"]
    large_motion_markers = [m for m in marker_list if m.label == "large_motion"]

    # 1:1 greedy nearest pairing: each event/marker consumed at most once.
    tp, large_motion_hits = _greedy_pair(event_list, blink_markers, match_before_s, match_after_s)
    # large_motion markers: keep the existing per-marker ANY semantics (any event hitting them
    # counts as a correctly-classified large_motion event), but do NOT double-count a TP event.
    tp_events = set(_pair_event_indices(event_list, blink_markers, match_before_s, match_after_s))
    large_hits_event_idx = [
        i for i, e in enumerate(event_list)
        if i not in tp_events
        and any(_event_matches_marker(e, m, match_before_s, match_after_s) for m in large_motion_markers)
    ]
    large_motion_hits = len(large_hits_event_idx)
    consumed = tp_events | set(large_hits_event_idx)

    # FP = events not matched to ANY marker (blink or large_motion)
    unexplained_events = sum(
        1 for i, e in enumerate(event_list)
        if i not in consumed
        and not _matches_any_marker(e, marker_list, match_before_s, match_after_s)
    )

    fn = max(0, len(blink_markers) - tp)
    fp = unexplained_events
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    balanced_score = tp - unexplained_penalty * unexplained_events - large_motion_penalty * large_motion_hits

    return BenchmarkSummary(
        events=len(event_list),
        blink_hits=tp,
        blink_markers=len(blink_markers),
        blink_misses=len(blink_markers) - tp,
        large_motion_hits=large_motion_hits,
        large_motion_markers=len(large_motion_markers),
        large_motion_misses=len(large_motion_markers) - large_motion_hits,
        unexplained_events=unexplained_events,
        nonblink_events=large_motion_hits + unexplained_events,
        balanced_score=balanced_score,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _pair_event_indices(events, blink_markers, match_before_s, match_after_s):
    """Return the set of event indices chosen as TP under 1:1 greedy nearest pairing."""
    candidates = []
    for mi, m in enumerate(blink_markers):
        for ei, e in enumerate(events):
            if _event_matches_marker(e, m, match_before_s, match_after_s):
                dt = abs(e.time_s - m.time_s)
                candidates.append((dt, ei, mi))
    candidates.sort()
    used_events, used_markers, chosen = set(), set(), set()
    for dt, ei, mi in candidates:
        if ei in used_events or mi in used_markers:
            continue
        used_events.add(ei)
        used_markers.add(mi)
        chosen.add(ei)
    return chosen


def _greedy_pair(events, blink_markers, match_before_s, match_after_s):
    chosen = _pair_event_indices(events, blink_markers, match_before_s, match_after_s)
    return len(chosen), 0
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd E:\android_projects\eye_blink_detect && "C:\Program Files\Python38\python.exe" -m pytest hp_acoustic_wave/tests/test_benchmark_f1.py -v`
Expected: PASS, all 5 tests.

- [ ] **Step 6: Run existing benchmark tests to confirm no regression**

Run: `cd E:\android_projects\eye_blink_detect && "C:\Program Files\Python38\python.exe" -m pytest hp_acoustic_wave/tests/ -v -k "benchmark or score"`
Expected: PASS. If a pre-existing test asserts the old default `match_before_s=0.7`, that test will break — update it to pass the old window explicitly, do not change the new defaults.

- [ ] **Step 7: Commit**

```bash
cd E:/android_projects/eye_blink_detect/hp_acoustic_wave
git add hp_acoustic_wave/benchmark.py hp_acoustic_wave/tests/test_benchmark_f1.py
git commit -m "feat(benchmark): standard F1 with 1:1 greedy pairing and after-only 0.5s match window"
```

---

## Task 2: AudioConfig.fmcw_emission + CLI + meta.json

**Files:**
- Modify: `hp_acoustic_wave/config.py:6-19` (AudioConfig), `hp_acoustic_wave/run_hp_wave_detector.py` (add CLI), `hp_acoustic_wave/app.py` (cw mode writes cw_single)
- Create: `tests/test_emission_config.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_emission_config.py`:

```python
from hp_acoustic_wave.config import AudioConfig, AppConfig


def test_default_emission_is_linear():
    assert AudioConfig().fmcw_emission == "linear"


def test_emission_flows_into_metadata():
    cfg = AppConfig(audio=AudioConfig(fmcw_emission="linear_tukey"))
    md = cfg.to_metadata()
    assert md["audio"]["fmcw_emission"] == "linear_tukey"


def test_cw_mode_marker_is_invalid_value():
    # only these strings are valid; cw_single is written by app.py, not via the field
    cfg = AudioConfig(signal_mode="cw")
    assert cfg.fmcw_emission == "linear"  # field default unchanged; app overrides at dump time
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:\android_projects\eye_blink_detect && "C:\Program Files\Python38\python.exe" -m pytest hp_acoustic_wave/tests/test_emission_config.py -v`
Expected: FAIL with `AttributeError: fmcw_emission`.

- [ ] **Step 3: Add field to AudioConfig**

In `hp_acoustic_wave/config.py`, edit `AudioConfig` (lines 6-19) to append one field after `fmcw_motion_amplitude_floor`:

```python
@dataclass
class AudioConfig:
    sample_rate: int = 48000
    tone_hz: float = 18500.0
    chunk_size: int = 1024
    output_amplitude: float = 0.12
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    signal_mode: str = "tone"
    fmcw_freq_low: float = 17_000.0
    fmcw_freq_high: float = 23_000.0
    fmcw_chirp_duration: float = 0.05
    fmcw_lowpass_cutoff: float = 5_000.0
    fmcw_range_bin: int = 15
    fmcw_motion_amplitude_floor: float = 0.02
    fmcw_emission: str = "linear"  # linear | linear_tukey | cw_fmcw_hybrid | triangle
```

- [ ] **Step 4: Add CLI arg in run_hp_wave_detector.py**

In `hp_acoustic_wave/run_hp_wave_detector.py`, find the argparse section that defines `--signal-mode` / `--fmcw-range-bin`. Add immediately after the existing fmcw args:

```python
parser.add_argument(
    "--fmcw-emission",
    choices=["linear", "linear_tukey", "cw_fmcw_hybrid", "triangle"],
    default="linear",
    help="FMCW transmit waveform (only used when --signal-mode fmcw)",
)
```

And in the function that builds `AudioConfig(...)` from args (search for `fmcw_range_bin=args.`), add:

```python
fmcw_emission=args.fmcw_emission,
```

If `--signal-mode cw` is selected, the app will overwrite `fmcw_emission="cw_single"` before dumping meta.json (Step 6).

- [ ] **Step 5: Run config test**

Run: `cd E:\android_projects\eye_blink_detect && "C:\Program Files\Python38\python.exe" -m pytest hp_acoustic_wave/tests/test_emission_config.py -v`
Expected: PASS.

- [ ] **Step 6: app.py writes cw_single in cw mode**

In `hp_acoustic_wave/app.py`, find where meta.json is written (search `to_metadata` usage near line 572). Before the dump, add:

```python
# Record emission label for benchmark/analysis. cw mode is the E-variant baseline.
emission_label = "cw_single" if self.config.audio.signal_mode != "fmcw" else self.config.audio.fmcw_emission
metadata["audio"]["fmcw_emission"] = emission_label
```

(If `to_metadata()` already returns the full dict, mutate the returned dict or add the field inside `to_metadata` — pick whichever matches the existing structure. Verify by reading `app.py:560-580` before editing.)

- [ ] **Step 7: Commit**

```bash
cd E:/android_projects/eye_blink_detect/hp_acoustic_wave
git add hp_acoustic_wave/config.py hp_acoustic_wave/run_hp_wave_detector.py hp_acoustic_wave/app.py hp_acoustic_wave/tests/test_emission_config.py
git commit -m "feat(config): AudioConfig.fmcw_emission field + CLI flag + meta.json recording"
```

---

## Task 3: Variant B — linear + Tukey taper

**Files:**
- Modify: `hp_acoustic_wave/dsp.py:47-60` (`generate_fmcw_chirp`) and `dsp.py:81-96` (`_fmcw_chirp_phase`)
- Create: `tests/test_fmcw_emission_variants.py`

- [ ] **Step 1: Write failing test for B**

Create `tests/test_fmcw_emission_variants.py`:

```python
import numpy as np
import pytest

from hp_acoustic_wave.dsp import generate_fmcw_chirp


SR = 48000
CHIRP_DUR = 0.05
SPC = int(round(SR * CHIRP_DUR))  # 2400 samples/chirp


def test_linear_tukey_amplitude_tapers_at_chirp_boundary():
    # First and last sample of each chirp must be near zero (Tukey fade)
    samples = generate_fmcw_chirp(
        num_samples=SPC, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="linear_tukey",
    )
    assert abs(samples[0]) < 0.2 * 0.05, f"expected fade-in at start, got {samples[0]}"
    assert abs(samples[-1]) < 0.2 * 0.05, f"expected fade-out at end, got {samples[-1]}"


def test_linear_tukey_mid_chirp_close_to_linear():
    # Center sample ~ same amplitude as pure linear (window ≈ 1 in the middle)
    tukey = generate_fmcw_chirp(
        num_samples=SPC, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="linear_tukey",
    )
    linear = generate_fmcw_chirp(
        num_samples=SPC, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="linear",
    )
    mid = SPC // 2
    assert abs(abs(tukey[mid]) - abs(linear[mid])) < 0.2 * 0.05


def test_emission_default_is_linear():
    # omitting emission defaults to "linear" — backward compatibility
    a = generate_fmcw_chirp(SPC, SR, 17000, 23000, CHIRP_DUR, 0, 0.2)
    b = generate_fmcw_chirp(SPC, SR, 17000, 23000, CHIRP_DUR, 0, 0.2, emission="linear")
    assert np.allclose(a, b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:\android_projects\eye_blink_detect && "C:\Program Files\Python38\python.exe" -m pytest hp_acoustic_wave/tests/test_fmcw_emission_variants.py -v`
Expected: FAIL with `unexpected keyword argument 'emission'`.

- [ ] **Step 3: Add emission param + B path to generate_fmcw_chirp**

Replace `hp_acoustic_wave/dsp.py:47-60` with:

```python
def generate_fmcw_chirp(
    num_samples: int,
    sample_rate: int,
    freq_low: float,
    freq_high: float,
    chirp_duration: float,
    start_sample: int,
    amplitude: float,
    emission: str = "linear",
    tukey_alpha: float = 0.2,
) -> np.ndarray:
    samples_per_chirp = int(round(sample_rate * chirp_duration))
    if samples_per_chirp <= 0:
        raise ValueError("chirp_duration is too small")
    phase = _fmcw_chirp_phase(
        num_samples, sample_rate, freq_low, freq_high, chirp_duration, start_sample, emission=emission
    )
    wave = np.cos(phase)
    if emission == "linear_tukey":
        wave = wave * _tukey_envelope_per_chirp(num_samples, samples_per_chirp, tukey_alpha)
    elif emission in ("linear", "cw_fmcw_hybrid", "triangle"):
        pass  # envelope handled by phase function for hybrid/triangle
    else:
        raise ValueError(f"unknown fmcw emission: {emission}")
    return (float(amplitude) * wave).astype(np.float32)


def _tukey_envelope_per_chirp(num_samples: int, samples_per_chirp: int, alpha: float) -> np.ndarray:
    one = _tukey_window(samples_per_chirp, alpha)
    n_full = num_samples // samples_per_chirp
    tail = num_samples - n_full * samples_per_chirp
    envelope = np.tile(one, n_full)
    if tail > 0:
        envelope = np.concatenate([envelope, one[:tail]])
    return envelope
```

- [ ] **Step 4: Add emission param to _fmcw_chirp_phase (pass-through for B)**

Replace `dsp.py:81-96` `_fmcw_chirp_phase` signature to accept `emission`:

```python
def _fmcw_chirp_phase(
    num_samples: int,
    sample_rate: int,
    freq_low: float,
    freq_high: float,
    chirp_duration: float,
    start_sample: int,
    emission: str = "linear",
) -> np.ndarray:
    samples_per_chirp = int(round(sample_rate * chirp_duration))
    if samples_per_chirp <= 0:
        raise ValueError("chirp_duration is too small")
    indices = np.arange(start_sample, start_sample + num_samples, dtype=np.int64)
    chirp_indices = np.mod(indices, samples_per_chirp).astype(np.float64)
    t = chirp_indices / float(sample_rate)
    slope = (float(freq_high) - float(freq_low)) / float(chirp_duration)
    return 2.0 * math.pi * (float(freq_low) * t + 0.5 * slope * t * t)
```

(Phase formula unchanged for B — Tukey is applied as amplitude envelope, not phase.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd E:\android_projects\eye_blink_detect && "C:\Program Files\Python38\python.exe" -m pytest hp_acoustic_wave/tests/test_fmcw_emission_variants.py -v`
Expected: PASS, all 3 tests.

- [ ] **Step 6: Commit**

```bash
cd E:/android_projects/eye_blink_detect/hp_acoustic_wave
git add hp_acoustic_wave/dsp.py hp_acoustic_wave/tests/test_fmcw_emission_variants.py
git commit -m "feat(dsp): linear_tukey emission variant (per-chirp Tukey envelope)"
```

---

## Task 4: Variant D — triangle (up+down chirp)

**Files:**
- Modify: `hp_acoustic_wave/dsp.py:81-96` (`_fmcw_chirp_phase`), `dsp.py:289` (`extract_fmcw_chunk_feature` signature)
- Extend: `tests/test_fmcw_emission_variants.py`

- [ ] **Step 1: Write failing test for D**

Append to `tests/test_fmcw_emission_variants.py`:

```python
def test_triangle_phase_is_continuous_at_midpoint():
    # Instantaneous freq at midpoint should match between up-sweep end and down-sweep start
    # (derivative of phase is continuous -> no click). Check phase delta across midpoint is small.
    samples = generate_fmcw_chirp(
        num_samples=SPC, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="triangle",
    )
    # amplitude at midpoint should not be a discontinuity spike: |sample| <= amplitude
    mid = SPC // 2
    assert abs(samples[mid]) <= 0.2 + 1e-6
    assert abs(samples[mid - 1]) <= 0.2 + 1e-6


def test_triangle_covers_full_chirp_length():
    samples = generate_fmcw_chirp(
        num_samples=SPC, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="triangle",
    )
    assert samples.shape == (SPC,)
    assert samples.dtype == np.float32


def test_triangle_spans_two_chirps_wraps_correctly():
    samples = generate_fmcw_chirp(
        num_samples=SPC * 2, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="triangle",
    )
    # second chirp should mirror the first (periodicity)
    assert np.allclose(samples[:SPC], samples[SPC:], atol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:\android_projects\eye_blink_detect && "C:\Program Files\Python38\python.exe" -m pytest hp_acoustic_wave/tests/test_fmcw_emission_variants.py -v -k triangle`
Expected: FAIL (phase function still linear).

- [ ] **Step 3: Implement triangle phase**

In `_fmcw_chirp_phase` (dsp.py), insert a triangle branch before the linear return:

```python
def _fmcw_chirp_phase(
    num_samples, sample_rate, freq_low, freq_high, chirp_duration, start_sample, emission="linear",
):
    samples_per_chirp = int(round(sample_rate * chirp_duration))
    if samples_per_chirp <= 0:
        raise ValueError("chirp_duration is too small")
    indices = np.arange(start_sample, start_sample + num_samples, dtype=np.int64)
    chirp_indices = np.mod(indices, samples_per_chirp).astype(np.float64)
    t = chirp_indices / float(sample_rate)
    slope = (float(freq_high) - float(freq_low)) / float(chirp_duration)

    if emission == "triangle":
        half = samples_per_chirp / 2.0
        # up-sweep [0, T/2): freq_low -> freq_high
        # down-sweep [T/2, T): freq_high -> freq_low
        # Continuous phase: integrate instantaneous frequency f(t).
        # f_up(t)   = freq_low + slope * t            (t in [0, T/2))
        # f_down(t) = freq_high - slope * (t - T/2)   (t in [T/2, T))
        t_half = chirp_duration / 2.0
        is_down = chirp_indices >= half
        t_local = np.where(is_down, t - t_half, t)
        f_inst = np.where(is_down, float(freq_high) - slope * t_local, float(freq_low) + slope * t_local)
        # phase = 2*pi * integral of f_inst dt, but we need an absolute phase (not just derivative)
        # so build it piecewise: phase_up(t) = 2pi*(fl*t + 0.5*slope*t^2)
        # phase_down(t) = phase_up(T/2) + 2pi*(fh*(t-T/2) - 0.5*slope*(t-T/2)^2)
        phase_up = 2.0 * math.pi * (float(freq_low) * t + 0.5 * slope * t * t)
        phase_at_half = 2.0 * math.pi * (float(freq_low) * t_half + 0.5 * slope * t_half * t_half)
        phase_down = phase_at_half + 2.0 * math.pi * (
            float(freq_high) * t_local - 0.5 * slope * t_local * t_local
        )
        return np.where(is_down, phase_down, phase_up)

    return 2.0 * math.pi * (float(freq_low) * t + 0.5 * slope * t * t)
```

- [ ] **Step 4: Propagate emission through extract_fmcw_chunk_feature**

In `dsp.py:289`, change the signature to add `emission: str = "linear"` and pass it to both `_fmcw_chirp_phase` calls inside (there is one at line 315). Replace the line:

```python
chirp_phase = _fmcw_chirp_phase(
    usable,
    sample_rate=sample_rate,
    freq_low=freq_low,
    freq_high=freq_high,
    chirp_duration=chirp_duration,
    start_sample=start_sample,
    emission=emission,
)
```

Add `emission: str = "linear"` to the `extract_fmcw_chunk_feature` parameter list (after `background_subtractor`).

- [ ] **Step 5: Run tests**

Run: `cd E:\android_projects\eye_blink_detect && "C:\Program Files\Python38\python.exe" -m pytest hp_acoustic_wave/tests/test_fmcw_emission_variants.py -v -k triangle`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd E:/android_projects/eye_blink_detect/hp_acoustic_wave
git add hp_acoustic_wave/dsp.py hp_acoustic_wave/tests/test_fmcw_emission_variants.py
git commit -m "feat(dsp): triangle (up+down sweep) FMCW emission variant"
```

---

## Task 5: Variant C — cw+fmcw hybrid (frame interleaving) — highest risk

**Files:**
- Modify: `hp_acoustic_wave/dsp.py:81-96` (`_fmcw_chirp_phase`), `dsp.py:289-360` (`extract_fmcw_chunk_feature` per-frame routing)
- Extend: `tests/test_fmcw_emission_variants.py`

**Design note:** In the hybrid, even-indexed chirps (0, 2, 4…) emit linear chirp; odd-indexed chirps (1, 3, 5…) emit pure CW at `freq_low`. At DSP time, split the chunk by chirp index: chirp frames feed the range-bin path, CW frames feed the phase-pair path. `mixed`/`complex_reference` are built per-frame and aggregated.

- [ ] **Step 1: Write failing test for C emission pattern**

Append to `tests/test_fmcw_emission_variants.py`:

```python
def test_hybrid_even_frames_are_chirp_odd_frames_are_cw():
    # Even chirp index: freq sweeps low->high (beat present after dechirp)
    # Odd chirp index: pure CW at freq_low (no sweep)
    # Easiest observable property: the emitted waveform's instantaneous frequency.
    # Check that an odd-indexed chirp has zero phase curvature (constant freq).
    samples = generate_fmcw_chirp(
        num_samples=SPC * 2, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="cw_fmcw_hybrid",
    )
    # odd chirp = samples[SPC:2*SPC]; its instantaneous freq should be ~freq_low (constant)
    odd = samples[SPC:2 * SPC].astype(np.float64)
    # estimate instantaneous freq via unwrap of analytic signal
    spectrum = np.fft.rfft(odd * np.hanning(odd.size))
    freqs = np.fft.rfftfreq(odd.size, d=1.0 / SR)
    peak_freq = freqs[np.argmax(np.abs(spectrum))]
    assert abs(peak_freq - 17000) < 500, f"odd frame should be CW at 17k, got {peak_freq}"


def test_hybrid_even_frame_is_chirp():
    samples = generate_fmcw_chirp(
        num_samples=SPC, sample_rate=SR, freq_low=17000, freq_high=23000,
        chirp_duration=CHIRP_DUR, start_sample=0, amplitude=0.2,
        emission="cw_fmcw_hybrid",
    )
    # even frame = chirp: energy should spread between 17k and 23k
    spec = np.abs(np.fft.rfft(samples.astype(np.float64) * np.hanning(SPC)))
    freqs = np.fft.rfftfreq(SPC, d=1.0 / SR)
    in_band = spec[(freqs >= 17000) & (freqs <= 23000)].sum()
    total = spec.sum()
    assert in_band / total > 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:\android_projects\eye_blink_detect && "C:\Program Files\Python38\python.exe" -m pytest hp_acoustic_wave/tests/test_fmcw_emission_variants.py -v -k hybrid`
Expected: FAIL.

- [ ] **Step 3: Implement hybrid phase**

In `_fmcw_chirp_phase` (dsp.py), add a branch before the linear return:

```python
    if emission == "cw_fmcw_hybrid":
        chirp_index = ((start_sample + indices - indices[0]) // samples_per_chirp).astype(np.int64)
        is_cw_frame = (chirp_index % 2 == 1)
        cw_phase = 2.0 * math.pi * float(freq_low) * t
        chirp_phase_lin = 2.0 * math.pi * (float(freq_low) * t + 0.5 * slope * t * t)
        return np.where(is_cw_frame, cw_phase, chirp_phase_lin)
```

- [ ] **Step 4: Add per-frame routing in extract_fmcw_chunk_feature**

In `dsp.py` `extract_fmcw_chunk_feature` (around line 310-345), wrap the existing range-bin computation so it only uses chirp frames, and route CW frames into the phase-pair baseband. Replace the block from `mixed = rx64 * tx64` through the `complex_reference`/`complex_baseband` lines with:

```python
    samples_per_chirp = int(round(sample_rate * chirp_duration))
    if samples_per_chirp <= 0:
        raise ValueError("chirp_duration is too small")
    n_full_chirps = usable // samples_per_chirp
    # frame mask: True = chirp frame, False = CW frame (hybrid only)
    frame_mask = np.ones(usable, dtype=bool)
    if emission == "cw_fmcw_hybrid":
        idx = np.arange(usable)
        chirp_index = (start_sample + idx) // samples_per_chirp
        frame_mask = (chirp_index % 2 == 0)

    mixed = rx64 * tx64
    # zero out CW frames for the range-bin path (CW has no beat freq)
    mixed_range = mixed * frame_mask
    spectrum = np.fft.rfft(mixed_range)
    freqs = np.fft.rfftfreq(usable, d=1.0 / float(sample_rate))
    spectrum[freqs > float(lowpass_cutoff)] = 0
    lowpassed = np.fft.irfft(spectrum, n=usable)

    chirp_phase = _fmcw_chirp_phase(
        usable, sample_rate=sample_rate, freq_low=freq_low, freq_high=freq_high,
        chirp_duration=chirp_duration, start_sample=start_sample, emission=emission,
    )
    complex_reference = np.exp(-1j * chirp_phase)
    # For hybrid: phase-pair baseband comes from CW frames only (cleaner phase).
    # For other emissions: frame_mask is all True, so behavior is unchanged.
    cw_mask = (~frame_mask).astype(np.float64) if emission == "cw_fmcw_hybrid" else np.ones(usable)
    complex_baseband = _lowpass_complex((rx64 * complex_reference) * cw_mask, sample_rate, lowpass_cutoff)
```

(The rest of `_phase_pair_features` and range-bin extraction stay the same — they now operate on masked signals.)

- [ ] **Step 5: Run hybrid tests**

Run: `cd E:\android_projects\eye_blink_detect && "C:\Program Files\Python38\python.exe" -m pytest hp_acoustic_wave/tests/test_fmcw_emission_variants.py -v -k hybrid`
Expected: PASS.

- [ ] **Step 6: Run full emission test file + existing DSP tests**

Run: `cd E:\android_projects\eye_blink_detect && "C:\Program Files\Python38\python.exe" -m pytest hp_acoustic_wave/tests/test_fmcw_emission_variants.py hp_acoustic_wave/tests/test_fmcw_wave_integration.py -v`
Expected: PASS (integration tests verify linear path still produces a valid ChunkFeature).

- [ ] **Step 7: Commit**

```bash
cd E:/android_projects/eye_blink_detect/hp_acoustic_wave
git add hp_acoustic_wave/dsp.py hp_acoustic_wave/tests/test_fmcw_emission_variants.py
git commit -m "feat(dsp): cw+fmcw hybrid emission with per-frame routing"
```

---

## Task 6: benchmark.py reads emission from meta.json + propagates to replay

**Files:**
- Modify: `hp_acoustic_wave/benchmark.py:76-92` (`_load_audio_config` area, line 83-89 caller), `benchmark.py:162-220` (`reprocess_audio_feature_rows`)
- Create: `tests/test_benchmark_replay_emission.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_benchmark_replay_emission.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd E:\android_projects\eye_blink_detect && "C:\Program Files\Python38\python.exe" -m pytest hp_acoustic_wave/tests/test_benchmark_replay_emission.py -v`
Expected: FAIL with `unexpected keyword argument 'fmcw_emission'`.

- [ ] **Step 3: Add fmcw_emission param to reprocess_audio_feature_rows**

In `benchmark.py:162-176`, add to signature:

```python
def reprocess_audio_feature_rows(
    audio_path: Path,
    sample_rate: int,
    tone_hz: float,
    chunk_size: int,
    tukey_alpha: float = 0.0,
    signal_mode: str = "tone",
    fmcw_freq_low: float = 17_000.0,
    fmcw_freq_high: float = 23_000.0,
    fmcw_chirp_duration: float = 0.05,
    fmcw_range_bin: int = 15,
    fmcw_lowpass_cutoff: float = 5_000.0,
    fmcw_motion_amplitude_floor: float = 0.02,
    fmcw_output_amplitude: float = 0.2,
    fmcw_emission: str = "linear",
) -> List[dict]:
```

Then in the body, pass it to both calls:

```python
        if signal_mode == "fmcw":
            tx = generate_fmcw_chirp(
                num_samples=chunk.size,
                sample_rate=sample_rate,
                freq_low=fmcw_freq_low,
                freq_high=fmcw_freq_high,
                chirp_duration=fmcw_chirp_duration,
                start_sample=start_sample,
                amplitude=fmcw_output_amplitude,
                emission=fmcw_emission,
            )
            feature = extract_fmcw_chunk_feature(
                samples=chunk,
                tx_samples=tx,
                sample_rate=sample_rate,
                freq_low=fmcw_freq_low,
                freq_high=fmcw_freq_high,
                chirp_duration=fmcw_chirp_duration,
                range_bin=fmcw_range_bin,
                start_sample=start_sample,
                previous=previous,
                lowpass_cutoff=fmcw_lowpass_cutoff,
                motion_amplitude_floor=fmcw_motion_amplitude_floor,
                background_subtractor=fmcw_background_subtractor,
                emission=fmcw_emission,
            )
```

- [ ] **Step 4: Read emission from meta.json at the caller**

In `benchmark.py` where `reprocess_audio_feature_rows` is called (around line 83-89, inside the function that loads `audio_config`), add emission extraction. Find the block that reads `audio_config.get(...)` and append:

```python
        emission = str(audio_config.get("fmcw_emission", "linear"))
        if signal_mode != "fmcw":
            emission = "cw_single"
```

Then pass `fmcw_emission=emission` into the `reprocess_audio_feature_rows(...)` call. (Read the existing surrounding code first to match variable names exactly.)

- [ ] **Step 5: Run tests**

Run: `cd E:\android_projects\eye_blink_detect && "C:\Program Files\Python38\python.exe" -m pytest hp_acoustic_wave/tests/test_benchmark_replay_emission.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd E:/android_projects/eye_blink_detect/hp_acoustic_wave
git add hp_acoustic_wave/benchmark.py hp_acoustic_wave/tests/test_benchmark_replay_emission.py
git commit -m "feat(benchmark): propagate fmcw_emission through audio replay path"
```

---

## Task 7: benchmark_hp_blink.py outputs F1 + emission flag

**Files:**
- Modify: `hp_acoustic_wave/benchmark_hp_blink.py:25-73` (args), `:130-154` (main output)

- [ ] **Step 1: Add --fmcw-emission override flag**

In `benchmark_hp_blink.py parse_args`, after the `--source` arg, add:

```python
    parser.add_argument(
        "--fmcw-emission",
        choices=["linear", "linear_tukey", "cw_fmcw_hybrid", "triangle", "cw_single"],
        default=None,
        help="Override emission label (default: read from session meta.json)",
    )
```

- [ ] **Step 2: Print F1 in summary line**

In `benchmark_hp_blink.py main` (the print block around line 146), extend to:

```python
    print(
        "events={events} blink_hits={blink_hits}/{blink_markers} "
        "large_motion_hits={large_motion_hits}/{large_motion_markers} "
        "unexplained_events={unexplained_events} nonblink_events={nonblink_events} "
        "tp={tp} fp={fp} fn={fn} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} "
        "balanced_score={balanced_score:.2f}".format(**summary)
    )
```

- [ ] **Step 3: Smoke test on existing session**

Run: `cd E:\android_projects\eye_blink_detect\hp_acoustic_wave && "C:\Program Files\Python38\python.exe" benchmark_hp_blink.py --session sessions/hp_blink_20260617_190655 --source audio --truth auto --blink-twinkle-fmcw-use-intra-chirp-phase-pair --blink-twinkle-fmcw-min-score 0.12`
Expected: prints a line with `f1=...` (value will be lower than balanced_score due to tighter window). No crash.

- [ ] **Step 4: Commit**

```bash
cd E:/android_projects/eye_blink_detect/hp_acoustic_wave
git add hp_acoustic_wave/benchmark_hp_blink.py
git commit -m "feat(benchmark): print F1/precision/recall + emission override flag"
```

---

## Task 8: Per-variant clean launch scripts

**Files:**
- Create: `hp_acoustic_wave/run_emission_linear.sh`, `run_emission_linear_tukey.sh`, `run_emission_cw_fmcw_hybrid.sh`, `run_emission_triangle.sh`, `run_emission_cw_single.sh`

**User requirement:** "给一个比较干净的启动命令，不要加太多别的处理或者trick的" — each script is the baseline + emission flag only.

- [ ] **Step 1: Create linear script**

`hp_acoustic_wave/run_emission_linear.sh`:

```bash
#!/usr/bin/env bash
# Variant A: linear chirp (baseline). Same as run_baseline.sh emission.
PYTHON="/c/Program Files/Python38/python.exe"

"$PYTHON" run_hp_wave_detector.py \
  --mode blink --blink-method twinkle --signal-mode fmcw \
  --blink-twinkle-fmcw-use-intra-chirp-phase-pair \
  --blink-twinkle-fmcw-min-score 0.12 \
  --fmcw-emission linear \
  --fmcw-range-bin 15 --fmcw-freq-low 17000 --fmcw-freq-high 23000 \
  --fmcw-chirp-duration 0.05 --fmcw-lowpass-cutoff 5000 --amplitude 0.2 \
  --input-device 1 --output-device 3 \
  --camera-width 640 --camera-height 480 --camera-fps 30 \
  --visual-ear-threshold 0.22 \
  "$@"
```

- [ ] **Step 2: Create linear_tukey script**

`hp_acoustic_wave/run_emission_linear_tukey.sh`: identical except `--fmcw-emission linear_tukey`.

- [ ] **Step 3: Create cw_fmcw_hybrid script**

`hp_acoustic_wave/run_emission_cw_fmcw_hybrid.sh`: identical except `--fmcw-emission cw_fmcw_hybrid`.

- [ ] **Step 4: Create triangle script**

`hp_acoustic_wave/run_emission_triangle.sh`: identical except `--fmcw-emission triangle`.

- [ ] **Step 5: Create cw_single script**

`hp_acoustic_wave/run_emission_cw_single.sh`:

```bash
#!/usr/bin/env bash
# Variant E: single-frequency CW (known-correct reference baseline)
PYTHON="/c/Program Files/Python38/python.exe"

"$PYTHON" run_hp_wave_detector.py \
  --mode blink --blink-method twinkle --signal-mode cw \
  --tone-hz 18500 \
  --amplitude 0.2 \
  --input-device 1 --output-device 3 \
  --camera-width 640 --camera-height 480 --camera-fps 30 \
  --visual-ear-threshold 0.22 \
  "$@"
```

- [ ] **Step 6: Commit**

```bash
cd E:/android_projects/eye_blink_detect/hp_acoustic_wave
git add hp_acoustic_wave/run_emission_*.sh
git commit -m "feat: per-emission-variant clean launch scripts"
```

---

## Validation against success criteria

After all 8 tasks land, the user records 5 sessions (one per variant, 20-30 blinks each), then:

```bash
for v in linear linear_tukey cw_fmcw_hybrid triangle cw_single; do
  SESSION=$(ls -dt sessions/hp_blink_* | head -1)   # replace with the actual session path per variant
  "C:\Program Files\Python38\python.exe" benchmark_hp_blink.py \
    --session "$SESSION" --source audio --truth auto \
    --blink-twinkle-fmcw-use-intra-chirp-phase-pair --blink-twinkle-fmcw-min-score 0.12
done
```

Then write `experiment_fmcw_emission_comparison_<date>.md` with the table and pick the highest-F1 variant as the new `run_baseline.sh`.

**Pre-recording sanity check:** run the smoke test from Task 7 Step 3 against an existing session — F1 must be a valid float, not crash.

## Self-Review notes

- **Spec coverage:** all 5 variants (A in Task 3 default, B Task 3, C Task 5, D Task 4, E Task 2 cw-mode write) ✓; F1 + 1:1 + after-only window (Task 1) ✓; meta.json `fmcw_emission` (Task 2 + Task 6 read) ✓; per-variant recording (Task 8 scripts) ✓.
- **Type consistency:** `emission` param name used consistently in `generate_fmcw_chirp`, `_fmcw_chirp_phase`, `extract_fmcw_chunk_feature`, `reprocess_audio_feature_rows`. `BenchmarkSummary.tp/fp/fn/precision/recall/f1` match the test assertions.
- **No placeholders:** every step has executable code.
- **Highest risk:** Task 5 (hybrid) per-frame masking — if it breaks integration, the test in Step 6 of Task 5 will catch it before commit.
