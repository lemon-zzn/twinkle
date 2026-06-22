# FMCW Blink Detector Redesign — Shape-Segmentation + BlinkListener LEVD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current peak + coherence + refractory blink detector (noise-driven periodic false positives) with two algorithm cores — Twinkle paper's shape-based segmentation and BlinkListener's LEVD bump detection — and verify on 9 recorded sessions that periodic firing is eliminated.

**Architecture:** Five phases in sequence. Phase 0 deletes dead code and min-score gates (foundation). Phase 1 splits the realtime waveform into phase trajectory + ungated coherence curves. Phase 2 adds `_TwinkleShapeSegmentationDetector` (pattern matching on phase trajectory). Phase 3 audits the existing `BlinkListenerBlinkDetector` and wires the new methods through CLI. Phase 4 benchmarks all three detectors on 9 sessions with an anti-periodic cluster-ratio metric.

**Tech Stack:** Python 3.8, numpy, OpenCV, existing `hp_acoustic_wave/{blink_detector,dsp,app,config}.py` framework. Python interpreter is `/c/Program Files/Python38/python.exe`. Tests use `pytest`. All test/benchmark commands run from the repo root `E:/android_projects/eye_blink_detect` so `from hp_acoustic_wave.X` imports resolve.

**Repo layout:**
- `hp_acoustic_wave/blink_detector.py` — detectors (`BlinkDetectionConfig`, `BlinkDetectionResult`, `_RobustEventGate`, `_TwinklePeakEventGate`, `BlinkListenerBlinkDetector`, `TwinkleTwinkleBlinkDetector`, `CompositeBlinkDetector`, `build_blink_detector`)
- `hp_acoustic_wave/config.py` — public config dataclasses (`BlinkConfig` is source of truth consumed by `app.py`)
- `hp_acoustic_wave/dsp.py` — `ChunkFeature` dataclass + FMCW/tone feature extraction (already populates `i_value`, `q_value`, `phase_pair_delta`)
- `hp_acoustic_wave/app.py` — realtime visualization
- `hp_acoustic_wave/run_hp_wave_detector.py`, `hp_acoustic_wave/benchmark_hp_blink.py` — CLIs
- `hp_acoustic_wave/run_emission_*.sh` — 5 launch scripts
- `hp_acoustic_wave/sessions/hp_blink_*` — recorded sessions
- `hp_acoustic_wave/tests/test_*.py` — pytest tests

**Hard requirements (from `hp_acoustic_wave/CLAUDE.md`):**
- No periodic false-positive firing ("周期猜测 blink"): event gaps must not cluster at the refractory period.
- On session 214655: events < 20 (baseline regression was 363).
- On sessions 184117 and 190655: `balanced_score >= 0`.
- Threshold must not lock at `min_score`.
- Test only 9 sessions: 184117, 190655, 214655 (CLAUDE.md) + 162711, 172844, 173242, 173057, 173428, 174829 (baseline doc).

---

## File Structure

| File | Phase | Responsibility |
|---|---|---|
| `hp_acoustic_wave/blink_detector.py` | 0,2,3 | Delete dead code + min-score gates; add shape detector; audit blinklistener |
| `hp_acoustic_wave/config.py` | 0,2 | Delete min-score fields; add shape params |
| `hp_acoustic_wave/run_hp_wave_detector.py` | 0,2,3 | Delete min-score CLI args; extend `--blink-method` choices |
| `hp_acoustic_wave/benchmark_hp_blink.py` | 0,2,4 | Delete min-score CLI args; extend `--blink-method` choices; add `--check-periodic` |
| `hp_acoustic_wave/run_emission_*.sh` (5 files) | 0 | Delete min-score lines |
| `hp_acoustic_wave/app.py` | 1 | Dual-curve bottom panel |
| `hp_acoustic_wave/tests/test_shape_segmentation.py` | 2 | New — shape detector unit tests |
| `hp_acoustic_wave/experiment_shape_vs_blinklistener_20260618.md` | 4 | Benchmark report |
| `hp_acoustic_wave/experiment_blinklistener_audit_20260618.md` | 3 | Audit findings (created only if Phase 3 finds bugs) |

---

## Task 1 — Phase 0.1: Delete orphaned metrics producers in `blink_detector.py`

**Context:** The metrics dict in `_fmcw_twinkle_score` (lines 686-708) emits many keys that no detector produces or consumes. We've already read the file; the keys `twinkle_phase_pair_library_*`, `twinkle_fmcw_has_spatial_spread`, `spread_*`, `dominance_*`, `spatial_spread_ok`, `twinkle_segment_*_votes`, `twinkle_shape_ok`, `twinkle_rhythm_*` are dead. This task keeps only the keys actually set by `_fmcw_twinkle_score` and removes the rest if they exist in the metrics dict (the file at L686-708 only sets keys listed there — orphaned ones elsewhere must be searched).

**Files:**
- Modify: `hp_acoustic_wave/blink_detector.py`

- [ ] **Step 1.1: Search for orphaned metric key producers**

Run: `cd E:/android_projects/eye_blink_detect && grep -nE "phase_pair_library|spatial_spread|spread_bins|spread_ratio|dominance_ratio|segment_coarse_votes|segment_fine_votes|shape_ok|rhythm_" hp_acoustic_wave/blink_detector.py`

Expected output: list every line where these keys appear. Record the line numbers — these are the producers to delete.

- [ ] **Step 1.2: Delete orphaned producers**

For each producer line found in Step 1.1 (likely only in `metrics` dict literals, none elsewhere — `_fmcw_twinkle_score` is the only metrics producer), delete the line. Be careful: only delete keys that no consumer references. If `grep` finds consumers in `app.py` or `benchmark_*.py`, do NOT delete those — only delete in `blink_detector.py` producers (and the consumers in Task 1.4 if found).

- [ ] **Step 1.3: Verify no syntax errors**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -c "from hp_acoustic_wave.blink_detector import build_blink_detector; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 1.4: Search for orphaned metric consumers**

Run: `cd E:/android_projects/eye_blink_detect && grep -rnE "phase_pair_library|spatial_spread|spread_bins|spread_ratio|dominance_ratio|segment_coarse_votes|segment_fine_votes|shape_ok|rhythm_" hp_acoustic_wave/ --include="*.py"`

Expected: empty, or only the producers you just deleted (residual). If consumers found in `_feature_row` or session_io, delete the column entries too.

- [ ] **Step 1.5: Commit**

```bash
cd E:/android_projects/eye_blink_detect
git add hp_acoustic_wave/blink_detector.py
git commit -m "refactor(blink): remove orphaned metrics producers (phase 0.1)"
```

---

## Task 2 — Phase 0.2: Delete `twinkle_fmcw_min_score`, `min_smoothness`, `min_deviation` gates

**Context:** These three gates (added in commit `ccce4f7` as Fix B/C) mask periodic false positives by zeroing scores. The new shape/LEVD detectors don't need them — pattern matching and adaptive thresholds do the work. We remove the gates entirely.

**Files:**
- Modify: `hp_acoustic_wave/blink_detector.py`
- Modify: `hp_acoustic_wave/config.py`
- Modify: `hp_acoustic_wave/run_hp_wave_detector.py`
- Modify: `hp_acoustic_wave/benchmark_hp_blink.py`
- Modify: `hp_acoustic_wave/run_emission_linear.sh`
- Modify: `hp_acoustic_wave/run_emission_linear_v2.sh` (if exists — else skip)
- Modify: `hp_acoustic_wave/run_emission_linear_tukey.sh`
- Modify: `hp_acoustic_wave/run_emission_triangle.sh`
- Modify: `hp_acoustic_wave/run_emission_cw_fmcw_hybrid.sh`

- [ ] **Step 2.1: Delete `_TwinklePeakEventGate.stats()` min_score_floor logic**

In `hp_acoustic_wave/blink_detector.py` at lines 140-153, replace the `stats` method body's floor computation. Current code (L147-152):

```python
        min_score_floor = (
            max(self.config.min_score, float(self.config.twinkle_fmcw_min_score))
            if is_fmcw
            else self.config.min_score
        )
        threshold = max(min_score_floor, baseline + self.config.threshold_k * robust_sigma)
        return baseline, mad, threshold
```

Replace with:

```python
        threshold = max(self.config.min_score, baseline + self.config.threshold_k * robust_sigma)
        return baseline, mad, threshold
```

(Removes the `min_score_floor` block — FMCW path now uses pure adaptive threshold like tone path.)

- [ ] **Step 2.2: Delete `twinkle_fmcw_min_score` usage in `_fmcw_twinkle_score`**

In `hp_acoustic_wave/blink_detector.py` at line 648, delete:

```python
        fmcw_min_score = max(float(self.config.twinkle_fmcw_min_score), 0.0)
```

At lines 682-684, replace the score-ok gate:

```python
        score_ok = bool(adjusted_score >= fmcw_min_score)
        if not score_ok:
            adjusted_score = 0.0
```

With (delete entirely — no floor):

```python
```

In the `metrics` dict (around L686-708), delete the keys:
- `"twinkle_fmcw_min_score": float(fmcw_min_score),`
- `"twinkle_fmcw_score_ok": float(score_ok),`

- [ ] **Step 2.3: Delete `twinkle_fmcw_min_smoothness` / `min_deviation` gates in `_fmcw_twinkle_score`**

In `hp_acoustic_wave/blink_detector.py` at lines 666-672 (Fix C block) and 686-708 (metrics dict), delete:

Block at L666-672:

```python
        # Fix C: gate on smoothness & deviation to break periodic noise-driven
        # firing (CLAUDE.md hard requirement). Coherence already multiplies by
        # smoothness, but if min_smoothness > 0 we additionally require the
        # peak frame's smoothness to clear the floor — noise with high local
        # jitter gets zeroed even when its deviation briefly crosses threshold.
        smoothness_ok = smoothness >= float(self.config.twinkle_fmcw_min_smoothness)
        deviation_ok = deviation >= float(self.config.twinkle_fmcw_min_deviation)
```

Update the conditional at L673-680 to remove the smoothness/deviation checks:

```python
        if not amplitude_ok or not amplitude_stable:
            adjusted_score = 0.0
        else:
            # Blend coherence with phase-pair delta; coherence is the primary
            # discriminator, phase_score adds short-lag energy.
            # Multiply by orthogonality to favor blink-like signatures
            base_score = max(coherence_score, phase_score * smoothness)
            adjusted_score = base_score * orthogonality_score
```

In the `metrics` dict, delete keys:
- `"twinkle_fmcw_min_smoothness": float(self.config.twinkle_fmcw_min_smoothness),`
- `"twinkle_fmcw_min_deviation": float(self.config.twinkle_fmcw_min_deviation),`
- `"twinkle_fmcw_smoothness_ok": float(smoothness_ok),`
- `"twinkle_fmcw_deviation_ok": float(deviation_ok),`

- [ ] **Step 2.4: Delete `BlinkDetectionConfig` fields**

In `hp_acoustic_wave/blink_detector.py` at lines 40-48, delete:

```python
    twinkle_fmcw_min_score: float = 0.09
```
(L40) — and lines 47-48:

```python
    twinkle_fmcw_min_smoothness: float = 0.0
    twinkle_fmcw_min_deviation: float = 0.0
```

Also delete the Fix C comment block at L44-46.

- [ ] **Step 2.5: Delete `BlinkConfig` fields in `config.py`**

In `hp_acoustic_wave/config.py` at lines 63, 67-69, delete:

```python
    twinkle_fmcw_min_score: float = 0.12  # Balanced threshold: reduces FP while maintaining recall
```
and:

```python
    # Fix C: smoothness & deviation gates to suppress periodic noise-driven triggers.
    twinkle_fmcw_min_smoothness: float = 0.0
    twinkle_fmcw_min_deviation: float = 0.0
```

- [ ] **Step 2.6: Delete CLI args in `run_hp_wave_detector.py`**

In `hp_acoustic_wave/run_hp_wave_detector.py` delete the three argument blocks at L189-212:

```python
    parser.add_argument(
        "--blink-twinkle-fmcw-min-score",
        type=float,
        default=0.09,
        help="Minimum FMCW Twinkle score before the selected range-bin phase profile can trigger",
    )
```

```python
    parser.add_argument(
        "--blink-twinkle-fmcw-min-smoothness",
        type=float,
        default=0.0,
        help="Fix C: zero score when coherence smoothness below floor (suppress periodic noise triggers)",
    )
```

```python
    parser.add_argument(
        "--blink-twinkle-fmcw-min-deviation",
        type=float,
        default=0.0,
        help="Fix C: zero score when phase-pair deviation below floor",
    )
```

In `build_app_config` (around L279-282), delete the three keyword assignments:

```python
            twinkle_fmcw_min_score=args.blink_twinkle_fmcw_min_score,
            twinkle_fmcw_min_smoothness=args.blink_twinkle_fmcw_min_smoothness,
            twinkle_fmcw_min_deviation=args.blink_twinkle_fmcw_min_deviation,
```

- [ ] **Step 2.7: Delete CLI args in `benchmark_hp_blink.py`**

In `hp_acoustic_wave/benchmark_hp_blink.py` delete the three args at L68, L70-73:

```python
    parser.add_argument("--blink-twinkle-fmcw-min-score", type=float, default=0.12)
```

```python
    parser.add_argument("--blink-twinkle-fmcw-min-smoothness", type=float, default=0.0,
                        help="Fix C: zero score when coherence smoothness < floor (suppress periodic noise triggers)")
    parser.add_argument("--blink-twinkle-fmcw-min-deviation", type=float, default=0.0,
                        help="Fix C: zero score when phase-pair deviation < floor")
```

In `build_config` (around L114-117), delete the three keyword assignments:

```python
        twinkle_fmcw_min_score=args.blink_twinkle_fmcw_min_score,
        twinkle_fmcw_min_smoothness=args.blink_twinkle_fmcw_min_smoothness,
        twinkle_fmcw_min_deviation=args.blink_twinkle_fmcw_min_deviation,
```

- [ ] **Step 2.8: Delete min-score lines in 5 launch scripts**

For each of `run_emission_linear.sh`, `run_emission_linear_tukey.sh`, `run_emission_triangle.sh`, `run_emission_cw_fmcw_hybrid.sh` in `hp_acoustic_wave/`, delete these three lines (they appear consecutively around L10-12):

```bash
  --blink-twinkle-fmcw-min-score 0.25 \
  --blink-twinkle-fmcw-min-smoothness 0.6 \
  --blink-twinkle-fmcw-min-deviation 0.30 \
```

(If `run_emission_linear_v2.sh` does not exist, skip it.)

- [ ] **Step 2.9: Verify imports**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -c "from hp_acoustic_wave.blink_detector import build_blink_detector; from hp_acoustic_wave.config import BlinkConfig; from hp_acoustic_wave.run_hp_wave_detector import build_app_config; from hp_acoustic_wave.benchmark_hp_blink import build_config; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 2.10: Run existing tests**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -m pytest hp_acoustic_wave/tests/ -v 2>&1 | tail -50`

Expected: tests run. Pre-existing failures related to `twinkle_fmcw_min_score` (e.g. `test_twinkle_fmcw_*`) may now break differently — record any new failures for Task 3 review. The cleanup goal is "no import errors" and "no test referencing deleted fields crashes with AttributeError."

- [ ] **Step 2.11: Commit**

```bash
cd E:/android_projects/eye_blink_detect
git add hp_acoustic_wave/blink_detector.py hp_acoustic_wave/config.py hp_acoustic_wave/run_hp_wave_detector.py hp_acoustic_wave/benchmark_hp_blink.py hp_acoustic_wave/run_emission_*.sh
git commit -m "refactor(blink): remove min-score/smoothness/deviation gates (phase 0.2)"
```

---

## Task 3 — Phase 0.3: Simplify `_candidate_trajectory_score` passthrough

**Context:** `twinkle_candidate_windows` defaults to `()` (empty tuple). When empty, `_candidate_trajectory_score` (L~613-632) is a passthrough — `votes` and `best_candidate_window` are zero/empty. The shape and LEVD detectors don't use this path. We inline it to remove dead code.

**Files:**
- Modify: `hp_acoustic_wave/blink_detector.py`

- [ ] **Step 3.1: Read `_candidate_trajectory_score` and its caller**

Run: `cd E:/android_projects/eye_blink_detect && grep -nE "_candidate_trajectory_score|twinkle_candidate_windows|twinkle_min_candidate_votes" hp_acoustic_wave/blink_detector.py`

Record the method definition line and every call site.

- [ ] **Step 3.2: Inline the passthrough at the call site**

In `TwinkleTwinkleBlinkDetector.update` (around L721-728), the call:

```python
        (
            score,
            trajectory_span,
            acceleration_rms,
            sign_changes,
            candidate_votes,
            best_candidate_window,
        ) = self._candidate_trajectory_score()
```

Replace with the empty-windows passthrough (since `twinkle_candidate_windows=()` always now):

```python
        # candidate-vote path removed (always empty); keep defaults
        trajectory_span = 0.0
        acceleration_rms = 0.0
        sign_changes = 0
        candidate_votes = 0
        best_candidate_window = 0
```

If `sign_changes` was actually computed inside `_candidate_trajectory_score` (it was — the method did real work computing sign_changes regardless of windows), DO NOT replace it with 0. Instead, extract only the sign-changes computation. To check: read the method body. If `sign_changes` depends on phase_steps window, keep that computation inline:

```python
        # sign_changes derived from recent phase steps (was in _candidate_trajectory_score)
        recent = list(self.phase_steps)[-self.config.short_window:]
        if len(recent) >= 2:
            signs = [1 if v >= 0 else -1 for v in recent]
            sign_changes = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i-1])
        else:
            sign_changes = 0
        trajectory_span = 0.0
        acceleration_rms = 0.0
        candidate_votes = 0
        best_candidate_window = 0
```

**Verify before committing** that the sign_changes formula matches the original method (read `_candidate_trajectory_score`).

- [ ] **Step 3.3: Delete `_candidate_trajectory_score` method**

Delete the entire method definition (the `def _candidate_trajectory_score(self): ... return representative_score, trajectory_span, acceleration_rms, sign_changes, votes, float(window_size)` block).

- [ ] **Step 3.4: Delete unused fields**

In `BlinkDetectionConfig` (blink_detector.py L34-35), delete:

```python
    twinkle_candidate_windows: Tuple[int, ...] = ()
    twinkle_min_candidate_votes: int = 1
```

In `config.py` (L46-47), delete:

```python
    twinkle_candidate_windows: tuple = ()
    twinkle_min_candidate_votes: int = 1
```

In `run_hp_wave_detector.py` — search for `blink_candidate_windows` / `blink_min_candidate_votes` and delete both parser args + the build_app_config kwargs (`candidate_windows=` and `twinkle_min_candidate_votes=`). Note the `candidate_windows` computation at L228-232 of `build_app_config` also becomes dead — delete it too.

In `benchmark_hp_blink.py` — search and delete the same args + `candidate_windows` computation + kwargs.

- [ ] **Step 3.5: Verify imports**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -c "from hp_acoustic_wave.blink_detector import build_blink_detector; from hp_acoustic_wave.run_hp_wave_detector import build_app_config; from hp_acoustic_wave.benchmark_hp_blink import build_config; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 3.6: Commit**

```bash
cd E:/android_projects/eye_blink_detect
git add hp_acoustic_wave/blink_detector.py hp_acoustic_wave/config.py hp_acoustic_wave/run_hp_wave_detector.py hp_acoustic_wave/benchmark_hp_blink.py
git commit -m "refactor(blink): inline empty candidate-vote passthrough (phase 0.3)"
```

---

## Task 4 — Phase 0.4: Unify `BlinkConfig` / `BlinkDetectionConfig`

**Context:** `config.py:BlinkConfig` and `blink_detector.py:BlinkDetectionConfig` duplicate ~30 fields with **divergent defaults** (`twinkle_fmcw_min_score` was 0.12 vs 0.09 — but we deleted it in Task 2, so the divergence is gone; still the duplication is a maintenance hazard). `app.py` does `BlinkDetectionConfig(**self.config.blink.__dict__)`, so any new field must exist in both. We make `BlinkConfig` the single source of truth.

**Files:**
- Modify: `hp_acoustic_wave/blink_detector.py`

- [ ] **Step 4.1: Verify fields match after Tasks 1-3**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -c "from hp_acoustic_wave.blink_detector import BlinkDetectionConfig as BDC; from hp_acoustic_wave.config import BlinkConfig as BC; a=set(BDC.__dataclass_fields__); b=set(BC.__dataclass_fields__); print('only_in_BDC:', a-b); print('only_in_BC:', b-a)"`

Expected: `only_in_BDC: set()` and `only_in_BC: set()` (or near-empty). Record any divergence.

- [ ] **Step 4.2: Replace `BlinkDetectionConfig` with re-export**

In `hp_acoustic_wave/blink_detector.py` at the top of the file, after the existing imports, add:

```python
from hp_acoustic_wave.config import BlinkConfig as BlinkDetectionConfig
```

Then delete the `@dataclass class BlinkDetectionConfig: ...` block (L11-48). This makes `BlinkDetectionConfig` an alias for `BlinkConfig` — single source of truth.

- [ ] **Step 4.3: Verify imports**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -c "from hp_acoustic_wave.blink_detector import BlinkDetectionConfig, build_blink_detector; from hp_acoustic_wave.config import BlinkConfig; assert BlinkDetectionConfig is BlinkConfig; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 4.4: Run all tests**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -m pytest hp_acoustic_wave/tests/ -v 2>&1 | tail -30`

Expected: no new failures (relative to Task 2 baseline).

- [ ] **Step 4.5: Commit**

```bash
cd E:/android_projects/eye_blink_detect
git add hp_acoustic_wave/blink_detector.py
git commit -m "refactor(blink): unify BlinkDetectionConfig as alias for BlinkConfig (phase 0.4)"
```

---

## Task 5 — Phase 1.1: Expose `last_ungated_coherence` on detectors

**Context:** `_fmcw_twinkle_score` computes `coherence_score` (raw deviation × smoothness) BEFORE the amplitude/stability gates zero `adjusted_score`. We expose this raw value so the realtime plot can show the ungated signal.

**Files:**
- Modify: `hp_acoustic_wave/blink_detector.py`

- [ ] **Step 5.1: Add `last_ungated_coherence` attribute to `TwinkleTwinkleBlinkDetector`**

In `hp_acoustic_wave/blink_detector.py` at `TwinkleTwinkleBlinkDetector.__init__` (around L408, after `self.current_unwrapped_phase = 0.0`), add:

```python
        self.last_ungated_coherence = 0.0
```

- [ ] **Step 5.2: Store ungated coherence in `_fmcw_twinkle_score`**

In `hp_acoustic_wave/blink_detector.py` in `_fmcw_twinkle_score` (around L657, after `coherence_score, smoothness, deviation = self._fmcw_coherence_score()`), add:

```python
        # Expose the raw coherence score (before amplitude/stability gates zero
        # the adjusted score) so the realtime viz can show the ungated signal.
        self.last_ungated_coherence = float(coherence_score)
```

- [ ] **Step 5.3: Add `last_ungated_coherence` attribute to `BlinkListenerBlinkDetector`**

In `hp_acoustic_wave/blink_detector.py` at `BlinkListenerBlinkDetector.__init__` (around L283, after the last deque init), add:

```python
        self.last_ungated_coherence = 0.0
```

- [ ] **Step 5.4: Verify**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -c "
from hp_acoustic_wave.blink_detector import build_blink_detector, BlinkDetectionConfig
for m in ['twinkle', 'blinklistener']:
    d = build_blink_detector(BlinkDetectionConfig(method=m))
    assert hasattr(d, 'last_ungated_coherence'), m
print('ok')"`

Expected: prints `ok`.

- [ ] **Step 5.5: Commit**

```bash
cd E:/android_projects/eye_blink_detect
git add hp_acoustic_wave/blink_detector.py
git commit -m "feat(blink): expose last_ungated_coherence attribute (phase 1.1)"
```

---

## Task 6 — Phase 1.2: Add dual history buffers in `app.py`

**Files:**
- Modify: `hp_acoustic_wave/app.py`

- [ ] **Step 6.1: Add history buffers at init**

In `hp_acoustic_wave/app.py` at lines 40-41, immediately after the existing `self.threshold_history = deque(maxlen=240)`, add:

```python
        self.phase_trajectory_history = deque(maxlen=240)
        self.ungated_coherence_history = deque(maxlen=240)
```

- [ ] **Step 6.2: Populate histories in `_process_audio_queue`**

In `hp_acoustic_wave/app.py` at line 244-245, replace the block:

```python
            plot_value = self.latest_score if self.config.mode == "blink" else feature.motion_energy
            self.energy_history.append(plot_value)
            self.threshold_history.append(self.latest_threshold)
```

With:

```python
            plot_value = self.latest_score if self.config.mode == "blink" else feature.motion_energy
            self.energy_history.append(plot_value)
            self.threshold_history.append(self.latest_threshold)
            if self.config.mode == "blink":
                if (
                    feature.signal_mode == "fmcw"
                    and feature.phase_pair_delta is not None
                ):
                    phase_val = float(feature.phase_pair_delta)
                else:
                    phase_val = float(feature.phase)
                self.phase_trajectory_history.append(phase_val)
                self.ungated_coherence_history.append(
                    float(getattr(self.detector, "last_ungated_coherence", 0.0))
                )
```

- [ ] **Step 6.3: Verify**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -c "from hp_acoustic_wave.app import AcousticApp; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 6.4: Commit**

```bash
cd E:/android_projects/eye_blink_detect
git add hp_acoustic_wave/app.py
git commit -m "feat(viz): add phase_trajectory_history + ungated_coherence_history (phase 1.2)"
```

---

## Task 7 — Phase 1.3: Split `_draw_overlay` into dual-curve panel

**Context:** Replace the single bottom panel (energy + threshold) with two stacked panels: top = phase trajectory (centered at 0, auto-scaled to ±max|val|), bottom = ungated coherence score + threshold line.

**Files:**
- Modify: `hp_acoustic_wave/app.py`

- [ ] **Step 7.1: Replace the bottom-panel drawing block**

In `hp_acoustic_wave/app.py` at lines 493-511, the current drawing block:

```python
        values = list(self.energy_history)
        thresholds = list(self.threshold_history)
        if values:
            max_value = max(max(values), max(thresholds) if thresholds else 0.0, self.config.detector.min_energy)
            max_value = max(max_value, 1e-6)
            left = 20
            right = width - 20
            top = panel_y + 20
            bottom = height + plot_h - 20
            cv2.rectangle(canvas, (left, top), (right, bottom), (80, 80, 80), 1)
            for series, color in ((values, (0, 255, 0)), (thresholds, (0, 180, 255))):
                if len(series) < 2:
                    continue
                points = []
                for idx, val in enumerate(series):
                    x = int(left + idx * (right - left) / max(1, len(series) - 1))
                    y = int(bottom - min(val / max_value, 1.0) * (bottom - top))
                    points.append((x, y))
                cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2)
        return canvas
```

Replace with the dual-curve panel:

```python
        # Dual-curve panel: top = phase trajectory (centered at 0),
        # bottom = ungated coherence score + threshold.
        left = 20
        right = width - 20
        panel_top = panel_y + 20
        panel_bottom = height + plot_h - 20
        mid_y = (panel_top + panel_bottom) // 2
        gap = 4
        top_top, top_bottom = panel_top, mid_y - gap
        bot_top, bot_bottom = mid_y + gap, panel_bottom

        if self.config.mode == "blink":
            # --- Top: phase trajectory (bipolar, centered at mid_top) ---
            phases = list(self.phase_trajectory_history)
            cv2.rectangle(canvas, (left, top_top), (right, top_bottom), (80, 80, 80), 1)
            cv2.putText(canvas, "phase trajectory", (left + 4, top_top + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 255), 1)
            if len(phases) >= 2:
                max_abs = max(abs(min(phases)), abs(max(phases)), 1e-6)
                center_y = (top_top + top_bottom) // 2
                span = (top_bottom - top_top) // 2 - 2
                points = []
                for idx, val in enumerate(phases):
                    x = int(left + idx * (right - left) / max(1, len(phases) - 1))
                    y = int(center_y - (val / max_abs) * span)
                    points.append((x, y))
                cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (0, 255, 0), 2)

            # --- Bottom: ungated coherence + threshold ---
            scores = list(self.ungated_coherence_history)
            thresholds = list(self.threshold_history)
            cv2.rectangle(canvas, (left, bot_top), (right, bot_bottom), (80, 80, 80), 1)
            cv2.putText(canvas, "ungated coherence", (left + 4, bot_top + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 220, 255), 1)
            if len(scores) >= 2:
                all_vals = list(scores) + list(thresholds)
                max_value = max(max(all_vals), self.config.detector.min_energy, 1e-6)
                # threshold (orange dashed)
                if len(thresholds) >= 2:
                    pts = []
                    for idx, val in enumerate(thresholds):
                        x = int(left + idx * (right - left) / max(1, len(thresholds) - 1))
                        y = int(bot_bottom - min(val / max_value, 1.0) * (bot_bottom - bot_top - 4))
                        pts.append((x, y))
                    for i in range(0, len(pts) - 1, 2):
                        cv2.line(canvas, pts[i], pts[i + 1], (0, 180, 255), 2)
                # ungated coherence (blue)
                pts = []
                for idx, val in enumerate(scores):
                    x = int(left + idx * (right - left) / max(1, len(scores) - 1))
                    y = int(bot_bottom - min(val / max_value, 1.0) * (bot_bottom - bot_top - 4))
                    pts.append((x, y))
                cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32)], False, (255, 180, 0), 2)
        else:
            values = list(self.energy_history)
            thresholds = list(self.threshold_history)
            if values:
                max_value = max(max(values), max(thresholds) if thresholds else 0.0, self.config.detector.min_energy)
                max_value = max(max_value, 1e-6)
                cv2.rectangle(canvas, (left, panel_top), (right, panel_bottom), (80, 80, 80), 1)
                for series, color in ((values, (0, 255, 0)), (thresholds, (0, 180, 255))):
                    if len(series) < 2:
                        continue
                    points = []
                    for idx, val in enumerate(series):
                        x = int(left + idx * (right - left) / max(1, len(series) - 1))
                        y = int(panel_bottom - min(val / max_value, 1.0) * (panel_bottom - panel_top))
                        points.append((x, y))
                    cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 2)
        return canvas
```

- [ ] **Step 7.2: Verify import**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -c "from hp_acoustic_wave.app import AcousticApp; print('ok')"`

Expected: prints `ok`.

- [ ] **Step 7.3: Smoke test the panel rendering**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -c "
import numpy as np
from hp_acoustic_wave.app import AcousticApp
import cv2
# build a minimal app instance without running audio
import argparse
from hp_acoustic_wave.config import AppConfig
app_cfg = AppConfig()
app_cfg.mode = 'blink'
class Stub: pass
app = AcousticApp.__new__(AcousticApp)
app.config = app_cfg
app.canvas_preview = None
import collections
app.energy_history = collections.deque(maxlen=240)
app.threshold_history = collections.deque(maxlen=240)
app.phase_trajectory_history = collections.deque([0.1,-0.2,0.3,-0.1]*10, maxlen=240)
app.ungated_coherence_history = collections.deque([0.1,0.2,0.05,0.3]*10, maxlen=240)
app.latest_score = 0.0
app.latest_threshold = 0.0
app.latest_time_s = 0.0
app.latest_event_id = 0
app.latest_detector_method = 'twinkle'
app.last_detection_time_s = None
app.last_detection_display_time_s = None
app.last_detection_energy = 0.0
app.last_detection_score = 0.0
app.last_detection_method = 'twinkle'
app.manual_marker_count = 0
import time
app.last_visual_blink_display_time_s = None
app.latest_visual_state = None
app.visual_labeling_enabled = False
app.visual_labeling_error = None
app.camera_enabled = True
# call only the draw code path
frame = np.zeros((480,640,3), dtype=np.uint8)
canvas = app._draw_overlay(cv2, frame)
assert canvas.shape == (480+160, 640, 3), canvas.shape
print('ok')"`

Expected: prints `ok`. If `AttributeError` for a missing attribute, add it to the stub (the test only validates the draw code path runs end-to-end).

- [ ] **Step 7.4: Commit**

```bash
cd E:/android_projects/eye_blink_detect
git add hp_acoustic_wave/app.py
git commit -m "feat(viz): dual-curve bottom panel (phase trajectory + ungated coherence) (phase 1.3)"
```

---

## Task 8 — Phase 2.1: Add shape config fields

**Files:**
- Modify: `hp_acoustic_wave/config.py`

- [ ] **Step 8.1: Append shape fields to `BlinkConfig`**

In `hp_acoustic_wave/config.py` at the end of `BlinkConfig` (after the `twinkle_fmcw_use_orthogonality` line, which is now the last field after Task 2 cleanup), add:

```python
    # Shape-based segmentation detector (Twinkle paper pattern matching)
    shape_smoothing_window: int = 5
    shape_detection_window: int = 30
    shape_min_edge_len_s: float = 0.05
    shape_max_edge_len_s: float = 0.5
    shape_edge_sigma_k: float = 3.0
    shape_refractory_s: float = 0.5
```

- [ ] **Step 8.2: Verify**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -c "from hp_acoustic_wave.config import BlinkConfig; c=BlinkConfig(); print(c.shape_smoothing_window, c.shape_edge_sigma_k)"`

Expected: `5 3.0`.

- [ ] **Step 8.3: Commit**

```bash
cd E:/android_projects/eye_blink_detect
git add hp_acoustic_wave/config.py
git commit -m "feat(config): add shape-segmentation config fields (phase 2.1)"
```

---

## Task 9 — Phase 2.2: Write failing tests for `_TwinkleShapeSegmentationDetector`

**Files:**
- Create: `hp_acoustic_wave/tests/test_shape_segmentation.py`

- [ ] **Step 9.1: Create the failing test file**

Create `hp_acoustic_wave/tests/test_shape_segmentation.py` with:

```python
"""Unit tests for _TwinkleShapeSegmentationDetector.

Shape-based segmentation detects a down-up (or up-down) edge pattern on the
phase trajectory within time bounds [min_edge_len, max_edge_len]. A single
noise peak does NOT form such a pattern, so periodic noise-driven firing is
structurally prevented.
"""
import numpy as np
import pytest

from hp_acoustic_wave.blink_detector import (
    BlinkDetectionConfig,
    _TwinkleShapeSegmentationDetector,
)
from hp_acoustic_wave.dsp import ChunkFeature


def make_feature(time_s, phase_value, signal_mode="fmcw"):
    """Build a minimal ChunkFeature. Shape detector only reads time_s,
    signal_mode, phase, phase_pair_delta."""
    return ChunkFeature(
        time_s=time_s,
        sample_index=int(time_s * 48000),
        i_value=0.0,
        q_value=0.0,
        amplitude=0.1,
        amplitude_delta=0.0,
        phase=float(phase_value),
        phase_delta=0.0,
        motion_energy=0.0,
        rms=0.1,
        peak_abs=0.1,
        signal_mode=signal_mode,
        range_bin=15,
        range_distance_m=0.4,
        phase_pair_delta=float(phase_value),
    )


def _run_series(detector, values, dt=0.05, start_t=0.0):
    events = []
    t = start_t
    for v in values:
        r = detector.update(make_feature(t, v))
        if r.is_event:
            events.append(t)
        t += dt
    return events


def test_shape_detects_down_up_edge_within_blink_duration():
    """A clear down-then-up edge spanning ~0.2s should trigger exactly one blink."""
    config = BlinkDetectionConfig(method="shape")
    det = _TwinkleShapeSegmentationDetector(config)
    # 1s baseline (tiny noise)
    rng = np.random.default_rng(0)
    baseline = list(rng.normal(0.0, 1e-4, 20))
    _run_series(det, baseline)
    # down-up edge from -0.3 to +0.3 over 5 frames (0.25s)
    edge = [-0.3, -0.2, 0.0, 0.2, 0.3, 0.0, 0.0, 0.0]
    events = _run_series(det, edge, start_t=1.0)
    assert len(events) >= 1, "expected at least one blink on a down-up edge"
    assert len(events) <= 2, "refractory should prevent double-fire on the same edge"


def test_shape_ignores_single_peak_noise():
    """Stationary random noise with no coherent edge pattern should not
    produce many false events (refractory bounds the count)."""
    config = BlinkDetectionConfig(method="shape")
    det = _TwinkleShapeSegmentationDetector(config)
    rng = np.random.default_rng(42)
    events = _run_series(det, list(rng.normal(0.0, 0.01, 200)))
    assert len(events) <= 2, f"random noise produced {len(events)} false positives"


def test_shape_refractory_prevents_double_fire():
    """A single clean blink should fire exactly once even if the edge is
    sharp enough to look like two peaks."""
    config = BlinkDetectionConfig(method="shape")
    det = _TwinkleShapeSegmentationDetector(config)
    # baseline
    _run_series(det, [0.0] * 20)
    # one V-shape blink over 0.25s
    events = _run_series(det, [-0.5, -0.4, 0.0, 0.4, 0.5, 0.0, 0.0], start_t=1.0)
    assert len(events) == 1, f"expected 1 blink, got {len(events)}"


def test_shape_supports_up_down_direction():
    """A real blink may produce an up-then-down edge. Both directions are valid."""
    config = BlinkDetectionConfig(method="shape")
    det = _TwinkleShapeSegmentationDetector(config)
    _run_series(det, [0.0] * 20)
    events = _run_series(det, [0.5, 0.4, 0.0, -0.4, -0.5, 0.0, 0.0], start_t=1.0)
    assert len(events) >= 1, "up-down edge should also trigger"


def test_shape_below_baseline_sigma_does_not_fire():
    """If edge magnitude is below shape_edge_sigma_k * baseline sigma, no fire."""
    config = BlinkDetectionConfig(method="shape", shape_edge_sigma_k=10.0)
    det = _TwinkleShapeSegmentationDetector(config)
    # baseline sigma is small (1e-4), but k=10 demands a large edge
    _run_series(det, [0.0] * 20)
    # edge of ~0.05 magnitude — vs sigma ~1e-4, this should still fire because
    # 0.05 >> 10*1e-4. So instead set sigma to a high value by feeding noise first.
    det2 = _TwinkleShapeSegmentationDetector(
        BlinkDetectionConfig(method="shape", shape_edge_sigma_k=10.0)
    )
    rng = np.random.default_rng(1)
    _run_series(det2, list(rng.normal(0.0, 0.01, 30)))  # baseline sigma ~0.01
    # now a small edge of 0.05: 0.05/0.01 = 5 sigma < 10 → no fire
    events = _run_series(det2, [0.05, 0.0, -0.05, 0.0], start_t=1.5)
    assert len(events) == 0, f"sub-threshold edge fired {len(events)} times"
```

- [ ] **Step 9.2: Run tests to verify they fail**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -m pytest hp_acoustic_wave/tests/test_shape_segmentation.py -v 2>&1 | tail -30`

Expected: FAIL with `ImportError: cannot import name '_TwinkleShapeSegmentationDetector'`.

- [ ] **Step 9.3: Commit the failing test**

```bash
cd E:/android_projects/eye_blink_detect
git add hp_acoustic_wave/tests/test_shape_segmentation.py
git commit -m "test(blink): add shape-segmentation failing tests (phase 2.2)"
```

---

## Task 10 — Phase 2.3: Implement `_TwinkleShapeSegmentationDetector`

**Files:**
- Modify: `hp_acoustic_wave/blink_detector.py`

- [ ] **Step 10.1: Add the detector class**

In `hp_acoustic_wave/blink_detector.py`, add the new class immediately BEFORE `class CompositeBlinkDetector` (around L795). Insert:

```python
class _TwinkleShapeSegmentationDetector:
    """Twinkle paper's shape-based segmentation on the phase trajectory.

    A blink produces a directional edge (down-up OR up-down) within a
    bounded time window (~0.1-0.5s). Noise-driven single peaks do NOT
    form such a pattern, so periodic false-positive firing is structurally
    prevented.

    The detector:
      1. Smooths the raw phase_pair_delta over `shape_smoothing_window` chunks.
      2. Maintains a `shape_detection_window` (~1.5s) sliding window.
      3. Computes baseline sigma via MAD (1.4826 * median(|x - median|)).
      4. Finds argmin/argmax in the window; if their time gap is within
         [shape_min_edge_len_s, shape_max_edge_len_s] and the magnitude
         exceeds shape_edge_sigma_k * baseline_sigma, fire one event.
      5. Refractory shape_refractory_s prevents double-fire on the same edge.
    """

    method = "shape"

    def __init__(self, config: "BlinkDetectionConfig"):
        self.config = config
        self.window: Deque[Tuple[float, float]] = deque(maxlen=config.shape_detection_window)
        self.smooth_buffer: Deque[float] = deque(maxlen=config.shape_smoothing_window)
        self.baseline: Deque[float] = deque(maxlen=200)  # ~10s @ 20Hz
        self.last_blink_time = -1e9
        self.last_ungated_coherence = 0.0

    def _trajectory_value(self, feature: ChunkFeature) -> float:
        if (
            feature.signal_mode == "fmcw"
            and feature.phase_pair_delta is not None
        ):
            return float(feature.phase_pair_delta)
        return float(feature.phase)

    def _baseline_sigma(self) -> Tuple[float, float, float]:
        if len(self.baseline) < 5:
            return 0.0, 0.0, 1e-6
        arr = np.asarray(self.baseline, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        sigma = max(mad * 1.4826, 1e-6)
        return med, mad, sigma

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        delta = self._trajectory_value(feature)
        self.smooth_buffer.append(delta)
        smoothed = float(np.mean(self.smooth_buffer))
        self.window.append((feature.time_s, smoothed))
        self.baseline.append(smoothed)

        med, mad, sigma = self._baseline_sigma()
        threshold = float(self.config.shape_edge_sigma_k * sigma)

        in_refractory = (feature.time_s - self.last_blink_time) < self.config.shape_refractory_s
        min_samples = max(3, int(self.config.shape_min_edge_len_s * 20.0))  # ~20Hz assumption
        not_enough_data = len(self.window) < min_samples

        metrics = {
            "shape_baseline_sigma": float(sigma),
            "shape_threshold": threshold,
            "shape_baseline_median": float(med),
            "shape_window_len": float(len(self.window)),
        }

        if in_refractory or not_enough_data:
            return BlinkDetectionResult(
                is_event=False,
                event_id=0,
                method=self.method,
                score=0.0,
                threshold=threshold,
                baseline=float(med),
                mad=float(mad),
                metrics={**metrics, "shape_edge_mag": 0.0, "shape_edge_len": 0.0},
            )

        times = [p[0] for p in self.window]
        vals = [p[1] for p in self.window]
        i_min = int(np.argmin(vals))
        i_max = int(np.argmax(vals))
        edge_len = abs(times[i_max] - times[i_min])
        edge_mag = abs(vals[i_max] - vals[i_min])
        confidence = edge_mag / sigma if sigma > 1e-9 else 0.0

        is_event = bool(
            self.config.shape_min_edge_len_s <= edge_len <= self.config.shape_max_edge_len_s
            and edge_mag >= threshold
        )

        if is_event:
            self.last_blink_time = feature.time_s

        return BlinkDetectionResult(
            is_event=is_event,
            event_id=(1 if is_event else 0),
            method=self.method,
            score=(float(confidence) if is_event else 0.0),
            threshold=threshold,
            baseline=float(med),
            mad=float(mad),
            metrics={
                **metrics,
                "shape_edge_mag": float(edge_mag),
                "shape_edge_len": float(edge_len),
            },
        )
```

- [ ] **Step 10.2: Register in `build_blink_detector`**

In `hp_acoustic_wave/blink_detector.py` at the `build_blink_detector` function (L831-839), replace with:

```python
def build_blink_detector(config: "BlinkDetectionConfig"):
    method = config.method.lower()
    if method == "blinklistener":
        return BlinkListenerBlinkDetector(config)
    if method in ("twinkle", "twinkletwinkle"):
        return TwinkleTwinkleBlinkDetector(config)
    if method == "shape":
        return _TwinkleShapeSegmentationDetector(config)
    if method == "both":
        return CompositeBlinkDetector(config)
    raise ValueError("blink method must be one of: blinklistener, twinkle, shape, both")
```

- [ ] **Step 10.3: Run tests**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -m pytest hp_acoustic_wave/tests/test_shape_segmentation.py -v 2>&1 | tail -30`

Expected: all 5 tests PASS. If `test_shape_below_baseline_sigma_does_not_fire` fails because the edge crossed threshold, adjust the test data rather than the detector — the detector should be a faithful implementation of the spec.

- [ ] **Step 10.4: Run the full test suite**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -m pytest hp_acoustic_wave/tests/ -v 2>&1 | tail -30`

Expected: no new failures relative to Task 4 baseline.

- [ ] **Step 10.5: Commit**

```bash
cd E:/android_projects/eye_blink_detect
git add hp_acoustic_wave/blink_detector.py
git commit -m "feat(blink): add _TwinkleShapeSegmentationDetector (phase 2.3)"
```

---

## Task 11 — Phase 2.4: Extend `--blink-method` CLI choices

**Files:**
- Modify: `hp_acoustic_wave/run_hp_wave_detector.py`
- Modify: `hp_acoustic_wave/benchmark_hp_blink.py`

- [ ] **Step 11.1: Update choices in `run_hp_wave_detector.py`**

In `hp_acoustic_wave/run_hp_wave_detector.py` at lines 25-29, change:

```python
    parser.add_argument(
        "--blink-method",
        choices=["blinklistener", "twinkle", "both"],
        default="twinkle",
        help="Blink detection method when --mode blink",
    )
```

To:

```python
    parser.add_argument(
        "--blink-method",
        choices=["blinklistener", "twinkle", "shape", "both"],
        default="twinkle",
        help="Blink detection method when --mode blink",
    )
```

- [ ] **Step 11.2: Update choices in `benchmark_hp_blink.py`**

In `hp_acoustic_wave/benchmark_hp_blink.py` at line 43, change:

```python
    parser.add_argument("--blink-method", choices=["blinklistener", "twinkle", "both"], default="twinkle")
```

To:

```python
    parser.add_argument("--blink-method", choices=["blinklistener", "twinkle", "shape", "both"], default="twinkle")
```

- [ ] **Step 11.3: Verify**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" hp_acoustic_wave/benchmark_hp_blink.py --help 2>&1 | grep blink-method`

Expected: shows `shape` in the choices list.

- [ ] **Step 11.4: Commit**

```bash
cd E:/android_projects/eye_blink_detect
git add hp_acoustic_wave/run_hp_wave_detector.py hp_acoustic_wave/benchmark_hp_blink.py
git commit -m "feat(cli): add 'shape' to --blink-method choices (phase 2.4)"
```

---

## Task 12 — Phase 3.1: Audit `BlinkListenerBlinkDetector`

**Context:** `BlinkListenerBlinkDetector` (L265-367) + `_RobustEventGate` (L63-114) implement the BlinkListener paper's I/Q LEVD approach. FMCW path already populates `feature.i_value`, `feature.q_value` (dsp.py:399-401). This task is a READ-ONLY audit — only write code if a real bug is found. Compare against the paper at `E:/android_projects/eye_blink_detect/FaceAcousticSensing/papers/BlinkListener "Listen" to Your Eye Blink Using Your Smartphone.pdf`.

**Files:**
- Read-only: `hp_acoustic_wave/blink_detector.py` (L63-117, L265-367)
- Read-only: `E:/android_projects/eye_blink_detect/FaceAcousticSensing/papers/BlinkListener "Listen" to Your Eye Blink Using Your Smartphone.pdf`
- Maybe-create: `hp_acoustic_wave/experiment_blinklistener_audit_20260618.md`

- [ ] **Step 12.1: Read the BlinkListener paper**

Open `E:/android_projects/eye_blink_detect/FaceAcousticSensing/papers/BlinkListener "Listen" to Your Eye Blink Using Your Smartphone.pdf` (Section 3.2 "Modeling the Eye Blink Process" + Section 6.3 "Real-time Detection"). Use the Read tool with `pages: "1-10"`.

Record:
- Paper's definition of "blink" in I/Q space (amplitude vs phase change).
- LEVD threshold formula (median + k×MAD? Or σ-based?).
- Refractory period definition.
- Whether the paper projects I/Q onto viewing positions or origin.

- [ ] **Step 12.2: Audit `_best_viewing_projection` (L295-311)**

Check:
- Does it center the I/Q vector at the median baseline (`center_i`, `center_q`) before projecting? **Yes** (L299-302). Good — this matches the paper's viewing-position concept.
- 12 angles are tested with `np.linspace(0, π, num=12, endpoint=False)` — that covers a half-circle. Since we take `abs(...)`, the other half is symmetric. **Correct.**
- Does the projection return the largest deviation from the median projection? **Yes** (L310). Good.

If any of these is wrong, flag in the audit doc.

- [ ] **Step 12.3: Audit `phase_stable_projection` (L319)**

```python
phase_stable_projection = best_projection if abs(feature.phase_delta) <= 0.12 else gated_projection
```

The `0.12` rad constant is hard-coded. Check the paper: does it specify a phase-stability threshold? If yes, confirm 0.12 matches. If no, recommend making it a config field (`blinklistener_phase_stable_threshold`). This is a candidate for a fix — record but do not change unless a clear paper citation says otherwise.

- [ ] **Step 12.4: Audit score normalization (L330-332)**

```python
raw_score = max(amplitude_bump, amplitude_range, phase_stable_projection, 0.75 * projection_range)
score_scale = max(baseline_amplitude, 1e-4)
score = raw_score / score_scale
```

Paper requires the score be **relative** (normalized to baseline amplitude). Confirm:
- `score_scale` uses `baseline_amplitude` (median of history), not current frame. **Yes** (L290-293 `_baseline_amplitude`). Good.
- `0.75 * projection_range` weight — check if the paper specifies this weight. If not documented, this is a tuning heuristic — flag but don't change.

- [ ] **Step 12.5: Audit `_RobustEventGate` (L63-114)**

Check:
- `threshold = max(min_score, baseline + threshold_k * robust_sigma)` at L78. With `min_score` removed in Task 2's logic (we removed the FMCW-specific floor but `min_score=0.006` is still the global default — that's fine for tone path), confirm tone-path behavior is unchanged.
- `release_ratio=0.4` (default) — paper specifies hysteresis? If yes, confirm 0.4.
- `baseline_freeze_s` (default 0.60) — prevents baseline contamination right after a detected event. Confirm paper does the same.

- [ ] **Step 12.6: Write audit findings**

Create `hp_acoustic_wave/experiment_blinklistener_audit_20260618.md` with:
- Section: paper algorithm summary.
- Section: implementation-vs-paper comparison (per Step 12.2-12.5).
- Section: bugs found (if any) — clearly marked.
- Section: recommended fixes (if any) with paper citations.

If the audit finds no bugs, write a short "No bugs found. Implementation matches paper" conclusion and commit only the doc.

- [ ] **Step 12.7: If a bug was found, fix it**

If Step 12.6 identifies a clear bug (e.g., wrong projection direction, wrong normalization), make the targeted fix in `blink_detector.py`. Re-run existing tests to ensure no regression. Otherwise skip this step.

- [ ] **Step 12.8: Commit**

```bash
cd E:/android_projects/eye_blink_detect
git add hp_acoustic_wave/experiment_blinklistener_audit_20260618.md hp_acoustic_wave/blink_detector.py
git commit -m "docs(audit): BlinkListenerBlinkDetector implementation audit (phase 3.1)"
```

---

## Task 13 — Phase 4.1: Add `--check-periodic` flag to benchmark

**Files:**
- Modify: `hp_acoustic_wave/benchmark_hp_blink.py`

- [ ] **Step 13.1: Add the CLI flag and the cluster-ratio function**

In `hp_acoustic_wave/benchmark_hp_blink.py`, first locate the end of `parse_args` (around L82-83). Add a new argument before `return parser.parse_args(argv)`:

```python
    parser.add_argument(
        "--check-periodic",
        action="store_true",
        help="Print anti-periodic cluster_ratio: fraction of event gaps within 0.2s of refractory_s.",
    )
```

- [ ] **Step 13.2: Add the helper function**

Add this function near the top of `benchmark_hp_blink.py` (after the imports):

```python
def check_periodic_cluster_ratio(event_times, refractory_s, tolerance=0.2):
    """Fraction of inter-event gaps within +/-tolerance of refractory_s.

    A high ratio (>0.5) indicates periodic noise-driven firing — the hallmark
    of the regression we're trying to eliminate. <0.3 is the target.
    """
    if len(event_times) < 3:
        return 0.0
    import numpy as np
    gaps = np.diff(sorted(event_times))
    clustered = float(sum(abs(float(g) - refractory_s) < tolerance for g in gaps))
    return clustered / float(len(gaps))
```

- [ ] **Step 13.3: Wire the flag into the report**

Find the place in `benchmark_hp_blink.py` where events are summarized and printed (search for `print(` or where `balanced` / `f1` is computed — typically near the end of `main` or `run_benchmark`). Add after the existing summary:

```python
    if args.check_periodic:
        refractory = getattr(config, "refractory_s", 1.05)
        cluster_ratio = check_periodic_cluster_ratio(
            [e.time_s for e in events], refractory
        )
        print(f"check_periodic: refractory={refractory:.2f}s cluster_ratio={cluster_ratio:.3f}")
        if cluster_ratio > 0.5:
            print("FAIL: cluster_ratio > 0.5 (periodic noise-driven firing)")
        elif cluster_ratio < 0.3:
            print("PASS: cluster_ratio < 0.3 (no periodic clustering)")
```

If the exact variable name for events differs in the file, adapt accordingly — read the surrounding code first.

- [ ] **Step 13.4: Verify**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" hp_acoustic_wave/benchmark_hp_blink.py --help 2>&1 | grep check-periodic`

Expected: shows `--check-periodic` in the help.

- [ ] **Step 13.5: Commit**

```bash
cd E:/android_projects/eye_blink_detect
git add hp_acoustic_wave/benchmark_hp_blink.py
git commit -m "feat(benchmark): add --check-periodic anti-periodic metric (phase 4.1)"
```

---

## Task 14 — Phase 4.2: Benchmark `shape` on 9 sessions

**Files:**
- Create: `hp_acoustic_wave/experiment_shape_vs_blinklistener_20260618.md`

- [ ] **Step 14.1: Run `shape` on the 3 CLAUDE.md sessions**

For each of `184117`, `190655`, `214655`, run:

```bash
cd E:/android_projects/eye_blink_detect
"/c/Program Files/Python38/python.exe" hp_acoustic_wave/benchmark_hp_blink.py \
  --session hp_acoustic_wave/sessions/hp_blink_20260617_184117 \
  --source audio --truth auto --signal-mode fmcw \
  --blink-method shape \
  --blink-twinkle-fmcw-use-intra-chirp-phase-pair \
  --check-periodic
```

(Repeat for 190655 and 214655.)

Record: events / TP / FP / FN / precision / recall / f1 / balanced / cluster_ratio.

If `--signal-mode` / `--blink-twinkle-fmcw-use-intra-chirp-phase-pair` are not valid args for `benchmark_hp_blink.py`, run without them — read `--help` first to see which FMCW flags exist. The shape detector needs `phase_pair_delta`, which requires the FMCW extraction path; check that benchmark_hp_blink.py's audio replay populates it (it should — it uses `extract_fmcw_chunk_feature`).

- [ ] **Step 14.2: Run `shape` on the 6 baseline-doc sessions**

For each of `162711`, `172844`, `173242`, `173057`, `173428`, `174829`, run the same command (substitute the session name). For `174829` (cw_single / tone), omit `--signal-mode fmcw` and `--blink-twinkle-fmcw-use-intra-chirp-phase-pair` — that session is tone mode. The shape detector falls back to `feature.phase` when `signal_mode != "fmcw"`.

- [ ] **Step 14.3: Hard-check the CLAUDE.md requirements**

For session 214655: events must be < 20 (baseline regression was 363).
For sessions 184117 and 190655: balanced_score must be >= 0.
For all 3: cluster_ratio must be < 0.3.

If any fails, record the failure and proceed to Task 15 — the failure informs the decision gate.

---

## Task 15 — Phase 4.3: Benchmark `blinklistener` on 9 sessions

**Files:**
- Append to: `hp_acoustic_wave/experiment_shape_vs_blinklistener_20260618.md`

- [ ] **Step 15.1: Run `blinklistener` on all 9 sessions**

Same command as Task 14 but with `--blink-method blinklistener`. For each session:

```bash
cd E:/android_projects/eye_blink_detect
"/c/Program Files/Python38/python.exe" hp_acoustic_wave/benchmark_hp_blink.py \
  --session hp_acoustic_wave/sessions/hp_blink_<NAME> \
  --source audio --truth auto --signal-mode fmcw \
  --blink-method blinklistener \
  --check-periodic
```

Record the same metrics.

- [ ] **Step 15.2: Hard-check**

Same checks as Task 14.3.

---

## Task 16 — Phase 4.4: Benchmark `twinkle` (baseline) on 9 sessions

**Files:**
- Append to: `hp_acoustic_wave/experiment_shape_vs_blinklistener_20260618.md`

- [ ] **Step 16.1: Run `twinkle` on all 9 sessions**

Same command but with `--blink-method twinkle --blink-twinkle-fmcw-use-intra-chirp-phase-pair` (since the launch scripts use this flag and we want a fair comparison). Record metrics.

- [ ] **Step 16.2: Hard-check**

Same checks.

---

## Task 17 — Phase 4.5: Write the experiment report + decision

**Files:**
- Create / finalize: `hp_acoustic_wave/experiment_shape_vs_blinklistener_20260618.md`

- [ ] **Step 17.1: Write the report**

Create `hp_acoustic_wave/experiment_shape_vs_blinklistener_20260618.md` with this structure:

```markdown
# Shape vs BlinkListener vs Twinkle — 9 session benchmark (2026-06-18)

## Setup

- Detectors: `shape` (new), `blinklistener` (existing), `twinkle` (baseline).
- Sessions: 184117, 190655, 214655 (CLAUDE.md) + 162711, 172844, 173242, 173057, 173428, 174829 (baseline doc).
- Anti-periodic metric: cluster_ratio = fraction of event gaps within ±0.2s of refractory_s. Target < 0.3.

## Results — per detector

| detector | session | events | TP | FP | FN | precision | recall | f1 | balanced | cluster_ratio |

(shape rows, then blinklistener rows, then twinkle rows)

## Hard-check summary

| check | shape | blinklistener | twinkle |
|---|---|---|---|
| 214655 events < 20 | ✅/❌ | ... | ... |
| 184117 balanced >= 0 | ... | ... | ... |
| 190655 balanced >= 0 | ... | ... | ... |
| all cluster_ratio < 0.3 | ... | ... | ... |

## Decision

(Based on the table: which detector wins on balanced + cluster_ratio?)
- If shape wins overall → recommend making `shape` the default `--blink-method`.
- If blinklistener wins on certain variants → recommend per-variant method selection.
- If both fail → recommend re-tuning `shape_edge_sigma_k` (try 2.0) and re-running Task 14.
```

Fill in the actual numbers from Tasks 14-16.

- [ ] **Step 17.2: Commit**

```bash
cd E:/android_projects/eye_blink_detect
git add hp_acoustic_wave/experiment_shape_vs_blinklistener_20260618.md
git commit -m "experiment: shape vs blinklistener vs twinkle on 9 sessions (phase 4)"
```

---

## Final review checklist

After all tasks complete:

- [ ] Phase 0 (Tasks 1-4): dead code gone, `BlinkDetectionConfig` is an alias for `BlinkConfig`, no import errors, existing tests pass.
- [ ] Phase 1 (Tasks 5-7): `last_ungated_coherence` attribute exists on twinkle + blinklistener detectors; app.py has dual history buffers; bottom panel shows phase trajectory (top) + ungated coherence (bottom).
- [ ] Phase 2 (Tasks 8-11): shape config fields present; `_TwinkleShapeSegmentationDetector` implemented; 5 unit tests pass; `--blink-method shape` works in both CLIs.
- [ ] Phase 3 (Task 12): audit doc written; bugs fixed if found.
- [ ] Phase 4 (Tasks 13-17): `--check-periodic` flag works; benchmark report filled with all 9 sessions × 3 detectors; hard-check decisions documented.
- [ ] No periodic firing: 214655 events < 20, cluster_ratio < 0.3 on all detectors.
