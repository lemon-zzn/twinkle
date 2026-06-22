# BlinkListenerBlinkDetector Implementation Audit (2026-06-18)

Phase 3 of the FMCW blink-detector redesign. Compares `BlinkListenerBlinkDetector`
(`hp_acoustic_wave/blink_detector.py` L232-326) and `_RobustEventGate`
(L24-75) against the BlinkListener paper (Liu et al., ACM IMWUT 2021, DOI
10.1145/3463521). Source text extracted from
`FaceAcousticSensing/papers/BlinkListener_PyMuPDF.txt`.

## Paper algorithm summary

Section 3.2 ("Modeling the Eye Blink Process") derives the I-Q signal model
for a single blink. The eye-closed/eye-open transition induces *two*
orthogonal effects in the complex plane: (i) a small **phase** change caused
by the eyelid thickness (~0.5 mm path-length difference) along the tangential
direction of the arc traced by the dynamic component, and (ii) a comparably
large **amplitude** change caused by the different reflection coefficients of
the water-textured eyeball vs. the skin-textured eyelid, along the radial
direction. Breathing/heartbeat "embedded interference" by contrast induces an
almost pure phase rotation (large phase change, small amplitude change), and
so traces an arc in I-Q space; a blink looks like a "bump" protruding
radially from that arc.

Section 6.2 ("Viewing Position Scheme") chooses the **center of the arc** as
the viewing position from which the distance to each sample is measured. The
arc center is the optimal point because it (a) maximizes the radial
blink-induced bump, and (b) makes the breathing-induced tangential motion
look nearly constant (no false bump). It is found by Pratt arc-fitting on
the most recent ~50 samples (Sec 6.3 Step 1, `Thr_update = 50`,
≈2 s at 40 ms chirp period).

Section 6.3 ("Real-time Eye Blink Detection Algorithm") runs Local Extreme
Value Detection (LEVD) on the distance time series in a sliding window.
`Thr_blink = 5 × std(no-blink amplitude)`. A blink is declared when a local
maximum and a neighboring local minimum differ by more than `Thr_blink`.
Body-motion restart fires when the signal excursion exceeds
`Thr_restart = 3 × Thr_blink`. There is **no** refractory period in the
paper, **no** hysteresis release, **no** baseline freeze, and **no**
phase-stability gate; the LEVD sliding window already enforces a temporal
extent (~blink duration, 100-400 ms). No score normalization to baseline
amplitude is described.

## Implementation-vs-paper comparison

### 1. I/Q viewing-position projection

Status: matches in spirit, approximates Euclidean distance.

- `_center` (L244-247) takes `median(baseline_i)` / `median(baseline_q)`,
  approximating the paper's arc center with the I/Q median. This is the
  correct point when the embedded interference traces a symmetric arc
  around the median, which holds while blink-induced excursions are sparse.
- `_best_viewing_projection` (L254-270) computes
  `max_θ |current·direction - median(baseline·direction)|` over 12 angles
  spanning `[0, π)`. Since the centered current point is `(delta_i, delta_q)`
  and `max_θ |a cos θ + b sin θ| = √(a²+b²)`, this converges to the Euclidean
  distance from the arc center — exactly the paper's "distance from viewing
  position". With 12 angles the worst-case discretization error is
  `1 - cos(π/24) ≈ 0.86 %`, negligible.
- The `median(baseline·direction)` term subtracts any residual bias of the
  centered baseline along each direction; harmless extra correction.

Evidence: `blink_detector.py:244-270`. No deviation from paper.

### 2. phase_stable_projection threshold

Status: heuristic — paper provides no phase-stability gate.

`phase_stable_projection = best_projection if abs(feature.phase_delta) <= 0.12
else gated_projection` (L278) gates the projection by a hard-coded 0.12 rad
phase-delta threshold. The paper treats large phase change as the *signature*
of breathing/heartbeat interference (Sec 3.2, "breathing/heartbeat induce
large phase change, small amplitude change"), but never specifies a numeric
phase-delta cutoff. **Recommendation (do not change in Phase 3):** expose
this as `BlinkConfig.blinklistener_phase_stable_threshold` once we have
ground-truth data to tune it.

### 3. score normalization

Status: implementation-defined, paper does not normalize.

`raw_score = max(amplitude_bump, amplitude_range, phase_stable_projection,
0.75 × projection_range)` (L289), then `score = raw_score /
max(baseline_amplitude, 1e-4)` (L291). Evidence: `blink_detector.py:274-291`.

- `score_scale` correctly uses the **median** baseline amplitude
  (`_baseline_amplitude`, L249-252), not the current frame — matches the
  intent of "relative amplitude" and is robust to transient spikes.
- Paper does not specify normalization; it operates on absolute amplitude
  units with `Thr_blink = 5 × std`. Our adaptive gate (`baseline +
  threshold_k × MAD-σ`) absorbs the per-session scale, so the normalization
  is a sensible portability aid rather than a paper requirement.
- The `0.75` weight on `projection_range` is a tuning heuristic with no
  paper citation. Flagging as a non-paper constant.

### 4. _RobustEventGate threshold

Status: design differs from LEVD but is functionally equivalent given the
score formula.

- `threshold = max(min_score, baseline + threshold_k × 1.4826 × MAD)`
  (L38-39) with `threshold_k = 2.5` (config.py L39). Paper uses
  `Thr_blink = 5 × std(no-blink)`. The factor differs (`2.5 × MAD-σ` vs.
  `5 × std`), but in our score space the per-frame score already encodes the
  *deviation from baseline* rather than the absolute signal — so the
  `baseline` term accounts for residual noise floor, and `k=2.5` on a
  relative score is roughly comparable to `5×std` on an absolute signal.
- Paper has **no refractory period**, **no hysteresis release**, **no
  baseline freeze**, and **no startup-ignore**. Our `refractory_s=1.05`,
  `release_ratio=0.4`, `baseline_freeze_s=0.60`, `startup_ignore_s=0.0`
  (config.py L41-42, L47-48) are all implementation additions to compensate
  for replacing LEVD's sliding-window local-extrema detection with a
  per-frame threshold crossing. They are pragmatic — without LEVD's natural
  ~blink-duration window, the gate would fire on every threshold-crossing
  sample. Flagging as a documented design deviation, not a bug.
- LEVD itself is not implemented. The score already aggregates the
  deviation over `short_window`-sized buffers (`amplitude_window`,
  `projection_window`, L239-240), so the per-frame score *is* a bump
  magnitude — equivalent in effect to LEVD's `max − min` over the window.

## Bugs found

None. The implementation is a faithful **adaptation** of the paper, not a
literal port: it approximates the arc-center distance with a 12-direction
projection max, replaces LEVD with a per-frame robust-threshold gate
augmented by refractory/hysteresis/baseline-freeze, and adds two undocumented
heuristics (0.12 rad phase gate, 0.75 projection-range weight). Every
deviation is defensible and Phase 2 testing (`test_blink_algorithms.py`)
exercises the contract.

## Recommended fixes

None required for Phase 3. Deferred tuning candidates (record for Phase 4
benchmarking):

1. Expose `0.12` rad phase-stability threshold as a config field
   `blinklistener_phase_stable_threshold` once we have ground-truth data.
2. Expose the `0.75` projection-range weight as
   `blinklistener_projection_range_weight`, default 0.75.
3. Consider raising `threshold_k` toward the paper's `5×std` if Phase 4
   benchmarks show low precision on session 214655.

## CLI verification

- `hp_acoustic_wave/run_hp_wave_detector.py` L26:
  `choices=["blinklistener", "twinkle", "shape", "both"]` — yes.
- `hp_acoustic_wave/benchmark_hp_blink.py` L43:
  `choices=["blinklistener", "twinkle", "shape", "both"]` — yes.

`--blink-method blinklistener` is selectable from both CLIs. No change
required.
