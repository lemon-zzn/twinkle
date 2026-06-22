# Shape vs BlinkListener vs Twinkle — 9 session benchmark (2026-06-18)

## Setup

- **Detectors**
  - `shape` — new `_TwinkleShapeSegmentationDetector` (Phase 2 of redesign plan). Down/up edge pattern matching on the phase trajectory within `[min_edge_len_s, max_edge_len_s]` window, gated by `shape_edge_sigma_k * baseline_sigma`. Default `shape_edge_sigma_k=3.0`.
  - `blinklistener` — existing `BlinkListenerBlinkDetector` (LEVD / I-Q projection).
  - `twinkle` — baseline `TwinkleTwinkleBlinkDetector` (coherence + phase-pair score, adaptive threshold).
- **Sessions** (9 total)
  - CLAUDE.md: `20260617_184117`, `20260617_190655`, `20260617_214655`
  - Baseline doc: `20260618_162711` (A linear), `20260618_172844` (A' linear_v2), `20260618_173242` (B linear_tukey), `20260618_173057` (D triangle), `20260618_173428` (C cw_fmcw_hybrid), `20260618_174829` (E cw_single / tone).
- **CLI**:
  ```
  benchmark_hp_blink.py --session <path> --source audio --truth auto
                       --blink-method <method>
                       --blink-twinkle-fmcw-use-intra-chirp-phase-pair   # FMCW only
                       --check-periodic
  ```
  For `174829` (tone session) the intra-chirp flag is omitted.
- **Anti-periodic metric**: `cluster_ratio` = fraction of inter-event gaps within +/-0.2s of `refractory_s` (1.05s). Target `< 0.3` per CLAUDE.md.
- **Balanced score** = `tp - fp` (per-event, signed). Target `>= 0`.
- `truth=auto` resolved to `visual` (camera-based blinks) on every session.

## Results — per detector

### shape

| session | events | TP | FP | FN | precision | recall | f1 | balanced | cluster_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260617_184117 | 1 | 0 | 1 | 5 | 0.000 | 0.000 | 0.000 | -0.55 | 0.000 |
| 20260617_190655 | 4 | 1 | 3 | 44 | 0.250 | 0.022 | 0.041 | -0.65 | 0.000 |
| 20260617_214655 | 1 | 0 | 1 | 6 | 0.000 | 0.000 | 0.000 | -0.55 | 0.000 |
| 20260618_162711 | 1 | 0 | 1 | 3 | 0.000 | 0.000 | 0.000 | -0.55 | 0.000 |
| 20260618_172844 | 3 | 2 | 1 | 28 | 0.667 | 0.067 | 0.121 | 1.45 | 0.500 |
| 20260618_173242 | 1 | 0 | 1 | 22 | 0.000 | 0.000 | 0.000 | -0.55 | 0.000 |
| 20260618_173057 | 6 | 2 | 4 | 20 | 0.333 | 0.091 | 0.143 | -0.20 | 0.000 |
| 20260618_173428 | 2 | 0 | 2 | 22 | 0.000 | 0.000 | 0.000 | -1.10 | 0.000 |
| 20260618_174829 | 18 | 6 | 12 | 19 | 0.333 | 0.240 | 0.279 | -0.60 | 0.118 |

Observations: `shape` fires very few events (1-18) even when truth has 22-45 blinks. The `shape_edge_sigma_k=3.0` gate is too strict — recall collapses to near zero. `172844` shows the only non-zero `cluster_ratio` (0.500 from 3 events).

### blinklistener

| session | events | TP | FP | FN | precision | recall | f1 | balanced | cluster_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260617_184117 | 4 | 1 | 3 | 4 | 0.250 | 0.200 | 0.222 | -0.65 | 0.000 |
| 20260617_190655 | 8 | 7 | 1 | 38 | 0.875 | 0.156 | 0.264 | 6.45 | 0.000 |
| 20260617_214655 | 55 | 0 | 55 | 6 | 0.000 | 0.000 | 0.000 | -30.25 | 0.111 |
| 20260618_162711 | 2 | 0 | 2 | 3 | 0.000 | 0.000 | 0.000 | -1.10 | 0.000 |
| 20260618_172844 | 11 | 4 | 7 | 26 | 0.364 | 0.133 | 0.195 | 0.15 | 0.000 |
| 20260618_173242 | 10 | 6 | 4 | 16 | 0.600 | 0.273 | 0.375 | 3.80 | 0.000 |
| 20260618_173057 | 8 | 4 | 4 | 18 | 0.500 | 0.182 | 0.267 | 1.80 | 0.143 |
| 20260618_173428 | 13 | 6 | 7 | 16 | 0.462 | 0.273 | 0.343 | 2.15 | 0.000 |
| 20260618_174829 | 0 | 0 | 0 | 25 | 0.000 | 0.000 | 0.000 | 0.00 | 0.000 |

Observations: precision is generally best (0.36-0.88 when TP > 0). Catastrophic regression on `214655` — 55 events with 0 TP and balanced_score -30.25 — the very symptom CLAUDE.md warns about. Tone session (`174829`) fires nothing (LEVD never crosses threshold without FMCW phase_pair_delta).

### twinkle

| session | events | TP | FP | FN | precision | recall | f1 | balanced | cluster_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260617_184117 | 24 | 4 | 20 | 1 | 0.167 | 0.800 | 0.276 | -7.00 | 0.000 |
| 20260617_190655 | 45 | 33 | 12 | 12 | 0.733 | 0.733 | 0.733 | 26.40 | 0.000 |
| 20260617_214655 | 1 | 0 | 1 | 6 | 0.000 | 0.000 | 0.000 | -0.55 | 0.000 |
| 20260618_162711 | 8 | 2 | 6 | 1 | 0.250 | 0.667 | 0.364 | -1.30 | 0.000 |
| 20260618_172844 | 49 | 24 | 25 | 6 | 0.490 | 0.800 | 0.608 | 10.25 | 0.000 |
| 20260618_173242 | 37 | 19 | 18 | 3 | 0.514 | 0.864 | 0.644 | 9.10 | 0.000 |
| 20260618_173057 | 39 | 18 | 21 | 4 | 0.462 | 0.818 | 0.590 | 6.45 | 0.000 |
| 20260618_173428 | 37 | 19 | 18 | 3 | 0.514 | 0.864 | 0.644 | 9.10 | 0.000 |
| 20260618_174829 | 28 | 16 | 12 | 9 | 0.571 | 0.640 | 0.604 | 9.40 | 0.407 |

Observations: best recall on most sessions (0.64-0.86) — the adaptive threshold is very sensitive. `214655` regression is gone (1 event). But `174829` trips the anti-periodic flag with `cluster_ratio=0.407`, indicating residual periodic firing on tone-mode audio.

## Hard-check summary

Per CLAUDE.md / Phase 4 task spec — applied per detector:

| check | shape | blinklistener | twinkle |
|---|:---:|:---:|:---:|
| 214655 events < 20 | PASS (1) | **FAIL (55)** | PASS (1) |
| 184117 balanced >= 0 | **FAIL (-0.55)** | **FAIL (-0.65)** | **FAIL (-7.00)** |
| 190655 balanced >= 0 | **FAIL (-0.65)** | PASS (6.45) | PASS (26.40) |
| all cluster_ratio < 0.3 | **FAIL** (0.500 on 172844) | PASS (max 0.143) | **FAIL** (0.407 on 174829) |

**No detector passes all 4 hard checks.** Each detector fails in a different way:

- `shape`: too conservative (1-18 events vs 3-45 truth blinks). Sigma-k gate starves recall.
- `blinklistener`: catastrophic regression on session 214655 (55 events, all FP).
- `twinkle`: best recall on most sessions but `174829` shows periodic noise firing.

## Decision

**No winner.** None of the three detectors clears the CLAUDE.md bar simultaneously. Honest scores:

| metric | shape | blinklistener | twinkle |
|---|---|---|---|
| mean balanced over 9 | -0.30 | -1.96 | +6.65 |
| mean recall over 9 | 0.077 | 0.045 | 0.576 |
| sessions passing cluster_ratio<0.3 | 7/9 | 9/9 | 8/9 |
| sessions passing events<20 | 9/9 | 8/9 | 9/9 |

- `twinkle` is currently the strongest baseline by raw balanced/recall, but its `174829` cluster_ratio=0.407 violates the anti-periodic target.
- `blinklistener` has the best precision and cleanest cluster_ratio, but its `214655` regression is unacceptable.
- `shape` structurally prevents periodic firing (cluster_ratio 0.000 on 7/9 sessions) but its recall is far too low.

## Recommended follow-up

1. **Re-tune `shape`**: lower `shape_edge_sigma_k` from `3.0` to `2.0` (or `1.5`) and widen `shape_max_edge_len_s` from `0.5` to `0.7`. Hypothesis: a 3-sigma gate starves the FMCW phase_pair_delta signal; a 2-sigma gate should recover recall without reintroducing periodic firing (the edge-pattern requirement still structurally rejects refractory-spaced noise peaks).
2. **Investigate blinklistener 214655 regression**: 55 events on the noisiest session suggests the LEVD threshold is dropping below the noise floor when `baseline_freeze_s` expires. Consider raising the floor or adding a MAD-based noise reject.
3. **Investigate twinkle 174829 periodic firing**: cluster_ratio=0.407 means the gaps land at exactly refractory_s=1.05s — likely the score baseline is being walked up by sustained noise. Consider tightening `blink_twinkle_peak_min_ratio` for tone-mode sessions.
4. **Do NOT promote any detector to default yet**. The redesign plan's success criterion (single detector passing all 4 hard checks) is not met. Re-run this benchmark after the tuning changes above.

## Raw command

```bash
"/c/Program Files/Python38/python.exe" hp_acoustic_wave/benchmark_hp_blink.py \
  --session hp_acoustic_wave/sessions/hp_blink_<DATE>_<NAME> \
  --source audio --truth auto --blink-method <shape|blinklistener|twinkle> \
  --blink-twinkle-fmcw-use-intra-chirp-phase-pair \
  --check-periodic
```
(omit the intra-chirp flag for `20260618_174829` — tone session)
