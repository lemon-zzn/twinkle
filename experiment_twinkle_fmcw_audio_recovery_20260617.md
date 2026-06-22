# Twinkle FMCW Audio Recovery (2026-06-17)

## Scope
- Code-only optimization (no CLI hyperparameter search)
- Target sessions only:
  - `sessions/hp_blink_20260617_184117`
  - `sessions/hp_blink_20260617_190655`
  - `sessions/hp_blink_20260617_214655` (empty/strict false-positive guard)
- Goal:
  - recover recall on `184117/190655`
  - keep `214655` at zero periodic blink guessing

## Root Cause Found
The all-zero regression in `source=audio` was caused by multiple hard rejects stacking:
1. FMCW spatial spread hard reject frequently failed after reprocess.
2. Phase-pair library low quality branch could hard-zero score.
3. Segment gate then had too few valid candidates to emit events.

Result was not "no motion"; it was over-suppression.

## Code Changes
File: `E:\android_projects\eye_blink_detect\hp_acoustic_wave\blink_detector.py`

1. Spatial spread gate:
- changed from hard reject to conditional attenuation for non-severe diffuse cases
- severe diffuse still hard rejected
- added metrics:
  - `twinkle_fmcw_spatial_penalty`
  - `twinkle_fmcw_spatial_likely_large_motion`

2. Phase-pair library branch:
- removed low-quality hard-zero path
- replaced with attenuation factor so trajectory is not globally disabled

3. Segment/peak logic:
- improved local-peak detection for plateau edges
- improved short-segment coarse/fine shape acceptance
- FMCW single-candidate acceptance now requires high consistency

## Verification
Command:

```powershell
python -m pytest hp_acoustic_wave/tests/test_blink_algorithms.py hp_acoustic_wave/tests/test_fmcw_wave_integration.py hp_acoustic_wave/tests/test_benchmark_fmcw.py hp_acoustic_wave/tests/test_session_io.py -q
```

Result: `47 passed`

## Benchmark (visual truth)

Command logic:
- `BlinkDetectionConfig(method="twinkle")`
- compare `source=features` and `source=audio`
- truth = `visual`

### Results

| Source | Session | Events | Hits | Recall | Unexplained | FP/Event |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| features | `184117` | 7 | 2/5 | 40.0% | 5 | 71.4% |
| features | `190655` | 14 | 16/45 | 35.6% | 3 | 21.4% |
| features | `214655` | 0 | 0/6 | 0.0% | 0 | 0.0% |
| audio | `184117` | 7 | 2/5 | 40.0% | 5 | 71.4% |
| audio | `190655` | 20 | 22/45 | 48.9% | 6 | 30.0% |
| audio | `214655` | 0 | 0/6 | 0.0% | 0 | 0.0% |

## Conclusion
1. The `source=audio` all-zero regression is fixed on `184117/190655`.
2. `214655` stays at zero events (no periodic blink guessing).
3. Recall improved, but false positives on `184117` remain high.
4. Next code step should focus on FP suppression without collapsing recall (especially in post-17s region of `184117`).
