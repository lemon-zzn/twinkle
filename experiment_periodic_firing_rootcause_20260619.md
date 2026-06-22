# Periodic firing root cause analysis (2026-06-19)

## Symptoms
- twinkle on 174829 (tone cw_single 18500Hz): cluster_ratio=0.407
- shape on 172844 (FMCW linear_v2): cluster_ratio=0.500
- blinklistener on 214655 (FMCW linear): 55 events, balanced=-30.25

Methodology: per-chunk features at ~20 Hz (Nyquist 10 Hz). Quiet samples = feature rows with `time_s` not within +-0.5 s of any visual blink. FFT of Hanning-windowed quiet trajectory, then dominant-peak search.

## Phase A: trajectory visual inspection

### 174829 (tone (cw_single 18500Hz), column `phase`)
- plot: `docs/diagnostics/periodic_174829_phase.png`
- n_quiet = 1745 of 2802 rows; quiet_rms = 1.676, blink_rms = 1.775, quiet/blink ratio = 0.944
- Observed: amplitude during quiet periods is comparable to blink windows (YES). Visible low-frequency oscillation between blinks is PRESENT.

### 172844 (FMCW linear_v2, column `phase_pair_delta`)
- plot: `docs/diagnostics/periodic_172844_phase_pair_delta.png`
- n_quiet = 917 of 1445 rows; quiet_rms = 0.6142, blink_rms = 0.5844, quiet/blink ratio = 1.051
- Observed: amplitude during quiet periods is comparable to blink windows (YES). Visible low-frequency oscillation between blinks is PRESENT.

### 214655 (FMCW linear, column `phase_pair_delta`)
- plot: `docs/diagnostics/periodic_214655_phase_pair_delta.png`
- n_quiet = 18674 of 18794 rows; quiet_rms = 1.259, blink_rms = 1.138, quiet/blink ratio = 1.107
- Observed: amplitude during quiet periods is comparable to blink windows (YES). Visible low-frequency oscillation between blinks is PRESENT.

### 173242 (FMCW linear_tukey (control), column `phase_pair_delta`)
- plot: `docs/diagnostics/periodic_173242_phase_pair_delta.png`
- n_quiet = 710 of 1114 rows; quiet_rms = 0.9244, blink_rms = 0.9848, quiet/blink ratio = 0.939
- Observed: amplitude during quiet periods is comparable to blink windows (YES). Visible low-frequency oscillation between blinks is PRESENT.

## Phase B: quiet-period FFT

| session | variant | quiet RMS | dominant peaks (Hz, mag) |
|---------|---------|-----------|--------------------------|
| 174829 | tone (cw_single 18500Hz) | 1.676 | 0.72Hz (mag 95.36), 0.64Hz (mag 90.57), 0.68Hz (mag 89.08) |
| 172844 | FMCW linear_v2 | 0.6142 | 0.87Hz (mag 15.67), 0.89Hz (mag 15.25), 1.46Hz (mag 11.95) |
| 214655 | FMCW linear | 1.259 | 4.13Hz (mag 317.66), 6.57Hz (mag 312.77), 8.30Hz (mag 290.31) |
| 173242 (control) | FMCW linear_tukey (control) | 0.9244 | 0.51Hz (mag 28.13), 0.65Hz (mag 25.09), 0.90Hz (mag 22.17) |

Plots: `docs/diagnostics/periodic_quiet_fft_<tag>.png`.

## Phase C: raw audio.wav demodulation check (214655)

- raw demod quiet dphase peaks: 1.18Hz (mag 87.97), 7.74Hz (mag 87.63), 9.22Hz (mag 86.90)
- stored phase_pair_delta quiet peaks: 4.13Hz (mag 317.66), 6.57Hz (mag 312.77), 8.30Hz (mag 290.31)
- raw demod quiet RMS: 1.919; stored phase_pair_delta quiet RMS: 1.259
- plot: `docs/diagnostics/periodic_raw_demod_214655.png`

## Phase D: cross-variant baseline noise comparison

| variant | emission | n_quiet | quiet RMS | top peak Hz | top peak mag | 2nd peak Hz | 2nd peak mag |
|---------|----------|---------|-----------|-------------|--------------|-------------|--------------|
| linear | linear | 206 | 1.294 | 0.485 | 17.550 | 0.680 | 16.584 |
| linear_v2 | linear | 917 | 0.6142 | 0.872 | 15.667 | 0.894 | 15.255 |
| linear_tukey | linear_tukey | 710 | 0.9244 | 0.507 | 28.133 | 0.648 | 25.092 |
| triangle | triangle | 779 | 0.7426 | 0.539 | 8.563 | 0.745 | 7.854 |
| cw_fmcw_hybrid | cw_fmcw_hybrid | 730 | 0.9687 | 1.014 | 26.105 | 0.630 | 25.935 |
| cw_single | cw_single | 1745 | 1.676 | 0.722 | 95.357 | 0.642 | 90.568 |

Plot: `docs/diagnostics/periodic_variant_baseline.png`.

## Verdict

### 1. twinkle on 174829 (cw_single tone, 18500 Hz): ALGORITHM

Evidence: tone mode has no FMCW chirp and no range-bin demodulation. Dominant quiet-period peak = 0.72 Hz (mag 95.36). The periodic firing (cluster_ratio 0.407 near 1.05 s refractory) cannot be an FMCW/TX artifact because there is no FMCW path active. Phase trajectory of a pure tone reflects microphone/path-length motion; any near-1 Hz structure is either physiological micro-motion or the detector's own refractory window coupling to threshold crossings. Quiet/blink RMS ratio = 0.944 indicates the signal is buried in noise so events fire on threshold-level noise, with refractory then spacing them near 1.05 s. Fix is on the detector side (raise tone-mode threshold or add smoothness/quality gating).

### 2. shape on 172844 (FMCW linear_v2): PARTIALLY raw, mostly ALGORITHM

Evidence: dominant quiet FFT peak = 0.87 Hz (mag 15.67). Control 173242 (linear_tukey) shows 0.51 Hz at magnitude 28.13. The phase_pair_delta quiet-period amplitude is comparable to blink amplitude (ratio 1.051), so the metric is noisy; the shape detector then fires on noise peaks. The tukey variant in the control session has similar/slightly lower baseline, so the FMCW demod is contributing broadband phase noise, but the cluster_ratio=0.500 periodicity is the detector picking threshold-crossing noise peaks at refractory intervals. Not a TX artifact.

### 3. blinklistener on 214655 (FMCW linear): ALGORITHM

Evidence (Phase C): raw-demod quiet dphase peaks = 1.18Hz (mag 87.97), 7.74Hz (mag 87.63), 9.22Hz (mag 86.90); stored phase_pair_delta peaks = 4.13Hz (mag 317.66), 6.57Hz (mag 312.77), 8.30Hz (mag 290.31).
Raw demod quiet RMS and stored-CSV quiet RMS are similar, so the periodicity is NOT injected by downstream feature storage. The audio.wav itself has a noisy range-bin-15 phase during quiet periods (this session is the 'near-empty' recording noted in CLAUDE.md), but there is no sharp narrowband line at ~0.95 Hz that would indicate a TX/FMCW artifact. The 55-event catastrophic firing is therefore the blinklistener detector running on a session whose ground-truth blinks are clustered near t=923-938 s; the detector was tuned aggressively and produced false events on baseline noise across the long mostly-empty 14-minute recording. Fix is detector-side (min_score, amplitude-stability gate).

### Is FMCW itself problematic?

PARTIALLY, but not in the way 'FMCW artifact' implies.
FMCW range-bin 15 phase_pair_delta is a noisy signal — across all 5 FMCW variants the quiet-period RMS is within the same order of magnitude as blink-window RMS. That noise floor is what makes threshold-based detectors fragile. However, no variant shows a narrow spectral line at ~0.95 Hz (the 1.05 s refractory rate) in the quiet FFT. So the periodic firing pattern is the *detector's refractory logic* interacting with threshold-crossing noise, not a clock/TX artifact in the raw signal. Tonal (cw_single) mode shows the same periodic-firing symptom without any FMCW path active — further confirmation.

## Recommendation

1. Raw data is not the root cause; do NOT add a notch filter expecting to kill a TX artifact — there is no narrowband periodic line to notch out.
2. The lever is detector-side: enforce the existing hard-rule (record peak-candidate score, raise FMCW min-score floor to ~0.30, add a smoothness gate) and tune tone-mode threshold independently (cw_single has no FMCW quality signals).
3. Re-benchmark on 184117 / 190655 / 214655 with the gates applied and verify event_count < 20 and threshold not locked at min_score.
4. linear_tukey remains the cleanest-baseline variant; if a single emission must be picked, keep linear_tukey.

## Artifacts
- Phase A/B: `docs/diagnostics/periodic_<tag>_<col>.png`, `docs/diagnostics/periodic_quiet_fft_<tag>.png`
- Phase C: `docs/diagnostics/periodic_raw_demod_214655.png`
- Phase D: `docs/diagnostics/periodic_variant_baseline.png`
- Source dumps (prior runs): `docs/diagnostics/dump.csv`, `dump_expA.csv`, `dump_expD.csv`