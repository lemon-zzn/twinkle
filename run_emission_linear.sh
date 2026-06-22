#!/usr/bin/env bash
set -euo pipefail
# Variant A: linear chirp (baseline). Same emission as run_baseline.sh.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
  else
    PYTHON="$(command -v python3 || command -v python)"
  fi
fi

"$PYTHON" run_hp_wave_detector.py \
  --mode blink \
  --blink-method twinkle \
  --signal-mode fmcw \
  --blink-twinkle-fmcw-use-intra-chirp-phase-pair \
  --fmcw-emission linear \
  --fmcw-range-bin 15 \
  --fmcw-freq-low 17000 \
  --fmcw-freq-high 23000 \
  --fmcw-chirp-duration 0.05 \
  --fmcw-lowpass-cutoff 5000 \
  --amplitude 0.2 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-backend any \
  --camera-fps 30 \
  --visual-ear-threshold 0.22 \
  "$@"
