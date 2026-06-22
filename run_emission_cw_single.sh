#!/usr/bin/env bash
set -euo pipefail
# Variant E: single-frequency CW (known-correct reference baseline)
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
  --signal-mode tone \
  --frequency 18500 \
  --amplitude 0.2 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-backend any \
  --camera-fps 30 \
  --visual-ear-threshold 0.22 \
  "$@"
