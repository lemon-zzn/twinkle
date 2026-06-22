#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
  else
    PYTHON="$(command -v python3 || command -v python)"
  fi
fi

DURATION="${DURATION:-12}"
RANGE_BIN="${RANGE_BIN:-16}"
AMPLITUDE="${AMPLITUDE:-0.2}"
DROP_SECONDS="${DROP_SECONDS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-refer/FMCW/outputs/emission_variants}"
RECORDING_DIR="${RECORDING_DIR:-refer/FMCW/live/emission_variants}"
EMISSIONS=("linear" "linear_tukey" "cw_fmcw_hybrid" "triangle")

for emission in "${EMISSIONS[@]}"; do
  echo
  echo "=== FMCW breathing emission: ${emission} ==="
  echo "Prepare posture now. Press Enter to start, or Ctrl-C to stop."
  read -r _
  "$PYTHON" fmcw_breathing_live.py \
    --duration "$DURATION" \
    --range-bin "$RANGE_BIN" \
    --drop-seconds "$DROP_SECONDS" \
    --invert-phase \
    --fmcw-emission "$emission" \
    --amplitude "$AMPLITUDE" \
    --no-show \
    --recording-dir "$RECORDING_DIR" \
    --output-dir "$OUTPUT_DIR"
done
