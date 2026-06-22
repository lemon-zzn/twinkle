#!/usr/bin/env bash
set -euo pipefail
# Live neural blink detector.
# Uses the CPU-only NumPy model trained from sessions/* audio.
#
# Usage:
#   ./run_neural_blink_live.sh
#   ./run_neural_blink_live.sh --input-device N --output-device M

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python"
  else
    PYTHON="$(command -v python3 || command -v python)"
  fi
fi

MODEL_PATH="${NEURAL_BLINK_MODEL:-$SCRIPT_DIR/models/neural_blink_model.npz}"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "missing neural model: $MODEL_PATH" >&2
  echo "train it first: $PYTHON train_neural_blink.py --session-root sessions --source audio --output models/neural_blink_model.npz" >&2
  exit 1
fi

"$PYTHON" run_hp_wave_detector.py \
  --mode blink \
  --blink-method neural \
  --neural-model-path "$MODEL_PATH" \
  --signal-mode fmcw \
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
