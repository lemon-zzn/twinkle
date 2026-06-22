#!/usr/bin/env bash
set -euo pipefail
# 最佳 baseline 启动脚本（FMCW + intra-chirp phase-pair）
# 依据：experiment_preprocess_20260618.md 验证的 baseline
#   - twinkle_fmcw_use_intra_chirp_phase_pair=True（走 phase-pair 路径，主信号）
#   - blink_min_score=0.12（平衡 FP 和召回）
#   - range_bin=15 ≈ 40cm 距离（参考 CIS3990 Lab 2 §3.1）
#   - amplitude=0.2（SNR 够，不削顶）
#
# 用法：./run_baseline.sh
# 如需指定设备：./run_baseline.sh --input-device N --output-device M

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
  --blink-min-score 0.12 \
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
