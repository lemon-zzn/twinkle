# FMCW 发射变体 baseline（2026-06-18）

> 5 个发射变体录制的 session 路径 + benchmark 分数 + 信号层统计。
> 后续优化以此为对比基线。

## 录制的 sessions

| Variant | 标签 | session 路径 | signal_mode | emission | tone_hz |
|---|---|---|---|---|---|
| A | linear | `sessions/hp_blink_20260618_162711` | fmcw | linear | — |
| A' | linear_v2 | `sessions/hp_blink_20260618_172844` | fmcw | linear | — |
| B | linear_tukey | `sessions/hp_blink_20260618_173242` | fmcw | linear_tukey | — |
| C | cw_fmcw_hybrid | `sessions/hp_blink_20260618_173428` | fmcw | cw_fmcw_hybrid | — |
| D | triangle | `sessions/hp_blink_20260618_173057` | fmcw | triangle | — |
| E | cw_single | `sessions/hp_blink_20260618_174829` | tone | cw_single | 18500 |

> 注：A (162711) 录得只有 3 个 visual blink，太短；A' (172844) 是 30 个，更长更可靠。
> cw_single (E) 用的是 tone 路径，不是 FMCW。

## Benchmark — 同一 config 跑全部 FMCW 变体

Config（"live"，已在 launch scripts 中）：
```
--blink-twinkle-fmcw-use-intra-chirp-phase-pair
--blink-twinkle-fmcw-min-score 0.25
--blink-twinkle-fmcw-min-smoothness 0.6
--blink-twinkle-fmcw-min-deviation 0.30
```

cw_single (E) 用 tone-mode config：
```
--blink-method twinkle --blink-min-score 0.006 --blink-refractory 1.05
```

| Variant | session | events | TP | FP | FN | precision | recall | f1 | balanced |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear (A) | 162711 | 5 | 1 | 4 | 2 | 0.200 | 0.333 | 0.250 | -1.20 |
| linear_v2 (A') | 172844 | 13 | 6 | 7 | 24 | 0.462 | 0.200 | 0.279 | 2.15 |
| **linear_tukey (B)** | 173242 | 22 | **14** | 8 | 8 | **0.636** | **0.636** | **0.636** | **9.60** |
| cw_fmcw_hybrid (C) | 173428 | 0 | 0 | 0 | 22 | — | 0.000 | 0.000 | 0.00 |
| triangle (D) | 173057 | 4 | 2 | 2 | 20 | 0.500 | 0.091 | 0.154 | 0.90 |
| cw_single (E) | 174829 | 28 | 16 | 12 | 9 | 0.571 | 0.640 | 0.604 | 9.40 |

**最优发射方式：`linear_tukey` (B)** — f1=0.636, recall=0.636, balanced=9.60。

cw_single (E) 作 reference 也很好（f1=0.604），用户已知该实现正确。
cw_fmcw_hybrid (C) 在 live config 下 0 事件 — hybrid 帧交错破坏了 intra-chirp coherence 计算，dev=0.30 直接全归零。如果要用 hybrid，需要单独调参（去掉 deviation gate 或换 trajectory 源）。

## Benchmark — baseline config（min_score=0.12，无 gates，用于对比）

```
--blink-twinkle-fmcw-use-intra-chirp-phase-pair
--blink-twinkle-fmcw-min-score 0.12
（min_smoothness=0, min_deviation=0 关闭）
```

| Variant | session | events | TP | FP | FN | precision | recall | f1 | balanced |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| linear (A) | 162711 | 8 | 2 | 6 | 1 | 0.250 | 0.667 | 0.364 | -1.30 |
| linear_v2 (A') | — | — | — | — | — | — | — | — | — |
| **linear_tukey (B)** | 173242 | 37 | **20** | 17 | 2 | 0.541 | **0.909** | **0.678** | **10.65** |
| cw_fmcw_hybrid (C) | 173428 | 37 | 19 | 18 | 3 | 0.514 | 0.864 | 0.644 | 9.10 |
| triangle (D) | 173057 | 23 | 10 | 13 | 12 | 0.435 | 0.455 | 0.444 | 2.85 |

> baseline 下事件数明显多（37 vs 22），周期误判严重。这就是实时看到"周期输出"的来源。

## 信号层统计

`compare_variants_stats.py` 输出（range-bin 15，前 800 个 chunk）：

| Variant | n_blinks | amp_mean | **amp_p2p** | dphase_q99 | blink_dphase_med |
|---|---:|---:|---:|---:|---:|
| linear (A) | 3 | 0.0355 | **0.967** | 2.82 | 0.41 |
| linear_v2 (A') | 30 | 0.0215 | 0.098 | 2.81 | 0.34 |
| **linear_tukey (B)** | 22 | 0.0136 | **0.045** | 2.78 | 0.24 |
| triangle (D) | 22 | 0.0225 | 0.144 | 2.76 | 0.12 |
| cw_fmcw_hybrid (C) | 22 | 0.0397 | 0.089 | **3.14** | **2.47** ← π，phase wrapping |
| cw_single (E) | 25 | 0.0234 | 0.055 | 1.67 | 0.06 |

**关键发现：**
- `linear` (A) 的 `amp_p2p=0.967` — amplitude 摆动幅度极大（chirp 起止不连续 → range-bin 能量不稳定），淹没 phase 信号。
- `linear_tukey` (B) 的 `amp_p2p=0.045` — 比 vanilla linear **稳定 22 倍**。Tukey 窗平滑 chirp 边界，range-bin 能量稳定。
- `cw_fmcw_hybrid` (C) 的 `blink_dphase_med=2.47` ≈ π — phase wrapping artifact，因为 CW/chirp 帧交错导致 range-bin phase 在每隔一帧跳π。这是 hybrid 设计的固有问题，不建议用于眨眼检测。
- `cw_single` (E) `dphase_q99=1.67` 最低 — 单频信号最稳定，但 blink 引起的 phase 变化也最弱（0.06），需要更敏感的 trajectory 源。

## 诊断图

- `docs/diagnostics/variant_comparison.png` — TX 频谱 + range-bin magnitude/phase + phase-step 分布（每个变体一行）
- `docs/diagnostics/raw_audio_waveforms.png` — **原始麦克风音频波形**（amplitude 包络 + 30ms 样本级），证明底层 audio.wav 从不"平"
