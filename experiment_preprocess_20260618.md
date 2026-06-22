# FMCW 预处理消融实验记录

日期：2026-06-18
目标：参考 `refer/FMCW/AIradar_processing.py` 和 `fmcw_breathing.py`，逐个验证
DSP 预处理改进对眨眼检测的影响。

测试 session：184117（5 blink, 41s）、190655（45 blink, 72s）、214655（6 blink, 940s，长安静期）

## Baseline（当前 dsp.py，无窗、无 DC 抑制、EMA 背景）

| Session | events | blink_hits | unexplained | nonblink | balanced_score |
|---------|--------|------------|-------------|----------|----------------|
| 184117  | 9      | 3/5        | 6           | 6        | -0.30          |
| 190655  | 13     | 11/45      | 4           | 4        | 8.80           |
| 214655  | 0      | 0/6        | 0           | 0        | 0.00           |

配置：`BlinkDetectionConfig(method="twinkle", twinkle_fmcw_use_intra_chirp_phase_pair=True, twinkle_fmcw_min_score=0.12)`

## 实验 A：range FFT 前 Hamming 窗

修改：`dsp.py:extract_fmcw_chunk_feature` 在 `np.fft.rfft(lowpassed)` 前乘 `np.hamming(usable)`。

| Session | events | hits | FP | score | vs baseline |
|---------|--------|------|-----|-------|------------|
| 184117  | 9      | 2/5  | 7   | -1.85 | 更差 (-1.55) |
| 190655  | 11     | 12/45 | 3   | 10.35 | 更好 (+1.55) |
| 214655  | 0      | 0/6  | 0   | 0.00  | 无变化      |

特征级分析（diagnostic_dump）：
- intra-chirp phase-pair（`ppd`、`sm9`）**几乎无变化**——因为 Hamming 窗只影响 range FFT 路径，phase-pair 路径（`complex_baseband = rx * exp(-j*chirp_phase)`）不受影响。
- cross-chirp range-bin phase（`rb_sm9`）blink 窗口 0.32→0.37（略升），non-blink 仍为 0.00。
- `rb_dev9` blink 窗口 0.13→0.08（偏差下降，因为 Hamming 窗压低旁瓣能量）。

**结论：放弃。** 检测主信号走 phase-pair 路径，range FFT 加窗对它无影响。对 range-bin phase 有轻微改善但不足以改变检测结果。184117 变差说明 range-bin 幅度/相位的小扰动反而干扰了 amplitude gate。

## 实验 B：DC bin 抑制

修改：`dsp.py` 在 rfft 后立即 `complex_bins[0] = 0.0`，在背景减除前。

| Session | events | hits | FP | score | vs baseline |
|---------|--------|------|-----|-------|------------|
| 184117  | 9      | 3/5  | 6   | -0.30 | 无变化      |
| 190655  | 13     | 11/45 | 4   | 8.80  | 无变化      |
| 214655  | 0      | 0/6  | 0   | 0.00  | 无变化      |

**结论：放弃。** DC bin（bin 0）距离我们的目标 bin（bin 15）很远，置零 bin 0 不影响 bin 15 的复数值。背景减除器也已隐式处理了 DC 偏置。完全无效。

## 实验 C：滑动窗背景减除

修改：`dsp.py:FmcwBackgroundSubtractor` 增加 `window` 参数，>0 时改为最近 N chirp 的**均值**而非 EMA。`benchmark.py` 中实例化时传入 window。

### 窗口大小扫描

| Window | 184117 hits/FP/score | 190655 hits/FP/score | 214655 hits/FP/score |
|--------|------|------|------|
| EMA α=0.02 (baseline) | 3/5, 6, -0.30 | 11/45, 4, 8.80 | 0/6, 0, 0.00 |
| **window=20 (1s)** | 2/5, 6, -1.30 | 9/45, 4, 6.80 | **1/6**, 0, **1.00** |
| window=40 (2s) | 3/5, 5, 0.25 | 9/45, 4, 6.80 | 0/6, 0, 0.00 |
| window=100 (5s) | 3/5, 6, -0.30 | 10/45, 4, 7.80 | 0/6, 1, -0.55 |

**关键发现：**
1. **window=20 是唯一在 214655 上得到 hit 的变体**（1/6）。短窗口能更好保留瞬态眨眼事件。
2. 但 window=20 损害了 184117（3→2）和 190655（11→9）的召回。
3. window=40 减少了 184117 的 FP（6→5）但同样降低 190655 召回。
4. 长窗口（100）反而引入 214655 误报。

**机理分析：** 背景减除只影响 range-bin 的**幅度和相位**（通过 `complex_bins`），**不影响** phase_pair_delta（走 `complex_baseband` 独立路径）。检测主信号是 phase-pair coherence，所以背景减除的改变只通过 amplitude gate 和 phase_delta/motion_energy 间接影响结果。window=20 能在 214655 上多检出一个 blink，说明该 blink 信号主要在 range-bin 幅度路径上而非 phase-pair 路径上。

**结论：放弃全局参数。** 没有一个窗口大小能同时改善三个 session。短窗口（20）能救 214655 但伤 190655。EMA（α=0.02）是合理的默认。如果要救 214655，应该考虑自适应窗口或直接在 phase-pair 路径上改进。

## 实验 D：phase-pair baseband 加 Hann 窗

修改：`dsp.py:extract_fmcw_chunk_feature` 在 `_phase_pair_features(complex_baseband)` 前对 baseband 乘 `np.hanning(size)`。

| Session | events | hits | FP | score | vs baseline |
|---------|--------|------|-----|-------|------------|
| 184117  | 9      | 2/5  | 7   | -1.85 | 更差 (-1.55) |
| 190655  | 12     | 11/45 | 4   | 8.80  | 无变化      |
| 214655  | 1      | **1/6** | 0   | **1.00** | **更好！首次召回** |

特征级分析（diagnostic_dump）：

| 指标 | Baseline blink | Exp-D blink | 变化 |
|------|------|------|------|
| peak_abs_ppd | 0.67 | 0.85 | **↑** 信号增强 |
| ppd_jitter_slice | 0.11 | 0.23 | ↑ 抖动增加 |
| sm9 (smoothness) | 0.79 | **0.66** | **↓** 平滑度下降 |
| dev9 (deviation) | 0.30 | 0.37 | ↑ 偏差增加 |
| global_ppd_std | 0.46 | 0.54 | ↑ 噪声底微升 |

**机理分析：** Hann 窗压低边缘采样点的权重，**提高了相位偏差（deviation↑）但降低了平滑度（sm9↓）**。因为 coherence = deviation × smoothness，两者乘积大致不变（0.30×0.79≈0.24 ≈ 0.37×0.66≈0.24），所以整体判别力没有提升，只是**移动了操作点**。214655 上某个 borderline blink 因此越过了阈值（deviation 占上风），184117 上某个 borderline blink 因此跌破了阈值（smoothness 占下风）。

**结论：放弃。** 不是真正的判别力提升，只是把同一个 borderline blink 在不同方向推过阈值。

---

## 总结

### 四个实验对比

| 实验 | 184117 | 190655 | 214655 | 净效果 |
|------|--------|--------|--------|--------|
| Baseline | 3/5, -0.30 | 11/45, 8.80 | 0/6, 0.00 | — |
| A: range FFT Hamming | 2/5, -1.85 | 12/45, 10.35 | 0/6, 0.00 | 混合，190655↑但184↓ |
| B: DC bin 抑制 | 3/5, -0.30 | 11/45, 8.80 | 0/6, 0.00 | **完全无效** |
| C: 滑动窗背景(20) | 2/5, -1.30 | 9/45, 6.80 | **1/6, 1.00** | 214↑但其他↓ |
| D: phase-pair Hann 窗 | 2/5, -1.85 | 11/45, 8.80 | **1/6, 1.00** | 214↑但184↓ |

### 关键发现

1. **214655 不是完全没救**：实验 C（window=20）和 D（Hann 窗）各自独立地在 214655 上召回 1/6 blink。说明 214655 的眨眼信号确实存在，只是处于检测阈值边缘。

2. **所有预处理改进都在做同一件事**：移动操作点。没有一个能同时提升三个 session。当前 EMA + 无窗 baseline 是 184117/190655 的最优点。

3. **检测主信号路径（intra-chirp phase-pair coherence）已经接近极限**：preprocessing 层面的改进只能微调，不能从根本上改变 phase_pair_delta 的 SNR。

4. **要救 214655 需要从检测算法层入手**，而不是 DSP 层。可能的路径：
   - 降低 214655 的检测阈值（自适应 SNR-aware threshold）
   - 检测长时间高噪声后启用更宽松的检测模式
   - 多 range-bin 融合（bin 13-17 投票）

所有实验代码已**还原为 baseline**，无残留修改。




