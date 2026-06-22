# FMCW 眨眼检测修复与 Coherence Score 优化记录

日期：2026-06-18  
项目：`E:\android_projects\eye_blink_detect\hp_acoustic_wave`

---

## 1. 背景

上一个 commit (`7d64a2f` "初版FMCW实现") 的代码是可用的。之后 working copy 中加了大量复杂逻辑（phase-pair library、spatial spread、background subtractor、复杂 segment gate 等），导致检测器退化成周期性假触发。

CLAUDE.md 中指定使用三个 session 来测试：

| Session | 时长 | 视觉眨眼数 | 说明 |
| --- | ---: | ---: | --- |
| `hp_blink_20260617_184117` | ~42s | 5 | 短 session |
| `hp_blink_20260617_190655` | ~70s | 45 | 主要测试 session |
| `hp_blink_20260617_214655` | ~940s | 6（仅最后几秒） | 长 session，基本为空 |

---

## 2. 问题诊断

### 2.1 周期性假触发

session 214655 的 events.csv 显示：

- **363 个 blink_candidate** 在 ~940 秒内
- 所有 threshold 锁定在 `0.006`（= `min_score`，最低限）
- 大量 score=0.000 的事件仍然触发
- 事件间隔稳定在 ~2.5 秒（等于 refractory 决定的最小间隔）

这就是"周期性猜测 blink"——检测器退化成了定时器。

### 2.2 根因

1. **`_TwinklePeakEventGate` 过度复杂**：从 538 行膨胀到 1300 行。多层 gate（segment shape voting、rhythm suppression、phase-pair library、spatial spread gate）之间的交互导致边界情况下事件泄漏。

2. **score=0 事件通过 gate**：`segment_close_candidate` 路径在 score=0 时仍能触发事件。

3. **baseline 无法建立**：大量 score=0 被写入 history，导致 baseline 和 threshold 都维持在 `min_score=0.006`，任何微弱信号都能超过阈值。

### 2.3 修改前 benchmark

| Session | Events | Blink Hits | Unexplained | Balanced Score |
| --- | ---: | ---: | ---: | ---: |
| 184117 | 7 | 2/5 | 5 | -0.75 |
| 190655 | 20 | 22/45 | 6 | 18.70 |
| 214655 | **363** | N/A | N/A | N/A |

---

## 3. 第一步：简化回 commit 7d64a2f 架构

### 3.1 方案

保留 `dsp.py` 中经过验证的 FMCW DSP（chirp 生成、range-bin 提取、intra-chirp phase-pair 特征），**只重写检测器**：

- `_TwinklePeakEventGate` 回到简单版本：只保留 local peak + rising edge 检测 + large motion suppression
- 移除：segment shape voting、rhythm suppression、phase-pair library、spatial spread gate、extreme motion suppress
- 保留 FMCW 基本支持：amplitude check、amplitude stability、min score、FMCW-specific refractory
- 同步清理 `config.py`、`run_hp_wave_detector.py`、`benchmark_hp_blink.py` 中不再使用的参数

### 3.2 代码量变化

| 文件 | 修改前行数 | 简化后行数 |
| --- | ---: | ---: |
| `blink_detector.py` | 1300 | 480 → 721（加 coherence 后） |
| `dsp.py` | 427 | 427（未改） |
| `config.py` | 117 | 108 |

### 3.3 简化后 benchmark

| Session | Events | Blink Hits | Unexplained | Balanced Score |
| --- | ---: | ---: | ---: | ---: |
| 184117 | 2 | 0/5 | 2 | -1.10 |
| 190655 | 6 | 5/45 | 2 | 3.90 |
| 214655 | **0** | 0/6 | 0 | 0.00 |

**周期性假触发完全消除**（214655: 363 → 0），但召回率极低（190655: 5/45）。

---

## 4. 第二步：诊断召回率低的原因

### 4.1 FMCW range-bin phase 的噪声特性

对 session 190655 的 1444 个 feature frame 做统计：

```
Total frames: 1444
Trajectory scores > 0: 1441 / 1444  (99.8%)
Sign changes >= 1: 1430 / 1444  (99.0%)
Sign changes >= 2: 1333  (92.3%)
```

**结论**：FMCW range-bin phase 在 HP 笔记本上极度嘈杂。几乎每一帧都有 trajectory reversal，所以 `_trajectory_score()` 几乎对所有帧都返回非零值。`_trajectory_score` 在 FMCW 模式下完全失去区分能力。

### 4.2 FMCW gate 全部拦截

```
fmcw_amplitude_ok: 1292/1444  (amplitude 通常足够)
fmcw_amplitude_stable: 大部分通过
fmcw_score_ok: 0/1444  (!!!)
```

`_fmcw_twinkle_score` 在检查 `score_ok` 时，虽然原始 trajectory score 很大（1-11），但因为 `_fmcw_twinkle_score` 用的是 `max(score, phase_score)`，而 FMCW 的 trajectory score 本身就无意义——同时 `fmcw_min_score=0.07` 的阈值对这些无意义的高分也没有过滤效果。

### 4.3 Phase-pair delta 的信号特性

分析 `phase_pair_delta`（同 chirp 内 phase-pair 差异）：

| 区段 | phase_pair_delta 特征 |
| --- | --- |
| 眨眼附近（如 vb=9.313） | 连续值高度一致：`0.596, 0.640, 0.595, 0.551, 0.641, 0.660, 0.570, 0.516` |
| 安静期（30-33s） | 随机跳动：`0.014, 0.353, -0.301, 0.194, -0.437, -0.354, -0.003, -0.349` |
| 整体分布 | abs median=0.287, abs p90=0.648 |

**关键发现**：区分眨眼和噪声的不是 phase-pair delta 的大小，而是 **连续值的平滑度/一致性**。

---

## 5. 第三步：Coherence Score 设计

### 5.1 核心思想

论文 TwinkleTwinkle 的核心观察是：眨眼引起的相位变化是 **平滑的方向性偏移**（眼睑下降-上升导致路径长度有规律变化），而噪声是 **随机跳动**。

我们将这个观察量化为 **coherence score**：

```python
coherence_score = deviation × smoothness
```

其中：

- **deviation** = 当前窗口的 phase-pair delta 中位数与长期基线中位数的偏差（信号强度）
- **smoothness** = 1 - jitter/jitter_scale（连续值的一致性）
  - jitter = 连续 phase-pair delta 之间差异的 RMS
  - jitter_scale = 长期 phase-pair delta 的标准差

### 5.2 为什么有效

| 场景 | deviation | smoothness | coherence_score |
| --- | --- | --- | --- |
| 眨眼 | 高（相位偏离基线） | 高（连续值平滑） | **高** |
| 安静 | 低（围绕基线波动） | 低（随机跳动） | **低** |
| 大动作 | 高 | 低（剧烈跳动） | **中等** |

coherence score 自然地同时要求"有信号"（deviation）和"信号可信"（smoothness），无需额外的 segment voting 或 rhythm 抑制。

### 5.3 Gate 层对 FMCW 的放松

FMCW 模式下的 phase 和 motion_energy 的量程与 tone 模式完全不同，因此在 `_TwinklePeakEventGate` 中放松了三个约束：

- `twinkle_max_peak_score`：FMCW coherence score 的量程不同，不用 ceiling 限制
- `twinkle_max_motion_energy`：FMCW motion_energy 天然偏高（range-bin 幅度波动大）
- `twinkle_max_sign_changes`：FMCW 相位嘈杂，sign changes 几乎总是 > 2

### 5.4 实现位置

```
blink_detector.py:
  TwinkleTwinkleBlinkDetector.__init__()
    - 新增 self.phase_pair_raw_history (deque)

  TwinkleTwinkleBlinkDetector._append_unwrapped_phase()
    - 新增：追加 phase_pair_delta 到 phase_pair_raw_history

  TwinkleTwinkleBlinkDetector._fmcw_coherence_score()   [新方法]
    - 计算 deviation, smoothness, coherence_score

  TwinkleTwinkleBlinkDetector._fmcw_twinkle_score()
    - 改为用 coherence_score 替代原始 trajectory score

  _TwinklePeakEventGate.update()
    - 新增 is_fmcw 参数
    - FMCW 模式下放松 score ceiling / motion energy / sign changes
```

---

## 6. 实验结果

### 6.1 最终 benchmark（默认参数 `fmcw_min_score=0.09`）

| Session | Events | Blink Hits | Unexplained | Balanced Score |
| --- | ---: | ---: | ---: | ---: |
| 184117 | 16 | **4/5** | 12 | -2.60 |
| 190655 | 36 | **34/45** | 8 | **29.60** |
| 214655 | **0** | 0/6 | 0 | 0.00 |

### 6.2 三版对比

| Session | 修改前 (broken) | 第一轮简化 | + Coherence Score |
| --- | --- | --- | --- |
| 184117 hits | 2/5 | 0/5 | **4/5** |
| 184117 unexplained | 5 | 2 | 12 |
| 184117 score | -0.75 | -1.10 | -2.60 |
| 190655 hits | 22/45 | 5/45 | **34/45** |
| 190655 unexplained | 6 | 2 | 8 |
| 190655 score | 18.70 | 3.90 | **29.60** |
| 214655 events | **363** | 0 | **0** |

### 6.3 参数调优实验

以 `--source audio --truth auto` 回测，扫描 `fmcw_min_score` 和 `fmcw_refractory`：

| fmcw_min_score | fmcw_refractory | 184117 hits | 184117 unexp | 184117 score | 190655 hits | 190655 unexp | 190655 score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.07 | 1.3 | 4/5 | 15 | -4.25 | 34/45 | 9 | 29.05 |
| **0.09** | **1.3** | **4/5** | **12** | **-2.60** | **34/45** | **8** | **29.60** |
| 0.12 | 1.3 | 4/5 | 11 | -2.05 | 31/45 | 5 | 28.25 |
| 0.15 | 1.5 | 4/5 | 9 | -0.95 | 29/45 | 7 | 25.15 |

选择 `fmcw_min_score=0.09` 作为默认值：

- 在 190655 上保持最高召回 34/45
- 比 0.07 减少了 7 个误报
- 比 0.12/0.15 召回更好

### 6.4 audio 和 features 回放一致性

两种回放源产生完全相同的结果，说明 FMCW replay 链路正确：

| Session | Source | Events | Hits | Unexplained | Score |
| --- | --- | ---: | ---: | ---: | ---: |
| 184117 | audio | 16 | 4/5 | 12 | -2.60 |
| 184117 | features | 16 | 4/5 | 12 | -2.60 |
| 190655 | audio | 36 | 34/45 | 8 | 29.60 |
| 190655 | features | 36 | 34/45 | 8 | 29.60 |

---

## 7. 当前限制与下一步

### 7.1 当前限制

1. **184117 误报偏多**（12 unexplained / 16 events）。这是一个仅 42 秒、只有 5 次眨眼的短 session，coherence score 在启动阶段（baseline 不够稳定时）容易产生假阳性。

2. **190655 仍有 11 次漏检**（34/45）。漏检主要集中在非常轻微或快速的眨眼，phase-pair delta 偏移不够明显。

3. **214655 零检测**。这个 940 秒的 session 前 ~920 秒没有眨眼（视觉标注只在最后几秒），算法在没有眨眼时正确保持安静，但也没能捕捉到最后几秒的 6 次眨眼。原因是 ~920 秒的安静期建立了非常稳定的 baseline，最后几秒的眨眼难以超过阈值。

### 7.2 下一步优化方向

1. **startup baseline 保护**：在前 N 秒内使用更保守的 baseline 更新策略，减少启动假阳性。

2. **多 range-bin 投票**：不只看 `range_bin=15`，而是对多个 bin 计算 coherence score，取投票一致的作为候选。这可以提供更独立的候选信息。

3. **自适应 baseline 衰减**：对超长安静 session（如 214655），baseline 不应该无限收紧。可以加一个 baseline 松弛机制，在长时间无事件后降低 threshold。

4. **amplitude-deviation 联合特征**：论文 BlinkListener 指出眨眼同时带来 amplitude 和 phase 变化。可以将 amplitude_delta 也纳入 coherence score 的计算。

5. **视觉标注对齐精度**：当前使用 ±0.8s 窗口匹配视觉和声学事件。如果声学延迟系统性偏高或偏低，更精确的对齐可能改善指标。

---

## 8. 代码架构总结

### 8.1 文件结构

```
hp_acoustic_wave/
  dsp.py                  (427 行, 未修改)  信号处理：chirp 生成、I/Q 解调、FMCW range-bin 提取、phase-pair 特征
  blink_detector.py       (721 行)          检测器：BlinkListener / TwinkleTwinkle / Composite + gate 逻辑
  config.py               (108 行)          配置 dataclass
  benchmark.py            (491 行)          回放引擎：feature/audio replay、事件评分
  benchmark_hp_blink.py   (156 行)          命令行 benchmark 入口
  run_hp_wave_detector.py (289 行)          实时运行入口
  app.py                                    实时应用（音频回调、摄像头、UI）
```

### 8.2 FMCW 检测流程

```
扬声器播放 17-23 kHz FMCW chirp (chirp_duration=0.05s)
        |
麦克风实时采集
        |
按 2400 samples (= 48000 * 0.05s) 分 chirp
        |
dsp.extract_fmcw_chunk_feature():
  rx * tx → 低通 → FFT → range_bin complex value → I/Q/amplitude/phase
  rx * complex_reference → 低通 → phase_pair_features() → phase_pair_delta
        |
TwinkleTwinkleBlinkDetector.update():
  _append_unwrapped_phase()  → 追踪 phase trajectory + phase_pair_raw_history
  _append_amplitude()        → 追踪 amplitude 稳定性
  _candidate_trajectory_score()  → (tone 模式有用, FMCW 模式几乎无区分力)
  _fmcw_coherence_score()    → deviation × smoothness [核心 FMCW 特征]
  _fmcw_twinkle_score()      → amplitude check + coherence → adjusted_score
        |
_TwinklePeakEventGate.update():
  adaptive threshold (median + k * MAD)
  local peak / rising edge detection
  refractory period (FMCW: 1.3s)
  large motion suppression
        |
BlinkDetectionResult (is_event, score, threshold, metrics)
```

### 8.3 启动命令

```powershell
& 'C:\Program Files\Python38\python.exe' .\run_hp_wave_detector.py `
  --mode blink --blink-method twinkle --signal-mode fmcw `
  --fmcw-range-bin 15 --amplitude 0.2 --input-device 1 --output-device 3 `
  --camera-width 640 --camera-height 480
```

### 8.4 回测命令

```powershell
& 'C:\Program Files\Python38\python.exe' .\benchmark_hp_blink.py `
  --session sessions\hp_blink_20260617_190655 `
  --source audio --truth auto
```

### 8.5 测试

27 个测试全部通过：

```powershell
cd E:\android_projects\eye_blink_detect
& 'C:\Program Files\Python38\python.exe' -m pytest hp_acoustic_wave/tests/test_blink_algorithms.py hp_acoustic_wave/tests/test_fmcw_wave_integration.py -v
```
