# Twinkle FMCW Blink 过拟合实验记录

日期：2026-06-17  
项目：`E:\android_projects\eye_blink_detect\hp_acoustic_wave`  
目标：以视觉标注为真值，尽量让声学眨眼检测接近视觉眨眼事件。

## 1. 实验数据

本轮只使用两组确认正确的数据：

| Session | 真值来源 | 视觉眨眼数 |
| --- | --- | ---: |
| `sessions\hp_blink_20260617_133830` | `events.csv` / `visual_blink` | 71 |
| `sessions\hp_blink_20260617_150955` | `events.csv` / `visual_blink` | 23 |

回测方式：

```powershell
& 'C:\Program Files\Python38\python.exe' .\benchmark_hp_blink.py `
  --session <session> `
  --source audio `
  --truth auto
```

说明：

- `--source audio` 会从 `audio.wav` 重新走 FMCW 特征提取和声学检测链路。
- `--truth auto` 会优先使用视觉 `visual_blink` 作为真值。
- 指标里的 `blink_hits` 表示视觉眨眼前后匹配窗口内有声学事件。
- `unexplained_events` 表示没有匹配到视觉眨眼的声学事件，可视为误判/误报。

## 2. 当前最好结果

当前最好参数组合：

| 参数 | 值 | 作用 |
| --- | ---: | --- |
| `twinkle_fmcw_use_intra_chirp_phase_pair` | `true` | 使用论文式同一 chirp 内 phase-pair 作为轨迹源 |
| `twinkle_candidate_windows` | `()` | 关闭多候选窗口投票，保留单条轨迹 |
| `twinkle_fmcw_refractory_s` | `1.4` | FMCW 眨眼候选最小间隔 |
| `twinkle_fmcw_min_score` | `0.05` | FMCW Twinkle 最小候选分数 |
| `twinkle_fmcw_max_amplitude_delta_ratio` | `0.45` | range-bin 幅值稳定性门限 |
| `twinkle_fmcw_phase_pair_weight` | `0.5` | phase-pair 分数权重 |
| `twinkle_segment_min_points` | `1` | 分段候选最低点数 |
| `twinkle_segment_max_sign_changes` | `8` | 放宽轨迹方向变化限制 |
| `twinkle_rhythm_max_count` | `8` | 基本关闭强节律抑制 |

最好回测结果：

| Session | Acoustic Events | Blink Hits | Recall | Unexplained Events | 误判占事件比例 | Balanced Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `133830` | 61 | 65 / 71 | 91.5% | 12 | 19.7% | 58.40 |
| `150955` | 33 | 21 / 23 | 91.3% | 15 | 45.5% | 12.75 |
| 合计 | 94 | 86 / 94 | 91.5% | 27 | 28.7% | 71.15 |

保守版本：

| 版本 | 主要变化 | 合计分数 | 召回 | 误判 |
| --- | --- | ---: | ---: | ---: |
| 最好召回版 | `refractory=1.4` | 71.15 | 86 / 94 | 27 |
| 稍保守版 | `refractory=1.6` | 70.35 | 83 / 94 | 23 |

保守版把 `150955` 的事件数从 33 降到 29，误判从 15 降到 13，但总召回从 86 降到 83。

## 3. 对比结果和结论

### 3.1 多候选轨迹投票

论文 TwinkleTwinkle 不是只看单个点，而是构造多个候选 phase interval / trajectory，再通过相似性、稳定性、edge/segment 约束筛选候选轨迹。

本轮把“多候选”实现为：在同一条 unwrapped phase 流上，用多个尾部窗口长度观察轨迹，例如 `(5, 7, 9, 11)`；多个窗口都出现有效反转轨迹时，才认为候选更可信。

实际结果：在这两组数据上，多候选投票变差。

| 配置 | 合计分数 | `133830` | `150955` |
| --- | ---: | --- | --- |
| 单窗口 `cand=(), votes=1` | 71.15 | `65/71`, `unexp=12` | `21/23`, `unexp=15` |
| 多候选 `cand=(5,7,9,11), votes=2` | 66.50 | `63/71`, `unexp=13` | `20/23`, `unexp=17` |

分析：

- 当前只有一条声学轨迹，多窗口之间高度相关，不像论文中多个 phase intervals 能提供更独立的候选信息。
- 多窗口投票会把同一段噪声在多个窗口里重复确认为“有效轨迹”，导致 `150955` 的误报反而增加。
- 多窗口还会改变局部峰值和分段时机，使部分真实眨眼错过最佳触发点，召回下降。

结论：当前最优不启用多候选窗口投票。后续如果要重新启用，应该先做真正的候选多样性，例如多个 chirp 内 phase pair、多个 range bin 或多个 phase interval，而不是只在同一轨迹上改窗口长度。

### 3.2 同一 chirp 内 phase-pair

论文的核心差异之一是：phase pair 来自同一 chirp 内两个采样点，而不是跨 chunk 的 phase delta。

本轮实现了同 chirp phase-pair：

- 在 `dsp.py` 中对 FMCW 接收信号做 complex baseband。
- 在同一 chirp 内选多组采样点，例如 `15%-85%`、`20%-80%`、`25%-75%` 等。
- 每组计算 `phase(second) - phase(first)`。
- 再用加权中位数、vote ratio 和 consistency 描述候选 pair 的一致性。

结果：同 chirp phase-pair 在 `133830` 上明显提高召回，是当前最好版本的核心来源。

关键取舍：

- 开启同 chirp phase-pair 后，`133830` 能稳定到 `65/71`。
- `150955` 仍然误判较多，说明这组数据中同 chirp phase-pair 对非眨眼扰动也敏感。
- 当前最好参数把 phase-pair quality gate 放宽到 `0.0`，因为严格质量门控会降低召回。

### 3.3 分段和节律约束

论文后续还有分段、边方向判断和连续眨眼数量识别。本轮实现了工程版门控：

- local peak / rising edge gate：只在局部峰值或阈值上升沿触发。
- segment gate：避免同一个连续高分段重复触发。
- rhythm gate：限制短时间过密事件。

实际搜索发现：

- 强节律抑制对当前两组数据帮助不明显。
- `133830` 的视觉眨眼本身比较密集，强节律抑制会误伤真实眨眼。
- 最好版本等价于放宽节律限制：`twinkle_rhythm_max_count=8`。

## 4. 核心代码位置

### 4.1 FMCW 同 chirp phase-pair 特征

文件：`dsp.py`

核心位置：

- `ChunkFeature.phase_pair_delta`
- `ChunkFeature.phase_pair_vote_ratio`
- `ChunkFeature.phase_pair_consistency`
- `ChunkFeature.phase_pair_candidate_count`
- `_phase_pair_features(baseband)`
- `extract_fmcw_chunk_feature(...)`

核心逻辑摘要：

```python
pair_fractions = (
    (0.15, 0.85),
    (0.20, 0.80),
    (0.25, 0.75),
    (0.30, 0.70),
    (0.35, 0.65),
    (0.40, 0.60),
)

delta = unwrap_delta(
    atan2(second.imag, second.real),
    atan2(first.imag, first.real),
)
```

然后用 weighted median 得到代表 phase-pair delta，并计算：

- `vote_ratio`：多少 pair 和中心值接近。
- `consistency`：pair 分布是否集中。
- `candidate_count`：有效 pair 数量。

### 4.2 Twinkle 检测器

文件：`blink_detector.py`

核心类：

- `BlinkDetectionConfig`
- `_TwinklePeakEventGate`
- `TwinkleTwinkleBlinkDetector`

核心路径：

```text
ChunkFeature
  -> TwinkleTwinkleBlinkDetector._trajectory_phase()
  -> _append_unwrapped_phase()
  -> _candidate_trajectory_score()
  -> _fmcw_twinkle_score()
  -> _TwinklePeakEventGate.update()
  -> BlinkDetectionResult
```

关键实现点：

- `_trajectory_phase()` 决定使用 range-bin phase 还是同 chirp phase-pair。
- `_append_unwrapped_phase()` 对 phase-pair 轨迹做基线中心化，减少 session 间绝对相位偏移。
- `_candidate_trajectory_score()` 支持多窗口候选，但当前最好版本关闭它。
- `_fmcw_twinkle_score()` 加入 amplitude stability、phase-pair score、FMCW min score。
- `_TwinklePeakEventGate.update()` 做 peak、segment、refractory、rhythm gate。

### 4.3 参数入口

文件：

- `run_hp_wave_detector.py`
- `benchmark_hp_blink.py`
- `config.py`

关键参数：

```text
--blink-twinkle-fmcw-use-intra-chirp-phase-pair
--blink-candidate-windows
--blink-min-candidate-votes
--blink-twinkle-fmcw-refractory
--blink-twinkle-fmcw-min-score
--blink-twinkle-fmcw-max-amplitude-delta-ratio
--blink-twinkle-fmcw-phase-pair-weight
--blink-twinkle-segment-min-points
--blink-twinkle-segment-max-sign-changes
--blink-twinkle-rhythm-max-count
```

## 5. 当前最好启动方式

最好召回版：

```powershell
& 'C:\Program Files\Python38\python.exe' .\run_hp_wave_detector.py `
  --mode blink --blink-method twinkle --signal-mode fmcw `
  --fmcw-range-bin 15 --amplitude 0.2 --input-device 1 --output-device 3 `
  --camera-width 640 --camera-height 480 `
  --blink-candidate-windows 0 --blink-min-candidate-votes 1 `
  --blink-twinkle-fmcw-use-intra-chirp-phase-pair `
  --blink-twinkle-phase-pair-vote-ratio-min 0.0 `
  --blink-twinkle-phase-pair-consistency-min 0.0 `
  --blink-twinkle-candidate-blend-weight 1.0 `
  --blink-twinkle-segment-min-points 1 `
  --blink-twinkle-segment-max-sign-changes 8 `
  --blink-twinkle-rhythm-max-count 8 `
  --blink-twinkle-fmcw-refractory 1.4 `
  --blink-twinkle-fmcw-min-score 0.05 `
  --blink-twinkle-fmcw-max-amplitude-delta-ratio 0.45 `
  --blink-twinkle-fmcw-phase-pair-weight 0.5 --blink-startup-ignore 0
```

稍保守版：

```powershell
& 'C:\Program Files\Python38\python.exe' .\run_hp_wave_detector.py `
  --mode blink --blink-method twinkle --signal-mode fmcw `
  --fmcw-range-bin 15 --amplitude 0.2 --input-device 1 --output-device 3 `
  --camera-width 640 --camera-height 480 `
  --blink-candidate-windows 0 --blink-min-candidate-votes 1 `
  --blink-twinkle-fmcw-use-intra-chirp-phase-pair `
  --blink-twinkle-phase-pair-vote-ratio-min 0.0 `
  --blink-twinkle-phase-pair-consistency-min 0.0 `
  --blink-twinkle-candidate-blend-weight 1.0 `
  --blink-twinkle-segment-min-points 1 `
  --blink-twinkle-segment-max-sign-changes 8 `
  --blink-twinkle-rhythm-max-count 8 `
  --blink-twinkle-fmcw-refractory 1.6 `
  --blink-twinkle-fmcw-min-score 0.05 `
  --blink-twinkle-fmcw-max-amplitude-delta-ratio 0.45 `
  --blink-twinkle-fmcw-phase-pair-weight 0.5
```

说明：

- `--blink-candidate-windows 0` 用来让解析后的候选窗口为空，因为代码会过滤掉非正窗口。
- 最好召回版更接近视觉真值，但 `150955` 误判偏多。
- 稍保守版更适合实时演示，减少“总是在 blink”的观感。

## 6. 后续优化方向

当前瓶颈不是“阈值还没调好”，而是候选轨迹缺少真正独立的可投票信息。

优先方向：

1. 多 range-bin 候选：不只看 `range_bin=15`，而是对眼部附近多个 bin 生成 phase trajectory，再按视觉真值选择最稳定 bin 或做投票。
2. 多 phase-pair interval：把同 chirp 内多组 phase-pair 不只聚成一个 delta，而是保留多条 trajectory，分别进入候选评分。
3. 分段后形态分类：对每个 segment 提取 duration、rise/fall ratio、peak shape、motion_energy，训练一个非常小的过拟合分类器。
4. 专门处理 `150955` 的误报：分析 unmatched event 前后的 phase-pair、amplitude_delta_ratio 和 visual EAR，找出误报共性。

当前建议：

- 实时使用最好召回版或稍保守版。
- 不建议启用当前的多窗口候选投票。
- 下一轮若继续优化，应从“多个独立候选轨迹”入手，而不是继续扩大单条轨迹上的窗口搜索。
