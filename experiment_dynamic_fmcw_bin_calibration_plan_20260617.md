# 动态 FMCW Range Bin 校准与多 Bin 眨眼检测计划

日期：2026-06-17

目标数据集：

| Session | 真值来源 | 当前问题 |
| --- | --- | --- |
| `sessions\hp_blink_20260617_133830` | 视觉 `visual_blink` | 固定 `range_bin=15` 表现较好，需要保持 |
| `sessions\hp_blink_20260617_150955` | 视觉 `visual_blink` | 召回较好但误判偏多，需要平衡 |
| `sessions\hp_blink_20260617_164043` | 视觉 `visual_blink` | 固定 `range_bin=15` 召回明显下降 |

## 1. 背景结论

之前最优配置主要依赖固定 `--fmcw-range-bin 15` 和同 chirp 内 phase-pair trajectory。该配置在 `133830`、`150955` 上有效，但在 `164043` 上召回下降明显。

已观察到：

```text
164043 + range_bin=15 + startup_ignore=2  -> 25/72
164043 + range_bin=15 + startup_ignore=0  -> 67/72
164043 + range_bin=24 + startup_ignore=2  -> 约 71/72
```

这说明至少有两个问题：

1. 固定 `range_bin=15` 不能适配不同坐姿和距离。
2. `startup_ignore_s=2` 当前会污染 baseline / segment gate 状态，使后续召回异常下降。

本计划先把 `blink-startup-ignore` 默认值改为 `0.0`，后续用显式校准阶段替代启动静默。

## 2. 总体目标

构建一个面向实时使用的校准流程：

```text
启动
  -> 视觉和声学同时进入校准期
  -> FMCW range profile 找候选距离带
  -> 使用视觉 blink 真值评估候选 bin 的声学响应
  -> 选择 Top-K range bins 和权重
  -> 校准完成，进入正式声学检测
  -> 视觉继续运行，只用于记录、显示和评测对比
```

校准完成后，声学检测不得读取视觉结果来触发、抑制或修正 blink event。视觉只作为真值日志继续存在。

## 3. 启动忽略参数处理

### 3.1 当前问题

`startup_ignore_s` 原本用于启动阶段不输出事件，但当前实现里启动期高 score 仍可能进入 history / baseline。这样会抬高后续阈值，尤其对 Twinkle peak / segment gate 影响很大。

### 3.2 立即修改

统一默认值：

```text
run_hp_wave_detector.py: --blink-startup-ignore default=0.0
benchmark_hp_blink.py: --blink-startup-ignore default=0.0
config.py: BlinkConfig.startup_ignore_s = 0.0
```

### 3.3 后续修正方向

保留参数，但修改语义：

```text
startup 期：
  - 不输出正式 acoustic blink event
  - 不更新 event / segment 触发状态
  - baseline 只接收低风险安静样本，或跳过高 score 样本
```

更推荐由新参数负责启动校准：

```text
--blink-calibration-seconds 10
```

## 4. FMCW 测距 Prior

FMCW 测距不直接决定最终 bin，只用于缩小候选范围。

原因：

```text
最大反射峰不一定是眼睛；
它可能来自脸部整体、鼻梁、胸口、桌面、设备串扰或多径反射。
```

计划：

1. 在校准期统计平均 range profile。
2. 排除过近串扰 bin。
3. 找稳定反射峰或多个反射带。
4. 得到候选 bin band，例如 `peak_bin ± 4`。
5. 如果 prior 不稳定，回退到固定搜索范围，例如 `8:32`。

## 5. 混合视觉校准

默认校准参数：

```text
--blink-calibration-seconds 10
--blink-calibration-max-seconds 25
--blink-calibration-min-visual-blinks 3
--blink-calibration-range-bins 8:32
--blink-calibration-top-k 3
```

流程：

1. 前 10 秒自然采集视觉和声学。
2. 如果视觉 blink 数量不少于 3 次，直接完成校准。
3. 如果视觉 blink 数量不足，提示用户补眨几次。
4. 达到最少 blink 数或最大校准时间后，进入评分和选 bin。

每个 bin 的校准评分：

```text
bin_score =
  visual_hit_score
  - false_positive_penalty
  + phase_pair_quality_bonus
  + fmcw_prior_bonus
  - noisy_baseline_penalty
```

评分含义：

| 指标 | 作用 |
| --- | --- |
| `visual_hit_score` | 视觉 blink 前后窗口内是否有声学峰 |
| `false_positive_penalty` | 非视觉 blink 区间的高分峰数量 |
| `phase_pair_quality_bonus` | phase-pair vote ratio / consistency |
| `fmcw_prior_bonus` | 是否落在 FMCW 稳定反射候选带 |
| `noisy_baseline_penalty` | 静默期 score 波动过大的惩罚 |

校准输出示例：

```json
{
  "selected_bins": [24, 23, 26],
  "bin_weights": [1.0, 0.82, 0.76],
  "primary_bin": 24,
  "calibration_visual_blinks": 5,
  "calibration_seconds": 12.4
}
```

## 6. 多 Bin 声学检测

校准后不再只跑一个 range bin，而是对 Top-K bins 运行多条独立声学轨迹。

推荐 V1：

```text
Top-K = 3
min_bin_votes = 1
event_merge_window = 0.25s
```

每个 bin 独立维护：

```text
phase trajectory
phase-pair trajectory
amplitude history
Twinkle detector state
event gate state
```

融合分两层：

1. 分数融合：

```text
combined_score = weighted max 或 weighted median
```

2. 事件融合：

```text
如果 Top-K 内至少 min_bin_votes 个 bin 在短窗口内触发，
则合并为一个 acoustic blink event。
```

第一版建议 `min_bin_votes=1`，优先恢复召回；如果误报变多，再尝试 `min_bin_votes=2`。

## 7. 视觉继续运行但不辅助声学

校准完成后，视觉仍然继续打开：

```text
visual_blink 写入 events.csv
visual_labels.csv 持续记录 EAR / blink 状态
UI 显示 visual 与 acoustic 的时间差异
```

但正式声学检测阶段只能读取：

```text
audio feature
selected_bins
bin_weights
detector internal state
```

禁止使用：

```text
visual_blink event
visual EAR
visual closed/open state
```

来触发、压制、修正或延迟 acoustic blink。

## 8. 离线评测计划

先扩展离线评测，再接实时 UI。

新增或扩展命令：

```powershell
& 'C:\Program Files\Python38\python.exe' .\benchmark_hp_blink.py `
  --session sessions\hp_blink_20260617_164043 `
  --source audio `
  --truth visual `
  --blink-calibration-seconds 10 `
  --blink-calibration-max-seconds 25 `
  --blink-calibration-min-visual-blinks 3 `
  --blink-calibration-range-bins 8:32 `
  --blink-calibration-top-k 3 `
  --blink-calibration-use-fmcw-prior `
  --blink-use-calibrated-bins
```

批量回测输出每组中间结果：

```text
session
calibration selected bins
per-bin calibration score
events
blink_hits
recall
unexplained_events
balanced_score
```

要求每个组合跑完立即打印，不等整轮搜索结束。

## 9. 验收目标

单组目标：

| Session | 目标 |
| --- | --- |
| `133830` | 保持或超过 `65/71` |
| `150955` | 保持或超过 `21/23`，同时观察误判 |
| `164043` | 从 `25/72` 提升到接近 `67/72` 或更高 |

三组合计目标：

```text
recall >= 90%
unexplained_events 不明显高于当前最优单 bin 总量
```

如果 recall 和误判存在明显冲突，输出两个 preset：

```text
best_recall
balanced
```

## 10. 实施顺序

1. 将 `blink-startup-ignore` 默认值统一改为 `0.0`。
2. 修正 startup ignore 的 baseline 污染语义。
3. 写离线 multi-bin feature replay，不先改实时 UI。
4. 在三组数据上扫 `range_bin=8..32`，保存 per-bin 结果。
5. 实现 calibration scorer，先选单个最佳 bin。
6. 扩展为 Top-K bins 和权重。
7. 实现多 bin detector state 与事件融合。
8. 接入实时校准参数、metadata 记录和 UI 状态。
9. 三组回测，输出新实验文档和推荐启动命令。

## 11. 第一版推荐启动方向

未来实时命令形态：

```powershell
& 'C:\Program Files\Python38\python.exe' .\run_hp_wave_detector.py `
  --mode blink --blink-method twinkle --signal-mode fmcw `
  --amplitude 0.2 --input-device 1 --output-device 3 `
  --camera-width 640 --camera-height 480 `
  --blink-startup-ignore 0 `
  --blink-calibration-seconds 10 `
  --blink-calibration-max-seconds 25 `
  --blink-calibration-min-visual-blinks 3 `
  --blink-calibration-range-bins 8:32 `
  --blink-calibration-top-k 3 `
  --blink-calibration-use-fmcw-prior `
  --blink-multibin-min-votes 1 `
  --blink-multibin-merge-window 0.25 `
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
  --blink-twinkle-fmcw-phase-pair-weight 0.5
```

注意：在多 bin 校准正式实现前，仍可保留 `--fmcw-range-bin` 作为 fallback。

## 12. 2026-06-17 离线实现进展

本轮已先实现离线 benchmark 路径，不改实时 UI：

```text
--blink-use-calibrated-bins
--blink-calibration-range-bins
--blink-calibration-top-k
--blink-calibration-preferred-bin
--blink-calibration-use-fmcw-prior
--blink-calibration-prior-weight
--blink-calibration-prior-band-radius
--blink-multibin-min-votes
--blink-multibin-merge-window
```

### 12.1 FMCW Prior 的实际定义

查阅 `refer\16.1 Acoustic Ranging - IoT Book.html` 后，采用的核心知识是：

```text
FMCW chirp 发射后，接收信号相对发射信号存在传播时延；
接收信号与发射参考混频并低通后得到 beat frequency；
beat frequency 和传播时延、距离成比例；
对混频结果做 FFT 后，峰值 bin 可以作为反射距离 profile。
```

但实验发现，最强静态反射峰不能直接当眼部 bin。例如 `164043` 的最强校准期反射在 bin 19，但该 bin 的眨眼检测明显差。因此本轮没有使用“最强幅值峰直接选 bin”，而是使用：

```text
视觉校准响应 + FMCW 反射距离带 prior
```

具体做法：

1. 每个候选 bin 在校准期计算视觉对齐的 acoustic calibration score。
2. 每个候选 bin 同时计算校准期 mean amplitude，形成粗略 FMCW range profile。
3. 用 calibration score 的连续响应带选择一个 blink-responsive range band。
4. FMCW amplitude 只参与选择反射带，不在同一带内细分强弱。
5. 带内 prior 只按距离中心衰减，避免幅值细节把 bin 24 错拉到 bin 26。

这样 prior 的角色是：

```text
缩小候选距离带，不替代视觉校准评分。
```

### 12.2 当前最好离线命令

```powershell
& 'C:\Program Files\Python38\python.exe' .\benchmark_hp_blink.py `
  --session <session> `
  --source audio `
  --truth auto `
  --blink-use-calibrated-bins `
  --blink-calibration-use-fmcw-prior `
  --blink-calibration-prior-weight 2.0 `
  --blink-calibration-prior-band-radius 4 `
  --blink-calibration-range-bins 8:28 `
  --blink-calibration-top-k 1 `
  --blink-calibration-preferred-bin 15 `
  --blink-multibin-min-votes 1 `
  --blink-candidate-windows 0 `
  --blink-min-candidate-votes 1 `
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
  --blink-twinkle-fmcw-phase-pair-weight 0.5
```

### 12.3 三组数据当前结果

Top-1 calibrated bin，10 秒校准，启用 FMCW response-band prior：

| Session | Selected Bin | Blink Hits | Recall | Unexplained | Balanced Score |
| --- | ---: | ---: | ---: | ---: | ---: |
| `133830` | 14 | 69 / 71 | 97.2% | 13 | 61.85 |
| `150955` | 15 | 22 / 23 | 95.7% | 15 | 13.75 |
| `164043` | 24 | 71 / 72 | 98.6% | 7 | 67.15 |
| 合计 | - | 162 / 166 | 97.6% | 35 | 142.75 |

对比固定 `range_bin=15`：

```text
164043 从 67/72 提升到 71/72；
150955 从 22/23 保持到 22/23；
133830 从 65/71 提升到 69/71。
```

### 12.4 Top-K 多 Bin 结论

在 `164043` 上测试 Top-K 多 bin：

| 配置 | Blink Hits | Unexplained | Balanced Score | 结论 |
| --- | ---: | ---: | ---: | --- |
| Top-1, selected bin 24 | 71 / 72 | 7 | 67.15 | 当前最好 |
| Top-3, min_votes=1 | 71 / 72 | 17 | 61.65 | 召回不变，误判增加 |
| Top-3, min_votes=2 | 66 / 72 | 5 | 63.25 | 误判下降，但召回损失 |

结论：

```text
当前不建议正式检测阶段启用 Top-K 多 bin 融合；
更稳的是校准阶段多 bin 搜索，检测阶段使用 Top-1 calibrated bin。
```

### 12.5 测试

已运行：

```powershell
$env:PYTHONPATH='E:\android_projects\eye_blink_detect'
& 'C:\Program Files\Python38\python.exe' -m pytest tests -q
```

结果：

```text
60 passed
```

## 13. 下一轮：多 Bin 大动作抑制

当前 Top-1 calibrated bin 的合计误判：

| Session | Acoustic Events | Unexplained | 误判占声学事件 |
| --- | ---: | ---: | ---: |
| `133830` | 65 | 13 | 20.0% |
| `150955` | 33 | 15 | 45.5% |
| `164043` | 61 | 7 | 11.5% |
| 合计 | 159 | 35 | 22.0% |

用户观察到点头、摇头、身体前后移动仍可能触发检测。下一轮不把多 bin 用作 blink 增强，而是反过来用于大动作拒绝：

```text
如果一个 acoustic candidate 附近，很多 range bins 同时出现事件，
说明更像头部/身体大范围运动；
如果只有校准出的 eye bin 附近有响应，更像局部眨眼。
```

离线先实现事件级 spread filter：

```text
for selected_event in selected_bin_events:
    spread_bins = count(range_bin with event within +/- motion_window)
    if spread_bins >= motion_reject_spread_bins:
        reject selected_event
```

计划先试：

```text
--blink-motion-reject-window 0.25
--blink-motion-reject-spread-bins 4, 6, 8, 10
```

验收目标：

```text
合计 unexplained 从 35 降到 20 以下；
召回尽量保持 >= 94%。
```

如果事件级 spread filter 误伤太多，再进入第二层特征：

```text
range_profile_energy_delta
range_center_shift
amplitude_delta_ratio spread
segment duration / width / baseline return
```
