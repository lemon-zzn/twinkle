# FMCW + Twinkle 眨眼检测实验代码修改记录

日期：2026-06-17

本文记录 `hp_acoustic_wave` 中把 FMCW 与 TwinkleTwinkle 思路接入眨眼检测后的问题分析、代码修改和实验结果。当前重点是 `blink` 模式下的 `--blink-method twinkle --signal-mode fmcw` 路径。

## 实验命令

主要采集命令：

```powershell
& 'C:\Program Files\Python38\python.exe' .\run_hp_wave_detector.py --mode blink --blink-method twinkle --signal-mode fmcw --fmcw-range-bin 15 --amplitude 0.2 --input-device 1 --output-device 3 --camera-width 640 --camera-height 480
```

该命令默认开启视觉标注。`events.csv` 中：

- `visual_blink` 是 MediaPipe EAR 视觉真值；
- `blink_candidate` 是声学 FMCW + Twinkle 候选事件。

## 关键问题与原因分析

### 1. 新增 FMCW Twinkle 指标后 CSV 写入崩溃

现象：

```text
ValueError: dict contains fields not in fieldnames:
'twinkle_fmcw_amplitude_ok', 'twinkle_fmcw_amplitude_stable',
'twinkle_fmcw_phase_score', 'twinkle_fmcw_range_bin',
'twinkle_fmcw_amplitude_delta_ratio'
```

原因：

- `app.py` 的 `_feature_row()` 已经写入新的 `twinkle_fmcw_*` 调试列；
- `session_io.py` 的 `FEATURE_FIELDS` 没有同步加入这些列；
- Python `csv.DictWriter` 默认 `extrasaction="raise"`，遇到额外字段直接抛异常。

修改：

- `session_io.py`
  - 在 `FEATURE_FIELDS` 中补齐 `twinkle_fmcw_range_bin`、`twinkle_fmcw_phase_score`、`twinkle_fmcw_amplitude_ok`、`twinkle_fmcw_amplitude_stable`、`twinkle_fmcw_amplitude_delta_ratio`。
  - 后续又补充 `twinkle_fmcw_min_score`、`twinkle_fmcw_score_ok`、`twinkle_effective_refractory_s`，方便定位门控原因。
- `tests/test_session_io.py`
  - 新增 `test_session_writer_accepts_fmcw_twinkle_feature_columns()`，防止再出现表头不同步。

### 2. `sessions/hp_blink_20260617_131250` 人工标注下误检较多

原始结果：

```text
source=events
events=49 blink_hits=16/30 unexplained_events=34 balanced_score=-2.70
```

分析：

- 当时 FMCW Twinkle 的声学阈值长期卡在 `0.006` 最低值；
- range-bin 15 的普通相位波动也能超过最低阈值；
- `blink_candidate` 很容易被重复触发；
- 离线 benchmark 当时没有恢复 `signal_mode=fmcw`，会错误走 tone 代理逻辑，导致回放结果不可信。

修改：

- `blink_detector.py`
  - 增加 FMCW 专用 score floor：`twinkle_fmcw_min_score`。
  - 增加 FMCW 专用触发间隔：`twinkle_fmcw_refractory_s`。
  - 在 FMCW 下先检查 selected range-bin 幅度、幅度稳定性和 score floor，再允许进入事件门控。
- `benchmark.py`
  - `_feature_from_row()` 恢复 `signal_mode`、`range_bin`、`range_distance_m`，确保 `features.csv` 离线回放仍走 FMCW 逻辑。
- `benchmark_hp_blink.py`
  - 补充 FMCW Twinkle 调参参数，支持离线扫参。

当时基于人工 marker 的保守调参：

```text
twinkle_fmcw_min_score = 0.09
twinkle_fmcw_refractory_s = 1.7
```

回放结果：

```text
source=features
events=33 blink_hits=16/30 unexplained_events=18 balanced_score=6.10
```

结论：

- 人工 marker 评估下，保守门控显著减少误检；
- 但该组 marker 不如后续视觉真值密集，不能单独决定最终默认参数。

### 3. `sessions/hp_blink_20260617_133830` 使用视觉标注作为真值后，发现主要问题是漏检

该组数据中视觉标注已确认准确：

```text
visual_blink_count = 71
auto_event_count = 25
```

原始声学候选按视觉真值评估：

```text
source=events
truth=visual
events=25 blink_hits=28/71 unexplained_events=3 balanced_score=26.35
```

补充说明：

- `benchmark.py` 的 `blink_hits` 是“视觉 blink 是否在窗口附近有声学候选”，一个声学候选可能覆盖相邻的多个视觉 blink；
- 因此 `blink_hits` 可能大于 `events`；
- 另用一对一匹配评估时，当前参数更直观。

参数扫描结论：

- 旧保守参数 `0.09 / 1.7s` 精度较高，但漏检偏多；
- `twinkle_fmcw_min_score = 0.07`、`twinkle_fmcw_refractory_s = 1.3` 在视觉真值下是更好的折中；
- 新默认参数相比旧参数，多抓到约 4 个视觉真 blink，额外增加约 1 个误检。

新默认参数回放：

```text
source=features
truth=visual
events=30 blink_hits=32/71 unexplained_events=4 balanced_score=29.80
```

一对一 `+-0.8s` 对齐：

```text
events=30
tp=26
fp=4
fn=45
precision=0.87
recall=0.37
```

结论：

- 当前 FMCW + Twinkle 声学候选精度已经比较好；
- 最大短板是召回率，说明并不是所有视觉眨眼都在 range-bin 15 上有足够稳定的声学响应；
- 后续优先尝试多 range-bin 或换 range-bin，而不是继续单纯降低阈值。

## 主要代码修改

### `blink_detector.py`

新增配置：

```python
twinkle_fmcw_min_amplitude: float = 1e-4
twinkle_fmcw_phase_pair_weight: float = 0.5
twinkle_fmcw_max_amplitude_delta_ratio: float = 0.45
twinkle_fmcw_min_score: float = 0.07
twinkle_fmcw_refractory_s: float = 1.3
```

关键逻辑：

- 维护 `amplitude_window`，计算 selected range-bin 幅度相对变化；
- `_fmcw_twinkle_score()` 在 `feature.signal_mode == "fmcw"` 时启用：
  - 幅度低于 `twinkle_fmcw_min_amplitude` 时拒绝；
  - 幅度变化比超过 `twinkle_fmcw_max_amplitude_delta_ratio` 时拒绝；
  - 使用 `phase_pair_delta * twinkle_fmcw_phase_pair_weight` 作为 FMCW phase score；
  - 最终 score 低于 `twinkle_fmcw_min_score` 时拒绝；
- `_TwinklePeakEventGate.update()` 支持传入 `refractory_s`，FMCW 路径使用 `twinkle_fmcw_refractory_s`。

### `config.py`

`BlinkConfig` 同步新增 FMCW Twinkle 参数，当前默认值：

```python
twinkle_fmcw_min_score = 0.07
twinkle_fmcw_refractory_s = 1.3
```

### `run_hp_wave_detector.py`

新增命令行参数：

```powershell
--blink-twinkle-fmcw-min-amplitude
--blink-twinkle-fmcw-phase-pair-weight
--blink-twinkle-fmcw-max-amplitude-delta-ratio
--blink-twinkle-fmcw-min-score
--blink-twinkle-fmcw-refractory
```

这些参数会写入 `BlinkConfig`，并通过 `AppConfig` 传给实时检测器。

### `app.py`

`features.csv` 增加 FMCW Twinkle 调试列：

```text
twinkle_fmcw_range_bin
twinkle_fmcw_phase_score
twinkle_fmcw_amplitude_ok
twinkle_fmcw_amplitude_stable
twinkle_fmcw_amplitude_delta_ratio
twinkle_fmcw_min_score
twinkle_fmcw_score_ok
twinkle_effective_refractory_s
```

这些列用于判断一次声学候选为什么通过或被挡住。

### `session_io.py`

同步扩展 `FEATURE_FIELDS`，保证 `app.py` 写出的所有列都能进入 `features.csv`。

同时视觉标注路径会写：

- `visual_labels.csv`
- `events.csv` 中的 `visual_blink`
- `features.csv` 中最近视觉状态字段

### `benchmark.py`

新增视觉真值评估能力：

- `truth="manual"`：使用 `manual_markers.csv`；
- `truth="visual"`：使用 `events.csv` 中的 `visual_blink`，必要时回退到 `visual_labels.csv`；
- `truth="auto"`：有视觉真值时优先使用视觉真值，否则用人工 marker。

`source="events"` 时会过滤掉 `visual_blink`，只把 `blink_candidate` 当作待评估的声学事件。

### `benchmark_hp_blink.py`

新增：

```powershell
--truth auto|manual|visual
```

默认 `auto`。因此对带视觉标注的新 session，直接运行：

```powershell
python benchmark_hp_blink.py --session sessions\hp_blink_20260617_133830 --source features
```

会自动使用视觉真值。

## 新增/更新测试

新增或扩展的测试覆盖：

- `tests/test_session_io.py`
  - CSV 表头支持 FMCW Twinkle 调试列；
  - 视觉标注文件写入。
- `tests/test_benchmark_fmcw.py`
  - `features.csv` 离线回放保留 `signal_mode=fmcw`。
- `tests/test_benchmark_visual_truth.py`
  - `benchmark_session(..., truth="visual")` 使用视觉真值；
  - `truth="auto"` 优先选择视觉真值。
- `tests/test_blink_algorithms.py`
  - FMCW 低幅度 range-bin 相位噪声不触发；
  - 低于 `twinkle_fmcw_min_score` 的小相位轨迹不触发；
  - FMCW 使用专用 refractory；
  - 默认 FMCW Twinkle 参数跟随视觉真值调参结果。
- `tests/test_fmcw_wave_integration.py`
  - CLI 接受 FMCW Twinkle 调参参数；
  - 默认参数为 `0.07 / 1.3s`。

完整验证命令：

```powershell
python -m pytest hp_acoustic_wave\tests tests\test_hp_acoustic_wave_benchmark.py tests\test_hp_acoustic_wave_session_io.py -q
```

当前结果：

```text
50 passed in 2.67s
```

## 当前结论

1. FMCW + Twinkle 在 range-bin 15 上已经能产生较准确的声学候选。
2. 视觉真值显示当前主要问题是召回不足，不是误检失控。
3. 单 range-bin 继续降低阈值会带来更多候选，但精度会下降；更合理的下一步是：
   - 尝试多个 range-bin；
   - 对多个 bin 的 phase profile 做投票；
   - 记录每个视觉 blink 附近不同 bin 的响应强度；
   - 再决定眨眼最稳定的 range-bin 或多 bin 组合。

## 后续建议

下一阶段建议做一个离线分析脚本：

```text
输入：audio.wav + visual_blink 真值时间
输出：range-bin 10-25 每个 bin 在视觉 blink 附近的 phase span / phase score / amplitude 稳定性
```

目标是回答：

- range-bin 15 是否真的是眨眼最强 bin；
- 是否存在某些 blink 在 bin 14/16/17 更明显；
- 多 bin 投票能否把召回从当前约 0.37 提高，同时保持 precision 在 0.8 以上。
