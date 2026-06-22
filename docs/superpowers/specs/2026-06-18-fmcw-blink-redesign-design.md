# FMCW 眨眼检测算法重做（shape-segmentation + BlinkListener LEVD）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 Twinkle 论文的 shape-based segmentation 和 BlinkListener 论文的 LEVD bump detection 替换当前的 peak + coherence score + refractory 检测器，消除噪声驱动的周期性假触发，并在 9 个 recorded session 上验证。

**Architecture:** 分 5 个 phase：(0) 删除 dead code + 统一 config；(1) 实时波形改成 phase trajectory + ungated coherence 双曲线；(2) 新增 `_TwinkleShapeSegmentationDetector`；(3) 审计现有 `BlinkListenerBlinkDetector` + CLI 接线；(4) 在 9 个 session 上 benchmark + anti-periodic 检查。

**Tech Stack:** Python 3.8, numpy, 现有 hp_acoustic_wave DSP/detector 框架。

---

## 背景

当前 `_TwinkleTwinkleBlinkDetector` + `_TwinklePeakEventGate`（peak + coherence + refractory）的周期性假触发根源：score 单调上升 → fire → refractory → 再上升 → fire。噪声驱动的 coherence peak 重复触发。

Fix A+B+C（已 commit `ccce4f7`）加了 smoothness/deviation/min_score 门来掩盖问题，但 trade-off 是 recall 严重下降（190655 从 27→13 TP）。本次"归零心态" — 移除所有 min/上下界门，让算法本身（pattern detection + adaptive LEVD）来区分 blink vs noise。

## 设计决策

1. **删除 `twinkle_fmcw_min_score`、`twinkle_fmcw_min_smoothness`、`twinkle_fmcw_min_deviation`** — 新检测器不依赖任何 min/max 门。
2. **不 unwrap phase trajectory** — 实时波形显示 raw `phase_pair_delta`，shape-segmentation 内部自己平滑。
3. **shape 支持双向（上凸+下凸）** — 真实 blink 任意方向都接受，取窗口内最大幅差。
4. **不做 composite detector** — 先 benchmark shape vs blinklistener 三选一，跑完看哪个赢。
5. **审计现有 BlinkListenerBlinkDetector，发现 bug 才改** — 不写新算法，只接 CLI + benchmark。
6. **测试集固定为 9 个 session** — CLAUDE.md 的 184117/190655/214655 + baseline doc 的 A/A'/B/C/D/E。

---

## Phase 0: Cleanup & Config Unification

**Files:**
- Modify: `hp_acoustic_wave/blink_detector.py` (删除 orphaned metrics, 删除 `twinkle_fmcw_min_score`/`min_smoothness`/`min_deviation` 字段和引用, 统一 `BlinkDetectionConfig`)
- Modify: `hp_acoustic_wave/config.py` (删除对应 `BlinkConfig` 字段)
- Modify: `hp_acoustic_wave/run_hp_wave_detector.py` (删除对应 CLI 参数)
- Modify: `hp_acoustic_wave/benchmark_hp_blink.py` (删除对应 CLI 参数)
- Modify: `hp_acoustic_wave/run_emission_*.sh` (5 个 launch scripts，删除 `--blink-twinkle-fmcw-min-score/--blink-twinkle-fmcw-min-smoothness/--blink-twinkle-fmcw-min-deviation` 行)

- [ ] **Step 0.1: 删除 orphaned metrics**

在 `blink_detector.py` 中（约 L380-440 的 metrics dict），删除以下 keys 及其引用：
- `twinkle_phase_pair_library_*` 系列
- `twinkle_fmcw_has_spatial_spread`, `spread_bins`, `spread_ratio`, `dominance_ratio`, `spatial_spread_ok`
- `twinkle_segment_coarse_votes`, `twinkle_segment_fine_votes`, `twinkle_shape_ok`
- `twinkle_rhythm_*` 系列

对应 `BlinkDetectionConfig`（L12-48）有字段就删字段。

- [ ] **Step 0.2: 删除 min-score 系列门**

在 `blink_detector.py`、`config.py`、`run_hp_wave_detector.py`、`benchmark_hp_blink.py`、5 个 `run_emission_*.sh` 中删除：
- `twinkle_fmcw_min_score`
- `twinkle_fmcw_min_smoothness`
- `twinkle_fmcw_min_deviation`

同时删除 `_TwinklePeakEventGate.stats()` 里的 `min_score_floor` 逻辑（Fix B），FMCW path 改回纯 adaptive `baseline + threshold_k × robust_sigma`。

- [ ] **Step 0.3: 简化 `_candidate_trajectory_score`**

当 `twinkle_candidate_windows` 为空时此函数是 passthrough（L613-615）。检查调用点（应该只在一处），inline 化或删除整个 method。如果字段 `twinkle_candidate_windows` / `twinkle_min_candidate_votes` 没其他用途，一并删掉。

- [ ] **Step 0.4: 统一 BlinkConfig / BlinkDetectionConfig**

选 `config.py:BlinkConfig` 为唯一 source of truth。`blink_detector.py:BlinkDetectionConfig` 改成 `from config import BlinkConfig as BlinkDetectionConfig` 的 re-export，或者 `class BlinkDetectionConfig(BlinkConfig): pass`。修正字段漂移。

- [ ] **Step 0.5: 保留以下不动**
- `_amplitude_phase_orthogonality_score`（默认 off，功能正常）
- `_trajectory_phase` 支持 intra-chirp phase-pair（shape-segmentation 要用）
- `FmcwBackgroundSubtractor`（app.py 用）

- [ ] **Step 0.6: 验证**

跑现有单元测试 `tests/test_blink_algorithms.py`、`tests/test_fmcw_wave_integration.py`。如果有测试引用删除的字段，同步删掉断言。预期：测试通过或只失败 pre-existing 的 phase-pair coherence 测试（这些是之前就 fail 的）。

- [ ] **Step 0.7: Commit**

```bash
git add hp_acoustic_wave/blink_detector.py hp_acoustic_wave/config.py hp_acoustic_wave/run_hp_wave_detector.py hp_acoustic_wave/benchmark_hp_blink.py hp_acoustic_wave/run_emission_*.sh
git commit -m "refactor(blink): remove dead code and min-score gates (phase 0)"
```

---

## Phase 1: 双曲线波形可视化

**Files:**
- Modify: `hp_acoustic_wave/app.py` (`_draw_overlay` L493-511，新增 history buffers)

- [ ] **Step 1.1: 暴露 ungated coherence**

在 `_fmcw_twinkle_score`（blink_detector.py L634-709）的 return 之前，添加：
```python
self.last_ungated_coherence = float(coherence_score)  # gates 应用前的原始值
```
在 `__init__` 里初始化 `self.last_ungated_coherence = 0.0`。
对 `BlinkListenerBlinkDetector`，初始化 `self.last_ungated_coherence = 0.0`（这个 method 没有 coherence 概念，下半波形显示已有的 `viewing_amplitude` / `raw_viewing_score` 即可）。

- [ ] **Step 1.2: 新增 history buffers**

在 `AcousticApp.__init__`（或 setup visualization 处）添加：
```python
self.phase_trajectory_history = deque(maxlen=self.config.visualization.energy_history_size)
self.ungated_coherence_history = deque(maxlen=self.config.visualization.energy_history_size)
```

- [ ] **Step 1.3: 填充 history**

在 `_extract_feature` 或 `_update_blink_detector` 调用后，append：
```python
phase_val = (
    feature.phase_pair_delta
    if (feature.signal_mode == "fmcw" and feature.phase_pair_delta is not None)
    else feature.phase
)
self.phase_trajectory_history.append(phase_val)
self.ungated_coherence_history.append(
    getattr(self.detector, "last_ungated_coherence", 0.0)
)
```

- [ ] **Step 1.4: 改造 `_draw_overlay` 双 panel**

把现有 bottom panel（L493-511）拆成上下两半：
- **上半（phase trajectory）**：绿色 polyline，y auto-scale 到 `±max_abs(values)`，居中 0。永远不归零（phase 是连续的）。
- **下半（ungated coherence）**：蓝色 polyline + threshold 橙色虚线 + min_score_floor 红色横线（如果 `min_score` 还在）。

布局：bottom panel 高度 `plot_h`，上半占 `plot_h // 2`，下半占 `plot_h // 2`。两个 panel 共享 x 轴。

- [ ] **Step 1.5: Commit**

```bash
git add hp_acoustic_wave/app.py hp_acoustic_wave/blink_detector.py
git commit -m "feat(viz): dual-curve bottom panel (phase trajectory + ungated coherence)"
```

---

## Phase 2: Shape-Segmentation Detector

**Files:**
- Modify: `hp_acoustic_wave/blink_detector.py` (新增 `_TwinkleShapeSegmentationDetector` 类)
- Modify: `hp_acoustic_wave/config.py` (新增 shape 参数)
- Modify: `hp_acoustic_wave/run_hp_wave_detector.py` (CLI flag)
- Modify: `hp_acoustic_wave/benchmark_hp_blink.py` (CLI flag)
- Create: `hp_acoustic_wave/tests/test_shape_segmentation.py`

### 参数（写到 BlinkConfig）

```python
shape_smoothing_window: int = 5      # chunks
shape_detection_window: int = 30    # chunks, ~1.5s @ 20Hz
shape_min_edge_len_s: float = 0.05
shape_max_edge_len_s: float = 0.5
shape_edge_sigma_k: float = 3.0     # edge_mag >= k × baseline_sigma
shape_refractory_s: float = 0.5
```

- [ ] **Step 2.1: 写失败测试**

`tests/test_shape_segmentation.py`：

```python
import numpy as np
import pytest
from hp_acoustic_wave.blink_detector import (
    BlinkDetectionConfig, _TwinkleShapeSegmentationDetector
)
from hp_acoustic_wave.dsp import ChunkFeature


def make_feature(time_s, phase_pair_delta, signal_mode="fmcw"):
    # ChunkFeature 字段完整列表见 dsp.py:ChunkFeature dataclass 定义；
    # shape detector 只用 time_s / signal_mode / phase / phase_pair_delta，其他给默认值。
    return ChunkFeature(
        time_s=time_s, signal_mode=signal_mode,
        i_value=0.0, q_value=0.0, amplitude=0.1, phase=float(phase_pair_delta),
        phase_delta=0.0, phase_pair_delta=phase_pair_delta,
        motion_energy=0.0, sign_changes=0,
        # 按 dsp.py 实际签名补全其他字段（如 magnitude / unwrapped_phase 等）
    )


def test_shape_detects_down_up_edge_in_blink_duration():
    """合成一个先下后上的 phase_pair_delta 序列，时间间隔 0.2s，应该触发一次 blink。"""
    config = BlinkDetectionConfig(blink_method="shape")
    det = _TwinkleShapeSegmentationDetector(config)
    events = []
    # baseline noise for 1s, then down-up edge
    dt = 0.05
    for i in range(20):
        f = make_feature(i * dt, 0.001 * (i % 3 - 1))  # small noise
        r = det.update(f)
        if r.is_event: events.append(f.time_s)
    # down-up edge: t=1.0 to 1.2
    trajectory = [(1.0, -0.5), (1.05, -0.4), (1.1, 0.0), (1.15, 0.4), (1.2, 0.5)]
    for t, v in trajectory:
        for _ in range(int(0.05 / dt)):
            f = make_feature(t, v)
            r = det.update(f)
            if r.is_event: events.append(f.time_s)
    assert len(events) >= 1, "down-up edge should trigger at least one blink"
    # refractory should prevent double-fire on same edge
    assert len(events) <= 2


def test_shape_ignores_single_peak_noise():
    """单个噪声峰不应该触发 — 不形成 down-up pattern。"""
    config = BlinkDetectionConfig(blink_method="shape")
    det = _TwinkleShapeSegmentationDetector(config)
    events = []
    dt = 0.05
    # random noise
    rng = np.random.default_rng(42)
    for i in range(200):
        f = make_feature(i * dt, rng.normal(0, 0.01))
        r = det.update(f)
        if r.is_event: events.append(f.time_s)
    assert len(events) <= 2, "random noise should not cause >2 false positives"


def test_shape_refractory_prevents_double_fire():
    """单次 blink 不应该触发多次（refractory 0.5s）。"""
    config = BlinkDetectionConfig(blink_method="shape")
    det = _TwinkleShapeSegmentationDetector(config)
    events = []
    dt = 0.05
    for i in range(20):
        f = make_feature(i * dt, 0.0)
        det.update(f)
    # single blink
    for v in [-0.5, -0.4, 0.0, 0.4, 0.5, 0.0, 0.0]:
        f = make_feature(1.0, v)
        r = det.update(f)
        if r.is_event: events.append(f.time_s)
    assert len(events) == 1
```

- [ ] **Step 2.2: 运行测试验证失败**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -m pytest hp_acoustic_wave/tests/test_shape_segmentation.py -v`
Expected: FAIL with "cannot import name `_TwinkleShapeSegmentationDetector`"

- [ ] **Step 2.3: 实现 `_TwinkleShapeSegmentationDetector`**

在 `blink_detector.py` 新增（参考 Section 3 的伪代码）：

```python
class _TwinkleShapeSegmentationDetector:
    """Twinkle paper's shape-based segmentation on phase trajectory.

    Detects a down-up edge (or up-down) within time bounds [min_edge_len, max_edge_len].
    Noise-driven single peaks do NOT form a coherent pattern, so periodic
    false-positive firing is structurally prevented.
    """
    method = "shape"

    def __init__(self, config: BlinkDetectionConfig):
        self.config = config
        self.window = deque(maxlen=config.shape_detection_window)
        self.smooth_buffer = deque(maxlen=config.shape_smoothing_window)
        self.baseline = deque(maxlen=200)  # 10s @ 20Hz
        self.last_blink_time = -1e9
        self.last_ungated_coherence = 0.0

    def _trajectory_value(self, feature: ChunkFeature) -> float:
        if (feature.signal_mode == "fmcw"
            and feature.phase_pair_delta is not None):
            return float(feature.phase_pair_delta)
        return float(feature.phase)

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        delta = self._trajectory_value(feature)
        self.smooth_buffer.append(delta)
        smoothed = float(np.mean(self.smooth_buffer))
        self.window.append((feature.time_s, smoothed))
        self.baseline.append(smoothed)

        # baseline sigma via MAD
        if len(self.baseline) >= 5:
            med = float(np.median(self.baseline))
            mad = float(np.median(np.abs(np.asarray(self.baseline) - med)))
            sigma = max(mad * 1.4826, 1e-6)
        else:
            sigma = 1e-6

        # refractory
        if feature.time_s - self.last_blink_time < self.config.shape_refractory_s:
            return BlinkDetectionResult(
                is_event=False, event_id=0, method=self.method,
                score=0.0, threshold=float(sigma * self.config.shape_edge_sigma_k),
                baseline=float(med if len(self.baseline) >= 5 else 0.0),
                mad=float(mad if len(self.baseline) >= 5 else 0.0),
                metrics={"edge_mag": 0.0, "edge_len": 0.0, "baseline_sigma": sigma},
            )

        # need enough samples for edge detection
        if len(self.window) < max(3, int(self.config.shape_min_edge_len_s * 20)):
            return BlinkDetectionResult(
                is_event=False, event_id=0, method=self.method,
                score=0.0, threshold=float(sigma * self.config.shape_edge_sigma_k),
                baseline=float(med if len(self.baseline) >= 5 else 0.0),
                mad=float(mad if len(self.baseline) >= 5 else 0.0),
                metrics={"edge_mag": 0.0, "edge_len": 0.0, "baseline_sigma": sigma},
            )

        times = [p[0] for p in self.window]
        vals = [p[1] for p in self.window]
        i_min = int(np.argmin(vals))
        i_max = int(np.argmax(vals))
        edge_len = abs(times[i_max] - times[i_min])
        edge_mag = abs(vals[i_max] - vals[i_min])

        is_event = (
            edge_len >= self.config.shape_min_edge_len_s
            and edge_len <= self.config.shape_max_edge_len_s
            and edge_mag >= self.config.shape_edge_sigma_k * sigma
        )

        if is_event:
            self.last_blink_time = feature.time_s
            confidence = edge_mag / sigma

        return BlinkDetectionResult(
            is_event=is_event,
            event_id=(1 if is_event else 0),
            method=self.method,
            score=(float(confidence) if is_event else 0.0),
            threshold=float(self.config.shape_edge_sigma_k * sigma),
            baseline=float(med if len(self.baseline) >= 5 else 0.0),
            mad=float(mad if len(self.baseline) >= 5 else 0.0),
            metrics={
                "edge_mag": float(edge_mag),
                "edge_len": float(edge_len),
                "baseline_sigma": float(sigma),
            },
        )
```

- [ ] **Step 2.4: 运行测试验证通过**

Run: `cd E:/android_projects/eye_blink_detect && "/c/Program Files/Python38/python.exe" -m pytest hp_acoustic_wave/tests/test_shape_segmentation.py -v`
Expected: PASS

- [ ] **Step 2.5: 注册到 `build_blink_detector`**

让 `build_blink_detector` 根据 `config.blink.method` 选 `_TwinkleShapeSegmentationDetector`。

- [ ] **Step 2.6: CLI flag**

`run_hp_wave_detector.py` 和 `benchmark_hp_blink.py` 的 `--blink-method` 已经接受 `twinkle`，扩展 choices 到 `["twinkle", "blinklistener", "shape"]`。

- [ ] **Step 2.7: Commit**

```bash
git add hp_acoustic_wave/blink_detector.py hp_acoustic_wave/config.py hp_acoustic_wave/run_hp_wave_detector.py hp_acoustic_wave/benchmark_hp_blink.py hp_acoustic_wave/tests/test_shape_segmentation.py
git commit -m "feat(blink): add shape-segmentation detector (twinkle paper pattern matching)"
```

---

## Phase 3: 审计 BlinkListenerBlinkDetector + CLI 接线

**Files:**
- Modify: `hp_acoustic_wave/blink_detector.py` (审计 `BlinkListenerBlinkDetector` L265-367 + `_RobustEventGate` L63-117，发现 bug 才改)
- Read-only: `E:\android_projects\eye_blink_detect\FaceAcousticSensing\papers\BlinkListener "Listen" to Your Eye Blink Using Your Smartphone.pdf` (对照 LEVD 公式)

- [ ] **Step 3.1: 审计 `BlinkListenerBlinkDetector`**

对照 BlinkListener 论文（Sec 3.2 + Sec 6.3）检查：
- 12 角度投影：是否取 viewing-position 幅差（论文核心），不是 origin 投影？看 `_best_viewing_projection` (L295-311)。
- LEVD 阈值：`_RobustEventGate` 的 `threshold_k`、`refractory_s`、`baseline_freeze_s` 是否合理？
- score 归一化：`raw_score / baseline_amplitude`（L331-332）是否符合论文的相对量定义？
- `phase_stable_projection` (L319)：`abs(feature.phase_delta) > 0.12` 的 0.12 常数 — 论文有依据吗？或者要改成 config 参数？

输出审计结论到 `experiment_blinklistener_audit_20260618.md`（或 brainstorming spec 的附录）。

- [ ] **Step 3.2: 发现 bug 才改**

如果审计发现实现错误（例如投影方向错、归一化分母错），修复。否则不改代码。

- [ ] **Step 3.3: CLI 接线（如果没有 Step 3.2）**

确认 `--blink-method blinklistener` 在 `run_hp_wave_detector.py` 和 `benchmark_hp_blink.py` 中可用（应该已经有，确认 choices 列表）。

- [ ] **Step 3.4: Commit（如果有改动）**

```bash
git add hp_acoustic_wave/blink_detector.py
git commit -m "fix(blinklistener): audit findings on I/Q + LEVD implementation"
```

---

## Phase 4: Benchmark + Anti-Periodic Verification

**Files:**
- Modify: `hp_acoustic_wave/benchmark_hp_blink.py` (新增 `--check-periodic` flag)
- Create: `hp_acoustic_wave/experiment_shape_vs_blinklistener_20260618.md`

### 测试集（9 个 session）

CLAUDE.md：184117, 190655, 214655
baseline doc：162711 (A linear), 172844 (A' linear_v2), 173242 (B linear_tukey), 173057 (D triangle), 173428 (C hybrid), 174829 (E cw_single)

- [ ] **Step 4.1: 新增 anti-periodic 检查**

在 `benchmark_hp_blink.py` 新增 `--check-periodic` flag，对每个 detector 的 event 输出：
```python
def check_periodic(event_times, refractory_s, tolerance=0.2):
    if len(event_times) < 3:
        return 0.0
    gaps = np.diff(sorted(event_times))
    clustered = sum(abs(g - refractory_s) < tolerance for g in gaps)
    return clustered / len(gaps)
```
打印 `cluster_ratio` per detector per session。

- [ ] **Step 4.2: 跑 benchmark — shape**

对 9 个 session 跑 `--blink-method shape`，session 路径已知（baseline doc）。命令模板：
```bash
"/c/Program Files/Python38/python.exe" benchmark_hp_blink.py \
  --session sessions/hp_blink_20260617_184117 --source audio --truth auto \
  --blink-method shape --check-periodic
```
收集 events/TP/FP/FN/f1/balanced/cluster_ratio。

- [ ] **Step 4.3: 跑 benchmark — blinklistener**

同样 9 个 session，`--blink-method blinklistener`。

- [ ] **Step 4.4: 跑 benchmark — twinkle（baseline 对比）**

同样 9 个 session，`--blink-method twinkle`（旧路径，无门 — 因为 Phase 0 删了 min_score）。

- [ ] **Step 4.5: 写结果到 `experiment_shape_vs_blinklistener_20260618.md`**

三个 detector × 9 session 的完整 table，加上 hard checks 结果：
- 214655: events < 20（baseline 363）
- 184117, 190655: balanced >= 0
- threshold 不锁死在 min_score（已删）
- cluster_ratio < 0.3（anti-periodic）

### 决策门

- 如果 shape 在多数 variant 上 balanced > twinkle → shape 成为默认。
- 如果 blinklistener 在某些 variant 上更好 → 提示用户按 variant 选 method。
- 如果两个都差 → 看诊断图调 `shape_edge_sigma_k`（降到 2.0 重跑）。

- [ ] **Step 4.6: Commit**

```bash
git add hp_acoustic_wave/benchmark_hp_blink.py hp_acoustic_wave/experiment_shape_vs_blinklistener_20260618.md
git commit -m "experiment: benchmark shape vs blinklistener vs twinkle on 9 sessions"
```

---

## 验证标准（汇总）

| 指标 | 阈值 | 备注 |
|---|---|---|
| 214655 events | < 20 | baseline 363，周期误判的源头 |
| 184117 balanced | >= 0 | 不能比"完全不检测"还差 |
| 190655 balanced | >= 0 | 同上 |
| threshold | 不锁死 | min_score 已删，FMCW 纯 adaptive |
| cluster_ratio | < 0.3 | anti-periodic metric |

## 文件清单

| 文件 | 操作 | Phase |
|---|---|---|
| `blink_detector.py` | 删 dead code + 删 min-score + 新增 shape detector + 审计 blinklistener | 0/2/3 |
| `config.py` | 删 min-score 字段 + 新增 shape 参数 | 0/2 |
| `run_hp_wave_detector.py` | 删 CLI + 扩展 `--blink-method` choices | 0/2 |
| `benchmark_hp_blink.py` | 删 CLI + 扩展 choices + 新增 `--check-periodic` | 0/2/4 |
| `run_emission_*.sh` (5 个) | 删 min-score 行 | 0 |
| `app.py` | 双曲线波形 | 1 |
| `tests/test_shape_segmentation.py` | 新建 | 2 |
| `experiment_shape_vs_blinklistener_20260618.md` | 新建 | 4 |
| `experiment_blinklistener_audit_20260618.md` | 新建（如审计有结论） | 3 |
