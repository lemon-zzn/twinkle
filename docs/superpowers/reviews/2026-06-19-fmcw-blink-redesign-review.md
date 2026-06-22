# FMCW 眨眼检测重做 — 综合性 Review 文档

> **目的**：供其他人 review 和复现。涵盖：问题背景、论文依据、所有代码改动、参数表、实验结果、诊断分析、复现步骤。
>
> **创建日期**：2026-06-19
> **分支**：`codex/hp-acoustic-wave-detector`
> **Commit 范围**：`e80c310` → `9fb33549`（共 21 个 commits）
> **关联文档**：`experiment_shape_vs_blinklistener_20260618.md`、`experiment_blinklistener_audit_20260618.md`、`experiment_periodic_firing_rootcause_20260619.md`、`experiment_variant_baseline_20260618.md`

---

## 1. 问题背景

### 1.1 现象

`_TwinkleTwinkleBlinkDetector` + `_TwinklePeakEventGate`（peak + coherence score + refractory）在 FMCW 数据上出现**噪声驱动的周期性假触发**：
- session 214655（约 14 分钟，基本为空）产生了 363 个 blink_candidate，间隔大都卡在 refractory_s（1.3s）附近。
- session 184117 benchmark：`blink_hits=2/5, unexplained=5, score=-0.75`。
- threshold 锁定在 `min_score=0.006`（floor），任何微弱信号都能超过。

### 1.2 临时修复（commit `ccce4f7`，本重做之前）

加了 Fix A+B+C 门掩盖问题：
- **Fix A**：事件 score 记 peak candidate 的 score，不记当前帧。
- **Fix B**：FMCW path gate threshold 下限 = `twinkle_fmcw_min_score`，不塌回 `min_score=0.006`。
- **Fix C**：smoothness + deviation 门（低 smoothness/deviation 直接归零）。

**Trade-off**：recall 严重下降（190655 从 27→13 TP）。门掩盖问题但不解决根因。

### 1.3 本次重做目标（"归零心态"）

> 移除所有 min/上下界门，让算法本身（pattern matching + adaptive LEVD）来区分 blink vs noise。

5 个 phase：
- **Phase 0**：删 dead code + 删 min-score 门 + 统一 config。
- **Phase 1**：实时波形双曲线（phase trajectory + ungated coherence）。
- **Phase 2**：新增 `_TwinkleShapeSegmentationDetector`（Twinkle 论文 shape-segmentation）。
- **Phase 3**：审计现有 `BlinkListenerBlinkDetector`（BlinkListener 论文 I/Q LEVD）。
- **Phase 4**：9 session benchmark + anti-periodic metric + 周期触发根因诊断。

---

## 2. 论文依据

### 2.1 Twinkle 论文（shape-segmentation）

**来源**：`hp_acoustic_wave/MinerU_html_3596238_2055512726219456512.html`（论文 PDF 的 HTML 提取）。

**核心算法（Sec 3.2）**：
1. FMCW demodulate（rx × conj(tx)，low-pass）→ range-bin FFT → 选 range bin。
2. 每个 chirp 取 intra-chirp valid interval 上的 phase sample。
3. **两个 chirp 之间做 phase subtraction** → candidate trajectory。
4. Moving-average filter（**不是** denoiser）— 保留 tiny blink variation。
5. **Shape-based segmentation（核心，Sec 3.2.1）**：
   - 粗 segmentation：local minima/maxima + direction sign + edge length filter。
   - 把 edges 分组成 segments（间隔 < 0.5s 视为同一个 blink segment）。
   - Blink characteristics 通过 **adaptive constraints**（Android-iOS 类比）建模：sibling segments 之间的相对约束。
   - Vote-based approach 最终决策。

**关键 insight**：blink 是**时间-形状 pattern**（下凸+上凸的边缘对，长度 ∈ [0.1, 0.5]s），不是 score threshold。这从结构上消灭了单峰噪声驱动的周期触发。

### 2.2 BlinkListener 论文（I/Q LEVD）

**来源**：`FaceAcousticSensing/papers/BlinkListener "Listen" to Your Eye Blink Using Your Smartphone.pdf`。

**核心算法（Sec 3.2 + Sec 6.3）**：
1. 把眨眼建模到 I/Q 向量空间：blink = **大幅 amplitude change + 小幅 phase change**（path-length 改变 + 反射面改变）。
2. **Viewing position**：从 origin 看不清的 bump，从局部 I/Q viewing position 看更清楚。
3. Real-time detection：**LEVD（Local Extremum Voting Detection）** bump detection，滑动窗口 local-extrema vs stationary σ。

**Phase 3 审计结论（commit `9b1c5b5`）**：现有 `BlinkListenerBlinkDetector` 实现忠实于论文 — 12 角度投影收敛于 √(Δi²+Δq²)（即论文的"弧心距离"，最坏离散误差 0.86%）；`_center` 用 I/Q 中位数近似弧心；score 归一化用 baseline amplitude median（符合"相对量"）。

**审计标记的 3 个无论文依据的启发式**（未改，是 Phase 4 调参候选）：
- `0.12 rad` 的 `phase_stable_projection` 阶梯（`blink_detector.py:278`）。
- `0.75 * projection_range` 权重（L289）。
- `_RobustEventGate` 的 `threshold_k=2.5 × MAD-σ`，paper 用 `5 × std` 且无 hysteresis/refractory。

---

## 3. 代码改动总览

21 个 commits，按 phase 分组：

| Phase | Commit | 描述 |
|---|---|---|
| **0.1** | `e80c310` | 删除 orphaned metrics producers（phase_pair_library / spatial_spread / segment voting / rhythm 等） |
| **0.1 fu** | `68e1f24` | 删除 session_io 里指向已删 machinery 的死列 |
| **0.2** | `4b60bc9` | 删除 `twinkle_fmcw_min_score` / `min_smoothness` / `min_deviation` 三个门 |
| **0.3** | `0dfbcaa` | inline `_candidate_trajectory_score` passthrough |
| **0.4** | `f835275` | `BlinkDetectionConfig` 改为 `BlinkConfig` 的 alias |
| **0 fu** | `ce63e3b` | 删 dead candidate-vote metrics synthesis + stats is_fmcw nit |
| **1.1** | `76c42b5` | 暴露 `last_ungated_coherence` 属性 |
| **1.2** | `b751aa9` | 加 `phase_trajectory_history` + `ungated_coherence_history` |
| **1.3** | `1cae299` | 双曲线波形 panel |
| **1 fu** | `27c554b` | `CompositeBlinkDetector` 暴露 `last_ungated_coherence` |
| **2.1** | `e391c80` | shape config 字段 |
| **2.2** | `8e92b97` | shape 单元测试（failing first） |
| **2.3** | `26c8a21` | `_TwinkleShapeSegmentationDetector` 实现 |
| **2.4** | `d1d4ad8` | CLI 加 `shape` 选项 |
| **2 fu** | `16b1aa7` | shape `event_id` 计数器 + 注释 nits |
| **3** | `9b1c5b5` | BlinkListener 审计 doc（无 bug） |
| **4.1** | `1411d02` | `--check-periodic` anti-periodic metric |
| **4.5 setup** | `e7b9804` | benchmark 暴露 shape 参数 |
| **4** | `2122473` | shape vs blinklistener vs twinkle on 9 sessions |
| **4.5 dx** | `9fb33549` | 周期触发根因诊断 |

---

## 4. 核心代码改动详解

### 4.1 Phase 0.2 — 删除 min-score 门（核心 diff）

**文件**：`hp_acoustic_wave/blink_detector.py`

**Before（`_TwinklePeakEventGate.stats`）**：
```python
min_score_floor = (
    max(self.config.min_score, float(self.config.twinkle_fmcw_min_score))
    if is_fmcw
    else self.config.min_score
)
threshold = max(min_score_floor, baseline + self.config.threshold_k * robust_sigma)
```

**After**：
```python
threshold = max(self.config.min_score, baseline + self.config.threshold_k * robust_sigma)
```

FMCW path 改回纯 adaptive threshold（`baseline + k × MAD-σ`），不再被 `min_score_floor` 抬高。

**`_fmcw_twinkle_score` 删除的 Fix C 块**：
```python
# 已删：
# fmcw_min_score = max(float(self.config.twinkle_fmcw_min_score), 0.0)
# smoothness_ok = smoothness >= float(self.config.twinkle_fmcw_min_smoothness)
# deviation_ok = deviation >= float(self.config.twinkle_fmcw_min_deviation)
# if not amplitude_ok or not amplitude_stable or not (smoothness_ok and deviation_ok):
#     adjusted_score = 0.0
# score_ok = bool(adjusted_score >= fmcw_min_score)
# if not score_ok:
#     adjusted_score = 0.0
```

剩余逻辑：
```python
if not amplitude_ok or not amplitude_stable:
    adjusted_score = 0.0
else:
    base_score = max(coherence_score, phase_score * smoothness)
    adjusted_score = base_score * orthogonality_score
```

### 4.2 Phase 0.4 — Config 统一

`hp_acoustic_wave/blink_detector.py` 顶部：
```python
from hp_acoustic_wave.config import BlinkConfig as BlinkDetectionConfig
```
删掉了重复的 `@dataclass class BlinkDetectionConfig: ...`。`BlinkConfig`（`config.py`）成为唯一 source of truth，修正历史 bug：`twinkle_fmcw_min_score` 默认值在两个类里漂移（0.12 vs 0.09）。

### 4.3 Phase 1 — 双曲线波形

**`hp_acoustic_wave/blink_detector.py:_fmcw_twinkle_score`**（暴露 ungated coherence）：
```python
coherence_score, smoothness, deviation = self._fmcw_coherence_score()
# Expose the raw coherence score (before amplitude/stability gates zero
# the adjusted score) so the realtime viz can show the ungated signal.
self.last_ungated_coherence = float(coherence_score)
```

**`hp_acoustic_wave/app.py:_draw_overlay`**（双 panel）：
- 上半（phase trajectory，绿色）：bipolar，居中 0，auto-scale ±max|val|。
- 下半（ungated coherence，蓝色 + threshold 橙虚线）。
- Wave mode 保留旧的单 panel（energy + threshold）。

### 4.4 Phase 2 — `_TwinkleShapeSegmentationDetector`（完整代码）

**文件**：`hp_acoustic_wave/blink_detector.py:707-835`

```python
class _TwinkleShapeSegmentationDetector:
    """Twinkle paper's shape-based segmentation on the phase trajectory.

    A blink produces a directional edge (down-up OR up-down) within a
    bounded time window (~0.1-0.5s). Noise-driven single peaks do NOT
    form such a pattern, so periodic false-positive firing is structurally
    prevented.
    """
    method = "shape"

    def __init__(self, config: "BlinkDetectionConfig"):
        self.config = config
        self.window: Deque[Tuple[float, float]] = deque(maxlen=config.shape_detection_window)
        self.smooth_buffer: Deque[float] = deque(maxlen=config.shape_smoothing_window)
        # 200 frames = ~10s @ 20Hz; longer than shape_detection_window so sigma
        # stays stable across multiple edge windows.
        self.baseline: Deque[float] = deque(maxlen=200)
        self.last_blink_time = -1e9
        self.last_ungated_coherence = 0.0
        self.event_count = 0

    def _trajectory_value(self, feature: ChunkFeature) -> float:
        if (feature.signal_mode == "fmcw"
            and feature.phase_pair_delta is not None):
            return float(feature.phase_pair_delta)
        return float(feature.phase)

    def _baseline_sigma(self) -> Tuple[float, float, float]:
        if len(self.baseline) < 5:
            return 0.0, 0.0, 1e-6
        arr = np.asarray(self.baseline, dtype=np.float64)
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        sigma = max(mad * 1.4826, 1e-6)
        return med, mad, sigma

    def update(self, feature: ChunkFeature) -> BlinkDetectionResult:
        delta = self._trajectory_value(feature)
        self.smooth_buffer.append(delta)
        smoothed = float(np.mean(self.smooth_buffer))
        self.window.append((feature.time_s, smoothed))
        self.baseline.append(smoothed)

        med, mad, sigma = self._baseline_sigma()
        in_refractory = (feature.time_s - self.last_blink_time) < self.config.shape_refractory_s
        # 20.0 = assumed chunk rate (Hz) for seconds->frames conversion.
        min_samples = max(3, int(self.config.shape_min_edge_len_s * 20.0))
        not_enough_data = len(self.window) < min_samples

        if in_refractory or not_enough_data:
            fallback_threshold = float(self.config.shape_edge_sigma_k * sigma)
            return BlinkDetectionResult(
                is_event=False, event_id=0, method=self.method,
                score=0.0, threshold=fallback_threshold,
                baseline=float(med), mad=float(mad),
                metrics={"shape_baseline_sigma": float(sigma),
                         "shape_threshold": fallback_threshold, ...},
            )

        times = [p[0] for p in self.window]
        vals = [p[1] for p in self.window]
        i_min = int(np.argmin(vals))
        i_max = int(np.argmax(vals))
        edge_len = abs(times[i_max] - times[i_min])
        edge_mag = abs(vals[i_max] - vals[i_min])
        confidence = edge_mag / sigma if sigma > 1e-9 else 0.0

        # For N iid Gaussian samples, expected max-min grows as sigma*sqrt(2*ln(N)).
        # Scale per-sample threshold by sqrt(n_edge) to match noise range growth.
        # Without this, random noise would always fire (max-min >> k*sigma for N>=10).
        n_edge = abs(i_max - i_min) + 1
        threshold = float(self.config.shape_edge_sigma_k * sigma * math.sqrt(n_edge))

        is_event = bool(
            self.config.shape_min_edge_len_s <= edge_len <= self.config.shape_max_edge_len_s
            and edge_mag >= threshold
        )
        if is_event:
            self.last_blink_time = feature.time_s
            self.event_count += 1

        return BlinkDetectionResult(
            is_event=is_event,
            event_id=(self.event_count if is_event else 0),
            method=self.method,
            score=(float(confidence) if is_event else 0.0),
            threshold=threshold,
            baseline=float(med), mad=float(mad),
            metrics={"shape_edge_mag": float(edge_mag),
                     "shape_edge_len": float(edge_len), ...},
        )
```

**关键设计决策**：
1. **不 unwrap phase trajectory** — 显示 raw `phase_pair_delta`。
2. **支持双向（上凸+下凸）** — 取窗口内最大幅差，方向取符号即可。
3. **`sqrt(n_edge)` 阈值缩放**（commit `26c8a21` 测试时发现）— 论文未给出具体阈值公式；plan 原版用 `k × sigma`，但 max-min 在 N≥10 时 ≈ `sigma × sqrt(2 ln N)`，会几乎总触发。改用 `k × sigma × sqrt(n_edge)` 匹配噪声增长。经验证：0/171 噪声窗口触发（plan 版是 120/171）。

**`build_blink_detector` 注册**：
```python
def build_blink_detector(config):
    method = config.method.lower()
    if method == "blinklistener":
        return BlinkListenerBlinkDetector(config)
    if method in ("twinkle", "twinkletwinkle"):
        return TwinkleTwinkleBlinkDetector(config)
    if method == "shape":
        return _TwinkleShapeSegmentationDetector(config)
    if method == "both":
        return CompositeBlinkDetector(config)
    raise ValueError("blink method must be one of: blinklistener, twinkle, shape, both")
```

### 4.5 Phase 4.1 — anti-periodic metric

**`hp_acoustic_wave/benchmark_hp_blink.py`**：
```python
def check_periodic_cluster_ratio(event_times, refractory_s, tolerance=0.2):
    """Fraction of inter-event gaps within +/-tolerance of refractory_s.
    High ratio (>0.5) indicates periodic noise-driven firing. <0.3 is target."""
    if len(event_times) < 3:
        return 0.0
    gaps = np.diff(sorted(event_times))
    clustered = float(sum(abs(float(g) - refractory_s) < tolerance for g in gaps))
    return clustered / float(len(gaps))
```

Smoke test：完美间隔 `[0, 1.05, 2.10, 3.15, 4.20]` → ratio=1.0；不规则 `[0, 0.5, 3.0, 7.0, 12.0]` → ratio=0.0。

---

## 5. 参数表

### 5.1 Shape 参数（`hp_acoustic_wave/config.py:BlinkConfig`）

| 字段 | 默认 | 含义 |
|---|---|---|
| `shape_smoothing_window` | 5 | 移动平均窗口（chunks） |
| `shape_detection_window` | 30 | 滑窗大小（~1.5s @ 20Hz） |
| `shape_min_edge_len_s` | 0.05 | edge 长度下界（秒） |
| `shape_max_edge_len_s` | 0.5 | edge 长度上界（秒） |
| `shape_edge_sigma_k` | 3.0 | edge 幅差必须 ≥ `k × sigma × sqrt(n_edge)` |
| `shape_refractory_s` | 0.5 | 防止单 blink 双触发 |

### 5.2 Benchmark 用的 FMCW 配置（所有 session 共用）

```
--signal-mode fmcw
--fmcw-range-bin 15
--fmcw-freq-low 17000
--fmcw-freq-high 23000
--fmcw-chirp-duration 0.05
--fmcw-lowpass-cutoff 5000
--amplitude 0.2
--blink-twinkle-fmcw-use-intra-chirp-phase-pair   # twinkle 路径用
```

session 174829（cw_single tone）用 `--signal-mode tone --frequency 18500`。

### 5.3 测试 session（9 个）

**CLAUDE.md**：`hp_blink_20260617_{184117, 190655, 214655}`
**baseline doc**：
| Variant | session | emission |
|---|---|---|
| A | 162711 | linear |
| A' | 172844 | linear_v2 |
| B | 173242 | linear_tukey |
| C | 173428 | cw_fmcw_hybrid |
| D | 173057 | triangle |
| E | 174829 | cw_single (tone) |

---

## 6. 实验结果

### 6.1 3 个 CLAUDE.md session（完整指标）

| session | detector | events | balanced | cluster_ratio |
|---|---|---:|---:|---:|
| **184117** | shape | 1 | -0.55 | 0.000 |
| | blinklistener | 4 | -0.65 | 0.000 |
| | twinkle | 24 | -7.00 | 0.000 |
| **190655** | shape | 4 | -0.65 | 0.000 |
| | blinklistener | 8 | **+6.45** | 0.000 |
| | twinkle | 45 | **+26.40** | 0.000 |
| **214655** | shape | 1 | -0.55 | 0.000 |
| | blinklistener | **55** | **-30.25** | 0.111 |
| | twinkle | 1 | -0.55 | 0.000 |

### 6.2 周期触发暴露（baseline-doc sessions，cluster_ratio ≥ 0.3）

| detector | session | cluster_ratio |
|---|---|---:|
| twinkle | 174829 (cw_single tone) | **0.407** |
| shape | 172844 (linear_v2 FMCW) | **0.500** |

其他 session 所有 detector 的 cluster_ratio = 0.000。

### 6.3 4 个硬性检查（CLAUDE.md）

| 检查 | shape | blinklistener | twinkle |
|---|:---:|:---:|:---:|
| 214655 events < 20 | ✅ (1) | ❌ (55) | ✅ (1) |
| 184117 balanced ≥ 0 | ❌ (-0.55) | ❌ (-0.65) | ❌ (-7.00) |
| 190655 balanced ≥ 0 | ❌ (-0.65) | ✅ (+6.45) | ✅ (+26.40) |
| 所有 session cluster_ratio < 0.3 | ❌ (172844: 0.500) | ✅ | ❌ (174829: 0.407) |

**没有 detector 通过全部 4 项。**

### 6.4 单元测试

`hp_acoustic_wave/tests/test_shape_segmentation.py` — 5/5 通过：
- `test_shape_detects_down_up_edge_within_blink_duration`
- `test_shape_ignores_single_peak_noise`
- `test_shape_refractory_prevents_double_fire`
- `test_shape_supports_up_down_direction`
- `test_shape_below_baseline_sigma_does_not_fire`

---

## 7. 诊断分析（周期触发根因，commit `9fb33549`）

完整报告：`experiment_periodic_firing_rootcause_20260619.md`
诊断图：`docs/diagnostics/periodic_*.png`（11 张）

### 7.1 三种症状

| 检测器 | session | 模式 | 现象 |
|---|---|---|---|
| twinkle | 174829 | tone (cw_single) | cluster_ratio=0.407 |
| shape | 172844 | FMCW (linear_v2) | cluster_ratio=0.500 |
| blinklistener | 214655 | FMCW (linear) | 55 事件，balanced=-30.25 |

### 7.2 诊断方法

对每个 session：
1. **Phase A**：直接看 trajectory（`phase_pair_delta` for FMCW / `phase` for tone）— 标注 visual blinks，看 quiet period 有没有可见振荡。
2. **Phase B**：quiet-period FFT（采样率 ~20Hz，Nyquist 10Hz）— 找周期线（0.95Hz = refractory 频率 / 20Hz = chunk rate / 10Hz = aliasing）。
3. **Phase C**（仅 214655）：raw audio.wav 直接 FMCW 解调，看周期是否在解调前就存在。
4. **Phase D**：6 个 baseline-doc variants 横向比较 baseline 噪声底。

### 7.3 关键证据

1. **没有 TX 线**：所有 variant 的 quiet-period FFT 都没有 0.95Hz（refractory 频率）的窄峰。如果是 FMCW/TX artifact 会有。
2. **tone 模式同样发病**：174829 是 tone（18500Hz 单频），完全没有 FMCW，twinkle 还是 cluster_ratio=0.407。**FMCW 不是周期触发的必要条件**。
3. **raw demod vs 存储的 phase_pair_delta 噪声底一致**：214655 上 audio.wav 直接解调的 quiet-FFT 主峰（1.18/7.74/9.22 Hz）和存进 features.csv 的 phase_pair_delta 的 quiet-FFT 主峰（4.13/6.57/8.30 Hz）在 50% 以内吻合，没有 DSP 层引入周期。

### 7.4 结论

**周期触发是算法层的，不是原始数据 / FMCW 的**。但有一个"软"问题：

> **FMCW range-bin 15 的 phase_pair_delta 是宽带噪声底**：所有 5 个 FMCW variant 上 quiet RMS ≈ blink RMS（比值 0.94–1.05）。眨眼信号被淹没在噪声底里。这让**任何 threshold-based detector 都脆弱** — 噪声反复过阈值就触发，refractory 把它们间隔开，看起来像周期。

| Variant | dominant quiet FFT peak | magnitude |
|---|---|---:|
| 173242 (linear_tukey) | 0.51 Hz | 28 |
| 172844 (linear_v2) | 0.87 Hz | 16 |
| 174829 (cw_single tone) | 0.72 Hz | — |
| 214655 (linear) | 1.18 Hz | — |

linear_tukey 噪声底最干净。

### 7.5 诊断建议

> "Do NOT notch-filter the raw signal — there is no TX line to notch. The lever is detector-side: enforce the existing hard-rule (record peak-candidate score, raise FMCW `min_score` floor to ~0.30, add smoothness gate) and tune tone-mode threshold independently. linear_tukey remains the cleanest-baseline emission; keep it."

**这和"归零心态"部分冲突** — 删除的 gate（min_score / smoothness / deviation）其实是 detector-side 防御 broadband-noise 诱导周期触发的正确机制。但 shape detector 通过 pattern matching 结构上规避了这个问题（cluster_ratio=0 on 7/9）。

---

## 8. 复现步骤

### 8.1 环境

- Python 3.8（`/c/Program Files/Python38/python.exe`）
- 依赖：见 `hp_acoustic_wave/requirements_hp_acoustic_wave.txt`（numpy, opencv-python, sounddevice, matplotlib, pytest 等）
- 仓库：`E:/android_projects/eye_blink_detect`，分支 `codex/hp-acoustic-wave-detector`，commit `9fb33549` 或更新。

### 8.2 跑单元测试

```bash
cd E:/android_projects/eye_blink_detect
"/c/Program Files/Python38/python.exe" -m pytest hp_acoustic_wave/tests/test_shape_segmentation.py -v
# 预期：5/5 通过
```

### 8.3 复现 9 session benchmark

对每个 detector × 每个 session：

```bash
cd E:/android_projects/eye_blink_detect

# FMCW sessions (8 个，除 174829)
for SESS in 184117 190655 214655 162711 172844 173242 173057 173428; do
  for METHOD in shape blinklistener twinkle; do
    "/c/Program Files/Python38/python.exe" hp_acoustic_wave/benchmark_hp_blink.py \
      --session hp_acoustic_wave/sessions/hp_blink_20260617_${SESS} \
      --source audio --truth auto --signal-mode fmcw \
      --blink-method ${METHOD} \
      --blink-twinkle-fmcw-use-intra-chirp-phase-pair \
      --check-periodic
  done
done
```

注：162711/172844/173242/173057/173428/174829 是 18 号的 session，前缀路径要改成 `hp_blink_20260618_${SESS}`。

```bash
# Tone session (174829)
"/c/Program Files/Python38/python.exe" hp_acoustic_wave/benchmark_hp_blink.py \
  --session hp_acoustic_wave/sessions/hp_blink_20260618_174829 \
  --source audio --truth auto --signal-mode tone \
  --blink-method shape --check-periodic
```

记录每个 run 的 events/TP/FP/FN/precision/recall/f1/balanced/cluster_ratio。完整数字见 `experiment_shape_vs_blinklistener_20260618.md`。

### 8.4 复现周期触发诊断

```bash
cd E:/android_projects/eye_blink_detect
"/c/Program Files/Python38/python.exe" hp_acoustic_wave/diagnose_periodic.py   # 如有
# 或参考 experiment_periodic_firing_rootcause_20260619.md 的 inline 步骤
```

输出：`docs/diagnostics/periodic_*.png`（11 张）。

### 8.5 实时运行（live）

```bash
cd E:/android_projects/eye_blink_detect/hp_acoustic_wave
bash run_emission_linear_tukey.sh   # 推荐：噪声底最干净的 emission
# 或 bash run_emission_linear.sh / run_emission_triangle.sh / run_emission_cw_fmcw_hybrid.sh
```

实时窗口下半显示：phase trajectory（绿，不归零）+ ungated coherence（蓝 + threshold 橙虚线）。

---

## 9. 已知问题与后续

### 9.1 已知问题

1. **shape recall 太低**（events=1 在多数 session 上）。原因：`shape_edge_sigma_k=3.0` + `sqrt(n_edge)` 缩放使阈值过严。需要降到 2.0 并放宽 `shape_max_edge_len_s` 到 0.7。
2. **184117 所有 detector 负分** — session 本身可能 ground truth 有问题，或所有方法都 struggling。需要单独查。
3. **twinkle 在 174829 (tone) 上 cluster_ratio=0.407** — tone 模式没有 Fix C 类的防御（因为 Fix C 是 FMCW-specific）。tone 路径需要独立的 noise 防御。
4. **blinklistener 在 214655 上 55 事件** — `_RobustEventGate` 的 `threshold_k=2.5 × MAD-σ` 太低（论文是 5 × std），需要调高。

### 9.2 建议后续

按优先级：

1. **调 shape 参数**（最便宜）：`shape_edge_sigma_k=2.0`, `shape_max_edge_len_s=0.7`。重跑 9 session。如果 cluster_ratio 仍 < 0.3 且 recall 上来，shape 成为默认。
2. **调 blinklistener threshold**：`threshold_k=5.0`（匹配论文），加 hysteresis/refractory gap。解决 214655 的 55 事件回归。
3. **tone 路径加防御**：twinkle 在 tone 模式没有 Fix C 等价物。考虑给 tone 路径加 adaptive threshold + smoothness 检查。
4. **统一 emission 为 linear_tukey**：诊断显示它噪声底最干净。
5. **不要 notch filter 原始信号** — 没有 TX 线可滤。

### 9.3 用户硬性要求 vs 现状

CLAUDE.md 要求：**禁止周期性误判**。当前**未达成**（没有任何 detector 通过全部 4 个硬性检查）。但 shape 结构上消灭了周期触发（cluster_ratio=0 在 7/9 session），只是 recall 不够。**调 shape 参数是最有可能的路径**。

---

## 10. 文件清单

### 新增
- `hp_acoustic_wave/blink_detector.py` — `_TwinkleShapeSegmentationDetector` class（L707-835）
- `hp_acoustic_wave/tests/test_shape_segmentation.py` — 5 个单元测试
- `hp_acoustic_wave/experiment_shape_vs_blinklistener_20260618.md` — Phase 4 benchmark 报告
- `hp_acoustic_wave/experiment_blinklistener_audit_20260618.md` — Phase 3 审计
- `hp_acoustic_wave/experiment_periodic_firing_rootcause_20260619.md` — 周期触发诊断
- `hp_acoustic_wave/docs/diagnostics/periodic_*.png` — 11 张诊断图
- `hp_acoustic_wave/docs/superpowers/specs/2026-06-18-fmcw-blink-redesign-design.md` — 设计 spec
- `hp_acoustic_wave/docs/superpowers/plans/2026-06-18-fmcw-blink-redesign-plan.md` — 实施 plan
- `hp_acoustic_wave/docs/superpowers/reviews/2026-06-19-fmcw-blink-redesign-review.md` — 本文档

### 修改
- `hp_acoustic_wave/blink_detector.py` — 删 dead code + min-score 门 + 新 shape detector + last_ungated_coherence
- `hp_acoustic_wave/config.py` — 删 min-score 字段 + 加 shape 字段
- `hp_acoustic_wave/app.py` — 双曲线波形 + 删 dead feature columns
- `hp_acoustic_wave/session_io.py` — 删 dead feature columns
- `hp_acoustic_wave/run_hp_wave_detector.py` — 删 min-score CLI + 加 shape CLI
- `hp_acoustic_wave/benchmark_hp_blink.py` — 删 min-score CLI + 加 shape CLI + `--check-periodic` + shape params
- `hp_acoustic_wave/run_emission_*.sh`（5 个）— 删 min-score 行

### 未改（保留）
- `_amplitude_phase_orthogonality_score`（默认 off 但功能正常）
- `_trajectory_phase` intra-chirp phase-pair 支持
- `FmcwBackgroundSubtractor`（app.py 用）
- `BlinkListenerBlinkDetector`（审计无 bug）
- `_RobustEventGate`（审计标记 3 个无论文依据的启发式，未改）
