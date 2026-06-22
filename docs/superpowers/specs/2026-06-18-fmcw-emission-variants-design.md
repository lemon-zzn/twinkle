# FMCW 发射变体对比实验 — 设计文档

日期：2026-06-18
状态：设计已批准，待实现

## 背景与动机

session `hp_blink_20260617_190655` 的 benchmark：`blink_hits=24/45`（之前一度 39/45，因回退 snap_020 后又退到 24）。session `hp_blink_20260618_150050`：`blink_hits=9/9, unexplained=16, events=26`。**召回 100%，但 FP 占 62%**。

用户指出当前评分有两个根本问题：
1. **balanced_score 不是 precision/recall**，FP 没有正确语义
2. **±0.8s 匹配窗 + 检测器每 0.5s 可能出事件**，导致每个 marker 几乎一定被覆盖，recall 结构性地偏向 100%

同时，FMCW 的发射端可能也有问题：
- 当前 `generate_fmcw_chirp` 是纯 `amplitude·cos(linear chirp)`，**无 taper**，chirp 边界不连续会造成 spectral leakage 污染 range bin
- `_tukey_window` helper 已存在（dsp.py:426）但只用在 CW/tone 路径，没用在 FMCW 发射上
- 用户希望对比多种发射策略，找出对眨眼（小幅度、近场 ~40cm）最敏感的一种

## 目标

1. 把 benchmark 评分从 `balanced_score` 升级为 **标准 F1**，匹配窗改为 after-only
2. 实现 **5 种发射变体**（含两条已知正确的对照基线）
3. 每变体录一份独立 session，用统一 F1 横评，选 F1 最高的作为新 baseline

## 非目标

- 不调检测器参数（`min_score`、phase-pair 开关、coherence 阈值等）—— 只看发射端影响
- 不动 `dsp.py` 的核心 DSP 链（dechirp → lowpass → range FFT → range bin → 相位/幅度）
- 不改 session 文件格式（只新增 meta.json 字段）

## Section 1 — 评分目标：标准 F1

### 公式

```
TP = #blink markers matched 1:1 by an event
FP = #unexplained events (no marker match)
FN = #blink markers without a match
precision = TP / (TP + FP)
recall    = TP / (TP + FN) = TP / #blink_markers
F1        = 2·P·R / (P+R),  0 if TP = 0
```

### 匹配规则

- **1:1 greedy nearest pairing**：每个 event 最多消费一个 marker，每个 marker 最多被一个 event 消费。配对按时间距离升序贪心分配。
- **`match_before_s = 0`**：event 必须出现在 marker 之后（用户确认：视觉眨眼判断一定快于声波，所以声波事件总是在视觉 marker 之后）
- **`match_after_s = 0.5`**：blink 动作 ~250ms + 检测器积分 ~0.5s 的上限

### 大动作处理

`large_motion_hits` **不**计入 FP。这些事件被检测器正确标记为「非眨眼」，既不是 TP 也不是 FP。只有真正未匹配任何 marker 的 `unexplained_events` 才算 FP。这与当前实现一致。

### 改动点

`benchmark.py:score_events`:
- 新增参数 `one_to_one: bool = True`（默认开启 1:1 配对）
- `BenchmarkSummary` 新增字段：`precision: float`, `recall: float`, `f1: float`, `tp: int`, `fp: int`, `fn: int`
- `balanced_score` 保留（向后兼容，traceability）
- `match_before_s` 默认改为 0.0，`match_after_s` 默认改为 0.5（向后兼容通过显式参数覆盖）

`benchmark_hp_blink.py` 输出加 `f1=...` 和 `precision=...` 字段。

## Section 2 — 五种发射变体

### 变体总表

| ID | 名称 | CLI 值 | 实现路径 |
|---|---|---|---|
| A | linear (基线) | `linear` | 现状 `generate_fmcw_chirp` |
| B | linear_tukey | `linear_tukey` | `generate_fmcw_chirp` + per-chirp Tukey α=0.2 envelope |
| C | cw_fmcw_hybrid | `cw_fmcw_hybrid` | 帧交替：偶数 chirp 帧 + 奇数 CW 帧 (freq_low) |
| D | triangle | `triangle` | 上扫+下扫拼接 (freq_low→freq_high→freq_low) |
| E | cw_single (对照) | `cw_single` | 现有 `extract_chunk_feature` 单频路径 |

### A — linear (基线，零成本)

```python
def generate_fmcw_chirp(..., emission: str = "linear") -> np.ndarray:
    if emission == "linear":
        return amplitude * np.cos(phase)  # 现状
```

### B — linear_tukey

```python
    if emission == "linear_tukey":
        # 每 samples_per_chirp 套一个 Tukey envelope
        n_chirps = ceil(num_samples / samples_per_chirp)
        envelope = np.tile(_tukey_window(samples_per_chirp, 0.2), n_chirps)[:num_samples]
        return (amplitude * envelope * np.cos(phase)).astype(np.float32)
```

### C — cw_fmcw_hybrid

```python
    if emission == "cw_fmcw_hybrid":
        # 偶数 chirp index → 线性 chirp；奇数 → 固定 freq_low 的 CW
        chirp_index = (start_sample + arange) // samples_per_chirp
        is_cw_frame = (chirp_index % 2 == 1)
        cw_phase = 2*pi * freq_low * t  # 不带 sweep
        phase = where(is_cw_frame, cw_phase, chirp_phase)
        return amplitude * cos(phase)
```

**DSP 端配合**：`extract_fmcw_chunk_feature` 需要按帧分路：
- chirp 帧：走原 dechirp → range FFT → range bin
- CW 帧：走 freq_low 复数 mixer → 直接取基带复数 → phase-pair 特征
- 整段 chunk 的 range bin 用 chirp 帧的均值，phase-pair 用 CW 帧的均值

### D — triangle

```python
    if emission == "triangle":
        # 上扫：[0, chirp_duration/2) freq_low → freq_high
        # 下扫：[chirp_duration/2, chirp_duration) freq_high → freq_low
        half = samples_per_chirp // 2
        t_up = t[:half], t_down = t[half:]
        phase_up = 2*pi*(freq_low*t_up + 0.5*slope*t_up**2)
        phase_dn = phase_up[-1] + 2*pi*(freq_high*t_down - 0.5*slope*t_down**2)
        ...
```

**DSP 端配合**：dechirp 用本地 tx 副本相乘（不假设线性），所以 beat 频率符号翻转会被自然吸收到 range FFT 的对称 bin 上。range bin 的幅度仍然有效，但相位在上下扫切换处有跳变 — phase-pair 路径需要 mask 掉切换帧。

### E — cw_single (对照基线，零成本)

用户确认实现一定对。走 `--signal-mode cw --tone-hz <freq>`，**不**通过 `--fmcw-emission` 进入（它不在 FMCW 路径里）。meta.json 仍记 `fmcw_emission: "cw_single"` 以便 benchmark 识别——写入由 runner 在 `--signal-mode cw` 时自动填入。

### 配置与 CLI

新增字段：
- `BlinkConfig.fmcw_emission: str = "linear"`（值域：linear / linear_tukey / cw_fmcw_hybrid / triangle）。`cw_single` 不在此枚举里——它由 `--signal-mode cw` 隐含，meta.json 中 fmcw_emission 字段在 cw 模式下被 runner 写成 `"cw_single"`
- `run_hp_wave_detector.py` 加 `--fmcw-emission`（默认 linear，仅 `--signal-mode fmcw` 时生效）
- `benchmark_hp_blink.py` 加 `--fmcw-emission`（仅在重算发射时用得上，replay audio 模式下从 meta.json 读）

## Section 3 — 录制与分析协议

### 录制流程

1. 跑 `run_hp_wave_detector.py --mode blink --blink-method twinkle --signal-mode fmcw --fmcw-emission <variant> ...`（其余参数同 `run_baseline.sh`）。变体 E 例外：用 `--signal-mode cw --tone-hz <freq>`，runner 自动把 meta.json 里 `fmcw_emission` 写成 `"cw_single"`
2. session 目录：`sessions/hp_blink_<timestamp>/`（命名不变，emission 写入 meta.json）
3. `meta.json` 在录制开始时立即落盘 `fmcw_emission` 字段，避免崩溃丢失
4. 每份录 20–30 个自然眨眼，期间穿插几次大幅度动作（点头/转身）用于 large_motion 排斥评估
5. session 落盘：`tx.wav` / `rx.wav` / `meta.json` / `events.csv` / `features.csv`（沿用现有格式）

### meta.json 新字段

```json
{
  "fmcw_emission": "linear_tukey",
  "fmcw_freq_low": 17000,
  "fmcw_freq_high": 23000,
  ...
}
```

### 分析流程

1. 每份 session 跑 `benchmark_hp_blink.py --session <path> --source audio --truth auto`
2. benchmark 从 meta.json 读 `fmcw_emission`，在 summary 里输出该字段
3. 汇总成表（5 变体 × {F1, precision, recall, FP, FN, events}）
4. 写 `experiment_fmcw_emission_comparison_<date>.md`
5. 选 F1 最高的变体作为新 baseline，更新 `run_baseline.sh`

### 关键不变量

- benchmark 对所有变体用同一套 F1 公式 + 0.5s after-only 匹配窗
- detector 配置保持 snap_020 状态，不调参
- DSP 核心链不动，只在外层加 emission 分支
- A 和 E 已实现且已验证，作为对照锚点

## 风险与权衡

| 风险 | 缓解 |
|---|---|
| C/D 改 dechirp 路径，可能与 snap_020 检测器期望的特征语义不匹配 | A/E 作为锚点；如果 C/D 端到端 F1 比 A 差，至少能定位是「发射端」问题而非「检测端」问题 |
| Tukey α=0.2 是经验值，不一定最优 | 本次不调 α；后续若 B 胜出，再扫 α |
| 1:1 配对会改变历史 session 的 benchmark 数字 | balanced_score 保留；F1 是新指标。旧 session 重跑 benchmark 时会同时得到两个数字 |
| 三角波切换帧的相位跳变 | DSP 端 mask 掉切换帧，phase-pair 路径跳过这些帧 |

## 文件清单

| 文件 | 操作 |
|---|---|
| `dsp.py` | `generate_fmcw_chirp` 加 `emission` 参数；新增 hybrid/triangle 发射序列；`extract_fmcw_chunk_feature` 加按帧分路（C）；`_tukey_window` 复用（B） |
| `blink_detector.py` | 不动（detector 对 emission 透明） |
| `config.py` | `BlinkConfig` 加 `fmcw_emission` 字段 |
| `run_hp_wave_detector.py` | 加 `--fmcw-emission`；meta.json 写入字段 |
| `benchmark.py` | `score_events` 加 `one_to_one` 参数；新增 F1/precision/recall；`match_before_s=0, match_after_s=0.5` 默认 |
| `benchmark_hp_blink.py` | 加 `--fmcw-emission`；从 meta.json 读 emission；summary 输出 F1 |
| `app.py` | 若 meta.json 是 app 写的，加 emission 字段写入 |

## 验证

- 对变体 A（已存在 sessions）：F1 数字应小于旧 balanced_score（因为更严），但应该明显反映 FP 问题
- 对变体 E：跑现有 `--signal-mode cw` session，确认 F1 公式工作
- B/C/D：用户录新数据后跑 benchmark，填对比表
