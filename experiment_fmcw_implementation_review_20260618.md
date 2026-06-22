# FMCW 实现正确性分析（对照参考材料）

日期：2026-06-18
目的：把 `hp_acoustic_wave/dsp.py` 的 FMCW 实现和三份参考资料对照，逐点核对哪些是正确的、哪些是偏差，为后续实验提供依据。

## 参考资料

| 来源 | 内容 | 关键点 |
|---|---|---|
| `refer/FMCW/CIS3990 Lab 2.html`（UPenn） | FMCW 手势 + 呼吸监测 lab | 标准流程、相位跟踪小位移、背景减除=全局均值 |
| `refer/FMCW/6.1820 - Lab 4.html`（MIT） | FMCW 手势识别 lab | 背景减除=**相邻 chirp 差分**、距离公式 |
| `refer/FMCW/chat-# PDF 文件总结：设备无关定位.txt` | Device-Free Localization 讲义 | 「差分消除家具反射，但需要用户移动」 |
| `refer/FMCW/AIradar_processing.py` | 雷达处理参考代码 | Hamming 窗、DC 抑制、bandpass 模拟 |

## 标准 FMCW 流程（两份 lab 一致）

1. 把收到的信号分段，每段对应一个 chirp（`samples_per_chirp`）
2. `mixed = rx * tx`（时域 dechirp）
3. 低通滤波
4. FFT → 找峰 → 得到 range bin
5. **背景减除**
   - CIS3990 §2.1：所有 chirp 取均值，再减去
   - 6.1820 §2.1：`chirp[i] - chirp[i-1]`（相邻差分）
6. 在 range bin 上跟踪幅度/相位
7. 距离公式：`distance = (Δf · v) / (2 · slope)`

## 逐点对照

### ✅ 与参考一致 / 正确的部分

| 项 | dsp.py | 参考 | 结论 |
|---|---|---|---|
| dechirp | `mixed = rx64 * tx64` (dsp.py:310) | 两 lab 同 | ✓ |
| range FFT | `np.fft.rfft(lowpassed)` (dsp.py:332) | 同 | ✓ |
| range bin 取值 | `complex_bins[range_bin]` (dsp.py:341) | 同 | ✓ |
| 距离公式 | `(delta_f · v / slope) / 2` (dsp.py:396-408) | CIS3990 §2.2 / 6.1820 §2.2 | ✓ 完全一致 |
| 相位跟踪 | `np.angle(complex_bins[range_bin])` (dsp.py:345) | CIS3990 §3.1 `np.angle()` | ✓ |
| 相位跟踪做小位移 | `phase_delta = unwrap(...)` (dsp.py:354) | CIS3990 §3 "use phase to track small movement" | ✓ |

**核心 DSP 主链是正确的**：dechirp → lowpass → range FFT → range bin → 相位/幅度 → 距离公式，和两份 lab 完全一致。

### ⚠️ 与参考存在偏差的部分

#### 1. 低通滤波实现方式 — 砖墙滤波（影响 phase-pair 路径）

- **dsp.py** (dsp.py:311-314)：
  ```python
  spectrum = np.fft.rfft(mixed)
  spectrum[freqs > lowpass_cutoff] = 0   # 直接置零
  lowpassed = np.fft.irfft(spectrum, n=usable)
  ```
  这是**理想砖墙滤波**，会在时域引入 sinc 振铃。
- **参考**：lab notebook 通常用 `scipy.signal` butterworth，或不显式加窗直接 FFT 找峰。
- **影响**：砖墙会污染 `complex_baseband` 的边缘样本，而 phase-pair 特征恰恰采样边缘点（0.15/0.85 ~ 0.40/0.60）。`_phase_pair_features` 取的就是这些位置，受振铃影响最大。

#### 2. 双重 FFT — 冗余

- **dsp.py**：`rfft(mixed)` → lowpass → `irfft` → `rfft(lowpassed)` 再取 range bin。
- 第一次 `spectrum` 已经是 mixed 的频谱。第二次 `rfft` 是对「频谱置零后逆变换再变换」的结果再做 FFT，**数值上等价于直接在第一次 spectrum 上取 bin**（因为 FFT/IFFT 可逆）。
- **问题**：冗余计算，且 irfft/rfft 来回会引入数值噪声。

#### 3. 背景减除策略不同（最关键的差异）

| 来源 | 方法 | 物理含义 | 时间常数 |
|---|---|---|---|
| CIS3990 §2.1 | `chirp - mean(all_chirps)` | 减去**静态**场景（家具反射） | 全程 |
| 6.1820 §2.1 | `chirp[i] - chirp[i-1]` | **差分**（只保留变化） | 1 帧 |
| DSP-Free Localization §4 | 差分消除家具反射 | **需要用户移动**才生效 | — |
| dsp.py | EMA α=0.02 | 介于两者之间 | ~2.5s（50 chirps） |

- **眨眼场景的矛盾**：眼皮动作是**瞬态**（~100-300ms），EMA α=0.02 的时间常数 2.5s 太慢，会把瞬态信号当成背景的一部分慢慢吸收。
- 这解释了 `experiment_preprocess_20260618.md` 中实验 C 的发现：**window=20（1s）能在 214655 上召回 1/6 blink**，而 EMA 完全召回不到。
- 6.1820 的相邻差分太激进（高通会放大呼吸/漂移噪声），CIS3990 的全局均值太慢（瞬态被吞）。

#### 4. tx 信号是实数 `cos`，但 phase-pair 路径用的是复指数

- `generate_fmcw_chirp` 返回 `cos(phase)`（实信号），所以 `mixed = rx * cos(phase)` 只混出**一个边带**，存在镜像分量。
- 但 phase-pair 路径 (dsp.py:323) 用：
  ```python
  complex_reference = np.exp(-1j * chirp_phase)
  complex_baseband = _lowpass_complex(rx64 * complex_reference, ...)
  ```
  这才是**正确的复 dechirp**，直接把 rx 乘复指数，没有镜像。
- **结论**：`complex_baseband`（phase-pair 路径）的相位信息**比** `complex_bins`（range-bin 路径，基于实数 mixed）更干净。
- 这也解释了为什么检测主信号走 phase-pair 路径，而 range-bin 幅度/相位的改进（实验 A/C）对结果影响很小。

#### 5. 没有 Hamming/Hann 窗（和 lab 一致，和 AIradar 不一致）

- `refer/FMCW/AIradar_processing.py:748` 用 `np.hamming(samples_per_chirp)` 在 range FFT 前。
- dsp.py 没有加窗（实验 A/D 都试过，已回退，均无净收益）。
- 两份 lab 的 jupyter notebook 没显式加窗，所以这点 dsp.py 和 lab 一致。

## 判断：实现是否正确？

**核心 DSP 是正确的**：dechirp → lowpass → range FFT → range bin → 相位/幅度 → 距离公式，这条主链和两份 lab 完全一致。

**phase-pair 路径本身是 dsp.py 的额外创新**，参考 lab 里没有（lab 只用 range-bin 的相位）。这条路径用复指数 dechirp 是更正确的做法，**保留它是对的**。

**但有 3 个值得改的偏差**，按影响排序：

| 优先级 | 偏差 | 预期影响 | 修改范围 |
|---|---|---|---|
| 高 | 背景减除时间常数（EMA α=0.02=2.5s 太慢） | 214655 召回为 0 的结构性原因 | `benchmark.py:183` 改 `window=` |
| 中 | 砖墙低通振铃污染 phase-pair 边缘样本 | phase-pair 路径 SNR | `dsp.py:311-314` 改 butterworth 或裁带 |
| 低 | 冗余的第二次 rfft | 数值噪声、计算浪费 | `dsp.py:332` 删掉 |

## 待做的实验（晚点做）

1. **背景减除**：滑窗均值 window=20~40（实验 C 已验证 window=20 能救 214655）
2. **低通滤波**：砖墙改 butterworth 或裁带（只保留 [−cutoff, cutoff] 区间，不置零）
3. **冗余 FFT**：直接在第一次 spectrum 上取 bin，删掉 irfft/rfft 来回

三个改动都**不影响 phase-pair 路径的核心判别力**（phase-pair 走独立的 `complex_baseband` 路径），只是让背景/幅度路径更干净，理论上不会伤 184117/190655。

## 参考：两份 lab 的差异速览

| 项 | CIS3990 Lab 2 (UPenn) | 6.1820 Lab 4 (MIT) |
|---|---|---|
| 应用 | 手势 + 呼吸监测 | 手势识别 |
| 背景减除 | `chirp - mean(all_chirps)` | `chirp[i] - chirp[i-1]` |
| 距离公式 | `(Δf/slope)·v/2` | `(Δf·v)/(2·slope)`（等价） |
| 相位用法 | §3.1 用相位跟踪呼吸 | 不用相位 |
| range_bin | 15（40cm 距离） | 取峰 |
