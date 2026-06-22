# FMCW 呼吸检测脚本使用说明

这个目录里的作业材料来自 `CIS3990 Lab 2.html`。现在已经把 notebook 里的核心流程整理成两个独立 Python 脚本：

- `fmcw_breathing.py`：离线处理 `.npz` 数据，生成呼吸图。
- `fmcw_breathing_live.py`：在线播放 FMCW chirp、麦克风采集、保存录音并生成呼吸图。

建议在 Windows PowerShell 里从项目目录运行：

```powershell
cd E:\android_projects\eye_blink_detect\hp_acoustic_wave
```

在当前 Mac 上，从项目目录运行：

```bash
cd /Users/a0/codex-work/hp_acoustic_wave
.venv/bin/python fmcw_breathing_live.py --list-devices
```

这台 Mac 当前默认设备是：

```text
input : [0] MacBook Pro麦克风
output: [1] MacBook Pro扬声器
```

所以通常不需要指定 `--input-device` / `--output-device`。如果外接了耳机或声卡，再根据 `--list-devices` 的输出手动指定。

## 1. 先跑离线数据

这里已经有两组预录数据：

- `refer\FMCW\breathing_1_rangebin=18.npz`
- `refer\FMCW\breathing_2_rangebin=21.npz`

运行：

```powershell
python fmcw_breathing.py --output-dir refer\FMCW\outputs
```

脚本会自动读取两个 `.npz`，并根据文件名使用 `range_bin=18` 和 `range_bin=21`。

输出在：

```text
refer\FMCW\outputs\
```

重点看这些图：

```text
breathing_1_rangebin_18_breath_rangebin18.png
breathing_2_rangebin_21_breath_rangebin21.png
```

这两张就是作业里的 `Breath Monitoring` 曲线。

## 2. 在线采集前先列设备

运行：

```powershell
python fmcw_breathing_live.py --list-devices
```

输出中：

- `>` 表示默认输入设备，通常是麦克风。
- `<` 表示默认输出设备，通常是耳机或扬声器。
- `(2 in, 0 out)` 是输入设备。
- `(0 in, 2 out)` 或 `(0 in, 8 out)` 是输出设备。

在当前这台 HP 上，我看到过类似设备：

```text
>  1 外部麦克风 ...
<  4 耳机 ...
   5 扬声器 ...
```

如果要用笔记本扬声器发声，通常指定 `--output-device 5`。如果你用外接麦克风，通常指定 `--input-device 1`。

## 3. 在线录制并生成图

推荐先从 10 秒录制开始：

```powershell
python fmcw_breathing_live.py --duration 10 --range-bin 15 --amplitude 0.2 --input-device 1 --output-device 5
```

操作姿势：

1. 让扬声器尽量正对胸口。
2. 距离大约 40 cm。
3. 运行命令后，保持电脑和身体尽量稳定。
4. 做一次或两次明显的吸气/呼气，再屏住呼吸。

运行结束后会保存：

```text
refer\FMCW\live\live_YYYYMMDD_HHMMSS.npz
refer\FMCW\outputs\live_YYYYMMDD_HHMMSS_rangebin_15_breath_rangebin15.png
refer\FMCW\outputs\live_YYYYMMDD_HHMMSS_rangebin_15_breath_rangebin15_clean.png
refer\FMCW\outputs\live_YYYYMMDD_HHMMSS_rangebin_15_range_spectrum.png
refer\FMCW\outputs\live_YYYYMMDD_HHMMSS_rangebin_15_rangebin_search.png
refer\FMCW\outputs\live_YYYYMMDD_HHMMSS_rangebin_15_summary.json
```

如果不想弹出图像窗口，只保存图片：

```powershell
python fmcw_breathing_live.py --duration 10 --range-bin 15 --amplitude 0.2 --input-device 1 --output-device 5 --no-show
```

Mac 默认设备采集可以直接用：

```bash
.venv/bin/python fmcw_breathing_live.py --duration 10 --range-bin 15 --amplitude 0.2 --no-show
```

当前脚本也支持眨眼实验里的 FMCW 发射变体：

```bash
.venv/bin/python fmcw_breathing_live.py --duration 12 --range-bin 16 --drop-seconds 1 --invert-phase --fmcw-emission linear_tukey --amplitude 0.2 --no-show
```

可选值：

```text
linear
linear_tukey
cw_fmcw_hybrid
triangle
```

如果要连续采四种变体，用：

```bash
./run_breathing_emission_variants.sh
```

可以用环境变量调整参数：

```bash
DURATION=12 RANGE_BIN=16 AMPLITUDE=0.2 DROP_SECONDS=1 ./run_breathing_emission_variants.sh
```

如果想录制时就实时看 `range_bin` 的相位曲线，加 `--live-plot`：

```powershell
python fmcw_breathing_live.py --duration 10 --range-bin 15 --amplitude 0.2 --input-device 1 --output-device 3 --live-plot
```

实时窗口里浅色线是当前已采集到的原始相位，深色线是平滑后的主趋势。这个实时显示只支持默认 callback stream，不和 `--playrec` 一起使用。

## 4. range_bin 怎么选

作业提示：胸口距离大约 40 cm 时，`range_bin` 通常在 `15` 附近。

如果 `range_bin=15` 的呼吸图不明显，可以试附近几个值：

```powershell
python fmcw_breathing_live.py --duration 10 --range-bin 12 --amplitude 0.2 --input-device 1 --output-device 5
python fmcw_breathing_live.py --duration 10 --range-bin 18 --amplitude 0.2 --input-device 1 --output-device 5
python fmcw_breathing_live.py --duration 10 --range-bin 21 --amplitude 0.2 --input-device 1 --output-device 5
```

也可以看 `rangebin_search.png`。这张图会把目标 bin 附近的多个相位曲线画在一起，哪条曲线最像平滑的吸气/呼气变化，就用对应的 `range_bin`。

## 5. amplitude 怎么调

默认建议：

```text
--amplitude 0.2
```

如果麦克风收到的信号太弱，可以逐步增大：

```powershell
--amplitude 0.3
--amplitude 0.4
```

不要一开始就开太大。FMCW 是 17 kHz 到 23 kHz 的 chirp，有些人仍然能听到高频部分，音量过大也可能让麦克风削顶。

## 6. 输出图怎么看

### Breath Monitoring

文件名类似：

```text
*_breath_rangebin15.png
```

这张图的纵轴是指定 range bin 的 unwrapped phase。呼吸会让胸口产生毫米级位移，这会体现在相位缓慢上升或下降。

如果后半段屏息，曲线应该趋于平缓。

如果是在线录制的 live 数据，原始相位可能会比作业预录数据更抖。脚本会额外输出一张：

```text
*_breath_rangebin15_clean.png
```

这张图会保留浅色原始相位，同时叠加一条边缘友好的平滑主曲线，更适合快速判断“先吸气、后呼气”这类整体趋势；是否真的选中了合适的目标距离，仍然要结合 `rangebin_search.png` 和 `range_spectrum.png` 看。

### Range Spectrum

文件名类似：

```text
*_range_spectrum.png
```

这张图显示背景减除后的 range-bin 能量随时间变化。白线是当前选中的 `range_bin` 对应距离。

### Summary JSON

文件名类似：

```text
*_summary.json
```

里面会有：

```json
{
  "range_bin": 15,
  "range_bin_distance_m": 0.42875,
  "breathing_present": true,
  "phase_peak_to_peak_rad": 6.1
}
```

`breathing_present` 是脚本给的粗略判断。作业截图主要看 `Breath Monitoring` 图，不要把 JSON 里的活动区间当成精确 inhale/exhale 分段。

## 7. 常见问题

### 设备列表乱码或报编码错

一般直接运行本脚本已经做了编码兜底。如果你自己写临时代码打印设备，PowerShell 里可以先设置：

```powershell
$env:PYTHONIOENCODING='utf-8'
```

### 没有明显呼吸曲线

按顺序检查：

1. 输出设备是不是扬声器，而不是耳机。
2. 输入设备是不是正在使用的麦克风。
3. 电脑扬声器是否正对胸口。
4. 距离是否大约 40 cm。
5. 试 `range_bin=10` 到 `25`。
6. 试把 `--amplitude` 从 `0.2` 提到 `0.3` 或 `0.4`。

### 声音从耳机出来了

先看设备列表：

```powershell
python fmcw_breathing_live.py --list-devices
```

然后把 `--output-device` 改成扬声器对应编号，例如：

```powershell
python fmcw_breathing_live.py --duration 10 --range-bin 15 --amplitude 0.2 --input-device 1 --output-device 5
```

### 想使用 notebook 原来的 playrec 风格

默认在线脚本使用 callback stream，更接近当前项目里的单频连续波实现。如果想用 notebook 的 `sd.playrec()` 方式，可以加：

```powershell
python fmcw_breathing_live.py --duration 10 --range-bin 15 --amplitude 0.2 --input-device 1 --output-device 5 --playrec
```

## 8. 验证脚本

从仓库父目录运行测试：

```powershell
cd E:\android_projects\eye_blink_detect
python -m pytest hp_acoustic_wave\tests -q
```

当前 FMCW 相关测试覆盖：

- 背景减除。
- FMCW range bin 到距离的公式。
- 预录 `.npz` 的呼吸相位提取。
- 在线录音设备参数和 callback 流程。
