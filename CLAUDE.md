阅读E:\android_projects\eye_blink_detect\hp_acoustic_wave\README.md 
  E:\android_projects\eye_blink_detect\hp_acoustic_wave\experiment_twinkle_fmcw_blink_20260617.md 
  ，阅读论文E:\android_projects\eye_blink_detect\hp_acoustic_wave\MinerU_html_3596238_2055512726219456512.html 我们一起来优化现在的代码，上 
  个commit的代码是可以用的，但是最近被改崩了，希望还是用FMCW来实现眨眼检测，但是现在的FMCW实现本身也可能有问题。 启动命令是这个& 
  'C:\Program Files\Python38\python.exe' .\run_hp_wave_detector.py 
    --mode blink --blink-method twinkle --signal-mode fmcw 
    --fmcw-range-bin 15 --amplitude 0.2 --input-device 1 --output-device 3 
    --camera-width 640 --camera-height 480 只拿 
  拿184117和sessions\hp_blink_20260617_190655、sessions\hp_blink_20260617_214655（这一个数据基本上为空）来测试 
  ，不要拿其他的，不要又出现周期性猜测blink了 。 我们先头脑风暴吧

## 硬性要求（不可违反）

- **禁止周期性误判（"周期猜测 blink"）**：FMCW coherence score 不得在 refractory_s 节拍上反复触发事件。
  现象：事件时间间隔大都卡在 refractory_s（1.3s）附近，与真实眨眼无关，全是噪声驱动。
  - **Why**：session 214655/150050 出现 18 个事件里只有 6 个 TP，其余 12 个是噪声驱动的周期假阳性，threshold 还锁死在 min_score。
  - **How to apply**：
    1. 事件的 `score` 必须记录 peak（middle candidate）的 score，不能记当前帧的 score（peak 通常在事件触发前一帧）。
    2. FMCW 路径 gate threshold 的下限 = `twinkle_fmcw_min_score`（不能塌回 `min_score=0.006`）。
    3. 单纯 A+B 不足以消灭周期误判 — 噪声会让 coherence score 反复超过 0.12。需要额外门控：要么把启动命令的 `--blink-twinkle-fmcw-min-score` 提到 ~0.30，要么在 `_fmcw_twinkle_score` 里加 smoothness 门（低 smoothness 直接归零）。
    4. 任何修改都必须在 184117 / 190655 / 214655 三个 session 上验证：事件数 < 20，balanced_score >= 0，threshold 不锁死在 min_score。