# GPU 实测 · 2026-10-08

操作者已有的 **AnatoFluxNet-Light-Direct / frozen-fold1-u7000**（12.16M 参数），同一冻结检查点、NVIDIA A800 80GB PCIe、PyTorch 2.1.0+cu121、MONAI 1.3.2。未在本次任务中训练或调参。

## 固定三维输入性能

每种模式 **5 次预热 + 30 次正式测量**；单输入 B×D×H×W = **1×32×96×96**，ROI 为 32×96×96，窗口批量 1，CPU 8 线程。固定随机 CT-like 输入只用于运行时测量。

| 模式 | P50 / ms | P95 / ms | volume/s | 峰值 allocated / MiB | 相对 CPU P50 |
|---|---:|---:|---:|---:|---:|
| CPU FP32 | 493.43 | 584.94 | 2.06 | 0 | 1.00× |
| A800 FP32 | 21.11 | 22.61 | 47.02 | 169.99 | 23.38× |
| A800 AMP FP16 | 8.67 | 11.33 | 108.30 | 298.09 | 56.89× |

[全部正式样本与协议](../benchmark_results/gpu/01_single_volume.json)。同步 wall time 包含预处理、H2D、MONAI 推理、后处理/D2H；不含模型冷加载、磁盘、网络或病例质量指标计算。吞吐按平均完整 runtime 计算，不能把 patch/s 写成完整临床病例/s。

FP16 allocated 显存高于 FP32，**+20% 显存回归门限失败**；原始报告保留 FAIL。AMP FP16 是混合精度推理，不是 INT8 量化。共享 GPU 的负载快照随 JSON 保存；性能会受其他任务影响。

四输入批量的 GPU FP32 / FP16 P50 分别为 **85.58 / 35.97 ms**，这是每批耗时；[原始批量报告](../benchmark_results/gpu/02_batch4.json)。48×128×128 大体积、32×96×96 滑窗 ROI、窗口批量 4 的 P50 为 **82.09 / 30.46 ms**；[原始滑窗报告](../benchmark_results/gpu/03_sliding_window.json)。

三种合成性能工作负载中，两种精度输出均为空掩膜。Mask agreement=1 是空预测的一致性，**不代表分割精度**。补充 replay 校验相同输入 SHA256 与冻结权重后记录前景数量，原始延迟样本没有覆盖。

## 公开验证病例

预先指定 AMOS22 fold1 验证列表中的首个病例，目标为 liver label 6；采用训练匹配的 96×256×256 ROI，FP16、窗口批量 1、overlap 0.25。未读取封存测试集，未按测量结果挑选病例。

| 样本数 | Dice | IoU | Precision | Recall | HD95 / mm | NSD @ 3 mm |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9629 | 0.9284 | 0.9765 | 0.9496 | 4.1875 | 0.9273 |

原始体积 103×768×768，spacing 为 5.0×0.5234375×0.5234375 mm。完整 runtime **5.219 s**，其中模型推理 **3.702 s**；峰值 allocated **3075.88 MiB**。这是单例测量，不能表述为 P50/P95 或整个验证集的平均精度。

[验证范围与汇总记录](../benchmark_results/gpu/public_validation.json) · [AMOS22 来源](https://github.com/Jianningli/amos)。影像、标注、概率图和权重留在操作者私有环境。该结果不是医院临床试验、医生计时实验或真实业务效果。

## 可用于项目介绍的事实

接入已有自研 12.16M 参数三维模型，基于 PyTorch/MONAI 实现 CPU/CUDA、AMP FP16、批量与 Gaussian sliding-window 推理；在 A800 上以 5 次预热 + 30 次正式样本测得固定 32×96×96 输入的 P50 从 CPU 493.43 ms 降至 GPU FP16 8.67 ms，约 56.89× 加速。完成 NiiVue 三视图、native-space 修订、病例指标、困难样本池及精度/性能回归链路；单个预先指定公开 AMOS22 验证病例 Dice 为 0.9629。

只有同一冻结模型执行了实际推理。本次实现了不同权重版本比较接口及测试，没有把 FP32→FP16 变化声称为 v1→v2 训练迭代效果。
