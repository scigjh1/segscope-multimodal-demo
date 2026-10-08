# SegScope AI GPU Runtime

前端复用 [NiiVue](https://github.com/niivue/niivue) 的完整绘图示例和 BSD-2-Clause 查看器；页面组件与样式来自 [Tabler](https://github.com/tabler/tabler)（MIT）。源码、版本、修改范围和原始许可证保存在 `static/vendor`，没有用一张医学影像截图替代真实查看器。

## 启动

GPU 环境与旧二维 OpenCV 环境分别安装；不要把两套 NumPy 版本约束混装。

```bash
python -m venv .venv-gpu
source .venv-gpu/bin/activate
pip install -r requirements-gpu.txt
export SEG_SCOPE_MODEL_ADAPTER=/absolute/operator/private/adapter.py
export SEG_SCOPE_DATA_DIR=/absolute/operator/private/case-store
python -m uvicorn gpu_runtime.api:app --host 127.0.0.1 --port 4186
```

Windows 使用 `.venv-gpu\Scripts\activate` 和 `$env:SEG_SCOPE_MODEL_ADAPTER=...`。打开 `http://127.0.0.1:4186/`。配置为空时 API 能展示设备、导入病例和查看报告，但明确禁止模型推理；不会显示假 CUDA 状态或生成假模型结果。

## 自己的模型

服务端可信适配器实现 `SegmentationModel`，暴露 `create_adapter()`。浏览器不能提交 Python 模块路径或权重路径。

```python
from gpu_runtime.adapters import SegmentationModel, TorchScriptAdapter

def create_adapter():
    return TorchScriptAdapter('/private/my_model.ts', model_id='MySeg-v1', version='1')
```

非 TorchScript 模型继承 `SegmentationModel`，按自己的训练协议实现 `load(device)`、`preprocess(image, spacing)`、`predict(model, tensor, spacing)` 和必要的 `postprocess(logits)`。运行时统一输入为 `B,C,D,H,W`，spacing 为 `Z,Y,X`；适配器负责训练轴序、模态和归一化匹配。二分类默认 sigmoid，多分类默认输出前景联合概率，不能把它当作单个病灶类别的概率。

本次实际接入操作者已有的 AnatoFluxNet-Light-Direct（12.16M 参数）及冻结检查点；适配器、模型源码、原始权重、训练材料均留在仓库之外。公共源码不把平台实现冒称为新分割算法。

## 病例工作流

- DICOM 单帧或 ZIP 单序列、3D NIfTI、PNG/JPEG 接入；DICOM 使用方向与位置排序，LPS 转 RAS，丢弃 Patient/Study 标签。增强多帧、倾斜层面、不规则间距需专业加载器，当前明确拒绝。
- CUDA 自动探测，CPU / CUDA 明确选择；FP32 / AMP FP16；MONAI Gaussian sliding-window 和窗口批量。请求不存在的 CUDA 时返回错误，不静默退回 CPU。
- 可选 spacing 重采样；概率恢复原始尺寸，Mask/概率/熵图导出原始 affine。PNG 没有物理间距，不编造毫米或 mL。
- 真实 NiiVue 三视图、3D、掩膜叠加和绘图/撤销；Accept、Reject；导出修订 Mask 后上传，原始 shape/affine 与 0/1 标签验证通过才记录 Edit。
- SQLite 保留病例、运行、复核和失败分类。Edit/Reject 进入 error pool；公开部分完成困难样本回流，不自动运行研究模型再训练。
- 无参考标注只显示低置信区域和体积/面积；FP/FN、Dice、IoU、Precision、Recall、HD95、NSD 需参考标注。熵风险比例是工作流启发规则，不是临床诊断概率。
- 单边空 Mask 的 HD95 为 `null` 并有状态；两边空 Mask Dice 为 1。3D 距离使用体素间距，不能与旧二维像素距离混用。

## 实测协议

```bash
python scripts/benchmark_gpu.py --adapter /private/adapter.py --gpu cuda:0 \
  --shape 32 96 96 --roi 32 96 96 --warmup 5 --runs 30 --out benchmark_results/gpu/my_run.json
```

保存所有正式样本的预处理、H2D、模型推理、后处理/D2H 和总耗时；计时前后同步 CUDA。P50/P95 由整批延迟分布计算，吞吐为批内 volume 数 / 平均完整 runtime 耗时。排除冷模型加载、磁盘及网络；FP16 是 AMP 混合精度，不是 INT8 量化。显存记录 PyTorch 进程 allocated/reserved high-water mark，不混同整卡 `nvidia-smi` 用量。

`benchmark_results/gpu` 保存三份当次 A800 实测：单输入 CPU/FP32/FP16、四输入批量、48×128×128 大体积滑窗。固定随机输入仅证明运行时性能；相同空预测的高一致率不能证明模型精度。另有预先指定的公开验证病例回放时，单独记录其数据范围和质量指标。

性能门限保留真实失败：本次单输入 FP16 显存高于 FP32，显存 +20% 门限未通过。未把 FP16 加速同时写成显存降低。

## 验证与边界

```bash
python -m unittest discover -s tests -p test_gpu_runtime.py -v
```

已验证滑窗批量、native-space roundtrip、各向异性 HD95、空掩膜、无 GT 时的 FP/FN 禁用、复核持久化。运行时与私有权重通过本地服务接通；Dockerfile 是打包入口，当前测量来自直接运行的 PyTorch/FastAPI 服务。

原始病例、参考标注、修订、概率图、研究模型、权重和 SSH 信息不随仓库公开。开放数据的访问权限不自动等于重分发权限；本次仅公开已声明样本范围的汇总指标与合成性能测量。
