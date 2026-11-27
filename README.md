# UP-MultiVendor

**开发中。** 这是基于 UP 的独立多厂商适配项目，尚未完成全部功能和各厂商真机验证；不是原版 UP 的正式发布。

从 [原版 UP 52aaeb9](https://github.com/ModelTC/United-Perception/tree/52aaeb9ed27a490820696d7754d19f98cd6ecd11) 重建。用 config 选择显卡，保留原训练、验证、独立推理和 ONNX 导出入口。

```bash
git clone https://github.com/jingfengou/UP-MultiVendor.git
cd UP-MultiVendor
```

| 正式对比任务 | 当前唯一维护的 benchmark 配方 |
|---|---|
| ResNet50 · FAIR | [resnet50_fair_native.yaml](configs/benchmarks/reference_aligned/resnet50_fair_native.yaml) |
| YOLOX-S · Megvii 配方对齐、UP BaseRunner | [yolox_s_megvii_aligned.yaml](configs/benchmarks/three_vendor/yolox_s_megvii_aligned.yaml) |
| YOLO26s · COCO 检测 | [yolo26s_detect.yaml](configs/benchmarks/reference_aligned/yolo26s_detect.yaml) |
| YOLO26s · ImageNet 分类 | [yolo26s_classify.yaml](configs/benchmarks/reference_aligned/yolo26s_classify.yaml) |
| DETR-R50 | [detr_r50.yaml](configs/benchmarks/reference_aligned/detr_r50.yaml) |
| YOLO11s | [yolo11s.yaml](configs/benchmarks/three_vendor/yolo11s.yaml) |

以上是各正式 benchmark 任务的唯一完整训练配置。`configs/benchmarks/_components/` 中的 YAML 是被主配方 include 的组件片段，不是独立训练配置。`configs/cls/`、`configs/det/` 下仍保留通用模型和平台 smoke-test 示例；它们不代表当前精度对比配方。尤其是 `configs/det/yolox/yolox_small.yaml` 是通用原版 UP 示例，正式 YOLOX-S 对比应使用表中的 Megvii 对齐配方。

## 架构

```text
config → RuntimeBackend → 设备、AMP、通信、计算库策略
UP 模型 → 标准 Torch 操作 → 厂商 Torch / 计算库
        → UP 专用操作 → NativeOps → 厂商实现
```

ResNet/ViT 的模型计算保持原版。YOLO11s 使用官方 Ultralytics 结构和损失；YOLO26、DETR 通过注册组件接入固定版本的参考实现，由 UP `BaseRunner` 训练。普通 ReLU、卷积和矩阵乘法直接用 Torch，不逐层套包装。原版 43 个 C++/CUDA 文件保留：NVIDIA 用 CUDA 编译，沐曦用 MACA cuBridge 编译为各自的扩展。

## 安装与选择厂商

Python 3.10 及以上。每家使用独立环境，先安装该厂商提供的 Torch、匹配的 TorchVision、设备插件与驱动。然后在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -r requirements/yolo11.txt
python -m pip install --no-deps -r requirements/yolo26.txt
python -m pip install -r requirements/export.txt
python -m pip install --no-deps -e .
```

这些命令不会主动安装 NVIDIA Torch。普通安装不编译 GPU 扩展。**YOLOX 的 NVIDIA / 沐曦 GPU 训练需要原 IoU 扩展**；在对应厂商环境里选择一条编译命令：

```bash
UP_BUILD_PLATFORM=nvidia python setup.py build_ext --inplace
# 无卡构建机还需显式设置 TORCH_CUDA_ARCH_LIST 为目标卡架构。

# MACA Torch + cuBridge 环境；产物与 NVIDIA 扩展分开存放。
UP_BUILD_PLATFORM=metax_cuda MAX_JOBS=2 python setup.py build_ext --inplace
```

沐曦的 16 个专用操作已接入原内核，并完成 A100 / C500 同输入对照：29 组用例、训练和推理各重复 3 次。**15 项通过；RoIAlign 的 FP32/FP64 通过，FP16 前向已对齐，反向仍有并行累加波动。** 沐曦构建仅对 RoIAlign 使用 `--fmad=false`，扩展检查的 26 组 FP16 前向均与 A100 相同；原生并行反向保留，部分结果超出 `rtol=1e-4, atol=1e-5`。操作清单、差异和构建要求见 [厂商接入指南](docs/platform/ADDING_VENDOR.md#沐曦原生扩展)。

config 中只修改厂商选择，数据和配方保持相同：

```yaml
runtime:
  platform: nvidia   # metax_cuda / ascend_npu / cambricon_mlu
  fp16: false
```

切换时先进入该厂商环境，再启动新进程。`platform` 选择当前环境里的实现，不会把 NVIDIA Torch 换成沐曦 Torch。NPU/MLU 分别需要 `torch_npu` / `torch_mlu`。

## 训练、验证、推理、导出

先在对应 YAML 中设置数据路径。下面的 ResNet 配置是通用 smoke-test 示例；正式对比请使用上表中唯一的 benchmark 配方。启动器默认单卡，多卡使用 `--ng 2`。正式 YOLO26 训练也走 UP `BaseRunner`，配置和命令见 [YOLO26 使用指南](docs/platform/YOLO26.md)。

```bash
# 训练；可在命令末尾覆盖厂商
python -m up train --config configs/cls/resnet/res50.yaml \
  --opts runtime.platform metax_cuda

# 验证：沿用训练后端，加载指定权重
python -m up train --evaluate --config configs/cls/resnet/res50.yaml \
  --opts runtime.platform metax_cuda saver.auto_resume false saver.resume_model /data/resnet50.pth

# 独立图片/目录推理：使用 inference 用途
python -m up inference --config configs/cls/resnet/res50.yaml \
  --ckpt /data/resnet50.pth --img_path /data/example.jpg --vis_dir /data/predictions \
  --opts runtime.platform metax_cuda

# ONNX：不读取训练集；固定 batch=1 和输入尺寸
python -m up to_onnx --config configs/cls/resnet/res50.yaml \
  --ckpt /data/resnet50.pth --save_prefix /data/resnet50.onnx \
  --input_size 3x224x224 --verify
```

也可使用 `scripts/train.sh`、`evaluate.sh`、`inference.sh`、`to_onnx.sh`。检测导出保留原始预测，NMS 在图外；输出含义见 [架构说明](docs/platform/ARCHITECTURE.md#onnx-输出)。

## 训练配置

benchmark 配方均由 UP 的统一训练、验证和推理入口执行；模型差异通过 YAML 选择已注册的网络、数据、增强、损失、优化器和评估组件。YOLOX-S 当前配方使用 UP 原生网络与 BaseRunner，按 Megvii 数据增强和配置对齐，不是“原版 UP 配方”，也不是 Megvii 原训练器。各任务的全局 batch、精度、随机种子和数据要求见 [训练指南](docs/platform/TRAINING.md) 与 [Reference Recipes](docs/platform/REFERENCE_RECIPES.md)。

## 厂商扩展

设备、AMP、通信和计算库策略通过 `RuntimeBackend` 扩展；专用操作在 `up/platform/<vendor>/native_ops.py` 中绑定。标准 Torch 操作直接使用当前厂商环境。厂商不支持的接口明确报错，所需实现及注册方式见 [厂商接入指南](docs/platform/ADDING_VENDOR.md)。

**开发交接（待真机验收）**：[寒武纪 MLU](up/platform/cambricon_mlu/) · [昇腾 NPU](up/platform/ascend_npu/)。各目录包含运行时、专用算子和检测 NMS 的实现入口及验收说明。

[架构说明](docs/platform/ARCHITECTURE.md) · [训练指南](docs/platform/TRAINING.md) · [厂商接入](docs/platform/ADDING_VENDOR.md) · [接口浏览](docs/platform/HANDOFF.html)

原版 UP 保留其 [Apache-2.0 许可](LICENSE)。外部模型源码和依赖的来源、版本与各自许可见设计文档。
