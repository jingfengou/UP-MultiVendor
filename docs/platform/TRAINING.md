# ImageNet / COCO 训练指南

正式对比使用下表中的 benchmark 配方。每个模型任务只维护一份完整可运行配置；`configs/benchmarks/_components/` 是主配置 include 的组件片段，不是独立配方。`configs/cls/`、`configs/det/` 下的通用 YAML 用于模型示例或平台 smoke test，不用于报告 benchmark 结果。

| 模型 / 任务 | 数据与指标 | 当前配方 | Epoch | 全局 batch（4 卡示例） | 精度 / seed |
|---|---|---|---:|---:|---|
| ResNet50 · FAIR | ImageNet-1K · 单中心裁剪 Top-1 | [resnet50_fair_native.yaml](../../configs/benchmarks/reference_aligned/resnet50_fair_native.yaml) | 90 | 256（64/卡） | FP32 / 0 |
| YOLOX-S · Megvii 配方对齐 | COCO2017 · AP50:95 | [yolox_s_megvii_aligned.yaml](../../configs/benchmarks/three_vendor/yolox_s_megvii_aligned.yaml) | 300 | 64（16/卡） | AMP / null |
| YOLO26s · 检测 | COCO2017 · 端到端 AP50:95 | [yolo26s_detect.yaml](../../configs/benchmarks/reference_aligned/yolo26s_detect.yaml) | 70 | 128（32/卡） | AMP / 0 |
| YOLO26s · 分类 | ImageNet-1K · Top-1 | [yolo26s_classify.yaml](../../configs/benchmarks/reference_aligned/yolo26s_classify.yaml) | 200 | 256（64/卡） | AMP / 0 |
| DETR-R50 | COCO2017 · AP50:95 | [detr_r50.yaml](../../configs/benchmarks/reference_aligned/detr_r50.yaml) | 150 | 32（8/卡） | FP32 / 42 + rank |
| YOLO11s | COCO2017 · AP50:95 | [yolo11s.yaml](../../configs/benchmarks/three_vendor/yolo11s.yaml) | 600 | 128（32/卡） | AMP / 131 |

表中的 batch 是跨所有进程的全局 batch；使用不同卡数时相应调整每卡 batch，保持全局 batch 和配方其余参数不变。ResNet50、YOLOX-S、YOLO26 和 DETR 由 UP `BaseRunner` 执行；YOLOX-S 使用 UP 原生网络与注册组件，按 Megvii 配方设置，不调用 Megvii 原训练器。YOLO26、DETR 的模型、增强、损失或评估组件通过 UP 注册器接入，不替换训练循环。YOLO11 使用注册的 Ultralytics 组件。配置细节见 [Reference Recipes](REFERENCE_RECIPES.md)。

## 数据与源码

- ResNet50：`UP_IMAGENET_ROOT` 下准备 `train/`、`val/` 和 `meta/{train,val}.txt`。清单类别映射在各厂商间保持一致；输出根目录通过 `UP_FAIR_OUTPUT_ROOT` 设置。
- YOLOX-S、DETR-R50：使用 COCO2017 `train2017/`、`val2017/` 和 `annotations/instances_{train,val}2017.json`。YOLOX-S 从 AFS 独立工作目录启动，并在其中建立 `coco/` 数据链接。
- YOLO26s 检测：设置 `YOLO26_DETECT_SOURCE`、`OBJECTS365_WEIGHTS` 和 `COCO_YOLO_YAML`。YOLO YAML 指向本地数据根目录，训练标签包含官方多边形标签。
- YOLO26s 分类：设置 `YOLO26_CLASSIFY_SOURCE`、`IMAGENET_ROOT`；数据使用 ImageFolder 目录结构。
- DETR-R50：设置 `DETR_SOURCE` 和 `COCO_ROOT`。固定源码需要 TorchVision ImageNet ResNet50 V1 骨干；离线环境提前放入对应 `TORCH_HOME` 缓存。
- YOLO11s：使用 COCO2017 和带轮廓的 `coco2017labels-segments` 标签，以支持官方 CopyPaste 配方。

所有厂商使用相同的数据与划分，并遵守配方指定的初始化规则；需要公共预训练权重时，各家使用同一文件。YOLO26 与 DETR 配置会验证固定源码摘要；检测还校验 Objects365 初始化权重。各任务的数据副本需可写，避免训练端修改共享原始数据。

需要固定外部源码时：

```bash
VENDOR_SOURCES="$PWD/../up-reference-sources"
mkdir -p "$VENDOR_SOURCES"

export YOLO26_DETECT_SOURCE="$VENDOR_SOURCES/yolo26-detect"
git clone https://github.com/ultralytics/ultralytics.git "$YOLO26_DETECT_SOURCE"
git -C "$YOLO26_DETECT_SOURCE" checkout --detach 557e95b99c24c5b0d4eb5973d5d54a20a799566a

export YOLO26_CLASSIFY_SOURCE="$VENDOR_SOURCES/yolo26-classify"
git clone --branch v8.4.0 --depth 1 https://github.com/ultralytics/ultralytics.git "$YOLO26_CLASSIFY_SOURCE"

export DETR_SOURCE="$VENDOR_SOURCES/detr"
git clone https://github.com/facebookresearch/detr.git "$DETR_SOURCE"
git -C "$DETR_SOURCE" checkout --detach 29901c51d7fe8712168b8d0d64351170bc0f83e0
```

## 启动与归档

先进入目标厂商独立环境，设置好表中任务需要的数据和源码变量，再为每次训练创建新的输出目录。`runtime.platform` 选择当前环境已安装的后端，不会自动切换 Torch。

```bash
PLATFORM=nvidia                 # C500: metax_cuda
NG=4
CFG=configs/benchmarks/reference_aligned/yolo26s_detect.yaml
RUN=/path/to/afs/up/$PLATFORM/yolo26s-detect
mkdir -p "$RUN"

python -m up train --config "$CFG" --ng "$NG" --launch pytorch \
  --opts runtime.platform "$PLATFORM" \
    saver.save_dir "$RUN/checkpoints" saver.results_dir "$RUN/results"
```

将 `CFG`、`RUN` 换成表中其他任务对应的路径即可。YOLOX-S 的完整启动、工作目录和 COCO 链接示例见 [四模型训练与评测交接](FOUR_MODEL_RUNBOOK.md)。

训练结束后，从最佳 checkpoint 另起进程做完整验证。验证保持同一配置、源码、数据和评估口径：

```bash
BEST="$RUN/checkpoints/ckpt_best.pth"
python -m up train --evaluate --config "$CFG" --ng "$NG" --launch pytorch \
  --opts runtime.platform "$PLATFORM" saver.auto_resume false \
    saver.resume_model "$BEST" \
    saver.save_dir "$RUN/eval-best/checkpoints" \
    saver.results_dir "$RUN/eval-best/results"
```

每次运行使用独立目录。完成后将实际运行配置、代码 commit、源码 revision、设备与 SDK/Torch 版本、每卡和全局 batch、训练及评估日志、最佳 checkpoint 和 SHA256 一并归档到自有 AFS/AOSS 的 `up/` 目录。分类报告 Top-1；检测报告 COCO AP50:95，并注明实际评估的 checkpoint。

官方参考配置与适配差异见 [Reference Recipes](REFERENCE_RECIPES.md)。各任务的完整数据要求和平台检查见 [四模型训练与评测交接](FOUR_MODEL_RUNBOOK.md)。
