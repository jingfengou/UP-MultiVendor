# ImageNet / COCO 训练指南

以下配置用于在不同厂商环境运行相同训练任务。先完成 [安装](../../README.md#安装与选择厂商)，再设置数据路径。

新增的 YOLO26s COCO 70 轮与 YOLO26s-cls ImageNet 200 轮，使用独立的官方训练插件。完整 YAML、初始化、训练、验证、预测及 ONNX 命令见 [YOLO26 使用指南](YOLO26.md)。下面保留已有 UP Runner 配置及 DETR 原始长配方。

| 模型 / 配置 | 数据集 | Epochs | 输入 | 卡数 × 每卡训练 batch | 每卡验证 batch |
|---|---|---:|---|---:|---:|
| [ResNet50](../../configs/benchmarks/three_vendor/resnet50.yaml) | ImageNet-1K | 100 | 224 | 4 × 256 | 256 |
| [YOLOX-S](../../configs/benchmarks/three_vendor/yolox_s.yaml) | COCO2017 | 300 | 训练 480–800，验证 640 | 4 × 8 | 8 |
| [YOLO11s](../../configs/benchmarks/three_vendor/yolo11s.yaml) | COCO2017 | 600 | 640，验证矩形补边 | 4 × 32 | 64 |
| [DETR-R50](../../configs/benchmarks/three_vendor/detr_r50_official.yaml) | COCO2017 | 500 | 官方多尺度；验证短边 800、长边不超过 1333 | 4 × 4 | 最终验证 2 |

前三个模型训练使用 AMP FP16，验证使用 FP32，seed 为 131；DDP、普通 BN、无梯度累积。全局 batch 分别是 1024、32、128。DETR 使用官方 FP32、FrozenBN、seed 42，全局 batch 16。配置会检查四卡 world size；运行前须在目标设备上确认显存与配套 SDK。

每家设备按 **ResNet50 → YOLOX-S → YOLO11s → DETR-R50** 顺序运行。

## 数据与配方

ImageNet 配置填写训练/验证清单与图片目录，保持类别映射一致。COCO 配置填写 train2017、val2017 图片目录、官方 `instances_*.json` 和 `evaluator.gt_file`。

YOLO11 还使用官方带轮廓的检测标签，供 CopyPaste 和几何增强使用；模型输出仍是检测框，不训练分割头：

```bash
curl -L --fail https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels-segments.zip -o coco2017labels-segments.zip
unzip coco2017labels-segments.zip -d /data
```

将 train / test 的 `label_dir` 分别设为 `/data/coco/labels/train2017` 和 `/data/coco/labels/val2017`。各厂商使用相同图片、标签和划分。

| 参数 | ResNet50 | YOLOX-S | YOLO11s |
|---|---|---|---|
| 优化器 | SGD，lr 0.4，momentum 0.9 | SGD，全局 batch 32 时 lr 0.005 | SGD，lr 0.01，momentum 0.9 |
| Weight decay | 0.0001 | 0.0005 | nominal batch 64，batch 128 时为 0.001 |
| 学习率 | 30 / 60 / 90 轮降为原来的 0.1 | 5 轮 warmup + 余弦衰减 | 3 轮 warmup + 线性衰减至初始的 0.01 |
| 训练收尾 | 保留分类增强 | 最后 15 轮关闭强增强并切换原训练阶段 | 最后 10 轮关闭 Mosaic、MixUp、CopyPaste |
| EMA | 不启用 | UP EMA | Ultralytics EMA |

完整参数以 YAML 为准。ResNet50 和 YOLOX 沿用 UP 配方；YOLO11 的网络、损失、增强和 EMA 使用 `ultralytics==8.3.225`。从零训练时使用公共随机初始化，不加载已训练权重。原生运行不强制确定性，TF32 关闭；厂商计算库开关见 [接入指南](ADDING_VENDOR.md)。

## 初始化与训练

每个模型生成一份随机初始权重，复制给各厂商使用。工具调用 UP 原来的 `set_random_seed()`，与 Runner 使用相同的种子规则；默认 `random_seed=131`、`rank_init=false` 时，Python / NumPy / CPU Torch 分别使用 `131`、`131²`、`131³`。权重旁的 JSON 记录种子函数、实际 Torch 种子和 SHA256，已有输出不覆盖。以下为 YOLO11 示例：

```bash
CFG=configs/benchmarks/three_vendor/yolo11s.yaml
python tools/prepare_training_init.py --config $CFG --output /data/up-init/yolo11s-seed131.pth

PLATFORM=nvidia  # metax_cuda / cambricon_mlu
RUN=/data/up-runs/${PLATFORM}/yolo11s
python -m up train --config $CFG --ng 4 --launch pytorch --opts runtime.platform $PLATFORM saver.pretrain_model /data/up-init/yolo11s-seed131.pth saver.save_dir $RUN/checkpoints saver.results_dir $RUN/results
```

ResNet50、YOLOX 替换配置、初始化文件和输出目录。每家先进入对应 Torch 环境，再选择 `runtime.platform`；配置不会自动安装或替换 Torch。

## 验证与恢复

在新进程验证 YOLO11 最终轮 EMA：

```bash
python -m up train --evaluate --config $CFG --ng 4 --launch pytorch --opts runtime.platform $PLATFORM saver.auto_resume false saver.pretrain_model $RUN/checkpoints/ckpt_e600.pth saver.save_dir $RUN/evaluate saver.results_dir $RUN/evaluate-results

# 从完整 epoch 检查点继续，保持原 600 轮配置与学习率时间轴。
python -m up train --config $CFG --ng 4 --launch pytorch --opts runtime.platform $PLATFORM saver.resume_model $RUN/checkpoints/ckpt_e100.pth saver.save_dir $RUN/checkpoints saver.results_dir $RUN/results
```

YOLO11 保存模型、优化器、LR、EMA 及各 rank 的 RNG、BN、scaler。独立图片推理与 ONNX 命令见 [README](../../README.md#训练验证推理导出)。

## 结果口径

分类报告 Top-1 / Top-5，检测报告 COCO AP / AP50 / AP75。最终轮与最佳验证轮分别记录，保持 EMA 和后处理设置一致。跨厂商比较固定数据、初始化、配置，并记录各家 Torch、驱动、SDK、实际 batch、耗时与峰值显存。

官方参考：[UP ResNet50 76.76% Top-1](https://github.com/ModelTC/United-Perception/blob/52aaeb9ed27a490820696d7754d19f98cd6ecd11/benchmark/classification_benchmark.md)、[UP YOLOX-S 40.4 AP](https://github.com/ModelTC/United-Perception/blob/52aaeb9ed27a490820696d7754d19f98cd6ecd11/benchmark/detection_benchmark.md)、[YOLO11s 47.0 AP](https://docs.ultralytics.com/models/yolo11/)。ResNet50 每卡 batch、YOLO11 多卡 BN 和种子等设置与官方记录存在差别；跨厂商基准使用同配置 NVIDIA 实测结果。

## DETR 官方配方

[官方 DETR-R50](https://github.com/facebookresearch/detr/tree/29901c51d7fe8712168b8d0d64351170bc0f83e0) 的 COCO val2017 成绩为 **42.0 AP**，对应 500 轮，第 400 轮后学习率降为十分之一。官方命令默认的 300 轮配置是另一套训练时长。

此项直接使用固定版本的官方模型、数据增强、训练循环和 COCO 评估器。UP 入口负责选择 NVIDIA / 沐曦运行时。它与 `configs/det/detr/detr_r50.yaml` 的 UP 适配器是两个入口；后者保留原功能，本项不使用它。

- AdamW：Transformer 学习率 `1e-4`，骨干 `1e-5`，weight decay `1e-4`；梯度裁剪 `0.1`。
- ImageNet 预训练 ResNet50 骨干；检测头随机初始化，不加载 COCO 检测权重。
- 使用官方水平翻转、多尺度缩放、随机裁剪，以及原始 COCO 类别 ID。
- 官方训练是 **8 × 2**；当前分配为 **4 × 4**，全局 batch 同为 16。每卡图片数会改变 padding 和采样分组，因此把这一项作为明确的硬件配置差别记录。
- 训练内验证沿用官方入口的每卡 batch 4；最终独立验证改为官方基准使用的每卡 batch 2。两家均用相同设置。

准备固定版本源码和一份公共初始化：

```bash
git clone https://github.com/facebookresearch/detr.git /opt/detr
git -C /opt/detr checkout 29901c51d7fe8712168b8d0d64351170bc0f83e0
export TORCH_HOME=/data/torch-cache
CFG=configs/benchmarks/three_vendor/detr_r50_official.yaml
python tools/train_official_detr.py --source /opt/detr --config $CFG \
  --platform nvidia --prepare-init /data/up-init/detr-r50-seed42.pth
```

初始化工具在 CPU 上加载 torchvision 的 ImageNet-1K V1 骨干，生成检测模型权重和 SHA256。将同一份初始化及骨干缓存复制到两家环境，再分别运行：

```bash
PLATFORM=nvidia  # 沐曦填 metax_cuda，并进入对应厂商 Torch 环境
RUN=/data/up-runs/${PLATFORM}/detr_r50
torchrun --standalone --nproc_per_node=4 tools/train_official_detr.py \
  --source /opt/detr --config $CFG --platform $PLATFORM \
  --coco-path /data/coco --resume /data/up-init/detr-r50-seed42.pth \
  --output-dir $RUN/train

torchrun --standalone --nproc_per_node=4 tools/train_official_detr.py \
  --source /opt/detr --config $CFG --platform $PLATFORM \
  --coco-path /data/coco --resume $RUN/train/checkpoint.pth \
  --output-dir $RUN/evaluate --eval
```

训练日志及逐轮 COCO AP 写入 `train/log.txt`，最新完整模型、优化器和学习率状态保存在 `train/checkpoint.pth`。恢复时使用同一命令，将 `--resume` 指向该检查点。
