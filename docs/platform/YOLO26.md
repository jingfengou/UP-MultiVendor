# YOLO26 训练与使用

两个任务共用 `up/tasks/yolo26/` 插件，网络、增强、损失、优化器、EMA 和验证流程使用各自固定版本的官方代码。UP 负责选择 NVIDIA / 沐曦运行时，沐曦专用操作使用 MACA/cuBridge 原生扩展。

| 任务 | 配置 | 初始化 | 轮数 / 尺寸 | 四卡训练 batch |
|---|---|---|---|---|
| YOLO26s 检测 | [yolo26s_coco70.yaml](../../configs/det/yolo26/yolo26s_coco70.yaml) | 官方 Objects365 150 轮权重，再构造 COCO 80 类头 | 70 / 640 | 4 × 32 = 128 |
| YOLO26s-cls 分类 | [yolo26s_imagenet200.yaml](../../configs/cls/yolo26/yolo26s_imagenet200.yaml) | 公共随机初始化 | 200 / 224 | 4 × 64 = 256 |

两家统一 seed 131、AMP FP16、TF32 关闭。分类显式使用 **SGD，初始学习率 0.1，200 轮余弦衰减**，对齐发布权重中的实际学习率记录。固定的 v8.4.0 代码在 `optimizer: auto` 时会改用 `0.01`，因此这里明确指定 SGD；插件会检查优化器和各参数组的初始学习率，训练恢复后也会检查。检测使用官方 **MuSGD**。

## 1. 准备环境与官方代码

先按 [README](../../README.md#安装与选择厂商) 安装对应厂商 Torch 和 UP 原生扩展，再安装辅助依赖：

```bash
python -m pip install --no-deps -r requirements/yolo26.txt

git clone https://github.com/ultralytics/ultralytics.git /opt/yolo26-detect
git -C /opt/yolo26-detect checkout 557e95b99c24c5b0d4eb5973d5d54a20a799566a
git clone --branch v8.4.0 --depth 1 \
  https://github.com/ultralytics/ultralytics.git /opt/yolo26-classify

export YOLO_CONFIG_DIR=/data/yolo-settings
export TORCH_HOME=/data/torch-cache
mkdir -p "$YOLO_CONFIG_DIR" /data/up-init
```

不需要 `pip install ultralytics`，启动时 `--source` 指定官方代码目录。插件检查源码和配置文件的 SHA256，再导入该版本。检测和分类分别启动新进程，也不会升级已经安装的 YOLO11。

检测固定的是发布权重记录的官方实验分支，包含该配方需要的 MuSGD 参数；分类发布权重没有记录历史 commit，采用固定的公开 v8.4.0 代码和其 200 轮训练参数。

## 2. 准备数据

检测使用官方 COCO 数据 YAML，目录中包含 `images/train2017`、`images/val2017`、对应 `labels/` 和 `annotations/instances_val2017.json`。本配方的 CopyPaste 需要官方 **coco2017labels-segments** 多边形标签。可使用固定源码中的 `ultralytics/cfg/datasets/coco.yaml`，复制后把 `path` 改为本机数据根目录，保留 80 类名称及 `val2017.txt` 验证列表。

分类使用 ImageFolder 目录：`imagenet/train/<类别>/图片`、`imagenet/val/<类别>/图片`，两边类别目录及排序相同。正式任务分别使用完整 COCO2017 和 ImageNet-1K。

## 3. 创建公共初始化

只生成一次，然后把 `.pt` 和同名 `.pt.json` 一起复制给两家。已存在的文件不会覆盖。

```bash
curl -L --fail \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-objv1-150.pt \
  -o /data/up-init/yolo26s-objv1-150.pt

CUDA_VISIBLE_DEVICES="" python -m up yolo26 init \
  --config configs/det/yolo26/yolo26s_coco70.yaml --source /opt/yolo26-detect \
  --pretrained /data/up-init/yolo26s-objv1-150.pt \
  --output /data/up-init/yolo26s-coco.pt

CUDA_VISIBLE_DEVICES="" python -m up yolo26 init \
  --config configs/cls/yolo26/yolo26s_imagenet200.yaml --source /opt/yolo26-classify \
  --output /data/up-init/yolo26s-cls.pt
```

检测会检查 Objects365 文件哈希，再按官方代码生成 80 类头。分类从 YAML 随机建模，不加载已经训练好的 ImageNet 分类权重。

## 4. 训练、恢复与验证

先进入对应厂商环境。下面的 `PLATFORM` 可选 `nvidia` 或 `metax_cuda`，`batch` 是全局 batch，已经写进 YAML。

```bash
PLATFORM=nvidia
CFG=configs/det/yolo26/yolo26s_coco70.yaml
SOURCE=/opt/yolo26-detect
DATA=/data/coco.yaml
INIT=/data/up-init/yolo26s-coco.pt
RUN=/data/up-runs/$PLATFORM/yolo26s-coco

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  tools/run_yolo26.py train --config "$CFG" --source "$SOURCE" \
  --platform "$PLATFORM" --data "$DATA" --weights "$INIT" --output "$RUN"

# 中断后恢复，继续原配置的完整训练预算。
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  tools/run_yolo26.py train --config "$CFG" --source "$SOURCE" \
  --platform "$PLATFORM" --data "$DATA" \
  --weights "$RUN/weights/last.pt" --output "$RUN" --resume

# 新进程验证；last.pt 与 best.pt 分别执行，分别保存结果。
CUDA_VISIBLE_DEVICES=0 python -m up yolo26 val \
  --config "$CFG" --source "$SOURCE" --platform "$PLATFORM" \
  --data "$DATA" --weights "$RUN/weights/last.pt" --output "$RUN/val-last"
```

分类复用这些命令，替换以下五个变量：

```bash
CFG=configs/cls/yolo26/yolo26s_imagenet200.yaml
SOURCE=/opt/yolo26-classify
DATA=/data/imagenet
INIT=/data/up-init/yolo26s-cls.pt
RUN=/data/up-runs/$PLATFORM/yolo26s-cls
```

训练中的验证继续使用 training 运行时；独立 `val` 和 `predict` 使用 inference 运行时。训练日志、逐轮指标和 `weights/last.pt`、`weights/best.pt` 由官方 Trainer 保存；恢复也由官方流程处理。

## 5. 图片预测与 ONNX

两类模型使用同一个命令，分别加载自己的配置、源码和权重：

```bash
CUDA_VISIBLE_DEVICES=0 python -m up yolo26 predict \
  --config "$CFG" --source "$SOURCE" --platform "$PLATFORM" \
  --weights "$RUN/weights/best.pt" --images /data/example.jpg \
  --output "$RUN/predictions"

CUDA_VISIBLE_DEVICES=0 python -m up yolo26 export \
  --config "$CFG" --source "$SOURCE" --platform "$PLATFORM" \
  --weights "$RUN/weights/best.pt"
```

导出前安装 `requirements/export.txt`。ONNX 写在权重旁边，固定 batch 1、FP32、opset 17；检测保留官方端到端输出，分类输出 1000 类概率。

## 对应代码

| 文件 | 作用 |
|---|---|
| [commands/yolo26.py](../../up/commands/yolo26.py) | `up yolo26` 参数和五种运行模式 |
| [tasks/yolo26/plugin.py](../../up/tasks/yolo26/plugin.py) | 官方模型初始化、训练、验证、预测、导出；绑定 UP 运行时 |
| [tasks/yolo26/source.py](../../up/tasks/yolo26/source.py) | 固定官方源码，阻止混用检测、分类或 YOLO11 的版本 |
| [run_yolo26.py](../../tools/run_yolo26.py) | 同一入口的脚本形式，供 torchrun 使用 |

源码、权重及其许可见 [Ultralytics](https://github.com/ultralytics/ultralytics)；官方检测配方说明见 [YOLO26 training recipe](https://docs.ultralytics.com/guides/yolo26-training-recipe/)。
