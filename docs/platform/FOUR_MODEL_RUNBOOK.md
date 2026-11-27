# 改版 UP 四模型训练与评测

寒武纪和昇腾共用下面的训练入口与实验配置。四个模型包含五项任务，YOLO26s 的分类和检测分别训练、评测和归档。两家的厂商后端目前待真机验收。

## 1. 获取代码和环境

在各自厂商的 Linux 容器中使用匹配的 PyTorch、TorchVision 和 SDK。寒武纪使用 `torch_mlu`、CNNL/CNCL，昇腾使用 `torch_npu`、CANN/HCCL。

```bash
git clone --branch main https://github.com/jingfengou/UP-MultiVendor.git
cd UP-MultiVendor
python -m pip install -r requirements.txt
UP_BUILD_PLATFORM=none python -m pip install --no-deps -e .
# 跑 YOLO26s 时额外安装
python -m pip install --no-deps -r requirements/yolo26.txt
export YOLO_AUTOINSTALL=false
```

| 文件 | 作用 |
| --- | --- |
| [scripts/train.sh](../../scripts/train.sh)、[scripts/evaluate.sh](../../scripts/evaluate.sh) | 启动训练、加载 UP checkpoint 跑验证集 |
| [up/commands/train.py](../../up/commands/train.py)、[up/runner/base_runner.py](../../up/runner/base_runner.py) | 两个入口共用的参数解析、训练和评测循环 |
| [up/platform/cambricon_mlu/](../../up/platform/cambricon_mlu/) | `runtime.platform=cambricon_mlu`，寒武纪负责人在此适配 |
| [up/platform/ascend_npu/](../../up/platform/ascend_npu/) | `runtime.platform=ascend_npu`，昇腾负责人在此适配 |

用厂商容器或调度系统分配可见卡。UP 根据 `runtime.platform` 选择设备和通信后端，默认按单台机器启动。

## 2. 选择任务并准备数据

| 任务与指标 | 训练、验证共用 YAML | 轮数 | 4 卡每卡 / 全局 batch | 训练精度 / seed | 启动前准备 |
| --- | --- | ---: | ---: | --- | --- |
| ResNet50 · ImageNet-1K Top-1，单中心裁剪 | [resnet50_fair_native.yaml](../../configs/benchmarks/reference_aligned/resnet50_fair_native.yaml) | 90 | 64 / 256 | FP32 / 0 | `UP_IMAGENET_ROOT`、`UP_FAIR_OUTPUT_ROOT` |
| YOLOX-S · COCO AP50:95 | [yolox_s_megvii_aligned.yaml](../../configs/benchmarks/three_vendor/yolox_s_megvii_aligned.yaml) | 300 | 16 / 64 | AMP / 未显式设 seed | 工作目录含 `coco/`，输出写入 `outputs/` |
| YOLO26s 检测 · COCO 端到端 AP50:95 | [yolo26s_detect.yaml](../../configs/benchmarks/reference_aligned/yolo26s_detect.yaml) | 70 | 32 / 128 | AMP / 0 | `YOLO26_DETECT_SOURCE`、`OBJECTS365_WEIGHTS`、`COCO_YOLO_YAML` |
| YOLO26s 分类 · ImageNet-1K Top-1 | [yolo26s_classify.yaml](../../configs/benchmarks/reference_aligned/yolo26s_classify.yaml) | 200 | 64 / 256 | AMP / 0 | `YOLO26_CLASSIFY_SOURCE`、`IMAGENET_ROOT` |
| DETR R50 · COCO AP50:95 | [detr_r50.yaml](../../configs/benchmarks/reference_aligned/detr_r50.yaml) | 150 | 8 / 32 | FP32 / 42 | `DETR_SOURCE`、`COCO_ROOT` |

启用 `runtime.audit_recipe` 的配方会在检查点目录记录各 rank 的实际种子和 AMP 状态。ResNet50 和 YOLOX-S 使用 UP 自带网络，分别按 FAIR 与 Megvii 配方设置训练和评测。YOLOX-S 只维护一份 Megvii 对齐配方；`random_seed: null` 表示不主动设置种子，对应作者源码默认行为。YOLO26s 和 DETR 通过注册组件在 UP `BaseRunner` 里使用固定版本的外部模型、数据与指标。分类 YOLO26s 使用公开 `v8.4.0` 的 MuSGD。配方细节见 [Reference Recipes](REFERENCE_RECIPES.md)。

表中每项任务对应唯一的正式 benchmark YAML。`configs/benchmarks/_components/` 只存放被主配方 include 的片段；`configs/cls/`、`configs/det/` 的通用示例不应用来替代表中的正式训练配置。

- ResNet50 的 `UP_IMAGENET_ROOT` 下有 `train/`、`val/` 和 `meta/{train,val}.txt`。每行是相对各自图片根目录的路径及整数类别，例如 `n01440764/example.JPEG 0`。交接时使用同一份类别映射和清单。
- YOLOX-S 的工作目录下需要 `coco/train2017/`、`coco/val2017/` 和 `coco/annotations/instances_{train,val}2017.json`；建议将 `coco/` 链接到数据集。DETR 的 `COCO_ROOT` 下使用相同目录结构。DETR 还需要 TorchVision 的 ImageNet 预训练 ResNet50 V1 骨干。离线时提前放入 `TORCH_HOME` 对应缓存目录。
- YOLO26s 检测的 `COCO_YOLO_YAML` 指向本地 COCO YAML，按固定源码的 `ultralytics/cfg/datasets/coco.yaml` 复制并修改 `path`。需要 `images/`、`labels/`、`train2017.txt`、`val2017.txt` 及 `annotations/instances_val2017.json`，训练使用官方多边形标签。`OBJECTS365_WEIGHTS` 指向初始化文件 [yolo26s-objv1-150.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-objv1-150.pt)。
- YOLO26s 分类的 `IMAGENET_ROOT` 使用 `train/<类别>/`、`val/<类别>/` 的 ImageFolder 结构。

两个 YOLO26s 任务使用各自可写的数据工作副本。上游数据检查会生成并更新标签缓存，可能修复损坏的 JPEG。共享数据按原样保留。评测保持 ImageNet-1K 全部验证样本或 COCO `val2017` 完整验证集。

需要外部源码时执行对应几行。三个 `*_SOURCE` 的值都是检出的仓库根目录。配置会验证源码摘要，检测还会校验初始权重。

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

## 3. 平台验收

1. 跑 [厂商接入指南的设备探针](ADDING_VENDOR.md#5-寒武纪与昇腾验收)，核对当前设备、前向、梯度、CNCL/HCCL。再用独立测试目录确认单卡前后向、优化器更新、保存和加载，以及双卡通信。
2. FAIR ResNet50、YOLOX-S 和 DETR 三份 YAML 显式设置了 `runtime.dnn`。厂商需在各自 `runtime.py::Backend.configure_dnn` 中映射可支持的计算库策略并记录实测值，当前模板会对未映射的选项报错。诊断其他功能时可在同目录复制 YAML，在诊断配置中将 `runtime.dnn` 设为空映射 `{}`。正式结果需注明计算库策略差异。
3. 两份 YOLO26s 配置使用的上游 AMP 检查会调用 `torch.cuda.get_device_name`。适配时在 [YOLO26ForUP.check_amp](../../up/tasks/det/plugins/yolo26/models/model.py) 接入厂商检查，核对 FP16 前后向、GradScaler 溢出跳步及恢复。若用 `trainer.execution.amp_check false` 跳过该探针，要先完成独立的厂商 AMP 验证，并保持 `runtime.fp16 true`。YOLOX-S 还需核对 [NMS](../../up/platform/ops.py) 和 OTA `overlap`，YOLO26s 验证需检查多卡汇总和在目标设备上的精度。
4. 四卡通过短训练及完整验证后再启动正式训练。卡数确需调整时，训练命令须同时设置 `runtime.expected_world_size` 和每卡 batch，使全局 batch 等于表中数值。ResNet50、YOLOX-S 的字段是 `dataset.train.batch_sampler.kwargs.batch_size`，YOLO26s、DETR 是 `dataset.train.batch_size`。保留学习率、精度、轮数和基础 seed，记录每卡 batch 对 BN 和采样的影响。

## 4. 训练并复测最佳 checkpoint

下面以 ResNet50 举例。在已验收的目标平台执行。跑其他任务时替换 `CFG` 和 `RUN`，并按任务准备表中的数据路径和源码绑定。

```bash
set -o pipefail
PLATFORM=cambricon_mlu  # 昇腾用 ascend_npu
NG=4
CFG=configs/benchmarks/reference_aligned/resnet50_fair_native.yaml
export UP_IMAGENET_ROOT=/data/imagenet

RUN=/path/to/owned-afs/up/$PLATFORM/resnet50-fair-01
export UP_FAIR_OUTPUT_ROOT="$RUN"
mkdir -p "$RUN"
# YOLO26s 还可设置自己的可写目录
# export YOLO_CONFIG_DIR="$RUN/yolo-config"
# mkdir -p "$YOLO_CONFIG_DIR/Ultralytics"

bash scripts/train.sh "$NG" "$CFG" \
  --opts runtime.platform "$PLATFORM" \
    saver.save_dir "$RUN/checkpoints" saver.results_dir "$RUN/results" \
  2>&1 | tee "$RUN/train.log"
```

YOLOX-S 的 YAML 将检查点和结果写到当前目录下的 `outputs/`，因此从 AFS 上的独立运行目录启动，并在该目录创建 `coco/` 数据链接：

```bash
RUN=/path/to/owned-afs/up/$PLATFORM/yolox-s-megvii
mkdir -p "$RUN"
ln -s /data/coco "$RUN/coco"
cd "$RUN"
python -m up train \
  --config /path/to/UP-MultiVendor/configs/benchmarks/three_vendor/yolox_s_megvii_aligned.yaml \
  --ng 4 --launch pytorch --opts runtime.platform "$PLATFORM"
```

ResNet50 训练结束后，从验证选出的 `ckpt_best.pth` 另起进程跑完整验证集，使用同一份配置、源码和数据路径：

```bash
BEST="$RUN/checkpoints/ckpt_best.pth"
EVAL="$RUN/eval-best"
mkdir -p "$EVAL"
UP_FAIR_OUTPUT_ROOT="$EVAL" \
  bash scripts/evaluate.sh "$NG" "$CFG" \
    --opts runtime.platform "$PLATFORM" saver.auto_resume false \
      saver.resume_model "$BEST" saver.save_dir "$EVAL/checkpoints" \
      saver.results_dir "$EVAL/results" \
  2>&1 | tee "$EVAL/eval.log"
```

YOLOX-S 的评测也从该任务的 AFS 工作目录启动，确保 `coco/` 链接仍在，并将输出重定向到新目录：

```bash
RUN=/path/to/owned-afs/up/$PLATFORM/yolox-s-megvii
cd "$RUN"
BEST="$RUN/outputs/checkpoints/ckpt_best.pth"
EVAL="$RUN/eval-best"
mkdir -p "$EVAL"
python -m up train --evaluate \
  --config /path/to/UP-MultiVendor/configs/benchmarks/three_vendor/yolox_s_megvii_aligned.yaml \
  --ng 4 --launch pytorch \
  --opts runtime.platform "$PLATFORM" saver.auto_resume false \
    saver.resume_model "$BEST" saver.save_dir "$EVAL/checkpoints" \
    saver.results_dir "$EVAL/results"
```

训练使用自己的可写 AFS 新目录。Saver 会更新目录中的最佳 checkpoint 和最新 checkpoint 链接。完成后把实际运行配置 `running_config.json`、代码 commit 和适配 diff、设备/SDK/Torch 版本、每卡和全局 batch、训练及评测日志、最佳 checkpoint 和 SHA256 一起归档到自己的 AFS 或 AOSS `up/` 目录。分类报 Top-1，检测报 AP50:95，记录实际加载的 checkpoint。
