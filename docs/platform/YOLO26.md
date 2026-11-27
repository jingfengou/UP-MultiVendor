# YOLO26 训练与使用

YOLO26s 检测与分类的正式对比统一使用 [`yolo26s_detect.yaml`](../../configs/benchmarks/reference_aligned/yolo26s_detect.yaml) 和 [`yolo26s_classify.yaml`](../../configs/benchmarks/reference_aligned/yolo26s_classify.yaml)，并由 UP `BaseRunner` 训练、验证和保存 checkpoint。旧包装器配置已清理，不再作为受支持的训练入口。

| 任务 | 唯一 benchmark 配置 | 全局 batch | Epoch | 训练精度 |
|---|---|---:|---:|---|
| YOLO26s COCO 检测 | [yolo26s_detect.yaml](../../configs/benchmarks/reference_aligned/yolo26s_detect.yaml) | 128 | 70 | AMP |
| YOLO26s ImageNet 分类 | [yolo26s_classify.yaml](../../configs/benchmarks/reference_aligned/yolo26s_classify.yaml) | 256 | 200 | AMP |

## 环境与数据

分别进入 NVIDIA 或沐曦的独立 Torch 环境，安装 UP 依赖及 YOLO26 辅助依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -r requirements/yolo26.txt
python -m pip install --no-deps -e .
export YOLO_AUTOINSTALL=false
```

配置会校验外部源码摘要。检测使用固定 Ultralytics 源码 revision `557e95b99c24c5b0d4eb5973d5d54a20a799566a`；分类使用公开 `v8.4.0`。检出到稳定目录后设置：

```bash
export YOLO26_DETECT_SOURCE=/opt/ultralytics-yolo26-detect
git clone https://github.com/ultralytics/ultralytics.git "$YOLO26_DETECT_SOURCE"
git -C "$YOLO26_DETECT_SOURCE" checkout --detach 557e95b99c24c5b0d4eb5973d5d54a20a799566a

export YOLO26_CLASSIFY_SOURCE=/opt/ultralytics-yolo26-classify
git clone --branch v8.4.0 --depth 1 https://github.com/ultralytics/ultralytics.git "$YOLO26_CLASSIFY_SOURCE"
```

检测还需要设置：

```bash
export OBJECTS365_WEIGHTS=/data/weights/yolo26s-objv1-150.pt
export COCO_YOLO_YAML=/data/coco.yaml
```

`COCO_YOLO_YAML` 按固定源码中的 COCO YAML 配置本地 `path`，并提供 `images/`、`labels/`、训练/验证列表和 `annotations/instances_val2017.json`。CopyPaste 需要 COCO2017 多边形标签。检测从官方 Objects365 初始化权重开始。

分类设置 `IMAGENET_ROOT` 为 ImageFolder 根目录，包含 `train/<class>/` 和 `val/<class>/`。分类配方使用随机初始化；不得加载已经训练的 ImageNet 分类权重作为初始权重。

## 训练与验证

下面以四卡 NVIDIA 检测为例。沐曦在自己的 Torch 环境中将 `PLATFORM` 改为 `metax_cuda`；调整卡数时保持 YAML 中的全局 batch 不变。

```bash
PLATFORM=nvidia
NG=4
CFG=configs/benchmarks/reference_aligned/yolo26s_detect.yaml
RUN=/path/to/afs/up/$PLATFORM/yolo26s-detect
mkdir -p "$RUN"

python -m up train --config "$CFG" --ng "$NG" --launch pytorch \
  --opts runtime.platform "$PLATFORM" \
    saver.save_dir "$RUN/checkpoints" saver.results_dir "$RUN/results"
```

分类使用同一入口和自己的新输出目录：

```bash
CFG=configs/benchmarks/reference_aligned/yolo26s_classify.yaml
RUN=/path/to/afs/up/$PLATFORM/yolo26s-classify
mkdir -p "$RUN"

python -m up train --config "$CFG" --ng "$NG" --launch pytorch \
  --opts runtime.platform "$PLATFORM" \
    saver.save_dir "$RUN/checkpoints" saver.results_dir "$RUN/results"
```

选择训练生成的最佳 checkpoint，另起进程评估：

```bash
BEST="$RUN/checkpoints/ckpt_best.pth"
python -m up train --evaluate --config "$CFG" --ng "$NG" --launch pytorch \
  --opts runtime.platform "$PLATFORM" saver.auto_resume false \
    saver.resume_model "$BEST" \
    saver.save_dir "$RUN/eval-best/checkpoints" \
    saver.results_dir "$RUN/eval-best/results"
```

独立图片推理使用 UP inference 入口：

```bash
python -m up inference --config "$CFG" --ng 1 \
  --ckpt "$BEST" --img_path /data/example.jpg \
  --vis_dir "$RUN/predictions" \
  --opts runtime.platform "$PLATFORM"
```

归档时将实际 YAML、代码 commit、源码 revision、初始化权重 SHA256、运行日志、评估结果、最佳 checkpoint 和 SHA256 一并放入 AFS/AOSS 的 `up/` 目录。旧的 `python -m up yolo26` 子命令仍保留作兼容入口；新实验不要用旧包装器任务配置或旧 runner。
