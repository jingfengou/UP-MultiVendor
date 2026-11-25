# Reference-aligned training

All recipes in `configs/benchmarks/reference_aligned/` use UP's existing
`BaseRunner`. UP owns batch iteration, backward, optimizer updates, validation
iteration, checkpointing and image inference. The YOLO26 and DETR plugins reuse
the fixed official network, loss, augmentation and metric components, not their
training loops.

## Plugin layout

YOLO26 and DETR follow the same plugin structure as UP's YOLOX:

```text
up/tasks/det/plugins/
  yolo26/                      # Shared detection/classification components
    models/                    # Official network/loss, adapted to UP dictionary batches
    data/                      # Dataset, augmentation and per-batch metrics
    utils/
      optimizer_helper.py      # UP BaseOptimizer plus official parameter groups/MuSGD
      lr_helper.py             # Batch warmup and epoch LR
      ema_helper.py            # Official EMA through UP's EMA registry
      hook_helper.py           # Mosaic, progressive loss and epoch transitions
      seed_helper.py           # Official seed and deterministic-algorithm settings
      inference_helper.py      # Image preprocessing and output conversion
      source_helper.py         # Fixed official source selection
  detr/
    models/                    # Official network, matching and auxiliary losses
    data/                      # Official COCO transforms, sampler and metrics
    utils/                     # Source selection and image-inference adapter
```

Each plugin registers its components with UP's existing registries; the YAML
selects them by name. `BaseRunner` owns the training/validation loop and
`BaseInference` owns image inference. DETR reuses UP's `set_random_seed` with
`rank_init: true`, `BaseOptimizer` with a backbone parameter group, and the base
StepLR builder stepped at epoch end. YOLO26 subclasses `BaseOptimizer`; only its
model-specific recipe rules stay in its helpers.

The earlier `up.tasks.yolo26` command remains available for existing scripts.
The recipes below select the registered UP plugins through the common commands.

## Recipes

| Configuration | Reference | Dataset | Epochs | Four GPUs: batch per GPU / total | Precision |
| --- | --- | --- | --- | --- | --- |
| `resnet50.yaml` | UP `52aaeb9e`, ResNet50 | ImageNet-1K | 100 | 256 / 1024 | FP32 |
| `yolox_s.yaml` | UP `52aaeb9e`, YOLOX-S | COCO2017 | 300 | 8 / 32 | FP32 |
| `yolo26s_detect.yaml` | Fixed YOLO26 detection source and released Objects365 initialization | COCO2017 | 70 | 32 / 128 | Official AMP |
| `yolo26s_classify.yaml` | Public Ultralytics `v8.4.0` classification source | ImageNet-1K | 200 | 64 / 256 | Official AMP |
| `detr_r50.yaml` | Official `29901c51`, 150-epoch recipe | COCO2017 | 150 | 4 / 16 | FP32 |

ResNet50 changes the original YAML's batch per GPU from 64 to 256 to retain total
batch 1024 on four GPUs. Its ordinary BN still uses each GPU's local batch.
DETR changes the reference eight GPUs times batch two to four GPUs times batch
four; validation stays at batch two per GPU. YOLOX retains its per-GPU batch.
YOLO26's published device field does not establish the original physical GPU
count; the recipes retain its published total batch.

Classification uses the public `v8.4.0` implementation with explicit SGD and
initial LR `0.1`, matching the released checkpoint's learning-rate log.
Do not select `optimizer: auto`: this version would replace the requested LR
with `0.01` for this training budget. Momentum remains `0.9` with Nesterov.
Its three groups contain ordinary weights (decay `0.0001`), normalization
weights (no decay) and biases (no decay).

The released classification checkpoint's logs contain four LR columns; it
retains neither optimizer state nor the original training commit. Its metadata
reports version `8.3.100`. This configuration matches its initial LR while
keeping the public `v8.4.0` three-group optimizer and cosine schedule.
The published weight's Top-1 is `75.982%` (rounded to `76.0%`).

## What stays with each reference

- ResNet50 and YOLOX use UP's original `set_random_seed`, initialization order,
  FP32, optimizer, learning-rate schedule and validation frequency. YOLOX retains
  its original BN averaging before scheduled validation.
- YOLO26 uses its fixed official seed initialization, AMP check, optimizer
  grouping, batch warmup, accumulation, EMA and epoch-state updates. These are
  registered UP components, not an embedded official Trainer.
- DETR keeps seed `42 + rank`, FrozenBN, original COCO category IDs, matching and
  auxiliary losses, AdamW groups, clipping and epoch-end StepLR.
- DNN flags are set only where the reference sets them. The effective library
  defaults are recorded separately from the requested YAML.

`runtime.audit_recipe: true` writes each rank's effective seed/RNG snapshots,
model-initialization hash, AMP/scaler state, BN/DDP settings, named optimizer
groups and loader settings to its new checkpoint directory.

## Run

Use the appropriate isolated vendor environment. Bind the fixed official source
and input paths required by a recipe; missing environment bindings fail early.
The source digest in each YAML prevents silently importing a different version.
YOLO26 COCO validation also needs `faster-coco-eval==1.6.7` in that environment:

```bash
python -m pip install faster-coco-eval==1.6.7
```

```bash
export YOLO26_DETECT_SOURCE=/opt/official/ultralytics-557e95b9
export OBJECTS365_WEIGHTS=/data/weights/yolo26s-objv1-150.pt
export COCO_YOLO_YAML=/data/coco-yolo.yaml

python -m up train \
  --config configs/benchmarks/reference_aligned/yolo26s_detect.yaml \
  --ng 4 --launch pytorch --backend dist \
  --opts runtime.platform nvidia \
    saver.save_dir outputs/my-new-run/checkpoints \
    saver.results_dir outputs/my-new-run/results
```

For C500, run the same command in its vendor environment with
`runtime.platform metax_cuda`. Select the allocated GPUs through
`CUDA_VISIBLE_DEVICES` before starting the process. Never mix vendor Torch
packages in one Python environment.

Training, validation and image inference use UP's common commands. Change
`CONFIG` to select a model and `PLATFORM` to select a vendor; do not call the
older model-specific official training wrappers for these recipes.

```bash
CONFIG=configs/benchmarks/reference_aligned/yolo26s_detect.yaml
PLATFORM=nvidia                 # C500: metax_cuda, in its own vendor environment
RUN=/data/outputs/my-new-run    # New directory; keep existing runs unchanged

python -m up train --config "$CONFIG" --ng 4 \
  --opts runtime.platform "$PLATFORM" \
    saver.save_dir "$RUN/checkpoints" saver.results_dir "$RUN/results"

# After a checkpoint is saved, select its actual filename.
CHECKPOINT=/data/outputs/my-new-run/checkpoints/ckpt.pth
python -m up train --evaluate --config "$CONFIG" --ng 4 \
  --opts runtime.platform "$PLATFORM" saver.resume_model "$CHECKPOINT" \
    saver.save_dir "$RUN/standalone-eval/checkpoints" \
    saver.results_dir "$RUN/standalone-eval/results"

python -m up inference --config "$CONFIG" --ng 1 \
  --ckpt "$CHECKPOINT" --img_path /data/images --vis_dir "$RUN/predictions" \
  --opts runtime.platform "$PLATFORM"
```

All commands need the same source/data/initial-weight environment bindings used
when constructing that model. Validation computes dataset metrics; image
inference reads images and writes visualized predictions. YOLO26 and DETR select
their registered inference adapters through `runtime.inferencer` in the YAML,
not through a separate model-specific command.

For ResNet50 and YOLOX, bind their existing UP dataset paths in the YAML or with
`--opts`; the recipes retain the original directory conventions. DETR requires
`DETR_SOURCE` and `COCO_ROOT`. Classification requires
`YOLO26_CLASSIFY_SOURCE` and `IMAGENET_ROOT`.

Each run must have a new output directory. `auto_resume: false` and
`require_empty: true` protect existing checkpoints. Resume an existing run only
by explicitly selecting its checkpoint and preserving its configuration.
These checkpoints restore model, optimizer, scheduler, AMP scaler, EMA and
model-specific training state. Resume YOLO26 and DETR at an epoch boundary.
They use full-state loading; load only checkpoints from a trusted source.
