# Reference-aligned training

For the five tasks in the vendor handoff, use the configuration and input
table in [Four-model training and evaluation](FOUR_MODEL_RUNBOOK.md).

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
| `resnet50_fair_native.yaml` | FAIR recipe with UP ResNet50 | ImageNet-1K | 90 | 64 / 256 | FP32 |
| `yolox_s_megvii_native.yaml` | Megvii recipe with UP YOLOX-S | COCO2017 | 300 | 16 / 64 | AMP |
| `yolo26s_detect.yaml` | Fixed YOLO26 detection source and released Objects365 initialization | COCO2017 | 70 | 32 / 128 | Official AMP |
| `yolo26s_classify.yaml` | Public Ultralytics `v8.4.0` classification source | ImageNet-1K | 200 | 64 / 256 | Official AMP |
| `detr_r50.yaml` | Official `29901c51`, 150-epoch recipe | COCO2017 | 150 | 8 / 32 | FP32 |

`resnet50.yaml` and `yolox_s.yaml` retain the original UP baselines at global
batches 1024 and 32 respectively. The FAIR and Megvii YAMLs include their UP
network definitions and select the reference training settings and registered
recipe components. DETR uses global batch 32 and validation batch two per GPU.
[Original DETR batch-size confirmation](https://github.com/facebookresearch/detr/issues/61#issuecomment-640496198)

YOLO26's published device field does not establish the original physical GPU
count; the recipes retain its published total batch.

Classification explicitly selects public `v8.4.0` MuSGD at initial LR `0.1`
with a cosine schedule and four base parameter groups. Hybrid updates in this
version use Muon and SGD coefficients of `0.5` each. The released checkpoint
records version `8.3.100` and different argument values for the coefficients.
Its original source commit and optimizer state are unavailable.
[YOLO26 paper, Table S1](https://arxiv.org/html/2606.03748v1#S1.T1) ·
[MuSGD in v8.4.0](https://github.com/ultralytics/ultralytics/blob/v8.4.0/ultralytics/optim/muon.py)

## What stays with each reference

- The original UP ResNet50 and YOLOX baselines use UP's `set_random_seed`, initialization order,
  FP32, optimizer, learning-rate schedule and validation frequency. YOLOX retains
  its original BN averaging before scheduled validation.
- The FAIR and Megvii recipes use seed `0` and select their own training
  augmentations, learning-rate settings and validation schedules.
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

For FAIR ResNet50 and Megvii YOLOX, set the `UP_*` environment bindings in the
handoff guide. The original UP baselines retain their dataset path keys. DETR requires
`DETR_SOURCE` and `COCO_ROOT`. Classification requires
`YOLO26_CLASSIFY_SOURCE` and `IMAGENET_ROOT`.

Each run must have a new output directory. `auto_resume: false` and
`require_empty: true` protect existing checkpoints. Resume an existing run only
by explicitly selecting its checkpoint and preserving its configuration.
These checkpoints restore model, optimizer, scheduler, AMP scaler, EMA and
model-specific training state. Resume YOLO26 and DETR at an epoch boundary.
They use full-state loading; load only checkpoints from a trusted source.
