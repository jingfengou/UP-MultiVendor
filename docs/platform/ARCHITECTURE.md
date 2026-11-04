# 架构

基于 [UP 52aaeb9](https://github.com/ModelTC/United-Perception/tree/52aaeb9ed27a490820696d7754d19f98cd6ecd11)，提供 ResNet50、ViT-B/16、YOLOX-S、YOLO11s、DETR-R50 的训练、验证、推理与 ONNX 导出。

```text
config / CLI → setup_runtime(platform, purpose)
                 ├─ 设备、AMP、通信、计算库策略
                 └─ NativeOps：当前厂商专用操作
UP 模型
  ├─ nn / functional / Tensor / autograd → 厂商 Torch
  └─ 原专用函数 / .apply() → NativeOps → 厂商实现
```

平台配置选择当前环境里的实现，不替换已安装的 Torch。同一进程初始化后不切换厂商或用途。标准 Torch 操作直接使用厂商设备后端，专用 CUDA 扩展单独适配。

## 公共流程

### 代码目录

| 位置 | 职责 |
|---|---|
| `up/commands/` | 解析配置，启动训练、推理或导出 |
| `up/runtime/backend.py` | `RuntimeBackend`、厂商注册、设备与运行时服务 |
| `up/runtime/state.py` | 临时进入验证模式并恢复各子模块状态 |
| `up/platform/native.py` | 专用操作契约、能力查询与工厂 |
| `up/platform/<vendor>/native_ops.py` | 各厂商显式绑定 |
| `up/platform/torch_native.py` | 共享 Torch NMS、overlap、Sigmoid Focal |
| `up/platform/ops.py` | 检测适配器使用的连续坐标 NMS |
| `up/extensions/python/` | 保留原函数和 `.apply()` 入口 |
| `up/extensions/csrc/` | 原 C++/CUDA 源码；NVIDIA 用 CUDA、沐曦用 cuBridge 显式编译 |
| `up/runner/`、`up/apis/` | 训练编排、验证和独立推理 |
| `up/tasks/det/plugins/yolo11/` | 官方计算与 UP 数据、训练、推理的连接 |
| `up/utils/general/toonnx_helper.py` | ONNX 导出及可选数值核对 |

### 训练与验证

1. `train` 读取配置并设置 `purpose=training`；worker 按 `local_rank` 绑卡并初始化通信。
2. Runner 创建 DataLoader、模型、AMP、优化器、调度器和 DDP。读取、解码及普通增强仍在数据管线中执行。
3. `batch2device() → to_device()` 递归迁移 Tensor，图片和标签分别保持所需 dtype。
4. 执行前向、损失、反向和优化器更新，硬件差异由运行时或专用操作处理。
5. 验证临时进入 `eval/no_grad`，结束后恢复子模块状态。`train --evaluate` 仍使用训练用途。

YOLO11 的增强、损失、EMA 和参数分组复用固定版本的 Ultralytics。专用 Runner 处理学习率、DDP、检查点和 COCO 结果汇总。配方与命令见 [训练指南](TRAINING.md)。

### 独立推理

`commands/inference.py → BaseInference` 设置 `purpose=inference`，加载模型与检查点。图片预处理后上卡，模型输出在评估或可视化边界下卡。检测输出为 `[batch_id,x1,y1,x2,y2,score,class_id]`。

### ONNX 输出

| 模型 | 输出 |
|---|---|
| ResNet50 / ViT-B/16 | `scores`；是否添加 softmax 由配置控制 |
| YOLO11s | `blobs_predictions`，`[B,84,N]`，像素 xywh 与 80 类概率 |
| YOLOX-S | 三层各自的 cls / box / obj Tensor，未解码、未 NMS |
| DETR-R50 | `blobs_boxes`：归一化 cxcywh；`blobs_logits`：类别与 no-object logits |

导出不创建训练数据加载器，使用模型副本在 CPU / FP32 执行，固定 batch=1 和输入尺寸。检查点含 EMA 时优先导出 EMA。检测 NMS 位于图外；`--verify` 使用 ONNX Runtime 核对输出，已有文件不会被覆盖。

### 来源与许可

- UP 和保留的 NVIDIA 扩展沿用 [Apache-2.0](../../LICENSE)。
- YOLO11 使用 `ultralytics==8.3.225`，遵循其 [AGPL-3.0 许可](https://github.com/ultralytics/ultralytics/blob/v8.3.225/LICENSE)。
- DETR 源码固定为 `29901c51d7fe8712168b8d0d64351170bc0f83e0`，许可证见 [official/LICENSE](../../up/tasks/det/plugins/detr/official/LICENSE)。

[训练指南](TRAINING.md) · [厂商接入](ADDING_VENDOR.md)
