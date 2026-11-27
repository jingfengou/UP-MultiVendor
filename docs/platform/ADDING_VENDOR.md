# 厂商接入指南

`RuntimeBackend` 提供设备与执行服务，`NativeOps` 提供专用操作。标准 Torch 操作直接复用厂商实现；不兼容部分在厂商后端处理，不复制模型或 Runner。

寒武纪和昇腾的开发入口与平台验收项见 [寒武纪 MLU](../../up/platform/cambricon_mlu/) 和 [昇腾 NPU](../../up/platform/ascend_npu/)。两家目录按 `runtime.py`、`native_ops.py`、`nms.py` 分工。

四模型正式任务使用 [训练与评测交接指南](FOUR_MODEL_RUNBOOK.md) 中的五份配置和启动命令。

## 厂商接入

### 1. 准备环境

| 平台配置 | 设备 API | 配套库 |
|---|---|---|
| `nvidia` | `torch.cuda` | NVIDIA Torch / TorchVision、CUDA、cuDNN、cuBLAS、NCCL |
| `metax_cuda` | 厂商 Torch 的 `torch.cuda` 兼容 API | MACA Torch / TorchVision、mcDNN、mcBLAS、mcCL 兼容通信接口 |
| `ascend_npu` | `torch.npu` | Torch、torch_npu、CANN、HCCL |
| `cambricon_mlu` | `torch.mlu` | Torch、torch_mlu、CNNL、CNRT、CNCL |

先安装匹配的厂商 Torch、SDK 和驱动，再安装 UP。C500 使用 cuda 名称不表示使用 NVIDIA 二进制。插件只导入当前厂商 SDK，不修改其他环境。

### 2. 适配运行时

父类位于 `up/runtime/backend.py::RuntimeBackend`。设备 API 兼容时直接复用；不兼容时继承父类，只覆盖差异。

| 方法 / 属性 | 接口职责 | NVIDIA 参考 |
|---|---|---|
| `set_device(local_rank)` | 绑定当前进程；不可用或越界报错 | `torch.cuda.set_device` |
| `device` | 返回当前进程的 `torch.device` | `torch.device('cuda', current_device())` |
| `module` | 设备、同步、内存和 RNG 服务 | `torch.cuda` |
| `amp` | autocast、GradScaler 及其状态保存与加载 | `torch.cuda.amp` |
| `memory_available(device, needed_memory, reserve_bytes)` | 输入 byte 数，返回容量是否满足要求 | 总容量减已分配量与保留量 |
| `configure_dnn(options, legacy_nocudnn)` | 映射显式计算库策略；不支持时报错 | cuDNN 与 matmul TF32 开关 |
| `sync_batch_norm(input, weight, bias, mean, variance, eps, momentum, group)` | 归一化、状态更新、反向和指定组通信 | Torch SyncBatchNorm |

`module` 应提供以下服务，内存单位为 byte：

- 设备：`is_available`、`device_count`、`current_device`、`set_device`、`get_device_properties`、`get_device_name`。
- 内存与同步：`memory_allocated`、`max_memory_allocated`、`reset_peak_memory_stats`、`empty_cache`、`synchronize`。
- 随机状态：`manual_seed`、`get_rng_state`、`set_rng_state`。状态只在同厂商环境保存与加载。
- AMP：`autocast(enabled=...)` 与 GradScaler 的 scale / step / update / state_dict / load_state_dict。

NPU / MLU 的显式 DNN 选项目前需要厂商映射；不支持的选项直接报错。不能把空函数或关闭 AMP 当作兼容实现。

`runtime.platform=cambricon_mlu` 或 `ascend_npu` 会直接选中各自 `runtime.py` 的 `Backend` 类。公共启动器和子进程按同一配置选后端，在通信初始化前绑定逻辑卡号。独立分发的后端可由 `PLUGINPATH` 加载，在 `setup_runtime()` 前调用 `up.runtime.register_backend(..., backend_class=自定义类, replace=True)` 覆盖已有厂商。

### 3. 绑定专用操作

契约位于 `up/platform/native.py::NativeOps`；实现位于 `up/platform/<vendor>/native_ops.py`。方法 docstring 列出参数、dtype、布局、坐标、空输入和梯度要求。NVIDIA 参考位于 `up/platform/nvidia/native_ops.py` 与 `up/extensions/csrc/`。

| 操作 | 沐曦 C500 | 昇腾 / 寒武纪 |
|---|---|---|
| `naive_nms`、`overlap`、`sigmoid_focal` | cuBridge 编译原内核 | 显式复用共享 Torch |
| `roi_align`、`psroi_align`、`psroi_pool`、`deform_conv` | cuBridge 编译原内核 | 未绑定，调用时报未实现 |
| `softmax_focal`、`cross_focal` | cuBridge 编译原内核 | 未绑定，调用时报未实现 |
| `bev_iou`、`iou3d`、`nms3d`、`nms3d_normal` | cuBridge 编译原内核 | 未绑定，调用时报未实现 |
| `points_in_boxes`、`roiaware`、`roipoint` | cuBridge 编译原内核 | 未绑定，调用时报未实现 |

只为实际使用的操作补适配；接口存在不等于当前模型会调用它。完整签名及源码位置可在 [接口浏览页](HANDOFF.html) 展开。

```python
from up.platform.native import NativeOps as Contract
from up.platform import torch_native

class NativeOps(Contract):
    naive_nms = staticmethod(torch_native.naive_nms)
    overlap = staticmethod(torch_native.overlap)
    sigmoid_focal = staticmethod(torch_native.sigmoid_focal)
```

上面的共享 Torch 示例用于接入新厂商。SDK 或自写内核在同一类覆盖同名方法。可微操作必须提供实际反向及所需状态；NMS 等不可微操作不用伪造梯度。新内核放在本厂商目录；复用原源码时分别编译，不能覆盖 NVIDIA 扩展。

### 沐曦原生扩展

```text
原 UP 函数 / .apply()
  → metax_cuda.NativeOps
  → 原 Python 包装（参数、反向、ONNX）
  → up.platform.metax_cuda._C
  → cuBridge 编译的 UP C++/CUDA 内核 → MACA
```

在 MACA Torch 与配套 SDK 环境中执行：

```bash
UP_BUILD_PLATFORM=metax_cuda MAX_JOBS=2 python setup.py build_ext --inplace
python -m up train --config configs/det/yolox/yolox_small.yaml \
  --opts runtime.platform metax_cuda
```

- 构建读取 `up/extensions/csrc/`，不修改原内核算法；NVIDIA 仍生成 `up.extensions._C`。
- 沐曦仅在编译 `roi_align/roi_align_kernel.cu` 时使用 `--fmad=false`，避免 FP16 乘加融合引起不同的中间舍入。其他源文件与 NVIDIA 的编译参数不变。升级后重新执行上面的构建命令即可。
- 构建时核对 Torch 厂商，沐曦必须提供 cuBridge。临时文件放在本次构建目录，支持普通用户容器。普通 Python 安装不自动编译。
- `NativeOps` 的 16 项绑定在 `up/platform/metax_cuda/native_ops.py`，完整签名见父类。旧包装内部分 `_nvidia_*` 名称保留以兼容原代码；实际二进制由 `up/extensions/ext.py` 按厂商加载。
- C500 的训练、验证和推理在首次绑定设备时检查完整二进制；未编译、版本不完整或加载失败会直接报错，不回退到旧 Torch 专用算子实现。`box_overlap`、`sigmoid_focal_loss` 兼容入口也进入 C500 原生接口。
- 所有 C500 运行环境先执行上述构建。ResNet50 / ViT 的标准卷积、BN、矩阵乘法仍由厂商 Torch 执行，UP 的 16 项专用操作和连续坐标 NMS 则使用这份独立二进制。
- 保留原支持范围：RoIAlign 的 MAX 分支未导出，RoIPointPool3d 无反向；原 NMS 内核的 CPU 掩码选择步骤也保持原样。

已用 A100（Torch `2.10.0+cu126`）和 C500（MACA Torch `2.10.0+metax3.8.1.0`、SDK `3.8.1.3`）对照相同输入与上游梯度，覆盖 16 项接口、29 组用例，训练与推理各重复 3 次。浮点容差为 `rtol=1e-4, atol=1e-5`，索引精确比较。

| 操作 | 跨卡最大绝对差：前向 / 梯度 | 结果 |
|---|---|---|
| `roi_align` | FP16：`0` / `4.88e-4`；FP32/FP64：最大 `5.96e-8` / `0` | 前向通过；FP32/FP64 反向通过；FP16 反向超差 |
| `psroi_align` | `2.98e-8` / `0` | 通过 |
| `psroi_pool` | `0` / `0` | 通过 |
| `deform_conv` | `1.91e-6` / `1.91e-6` | 通过 |
| `naive_nms` | 索引一致 / 无反向 | 通过 |
| `sigmoid_focal` | `5.96e-8` / `1.49e-8` | 通过 |
| `softmax_focal` | `1.19e-7` / `0` | 通过 |
| `cross_focal` | `0` / `1.86e-9` | 通过 |
| `overlap` | `0` / 无反向 | 通过 |
| `bev_iou` | `0` / 无反向 | 通过 |
| `iou3d` | `0` / `0` | 通过原有高度、体积梯度 |
| `nms3d` | 索引一致 / 无反向 | 通过 |
| `nms3d_normal` | 索引一致 / 无反向 | 通过 |
| `points_in_boxes` | 索引一致 / 无反向 | 通过 |
| `roiaware` | `0` / `0` | max、avg 通过 |
| `roipoint` | `0` / 无反向 | 通过 |

RoIAlign 另外检查了 26 组 FP16 输入，包括两种对齐方式、固定/自适应采样、边界、小框及重叠框：使用新构建的完整扩展，经 UP 训练和推理入口各重复 3 次，前向均与 A100 相同。8 组 FP32/FP64 控制用例的前向也通过原容差。

FP16 反向仍使用原生 `atomicAdd` 并行累加。多个线程写入同一像素时，加法顺序会变；半精度舍入使两家各自重复运行和跨卡对比都可能超差。本次扩展用例跨卡梯度最大差 `0.01953125`。独立 FP64 参考检查没有发现某一家始终更准确，因此保留并行反向，不改为串行计算，也不放宽容差。需要已验证的更高精度路径时可显式选择 FP32。

在 C500 上对四种 RoIAlign 工作量做交替计时，关闭乘加融合后的前向耗时中位数变化为 `-0.24%` 至 `+1.39%`，未观察到明显降速。这是算子计时，不是整网训练测速。DeformConv 也有原生反向波动，既有重复检查最大 `2.38e-6`，在原容差内。16 项接口的检查范围还包括连续布局、RoIAlign 与 overlap 空输入、DCN groups=1/2、NMS offset=0/1；新增形状或变体按父类契约验证。

移植保留了内核计算公式，补了两处设备适配：PSRoIPool 在输入设备直接分配连续 int32 工作区；RoIAlign FP64 反向使用 256 线程块，避免原 1024 线程块在 A100 上因寄存器不足无法启动。FP16/FP32 线程块大小保持原样。

YOLOX、YOLO11 的普通检测 NMS 入口是 `up/platform/ops.py::nms(boxes, scores, threshold)`：

- NVIDIA 调用 TorchVision 的 GPU NMS。
- C500 调用 `up/platform/metax_cuda/nms.py`，用当前厂商 Torch 排序，再由 `up/extensions/csrc/nms/continuous_nms.cu` 在 GPU 上计算重叠和完成筛选。按上面的 `UP_BUILD_PLATFORM=metax_cuda` 命令重新编译后使用。
- 昇腾和寒武纪分别路由到各自目录的 `nms.py`。现有绑定使用共享 Torch 实现，厂商可在此替换为符合契约的内核。

输入为连续坐标 `boxes[N,4]` 和同设备、同 dtype 的 `scores[N]`，输出为同设备的 int64 原框索引。按分数降序保留框，IoU **大于**阈值时抑制，无 `+1` 坐标偏移，也无反向。C500 保留原厂商排序对同分框的处理，支持 FP16、FP32、FP64；缺少已编译扩展时明确报错。这个入口与上表带 `offset` 的旧 `naive_nms` 分开。

这些专用内核不替换 ResNet50 的 Torch 卷积和 BN。

### 4. 增加模型或操作

- Torch 已支持的操作：复用，核对新形状、dtype、前向和反向。
- 已有平台接口：复用调用；未支持的变体再补厂商实现。
- 通用 Torch 可组合的操作：维护一份共享实现，各厂商显式绑定。
- 专用内核：先定义统一契约和参考结果，再实现厂商内核。

接入时核对前向、反向和状态，然后运行训练、训练内验证、保存加载、独立推理与多卡通信。保持原算子容差，缺失能力明确报错，不新增静默 CPU 代算。没有 PyTorch 设备后端的厂商还需完成设备注册、Tensor 存储、拷贝、派发、stream 与 autograd 桥接。

### 5. 寒武纪与昇腾验收

在匹配的厂商环境以及仓库根目录执行。将 `PLATFORM` 改为本次验收的平台。

```bash
PLATFORM=cambricon_mlu  # 或 ascend_npu
PLATFORM="$PLATFORM" python - <<'PY'
import os
import torch
from up.runtime import setup_runtime

rt = setup_runtime({'runtime': {'platform': os.environ['PLATFORM']}})
rt.set_device(0)
value = torch.ones(2, device=rt.device, requires_grad=True)
value.square().sum().backward()
rt.module.synchronize()
torch.testing.assert_close(value.grad.cpu(), torch.full((2,), 2.0))
print(type(rt).__module__, rt.device, rt.dist_backend, torch.__version__)
print(rt.native_ops.capabilities())
PY
```

`capabilities()` 记录算子绑定情况。随后在配好数据和独立输出目录的 YAML 上验证公共训练入口。下方的 `res50.yaml` 和 `yolox_small.yaml` 是验证平台设备、算子和公共 Runner 的通用 smoke-test 配置，不是当前正式精度对比配方；跨厂商 benchmark 应使用[四模型训练与评测交接](FOUR_MODEL_RUNBOOK.md)表格中的唯一主 YAML：

```bash
python -m up train --ng 1 --config configs/cls/resnet/res50.yaml \
  --opts runtime.platform "$PLATFORM"
python -m up train --ng 1 --config configs/det/yolox/yolox_small.yaml \
  --opts runtime.platform "$PLATFORM"
```

1. 固定初始化、输入和随机种子，记录 ResNet50 与 YOLOX-S 的单卡 FP32 前后向和优化器更新。YOLOX-S 验证 OTA `overlap`、常规 NMS 及输出索引所在设备。
2. 按目标训练配置验证 AMP 溢出跳步、恢复 scaler，并用 `--ng 2` 检查双卡 DDP、集合通信及配置用到的 SyncBN。
3. 用同一份 checkpoint 测试恢复训练、完整验证和独立推理。比较精度时保持官方配方与全局 batch 一致，记录提交、设备型号、Torch/SDK/驱动版本和实测结果。

[架构说明](ARCHITECTURE.md) · [训练指南](TRAINING.md)
