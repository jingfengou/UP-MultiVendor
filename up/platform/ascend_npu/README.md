# 昇腾 NPU 适配

该目录供昇腾负责人接入和验收。目标环境使用配套的 PyTorch、`torch_npu`、CANN 和 HCCL，设备名为 `npu`。当前状态为待真机验收。

| 修改位置 | 负责的接口 |
| --- | --- |
| [runtime.py](runtime.py) | 继承 `RuntimeBackend`，按需覆盖绑卡、显存/RNG、AMP、CANN 策略和 SyncBN。默认使用 `torch.npu`，通信后端为 HCCL。 |
| [native_ops.py](native_ops.py) | 按 `NativeOps` 契约绑定专用算子。`overlap`、`naive_nms`、`sigmoid_focal` 已绑定共享 Torch 实现。其余 13 项的签名、反向要求及 NVIDIA 参考见代码注释。 |
| [nms.py](nms.py) | YOLOX 等检测模型的连续坐标 NMS。核对阈值边界、同分框、空输入和返回的本卡 int64 索引。 |

```yaml
runtime:
  platform: ascend_npu
```

`runtime.platform` 已接入公共训练、验证和推理入口。910B 上先跑设备探针和单卡前后向，再验 HCCL 双卡同步及 checkpoint 恢复。若 SDK 缺少已启用的算子，可在本目录绑定兼容接口或添加独立编译的源码，保留公共参数、输出与梯度契约。

[公共接口和验收命令](../../../docs/platform/ADDING_VENDOR.md) · [专用算子契约](../native.py) · [torch_npu](https://github.com/Ascend/pytorch)
