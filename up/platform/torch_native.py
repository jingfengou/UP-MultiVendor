"""我们维护的共享 Torch 实现；各厂商文件显式绑定后在输入设备上执行。

负责人先验证本卡 Torch 的前向/反向、边界及性能，不用复制本文件。
需要专用 SDK 优化时，在自己 native_ops.py 的同名方法替换绑定，仍遵守
up/platform/native.py 的契约；不要把公共数学实现改成某一家 SDK 调用。
"""
import torch
from torch.nn import functional as F


def naive_nms(dets, thresh, *, offset):
    """910B/MLU590 的共享 offset-NMS；C500 已绑定独立原生二进制。

    dets[N,5] 为已排好序的框/分数，offset 为0或1；返回同设备 int64 索引[K]。
    保留原 FP32 转换、输入顺序和严格 > 阈值，无反向要求。
    NVIDIA 参考: up/extensions/python/nms.py::_nvidia_naive_nms。
    常规检测连续坐标 NMS 在 ops.py，不经过本函数。
    """
    if dets.ndim != 2 or dets.shape[1] != 5 or offset not in (0, 1):
        raise ValueError('naive_nms expects [N,5], offset=0 or 1')
    dets = dets.float()  # Original UP casts NMS operands to FP32 before the kernel.
    # UP's native entry receives boxes sorted by its caller. Retain input order.
    order = torch.arange(len(dets), device=dets.device)
    keep = []
    area = (dets[:, 2:4] - dets[:, :2] + offset).prod(-1)
    while order.numel():
        current = order[0]
        keep.append(current)
        rest = order[1:]
        lt = torch.maximum(dets[current, :2], dets[rest, :2])
        rb = torch.minimum(dets[current, 2:4], dets[rest, 2:4])
        intersection = (rb - lt + offset).clamp(min=0).prod(-1)
        iou = intersection / (area[current] + area[rest] - intersection)
        order = rest[~(iou > thresh)]
    return torch.stack(keep).long() if keep else dets.new_empty((0,), dtype=torch.long)


def overlap(b1, b2, mode='IoU'):
    """910B/MLU590 的共享框重叠实现；C500 的 YOLOX 使用原生内核。

    b1[N,>=4]/b2[M,>=4] -> 同 b1 设备/dtype 的 [N,M]；任一空输入返回 [0]。
    mode 为 IoU/IoF/IoS，保留 +1 坐标和分母最小值1；原操作不提供几何梯度。
    NVIDIA 参考: up/extensions/python/iou_overlap.py::_nvidia_gpu_iou_overlap。
    验证三个 mode 和空输入后再认定本卡通过；不是直接换成连续坐标 IoU。
    """
    if mode not in ('IoU', 'IoF', 'IoS'):
        raise ValueError('overlap mode must be IoU, IoF or IoS')
    with torch.no_grad():  # Original matching/geometry extension has no backward.
        if b1.numel() == 0 or b2.numel() == 0:
            return b1.new_zeros((0,))
        lt = torch.maximum(b1[:, None, :2], b2[None, :, :2])
        rb = torch.minimum(b1[:, None, 2:4], b2[None, :, 2:4])
        intersection = (rb - lt + 1).clamp(min=0).prod(-1)
        a = (b1[:, 2:4] - b1[:, :2] + 1).prod(-1)[:, None]
        b = (b2[:, 2:4] - b2[:, :2] + 1).prod(-1)[None, :]
        denominator = a + b - intersection if mode == 'IoU' else a if mode == 'IoF' else b
        return intersection / denominator.clamp(min=1)


def sigmoid_focal(preds, targets, weight_pos, gamma, alpha, num_classes, reduction):
    """已提供的 Sigmoid Focal；训练梯度由厂商 Torch autograd 计算。

    preds[N,C] logits；targets[N] 中0为背景、1..C为前景、-1忽略；
    weight_pos 为单元素 normalizer；gamma/alpha 为 Focal 参数，num_classes=C。
    none 返回 float32 [N,C]，sum/mean 沿用原接口的求和结果 [1]。
    只要求 preds 梯度；负责人验证 loss/梯度和忽略类别，不必另写 backward。
    NVIDIA 参考: up/extensions/python/focal_loss.py::SigmoidFocalLossFunction。
    """
    if preds.shape[-1] != num_classes or reduction not in ('none', 'sum', 'mean'):
        raise ValueError('Invalid focal class count or reduction')
    classes = torch.arange(1, num_classes + 1, device=preds.device)
    positive = targets[:, None] == classes
    p = preds.sigmoid()
    pt = torch.where(positive, p, 1 - p)
    weight = torch.where(positive, alpha, 1 - alpha)
    loss = F.binary_cross_entropy_with_logits(preds, positive.to(preds), reduction='none')
    loss = loss * weight * (1 - pt).pow(gamma) * (targets[:, None] >= 0)
    loss = loss / weight_pos.detach().clamp(min=1)
    return loss.float() if reduction == 'none' else loss.sum().float().reshape(1)
