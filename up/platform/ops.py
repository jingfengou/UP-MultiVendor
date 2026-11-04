"""Shared detection helpers; complete native operators live behind NativeOps."""
import torch
from up.runtime import get_runtime


def box_overlap(b1, b2, mode='IoU'):
    """Route the compatibility entry to C500's native overlap, preserving +1 coordinates."""
    runtime = get_runtime()
    if runtime.platform == 'metax_cuda':
        return runtime.native_ops.overlap(b1, b2, mode)
    from .torch_native import overlap
    return overlap(b1, b2, mode)


def sigmoid_focal_loss(preds, targets, weight_pos, gamma, alpha, num_classes, reduction):
    """Use C500's native forward/backward even through the compatibility entry."""
    runtime = get_runtime()
    if runtime.platform == 'metax_cuda':
        return runtime.native_ops.sigmoid_focal(
            preds, targets, weight_pos, gamma, alpha, num_classes, reduction)
    from .torch_native import sigmoid_focal
    return sigmoid_focal(preds, targets, weight_pos, gamma, alpha, num_classes, reduction)


def nms(boxes, scores, threshold):
    """YOLO11/YOLOX 实际使用的连续坐标 NMS；公共调用入口由我们维护。

    boxes[N,4] 为 xyxy、scores[N] 为分数、threshold 为 IoU 阈值；
    返回输入设备上的 int64 原框索引[K]，无反向要求。这里没有 +1 offset。
    NVIDIA 使用 torchvision.ops.nms，C500 使用 MACA 编译的连续坐标 NMS。
    其他厂商使用下面的共享 Torch 实现。旧 NativeOps.naive_nms 带 offset，
    与此接口分开，不能直接替换。
    """
    platform = get_runtime().platform
    if platform == 'nvidia':
        from torchvision.ops import nms as torchvision_nms
        return torchvision_nms(boxes, scores, threshold)
    if platform == 'metax_cuda':
        from .metax_cuda.nms import nms as metax_nms
        return metax_nms(boxes, scores, threshold)

    # Same-device reference path. No CUDA binary or implicit CPU fallback.
    order = scores.argsort(descending=True)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while order.numel():
        index = order[0]
        keep.append(index)
        remaining = order[1:]
        lt = torch.maximum(boxes[index, :2], boxes[remaining, :2])
        rb = torch.minimum(boxes[index, 2:], boxes[remaining, 2:])
        wh = (rb - lt).clamp(min=0)
        intersection = wh[:, 0] * wh[:, 1]
        iou = intersection / (area[index] + area[remaining] - intersection)
        order = remaining[~(iou > threshold)]
    return torch.stack(keep) if keep else order
