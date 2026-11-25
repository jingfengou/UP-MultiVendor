"""Shared detection helpers; complete native operators live behind NativeOps."""
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
    昇腾和寒武纪从各自的 nms.py 进入；默认绑定共享 Torch 实现。
    旧 NativeOps.naive_nms 带 offset，与此接口分开。
    """
    platform = get_runtime().platform
    if platform == 'nvidia':
        from torchvision.ops import nms as torchvision_nms
        return torchvision_nms(boxes, scores, threshold)
    if platform == 'metax_cuda':
        from .metax_cuda.nms import nms as platform_nms
    elif platform == 'ascend_npu':
        from .ascend_npu.nms import nms as platform_nms
    elif platform == 'cambricon_mlu':
        from .cambricon_mlu.nms import nms as platform_nms
    else:
        from .torch_native import continuous_nms as platform_nms
    return platform_nms(boxes, scores, threshold)
