"""Cambricon binding for the continuous-coordinate detection NMS."""

from up.platform.torch_native import continuous_nms


def nms(boxes, scores, threshold):
    """Return same-device int64 indices into xyxy boxes[N,4], without gradients.

    Preserve descending scores, the strict IoU > threshold rule and empty
    input. Bind an MLU kernel here after validating its dtype and tie behavior.
    """
    return continuous_nms(boxes, scores, threshold)
