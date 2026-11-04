"""Continuous-coordinate YOLO NMS using the separately compiled MACA binary."""
import torch


def nms(boxes, scores, threshold):
    """Return same-device int64 indices in descending score order, without gradients.

    boxes[N,4] contains xyxy coordinates, scores[N] contains matching scores.
    Keep the previous vendor Torch sort, including its handling of equal scores.
    Only the pairwise IoU and greedy suppression move into the native extension.
    This entry has no coordinate offset and is separate from legacy naive_nms.
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError('NMS boxes must have shape [N, 4]')
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise ValueError('NMS scores must have shape [N]')
    if boxes.device != scores.device or boxes.device.type != 'cuda':
        raise ValueError('C500 NMS requires boxes and scores on the same GPU')
    if boxes.dtype not in (torch.float16, torch.float32, torch.float64):
        raise TypeError('C500 NMS supports float16, float32 and float64 boxes')
    if boxes.dtype != scores.dtype:
        raise TypeError('NMS boxes and scores must have the same dtype')
    from up.extensions.ext import load_extension
    extension = load_extension('metax_cuda')
    with torch.no_grad():
        order = scores.argsort(descending=True)
        # Match the old Torch scalar comparison, including FP16 threshold rounding.
        keep = extension.naive_nms.continuous_mask(boxes[order].contiguous(), threshold)
        return order[keep]
