# 保留原调用名，集中由平台决定使用 TorchVision 还是同设备 Torch 实现。
from up.platform.ops import nms as t_nms


def nms(cls_rois, nms_cfg={}):
    thresh = nms_cfg.get('nms_iou_thresh', 0.5)
    idx = t_nms(cls_rois[:, :4], cls_rois[:, 4], thresh)
    return cls_rois[idx], idx
