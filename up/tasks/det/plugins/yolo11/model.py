"""UP batch adapter for the official Ultralytics YOLO11s, not a renamed YOLOX.

The architecture, assignment and losses come from the pinned optional package.
Only labels, coordinates and UP's dictionary I/O are converted here.
"""
from types import SimpleNamespace

import torch
from torch import nn

from up.platform.ops import nms
from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY


@MODULE_ZOO_REGISTRY.register('yolo11s')
class Yolo11ForUP(nn.Module):
    def __init__(self, num_classes=81, score_threshold=0.25, nms_threshold=0.7,
                 max_detections=300, loss_gains=None, multi_label=False):
        super().__init__()
        # Lazy optional import: ResNet, ViT and DETR don't require Ultralytics.
        import ultralytics
        if ultralytics.__version__ != '8.3.225':
            raise RuntimeError('YOLO11 adapter requires ultralytics==8.3.225')
        from ultralytics.nn.tasks import DetectionModel
        from ultralytics.utils import DEFAULT_CFG_DICT
        self.detector = DetectionModel('yolo11s.yaml', nc=num_classes - 1, verbose=False)
        options = dict(DEFAULT_CFG_DICT)
        if loss_gains:
            if set(loss_gains) - {'box', 'cls', 'dfl'}:
                raise ValueError('loss_gains accepts box, cls and dfl')
            options.update(loss_gains)
        self.detector.args = SimpleNamespace(**options)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.multi_label = multi_label
        self.num_classes = num_classes
        self.deploy = False

    def targets(self, batch):
        images = batch['image']
        height, width = images.shape[-2:]
        rows, labels, indices = [], [], []
        for index, boxes in enumerate(batch['gt_bboxes']):
            boxes = boxes[boxes[:, 4] > 0]
            if boxes.numel() and (boxes[:, 4] >= self.num_classes).any():
                raise ValueError('YOLO11 labels must be UP foreground IDs 1..num_classes-1')
            xyxy = boxes[:, :4]
            center = (xyxy[:, :2] + xyxy[:, 2:]) / 2
            size = xyxy[:, 2:] - xyxy[:, :2]
            rows.append(torch.cat((center, size), -1) / images.new_tensor([width, height, width, height]))
            labels.append(boxes[:, 4:5] - 1)
            indices.append(images.new_full((len(boxes),), index))
        return {'img': images, 'bboxes': torch.cat(rows), 'cls': torch.cat(labels),
                'batch_idx': torch.cat(indices)}

    def forward(self, batch):
        images = batch['image']  # RGB [0,1]; padding and boxes use continuous coordinates.
        if self.training:
            # The official dataset already supplies normalized xywh and class IDs.
            # Keep that target representation intact instead of round-tripping boxes.
            targets = dict(batch['yolo11_targets'], img=images) if 'yolo11_targets' in batch else self.targets(batch)
            loss, components = self.detector.loss(targets)
            # Official loss is a weighted 3-vector times batch size. Sum once.
            return {'YOLO11.loss': loss.sum(), 'YOLO11.box_component': components[0],
                    'YOLO11.cls_component': components[1], 'YOLO11.dfl_component': components[2]}
        predictions = self.detector(images)
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions
        if self.deploy:
            # Raw decoded xywh + class probabilities; NMS stays outside ONNX.
            return {'blobs_predictions': predictions}
        return {'dt_bboxes': self.postprocess(predictions)}

    def postprocess(self, predictions):
        """Official candidate selection; only NMS dispatch is vendor-specific.

        COCO validation uses multi_label=True and confidence=0.001. Interactive
        inference may retain the single-label option. Coordinates remain continuous.
        """
        rows = []
        for index, prediction in enumerate(predictions.transpose(1, 2)):
            prediction = prediction[prediction[:, 4:].amax(-1) > self.score_threshold]
            if self.multi_label and self.num_classes > 2:
                selected, classes = torch.where(prediction[:, 4:] > self.score_threshold)
                boxes, scores = prediction[selected, :4], prediction[selected, classes + 4]
            else:
                scores, classes = prediction[:, 4:].max(-1)
                boxes = prediction[:, :4]
            boxes = torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                               boxes[:, :2] + boxes[:, 2:] / 2), -1)
            # Match official class-aware NMS. Its 30k cap is applied before NMS.
            if len(boxes) > 30000:
                order = scores.argsort(descending=True)[:30000]
                boxes, scores, classes = boxes[order], scores[order], classes[order]
            keep = nms(boxes + classes[:, None] * 7680, scores, self.nms_threshold)[:self.max_detections]
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
            rows.append(torch.cat((boxes.new_full((len(boxes), 1), index), boxes,
                                   scores[:, None], (classes + 1).to(boxes)[:, None]), -1))
        return torch.cat(rows)
