import torch
from torch import nn
from torchvision.models import resnet50

from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from .official.backbone import BackboneBase, FrozenBatchNorm2d, Joiner
from .official.detr import DETR, PostProcess, SetCriterion
from .official.matcher import HungarianMatcher
from .official.misc import NestedTensor
from .official.position_encoding import PositionEmbeddingSine
from .official.transformer import Transformer


@MODULE_ZOO_REGISTRY.register('detr')
class DetrForUP(nn.Module):
    """Official DETR with UP's batch, label and bounding-box conventions.

    ``num_classes`` includes UP's background class; DETR adds its own no-object
    class after the foreground classes. Images must use top-left alignment.
    """

    def __init__(self, num_classes=81, num_queries=100, aux_loss=True, aligned=True):
        super().__init__()
        self.offset = 0 if aligned else 1
        # 使用随仓库保留的官方 DETR 结构；此类仅负责连接 UP 的输入、标签和输出约定。
        body = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
        backbone = BackboneBase(body, train_backbone=True, num_channels=2048,
                                return_interm_layers=False)
        backbone = Joiner(backbone, PositionEmbeddingSine(128, normalize=True))
        backbone.num_channels = 2048
        transformer = Transformer(d_model=256, nhead=8, num_encoder_layers=6,
                                  num_decoder_layers=6, dim_feedforward=2048,
                                  dropout=0.1, return_intermediate_dec=True)
        # UP 的类别数包含背景；官方 DETR 只接收前景数，并自行添加 no-object 类。
        self.detector = DETR(backbone, transformer, num_classes - 1, num_queries, aux_loss)
        self.weights = {'loss_ce': 1.0, 'loss_bbox': 5.0, 'loss_giou': 2.0}
        if aux_loss:
            # 六层 decoder 的前五层沿用相同损失权重，最终层使用无后缀的键。
            self.weights.update({name + '_' + str(i): weight
                                 for i in range(5) for name, weight in list(self.weights.items())})
        self.criterion = SetCriterion(num_classes - 1, HungarianMatcher(1, 5, 2),
                                      self.weights, 0.1, ['labels', 'boxes', 'cardinality'])
        self.postprocess = PostProcess()
        self.deploy = False

    def forward(self, batch):
        images, information = batch['image'], batch['image_info']
        # NestedTensor 中 True 表示 padding；有效图像固定从左上角开始，避免位置编码错位。
        mask = images.new_ones((images.shape[0], images.shape[2], images.shape[3]), dtype=torch.bool)
        for index, info in enumerate(information):
            if float(info[6]) or float(info[7]):
                raise ValueError('DETR requires top-left aligned images without left/top padding.')
            mask[index, :int(info[0]), :int(info[1])] = False
        predictions = self.detector(NestedTensor(images, mask))
        if self.deploy:
            # Preserve official tensors; matching and pixel-space postprocessing
            # are training/inference work, not part of the exported network.
            return {'blobs_logits': predictions['pred_logits'],
                    'blobs_boxes': predictions['pred_boxes']}
        if self.training:
            targets = []
            for boxes, info in zip(batch['gt_bboxes'], information):
                boxes = boxes[boxes[:, 4] > 0]
                height, width = float(info[0]), float(info[1])
                scale = info[2]
                sh, sw = scale if isinstance(scale, (tuple, list)) else (scale, scale)
                xyxy = boxes[:, :4].clone()
                # 先统一 UP 的闭区间端点约定，再转成 DETR 的归一化 cx/cy/w/h。
                xyxy[:, 2] += self.offset * sw
                xyxy[:, 3] += self.offset * sh
                xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, width)
                xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, height)
                x0, y0, x1, y1 = xyxy.unbind(-1)
                normalized = torch.stack(((x0 + x1) / (2 * width), (y0 + y1) / (2 * height),
                                          (x1 - x0) / width, (y1 - y0) / height), dim=-1)
                # UP 前景标签从 1 开始，DETR 前景标签从 0 开始。
                targets.append({'boxes': normalized, 'labels': boxes[:, 4].long() - 1})
            losses = self.criterion(predictions, targets)
            return {'DETR.' + name: loss * self.weights[name]
                    for name, loss in losses.items() if name in self.weights}
        # 推理把官方检测结果转回 UP 的图像索引/xyxy/分数/类别七列格式。
        sizes = images.new_tensor([[float(info[0]), float(info[1])] for info in information])
        rows = []
        for index, (result, info) in enumerate(zip(self.postprocess(predictions, sizes), information)):
            scale = info[2]
            sh, sw = scale if isinstance(scale, (tuple, list)) else (scale, scale)
            boxes = result['boxes'].clone()
            boxes[:, 2] -= self.offset * sw
            boxes[:, 3] -= self.offset * sh
            rows.append(torch.cat((boxes.new_full((len(boxes), 1), index), boxes,
                                   result['scores'][:, None], (result['labels'] + 1).to(boxes)[:, None]), dim=1))
        return {'dt_bboxes': torch.cat(rows, dim=0)}
