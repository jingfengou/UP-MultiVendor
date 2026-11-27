"""Official COCO model/criterion plugged into the existing UP BaseRunner."""
from argparse import Namespace

import torch
from torch import nn

from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from ..utils.source_helper import load_source


@MODULE_ZOO_REGISTRY.register('official_detr')
class OfficialDetr(nn.Module):
    """Keep official 91 COCO slots, FrozenBN, NestedTensor and auxiliary loss.

    The runner seeds before construction. Labels retain original COCO IDs,
    instead of the older UP adapter's contiguous 80-class mapping.
    """

    def __init__(self, source, package_sha256, model_args):
        super().__init__()
        official = load_source(source, package_sha256)
        self.detector, self.criterion, self.postprocessors = official.build_model(
            Namespace(**model_args))
        self.deploy = False

    def forward(self, batch):
        if self.deploy:
            from util.misc import NestedTensor
            image = batch['image']
            samples = NestedTensor(image, torch.zeros(
                (image.shape[0], image.shape[2], image.shape[3]),
                dtype=torch.bool, device=image.device))
        else:
            samples = batch['samples']
        predictions = self.detector(samples)
        if self.deploy:
            return {'blobs_logits': predictions['pred_logits'],
                    'blobs_boxes': predictions['pred_boxes']}
        if self.training:
            components = self.criterion(predictions, batch['targets'])
            weights = self.criterion.weight_dict
            total = sum(components[key] * weights[key]
                        for key in components if key in weights)
            return {'model.loss': total,
                    'metrics': {key: value.detach() for key, value in components.items()}}
        if batch['targets'] and 'labels' in batch['targets'][0]:
            # Official evaluation also runs criterion, without backward/update.
            self.criterion(predictions, batch['targets'])
        sizes = torch.stack([target['orig_size'] for target in batch['targets']])
        return {'official_predictions': self.postprocessors['bbox'](predictions, sizes)}
