"""DETR image preprocessing/output conversion for UP's common inference command."""
from importlib import import_module

import numpy as np
from PIL import Image
import torch

from up.apis.inference import BaseInference
from up.utils.general.registry_factory import INFERENCER_REGISTRY
from .source_helper import load_source


@INFERENCER_REGISTRY.register('official_detr')
class DetrInference(BaseInference):
    def resume(self):
        checkpoint = self.saver.load_checkpoint(self.ckpt)
        ema = checkpoint.get('ema') or {}
        self.detector.load_state_dict(ema.get('ema_state_dict', checkpoint['model']), strict=True)

    def build_data(self):
        source = self.config['official']
        load_source(source['source'], source['package_sha256'])
        self.transform = import_module('datasets.coco').make_coco_transforms('val')
        self.nested = import_module('util.misc').nested_tensor_from_tensor_list
        self.color_mode = 'RGB'

    def fetch(self, filenames):
        images, tensors, targets = [], [], []
        for filename in filenames:
            image = Image.open(filename).convert('RGB')
            tensor, _ = self.transform(image, None)
            width, height = image.size
            images.append(np.asarray(image))
            tensors.append(tensor)
            targets.append({'orig_size': torch.tensor([height, width], device=self.runtime.device)})
        return {'samples': self.nested(tensors).to(self.runtime.device), 'targets': targets,
                'origin_image': images, 'filenames': filenames}

    def map_back(self, output):
        rows = []
        for image, filename, result in zip(output['origin_image'], output['filenames'],
                                           output['official_predictions']):
            boxes = torch.cat((result['boxes'], result['scores'][:, None],
                               result['labels'][:, None]), dim=1).cpu().numpy()
            height, width = image.shape[:2]
            rows.append({'image': image,
                         'image_info': [height, width, 1., height, width, False, 0, 0, filename],
                         'dt_bboxes': boxes})
        return rows
