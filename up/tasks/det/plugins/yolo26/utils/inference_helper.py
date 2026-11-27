"""YOLO26 image preprocessing/output conversion for UP's common inference command."""
import numpy as np
from PIL import Image
import torch

from up.apis.inference import BaseInference
from up.utils.general.registry_factory import INFERENCER_REGISTRY
from ..models.model import official_model
from .source_helper import load_source


@INFERENCER_REGISTRY.register('official_yolo26')
class YOLO26Inference(BaseInference):
    def resume(self):
        checkpoint = self.saver.load_checkpoint(self.ckpt)
        ema = checkpoint.get('ema') or {}
        self.detector.load_state_dict(ema.get('ema_state_dict', checkpoint['model']), strict=True)

    def build_data(self):
        source = self.config['official']
        load_source(source['source'], source['package_sha256'])
        from ultralytics.data.augment import LetterBox, classify_transforms
        self.color_mode = 'RGB'
        self.recipe = self.config['dataset']['recipe']
        self.image_size = self.recipe['imgsz']
        self.task = self.recipe['task']
        self.transform = (classify_transforms(self.image_size) if self.task == 'classify'
                          else LetterBox((self.image_size, self.image_size), auto=True, stride=32))

    def fetch(self, filenames):
        tensors, originals = [], []
        for filename in filenames:
            image = Image.open(filename).convert('RGB')
            originals.append(image)
            if self.task == 'classify':
                tensor = self.transform(image)
            else:
                # Official preprocessing starts with BGR and flips before transfer.
                bgr = np.asarray(image)[:, :, ::-1]
                transformed = self.transform(image=bgr)
                tensor = torch.from_numpy(np.ascontiguousarray(transformed[:, :, ::-1].transpose(2, 0, 1)))
            tensors.append(tensor)
        return {'img': torch.stack(tensors).to(self.runtime.device),
                'origin_image': originals, 'filenames': filenames}

    def map_back(self, output):
        from ultralytics.utils import ops
        from ultralytics.utils import nms
        network = official_model(self.detector).network
        previous = None
        if self.runtime.platform == 'metax_cuda':
            import torchvision
            from up.platform.metax_cuda.nms import nms as native_nms
            previous = torchvision.ops.nms
            torchvision.ops.nms = native_nms
        try:
            predictions = nms.non_max_suppression(output['official_predictions'],
                conf_thres=0.25, iou_thres=self.recipe['iou'], max_det=self.recipe['max_det'],
                end2end=network.end2end)
        finally:
            if previous is not None:
                torchvision.ops.nms = previous
        results = []
        for prediction, image, filename in zip(predictions, output['origin_image'], output['filenames']):
            width, height = image.size
            prediction[:, :4] = ops.scale_boxes(output['img'].shape[2:], prediction[:, :4], (height, width))
            results.append({'image': np.asarray(image),
                            'image_info': [height, width, 1., height, width, False, 0, 0, filename],
                            'dt_bboxes': prediction.cpu().numpy()})
        return results

    def vis_cls(self, output):
        from PIL import ImageDraw
        from pathlib import Path
        predictions = output['official_predictions']
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        for image, filename, scores in zip(output['origin_image'], output['filenames'], predictions):
            index = int(scores.argmax())
            ImageDraw.Draw(image).text((10, 10), str(index), fill=(255, 255, 255))
            image.save(Path(self.vis_dir) / Path(filename).name)
