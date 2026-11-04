"""Independent inference uses the same official resize/padding inversion as eval."""
import math

import cv2
import numpy as np
import torch
from easydict import EasyDict
from ultralytics.data.augment import LetterBox

from up.apis.inference import BaseInference
from up.data.image_reader import build_image_reader
from .data import original_boxes


class Yolo11Inference(BaseInference):
    def build_data(self):
        cfg = self.config['dataset']['test']['dataset']['kwargs']
        self.imgsz = cfg['imgsz']
        self.color_mode = cfg['image_reader']['kwargs']['color_mode']
        self.image_reader = build_image_reader(cfg['image_reader'])

    def fetch_single(self, filename, input_shape=None):
        image = self.image_reader.read(filename) if isinstance(filename, str) else filename
        h0, w0 = image.shape[:2]
        ratio = self.imgsz / max(h0, w0)
        h, w = min(math.ceil(h0 * ratio), self.imgsz), min(math.ceil(w0 * ratio), self.imgsz)
        resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR) if (h, w) != (h0, w0) else image
        # Independent pictures use a square canvas. A caller comparing against
        # rectangular validation can supply that validation batch's exact canvas.
        shape = input_shape or (self.imgsz, self.imgsz)
        padded = LetterBox(shape, scaleup=False)(image=resized)
        left, top = round((shape[1] - w) / 2 - 0.1), round((shape[0] - h) / 2 - 0.1)
        tensor = torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1))).unsqueeze(0)
        return EasyDict(image=tensor.to(self.runtime.device).float() / 255,
                        origin_image=[image], ori_shape=[(h0, w0)],
                        ratio_pad=[((h / h0, w / w0), (left, top))],
                        image_info=[(h, w, (h / h0, w / w0), h0, w0, False, left, top, filename)],
                        filenames=[filename])

    def fetch(self, filename_list):
        batches = [self.fetch_single(filename) for filename in filename_list]
        return EasyDict({key: torch.cat([batch[key] for batch in batches]) if key == 'image'
                         else [value for batch in batches for value in batch[key]]
                         for key in batches[0]})

    def map_back(self, output):
        return [{'image': image, 'image_info': output['image_info'][index],
                 'dt_bboxes': original_boxes(output, index).cpu().numpy()}
                for index, image in enumerate(output['origin_image'])]
