"""COCO IDs + official YOLO labels → YOLODataset → UP batch/evaluator.

Imported only when the YOLO11 dataset is selected. Image decoding, Mosaic,
MixUp, LetterBox and label transforms are the pinned upstream implementation.
No vendor SDK is imported and no label/cache files are written beside datasets.
"""
import json
from pathlib import Path

import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.ops import scale_boxes, segments2boxes

from up.data.metrics.base_evaluator import build_evaluator


def original_boxes(output, index):
    """Invert the exact resize/padding metadata using the official validator math."""
    boxes = output['dt_bboxes'][output['dt_bboxes'][:, 0] == index, 1:].clone()
    scale_boxes(output['image'].shape[2:], boxes[:, :4], output['ori_shape'][index],
                ratio_pad=output['ratio_pad'][index])
    return boxes


class Yolo11CocoDataset(YOLODataset):
    def __init__(self, meta_file, image_reader, label_dir, transformer=None, evaluator=None,
                 training=True, imgsz=640, batch_size=32, hyp=None):
        with open(meta_file, encoding='utf-8') as stream:
            self.coco_json = json.load(stream)
        self.image_root = Path(image_reader['kwargs']['image_dir'])
        self.label_root = Path(label_dir)
        if not self.label_root.is_dir():
            raise FileNotFoundError('Install official coco2017labels-segments first: ' + str(self.label_root))
        self.records = sorted(self.coco_json['images'], key=lambda row: row['file_name'])
        self.annotated_images = {ann['image_id'] for ann in self.coco_json['annotations']
                                 if not ann.get('iscrowd', 0) and ann['bbox'][2] > 0 and ann['bbox'][3] > 0}
        categories = sorted(self.coco_json['categories'], key=lambda row: row['id'])
        self.class_to_category = {i + 1: row['id'] for i, row in enumerate(categories)}
        self.path_to_id = {str(self.image_root / row['file_name']): row['id'] for row in self.records}
        self.classes = ['__background__'] + [row['name'] for row in categories]
        self.hyp = IterableSimpleNamespace(**dict(DEFAULT_CFG_DICT, **(hyp or {})))
        self.evaluator = build_evaluator(evaluator) if evaluator else None
        super().__init__(img_path=str(self.image_root), imgsz=imgsz, batch_size=batch_size,
                         augment=training, hyp=self.hyp, cache=False, rect=not training,
                         stride=32, pad=0.0 if training else 0.5,
                         data={'names': self.classes[1:], 'nc': len(categories)}, task='detect')
        # Workers need the parsed labels, not a second copy of the full COCO JSON.
        del self.coco_json

    def get_img_files(self, img_path):
        return [str(self.image_root / row['file_name']) for row in self.records]

    def get_labels(self):
        """Read official normalized polygon labels without rewriting source images.

        Official detection training also uses segments for CopyPaste and derives
        boxes with segments2boxes. Format still emits detection targets, not masks.
        COCO JSON supplies image/category IDs and evaluator ground truth only.
        """
        labels = []
        for image, path in zip(self.records, self.im_files):
            label_file = self.label_root / Path(image['file_name']).with_suffix('.txt')
            if not label_file.exists() and image['id'] in self.annotated_images:
                raise FileNotFoundError('Missing official label for annotated image: ' + str(label_file))
            rows = [line.split() for line in label_file.read_text().splitlines() if line.strip()] if label_file.exists() else []
            segments = []
            if rows:
                if any(len(row) < 7 or (len(row) - 1) % 2 for row in rows):
                    raise ValueError('This recipe requires official polygon labels: ' + str(label_file))
                classes = np.asarray([row[0] for row in rows], dtype=np.float32).reshape(-1, 1)
                segments = [np.asarray(row[1:], dtype=np.float32).reshape(-1, 2) for row in rows]
                if (classes < 0).any() or (classes >= len(self.classes) - 1).any() or (classes != classes.astype(int)).any():
                    raise ValueError('Invalid COCO class index: ' + str(label_file))
                values = np.concatenate((classes, segments2boxes(segments)), 1)
                # Match upstream label-file validation, including its small
                # coordinate margin. This is unrelated to GPU numeric tolerances.
                if not np.isfinite(values).all() or values.min() < -0.01 or values[:, 1:].max() > 1.01:
                    raise ValueError('Invalid normalized label coordinates: ' + str(label_file))
                _, unique = np.unique(values, axis=0, return_index=True)
                if len(unique) < len(values):
                    values = values[unique]
                    segments = [segments[index] for index in unique]
            else:
                values = np.zeros((0, 5), dtype=np.float32)
            labels.append(dict(im_file=path, shape=(image['height'], image['width']),
                               cls=values[:, :1], bboxes=values[:, 1:], segments=segments,
                               normalized=True, bbox_format='xywh'))
        return labels

    def close_strong_augmentation(self):
        # Invoked before creating new epoch workers; prefetched old batches cannot leak in.
        self.close_mosaic(self.hyp)

    @staticmethod
    def collate_fn(batch):
        result = YOLODataset.collate_fn(batch)
        targets = {key: result[key] for key in ('bboxes', 'cls', 'batch_idx')}
        return {'image': result['img'], 'yolo11_targets': targets,
                'filenames': result['im_file'], 'ori_shape': result['ori_shape'],
                'ratio_pad': result.get('ratio_pad', ()), 'image_info': [()] * len(batch)}

    def dump(self, output):
        results = []
        for index, filename in enumerate(output['filenames']):
            for x1, y1, x2, y2, score, label in original_boxes(output, index).cpu().tolist():
                results.append({'image_id': self.path_to_id[filename],
                                'category_id': self.class_to_category[int(label)],
                                'bbox': [x1, y1, x2 - x1, y2 - y1], 'score': score})
        return results

    def evaluate(self, res_file, res=None):
        return self.evaluator.eval(res_file, res) if self.evaluator else {}
