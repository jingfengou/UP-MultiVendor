# Standard Library
import math
import copy
import random

# Import from third library
import cv2
import torch
import numpy as np

from up.tasks.det.plugins.yolov5.data.transforms import RandomPespective, box_candidates
from up.data.datasets.transforms import Augmentation
from up.utils.general.registry_factory import AUGMENTATION_REGISTRY

__all__ = ['YoloXRandomPespective', 'YoloxMegviiAffine', 'YoloxAdditiveHSV', 'YoloxMixUp_cv2']


@AUGMENTATION_REGISTRY.register('random_perspective_yolox')
class YoloXRandomPespective(RandomPespective):
    def __init__(self,
                 degrees=0.0,
                 translate=0.1,
                 scale=0.5,
                 shear=0.0,
                 perspective=0.0,
                 fill_color=0,
                 max_try=1,
                 border=(0, 0),
                 clip_box=True):
        super(YoloXRandomPespective, self).__init__(degrees,
                                                    translate,
                                                    scale,
                                                    shear,
                                                    perspective,
                                                    fill_color,
                                                    max_try,
                                                    border,
                                                    clip_box)
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.fill_color = fill_color
        self.border = border
        self.max_try = max_try

    def generate_trans_matrix(self, img, width, height):
        # Center
        C = np.eye(3)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(self.scale[0], self.scale[1])
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height  # y translation (pixels)

        # Combined rotation matrix
        # M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        M = T @ S @ R @ C
        return M, s


@AUGMENTATION_REGISTRY.register('yolox_megvii_affine')
class YoloxMegviiAffine(YoloXRandomPespective):
    """Crop the mosaic from its origin. Keep clipped boxes for the final size filter."""

    def generate_trans_matrix(self, img, width, height):
        angle = random.uniform(-self.degrees, self.degrees)
        scale = random.uniform(*self.scale)
        rotation = cv2.getRotationMatrix2D((0, 0), angle, scale)
        shear_x, shear_y = (math.tan(math.radians(random.uniform(-self.shear, self.shear))) for _ in range(2))
        matrix = np.eye(3)
        matrix[:2] = np.array([[1., shear_y], [shear_x, 1.]]) @ rotation
        matrix[0, 2] = random.uniform(-self.translate, self.translate) * width
        matrix[1, 2] = random.uniform(-self.translate, self.translate) * height
        return matrix, scale

    def label_transform(self, targets, matrix, width, height, scale):
        result = targets.clone()
        if result.shape[1] == 5:
            result = result[result[:, 4] > 0]  # UP may represent an empty GT by a zero row.
        if not len(result):
            return result
        corners = result[:, [0, 1, 2, 3, 0, 3, 2, 1]].double().reshape(-1, 4, 2)
        affine = torch.as_tensor(matrix[:2], device=result.device, dtype=corners.dtype)
        corners = corners @ affine[:, :2].T + affine[:, 2]
        bounds = torch.cat((corners.amin(dim=1), corners.amax(dim=1)), dim=1)
        result[:, [0, 2]] = bounds[:, [0, 2]].clamp(0, width).to(result.dtype)
        result[:, [1, 3]] = bounds[:, [1, 3]].clamp(0, height).to(result.dtype)
        return result


@AUGMENTATION_REGISTRY.register('yolox_additive_hsv')
class YoloxAdditiveHSV(Augmentation):
    """Megvii's independently gated additive HSV perturbations for BGR images."""

    def augment(self, data):
        if not (data.gt_bboxes[:, 4] > 0).any():
            return data
        delta = np.random.uniform(-1., 1., 3) * [5, 30, 30]
        delta = (delta * np.random.randint(0, 2, 3)).astype(np.int16)
        hsv = cv2.cvtColor(data.image, cv2.COLOR_BGR2HSV).astype(np.int16) + delta
        hsv[..., 0] %= 180
        hsv[..., 1:] = hsv[..., 1:].clip(0, 255)
        output = copy.copy(data)
        output.image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return output


@AUGMENTATION_REGISTRY.register('yolox_mixup_cv2')
class YoloxMixUp_cv2(Augmentation):
    def __init__(self,
                 input_size,
                 mixup_scale,
                 extra_input=True,
                 dataset=None,
                 flip_prob=1,
                 fill_color=0,
                 clip_box=True,
                 megvii_labels=False):
        super(YoloxMixUp_cv2, self).__init__()
        assert extra_input and dataset is not None
        self.dataset = dataset
        self.input_dim = input_size
        self.mixup_scale = mixup_scale
        self.flip_prob = flip_prob
        self.fill_color = fill_color
        self.clip_box = clip_box
        self.megvii_labels = megvii_labels

    def augment(self, data):
        if self.megvii_labels and not (data.gt_bboxes[:, 4] > 0).any():
            return data
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > self.flip_prob

        # data
        origin_input = copy.copy(data)
        origin_img = data['image']
        origin_labels = np.array(data['gt_bboxes'])

        # cp data
        idx_other = np.random.randint(0, len(self.dataset))
        if self.megvii_labels:
            while not self.dataset.get_image_classes(idx_other):
                idx_other = random.randrange(len(self.dataset))
        data_other = self.dataset.get_input(idx_other)
        img = data_other['image']
        cp_labels = np.array(data_other['gt_bboxes'])

        channels = 3           # default is rgb
        if len(data.image.shape) == 2:
            channels = 1
        if not isinstance(self.fill_color, list):
            self.fill_color = [self.fill_color] * channels
        cp_img = []
        for c in range(channels):
            canvas_c = np.full((self.input_dim[0], self.input_dim[1]), self.fill_color[c], dtype=np.uint8)
            cp_img.append(canvas_c)
        if channels == 1:
            cp_img = cp_img[0]
        else:
            cp_img = np.stack(cp_img, axis=-1)
        cp_scale_ratio = min(self.input_dim[0] / img.shape[0], self.input_dim[1] / img.shape[1])

        # resize cp img
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        cp_img[: int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)] = resized_img
        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        # flip cp img
        if FLIP:
            if len(data.image.shape) == 2:
                cp_img = cp_img[:, ::-1]
            else:
                cp_img = cp_img[:, ::-1, :]

        # pad
        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        if len(data.image.shape) == 2:
            padded_img = np.zeros(
                (max(origin_h, target_h), max(origin_w, target_w))
            ).astype(np.uint8)
        else:
            padded_img = np.zeros(
                (max(origin_h, target_h), max(origin_w, target_w), 3)
            ).astype(np.uint8)
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[y_offset: y_offset + target_h, x_offset: x_offset + target_w]

        def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
            bbox[:, 0::2] = bbox[:, 0::2] * scale_ratio + padw
            bbox[:, 1::2] = bbox[:, 1::2] * scale_ratio + padh
            if self.clip_box:
                bbox[:, 0::2] = np.clip(bbox[:, 0::2], 0, w_max)
                bbox[:, 1::2] = np.clip(bbox[:, 1::2], 0, h_max)
            return bbox

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = cp_bboxes_transformed_np[:, 0::2] - x_offset
        cp_bboxes_transformed_np[:, 1::2] = cp_bboxes_transformed_np[:, 1::2] - y_offset
        if self.clip_box:
            cp_bboxes_transformed_np[:, 0::2] = np.clip(
                cp_bboxes_transformed_np[:, 0::2], 0, target_w
            )
            cp_bboxes_transformed_np[:, 1::2] = np.clip(
                cp_bboxes_transformed_np[:, 1::2], 0, target_h
            )

        if self.megvii_labels:
            keep_list = cp_labels[:, 4] > 0
        else:
            keep_list = box_candidates(cp_bboxes_origin_np.T, cp_bboxes_transformed_np.T, 5)

        if keep_list.sum() >= 1.0:
            cls_labels = cp_labels[keep_list, 4:5].copy()
            box_labels = cp_bboxes_transformed_np[keep_list]
            labels = np.hstack((box_labels, cls_labels))
            origin_labels = np.vstack((origin_labels, labels))
            origin_img = origin_img.astype(np.float32)
            origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

            origin_input['image'] = origin_img.astype(np.uint8)
            origin_input['gt_bboxes'] = torch.tensor(origin_labels)

        return origin_input
