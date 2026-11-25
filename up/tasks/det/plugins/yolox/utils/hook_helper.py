import random

import torch
import torch.nn.functional as F

from up.utils.general.registry_factory import HOOK_REGISTRY
from up.utils.general.hook_helper import Hook
from up.utils.env.dist_helper import broadcast, env
from up.utils.general.log_helper import default_logger as logger


__all__ = ['YoloxNoaug', 'YoloxBatchResize']


@HOOK_REGISTRY.register('yolox_noaug')
class YoloxNoaug(Hook):
    def __init__(self, runner, no_aug_epoch=15, max_epoch=300, transformer=[], test_freq=1, save_freq=1,
                 switch_before_data=False):
        super(YoloxNoaug, self).__init__(runner)
        self.no_aug_epoch = no_aug_epoch
        self.max_epoch = max_epoch
        self.transformer = transformer
        self.flag = False
        self.test_freq = test_freq
        self.save_freq = save_freq
        self.switch_before_data = switch_before_data

    def before_data(self, cur_iter):
        if self.switch_before_data and self.runner_ref().model.training:
            self.switch_augmentation(cur_iter)

    def before_forward(self, cur_iter, input):
        if not self.switch_before_data:
            self.switch_augmentation(cur_iter)

    def switch_augmentation(self, cur_iter):
        runner = self.runner_ref()
        if cur_iter >= runner.data_loaders['train'].get_epoch_size() * (self.max_epoch - self.no_aug_epoch):
            if not self.flag:
                logger.info(f"rebuild dataset transformer cfg {self.transformer}")
                runner.config['dataset']['train']['dataset']['kwargs']['transformer'] = self.transformer
                del runner.data_loaders
                runner.data_iterators = {}
                import gc
                gc.collect()
                logger.info("rebuild dataloader")
                runner.build_dataloaders()
                if self.switch_before_data:
                    runner.data_loaders['train'].batch_sampler.set_start_iter(cur_iter)
                runner.data_iterators['train'] = iter(runner.data_loaders["train"])
                try:
                    if env.world_size > 1:
                        runner.model.module.yolox_post.use_l1 = True
                    else:
                        runner.model.yolox_post.use_l1 = True
                except:  # noqa
                    pass
                runner.test_freq = self.test_freq
                runner.save_freq = self.save_freq
                self.flag = True


@HOOK_REGISTRY.register('yolox_batch_resize')
class YoloxBatchResize(Hook):
    def __init__(self, runner, input_size=640, random_size=(14, 26), interval=10):
        super().__init__(runner)
        self.size = self.input_size = input_size
        self.random_size = random_size
        self.interval = interval
        self.rng = random.Random(runner.config['runtime']['random_seed'])
        self.resize_index = 0

    def before_forward(self, cur_iter, input):
        image = input['image']
        resize_index = cur_iter // self.interval
        if resize_index != self.resize_index:
            if env.rank == 0:
                for _ in range(self.resize_index, resize_index):
                    self.size = self.rng.randint(*self.random_size) * 32
            if env.world_size > 1:
                size = torch.tensor([self.size], device=image.device, dtype=torch.int64)
                broadcast(size, 0)
                self.size = size.item()
            self.resize_index = resize_index

        if self.size != self.input_size:
            input['image'] = F.interpolate(image, size=(self.size, self.size),
                                           mode='bilinear', align_corners=False)
            scale = self.size / self.input_size
            for boxes in input['gt_bboxes']:
                boxes[:, :4].mul_(scale)
            for info in input['image_info']:
                info[:2] = [self.size, self.size]
                factors = info[2]
                info[2] = [factor * scale for factor in factors] if isinstance(
                    factors, (list, tuple)) else factors * scale
