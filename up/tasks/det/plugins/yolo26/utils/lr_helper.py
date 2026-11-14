"""YOLO26 epoch LR and batch warmup, called by UP's common BaseRunner."""
import math

import numpy as np
import torch

from up.utils.general.registry_factory import LR_SCHEDULER_REGISTY
from up.utils.model.lr_helper import BaseLRScheduler


@LR_SCHEDULER_REGISTY.register('official_yolo26')
class YOLO26Scheduler(BaseLRScheduler):
    def build_scheduler(self):
        return YOLO26Schedule(self.optimizer, self.data_size, self.cfg_scheduler['kwargs'])


class YOLO26Schedule:
    """Epoch LambdaLR plus official batch LR/momentum/accumulation warmup."""
    def __init__(self, optimizer, data_size, options):
        self.optimizer, self.data_size, self.options = optimizer, data_size, options
        self.accumulate = max(round(options['nbs'] / options['batch']), 1)
        self.warmup = (max(round(options['warmup_epochs'] * data_size), 100)
                       if options['warmup_epochs'] > 0 else -1)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.factor)
        self.scheduler.last_epoch = -1

    def factor(self, epoch):
        options = self.options
        if options['cos_lr']:
            return (1 - math.cos(epoch * math.pi / options['epochs'])) / 2 * (
                options['lrf'] - 1) + 1
        return max(1 - epoch / options['epochs'], 0) * (1 - options['lrf']) + options['lrf']

    def step(self):
        self.scheduler.step()

    def before_batch(self, iteration):
        if iteration > self.warmup:
            return
        options = self.options
        self.accumulate = max(1, int(np.interp(
            iteration, [0, self.warmup], [1, options['nbs'] / options['batch']]).round()))
        epoch = iteration // self.data_size
        for index, group in enumerate(self.optimizer.param_groups):
            group['lr'] = np.interp(iteration, [0, self.warmup], [
                options['warmup_bias_lr'] if index == 0 else 0,
                group['initial_lr'] * self.factor(epoch)])
            if 'momentum' in group:
                group['momentum'] = np.interp(iteration, [0, self.warmup],
                                              [options['warmup_momentum'], options['momentum']])

    def state_dict(self):
        return {'scheduler': self.scheduler.state_dict(), 'accumulate': self.accumulate}

    def load_state_dict(self, state):
        self.scheduler.load_state_dict(state['scheduler'])
        self.accumulate = state['accumulate']

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]
