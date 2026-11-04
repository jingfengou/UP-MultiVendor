"""YOLO11 training policy shared by every vendor; model math stays upstream.

The recipe is pinned to Ultralytics 8.3.225. These small adapters expose its
optimizer/EMA and epoch LR policy through UP's Runner interfaces.
"""
import numpy as np


class Yolo11Schedule:
    """Linear epoch decay plus official per-batch LR/momentum warmup.

    last_iter is the number of completed batches, including AMP-skipped batches.
    State is saved after step(), so restoring never repeats a completed update.
    """
    def __init__(self, optimizer, steps_per_epoch, epochs, lr=0.01, lrf=0.01,
                 warmup_epochs=3, warmup_momentum=0.8, momentum=0.9, warmup_bias_lr=0.0):
        self.optimizer = optimizer
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.lr = lr
        self.lrf = lrf
        self.warmup_steps = max(round(warmup_epochs * steps_per_epoch), 100) if warmup_epochs else 0
        self.warmup_momentum = warmup_momentum
        self.momentum = momentum
        self.warmup_bias_lr = warmup_bias_lr
        self.last_iter = 0
        self.apply()

    def apply(self):
        epoch = self.last_iter // self.steps_per_epoch
        target = self.lr * (max(1 - epoch / self.epochs, 0) * (1 - self.lrf) + self.lrf)
        for index, group in enumerate(self.optimizer.param_groups):
            group['initial_lr'] = self.lr
            group['lr'] = target
            group['momentum'] = self.momentum
            if self.warmup_steps and self.last_iter <= self.warmup_steps:
                group['lr'] = float(np.interp(self.last_iter, [0, self.warmup_steps],
                                             [self.warmup_bias_lr if index == 0 else 0.0, target]))
                group['momentum'] = float(np.interp(self.last_iter, [0, self.warmup_steps],
                                                   [self.warmup_momentum, self.momentum]))

    def step(self):
        self.last_iter += 1
        self.apply()

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state):
        for key, value in self.state_dict().items():
            if key != 'last_iter' and state[key] != value:
                raise ValueError('YOLO11 resume changed scheduler field: ' + key)
        self.last_iter = state['last_iter']
        self.apply()


def build_optimizer(model, global_batch, lr=0.01, momentum=0.9, weight_decay=0.0005, nominal_batch=64):
    """Use the actual pinned official grouping, with its resolved long-run SGD.

    'auto' resolves to SGD/0.01/0.9 for this 600-epoch COCO run. Keeping that
    choice explicit prevents a short validation run from switching to AdamW.
    Biases and normalization weights have no decay; other weights do.
    """
    from ultralytics.engine.trainer import BaseTrainer
    if global_batch < nominal_batch:
        raise ValueError('This fixed-batch recipe requires global batch >= nominal batch; no hidden accumulation')
    return BaseTrainer.build_optimizer(None, model, name='SGD', lr=lr, momentum=momentum,
                                       decay=weight_decay * global_batch / nominal_batch)


class Yolo11EMA:
    """Keep the official FP32 EMA, including update count, in UP checkpoints."""
    def __init__(self, model):
        from ultralytics.utils.torch_utils import ModelEMA
        self.impl = ModelEMA(model, decay=0.9999, tau=2000)
        self.model = self.impl.ema

    def step(self, model, curr_step=None):
        self.impl.update(model)

    def state_dict(self):
        return {'ema_state_dict': self.model.state_dict(), 'updates': self.impl.updates}

    def load_state_dict(self, state):
        self.model.load_state_dict(state['ema_state_dict'])
        self.impl.updates = state['updates']
