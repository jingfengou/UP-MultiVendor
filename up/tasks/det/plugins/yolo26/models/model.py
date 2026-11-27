"""Pinned official YOLO26 model/loss as a UP dictionary-batch module."""
import random

import torch
from torch import nn

from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from ..utils.source_helper import load_source, sha256


@MODULE_ZOO_REGISTRY.register('official_yolo26')
class YOLO26ForUP(nn.Module):
    """Build the official network directly, without constructing a Trainer.

    Images and targets keep official batch keys. Detection normalization happens
    once here, after the runtime transfers tensors to the selected device.
    """

    def __init__(self, source, package_sha256, task, recipe, classes, names,
                 pretrained=None, pretrained_sha256=None):
        super().__init__()
        load_source(source, package_sha256)
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import ClassificationModel, DetectionModel, load_checkpoint
        self.task, self.options = task, get_cfg(overrides=recipe)
        if task == 'detect':
            if pretrained is None or sha256(pretrained) != pretrained_sha256:
                raise ValueError('Detection requires verified official Objects365 initialization')
            weights, _ = load_checkpoint(pretrained, device='cpu')
            self.network = DetectionModel(weights.yaml, nc=classes, ch=3, verbose=False)
            self.network.load(weights)
            # The official trainer always freezes fixed DFL projection parameters.
            for name, parameter in self.network.named_parameters():
                if '.dfl' in name:
                    parameter.requires_grad_(False)
        elif task == 'classify':
            if pretrained is not None:
                raise ValueError('Classification recipe starts from random initialization')
            self.network = ClassificationModel(recipe['model'], nc=classes, ch=3, verbose=False)
            for module in self.network.modules():
                if not self.options.pretrained and hasattr(module, 'reset_parameters'):
                    module.reset_parameters()
                if isinstance(module, nn.Dropout) and self.options.dropout:
                    module.p = self.options.dropout
        else:
            raise ValueError('Expected detect or classify')
        self.network.args = self.options
        self.network.names = dict(enumerate(names)) if isinstance(names, list) else names
        self.network.nc = classes
        self.deploy = False

    def check_amp(self):
        """Run the pinned official AMP guard without changing global callbacks."""
        from ultralytics.utils import callbacks
        from ultralytics.utils.checks import check_amp
        previous = callbacks.default_callbacks.copy()
        try:
            return check_amp(self.network)
        finally:
            callbacks.default_callbacks = previous

    def forward(self, batch):
        if self.deploy:
            result = self.network(batch['image'])
            result = result[0] if isinstance(result, tuple) else result
            return {'scores': result} if self.task == 'classify' else {'blobs_predictions': result}
        if self.task == 'detect':
            # Official preprocess consumes this draw even when multi_scale is zero.
            # Nonzero multi-scale needs the exact upstream interpolation protocol.
            if self.training:
                if self.options.multi_scale:
                    raise NotImplementedError('Nonzero official multi_scale is not integrated yet')
                random.random()
            dtype = torch.float32 if self.training else next(self.network.parameters()).dtype
            batch['img'] = batch['img'].to(dtype=dtype) / 255
        elif not self.training:
            batch['img'] = batch['img'].to(dtype=next(self.network.parameters()).dtype)
        if self.training:
            total, components = self.network(batch)
            return {'model.loss': total.sum(), 'metrics': components.detach()}
        return {'official_predictions': self.network(batch['img'])}

    def training_state(self):
        """Progressive loss weights are Python state, not model state_dict buffers."""
        criterion = getattr(self.network, 'criterion', None)
        return {key: getattr(criterion, key) for key in ('updates', 'o2m', 'o2o', 'o2m_copy')
                if hasattr(criterion, key)}

    def load_training_state(self, state):
        if state:
            if not hasattr(self.network, 'criterion'):
                self.network.criterion = self.network.init_criterion()
            for key, value in state.items():
                setattr(self.network.criterion, key, value)


def official_model(model):
    """Find the plugin network inside UP's ModelHelper and optional DDP wrapper."""
    if hasattr(model, 'module'):
        model = model.module
    matches = [module for module in model.modules() if isinstance(module, YOLO26ForUP)]
    if len(matches) != 1:
        raise ValueError('Expected one registered YOLO26 adapter')
    return matches[0]
