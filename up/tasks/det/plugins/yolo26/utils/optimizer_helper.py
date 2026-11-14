"""YOLO26 parameter grouping using UP's existing optimizer-builder protocol."""
from torch import nn

from up.utils.general.registry_factory import OPTIMIZER_REGISTRY
from up.utils.model.optimizer_helper import BaseOptimizer
from ..models.model import official_model


@OPTIMIZER_REGISTRY.register('official_yolo26')
class YOLO26Optimizer(BaseOptimizer):
    def get_trainable_params(self, cfg_optim=None):
        """Keep the pinned recipe's group order, decay and parameter names."""
        options = official_model(self.model).options
        if options.optimizer not in ('SGD', 'MuSGD'):
            raise NotImplementedError('Resolve the reference optimizer to SGD or MuSGD')
        groups, names = [[], [], [], []], [[], [], [], []]
        norm_types = tuple(value for key, value in nn.__dict__.items() if 'Norm' in key)
        for module_name, module in official_model(self.model).network.named_modules():
            for name, parameter in module.named_parameters(recurse=False):
                full_name = module_name + '.' + name if module_name else name
                if parameter.ndim >= 2 and options.optimizer == 'MuSGD':
                    index = 3
                elif 'bias' in full_name:
                    index = 2
                elif isinstance(module, norm_types) or 'logit_scale' in full_name:
                    index = 1
                else:
                    index = 0
                groups[index].append(parameter)
                names[index].append(full_name)
        accumulate = max(round(options.nbs / options.batch), 1)
        decay = options.weight_decay * options.batch * accumulate / options.nbs
        if options.optimizer == 'SGD':
            # The pinned classifier uses complete dictionaries and unnamed groups.
            common = dict(lr=options.lr0, momentum=options.momentum, nesterov=True)
            return [
                dict(params=groups[0], weight_decay=decay, **common),
                dict(params=groups[1], weight_decay=0.0, **common),
                dict(params=groups[2], **common),
            ]
        result = [dict(params=groups[2], param_names=names[2])]
        if groups[3]:
            result.append(dict(params=groups[3], lr=options.lr0, weight_decay=decay,
                               momentum=options.momentum, nesterov=True,
                               use_muon=True, param_names=names[3]))
        result.extend([
            dict(params=groups[0], weight_decay=decay, param_names=names[0]),
            dict(params=groups[1], weight_decay=0.0, param_names=names[1]),
        ])
        return result

    def get_optimizer(self, cfg_optim):
        """Use UP's SGD construction; bind the official MuSGD class when required."""
        options = official_model(self.model).options
        if cfg_optim['type'] != options.optimizer:
            raise ValueError('Optimizer type must match the model recipe')
        if options.optimizer == 'SGD':
            return super().get_optimizer(cfg_optim)
        from ultralytics.optim import MuSGD
        groups = cfg_optim['kwargs']['params']
        first, *remaining = groups
        optimizer = MuSGD(first['params'], lr=options.lr0, momentum=options.momentum,
                          nesterov=True, muon=options.muon_w, sgd=options.sgd_w,
                          param_names=first['param_names'], cls_w=options.cls_w)
        for group in remaining:
            optimizer.add_param_group(group)
        return optimizer
