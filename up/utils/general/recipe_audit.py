"""Record constructed runtime behavior rather than only requested YAML values."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from up.utils.env.dist_helper import env
from .hook_helper import Hook
from .registry_factory import HOOK_REGISTRY


def tensor_digest(tensor):
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def rng_snapshot(runtime):
    numpy_state = np.random.get_state()
    return {
        'torch_initial_seed': torch.initial_seed(),
        'device_initial_seed': runtime.module.initial_seed(),
        'python_state_sha256': hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        'numpy_state_sha256': hashlib.sha256(numpy_state[1].tobytes() +
            repr((numpy_state[0], *numpy_state[2:])).encode()).hexdigest(),
        'torch_state_sha256': tensor_digest(torch.get_rng_state()),
        'device_state_sha256': tensor_digest(runtime.module.get_rng_state()),
    }


def state_digest(state):
    """Hash names, tensor metadata and bytes, without writing model weights."""
    record = {name: {'shape': list(value.shape), 'dtype': str(value.dtype),
                     'sha256': tensor_digest(value)}
              for name, value in state.items() if torch.is_tensor(value)}
    payload = json.dumps(record, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def loader_description(loader):
    loader = getattr(loader, 'loader', loader)
    sampler = getattr(loader, 'sampler', None)
    generator = getattr(loader, 'generator', None)
    initialize = getattr(loader, 'worker_init_fn', None)
    return {'type': type(loader).__name__, 'num_workers': loader.num_workers,
            'pin_memory': loader.pin_memory, 'drop_last': loader.drop_last,
            'sampler': type(sampler).__name__,
            'sampler_seed': getattr(sampler, 'seed', None),
            'generator_seed': generator.initial_seed() if generator is not None else None,
            'worker_init': (initialize.__module__ + '.' + initialize.__qualname__)
                           if initialize is not None else None}


@HOOK_REGISTRY.register('recipe_audit')
class RecipeAudit(Hook):
    """Write each rank's real optimizer, BN, DDP, AMP and initialization state."""
    def before_train(self):
        runner = self.runner_ref()
        model = runner.model.module if isinstance(runner.model, DistributedDataParallel) else runner.model
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        groups = []
        for group in runner.optimizer.param_groups:
            groups.append({key: value for key, value in group.items()
                           if key not in ('params', 'param_names')})
            groups[-1]['parameters'] = [names[id(parameter)] for parameter in group['params']]
        bn = {name: {'type': type(module).__name__, 'eps': getattr(module, 'eps', None),
                     'momentum': getattr(module, 'momentum', None), 'training': module.training,
                     'state_sha256': state_digest(module.state_dict())}
              for name, module in model.named_modules()
              if 'BatchNorm' in type(module).__name__}
        ddp = None
        if isinstance(runner.model, DistributedDataParallel):
            ddp = {'broadcast_buffers': runner.model.broadcast_buffers,
                   'find_unused_parameters': runner.model.find_unused_parameters}
        record = {
            'platform': runner.runtime.platform, 'purpose': runner.runtime.purpose,
            'rank': env.rank, 'world_size': env.world_size,
            'torch': str(torch.__version__), 'rng_after_seed': runner._initial_rng,
            'rng_before_model': runner._before_model_rng,
            'rng_after_build': rng_snapshot(runner.runtime),
            'amp_enabled': runner.fp16,
            'scaler': runner.scaler.state_dict() if runner.fp16 else None,
            'dnn': {'benchmark': torch.backends.cudnn.benchmark,
                    'deterministic': torch.backends.cudnn.deterministic,
                    'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                    'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                    'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                    'warn_only': torch.is_deterministic_algorithms_warn_only_enabled()},
            'optimizer': type(runner.optimizer).__name__, 'parameter_groups': groups,
            'initial_model_sha256': state_digest(model.state_dict()),
            'normalization': bn, 'ddp': ddp,
            'loaders': {name: loader_description(loader)
                        for name, loader in runner.data_loaders.items()},
            'scheduler': type(runner.lr_scheduler).__name__, 'execution': runner.execution,
            'train_batches_per_epoch': runner.train_epoch_size,
            'train_batch_per_rank': runner.train_batch_size,
        }
        path = Path(runner.saver.save_dir) / (
            'recipe-rank' + str(env.rank) + '-start' + str(runner.start_iter) + '.json')
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2)
