"""Official YOLO26 data and metric components driven by UP's own loops."""
from copy import copy
import json
from pathlib import Path

import torch

from up.data.metrics.base_evaluator import Metric
from up.utils.env.dist_helper import env
from up.utils.general.registry_factory import DATA_BUILDER_REGISTY
from ..utils.source_helper import load_source
from ..models.model import official_model


class YOLO26Dataset:
    """Keep the official dataset, transforms and per-batch metric calculation."""
    def __init__(self, dataset, task, args, data, loader, training=False):
        self.official, self.task, self.args, self.data = dataset, task, args, data
        self.loader = loader
        self.metrics_in_place = True
        self.training_split = training

    def __len__(self):
        return len(self.official)

    def bind_model(self, model):
        network = official_model(model).network
        network.names = self.data['names']
        network.nc = self.data['nc']
        if self.task == 'classify' and not self.training_split:
            network.transforms = self.official.torch_transforms

    def begin_evaluation(self, model, device, amp, output_dir):
        from ultralytics.models.yolo.detect import DetectionValidator
        from ultralytics.models.yolo.classify import ClassificationValidator
        adapter = official_model(model)
        options = copy(self.args)
        options.half = bool(amp and device.type != 'cpu')
        cls = DetectionValidator if self.task == 'detect' else ClassificationValidator
        # Only single-batch methods are reused. Never call Validator.__call__().
        self.validator = cls(self.loader, Path(output_dir), args=vars(options))
        self.validator.device, self.validator.data = device, self.data
        self.validator.training = True
        self.validator.init_metrics(adapter.network)
        if self.task == 'detect':
            self.validator.iouv = torch.linspace(0.5, 0.95, 10, device=device)
            self.validator.niou = 10
        self.model = adapter.network
        self.original_dtype = next(self.model.parameters()).dtype
        self.loss_components = None
        self.model.half() if options.half else self.model.float()

    def dump(self, output):
        validator = self.validator
        raw = output['official_predictions']
        if getattr(self.model, 'criterion', None) is None:
            self.model.criterion = self.model.init_criterion()
        # Official train-time validation computes criterion as well as predictions.
        # Keep this per-batch work even though it does not update the optimizer.
        components = self.model.loss(output, raw)[1].detach()
        self.loss_components = (components.clone() if self.loss_components is None
                                else self.loss_components + components)
        from up.runtime import get_runtime
        if get_runtime().platform == 'metax_cuda' and self.task == 'detect':
            import torchvision
            from up.platform.metax_cuda.nms import nms
            previous = torchvision.ops.nms
            torchvision.ops.nms = nms
            try:
                predictions = validator.postprocess(raw)
            finally:
                torchvision.ops.nms = previous
        else:
            predictions = validator.postprocess(raw)
        validator.update_metrics(predictions, output)
        return []

    def finish_evaluation(self):
        # This method is called on every rank before UP evaluates on rank zero.
        self.validator.gather_stats()
        if self.loss_components is not None and env.distributed:
            torch.distributed.reduce(self.loss_components, dst=0, op=torch.distributed.ReduceOp.AVG)

    def restore_evaluation_dtype(self):
        self.model.to(dtype=self.original_dtype)

    def evaluate(self, res_file, results=None):
        validator = self.validator
        stats = validator.get_stats()
        if self.loss_components is not None:
            components = self.loss_components.float().cpu().reshape(-1) / len(self.loader)
            for index, value in enumerate(components):
                stats['val/component_' + str(index)] = float(value)
        validator.finalize_metrics()
        validator.print_results()
        if self.task == 'detect' and validator.args.save_json and validator.jdict:
            output = validator.save_dir / 'predictions.json'
            with output.open('w', encoding='utf-8') as stream:
                json.dump(validator.jdict, stream)
            stats = validator.eval_json(stats)
        # Official train-time Validator serializes metrics to five decimal places.
        # Its fitness, not an alternative metric, selects the best checkpoint.
        metric = Metric({key: round(float(value), 5) for key, value in stats.items()})
        metric.set_cmp_key('fitness')
        return metric


class YOLO26Loader:
    def __init__(self, loader, dataset):
        self.loader, self.dataset = loader, dataset
        self.batch_sampler = loader.batch_sampler
        self.batch_size = loader.batch_size
        self.start_iter = 0

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(self.start_iter % len(self.loader)):
            next(iterator)
        self.start_iter = 0
        yield from iterator

    def get_epoch_size(self):
        return len(self.loader)

    def set_start_iter(self, iteration):
        if iteration % len(self.loader):
            raise ValueError('Official YOLO26 resume requires an epoch boundary')
        self.start_iter = iteration

    def set_epoch(self, epoch):
        if hasattr(self.loader.sampler, 'set_epoch'):
            self.loader.sampler.set_epoch(epoch)

    def reset(self):
        self.loader.reset()


@DATA_BUILDER_REGISTY.register('official_yolo26')
class YOLO26DataBuilder:
    def __init__(self, config, data_source, reload_cfg=None):
        self.config, self.phase = config, data_source.split(':')[-1]

    def build_dataloader(self):
        config = self.config
        load_source(config['source'], config['package_sha256'])
        from ultralytics.cfg import get_cfg
        from ultralytics.data import build_dataloader, build_yolo_dataset
        from ultralytics.data.dataset import ClassificationDataset
        from ultralytics.data.utils import check_cls_dataset, check_det_dataset
        from ultralytics.utils.torch_utils import torch_distributed_zero_first
        args = get_cfg(overrides=config['recipe'])
        training = self.phase == 'train'
        mode = 'train' if training else 'val'
        data = (check_det_dataset(config['data'], autodownload=False) if args.task == 'detect'
                else check_cls_dataset(config['data']))
        batch = config[self.phase]['batch_size']
        with torch_distributed_zero_first(env.rank if env.distributed else -1):
            if args.task == 'detect':
                dataset = build_yolo_dataset(args, data[mode], batch, data, mode=mode,
                                             rect=not training, stride=32)
            else:
                dataset = ClassificationDataset(data[mode], args=args, augment=training, prefix=mode)
        # ClassificationTrainer omits shuffle= even for train-time validation,
        # so its fixed version uses build_dataloader's default True there.
        shuffle = (args.task == 'classify' or training) and not getattr(dataset, 'rect', False)
        loader = build_dataloader(dataset, batch, args.workers * (
            2 if args.task == 'detect' and not training else 1),
            shuffle=shuffle, rank=env.rank if env.distributed else -1,
            drop_last=args.compile and training)
        wrapper = YOLO26Dataset(dataset, args.task, args, data, loader, training=training)
        return YOLO26Loader(loader, wrapper)
