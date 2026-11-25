"""Official COCO dataset/sampler under UP data and evaluation interfaces."""
from argparse import Namespace
import importlib

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from up.data.metrics.base_evaluator import Metric
from up.utils.env.dist_helper import env
from up.utils.general.registry_factory import DATA_BUILDER_REGISTY
from ..utils.source_helper import load_source


class DetrDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.official = dataset
        self.metrics_in_place = True

    def __len__(self):
        return len(self.official)

    def __getitem__(self, index):
        return self.official[index]

    def begin_evaluation(self, model, device, amp, output_dir):
        from datasets.coco_eval import CocoEvaluator
        self.evaluator = CocoEvaluator(self.official.coco, ['bbox'])

    def dump(self, output):
        self.evaluator.update({target['image_id'].item(): prediction
                               for target, prediction in zip(output['targets'], output['official_predictions'])})
        return []

    def finish_evaluation(self):
        self.evaluator.synchronize_between_processes()

    def evaluate(self, res_file, results=None):
        self.evaluator.accumulate()
        self.evaluator.summarize()
        statistics = self.evaluator.coco_eval['bbox'].stats
        metric = Metric({'bbox.AP': float(statistics[0]),
                         'bbox.AP50': float(statistics[1])})
        metric.set_cmp_key('bbox.AP')
        return metric


class DetrLoader:
    """UP loader protocol with official sampler, collation and finite epochs."""
    def __init__(self, loader):
        self.loader = loader
        self.dataset, self.batch_sampler = loader.dataset, loader.batch_sampler
        self.batch_size = loader.batch_size
        self.start_iter = 0

    def __iter__(self):
        iterator = iter(self.loader)
        for _ in range(self.start_iter % len(self.loader)):
            next(iterator)
        self.start_iter = 0
        for samples, targets in iterator:
            yield {'samples': samples, 'targets': targets}

    def get_epoch_size(self):
        return len(self.loader)

    def set_start_iter(self, iteration):
        if iteration % len(self.loader):
            raise ValueError('Official DETR resume requires an epoch boundary')
        self.start_iter = iteration

    def set_epoch(self, epoch):
        if hasattr(self.loader.sampler, 'set_epoch'):
            self.loader.sampler.set_epoch(epoch)


@DATA_BUILDER_REGISTY.register('official_detr')
class DetrDataBuilder:
    def __init__(self, config, data_source, reload_cfg=None):
        self.config = config
        self.phase = data_source.split(':')[-1]

    def build_dataloader(self):
        config = self.config
        load_source(config['source'], config['package_sha256'])
        datasets = importlib.import_module('datasets')
        misc = importlib.import_module('util.misc')
        training = self.phase == 'train'
        dataset = DetrDataset(datasets.build_dataset(
            image_set='train' if training else 'val', args=Namespace(**config['args'])))
        if env.distributed:
            sampler = DistributedSampler(dataset, shuffle=training)
        else:
            sampler = RandomSampler(dataset) if training else SequentialSampler(dataset)
        loader = DataLoader(dataset, batch_size=config[self.phase]['batch_size'],
                            sampler=sampler, drop_last=training,
                            collate_fn=misc.collate_fn, num_workers=config['num_workers'])
        return DetrLoader(loader)
