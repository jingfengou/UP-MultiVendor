"""Finite epoch loading lets close-mosaic and epoch resume reach every worker."""
import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from up.data.data_builder import BaseDataLoaderBuilder
from up.utils.env.dist_helper import env
from up.utils.general.registry_factory import DATA_BUILDER_REGISTY, DATALOADER_REGISTRY, DATASET_REGISTRY


@DATASET_REGISTRY.register('yolo11_coco')
def build_dataset(**kwargs):
    from .data import Yolo11CocoDataset
    return Yolo11CocoDataset(**kwargs)


class EpochBatchSampler:
    def __init__(self, batch_sampler):
        self.batch_sampler = batch_sampler
        self.batch_size = batch_sampler.batch_size
        self.start_iter = 0

    def set_start_iter(self, start_iter):
        self.start_iter = start_iter

    def __len__(self):
        return len(self.batch_sampler)

    def __iter__(self):
        return iter(self.batch_sampler)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed)
    np.random.seed(seed)


@DATALOADER_REGISTRY.register('yolo11')
class Yolo11Loader(DataLoader):
    def __init__(self, dataset, batch_sampler, num_workers=4, seed=131):
        self.seed = seed
        generator = torch.Generator().manual_seed(seed + env.rank)
        super().__init__(dataset, batch_sampler=batch_sampler, num_workers=num_workers,
                         collate_fn=dataset.collate_fn, worker_init_fn=seed_worker,
                         generator=generator, persistent_workers=False)

    def get_epoch_size(self):
        return len(self.batch_sampler)

    def get_data_size(self):
        return self.get_epoch_size()

    def set_epoch(self, epoch):
        sampler = self.batch_sampler.batch_sampler.sampler
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)
        # Worker seeds depend on epoch/rank, not on how many validation iterators ran.
        self.generator.manual_seed(self.seed + env.rank + epoch * 100003)


@DATA_BUILDER_REGISTY.register('yolo11')
class Yolo11DataBuilder(BaseDataLoaderBuilder):
    def build_sampler(self, cfg_sampler, dataset):
        if self.phase == 'test':
            from ultralytics.data.build import ContiguousDistributedSampler
            # A rectangular batch must stay together. UP's usual strided test
            # sampler would mix different padded shapes from different batches.
            return ContiguousDistributedSampler(dataset, num_replicas=env.world_size,
                                                rank=env.rank, batch_size=dataset.batch_size)
        return torch.utils.data.DistributedSampler(dataset, num_replicas=env.world_size,
                                                   rank=env.rank, seed=self.cfg['dataloader']['kwargs']['seed'])

    def build_batch_sampler(self, cfg_batch_sampler, dataset):
        wrapped = super().build_batch_sampler(cfg_batch_sampler, dataset)
        return EpochBatchSampler(wrapped.batch_sampler)

    def build_dataset(self, cfg_dataset):
        cfg_dataset = copy.deepcopy(cfg_dataset)
        batch = self.update_cfg()['batch_sampler']['kwargs']['batch_size']
        cfg_dataset['kwargs']['batch_size'] = batch
        return super().build_dataset(cfg_dataset)
