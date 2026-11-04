"""YOLO11 recipe in UP's train/evaluate/save path, with vendor-neutral device calls.

Only recipe differences live here. The shared BaseRunner and other four models
retain their training loops. All devices use this same code and native Torch.
"""
import random

import numpy as np
import torch
import torch.distributed as dist

from up.runner.base_runner import BaseRunner
from up.utils.env.dist_helper import env
from up.utils.general.registry_factory import RUNNER_REGISTRY
from .recipe import Yolo11EMA, Yolo11Schedule, build_optimizer


@RUNNER_REGISTRY.register('yolo11')
class Yolo11Runner(BaseRunner):
    def build_optimizer(self, cfg_optimizer):
        global_batch = self.train_batch_size * env.world_size
        expected = self.config['trainer']['global_batch']
        if global_batch != expected:
            raise ValueError('YOLO11 requires global batch {}; got {}'.format(expected, global_batch))
        return build_optimizer(self.model, global_batch, **cfg_optimizer)

    def build_lr_scheduler(self, cfg_lr_scheduler):
        return Yolo11Schedule(self.optimizer, self.train_epoch_size,
                              self.config['trainer']['max_epoch'], **cfg_lr_scheduler)

    def build_ema(self):
        self.ema = Yolo11EMA(self.model)

    def resume_model(self):
        super().resume_model()
        ckpt = self.saver.load_pretrain_or_resume()
        if not ckpt.get('ema'):
            self.ema.model.load_state_dict(self.model.state_dict())
        self.successful_updates = ckpt.get('yolo11', {}).get('successful_updates', 0)
        self.amp_skips = ckpt.get('yolo11', {}).get('amp_skips', 0)
        self.best_metric = ckpt.get('yolo11', {}).get('best_metric', -1.0)
        self._resume_states = ckpt.get('rank_states')

    def prepare_dist_model(self):
        super().prepare_dist_model()
        if self.training and self.start_iter % self.train_epoch_size:
            raise ValueError('YOLO11 resumes from complete-epoch checkpoints; partial epochs are not accepted')
        if self.training and self._resume_states:
            if len(self._resume_states) != env.world_size:
                raise ValueError('Resume must keep the original world size')
            state = self._resume_states[env.rank]
            random.setstate(state['python'])
            name, keys, position, has_gauss, cached_gaussian = state['numpy']
            np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, has_gauss, cached_gaussian))
            torch.set_rng_state(state['torch'].cpu())
            self.runtime.module.set_rng_state(state['device'].cpu())
            if self.fp16:
                self.scaler.load_state_dict(state['scaler'])
            # DDP broadcast_buffers=False: restore each rank's own local BN state.
            model = self.model.module if hasattr(self.model, 'module') else self.model
            for name, buffer in model.named_buffers():
                buffer.copy_(state['buffers'][name].to(buffer.device))

    def batch2device(self, batch):
        batch = super().batch2device(batch)
        # The official collator yields RGB uint8. Divide exactly once, on the device.
        batch['image'] = batch['image'] / 255.0
        return batch

    def backward(self, loss):
        # Official DetectionTrainer multiplies batch-summed loss by world_size
        # before DDP averages gradients. Keep the same scale on all vendors.
        return super().backward(loss * env.world_size)

    def update(self):
        if self.fp16:
            self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        previous_scale = self.scaler.get_scale() if self.fp16 else None
        super().update()
        skipped = self.fp16 and self.scaler.get_scale() < previous_scale
        self.amp_skips += int(skipped)
        self.successful_updates += int(not skipped)

    def train(self):
        if any(h['type'].startswith('auto_save_best') for h in self.config.get('hooks', [])):
            raise ValueError('YOLO11 saves complete best checkpoints itself; remove auto_save_best hooks')
        self._hooks('before_train')
        self.model.to(self.device).train()
        loader = self.data_loaders['train']
        epochs = self.config['trainer']['max_epoch']
        close = self.config['trainer']['close_mosaic']
        for iter_idx in range(self.start_iter, self.max_iter):
            if iter_idx % self.train_epoch_size == 0:
                epoch = iter_idx // self.train_epoch_size
                if close and epoch >= epochs - close:
                    loader.dataset.close_strong_augmentation()
                loader.set_epoch(epoch)
                if hasattr(self, 'data_iterators'):
                    self.data_iterators.pop('train', None)
            loss = self.forward_train(self.get_batch('train'))
            self.backward(loss)
            self.update()
            # Match the official trainer: EMA advances once per attempted update,
            # including overflow skips; successful optimizer updates are logged apart.
            self.ema.step(self.model)
            self.lr_scheduler.step()
            if self.is_test(iter_idx):
                if env.distributed:
                    # Official validator uses rank-0 EMA buffers on every rank.
                    for buffer in self.ema.model.buffers():
                        dist.broadcast(buffer, src=0)
                metrics = self.evaluate()
                value = metrics['bbox.AP'] if env.is_master() else None
                score = [float(value) if value is not None else None]
                if env.distributed:
                    dist.broadcast_object_list(score, src=0)
                if score[0] > self.best_metric:
                    self.best_metric = score[0]
                    self.save(auto_save='best', lns=False)
            if self.is_save(iter_idx):
                self.save()
        self._hooks('after_train')

    def get_dump_dict(self):
        result = super().get_dump_dict()
        model = self.model.module if hasattr(self.model, 'module') else self.model
        numpy_state = np.random.get_state()
        # Use plain values rather than pickled NumPy classes so default safe
        # torch.load can restore this checkpoint on current vendor Torch builds.
        numpy_state = (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])
        local = {'python': random.getstate(), 'numpy': numpy_state,
                 'torch': torch.get_rng_state(), 'device': self.runtime.module.get_rng_state(),
                 'buffers': {key: value.detach().cpu() for key, value in model.named_buffers()}}
        if self.fp16:
            local['scaler'] = self.scaler.state_dict()
        states = [None] * env.world_size
        if env.distributed:
            dist.all_gather_object(states, local)
        else:
            states[0] = local
        result.update(epoch=self.cur_iter // self.train_epoch_size,
                      rank_states=states,
                      yolo11={'successful_updates': self.successful_updates, 'amp_skips': self.amp_skips,
                              'best_metric': self.best_metric})
        return result
