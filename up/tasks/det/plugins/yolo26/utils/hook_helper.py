"""YOLO26 epoch transitions registered as UP hooks."""
import torch
from torch import nn

from up.utils.env.dist_helper import env
from up.utils.general.hook_helper import Hook
from up.utils.general.registry_factory import HOOK_REGISTRY
from ..models.model import official_model


@HOOK_REGISTRY.register('official_yolo26')
class YOLO26EpochHook(Hook):
    def before_train(self):
        runner = self.runner_ref()
        if runner.start_iter % runner.train_epoch_size:
            raise ValueError('Official YOLO26 resume requires an epoch boundary')
        from ultralytics.utils.torch_utils import EarlyStopping
        self.stopper = EarlyStopping(patience=official_model(runner.model).options.patience)

    def after_eval(self, metrics):
        runner = self.runner_ref()
        if not runner.training:
            return
        stop = self.stopper(runner.cur_epoch(), metrics['fitness']) if env.is_master() else False
        if env.distributed:
            decision = [stop]
            torch.distributed.broadcast_object_list(decision, src=0)
            stop = decision[0]
        runner.stop_training = stop

    def before_epoch(self, epoch):
        runner = self.runner_ref()
        adapter = official_model(runner.model)
        network = adapter.network
        network.train()
        for name, module in network.named_modules():
            if '.dfl' in name and isinstance(module, nn.BatchNorm2d):
                module.eval()
        if hasattr(network, 'set_epoch'):
            network.set_epoch(epoch)
        loader = runner.data_loaders['train']
        close_epoch = adapter.options.epochs - adapter.options.close_mosaic
        resuming_closed = epoch == runner.start_iter // runner.train_epoch_size and epoch > close_epoch
        if adapter.options.close_mosaic and (epoch == close_epoch or resuming_closed):
            dataset = loader.dataset.official
            if hasattr(dataset, 'mosaic'):
                dataset.mosaic = False
            if hasattr(dataset, 'close_mosaic'):
                from copy import copy
                dataset.close_mosaic(hyp=copy(adapter.options))
                loader.reset()

    def after_epoch(self, epoch):
        runner = self.runner_ref()
        adapter = official_model(runner.model)
        criterion = getattr(adapter.network, 'criterion', None)
        if getattr(adapter.options, 'o2m', 1.0) != 1.0 and hasattr(criterion, 'update'):
            criterion.update()
        if runner.ema is not None:
            runner.ema.official.update_attr(adapter.network,
                include=['yaml', 'nc', 'args', 'names', 'stride', 'class_weights'])
            if env.distributed:
                for buffer in runner.ema.official.ema.buffers():
                    torch.distributed.broadcast(buffer, src=0)
