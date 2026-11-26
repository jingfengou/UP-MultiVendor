from up.utils.general.registry_factory import HOOK_REGISTRY
from up.utils.general.hook_helper import Hook
from up.utils.env.dist_helper import env
from up.utils.general.log_helper import default_logger as logger
from up.tasks.det.plugins.yolox.models.preprocess.multiscale import Multiscale


__all__ = ['YoloxNoaug', 'YoloxBatchResize']


@HOOK_REGISTRY.register('yolox_noaug')
class YoloxNoaug(Hook):
    def __init__(self, runner, no_aug_epoch=15, max_epoch=300, transformer=[],
                 test_freq=1, save_freq=1, switch_before_data=False):
        super(YoloxNoaug, self).__init__(runner)
        self.no_aug_epoch = no_aug_epoch
        self.max_epoch = max_epoch
        self.transformer = transformer
        self.flag = False
        self.test_freq = test_freq
        self.save_freq = save_freq
        self.switch_before_data = switch_before_data

    def switch(self, cur_iter):
        runner = self.runner_ref()
        if self.flag or cur_iter < runner.data_loaders['train'].get_epoch_size() * (
                self.max_epoch - self.no_aug_epoch):
            return

        logger.info(f"rebuild dataset transformer cfg {self.transformer}")
        runner.config['dataset']['train']['dataset']['kwargs']['transformer'] = self.transformer
        del runner.data_loaders
        if hasattr(runner, 'data_iterators'):
            runner.data_iterators.pop('train', None)
        import gc
        gc.collect()
        if not hasattr(runner, 'data_iterators'):
            runner.data_iterators = {}
        logger.info("rebuild dataloader")
        runner.build_dataloaders()
        runner.data_iterators['train'] = iter(runner.data_loaders["train"])
        try:
            if env.world_size > 1:
                runner.model.module.yolox_post.use_l1 = True
            else:
                runner.model.yolox_post.use_l1 = True
        except:  # noqa
            pass
        runner.test_freq = self.test_freq
        runner.save_freq = self.save_freq
        self.flag = True

    def before_data(self, cur_iter):
        if self.switch_before_data:
            self.switch(cur_iter)

    def before_forward(self, cur_iter, input):
        if not self.switch_before_data:
            self.switch(cur_iter)


@HOOK_REGISTRY.register('yolox_batch_resize')
class YoloxBatchResize(Hook):
    def __init__(self, runner, input_size=640, random_size=(15, 25), interval=10, scale_step=32):
        super(YoloxBatchResize, self).__init__(runner)
        self.input_size = input_size
        self.interval = interval
        self.iter_count = 0
        self.resizer = Multiscale(
            random_size=random_size,
            scale_step=scale_step,
            scale_rate=interval,
        )
        self.resizer.cur_size = [input_size, input_size]

    def before_forward(self, cur_iter, input):
        if self.iter_count and self.iter_count % self.interval == 0:
            self.resizer.cur_size = self.resizer.get_target_size(input)
        self.resizer.rescale(input)
        self.iter_count += 1
