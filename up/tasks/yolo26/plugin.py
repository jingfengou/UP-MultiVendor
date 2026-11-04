"""Use official YOLO26 algorithms; UP supplies the selected device/native boundary."""
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import os
from pathlib import Path

from .source import load_source, sha256


class OfficialYOLO26:
    """One detection or classification recipe, loaded lazily in a fresh process.

    The official Trainer owns data augmentation, loss, optimizer, AMP, EMA and
    checkpoint recovery. Its Validator/Predictor/Exporter own postprocessing.
    No YOLO11 adapter, loss or parameter grouping is reused here.
    """

    def __init__(self, config, source, platform=None):
        self.config = deepcopy(config)
        self.task = config['plugin']['task']
        if self.task not in ('detect', 'classify'):
            raise ValueError('YOLO26 task must be detect or classify')
        self.platform = platform or config['runtime']['platform']
        if self.platform not in ('nvidia', 'metax_cuda'):
            raise NotImplementedError('This official CUDA-compatible trainer supports NVIDIA and MetaX')
        self.source = Path(source).resolve()
        self.runtime = None
        self._loaded = False

    def load(self):
        """Verify the source pin before importing the official model and trainers."""
        if not self._loaded:
            os.environ['YOLO_AUTOINSTALL'] = 'false'
            load_source(self.source, self.config['official']['package_sha256'])
            from ultralytics.cfg import get_cfg
            # Catch misspelled or unsupported official arguments before using a GPU.
            get_cfg(overrides=self.recipe())
            self._loaded = True

    def recipe(self):
        """Return official arguments; batch is global, device IDs are process-local."""
        recipe = dict(self.config['train'])
        recipe.update(task=self.task, mode='train', model=self.config['plugin']['model'])
        return recipe

    def trainer_class(self):
        self.load()
        if self.task == 'detect':
            from ultralytics.models.yolo.detect import DetectionTrainer
            return DetectionTrainer
        from ultralytics.models.yolo.classify import ClassificationTrainer
        return ClassificationTrainer

    def check_optimizer(self, optimizer):
        """Check the optimizer actually built, including a resumed schedule's base LR."""
        if type(optimizer).__name__ != self.config['plugin']['optimizer_class']:
            raise RuntimeError('Official optimizer selection differs from the YAML contract')
        expected = self.config['plugin'].get('optimizer_initial_lr')
        if expected is not None:
            # LambdaLR keeps initial_lr even after decay or checkpoint recovery.
            # Checking args.lr0 alone misses the official auto optimizer override.
            for group in optimizer.param_groups:
                if 'initial_lr' not in group or not math.isclose(
                        group['initial_lr'], expected, rel_tol=1e-12, abs_tol=0.0):
                    raise RuntimeError('Optimizer base learning rate differs from the YAML contract')

    def build_model(self, pretrained=None):
        """Build common CPU initialization with the official task's get_model().

        Detection transfers official Objects365 weights to an 80-class head.
        Classification constructs a random 1000-class model, never a trained
        ImageNet checkpoint. Create this once and copy the same file to each host.
        """
        import torch
        self.load()
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import load_checkpoint
        from ultralytics.utils.torch_utils import init_seeds

        if torch.cuda.is_initialized():
            raise RuntimeError('Create common initialization in a fresh CPU-only process')
        options = self.recipe()
        init_seeds(options['seed'], deterministic=options['deterministic'])
        trainer = object.__new__(self.trainer_class())
        trainer.args = get_cfg(overrides=options)
        trainer.data = {'nc': self.config['plugin']['classes'], 'channels': 3}
        if self.task == 'detect':
            if pretrained is None or sha256(pretrained) != self.config['initialization']['sha256']:
                raise ValueError('Supply the SHA256-matched official Objects365 initialization')
            weights, _ = load_checkpoint(str(pretrained), device='cpu')
            model = trainer.get_model(cfg=weights.yaml, weights=weights, verbose=False)
        else:
            if pretrained is not None:
                raise ValueError('The ImageNet recipe starts from random weights')
            model = trainer.get_model(cfg=options['model'], verbose=False)
        model.args = vars(trainer.args)
        return model

    def prepare_initialization(self, output, pretrained=None):
        import torch
        output = Path(output)
        receipt = output.with_suffix(output.suffix + '.json')
        if output.exists() or receipt.exists():
            raise FileExistsError('Common initialization already exists: ' + str(output))
        model = self.build_model(pretrained)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Official loaders expect their model object, not UP's dictionary-batch adapter.
        with output.open('xb') as stream:
            torch.save({'model': model, 'train_args': self.recipe(), 'epoch': -1,
                        'version': 'pinned-official'}, stream)
        metadata = {'sha256': sha256(output), 'task': self.task, 'seed': self.config['train']['seed'],
                    'package_sha256': self.config['official']['package_sha256'],
                    'parameters': sum(p.numel() for p in model.parameters())}
        with receipt.open('x', encoding='utf-8') as stream:
            json.dump(metadata, stream, indent=2)
        return metadata

    @contextmanager
    def execution(self, purpose):
        """Keep training eval on training runtime; bind standalone work to inference."""
        self.load()
        import torchvision
        from up.runtime import setup_runtime

        config = deepcopy(self.config)
        config['runtime']['platform'] = self.platform
        self.runtime = setup_runtime(config, purpose=purpose)
        self.runtime.set_device(int(os.environ.get('LOCAL_RANK', '0')))
        self.runtime.configure_dnn(config['runtime']['dnn'])
        previous_nms = torchvision.ops.nms
        if self.platform == 'metax_cuda':
            # set_device already requires the native MACA/cuBridge extension.
            # End-to-end detection bypasses NMS; other official paths use this
            # continuous-coordinate native binding, never the old Torch fallback.
            from up.platform.metax_cuda.nms import nms
            torchvision.ops.nms = nms
        try:
            yield
        finally:
            torchvision.ops.nms = previous_nms

    def train(self, data, weights, output, resume=False):
        """Run the untouched official trainer under externally launched torchrun."""
        import torch
        world = int(os.environ.get('WORLD_SIZE', '1'))
        expected = self.config['runtime']['expected_world_size']
        if world != expected or int(os.environ.get('LOCAL_WORLD_SIZE', '1')) != world:
            raise ValueError('Launch one host with torchrun --nproc_per_node=' + str(expected))
        if self.config['train']['batch'] % world:
            raise ValueError('Global batch must be divisible by world size')
        weights, output = Path(weights).resolve(), Path(output).resolve()
        if not weights.is_file():
            raise FileNotFoundError(weights)
        if not resume and int(os.environ.get('RANK', '0')) == 0 and output.exists():
            raise FileExistsError('Choose a new training output directory')
        self.load()
        from ultralytics.nn.tasks import load_checkpoint
        from ultralytics.utils.torch_utils import unwrap_model
        reference, checkpoint = load_checkpoint(str(weights), device='cpu')
        if not resume:
            if checkpoint.get('epoch') != -1:
                raise ValueError('Fresh training requires the common initialization')
            receipt = json.loads(weights.with_suffix(weights.suffix + '.json').read_text())
            if (receipt['sha256'] != sha256(weights) or receipt['task'] != self.task
                    or receipt['package_sha256'] != self.config['official']['package_sha256']
                    or receipt['seed'] != self.config['train']['seed']):
                raise ValueError('Common initialization does not match the selected recipe')
        options = self.recipe()
        options.update(model=str(weights), data=str(Path(data).resolve()),
                       device=','.join(str(i) for i in range(world)), project=str(output.parent),
                       name=output.name, exist_ok=True, resume=str(weights) if resume else False)
        expected_state = {k: v.detach().cpu().clone() for k, v in reference.state_dict().items()}
        del reference

        def prepared(trainer):
            # Official resume restores saved args. Reject a different budget/recipe
            # instead of silently continuing with the checkpoint's old schedule.
            for key, value in self.config['train'].items():
                if getattr(trainer.args, key) != value:
                    raise ValueError('Checkpoint/trainer recipe differs at ' + key)
            if bool(trainer.amp) != options['amp']:
                raise RuntimeError('Official AMP check changed the requested precision')
            self.check_optimizer(trainer.optimizer)
            if not resume:
                state = unwrap_model(trainer.model).state_dict()
                if state.keys() != expected_state.keys() or any(
                        not torch.equal(value.detach().cpu(), expected_state[key])
                        for key, value in state.items()):
                    raise RuntimeError('Trainer did not load the common initialization exactly')
            expected_state.clear()
            self.runtime.configure_dnn(self.config['runtime']['dnn'])

        with self.execution('training'):
            trainer = self.trainer_class()(overrides=options)
            trainer.add_callback('on_pretrain_routine_end', prepared)
            try:
                trainer.train()
            finally:
                if torch.distributed.is_initialized():
                    torch.distributed.destroy_process_group()
        return trainer

    def validate(self, data, weights, output):
        """Full independent validation with the same official task evaluator."""
        self.load()
        if self.task == 'detect':
            from ultralytics.models.yolo.detect import DetectionValidator as Validator
        else:
            from ultralytics.models.yolo.classify import ClassificationValidator as Validator
        output = Path(output).resolve()
        if output.exists():
            raise FileExistsError(output)
        options = self.recipe()
        options.update(self.config['validation'], model=str(Path(weights).resolve()),
                       data=str(Path(data).resolve()), mode='val', device='0',
                       project=str(output.parent), name=output.name)
        with self.execution('inference'):
            return Validator(args=options, save_dir=output)()

    def predict(self, images, weights, output):
        """Use official preprocessing/postprocessing and stream saved predictions."""
        self.load()
        from ultralytics import YOLO
        output = Path(output).resolve()
        if output.exists():
            raise FileExistsError(output)
        with self.execution('inference'):
            model = YOLO(str(Path(weights).resolve()), task=self.task)
            if self.task == 'classify' and not hasattr(model.model, 'transforms'):
                # Trained classifiers carry their validation transform. Common
                # initialization has not built a dataloader yet; use that same
                # official Resize/CenterCrop transform for its first prediction.
                from ultralytics.data.augment import classify_transforms
                model.model.transforms = classify_transforms(self.config['train']['imgsz'])
            # Consume the stream so large image/video sources do not accumulate in RAM.
            for _ in model.predict(source=images, device='0', stream=True, save=True,
                                   project=str(output.parent), name=output.name,
                                   imgsz=self.config['train']['imgsz']):
                pass

    def export(self, weights):
        """Official ONNX export; the graph is written alongside the supplied weights."""
        self.load()
        from ultralytics import YOLO
        if Path(weights).with_suffix('.onnx').exists():
            raise FileExistsError(Path(weights).with_suffix('.onnx'))
        with self.execution('inference'):
            model = YOLO(str(Path(weights).resolve()), task=self.task)
            return model.export(format='onnx', device='0', imgsz=self.config['train']['imgsz'],
                                **self.config['export'])
