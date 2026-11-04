"""Export selected UP networks without building datasets or a training Runner."""
import copy
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

from up.models.model_helper import ModelHelper
from up.runtime.state import evaluation_mode
from up.utils.general.global_flag import ALIGNED_FLAG
from up.utils.general.cfg_helper import validate_model_coordinates
from up.utils.general.registry_factory import TOONNX_REGISTRY
from up.utils.general.saver_helper import Saver


def build_model(cfg):
    model = ModelHelper(copy.deepcopy(cfg['net']))
    saver_cfg = copy.deepcopy(cfg.get('saver', {}))
    if saver_cfg:
        saver = Saver(saver_cfg)
        checkpoint = saver.load_pretrain_or_resume()
        ema = checkpoint.get('ema') or {}
        state = ema.get('ema_state_dict', checkpoint['model'])
        if state:
            # Export must not silently deploy random, unmatched parameters.
            model.load_state_dict(state, strict=True)
    return model


@contextmanager
def export_mode(model):
    """Temporarily select raw deployment outputs; restore caller's model flags."""
    states = [(m, 'deploy' in m.__dict__, getattr(m, 'deploy', None)) for m in model.modules()]
    try:
        for module, _, _ in states:
            module.deploy = True
        with evaluation_mode(model):
            yield
    finally:
        for module, existed, value in states:
            if existed:
                module.deploy = value
            else:
                del module.deploy


class ExportWrapper(nn.Module):
    def __init__(self, model, task, add_softmax=False):
        super().__init__()
        self.model, self.task, self.add_softmax = model, task, add_softmax
        if task not in ('cls', 'det'):
            raise ValueError('This branch exports cls and det models')
        self.output_names = None

    def forward(self, image):
        b, _, h, w = image.shape
        # Export has one fixed image extent, with no left/top padding. Both
        # detection adapters share UP's eight-field image_info convention.
        output = self.model({'image': image, 'image_info': [[h, w, 1., h, w, False, 0, 0]] * b})
        if self.task == 'cls':
            # Original UP exports scores, optionally applying another softmax.
            value = output['scores']
            if isinstance(value, list):
                value = torch.cat(value, 1)
            if self.add_softmax:
                value = value.softmax(1)
            self.output_names = ['scores']
            return (value,)
        names = sorted(name for name in output if name.startswith('blobs_'))
        if not names:
            raise RuntimeError('Detection adapter did not provide raw ONNX outputs')
        self.output_names = names
        return tuple(output[name] for name in names)


@TOONNX_REGISTRY.register('base')
class BaseToOnnx:
    def __init__(self, cfg, save_prefix, input_size=None, model=None, input_channel=3):
        self.cfg = copy.deepcopy(cfg)
        validate_model_coordinates(cfg)
        self.input_size = input_size or tuple(cfg.get('export', {}).get('input_size', (input_channel, 224, 224)))
        if len(self.input_size) != 3 or min(self.input_size) <= 0:
            raise ValueError('input_size must be positive CxHxW')
        self.output = Path(save_prefix)
        if self.output.suffix != '.onnx':
            self.output = Path('toonnx') / (str(save_prefix) + '.onnx')
        self.model = model

    def process(self):
        import onnx
        if self.output.exists():
            raise FileExistsError('Refusing to overwrite ONNX artifact: ' + str(self.output))
        self.output.parent.mkdir(parents=True, exist_ok=True)
        # Use a copy: exporting must not move a caller's live optimizer/model
        # tensors to CPU or change their dtype and parameter identity.
        model = copy.deepcopy(self.model) if self.model is not None else build_model(self.cfg)
        model = model.cpu().float()
        ALIGNED_FLAG.aligned = self.cfg.get('runtime', {}).get('aligned', False)
        ALIGNED_FLAG.offset = 0 if ALIGNED_FLAG.aligned else 1
        task = self.cfg.get('runtime', {}).get('task_names', 'det')
        wrapper = ExportWrapper(model, task, self.cfg.get('to_kestrel', {}).get('add_softmax', False))
        image = torch.randn(1, *self.input_size)
        with export_mode(model), torch.no_grad():
            reference = wrapper(image)
            torch.onnx.export(wrapper, image, str(self.output), input_names=['data'],
                              output_names=wrapper.output_names,
                              opset_version=self.cfg.get('export', {}).get('opset', 17),
                              dynamo=False, do_constant_folding=True)
        onnx.checker.check_model(str(self.output))
        if self.cfg.get('export', {}).get('verify', False):
            import numpy as np
            import onnxruntime as ort
            session = ort.InferenceSession(str(self.output), providers=['CPUExecutionProvider'])
            actual = session.run(None, {'data': image.numpy()})
            for want, got in zip(reference, actual):
                np.testing.assert_allclose(got, want.numpy(), rtol=1e-4, atol=1e-5)
        return str(self.output)
