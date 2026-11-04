"""Load only the selected vendor's compiled UP kernels, on first use."""
import importlib
from functools import lru_cache

from up.runtime import get_runtime


@lru_cache(maxsize=None)
def load_extension(platform):
    """CUDA-compatible device names do not identify a binary's vendor."""
    if platform == 'metax_cuda':
        from torch import _utils_internal
        if not getattr(_utils_internal, 'USE_MACA', False):
            raise RuntimeError('MetaX native kernels require MACA PyTorch')
        module_name = 'up.platform.metax_cuda._C'
    elif platform == 'nvidia':
        from torch import _utils_internal
        if getattr(_utils_internal, 'USE_MACA', False):
            raise RuntimeError('NVIDIA native kernels require NVIDIA PyTorch')
        module_name = 'up.extensions._C'
    else:
        raise NotImplementedError('{} has no compiled UP extension binding'.format(platform))
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        if platform == 'nvidia':
            # Retain the original installation path; never try it for MetaX.
            try:
                return importlib.import_module('up_extensions._C')
            except ModuleNotFoundError as fallback_error:
                if fallback_error.name not in ('up_extensions', 'up_extensions._C'):
                    raise
        raise RuntimeError('{} native extension is not built. In the matching vendor '
                           'environment run UP_BUILD_PLATFORM={} python setup.py '
                           'build_ext --inplace'.format(platform, platform)) from error


class _LazyExtension:
    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        # 共用原 Python/autograd 包装，但只加载当前厂商编译的内核。
        extension = load_extension(get_runtime().platform)
        return getattr(getattr(extension, self.name), name)


roi_align = _LazyExtension('roi_align')
psroi_align = _LazyExtension('psroi_align')
psroi_pooling = _LazyExtension('psroi_pooling')
naive_nms = _LazyExtension('naive_nms')
softer_nms = _LazyExtension('softer_nms')
deform_conv_v1 = _LazyExtension('deform_conv_v1')
focal_loss = _LazyExtension('focal_loss')
cross_focal_loss = _LazyExtension('cross_focal_loss')
overlaps = _LazyExtension('overlaps')
roipoint_pool3d = _LazyExtension('roipoint_pool3d')
roiaware_pool3d = _LazyExtension('roiaware_pool3d')
iou3d_nms = _LazyExtension('iou3d_nms')
