"""Python install by default; explicitly build UP kernels for one vendor."""
import copy
import os
from pathlib import Path
import re
from setuptools import find_packages, setup


def extensions():
    platform = os.environ.get('UP_BUILD_PLATFORM', 'none')
    if platform == 'none':
        return [], {}
    if platform not in ('nvidia', 'metax_cuda'):
        raise ValueError('UP_BUILD_PLATFORM accepts none, nvidia or metax_cuda')
    # Import build tools only for an explicit native build; no GPU probe.
    from torch import _utils_internal
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    is_maca = bool(getattr(_utils_internal, 'USE_MACA', False))
    if is_maca != (platform == 'metax_cuda'):
        raise RuntimeError('UP_BUILD_PLATFORM={} does not match the installed PyTorch vendor'.format(platform))
    if CUDA_HOME is None:
        raise RuntimeError('Native build needs the matching vendor PyTorch and CUDA toolkit / MACA cuBridge')
    if platform == 'metax_cuda':
        if not (Path(CUDA_HOME) / 'bin' / 'cucc').is_file():
            raise RuntimeError('MetaX build needs MACA cuBridge (CUDA_HOME/bin/cucc)')
        # cuBridge uses this CUDA compatibility macro, not an NVIDIA device.
        # Specifying it also avoids probing active GPUs during a CPU-only build.
        os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '8.0')
    source = Path(__file__).parent / 'up' / 'extensions' / 'csrc'
    files = sorted(str(p) for pattern in ('*.cpp', '*.cu') for p in source.rglob(pattern))
    module_name = 'up.extensions._C' if platform == 'nvidia' else 'up.platform.metax_cuda._C'
    module = CUDAExtension(module_name, files, include_dirs=[str(source)],
                           define_macros=[('WITH_CUDA', None)],
                           extra_compile_args={'cxx': [], 'nvcc': [
                               '-DCUDA_HAS_FP16=1', '-D__CUDA_NO_HALF_OPERATORS__',
                               '-D__CUDA_NO_HALF_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__']})
    class VendorBuildExtension(BuildExtension):
        def finalize_options(self):
            super().finalize_options()
            # Never reuse another vendor's object files when building in one checkout.
            self.build_temp = str(Path(self.build_temp) / platform)
            if platform == 'metax_cuda':
                # Non-root containers may have HOME=/; cuBridge needs writable scratch.
                scratch = Path(self.build_temp).resolve() / 'cubridge'
                scratch.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault('CUBRIDGE_HOME', str(scratch))

        def build_extension(self, extension):
            if platform != 'metax_cuda':
                return super().build_extension(extension)
            compile_sources = self.compiler.compile
            unfused_sources = {
                (source / 'roi_align' / 'roi_align_kernel.cu').resolve(),
                (source / 'nms' / 'continuous_nms.cu').resolve(),
            }

            def compile_with_roi_flags(sources, output_dir=None, macros=None,
                                       include_dirs=None, debug=0, extra_preargs=None,
                                       extra_postargs=None, depends=None):
                # RoIAlign keeps its validated rounding; NMS must retain the old
                # Torch IoU arithmetic at threshold boundaries.
                # Separate object directories also keep Ninja's flag caches isolated.
                roi = [p for p in sources if Path(p).resolve() in unfused_sources]
                others = [p for p in sources if Path(p).resolve() not in unfused_sources]
                objects = []
                for group, is_roi in ((others, False), (roi, True)):
                    if not group:
                        continue
                    flags = copy.deepcopy(extra_postargs)
                    directory = output_dir
                    if is_roi:
                        flags['nvcc'].append('--fmad=false')
                        directory = str(Path(output_dir) / 'roi_align_fmad_off')
                    objects.extend(compile_sources(
                        group, output_dir=directory, macros=macros,
                        include_dirs=include_dirs, debug=debug,
                        extra_preargs=extra_preargs, extra_postargs=flags,
                        depends=depends))
                return objects

            # BuildExtension has already configured the vendor compiler at this point.
            self.compiler.compile = compile_with_roi_flags
            try:
                return super().build_extension(extension)
            finally:
                self.compiler.compile = compile_sources

    return [module], {'build_ext': VendorBuildExtension}


ext_modules, cmdclass = extensions()
version = re.search(r'__version__ = "(.*?)"', Path('up/__init__.py').read_text(encoding='utf-8')).group(1)
setup(name='UP', version=version, packages=find_packages(), python_requires='>=3.10',
      description='UP with config-selected accelerator runtimes and official model plugins',
      ext_modules=ext_modules, cmdclass=cmdclass, include_package_data=True,
      package_data={'up.tasks.det.plugins.detr.official': ['LICENSE']},
      extras_require={'yolo11': ['ultralytics==8.3.225'],
                      'onnx': ['onnx>=1.16,<2', 'onnxruntime>=1.18,<2']})
