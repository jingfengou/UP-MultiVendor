"""公共设备服务；由我们接 UP 调用，各显卡负责人适配不兼容的 SDK API。

接口与注册方式见 docs/platform/ADDING_VENDOR.md。

兼容本类接口的厂商 Torch 可直接复用。确需差异实现时，在交付插件
vendor_plugins/<vendor>_runtime.py 中继承 RuntimeBackend，只覆盖不兼容
的方法，再通过 register_backend(..., backend_class=..., replace=True) 注册。
插件在 setup_runtime 之前加载，示例见 docs/platform/ADDING_VENDOR.md。
普通 nn/functional/Tensor/autograd 操作仍由厂商 Torch 执行，不在这里重写。
"""
import importlib
import os

import torch


# 每项依次指定设备 API、通信后端、注册设备所需的扩展；厂商 Torch 须在启动前安装好。
# NVIDIA 和沐曦虽然都使用 cuda 名称，实际由各自环境中的 Torch/SDK 执行。
_PLATFORMS = {
    'nvidia': ('cuda', 'nccl', None),
    'ascend_npu': ('npu', 'hccl', 'torch_npu'),
    'cambricon_mlu': ('mlu', 'cncl', 'torch_mlu'),
    'metax_cuda': ('cuda', 'nccl', None),
}
_runtime = None
_backend_classes = {}


class RuntimeBackend:
    """厂商运行时父类；device_type 是设备名，platform 是厂商身份。

负责人首先验证 module 和 amp 是否满足 UP 的调用；兼容则不写子类。
module 的接口、单位与状态要求列在 __init__，其他方法各自列出适配要求。
通信仍走 torch.distributed；dist_backend 选择当前环境已注册的通信库。
"""

    def __init__(self, platform, *, purpose='training'):
        """选 SDK 和设备模块，不初始化设备、不创建模型。

        参数: platform 为注册名；purpose 固定为 training 或 inference。
        状态: module 指向 torch.cuda/npu/mlu；必要时由厂商子类换成兼容 adapter。
        module 需提供（NVIDIA 参考均为 torch.cuda 下同名函数）：
          is_available()/device_count()/current_device() -> bool/int/int；
          set_device(index)、get_device_properties(index).total_memory；
          get_device_name(device)、memory_allocated(device=None)、max_memory_allocated(device=None)；
          reset_peak_memory_stats()、empty_cache()、synchronize()；
          manual_seed(seed)、get_rng_state()、set_rng_state(state)。
        显存单位为 byte；RNG 保存/恢复使用本厂商状态，不跨厂商复制状态字节。
        empty_cache 只释放未用缓存，不能破坏活跃 Tensor；同步要等待真实设备执行。
        负责人只在这些 API 不兼容时提供 adapter，不要吞掉 SDK 初始化错误。
        """
        if platform not in _PLATFORMS:
            raise ValueError('Unknown runtime.platform: {}'.format(platform))
        if purpose not in ('training', 'inference'):
            raise ValueError('purpose must be training or inference')
        self.platform = platform
        self.purpose = purpose
        self._native_ops = None
        self._native_checked = False
        self._bound = False
        self.pid = os.getpid()
        self.device_type, self.dist_backend, extension = _PLATFORMS[platform]
        if extension is not None:
            # 先导入 torch_npu/torch_mlu 注册设备，之后才能取得 torch.npu/torch.mlu。
            try:
                importlib.import_module(extension)
            except ImportError as error:
                raise RuntimeError('{} requires a matching {} installation: {}'.format(
                    platform, extension, error)) from error
        self.module = getattr(torch, self.device_type, None)
        if self.module is None:
            raise RuntimeError('{} does not provide torch.{}'.format(platform, self.device_type))

    @property
    def device(self):
        """返回当前进程绑定的 torch.device，不分配新卡或迁移模型。

        NVIDIA 参考: torch.device('cuda', torch.cuda.current_device())。
        厂商需保证 device_type 与 Tensor 的实际设备一致，逻辑编号以可见卡为准。
        """
        self._check_native_extension()
        self._bound = True
        return torch.device(self.device_type, self.module.current_device())

    @property
    def amp(self):
        """返回当前厂商的 autocast/GradScaler 接口，兼容时复用 module.amp。

        NVIDIA 参考: torch.cuda.amp。UP 用 autocast(enabled=bool) 上下文，
        GradScaler(enabled=bool) 的 scale/step/update/state_dict/load_state_dict。
        若厂商签名不同，负责人在子类此属性返回 adapter；保留溢出检测、跳步
        和 scaler 恢复语义，不把关闭 AMP 当成实现完成。
        """
        # AMP 使用当前设备自己的实现，模型和优化器仍按原来的 PyTorch 流程工作。
        self._bound = True
        return self.module.amp

    def set_device(self, local_rank):
        """local_rank: 可见逻辑卡号 int；返回 None，绑定当前进程的设备。

        NVIDIA 参考: torch.cuda.is_available/device_count/set_device。
        厂商需保留不可用/越界报错，在进程组和模型创建前完成绑卡。
        """
        if not self.module.is_available():
            raise RuntimeError('{} is unavailable in this PyTorch environment'.format(self.platform))
        if local_rank < 0 or local_rank >= self.module.device_count():
            raise RuntimeError('Local rank {} exceeds visible device count'.format(local_rank))
        self._check_native_extension()
        self.module.set_device(local_rank)
        self._bound = True

    def _check_native_extension(self):
        # C500 jobs must use the compiled UP operators. Check once per process,
        # before DDP/model creation; importing UP or querying capabilities stays lazy.
        if self.platform == 'metax_cuda' and not self._native_checked:
            from up.platform.metax_cuda.native_ops import require_native_extension
            require_native_extension()
            self._native_checked = True

    @property
    def native_ops(self):
        """惰性返回本厂商 NativeOps，缓存后复用；此装配流程由我们维护。

        厂商只改 up/platform/<vendor>/native_ops.py，不在这里根据 Tensor
        恰好叫 cuda 就选 NVIDIA，也不在训练内 eval 时切换用途或算子供应者。
        """
        if self._native_ops is None:
            from up.platform.native import build_native_ops
            self._native_ops = build_native_ops(self)
        self._bound = True
        return self._native_ops

    def sync_batch_norm(self, input, weight, bias, mean, variance, eps, momentum, group):
        """TaskBN 的跨卡计算边界；厂商仅在该 Torch 私有 ABI 不兼容时覆盖。

        输入: input[N,C,...]、可选 weight/bias[C]、运行 mean/variance[C]、
        eps/momentum 标量和调用方指定的进程组 group（None 表示默认组）。
        输出: 与 input 同形状/设备的 Tensor；保留输入/仿射参数梯度及运行统计更新。
        NVIDIA 参考: torch.nn.modules._functions.SyncBatchNorm.apply，最后一项
        是当前 group 的 world_size；不能改成普通单卡 BatchNorm。
        TaskBN 选择哪套运行统计仍由 UP 负责，负责人不在此改变 task 逻辑。
        """
        # Keep TaskBN's state selection; isolate the private Torch ABI here.
        from torch.nn.modules._functions import SyncBatchNorm
        import torch.distributed as dist
        return SyncBatchNorm.apply(input, weight, bias, mean, variance, eps,
                                   momentum, group, dist.get_world_size(group))

    def memory_available(self, device, needed_memory, reserve_bytes=0.5 * 1024 ** 3):
        """按原 UP 估算是否够显存，返回 bool；不是实际分配或显存预约。

        参数: device 为实际 torch.device；needed_memory/reserve_bytes 均为 byte。
        NVIDIA 参考: get_device_properties(device.index).total_memory 减去
        memory_allocated(device) 和 reserve_bytes。CPU/设备不可用返回 False。
        厂商 API 兼容就复用；差异实现保留该估算规则，不借机更改模型回退条件。
        """
        if device.type == 'cpu' or not self.module.is_available():
            return False
        total = self.module.get_device_properties(device.index).total_memory
        allocated = self.module.memory_allocated(device)
        return needed_memory < total - allocated - reserve_bytes

    def configure_dnn(self, options=None, legacy_nocudnn=None):
        """设置显式请求的计算库策略，返回 None；无效或不支持选项报错。

        options: enabled/benchmark/deterministic/allow_tf32 到 bool 的字典；
        legacy_nocudnn: 原 CLI 参数。NVIDIA 参考为 torch.backends.cudnn。
        C500 负责人核对兼容开关是否实际控制所装 MACA 的计算库。
        910B/MLU590 负责人若要支持这些选项，需映射到真实 CANN/CNNL 能力；
        当前无显式选项时用厂商默认值，有选项则报不支持，不静默忽略。
        开关的数值效果和性能需真机测试，名称相同不能当作语义相同。
        """
        options = dict(options or {})
        valid = {'enabled', 'benchmark', 'deterministic', 'allow_tf32'}
        if set(options) - valid or any(type(value) is not bool for value in options.values()):
            raise ValueError('runtime.dnn accepts boolean {}'.format(sorted(valid)))
        if legacy_nocudnn:
            if options.get('enabled') is True:
                raise ValueError('--nocudnn conflicts with runtime.dnn.enabled=true')
            options['enabled'] = False
        elif legacy_nocudnn is not None and self.device_type == 'cuda':
            # Preserve the original Runner default without overriding explicit YAML.
            options.setdefault('enabled', True)
        if not options:
            return
        if self.device_type != 'cuda':
            raise NotImplementedError('{}: runtime.dnn {} needs a verified vendor policy; '
                                      'cuDNN flags cannot control CANN/CNNL'.format(
                                          self.platform, sorted(options)))
        api = torch.backends.cudnn
        for name in options:
            if not hasattr(api, name):
                raise NotImplementedError('{} does not expose the DNN policy {}'.format(self.platform, name))
        for name, value in options.items():
            setattr(api, name, value)
        if 'allow_tf32' in options:
            # Convolution and GEMM have separate TF32 switches in CUDA-compatible
            # Torch. A recipe that disables TF32 must apply to both paths.
            torch.backends.cuda.matmul.allow_tf32 = options['allow_tf32']


def register_backend(platform, *, device_type, dist_backend, extension=None,
                     backend_class=RuntimeBackend, replace=False):
    """登记厂商设备类，返回 None；调用时不导入 SDK、不初始化设备。

    我们维护注册机制；负责人在自己的插件里调用，必须早于 setup_runtime。
    platform 是厂商名；device_type 是 Tensor 设备名；dist_backend 是该环境
    注册的通信后端名；extension 是负责注册设备的可选 Python 模块。
    backend_class 为 RuntimeBackend 子类。覆盖已有厂商登记时显式传 replace=True。
    """
    if _runtime is not None and _runtime.platform == platform:
        raise RuntimeError('Register device adapters before setup_runtime selects that platform')
    if platform in _PLATFORMS and not replace:
        raise ValueError('Platform already registered: ' + platform)
    _PLATFORMS[platform] = (device_type, dist_backend, extension)
    _backend_classes[platform] = backend_class


def setup_runtime(config, opts=None, *, purpose='training'):
    # 必须在创建工作进程/通信组之前解析厂商，训练和推理入口共用这一步。
    # Resolve the same --opts overrides as the runner, before creating workers/groups.
    if opts:
        from up.utils.general.cfg_helper import merge_opts_into_cfg
        config = merge_opts_into_cfg(opts, config)
    platform = config.get('runtime', {}).get('platform', 'nvidia')
    expected = _PLATFORMS.get(platform)
    device = config.get('runtime', {}).get('device')
    if expected and device is not None and str(device).split(':')[0] != expected[0]:
        raise ValueError('runtime.device={} conflicts with runtime.platform={}; expected {}'.format(
            device, platform, expected[0]))
    global _runtime
    # fork/spawn workers own their context; never retain a parent's device handle.
    if _runtime is not None and _runtime.pid != os.getpid():
        _runtime = None
    if _runtime is not None and (_runtime.platform, _runtime.purpose) != (platform, purpose):
        if _runtime._bound:
            raise RuntimeError('The active platform/purpose is fixed; start a new process to change it')
        _runtime = None
    if _runtime is None:
        _runtime = _backend_classes.get(platform, RuntimeBackend)(platform, purpose=purpose)
    return _runtime


def get_runtime():
    global _runtime
    if _runtime is not None and _runtime.pid != os.getpid():
        cls = _backend_classes.get(_runtime.platform, RuntimeBackend)
        _runtime = cls(_runtime.platform, purpose=_runtime.purpose)
    if _runtime is None:
        _runtime = RuntimeBackend('nvidia')
    return _runtime
