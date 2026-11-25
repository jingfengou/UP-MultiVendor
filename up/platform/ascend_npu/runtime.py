"""Ascend NPU runtime entry selected by runtime.platform=ascend_npu."""

from up.runtime.backend import RuntimeBackend


class Backend(RuntimeBackend):
    """Keep compatible torch.npu services and override SDK differences here.

    The parent imports torch_npu when this backend is selected, exposes
    torch.npu as module, and selects HCCL. See README.md for device checks.
    """

    # set_device(local_rank) binds a visible logical NPU before HCCL starts.
    # Adapt module in __init__ only for SDK signature differences. Memory
    # counters use bytes and RNG state must round-trip in the same SDK.
    #
    # Override the amp property if required by the installed torch_npu.
    # Preserve autocast(enabled=...), GradScaler overflow and saved state.
    #
    # Implement configure_dnn(options=None, legacy_nocudnn=None) only for
    # verified CANN policies. Keep a clear error for unsupported options.
    #
    # Override sync_batch_norm(...) if the Torch ABI is incompatible. Keep
    # the caller's HCCL group, running statistics and input/affine gradients.
