"""Cambricon MLU runtime entry selected by runtime.platform=cambricon_mlu."""

from up.runtime.backend import RuntimeBackend


class Backend(RuntimeBackend):
    """Keep compatible torch.mlu services and override SDK differences here.

    The parent imports torch_mlu when this backend is selected, exposes
    torch.mlu as module, and selects CNCL. See README.md for device checks.
    """

    # set_device(local_rank) binds a visible logical MLU before CNCL starts.
    # Adapt module in __init__ only for SDK signature differences. Memory
    # counters use bytes and RNG state must round-trip in the same SDK.
    #
    # Override the amp property if required by the installed torch_mlu.
    # Preserve autocast(enabled=...), GradScaler overflow and saved state.
    #
    # Implement configure_dnn(options=None, legacy_nocudnn=None) only for
    # verified CNNL policies. Keep a clear error for unsupported options.
    #
    # Override sync_batch_norm(...) if the Torch ABI is incompatible. Keep
    # the caller's CNCL group, running statistics and input/affine gradients.
