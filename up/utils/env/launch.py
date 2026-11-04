#!/usr/bin/env python3
# Code are based on
# https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/launch.py
# Copyright (c) Facebook, Inc. and its affiliates.

from datetime import timedelta
from ..general.log_helper import default_logger as logger

import torch.distributed as dist
import torch.multiprocessing as mp
from up.utils.env import dist_helper as dist_helper
from up.runtime import setup_runtime

__all__ = ["launch"]

DEFAULT_TIMEOUT = timedelta(minutes=30)


def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    backend=None,
    dist_url='auto',
    args=(),
    timeout=DEFAULT_TIMEOUT,
    start_method='fork'
):
    """
    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine (one per machine)
        dist_url (str): url to connect to for distributed training, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to auto to automatically select a free port on localhost
        args (tuple): arguments passed to main_func
    """
    runtime = setup_runtime({'runtime': {'platform': getattr(args, 'platform', 'nvidia')}},
                            purpose=getattr(args, 'purpose', 'training'))
    # 用户未覆盖通信后端时，使用所选设备对应的 NCCL/HCCL/CNCL。
    backend = backend or runtime.dist_backend
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if dist_url == "auto":
            assert (
                num_machines == 1
            ), "dist_url=auto cannot work with distributed training."
            port = dist_helper.find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"

        mp.start_processes(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                backend,
                dist_url,
                args,
            ),
            daemon=False,
            start_method=start_method,
        )
    else:
        # 单卡没有 DDP 包装，但仍需绑定设备，以便模型和临时张量使用同一张卡。
        runtime.set_device(0)
        main_func(*(args,))


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    backend,
    dist_url,
    args,
    timeout=DEFAULT_TIMEOUT,
):
    runtime = setup_runtime({'runtime': {'platform': getattr(args, 'platform', 'nvidia')}},
                            purpose=getattr(args, 'purpose', 'training'))
    # 每个子进程先绑定 local_rank，再初始化通信组，避免各 rank 默认占用第 0 张卡。
    assert num_gpus_per_machine <= runtime.module.device_count()
    runtime.set_device(local_rank)
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    logger.info("Rank {} initialization finished.".format(global_rank))
    try:
        dist.init_process_group(
            backend=backend,
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception:
        logger.error("Process group URL: {}".format(dist_url))
        raise

    # Setup the local process group (which contains ranks within the same machine)
    assert dist_helper._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            dist_helper._LOCAL_PROCESS_GROUP = pg

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172

    dist_helper.barrier()

    main_func(args)
