from __future__ import division

# Standard Library
import argparse
import sys

# Import from third library
import torch.multiprocessing as mp

from up.utils.env.dist_helper import setup_distributed, finalize, gpu_check
from up.utils.general.yaml_loader import load_yaml  # IncludeLoader
from up.utils.env.launch import launch
from up.runtime import setup_runtime
from up.utils.general.user_analysis_helper import send_info

# Import from local
from .subcommand import Subcommand
from up.utils.general.registry_factory import SUBCOMMAND_REGISTRY, RUNNER_REGISTRY
from up.utils.general.global_flag import DIST_BACKEND


__all__ = ['Train']


@SUBCOMMAND_REGISTRY.register('train')
class Train(Subcommand):
    def add_subparser(self, name, parser):
        sub_parser = parser.add_parser(name,
                                       description='subcommand for training',
                                       help='train a model')

        sub_parser.add_argument('-e',
                                '--evaluate',
                                dest='evaluate',
                                action='store_true',
                                help='evaluate model on validation set')
        sub_parser.add_argument(
            '--fork-method',
            dest='fork_method',
            type=str,
            default='spawn',
            choices=['spawn', 'fork'],
            help='method to fork subprocess, especially for dataloader')
        sub_parser.add_argument('--backend',
                                dest='backend',
                                type=str,
                                default='dist',
                                choices=['dist'],
                                help='model backend')
        sub_parser.add_argument(
            '--nocudnn',
            dest='nocudnn',
            action='store_true',
            help='Whether to use cudnn backend or not. Please disable cudnn when running on V100'
        )
        sub_parser.add_argument('--config',
                                dest='config',
                                required=True,
                                help='settings of detection in yaml format')
        sub_parser.add_argument('--display',
                                dest='display',
                                type=int,
                                default=20,
                                help='display intervel')
        sub_parser.add_argument('--ng', '--num_gpus_per_machine',
                                dest='num_gpus_per_machine',
                                type=int,
                                default=1,
                                help='num_gpus_per_machine')
        sub_parser.add_argument('--nm', '--num_machines',
                                dest='num_machines',
                                type=int,
                                default=1,
                                help='num_machines')
        sub_parser.add_argument('--launch',
                                dest='launch',
                                type=str,
                                default='pytorch',
                                help='launch backend')
        sub_parser.add_argument('--port',
                                dest='port',
                                type=int,
                                default=13333,
                                help='dist port')
        sub_parser.add_argument('--no_running_config',
                                action='store_true',
                                help='disable display running config')
        sub_parser.add_argument('--phase', default='train', help="train phase")
        sub_parser.add_argument('--cfg_type',
                                dest='cfg_type',
                                type=str,
                                default='up',
                                help='config type (up or pod)')
        sub_parser.add_argument('--opts',
                                help='options to replace yaml config',
                                default=None,
                                nargs=argparse.REMAINDER)
        sub_parser.add_argument('--test_gpu',
                                dest='test_gpu',
                                action='store_true',
                                help='test if gpus work properly before training')

        sub_parser.set_defaults(run=_main)
        return sub_parser


def main(args):
    if args.test_gpu:
        gpu_check()
    cfg = load_yaml(args.config, args.cfg_type)
    # dist 选择 PyTorch 执行路径，只有实际 world_size > 1 时 Runner 才包装 DDP。
    cfg['args'] = {
        'ddp': args.backend == 'dist',
        'config_path': args.config,
        'nocudnn': args.nocudnn,
        'display': args.display,
        'no_running_config': args.no_running_config,
        'opts': args.opts
    }
    train_phase = args.phase
    cfg['runtime'] = cfg.setdefault('runtime', {})
    runner_cfg = cfg['runtime'].get('runner', {})
    runner_cfg['type'] = runner_cfg.get('type', 'base')
    runner_cfg['kwargs'] = runner_cfg.get('kwargs', {})
    cfg['runtime']['runner'] = runner_cfg
    training = True
    if args.evaluate:
        training = False
        train_phase = "eval"
    send_info(cfg, train_phase)
    runner_cfg['kwargs']['training'] = training
    runner = RUNNER_REGISTRY.get(runner_cfg['type'])(cfg, **runner_cfg['kwargs'])
    train_func = {"train": runner.train, "eval": runner.evaluate}
    # 训练内验证和 --evaluate 共用同一 Runner 和设备选择。

    train_func[train_phase]()
    finalize()


def _main(args):
    DIST_BACKEND.backend = args.backend
    # 先合并 --opts 选定厂商，再把相同选择交给启动器和各个工作进程。
    runtime = setup_runtime(load_yaml(args.config, args.cfg_type), args.opts, purpose='training')
    args.purpose = 'training'
    args.platform = runtime.platform
    if runtime.platform != 'nvidia':
        # 非 NVIDIA SDK 使用新解释器进程初始化，避免继承父进程的设备状态。
        args.fork_method = 'spawn'
    if args.launch == 'pytorch':
        launch(main, args.num_gpus_per_machine, args.num_machines, args=args, start_method=args.fork_method)
    else:
        mp.set_start_method(args.fork_method, force=True)
        fork_method = mp.get_start_method(allow_none=True)
        assert fork_method == args.fork_method
        sys.stdout.flush()
        setup_distributed(args.port, args.launch, args.backend)
        main(args)
