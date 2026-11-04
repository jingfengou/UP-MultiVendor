"""Original to_onnx command, using the inference platform and a CPU exporter."""
import argparse

from .subcommand import Subcommand
from up.runtime import setup_runtime
from up.utils.general.cfg_helper import merge_opts_into_cfg
from up.utils.general.registry_factory import SUBCOMMAND_REGISTRY
from up.utils.general.toonnx_helper import BaseToOnnx
from up.utils.general.yaml_loader import load_yaml


@SUBCOMMAND_REGISTRY.register('to_onnx')
class ToOnnx(Subcommand):
    def add_subparser(self, name, parser):
        p = parser.add_parser(name, help='Export ResNet50, YOLO11s, ViT-B/16 or DETR-R50 to ONNX')
        p.add_argument('--config', required=True)
        p.add_argument('--save_prefix', required=True)
        p.add_argument('--input_size', type=lambda value: tuple(map(int, value.split('x'))))
        p.add_argument('--ckpt', help='UP checkpoint; overrides saver resume/pretrain selection')
        p.add_argument('--cfg_type', default='up')
        p.add_argument('--verify', action='store_true', help='Compare ONNX Runtime outputs at original tolerance')
        p.add_argument('--opts', nargs=argparse.REMAINDER)
        p.set_defaults(run=_main)
        return p


def _main(args):
    cfg = merge_opts_into_cfg(args.opts or [], load_yaml(args.config, args.cfg_type))
    setup_runtime(cfg, purpose='inference')
    if args.ckpt:
        cfg.setdefault('saver', {}).update(auto_resume=False, resume_model=args.ckpt)
    cfg.setdefault('export', {})['verify'] = args.verify
    result = BaseToOnnx(cfg, args.save_prefix, args.input_size).process()
    print('ONNX exported: ' + result)
    return result
