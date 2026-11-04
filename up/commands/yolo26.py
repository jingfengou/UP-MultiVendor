"""CLI for the two version-pinned official YOLO26 tasks."""
from pathlib import Path

from .subcommand import Subcommand
from up.utils.general.registry_factory import SUBCOMMAND_REGISTRY


def add_arguments(parser):
    parser.add_argument('mode', choices=('init', 'train', 'val', 'predict', 'export'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True, help='Pinned official Ultralytics checkout')
    parser.add_argument('--platform', choices=('nvidia', 'metax_cuda'))
    parser.add_argument('--data', help='Official COCO dataset YAML or ImageNet ImageFolder root')
    parser.add_argument('--weights', type=Path, help='Common init or official task checkpoint')
    parser.add_argument('--pretrained', type=Path, help='Objects365 weights, only for detection init')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--images', help='Image, directory or video for prediction')
    parser.add_argument('--resume', action='store_true')
    parser.set_defaults(run=run)
    return parser


@SUBCOMMAND_REGISTRY.register('yolo26')
class Yolo26(Subcommand):
    def add_subparser(self, name, parser):
        return add_arguments(parser.add_parser(name, help='Official YOLO26 detection/classification'))


def run(args):
    import os
    import yaml
    from up.tasks.yolo26 import OfficialYOLO26

    if args.resume and args.mode != 'train':
        raise ValueError('--resume is only valid for train')
    required = {'init': ('output',), 'train': ('data', 'weights', 'output'),
                'val': ('data', 'weights', 'output'), 'predict': ('images', 'weights', 'output'),
                'export': ('weights',)}
    for field in required[args.mode]:
        if getattr(args, field) is None:
            raise ValueError('--' + field + ' is required for ' + args.mode)
    if args.mode != 'train' and int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('Initialization, validation, prediction and export use one process')
    if args.mode != 'init' and not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    plugin = OfficialYOLO26(config, args.source, args.platform)
    if args.mode == 'init':
        print(plugin.prepare_initialization(args.output, args.pretrained))
    elif args.mode == 'train':
        plugin.train(args.data, args.weights, args.output, args.resume)
    elif args.mode == 'val':
        print(plugin.validate(args.data, args.weights, args.output))
    elif args.mode == 'predict':
        plugin.predict(args.images, args.weights, args.output)
    else:
        print(plugin.export(args.weights))
