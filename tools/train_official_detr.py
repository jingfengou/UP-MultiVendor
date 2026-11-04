"""Run pinned official DETR with the selected NVIDIA or MetaX Torch environment.

The official checkout owns the model, augmentation, loss, optimizer loop and
COCO evaluator. UP selects the device runtime; it does not substitute its older
DETR adapter or relabel COCO categories for this benchmark.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

REVISION = '29901c51d7fe8712168b8d0d64351170bc0f83e0'
REPOSITORY = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_official(source):
    source = source.resolve()
    revision = subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('DETR source must be at ' + REVISION)
    subprocess.run(['git', '-C', str(source), 'diff', '--exit-code', 'HEAD'], check=True)
    # These generic package names must resolve to the pinned external checkout.
    for name in ('models', 'datasets', 'util', 'engine'):
        if name in sys.modules:
            raise RuntimeError('Load official DETR in a fresh process: ' + name)
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('_official_detr_main', source / 'main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_initialization(official, args, output, config):
    import numpy as np
    import torch

    receipt = output.with_suffix(output.suffix + '.json')
    if output.exists() or receipt.exists():
        raise FileExistsError('Preserve the existing common initialization')
    args.device = 'cpu'
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Official build_model loads the ImageNet ResNet50 backbone and randomly
    # initializes the detector. Never initialize training with COCO model-zoo weights.
    model, _, _ = official.build_model(args)
    weights = model.state_dict()
    if any(t.device.type != 'cpu' for t in weights.values()):
        raise RuntimeError('Common initialization must be created on CPU')
    backbone = Path(torch.hub.get_dir()) / 'checkpoints/resnet50-0676ba61.pth'
    backbone_sha = sha256(backbone)
    if not backbone_sha.startswith('0676ba61'):
        raise ValueError('Unexpected torchvision ImageNet-1K V1 backbone')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as stream:
        torch.save({'model': weights}, stream)
    metadata = {
        'official_revision': REVISION, 'seed': args.seed, 'torch': str(torch.__version__),
        'config_sha256': sha256(config), 'sha256': sha256(output),
        'backbone': 'torchvision ResNet50 IMAGENET1K_V1',
        'backbone_sha256': backbone_sha, 'coco_pretrained': False,
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    with receipt.open('x') as stream:
        json.dump(metadata, stream, indent=2)
    print(json.dumps(metadata))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=REPOSITORY /
                        'configs/benchmarks/three_vendor/detr_r50_official.yaml')
    parser.add_argument('--platform', choices=['nvidia', 'metax_cuda'], required=True)
    parser.add_argument('--coco-path', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--prepare-init', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--eval', action='store_true')
    options = parser.parse_args()

    import torch
    import yaml

    config = yaml.safe_load(options.config.read_text(encoding='utf-8'))
    official = load_official(options.source)
    args = official.get_args_parser().parse_args([])
    runtime_keys = {'expected_world_size', 'evaluation_batch_size'}
    unknown = set(config) - set(vars(args)) - runtime_keys
    if unknown:
        raise ValueError('Unknown official DETR arguments: ' + str(sorted(unknown)))
    for key, value in config.items():
        if key not in runtime_keys:
            setattr(args, key, value)
    if options.prepare_init:
        prepare_initialization(official, args, options.prepare_init, options.config)
        return
    if not options.coco_path or not options.output_dir or not options.resume:
        parser.error('Training/evaluation require --coco-path, --output-dir and --resume')
    if not options.resume.is_file():
        raise FileNotFoundError(options.resume)
    if int(os.environ.get('WORLD_SIZE', '1')) != config['expected_world_size']:
        raise ValueError('Launch the configured number of processes with torchrun')

    sys.path.insert(0, str(REPOSITORY))
    from up.runtime import setup_runtime

    runtime = setup_runtime({'runtime': {'platform': options.platform}}, purpose='training')
    runtime.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    # The official FP32 baseline predates TF32. Use vendor-native FP32 libraries,
    # with no AMP, custom GEMM or deterministic replacement kernel.
    runtime.configure_dnn({'benchmark': False, 'deterministic': False, 'allow_tf32': False})
    args.device = str(runtime.device)
    args.coco_path = str(options.coco_path.resolve())
    args.output_dir = str(options.output_dir.resolve())
    args.resume = str(options.resume.resolve())
    args.eval = options.eval
    if args.eval:
        args.batch_size = config['evaluation_batch_size']
    options.output_dir.mkdir(parents=True, exist_ok=True)
    # PyTorch 2.6+ uses weights_only=True. Official local checkpoints also store
    # the argparse namespace; allow that specific type instead of unrestricted pickle.
    with torch.serialization.safe_globals([argparse.Namespace]):
        official.main(args)


if __name__ == '__main__':
    main()
