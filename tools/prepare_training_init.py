"""Create one CPU random initialization for all vendor runs of a model.

No pretrained weights or GPU context is used. Copy the output and its SHA256
receipt unchanged to the three machines; do not generate one file per vendor.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    receipt = args.output.with_suffix(args.output.suffix + '.json')
    if args.output.exists() or receipt.exists():
        raise FileExistsError('Use a new output path; existing initialization is preserved')

    import torch
    from up.models.model_helper import ModelHelper
    from up.utils.env.gene_env import set_random_seed
    from up.utils.general.yaml_loader import load_yaml
    from up.utils.general.global_flag import ALIGNED_FLAG

    cfg = load_yaml(str(args.config))
    runtime = cfg.get('runtime', {})
    seed = runtime.get('random_seed', 131)
    rank_init = runtime.get('rank_init', False)
    # Use the Runner's seed mapping before creating the shared CPU weights.
    set_random_seed(seed, rank_init)
    ALIGNED_FLAG.aligned = runtime.get('aligned', False)
    ALIGNED_FLAG.offset = 0 if ALIGNED_FLAG.aligned else 1
    model = ModelHelper(copy.deepcopy(cfg['net']))
    if any(value.device.type != 'cpu' for value in model.state_dict().values() if torch.is_tensor(value)):
        raise RuntimeError('Initialization must remain on CPU')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        torch.save({'model': model.state_dict()}, stream)
    result = {'seed': seed, 'rank_init': rank_init,
              'seed_function': 'up.utils.env.gene_env.set_random_seed',
              'torch_initial_seed': torch.initial_seed(),
              'torch': str(torch.__version__), 'pretrained': False,
              'model_config': cfg['net'], 'bytes': args.output.stat().st_size,
              'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest()}
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    print(json.dumps({'output': str(args.output), 'sha256': result['sha256'], 'seed': seed}))


if __name__ == '__main__':
    main()
