"""Seed policy specific to the pinned YOLO26 recipe."""
from .source_helper import load_source


def initialize_seed(config, rank):
    """Preserve official seed + 1 + rank and its deterministic-algorithm flags."""
    source = config['official']
    load_source(source['source'], source['package_sha256'])
    from ultralytics.utils.torch_utils import init_seeds
    options = config['runtime']
    init_seeds(options['random_seed'] + 1 + rank,
               deterministic=options['deterministic'])
