"""Select a verified official source tree before importing Ultralytics."""
import hashlib
import importlib
import json
from pathlib import Path
import sys


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_digest(source):
    """Hash package code/configs, independently of checkout paths or Git metadata."""
    source = Path(source).resolve()
    package = source / 'ultralytics'
    files = {p.relative_to(source).as_posix(): sha256(p)
             for p in sorted(package.rglob('*'))
             if p.is_file() and p.suffix in ('.py', '.yaml', '.toml')}
    if not files:
        raise FileNotFoundError('Expected an official checkout containing ultralytics/')
    return hashlib.sha256(json.dumps(
        files, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()


def load_source(source, expected_digest):
    source = Path(source).resolve()
    if source_digest(source) != expected_digest:
        raise ValueError('Official YOLO26 code/configs do not match this YAML source pin')
    existing = sys.modules.get('ultralytics')
    if existing is not None and Path(existing.__file__).resolve().parent != source / 'ultralytics':
        raise RuntimeError('Run each YOLO version in a fresh process; another Ultralytics is loaded')
    sys.path.insert(0, str(source))
    module = importlib.import_module('ultralytics')
    if Path(module.__file__).resolve().parent != source / 'ultralytics':
        raise RuntimeError('Ultralytics was imported from a different source tree')
    return module
