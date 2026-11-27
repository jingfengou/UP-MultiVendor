"""Verify the plugin's official model/data source without importing its training engine."""
import hashlib
import importlib
import json
from pathlib import Path
import sys


def package_digest(source):
    source = Path(source).resolve()
    files = [source / 'main.py', source / 'engine.py']
    for name in ('models', 'datasets', 'util'):
        package = source / name
        if not (package / '__init__.py').is_file():
            raise FileNotFoundError('Missing official DETR package: ' + name)
        files.extend(sorted(package.rglob('*.py')))
    if any(not path.is_file() for path in files):
        raise FileNotFoundError('Expected official DETR source')
    content = {path.relative_to(source).as_posix(): hashlib.sha256(
        path.read_bytes()).hexdigest() for path in files}
    return hashlib.sha256(json.dumps(content, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def load_source(source, expected_digest):
    """One pinned checkout per process; reject conflicting top-level packages."""
    source = Path(source).resolve()
    if package_digest(source) != expected_digest:
        raise ValueError('DETR source does not match official.package_sha256')
    for name in ('models', 'datasets', 'util'):
        existing = sys.modules.get(name)
        if existing is not None:
            filename = getattr(existing, '__file__', None)
            if filename is None or Path(filename).resolve().parent != source / name:
                raise RuntimeError('DETR import conflicts with package ' + name)
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    return importlib.import_module('models')
