"""Temporary eval restores each module's mode, including intentionally frozen BN."""
from contextlib import contextmanager


@contextmanager
def evaluation_mode(*models):
    modules = {module: module.training for model in models if model is not None
               for module in model.modules()}
    try:
        for model in models:
            if model is not None:
                model.eval()
        yield
    finally:
        # Calling train(True) recursively would overwrite frozen child modes.
        for module, training in modules.items():
            module.training = training
