from .model import Yolo11ForUP  # noqa: F401
from .loader import Yolo11DataBuilder, Yolo11Loader, build_dataset  # noqa: F401
from .runner import Yolo11Runner  # noqa: F401
from up.utils.general.registry_factory import INFERENCER_REGISTRY


@INFERENCER_REGISTRY.register('yolo11')
def build_inference(*args, **kwargs):
    # Other models do not import the optional Ultralytics data/inference modules.
    from .inference import Yolo11Inference
    return Yolo11Inference(*args, **kwargs)
