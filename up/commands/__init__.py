# 瘦身版只公开训练/验证和独立推理，避免导入已移除任务的命令依赖。
from .train import Train  # noqa
from .inference import Inference # noqa
from .to_onnx import ToOnnx  # noqa
from .yolo26 import Yolo26  # noqa
