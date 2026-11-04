"""Optional official YOLO26 training and inference plugin.

Importing UP does not import Ultralytics. Each invocation selects one pinned
official source tree, so YOLO11 and the two YOLO26 versions stay independent.
"""
from .plugin import OfficialYOLO26  # noqa: F401
