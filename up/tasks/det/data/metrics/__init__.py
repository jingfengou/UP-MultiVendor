# 保留四模型检测配置使用的评估器，不再通过注册入口加载已移除任务。
from .coco_evaluator import CocoEvaluator  # noqa
from .custom_evaluator import CustomEvaluator  # noqa
