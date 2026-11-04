# 从原 3D 任务抽出的纯几何工具，保留扩展入口时无需导入整套 3D 任务注册代码。
import numpy as np
import torch


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)
    large_boxes3d = boxes3d.clone()
    large_boxes3d[:, 3:6] += boxes3d.new_tensor(extra_width)[None, :]
    return large_boxes3d
