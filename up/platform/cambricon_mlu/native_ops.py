"""寒武纪 MLU590负责人：本厂商专用操作在本文件接入。

厂商环境、运行时与操作契约见 docs/platform/ADDING_VENDOR.md。

【我们统一改】维护父类契约、共享 Torch 实现和模型调用；先核对厂商已有
API，明确各项应绑定现有 SDK、补共享实现，还是开发专用内核。
【各显卡负责人改】按确定的方案，在下面 NativeOps 类实现对应方法，
或绑定符合契约的真实 SDK。输入/输出/反向要求见父类同名函数的 docstring。

设备: torch_mlu 注册的 mlu 设备；不要将输入搬到 CUDA 或 CPU 代算。
Neuware 环境先核对 torch.mlu/amp、CNCL 和显式 DNN 策略。
设备/AMP 不兼容时另写 vendor_plugins/cambricon_mlu_runtime.py 的 RuntimeBackend
子类并注册；不要在本算子文件改 Runner、模型或厂商选择逻辑。

当前 3 项已复用、13 项继承未实现报错；未接入不表示厂商 SDK 没有这个能力。
五份默认配置中 YOLOX 的 GPU OTA 匹配需要 overlap；其他 15 项默认未用。
常规检测 NMS 另走 up/platform/ops.py::nms，不是这里的 naive_nms。
验收: 同操作数前向/所需梯度/状态/空输入，rtol=1e-4、atol=1e-5；
索引按契约精确比较。记录本卡环境和结果，不用导入成功代替真机测试。
"""
from up.platform.native import NativeOps as NativeOpsContract
from up.platform import torch_native


class NativeOps(NativeOpsContract):
    """本文件就是寒武纪 MLU590的修改位置；父类是参数、输出及梯度契约。

    已复用项保留下面的显式绑定，先验证。待接入项的注释签名是填写位置
    索引；实际开发时在本类定义同名方法。没有真实实现时继续继承报错，
    不加 pass、返回零或只调用 super() 的空覆盖方法。
    SDK Python 绑定放同目录 kernels/，需要编译的源码放 csrc/；不要覆盖原 _C。
    """

    # 已复用：dets[N,5] -> 本设备 int64 索引；保留输入顺序及 offset，无反向。
    # NVIDIA 参考: up/extensions/python/nms.py::_nvidia_naive_nms。
    naive_nms = staticmethod(torch_native.naive_nms)

    # 已复用、YOLOX 必需：b1/b2 -> IoU/IoF/IoS 矩阵；+1 坐标，不求几何梯度。
    # NVIDIA 参考: up/extensions/python/iou_overlap.py::_nvidia_gpu_iou_overlap。
    overlap = staticmethod(torch_native.overlap)

    # 已复用：保留背景0/忽略-1、normalizer、reduction；preds 梯度由 autograd 提供。
    # NVIDIA 参考: up/extensions/python/focal_loss.py::SigmoidFocalLossFunction。
    sigmoid_focal = staticmethod(torch_native.sigmoid_focal)

    # 以下 13 项尚未接入、五份默认配置未用。完整契约在父类同名函数。

    # def roi_align(self, features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode, *, aligned):
    #     区域特征采样；需 features 梯度；核对 aligned 和 sampling_ratio。
    #     NVIDIA 参考: up/extensions/python/roi_align.py::RoIAlignFunction

    # def psroi_align(self, features, rois, group_size, spatial_scale, output_dim, sampling_ratio):
    #     位置敏感采样；需 features 梯度及通道映射，不能用普通 RoIAlign 替代。
    #     NVIDIA 参考: up/extensions/python/psroi_align.py::PSRoIAlignFunction

    # def psroi_pool(self, features, rois, group_size, spatial_scale, output_dim):
    #     位置敏感池化；需 features 梯度，保留 bin 取整和通道映射。
    #     NVIDIA 参考: up/extensions/python/psroi_pool.py::PSRoIPoolFunction

    # def deform_conv(self, input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups):
    #     DCNv1；需 input/offset/weight 梯度，核对 groups/deformable_groups。
    #     NVIDIA 参考: up/extensions/python/deformable_conv.py::DeformableConvFunction

    # def softmax_focal(self, preds, targets, weight_pos, gamma, alpha, num_classes):
    #     多类 Focal；需 preds 梯度，保留背景权重和 normalizer。
    #     NVIDIA 参考: up/extensions/python/focal_loss.py::SoftmaxFocalLossFunction

    # def cross_focal(self, preds, targets, weight_pos, gamma, alpha, num_classes, gt_class_to_avoid, reduction):
    #     带屏蔽表的 Focal；需 preds 梯度，按原 neg_map 解码。
    #     NVIDIA 参考: up/extensions/python/cross_focal_loss.py::CrossSigmoidFocalLossFunction

    # def bev_iou(self, boxes_a, boxes_b):
    #     旋转 BEV IoU；原扩展无几何反向，核对角度和边界。
    #     NVIDIA 参考: up/extensions/python/iou3d_nms_utils.py::_nvidia_boxes_iou_bev

    # def iou3d(self, boxes_a, boxes_b):
    #     三维 IoU；保留原 BEV 无梯度、外围高度/体积可求导的图结构。
    #     NVIDIA 参考: up/extensions/python/iou3d_nms_utils.py::_nvidia_boxes_iou3d_gpu

    # def nms3d(self, boxes, scores, thresh, pre_maxsize=None, **kwargs):
    #     旋转框 NMS；不求梯度，返回 (原输入索引, None)。
    #     NVIDIA 参考: up/extensions/python/iou3d_nms_utils.py::_nvidia_nms_gpu

    # def nms3d_normal(self, boxes, scores, thresh, **kwargs):
    #     轴对齐 3D NMS；不求梯度，不能借用旋转 IoU 规则。
    #     NVIDIA 参考: up/extensions/python/iou3d_nms_utils.py::_nvidia_nms_normal_gpu

    # def points_in_boxes(self, points, boxes):
    #     点框归属；不求梯度，输出 int32，未命中为 -1。
    #     NVIDIA 参考: up/extensions/python/roiaware_pool3d_utils.py::_nvidia_points_in_boxes_gpu

    # def roiaware(self, rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
    #     体素池化；需 pts_feature 梯度，保留 max/avg 和截断规则。
    #     NVIDIA 参考: up/extensions/python/roiaware_pool3d_utils.py::RoIAwarePool3dFunction

    # def roipoint(self, points, point_features, boxes3d, pool_extra_width, num_sampled_points=512):
    #     区域点采样；原版无 backward，保留补采样和空框标志。
    #     NVIDIA 参考: up/extensions/python/roipoint_pool3d_utils.py::RoIPointPool3dFunction
