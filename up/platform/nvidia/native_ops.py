"""NVIDIA 参考绑定；我们维护此文件，其他厂商负责人按同名方法查原实现。

参数和梯度契约见 up/platform/native.py；此处继续调用原 UP Python 包装与
C++/CUDA，其他厂商不要加载本文件指向的 NVIDIA 二进制。
"""
from torch.autograd import Function
from up.platform.native import NativeOps as Contract

class NativeOps(Contract):
    def roi_align(self, features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode, *, aligned):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.roi_align。

        计算内核: up/extensions/csrc/roi_align/；下方 import 是实际 Python 包装入口。
        """
        from up.utils.general.global_flag import ALIGNED_FLAG
        if aligned != ALIGNED_FLAG.aligned:
            raise ValueError('Alignment must match the selected UP configuration')
        from up.extensions.python.roi_align import RoIAlignFunction
        return Function.apply.__func__(RoIAlignFunction, features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode)

    def psroi_align(self, features, rois, group_size, spatial_scale, output_dim, sampling_ratio):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.psroi_align。

        计算内核: up/extensions/csrc/psroi_align/；下方 import 是实际 Python 包装入口。
        """
        from up.extensions.python.psroi_align import PSRoIAlignFunction
        return Function.apply.__func__(PSRoIAlignFunction, features, rois, group_size, spatial_scale, output_dim, sampling_ratio)

    def psroi_pool(self, features, rois, group_size, spatial_scale, output_dim):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.psroi_pool。

        计算内核: up/extensions/csrc/psroi_pooling/；下方 import 是实际 Python 包装入口。
        """
        from up.extensions.python.psroi_pool import PSRoIPoolFunction
        return Function.apply.__func__(PSRoIPoolFunction, features, rois, group_size, spatial_scale, output_dim)

    def deform_conv(self, input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.deform_conv。

        计算内核: up/extensions/csrc/deform_conv/；下方 import 是实际 Python 包装入口。
        """
        from up.extensions.python.deformable_conv import DeformableConvFunction
        return Function.apply.__func__(DeformableConvFunction, input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups)

    def naive_nms(self, dets, thresh, *, offset):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.naive_nms。

        计算内核: up/extensions/csrc/nms/；下方 import 是实际 Python 包装入口。
        """
        from up.utils.general.global_flag import ALIGNED_FLAG
        if offset != ALIGNED_FLAG.offset:
            raise ValueError('Alignment must match the selected UP configuration')
        from up.extensions.python.nms import _nvidia_naive_nms
        return _nvidia_naive_nms(dets, thresh)

    def sigmoid_focal(self, preds, targets, weight_pos, gamma, alpha, num_classes, reduction):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.sigmoid_focal。

        计算内核: up/extensions/csrc/focal_loss/；下方 import 是实际 Python 包装入口。
        """
        from up.extensions.python.focal_loss import SigmoidFocalLossFunction
        return Function.apply.__func__(SigmoidFocalLossFunction, preds, targets, weight_pos, gamma, alpha, num_classes, reduction)

    def softmax_focal(self, preds, targets, weight_pos, gamma, alpha, num_classes):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.softmax_focal。

        计算内核: up/extensions/csrc/focal_loss/；下方 import 是实际 Python 包装入口。
        """
        from up.extensions.python.focal_loss import SoftmaxFocalLossFunction
        return Function.apply.__func__(SoftmaxFocalLossFunction, preds, targets, weight_pos, gamma, alpha, num_classes)

    def cross_focal(self, preds, targets, weight_pos, gamma, alpha, num_classes, gt_class_to_avoid, reduction):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.cross_focal。

        计算内核: up/extensions/csrc/cross_focal_loss/；下方 import 是实际 Python 包装入口。
        """
        from up.extensions.python.cross_focal_loss import CrossSigmoidFocalLossFunction
        return Function.apply.__func__(CrossSigmoidFocalLossFunction, preds, targets, weight_pos, gamma, alpha, num_classes, gt_class_to_avoid, reduction)

    def overlap(self, b1, b2, mode='IoU'):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.overlap。

        计算内核: up/extensions/csrc/iou_overlap/；下方 import 是实际 Python 包装入口。
        """
        from up.extensions.python.iou_overlap import _nvidia_gpu_iou_overlap
        return _nvidia_gpu_iou_overlap(b1, b2, mode)

    def bev_iou(self, boxes_a, boxes_b):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.bev_iou。

        计算内核: up/extensions/csrc/iou3d_nms/；下方 import 是实际 Python 包装入口。
        """
        from up.extensions.python.iou3d_nms_utils import _nvidia_boxes_iou_bev
        return _nvidia_boxes_iou_bev(boxes_a, boxes_b)

    def iou3d(self, boxes_a, boxes_b):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.iou3d。

        计算内核: up/extensions/csrc/iou3d_nms/；下方 import 是实际 Python 包装入口。
        """
        from up.extensions.python.iou3d_nms_utils import _nvidia_boxes_iou3d_gpu
        return _nvidia_boxes_iou3d_gpu(boxes_a, boxes_b)

    def nms3d(self, boxes, scores, thresh, pre_maxsize=None, **kwargs):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.nms3d。

        计算内核: up/extensions/csrc/iou3d_nms/；下方 import 是实际 Python 包装入口。
        """
        from up.extensions.python.iou3d_nms_utils import _nvidia_nms_gpu
        return _nvidia_nms_gpu(boxes, scores, thresh, pre_maxsize, **kwargs)

    def nms3d_normal(self, boxes, scores, thresh, **kwargs):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.nms3d_normal。

        计算内核: up/extensions/csrc/iou3d_nms/；下方 import 是实际 Python 包装入口。
        """
        from up.extensions.python.iou3d_nms_utils import _nvidia_nms_normal_gpu
        return _nvidia_nms_normal_gpu(boxes, scores, thresh, **kwargs)

    def points_in_boxes(self, points, boxes):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.points_in_boxes。

        计算内核: up/extensions/csrc/roiaware_pool3d/；下方 import 是实际 Python 包装入口。
        """
        from up.extensions.python.roiaware_pool3d_utils import _nvidia_points_in_boxes_gpu
        return _nvidia_points_in_boxes_gpu(points, boxes)

    def roiaware(self, rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.roiaware。

        计算内核: up/extensions/csrc/roiaware_pool3d/；下方 import 是实际 Python 包装入口。
        """
        from up.extensions.python.roiaware_pool3d_utils import RoIAwarePool3dFunction
        return Function.apply.__func__(RoIAwarePool3dFunction, rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method)

    def roipoint(self, points, point_features, boxes3d, pool_extra_width, num_sampled_points=512):
        """NVIDIA 原实现；完整参数/输出/梯度见 native.NativeOps.roipoint。

        计算内核: up/extensions/csrc/roipoint_pool3d/；下方 import 是实际 Python 包装入口。
        """
        from up.extensions.python.roipoint_pool3d_utils import RoIPointPool3dFunction
        return Function.apply.__func__(RoIPointPool3dFunction, points, point_features, boxes3d, pool_extra_width, num_sampled_points)
