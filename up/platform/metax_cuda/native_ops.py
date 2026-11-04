"""C500 bindings to UP kernels compiled by MACA cuBridge.

Build with UP_BUILD_PLATFORM=metax_cuda in a matching MACA PyTorch environment.
The source in up/extensions/csrc is shared; the binary is this package's _C,
never NVIDIA's up.extensions._C. Original Python wrappers retain autograd,
saved tensors, coordinate rules and ONNX symbols. Missing builds fail explicitly.

Method contracts and the NVIDIA reference locations are in native.NativeOps.
ResNet50 uses vendor Torch convolution/BN, not these specialized UP operators.
"""
from torch.autograd import Function
from up.platform.native import NativeOps as Contract


def require_native_extension():
    """Check the full MACA binary before training/inference acquires a device.

    CPU imports and capability queries stay lazy. A missing or older binary
    must stop execution instead of selecting a Torch replacement.
    """
    from up.extensions.ext import load_extension
    extension = load_extension('metax_cuda')
    required = {
        'roi_align': ('forward_avg_cuda', 'backward_avg_cuda'),
        'psroi_align': ('forward_cuda', 'backward_cuda'),
        'psroi_pooling': ('forward_cuda', 'backward_cuda'),
        'deform_conv_v1': ('forward_cuda', 'backward_input_cuda', 'backward_parameters_cuda'),
        'naive_nms': ('gpu_nms', 'continuous_mask'),
        'focal_loss': ('sigmoid_forward_cuda', 'sigmoid_backward_cuda',
                       'softmax_forward_cuda', 'softmax_backward_cuda'),
        'cross_focal_loss': ('cross_sigmoid_forward_cuda', 'cross_sigmoid_backward_cuda'),
        'overlaps': ('iou',),
        'iou3d_nms': ('boxes_iou_bev_gpu', 'boxes_overlap_bev_gpu', 'nms_gpu', 'nms_normal_gpu'),
        'roiaware_pool3d': ('forward', 'backward', 'points_in_boxes_gpu'),
        'roipoint_pool3d': ('forward',),
    }
    missing = [group + '.' + name for group, names in required.items() for name in names
               if not callable(getattr(getattr(extension, group, None), name, None))]
    if missing:
        raise RuntimeError('Incomplete MetaX native extension: {}. Rebuild with '
                           'UP_BUILD_PLATFORM=metax_cuda; Torch fallback is disabled.'.format(
                               ', '.join(missing)))
    return extension


class NativeOps(Contract):
    """Explicit MetaX bindings; legacy wrapper names do not select the binary."""

    def roi_align(self, features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode, *, aligned):
        """ROI sampling and feature backward; preserve UP's alignment setting."""
        from up.utils.general.global_flag import ALIGNED_FLAG
        if aligned != ALIGNED_FLAG.aligned:
            raise ValueError('Alignment must match the selected UP configuration')
        from up.extensions.python.roi_align import RoIAlignFunction
        # Bypass the public .apply dispatcher to avoid routing into ourselves again.
        return Function.apply.__func__(RoIAlignFunction, features, rois, pooled_h, pooled_w, spatial_scale, sampling_ratio, pooled_mode)

    def psroi_align(self, features, rois, group_size, spatial_scale, output_dim, sampling_ratio):
        """Position-sensitive sampling, including channel mapping and backward."""
        from up.extensions.python.psroi_align import PSRoIAlignFunction
        return Function.apply.__func__(PSRoIAlignFunction, features, rois, group_size, spatial_scale, output_dim, sampling_ratio)

    def psroi_pool(self, features, rois, group_size, spatial_scale, output_dim):
        """Position-sensitive pooling with the original bin rounding/backward."""
        from up.extensions.python.psroi_pool import PSRoIPoolFunction
        return Function.apply.__func__(PSRoIPoolFunction, features, rois, group_size, spatial_scale, output_dim)

    def deform_conv(self, input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups):
        """DCNv1; compute input, offset and weight gradients with the UP kernels."""
        from up.extensions.python.deformable_conv import DeformableConvFunction
        return Function.apply.__func__(DeformableConvFunction, input, offset, weight, out_channels, kernel_size, stride, pad, dilations, groups, deformable_groups)

    def naive_nms(self, dets, thresh, *, offset):
        """Sorted, offset-aware NMS; preserve the original mask/selection path."""
        from up.utils.general.global_flag import ALIGNED_FLAG
        if offset != ALIGNED_FLAG.offset:
            raise ValueError('Alignment must match the selected UP configuration')
        from up.extensions.python.nms import _nvidia_naive_nms
        return _nvidia_naive_nms(dets, thresh)

    def sigmoid_focal(self, preds, targets, weight_pos, gamma, alpha, num_classes, reduction):
        """Sigmoid Focal with original label, normalizer and logits-gradient rules."""
        from up.extensions.python.focal_loss import SigmoidFocalLossFunction
        return Function.apply.__func__(SigmoidFocalLossFunction, preds, targets, weight_pos, gamma, alpha, num_classes, reduction)

    def softmax_focal(self, preds, targets, weight_pos, gamma, alpha, num_classes):
        """Softmax Focal; preserve background weighting and logits backward."""
        from up.extensions.python.focal_loss import SoftmaxFocalLossFunction
        return Function.apply.__func__(SoftmaxFocalLossFunction, preds, targets, weight_pos, gamma, alpha, num_classes)

    def cross_focal(self, preds, targets, weight_pos, gamma, alpha, num_classes, gt_class_to_avoid, reduction):
        """Cross Focal; keep packed neg_map and the original scalar output."""
        from up.extensions.python.cross_focal_loss import CrossSigmoidFocalLossFunction
        return Function.apply.__func__(CrossSigmoidFocalLossFunction, preds, targets, weight_pos, gamma, alpha, num_classes, gt_class_to_avoid, reduction)

    def overlap(self, b1, b2, mode='IoU'):
        """YOLOX OTA box overlap; retain +1 coordinates and empty-output shape."""
        from up.extensions.python.iou_overlap import _nvidia_gpu_iou_overlap
        return _nvidia_gpu_iou_overlap(b1, b2, mode)

    def bev_iou(self, boxes_a, boxes_b):
        """Rotated BEV IoU; the original geometry kernel has no backward."""
        from up.extensions.python.iou3d_nms_utils import _nvidia_boxes_iou_bev
        return _nvidia_boxes_iou_bev(boxes_a, boxes_b)

    def iou3d(self, boxes_a, boxes_b):
        """3D IoU; preserve the differentiable outer height/volume expressions."""
        from up.extensions.python.iou3d_nms_utils import _nvidia_boxes_iou3d_gpu
        return _nvidia_boxes_iou3d_gpu(boxes_a, boxes_b)

    def nms3d(self, boxes, scores, thresh, pre_maxsize=None, **kwargs):
        """Rotated 3D NMS; return original indices and None, with no backward."""
        from up.extensions.python.iou3d_nms_utils import _nvidia_nms_gpu
        return _nvidia_nms_gpu(boxes, scores, thresh, pre_maxsize, **kwargs)

    def nms3d_normal(self, boxes, scores, thresh, **kwargs):
        """Axis-aligned 3D NMS; do not substitute rotated overlap rules."""
        from up.extensions.python.iou3d_nms_utils import _nvidia_nms_normal_gpu
        return _nvidia_nms_normal_gpu(boxes, scores, thresh, **kwargs)

    def points_in_boxes(self, points, boxes):
        """Point-to-box assignment; return int32 indices, with -1 for no hit."""
        from up.extensions.python.roiaware_pool3d_utils import _nvidia_points_in_boxes_gpu
        return _nvidia_points_in_boxes_gpu(points, boxes)

    def roiaware(self, rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method):
        """Voxel max/avg pooling; retain saved indices and feature backward."""
        from up.extensions.python.roiaware_pool3d_utils import RoIAwarePool3dFunction
        return Function.apply.__func__(RoIAwarePool3dFunction, rois, pts, pts_feature, out_size, max_pts_each_voxel, pool_method)

    def roipoint(self, points, point_features, boxes3d, pool_extra_width, num_sampled_points=512):
        """Point sampling and empty flags; original backward remains unsupported."""
        from up.extensions.python.roipoint_pool3d_utils import RoIPointPool3dFunction
        return Function.apply.__func__(RoIPointPool3dFunction, points, point_features, boxes3d, pool_extra_width, num_sampled_points)
