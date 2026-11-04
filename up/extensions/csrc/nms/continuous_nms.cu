// Continuous-coordinate NMS for YOLO. Input order is chosen by vendor Torch.
// Both the pairwise suppression mask and greedy selection stay on the GPU.
#include "nms/nms.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

namespace {
constexpr int kBits = 64;

template <typename T>
__device__ bool overlaps(const T* a, const T* b, T threshold) {
  // Named intermediates preserve the old elementwise Torch rounding.
  T width = (a[2] < b[2] ? a[2] : b[2]) -
            (a[0] > b[0] ? a[0] : b[0]);
  T height = (a[3] < b[3] ? a[3] : b[3]) -
             (a[1] > b[1] ? a[1] : b[1]);
  width = width > T(0) ? width : T(0);
  height = height > T(0) ? height : T(0);
  T intersection = width * height;
  T aw = a[2] - a[0], ah = a[3] - a[1];
  T bw = b[2] - b[0], bh = b[3] - b[1];
  T area_a = aw * ah, area_b = bw * bh;
  T total = area_a + area_b;
  T denominator = total - intersection;
  T iou = intersection / denominator;
  return iou > threshold;
}

template <typename T>
__global__ void pairwise_mask(const T* boxes, int64_t count, int64_t words,
                              T threshold, unsigned long long* mask) {
  const int64_t row_word = blockIdx.y, col_word = blockIdx.x;
  if (col_word < row_word) return;
  __shared__ T columns[kBits * 4];
  const int lane = threadIdx.x;
  const int64_t column = col_word * kBits + lane;
  if (column < count) {
    for (int d = 0; d < 4; ++d) columns[lane * 4 + d] = boxes[column * 4 + d];
  }
  __syncthreads();
  const int64_t row = row_word * kBits + lane;
  if (row >= count) return;
  const int limit = count - col_word * kBits < kBits ? count - col_word * kBits : kBits;
  const int begin = row_word == col_word ? lane + 1 : 0;
  unsigned long long bits = 0;
  for (int j = begin; j < limit; ++j) {
    if (overlaps(boxes + row * 4, columns + j * 4, threshold)) bits |= 1ULL << j;
  }
  mask[row * words + col_word] = bits;
}

__global__ void select_kept(const unsigned long long* mask, int64_t count,
                            int64_t words, unsigned long long* removed, bool* keep) {
  // One block makes greedy decisions in score order; its threads merge mask words.
  // No atomics, CPU mask copy or per-box kernel launch is required.
  for (int64_t word = 0; word < words; ++word) {
    unsigned long long current = removed[word];
    __syncthreads();
    const int limit = count - word * kBits < kBits ? count - word * kBits : kBits;
    for (int bit = 0; bit < limit; ++bit) {
      if (current & (1ULL << bit)) continue;
      const int64_t index = word * kBits + bit;
      if (threadIdx.x == 0) keep[index] = true;
      for (int64_t j = word + threadIdx.x; j < words; j += blockDim.x) {
        removed[j] |= mask[index * words + j];
      }
      __syncthreads();
      current = removed[word];
      // Finish every reader before the next accepted box writes this word.
      __syncthreads();
    }
  }
}
}  // namespace

at::Tensor continuous_nms_mask(at::Tensor boxes, double threshold) {
  TORCH_CHECK(boxes.is_cuda() && boxes.is_contiguous(),
              "continuous NMS requires contiguous GPU boxes");
  TORCH_CHECK(boxes.dim() == 2 && boxes.size(1) == 4, "NMS boxes must be [N, 4]");
  const c10::cuda::CUDAGuard guard(boxes.device());
  const int64_t count = boxes.size(0), words = (count + kBits - 1) / kBits;
  auto keep = at::zeros({count}, boxes.options().dtype(at::kBool).requires_grad(false));
  if (count == 0) return keep;
  TORCH_CHECK(words <= 65535, "Too many boxes for the NMS launch grid");
  auto mask = at::empty({count, words}, boxes.options().dtype(at::kLong).requires_grad(false));
  auto removed = at::zeros({words}, boxes.options().dtype(at::kLong).requires_grad(false));
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(boxes.scalar_type(), "continuous_nms_mask", [&] {
    pairwise_mask<scalar_t><<<dim3(words, words), kBits, 0, stream>>>(
        boxes.data_ptr<scalar_t>(), count, words, scalar_t(threshold),
        reinterpret_cast<unsigned long long*>(mask.data_ptr<int64_t>()));
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  select_kept<<<1, 256, 0, stream>>>(
      reinterpret_cast<unsigned long long*>(mask.data_ptr<int64_t>()), count, words,
      reinterpret_cast<unsigned long long*>(removed.data_ptr<int64_t>()), keep.data_ptr<bool>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return keep;
}
