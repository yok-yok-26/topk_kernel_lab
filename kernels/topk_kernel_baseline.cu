#include "topk.h"

#include <cuda_runtime.h>
#include <float.h>

#include "cuda_check.cuh"

namespace topk_lab {
namespace {



//////////////////////////////////////////////////////////////
// Baseline implementation: one thread per row. Not fast, but simple and easy to understand
//////////////////////////////////////////////////////////////  
__global__ void topk_naive_one_thread_per_row(const float* __restrict__ input,
                                              float* __restrict__ values,
                                              int* __restrict__ indices,
                                              int batch,
                                              int n,
                                              int k) {
  const int row = blockIdx.x;
  if (row >= batch || threadIdx.x != 0) {
    return;
  }

  int chosen[kMaxK];
  for (int i = 0; i < kMaxK; ++i) {
    chosen[i] = -1;
  }

  const float* row_in = input + static_cast<long long>(row) * n;
  float* row_values = values + static_cast<long long>(row) * k;
  int* row_indices = indices + static_cast<long long>(row) * k;

  for (int out = 0; out < k; ++out) {
    float best_value = -FLT_MAX;
    int best_index = -1;

    for (int col = 0; col < n; ++col) {
      bool already_chosen = false;
      for (int prev = 0; prev < out; ++prev) {
        already_chosen = already_chosen || (chosen[prev] == col);
      }
      if (already_chosen) {
        continue;
      }

      const float candidate = row_in[col];
      if (candidate > best_value ||
          (candidate == best_value && (best_index < 0 || col < best_index))) {
        best_value = candidate;
        best_index = col;
      }
    }

    chosen[out] = best_index;
    row_values[out] = best_value;
    row_indices[out] = best_index;
  }
}

}  // namespace

cudaError_t topk_cuda(const float* input,
                      float* values,
                      int* indices,
                      TopKShape shape,
                      cudaStream_t stream) {
  if (input == nullptr || values == nullptr || indices == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (shape.batch < 0 || shape.n < 0 || shape.k < 0 || shape.k > kMaxK ||
      shape.k > shape.n) {
    return cudaErrorInvalidValue;
  }
  if (shape.batch == 0 || shape.k == 0) {
    return cudaSuccess;
  }

  // Baseline intentionally favors clarity over speed. Your first optimization
  // exercise is to replace the one-thread-per-row work with a parallel kernel.
  const dim3 grid(shape.batch);
  const dim3 block(1);
  topk_naive_one_thread_per_row<<<grid, block, 0, stream>>>(
      input, values, indices, shape.batch, shape.n, shape.k);
  CUDA_KERNEL_CHECK();

#ifdef DEBUG_CUDA_SYNC
  CUDA_CHECK(cudaStreamSynchronize(stream));
#endif

  return cudaSuccess;
}

}  // namespace topk_lab
