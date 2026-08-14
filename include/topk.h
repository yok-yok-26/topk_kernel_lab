#pragma once

#include <cuda_runtime.h>

namespace topk_lab {

constexpr int kMaxK = 128;

struct TopKShape {
  int batch;
  int n;
  int k;
};

// Contract v0:
// input: row-major float32 [batch, n]
// values: row-major float32 [batch, k], sorted descending per row
// indices: row-major int32 [batch, k], original column indices
// ties: lower source index wins
cudaError_t topk_cuda(const float* input,
                      float* values,
                      int* indices,
                      TopKShape shape,
                      cudaStream_t stream = nullptr);

cudaError_t topk_cuda_seme(const float* input,
                           float* values,
                           int* indices,
                           float* tmp_a_val,
                           int* tmp_a_idx,
                           float* tmp_b_val,
                           int* tmp_b_idx,
                           TopKShape shape,
                           cudaStream_t stream = nullptr);

cudaError_t topk_cuda_warp(const float* input,
                           float* values,
                           int* indices,
                           float* tmp_a_val,
                           int* tmp_a_idx,
                           float* tmp_b_val,
                           int* tmp_b_idx,
                           TopKShape shape,
                           cudaStream_t stream = nullptr);

cudaError_t topk_cuda_SoA_warp(const float* input,
                               float* values,
                               int* indices,
                               float* tmp_a_val,
                               int* tmp_a_idx,
                               float* tmp_b_val,
                               int* tmp_b_idx,
                               TopKShape shape,
                               cudaStream_t stream = nullptr);

}  // namespace topk_lab
