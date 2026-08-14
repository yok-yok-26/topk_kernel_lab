#include "topk.h"

#include <cuda_runtime.h>
#include <float.h>

#include "cuda_check.cuh"

namespace topk_lab {

struct TopKPair {
    float val;
    int   idx;
};

int next_power_of_2(int x)
{
    int n = 1;
    while (n < x) {
        n <<= 1;
    }
    return n;
}

__device__ __forceinline__ 
int next_power_of_2_device(int x)
{
    int n = 1;
    while (n < x) {
        n <<= 1;
    }
    return n;
}

__device__ __forceinline__ 
bool pair_greater(const TopKPair& a, const TopKPair& b){

    if (a.val < b.val) return false;
    if (a.val > b.val) return true;

    return (a.idx < b.idx);
}

__device__ __forceinline__ 
bool pair_less(const TopKPair& a, const TopKPair& b){

    if (a.val > b.val) return false;
    if (a.val < b.val) return true;

    return (a.idx > b.idx);
}

__device__ __forceinline__ 
void pair_greater_swap(TopKPair& a, TopKPair& b, bool ascending){

    bool need_swap;

    if (ascending)
    {
        need_swap = pair_greater(a, b);
    }
    else
    {
        need_swap = pair_less(a,b);
    }
    
    if (need_swap)
    {
        TopKPair tmp;
        tmp = a;
        a = b;
        b = tmp;
    }
}

template<size_t BLOCK_SIZE>
__device__
void bitonic_sort_power2_ascending(TopKPair* data, int N){

    for (int k = 2; k <= N; k<<=1)
    {
        for (int j = k>>1; j > 0; j>>=1)
        {
            for (int i = threadIdx.x; i < N; i+=BLOCK_SIZE)
            {
                bool ascending;
                int ixj = i ^ j;

                if (ixj > i)
                {
                    if ((i & k) == 0)
                    {
                        ascending = true;
                    }
                    else
                    {
                        ascending = false;
                    }
                    
                    pair_greater_swap(data[i], data[ixj], ascending);
                }
                
            }
            __syncthreads();
        }        
    }

}

//////////////////////////////////////////////////////////////
// type2 
//////////////////////////////////////////////////////////////
namespace {


template<size_t N>
__device__
void bitonic_sort_power2_warp_ascending(TopKPair& local){

    unsigned mask = __activemask();

    for (int k = 2; k <= N; k<<=1)     // !!!!!N!!! 一定不是BLOCK_SIZE
    {
        for (int j = k>>1; j > 0; j>>=1)
        {
            int i = threadIdx.x & 31;            // 注意⚠️ i的表示范围，不是block，是warp
            TopKPair other{-MAXFLOAT, -1};
            other.val = __shfl_xor_sync(mask, local.val, j);
            other.idx = __shfl_xor_sync(mask, local.idx, j);
            bool ascending = ((i & k) == 0);    // 注意⚠️ 运算符的优先级 ！！！！！！！！！！！！！！
            int ixj = i ^ j;

            if (ascending)
            {
                if (i < ixj)
                {
                    if (other.val < local.val)
                    {
                        local = other;
                    }
                    else if (other.val > local.val) {}
                    else
                    {
                        if (local.idx < other.idx){
                            local = other;
                        }
                    }
                }
                else
                {
                    if (other.val > local.val)
                    {
                        local = other;
                    }
                    else if (other.val < local.val) {}
                    else
                    {
                        if (local.idx > other.idx){
                            local = other;
                        }
                    }
                }
            }
            else
            {
                if (i < ixj)
                {
                    if (other.val > local.val)
                    {
                        local = other;
                    }
                    else if (other.val < local.val) {}
                    else
                    {
                        if (local.idx > other.idx){
                            local = other;
                        }
                    }                    
                }
                else
                {
                    if (other.val < local.val)
                    {
                        local = other;
                    }
                    else if (other.val > local.val) {}
                    else
                    {
                        if (local.idx < other.idx){
                            local = other;
                        }
                    }
                }
            }

            __syncwarp();
        }        
    }

}



/// @brief block 内不循环！
template<size_t BLOCK_SIZE, size_t WARP_SIZE>
__global__
void topk_bitonic_per_block_warp_kernel(
    const float* __restrict__ input,
    int total_num,
    float* __restrict__ out_val,
    int* __restrict__ out_idx,
    int K,
    int N, 
    int Batch)
{

    if (K > WARP_SIZE) return;

    int warp_num = blockDim.x >> 5;
    if (N != next_power_of_2_device(warp_num * K)) return;
    
    extern __shared__ TopKPair seme[];
    int tx = threadIdx.x;
    int ibatch = blockIdx.y;

    int gtx = blockDim.x * blockIdx.x + threadIdx.x;

    TopKPair local{-MAXFLOAT, -1};
    if (gtx < total_num)
    {
        local.val = input[gtx + ibatch * total_num];
        local.idx = gtx;
    }
    
    bitonic_sort_power2_warp_ascending<WARP_SIZE>(local);

    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane_id >= WARP_SIZE - K)
    {
        int sdx = warp_id * K + (WARP_SIZE - lane_id - 1);
        seme[sdx] = local;
    }
    
    if (tx + K * warp_num < N)
    {
        seme[tx + K * warp_num].val = -MAXFLOAT;
        seme[tx + K * warp_num].idx = -1;
    }
    
    __syncthreads();


    bitonic_sort_power2_ascending<BLOCK_SIZE>(seme, N);

    if (tx < N && tx >= (N - K))
    {
        out_val[(N - tx - 1) + blockIdx.x * K + ibatch * K * gridDim.x] = seme[tx].val;
        out_idx[(N - tx - 1) + blockIdx.x * K + ibatch * K * gridDim.x] = seme[tx].idx;
    }
    
}




/// @brief block 内不循环！
template<size_t BLOCK_SIZE, size_t WARP_SIZE>
__global__
void topk_bitonic_per_block_warp_kernel_pingpong(
    float* __restrict__ in_val,
    int* __restrict__ in_idx,
    int total_num,
    float* __restrict__ out_val,
    int* __restrict__ out_idx,
    int K,
    int N, 
    int Batch)
{

    if (K > WARP_SIZE) return;

    int warp_num = blockDim.x >> 5;
    if (N != next_power_of_2_device(warp_num * K)) return;
    
    extern __shared__ TopKPair seme[];
    int tx = threadIdx.x;
    int ibatch = blockIdx.y;

    int gtx = blockDim.x * blockIdx.x + threadIdx.x;

    TopKPair local{-MAXFLOAT, -1};
    if (gtx < total_num)
    {
        local.val = in_val[gtx + ibatch * total_num];
        local.idx = in_idx[gtx + ibatch * total_num];
    }
    
    bitonic_sort_power2_warp_ascending<WARP_SIZE>(local);

    int lane_id = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane_id >= WARP_SIZE - K)
    {
        int sdx = warp_id * K + (WARP_SIZE - lane_id - 1);
        seme[sdx] = local;
    }
    
    if (tx + K * warp_num < N)
    {
        seme[tx + K * warp_num].val = -MAXFLOAT;
        seme[tx + K * warp_num].idx = -1;
    }
    
    __syncthreads();


    bitonic_sort_power2_ascending<BLOCK_SIZE>(seme, N);

    if (tx < N && tx >= (N - K))
    {
        out_val[(N - tx - 1) + blockIdx.x * K + ibatch * K * gridDim.x] = seme[tx].val;
        out_idx[(N - tx - 1) + blockIdx.x * K + ibatch * K * gridDim.x] = seme[tx].idx;
    }
    
}


}  // namespace


cudaError_t topk_cuda_warp(const float* input,
                      float* values,
                      int* indices,
                      float* tmp_a_val,
                      int* tmp_a_idx,
                      float* tmp_b_val,
                      int* tmp_b_idx,
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

  const int BLOCK_SIZE = 256;
  const int N = next_power_of_2((BLOCK_SIZE / 32) * shape.k); 
  dim3 grid((shape.n + BLOCK_SIZE - 1) / BLOCK_SIZE, shape.batch);
  const dim3 block(BLOCK_SIZE);

  bool ping{true};
  float* out_val_tmp = tmp_a_val;
  int* out_idx_tmp = tmp_a_idx;
  float* in_val_tmp = tmp_b_val;
  int* in_idx_tmp = tmp_b_idx;
  
  int out_current_num{shape.n};
  topk_bitonic_per_block_warp_kernel<BLOCK_SIZE, 32><<<grid, block, N * sizeof(TopKPair), stream>>>(
      input, shape.n, out_val_tmp, out_idx_tmp, shape.k, N, shape.batch);

  if (out_current_num > BLOCK_SIZE)
  {
    do
    {
      ping = !ping;
      if (ping)
      {
        out_val_tmp = tmp_a_val;
        out_idx_tmp = tmp_a_idx;
        in_val_tmp = tmp_b_val;
        in_idx_tmp = tmp_b_idx;
      }
      else
      {
        out_val_tmp = tmp_b_val;
        out_idx_tmp = tmp_b_idx;
        in_val_tmp = tmp_a_val;
        in_idx_tmp = tmp_a_idx;
      }
      
      out_current_num = grid.x * shape.k;
      grid.x = (out_current_num + BLOCK_SIZE - 1) / BLOCK_SIZE;
      topk_bitonic_per_block_warp_kernel_pingpong<BLOCK_SIZE, 32><<<grid, block, N * sizeof(TopKPair), stream>>>(
        in_val_tmp, in_idx_tmp, out_current_num, out_val_tmp, out_idx_tmp, shape.k, N, shape.batch);
    } while (out_current_num > BLOCK_SIZE);

  }

  CUDA_CHECK(cudaMemcpyAsync(values, out_val_tmp, static_cast<size_t>(shape.batch) * shape.k * sizeof(float), cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(indices, out_idx_tmp, static_cast<size_t>(shape.batch) * shape.k * sizeof(int), cudaMemcpyDeviceToDevice, stream));

  CUDA_KERNEL_CHECK();

#ifdef DEBUG_CUDA_SYNC
  CUDA_CHECK(cudaStreamSynchronize(stream));
#endif

  return cudaSuccess;
}





}  // namespace topk_lab
