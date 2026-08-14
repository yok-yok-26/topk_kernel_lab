#include "topk.h"

#include <cuda_runtime.h>
#include <float.h>

#include "cuda_check.cuh"

namespace topk_lab {

//////////////////////////////////////////////////////////////
// type1 
//////////////////////////////////////////////////////////////
namespace {
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



/// @brief 
/// @param a 
/// @param b 
/// @param ascending 
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

    for (int k = 2; k <= N; k<<=1)     // !!!!!N!!! 一定不是BLOCK_SIZE
    {
        for (int j = k>>1; j > 0; j>>=1)
        {
            for (int i = threadIdx.x; i < N; i+=BLOCK_SIZE)
            {
                bool ascending;
                int ixj = i ^ j;

                if (ixj > i)    // 是否跳过这个数据 !!!!!!!!注意一定是i
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



/// @param input 输入数组
/// @param total_num 输入数组长度
/// @param out_val 输出topk的值
/// @param out_idx 输出topk的索引
/// @param K topk的k值
/// @param N bitonic排序的长度，必须是BLOCK_SIZE + K的下一个2的幂次方
template<size_t BLOCK_SIZE>
__global__
void topk_bitonic_per_block_kernel(
    const float* __restrict__ input,
    int total_num,
    float* __restrict__ out_val,
    int* __restrict__ out_idx,
    int K,
    int N, 
    int Batch)
{

    if (blockDim.x != BLOCK_SIZE) return;
    
    extern __shared__ TopKPair seme[];
    TopKPair* s_topk = seme;
    TopKPair* s_batch = &seme[K];
    int tx = threadIdx.x;
    int ibatch = blockIdx.y;

    // // debug
    // if (total_num == 7 && blockIdx.x == 0 && tx == 0 && blockIdx.y == 0)
    // {
    //     printf("input[0] = %f\n", input[0]);
    //     printf("input[1] = %f\n", input[1]);
    //     printf("input[2] = %f\n", input[2]);
    //     printf("input[3] = %f\n", input[3]);
    //     printf("input[4] = %f\n", input[4]);
    //     printf("input[5] = %f\n", input[5]);
    //     printf("input[6] = %f\n", input[6]);
    // }


    // 赋初值
    for (int i = tx; i < K; i+=BLOCK_SIZE)
    {
        s_topk[i].idx = -1;
        s_topk[i].val = -MAXFLOAT;
    }
    // for (int i = tx + K; i < N; i+=BLOCK_SIZE)
    // {
    //     // 注意边界 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     if (i >= (K + BLOCK_SIZE))
    //     {
    //         // !!!!!!!!!!!!!!!!!!! s_batch会导致分段错误 !!!!!!!!!!!!!!!!!!
    //         seme[i].idx = -1;
    //         seme[i].val = -MAXFLOAT/400.0f;  // 注意不能直接赋值 -MAXFLOAT，否则可能会被优化掉，导致排序不正确
    //     }
    // }
    __syncthreads();

    /*
    致命错误
    for (int i = tx; i < total_num; i+=(gridDim.x * BLOCK_SIZE))
    {
        int gtx = i + BLOCK_SIZE * blockIdx.x;
    */

    for (int base = blockIdx.x * BLOCK_SIZE; base < total_num; base+=(gridDim.x * BLOCK_SIZE))
    {
        int gtx = base + tx;
        if (gtx < total_num)
        {
            s_batch[tx].idx = gtx;
            s_batch[tx].val = input[gtx + ibatch * total_num];
        }
        else
        {
            s_batch[tx].idx = -1;
            s_batch[tx].val = -MAXFLOAT;
        }

        for (int p = tx + K + BLOCK_SIZE; p < N; p+=BLOCK_SIZE)
        {
            seme[p].idx = -1;
            seme[p].val = -MAXFLOAT/200.0f;
        }
        __syncthreads();
        
        bitonic_sort_power2_ascending<BLOCK_SIZE>(seme, N);

        // 归并到topk !!!!!!!!!!!!!!!!!!!!!!
        if (tx < K)
        {
            seme[tx].idx = seme[N - 1 - tx].idx;
            seme[tx].val = seme[N - 1 - tx].val;
        }
        __syncthreads();
    }

    // // debug
    // if (total_num == 7 && blockIdx.x == 0 && tx == 0 && blockIdx.y == 0)
    // {
    //   for (int j = 0; j < N; j++)
    //   {
    //     printf("seme[%d] %d ", j, seme[j].idx);
    //     printf("seme[%d] %f\n", j, seme[j].val);
    //   }
    // }
    
    
    for (int i = tx; i < K; i+=BLOCK_SIZE)
    {
        out_idx[i + blockIdx.x * K + ibatch * K * gridDim.x] = seme[i].idx;
        out_val[i + blockIdx.x * K + ibatch * K * gridDim.x] = seme[i].val;
    }

}


template<size_t BLOCK_SIZE>
__global__
void topk_bitonic_per_block_kernel_pingpong(
    float* __restrict__ in_val,
    int* __restrict__ in_idx,
    int total_num,
    float* __restrict__ out_val,
    int* __restrict__ out_idx,
    int K,
    int N, 
    int Batch)
{

    extern __shared__ TopKPair seme[];
    TopKPair* s_topk = seme;
    TopKPair* s_batch = &seme[K];
    int tx = threadIdx.x;
    int ibatch = blockIdx.y;

    // 赋初值
    for (int i = tx; i < K; i+=BLOCK_SIZE)
    {
        s_topk[i].idx = -1;
        s_topk[i].val = -MAXFLOAT;
    }
    // for (int i = tx + K; i < N; i+=BLOCK_SIZE)
    // {
    //     // 注意边界 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //     if (i >= (K + BLOCK_SIZE))
    //     {
    //         // !!!!!!!!!!!!!!!!!!! s_batch会导致分段错误 !!!!!!!!!!!!!!!!!!
    //         seme[i].idx = -1;
    //         seme[i].val = -MAXFLOAT;
    //     }
    // }
    __syncthreads();

    // for (int i = tx; i < total_num; i+=(gridDim.x * BLOCK_SIZE))
    // {
    //     int gtx = i + BLOCK_SIZE * blockIdx.x;
    for (int base = blockIdx.x * BLOCK_SIZE; base < total_num; base+=(gridDim.x * BLOCK_SIZE))
    {
        int gtx = base + tx;
        if (gtx < total_num)
        {
            s_batch[tx].idx = in_idx[gtx + ibatch * total_num];
            s_batch[tx].val = in_val[gtx + ibatch * total_num];
        }
        else
        {
            s_batch[tx].idx = -1;
            s_batch[tx].val = -MAXFLOAT;
        }

        for (int p = tx + K + BLOCK_SIZE; p < N; p+=BLOCK_SIZE)
        {
            seme[p].idx = -1;
            seme[p].val = -MAXFLOAT;
        }
        
        __syncthreads();
        
        bitonic_sort_power2_ascending<BLOCK_SIZE>(seme, N);

        // 归并到topk !!!!!!!!!!!!!!!!!!!!!!
        if (tx < K)
        {
            seme[tx].idx = seme[N - 1 - tx].idx;
            seme[tx].val = seme[N - 1 - tx].val;
        }
        __syncthreads();
    }
    
    for (int i = tx; i < K; i+=BLOCK_SIZE)
    {
        out_idx[i + blockIdx.x * K + ibatch * K * gridDim.x] = seme[i].idx;
        out_val[i + blockIdx.x * K + ibatch * K * gridDim.x] = seme[i].val;
    }

}
}  // namespace


cudaError_t topk_cuda_seme(const float* input,
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
  const int N = next_power_of_2(BLOCK_SIZE + shape.k);
  dim3 grid((shape.n + BLOCK_SIZE - 1) / BLOCK_SIZE, shape.batch);
  const dim3 block(BLOCK_SIZE);

  bool ping{true};
  float* out_val_tmp = tmp_a_val;
  int* out_idx_tmp = tmp_a_idx;
  float* in_val_tmp = tmp_b_val;
  int* in_idx_tmp = tmp_b_idx;
  
  int out_current_num{shape.n};
  topk_bitonic_per_block_kernel<BLOCK_SIZE><<<grid, block, N * sizeof(TopKPair), stream>>>(
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
      topk_bitonic_per_block_kernel_pingpong<BLOCK_SIZE><<<grid, block, N * sizeof(TopKPair), stream>>>(
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
