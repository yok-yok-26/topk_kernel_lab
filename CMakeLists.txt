cmake_minimum_required(VERSION 3.22)
project(topk_kernel_lab LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h PATHS /usr/include/x86_64-linux-gnu REQUIRED)
find_library(TENSORRT_LIBRARY nvinfer REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

option(TOPK_DEBUG_SYNC "Synchronize after kernel launches in wrappers" OFF)

add_library(topk_cuda STATIC
  kernels/topk_kernel_baseline.cu
  kernels/topk_kernelv1.cu
  kernels/topk_kernelv2.cu
  kernels/topk_kernelv3.cu
)
target_include_directories(topk_cuda PUBLIC include ${CUDAToolkit_INCLUDE_DIRS})
target_compile_options(topk_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
)
if(TOPK_DEBUG_SYNC)
  target_compile_definitions(topk_cuda PUBLIC DEBUG_CUDA_SYNC=1)
endif()

add_executable(test_topk tests/test_topk.cpp)
target_link_libraries(test_topk PRIVATE topk_cuda CUDA::cudart)

set_source_files_properties(benchmarks/bench_topk.cpp PROPERTIES LANGUAGE CUDA)
add_executable(bench_topk benchmarks/bench_topk.cpp)
target_include_directories(bench_topk PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(bench_topk PRIVATE topk_cuda CUDA::cudart ${TENSORRT_LIBRARY})
