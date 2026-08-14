#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "cuda_check.cuh"
#include "topk.h"

namespace {

bool profiler_capture_requested() {
  return std::getenv("TOPK_PROFILE_SINGLE_OP") != nullptr;
}

void start_profiler_capture() {
  if (profiler_capture_requested()) {
    CUDA_CHECK(cudaProfilerStart());
  }
}

void stop_profiler_capture() {
  if (profiler_capture_requested()) {
    CUDA_CHECK(cudaProfilerStop());
  }
}

enum class TopKMode {
  kNaive,
  kV1,
  kV2,
  kV3,
  kCubSort,
  kCubRadixSort,
  kTensorRtTopK,
};

const char* mode_name(TopKMode mode) {
  switch (mode) {
    case TopKMode::kNaive: return "naive";
    case TopKMode::kV1: return "v1";
    case TopKMode::kV2: return "v2";
    case TopKMode::kV3: return "v3";
    case TopKMode::kCubSort: return "library_cub_stable_segmented_sort";
    case TopKMode::kCubRadixSort: return "library_cub_segmented_radix_sort";
    case TopKMode::kTensorRtTopK: return "library_tensorrt_topk";
  }
  return "unknown";
}

bool parse_mode(const std::string& name, TopKMode* mode) {
  if (name == "naive") {
    *mode = TopKMode::kNaive;
    return true;
  }
  if (name == "v1" || name == "seme") {
    *mode = TopKMode::kV1;
    return true;
  }
  if (name == "v2" || name == "warp") {
    *mode = TopKMode::kV2;
    return true;
  }
  if (name == "v3" || name == "soa_warp" || name == "SoA_warp") {
    *mode = TopKMode::kV3;
    return true;
  }
  if (name == "cub_sort" || name == "library_cub_segmented_sort"
      || name == "library_cub_stable_segmented_sort") {
    *mode = TopKMode::kCubSort;
    return true;
  }
  if (name == "cub_radix_sort" || name == "library_cub_segmented_radix_sort") {
    *mode = TopKMode::kCubRadixSort;
    return true;
  }
  if (name == "tensorrt_topk" || name == "library_tensorrt_topk") {
    *mode = TopKMode::kTensorRtTopK;
    return true;
  }
  return false;
}

cudaError_t launch_topk(TopKMode mode,
                        const float* input,
                        float* values,
                        int* indices,
                        float* tmp_a_val,
                        int* tmp_a_idx,
                        float* tmp_b_val,
                        int* tmp_b_idx,
                        topk_lab::TopKShape shape) {
  switch (mode) {
    case TopKMode::kNaive:
      return topk_lab::topk_cuda(input, values, indices, shape);
    case TopKMode::kV1:
      return topk_lab::topk_cuda_seme(input, values, indices, tmp_a_val, tmp_a_idx, tmp_b_val, tmp_b_idx, shape);
    case TopKMode::kV2:
      return topk_lab::topk_cuda_warp(input, values, indices, tmp_a_val, tmp_a_idx, tmp_b_val, tmp_b_idx, shape);
    case TopKMode::kV3:
      return topk_lab::topk_cuda_SoA_warp(input, values, indices, tmp_a_val, tmp_a_idx, tmp_b_val, tmp_b_idx, shape);
    case TopKMode::kCubSort:
    case TopKMode::kCubRadixSort:
    case TopKMode::kTensorRtTopK:
      return cudaErrorInvalidValue;
  }
  return cudaErrorInvalidValue;
}

void run_one(TopKMode mode, int batch, int n, int k, int warmup, int iters, std::ostream& csv) {
  std::mt19937 rng(2026 + batch + n + k);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> h_input(static_cast<size_t>(batch) * n);
  for (float& v : h_input) v = dist(rng);

  float* d_input = nullptr;
  float* d_values = nullptr;
  int* d_indices = nullptr;
  float* d_tmp_a_val = nullptr;
  int* d_tmp_a_idx = nullptr;
  float* d_tmp_b_val = nullptr;
  int* d_tmp_b_idx = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_values, static_cast<size_t>(batch) * k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_indices, static_cast<size_t>(batch) * k * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_tmp_a_val, h_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tmp_a_idx, h_input.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_tmp_b_val, h_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tmp_b_idx, h_input.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

  for (int i = 0; i < warmup; ++i) {
    CUDA_CHECK(launch_topk(mode, d_input, d_values, d_indices, d_tmp_a_val, d_tmp_a_idx, d_tmp_b_val, d_tmp_b_idx, {batch, n, k}));
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  start_profiler_capture();
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(launch_topk(mode, d_input, d_values, d_indices, d_tmp_a_val, d_tmp_a_idx, d_tmp_b_val, d_tmp_b_idx, {batch, n, k}));
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  stop_profiler_capture();
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  const double avg_ms = static_cast<double>(ms) / iters;
  const double read_gb = static_cast<double>(batch) * n * sizeof(float) / 1.0e9;
  const double write_gb = static_cast<double>(batch) * k * (sizeof(float) + sizeof(int)) / 1.0e9;
  const double bandwidth_gbs = (read_gb + write_gb) / (avg_ms / 1000.0);
  csv << mode_name(mode) << "," << batch << "," << n << "," << k << ","
      << avg_ms << "," << bandwidth_gbs << "\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_indices));
  CUDA_CHECK(cudaFree(d_tmp_a_val));
  CUDA_CHECK(cudaFree(d_tmp_a_idx));
  CUDA_CHECK(cudaFree(d_tmp_b_val));
  CUDA_CHECK(cudaFree(d_tmp_b_idx));
}

void run_one_cub_sort(
    TopKMode mode, int batch, int n, int k, int warmup, int iters, std::ostream& csv) {
  std::mt19937 rng(2026 + batch + n + k);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> h_input(static_cast<size_t>(batch) * n);
  for (float& v : h_input) v = dist(rng);

  std::vector<int> h_indices(static_cast<size_t>(batch) * n);
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < n; ++i) {
      h_indices[static_cast<size_t>(b) * n + i] = i;
    }
  }
  std::vector<int> h_begin(batch);
  std::vector<int> h_end(batch);
  for (int b = 0; b < batch; ++b) {
    h_begin[b] = b * n;
    h_end[b] = (b + 1) * n;
  }

  const int num_items = batch * n;
  float* d_keys_in = nullptr;
  float* d_keys_out = nullptr;
  int* d_vals_in = nullptr;
  int* d_vals_out = nullptr;
  int* d_begin = nullptr;
  int* d_end = nullptr;
  void* d_temp = nullptr;
  size_t temp_bytes = 0;

  CUDA_CHECK(cudaMalloc(&d_keys_in, h_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_keys_out, h_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_vals_in, h_indices.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_vals_out, h_indices.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_begin, h_begin.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_end, h_end.size() * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_keys_in, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_vals_in, h_indices.data(), h_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_begin, h_begin.data(), h_begin.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_end, h_end.data(), h_end.size() * sizeof(int), cudaMemcpyHostToDevice));

  auto invoke_sort = [&]() {
    if (mode == TopKMode::kCubRadixSort) {
      return cub::DeviceSegmentedRadixSort::SortPairsDescending(
          d_temp, temp_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out,
          num_items, batch, d_begin, d_end);
    }
    return cub::DeviceSegmentedSort::StableSortPairsDescending(
        d_temp, temp_bytes, d_keys_in, d_keys_out, d_vals_in, d_vals_out,
        num_items, batch, d_begin, d_end);
  };

  CUDA_CHECK(invoke_sort());
  CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));

  for (int i = 0; i < warmup; ++i) {
    CUDA_CHECK(invoke_sort());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  start_profiler_capture();
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    CUDA_CHECK(invoke_sort());
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  stop_profiler_capture();
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<float> h_keys_out(h_input.size());
  std::vector<int> h_vals_out(h_indices.size());
  CUDA_CHECK(cudaMemcpy(
      h_keys_out.data(), d_keys_out, h_keys_out.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      h_vals_out.data(), d_vals_out, h_vals_out.size() * sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> expected_order(n);
  for (int b = 0; b < batch; ++b) {
    const float* input_row = h_input.data() + static_cast<size_t>(b) * n;
    std::iota(expected_order.begin(), expected_order.end(), 0);
    std::partial_sort(
        expected_order.begin(), expected_order.begin() + k, expected_order.end(),
        [input_row](int lhs, int rhs) {
          return input_row[lhs] > input_row[rhs]
              || (input_row[lhs] == input_row[rhs] && lhs < rhs);
        });
    for (int rank = 0; rank < k; ++rank) {
      const size_t sorted = static_cast<size_t>(b) * n + rank;
      if (h_vals_out[sorted] != expected_order[rank]
          || h_keys_out[sorted] != input_row[expected_order[rank]]) {
        throw std::runtime_error(
            std::string(mode_name(mode)) + " correctness failed at batch=" + std::to_string(b)
            + " rank=" + std::to_string(rank));
      }
    }
  }
  {
    std::ofstream correctness(
        "reports/correctness/" + std::string(mode_name(mode)) + "_"
        + std::to_string(batch) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".log");
    correctness << "PASS mode=" << mode_name(mode)
                << " batch=" << batch << " n=" << n << " k=" << k
                << " sorted_values=pass lower_index_tie=pass"
                << " caveat=full_row_sort\n";
  }

  const double avg_ms = static_cast<double>(ms) / iters;
  const double read_gb = static_cast<double>(batch) * n * (sizeof(float) + sizeof(int)) / 1.0e9;
  const double write_gb = static_cast<double>(batch) * n * (sizeof(float) + sizeof(int)) / 1.0e9;
  const double bandwidth_gbs = (read_gb + write_gb) / (avg_ms / 1000.0);
  csv << mode_name(mode) << "," << batch << "," << n << "," << k << ","
      << avg_ms << "," << bandwidth_gbs << "\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_keys_in));
  CUDA_CHECK(cudaFree(d_keys_out));
  CUDA_CHECK(cudaFree(d_vals_in));
  CUDA_CHECK(cudaFree(d_vals_out));
  CUDA_CHECK(cudaFree(d_begin));
  CUDA_CHECK(cudaFree(d_end));
  CUDA_CHECK(cudaFree(d_temp));
}

class TensorRtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* message) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << message << "\n";
    }
  }
};

template <typename T>
std::unique_ptr<T> checked_trt_ptr(T* ptr, const char* what) {
  if (ptr == nullptr) {
    throw std::runtime_error(std::string("TensorRT failed to create ") + what);
  }
  return std::unique_ptr<T>(ptr);
}

void validate_tensorrt_result(const std::vector<float>& input,
                              const std::vector<float>& values,
                              const std::vector<int>& indices,
                              int batch,
                              int n,
                              int k) {
  std::vector<float> row(n);
  for (int b = 0; b < batch; ++b) {
    const float* input_row = input.data() + static_cast<size_t>(b) * n;
    std::copy(input_row, input_row + n, row.begin());
    std::partial_sort(row.begin(), row.begin() + k, row.end(), std::greater<float>());
    for (int rank = 0; rank < k; ++rank) {
      const size_t out = static_cast<size_t>(b) * k + rank;
      const int idx = indices[out];
      if (idx < 0 || idx >= n || values[out] != row[rank] || input_row[idx] != values[out]) {
        throw std::runtime_error(
            "TensorRT TopK correctness failed at batch=" + std::to_string(b)
            + " rank=" + std::to_string(rank));
      }
    }
  }
}

void run_one_tensorrt_topk(int batch, int n, int k, int warmup, int iters, std::ostream& csv) {
  std::mt19937 rng(2026 + batch + n + k);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> h_input(static_cast<size_t>(batch) * n);
  for (float& v : h_input) v = dist(rng);

  float* d_input = nullptr;
  float* d_values = nullptr;
  int* d_indices = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_values, static_cast<size_t>(batch) * k * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_indices, static_cast<size_t>(batch) * k * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice));

  TensorRtLogger logger;
  auto builder = checked_trt_ptr(nvinfer1::createInferBuilder(logger), "builder");
  auto network = checked_trt_ptr(builder->createNetworkV2(0U), "network");
  auto config = checked_trt_ptr(builder->createBuilderConfig(), "builder config");
  config->setMaxAuxStreams(0);

  auto* input = network->addInput("input", nvinfer1::DataType::kFLOAT, nvinfer1::Dims2{batch, n});
  if (input == nullptr) {
    throw std::runtime_error("TensorRT failed to create input tensor");
  }
  auto* topk = network->addTopK(
      *input, nvinfer1::TopKOperation::kMAX, k, 1U << 1);
  if (topk == nullptr) {
    throw std::runtime_error("TensorRT failed to create TopK layer");
  }
  topk->getOutput(0)->setName("values");
  topk->getOutput(1)->setName("indices");
  network->markOutput(*topk->getOutput(0));
  network->markOutput(*topk->getOutput(1));

  auto plan = checked_trt_ptr(
      builder->buildSerializedNetwork(*network, *config), "serialized engine");
  auto runtime = checked_trt_ptr(nvinfer1::createInferRuntime(logger), "runtime");
  auto engine = checked_trt_ptr(
      runtime->deserializeCudaEngine(plan->data(), plan->size()), "engine");
  auto context = checked_trt_ptr(engine->createExecutionContext(), "execution context");
  if (!context->setTensorAddress("input", d_input)
      || !context->setTensorAddress("values", d_values)
      || !context->setTensorAddress("indices", d_indices)) {
    throw std::runtime_error("TensorRT failed to bind tensor addresses");
  }

  for (int i = 0; i < warmup; ++i) {
    if (!context->enqueueV3(nullptr)) {
      throw std::runtime_error("TensorRT TopK warmup enqueue failed");
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  start_profiler_capture();
  CUDA_CHECK(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    if (!context->enqueueV3(nullptr)) {
      throw std::runtime_error("TensorRT TopK timed enqueue failed");
    }
  }
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  stop_profiler_capture();
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  std::vector<float> h_values(static_cast<size_t>(batch) * k);
  std::vector<int> h_indices(static_cast<size_t>(batch) * k);
  CUDA_CHECK(cudaMemcpy(h_values.data(), d_values, h_values.size() * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_indices.data(), d_indices, h_indices.size() * sizeof(int), cudaMemcpyDeviceToHost));
  validate_tensorrt_result(h_input, h_values, h_indices, batch, n, k);
  {
    std::ofstream correctness(
        "reports/correctness/library_tensorrt_topk_"
        + std::to_string(batch) + "_" + std::to_string(n) + "_" + std::to_string(k) + ".log");
    correctness << "PASS_RELAXED_TIE mode=library_tensorrt_topk"
                << " batch=" << batch << " n=" << n << " k=" << k
                << " sorted_values=pass indices_reference_values=pass"
                << " caveat=tie_index_order_unspecified\n";
  }

  const double avg_ms = static_cast<double>(ms) / iters;
  const double read_gb = static_cast<double>(batch) * n * sizeof(float) / 1.0e9;
  const double write_gb = static_cast<double>(batch) * k * (sizeof(float) + sizeof(int)) / 1.0e9;
  const double bandwidth_gbs = (read_gb + write_gb) / (avg_ms / 1000.0);
  csv << "library_tensorrt_topk," << batch << "," << n << "," << k << ","
      << avg_ms << "," << bandwidth_gbs << "\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_indices));
}

}  // namespace

int main(int argc, char** argv) {
  const int batch = argc > 1 ? std::atoi(argv[1]) : 1024;
  const int n = argc > 2 ? std::atoi(argv[2]) : 4096;
  const int k = argc > 3 ? std::atoi(argv[3]) : 32;
  const int iters = argc > 4 ? std::atoi(argv[4]) : 50;
  const std::string mode_arg = argc > 5 ? argv[5] : "all";
  const int warmup = argc > 6 ? std::atoi(argv[6]) : 5;

  std::vector<TopKMode> modes;
  if (mode_arg == "all") {
    modes = {TopKMode::kNaive, TopKMode::kV1, TopKMode::kV2, TopKMode::kV3,
             TopKMode::kTensorRtTopK, TopKMode::kCubSort, TopKMode::kCubRadixSort};
  } else {
    TopKMode mode;
    if (!parse_mode(mode_arg, &mode)) {
      std::cerr << "unknown mode: " << mode_arg
                << " (expected all, naive, v1, v2, v3, library_tensorrt_topk,"
                   " library_cub_stable_segmented_sort, library_cub_segmented_radix_sort;"
                   " aliases: seme, warp, soa_warp, tensorrt_topk, cub_sort,"
                   " cub_radix_sort)\n";
      return 2;
    }
    modes = {mode};
  }

  std::filesystem::create_directories("reports/benchmark");
  std::filesystem::create_directories("reports/correctness");
  const char* csv_env = std::getenv("TOPK_BENCH_CSV");
  const std::string csv_path = csv_env != nullptr ? csv_env : "reports/benchmark/latest.csv";
  std::ofstream csv(csv_path);
  csv << "mode,batch,n,k,avg_ms,effective_gbs\n";
  for (const auto mode : modes) {
    if (mode == TopKMode::kCubSort || mode == TopKMode::kCubRadixSort) {
      run_one_cub_sort(mode, batch, n, k, warmup, iters, csv);
    } else if (mode == TopKMode::kTensorRtTopK) {
      run_one_tensorrt_topk(batch, n, k, warmup, iters, csv);
    } else {
      run_one(mode, batch, n, k, warmup, iters, csv);
    }
  }
  std::cout << "wrote " << csv_path << "\n";
  return 0;
}
