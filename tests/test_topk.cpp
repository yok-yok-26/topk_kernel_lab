#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.cuh"
#include "topk.h"

namespace {

struct CaseDef {
  std::string name;
  int batch;
  int n;
  int k;
  int pattern;
};

enum class TopKMode {
  kNaive,
  kV1,
  kV2,
  kV3,
};

const char* mode_name(TopKMode mode) {
  switch (mode) {
    case TopKMode::kNaive: return "naive";
    case TopKMode::kV1: return "v1";
    case TopKMode::kV2: return "v2";
    case TopKMode::kV3: return "v3";
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
  return false;
}


bool supports_case(TopKMode mode, const CaseDef& c) {
  if ((mode == TopKMode::kV2 || mode == TopKMode::kV3) && c.k > 32) {
    return false;
  }
  return true;
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
  }
  return cudaErrorInvalidValue;
}

void fill_case(const CaseDef& c, std::vector<float>& x) {
  std::mt19937 rng(1234 + c.batch * 17 + c.n * 3 + c.k + c.pattern * 101);
  std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
  x.assign(static_cast<size_t>(c.batch) * c.n, 0.0f);
  for (int b = 0; b < c.batch; ++b) {
    for (int i = 0; i < c.n; ++i) {
      float v = 0.0f;
      switch (c.pattern) {
        case 0: v = static_cast<float>(i % 7); break;
        case 1: v = dist(rng); break;
        case 2: v = (i % 2 == 0) ? 1.0f : -1.0f; break;
        case 3: v = (i == c.n / 2) ? 1000.0f : -static_cast<float>(i); break;
        case 4: v = static_cast<float>((i * 13 + b * 5) % 23); break;
        default: v = dist(rng); break;
      }
      x[static_cast<size_t>(b) * c.n + i] = v;
    }
  }
}

void cpu_topk(const std::vector<float>& x,
              int batch,
              int n,
              int k,
              std::vector<float>& values,
              std::vector<int>& indices) {
  values.assign(static_cast<size_t>(batch) * k, 0.0f);
  indices.assign(static_cast<size_t>(batch) * k, -1);
  for (int b = 0; b < batch; ++b) {
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    const float* row = x.data() + static_cast<size_t>(b) * n;
    std::stable_sort(order.begin(), order.end(), [&](int lhs, int rhs) {
      if (row[lhs] == row[rhs]) return lhs < rhs;
      return row[lhs] > row[rhs];
    });
    for (int j = 0; j < k; ++j) {
      indices[static_cast<size_t>(b) * k + j] = order[j];
      values[static_cast<size_t>(b) * k + j] = row[order[j]];
    }
  }
}

bool run_case(TopKMode mode, const CaseDef& c, std::ostream& log) {
  std::vector<float> h_input;
  fill_case(c, h_input);
  std::vector<float> ref_values;
  std::vector<int> ref_indices;
  cpu_topk(h_input, c.batch, c.n, c.k, ref_values, ref_indices);

  float* d_input = nullptr;
  float* d_values = nullptr;
  int* d_indices = nullptr;
  const size_t input_bytes = h_input.size() * sizeof(float);
  const size_t output_count = static_cast<size_t>(c.batch) * c.k;
  CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
  CUDA_CHECK(cudaMalloc(&d_values, output_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_indices, output_count * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));

  float* d_tmp_a_val = nullptr;
  int* d_tmp_a_idx = nullptr;
  float* d_tmp_b_val = nullptr;
  int* d_tmp_b_idx = nullptr;
  CUDA_CHECK(cudaMalloc(&d_tmp_a_val, h_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tmp_a_idx, h_input.size() * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_tmp_b_val, h_input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tmp_b_idx, h_input.size() * sizeof(int)));

  const cudaError_t launch_status = launch_topk(
      mode, d_input, d_values, d_indices, d_tmp_a_val, d_tmp_a_idx, d_tmp_b_val, d_tmp_b_idx, {c.batch, c.n, c.k});
  if (launch_status != cudaSuccess) {
    log << "FAIL mode=" << mode_name(mode) << " case=" << c.name
        << " launch_status=" << cudaGetErrorString(launch_status) << "\n";
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_values));
    CUDA_CHECK(cudaFree(d_indices));
    CUDA_CHECK(cudaFree(d_tmp_a_val));
    CUDA_CHECK(cudaFree(d_tmp_a_idx));
    CUDA_CHECK(cudaFree(d_tmp_b_val));
    CUDA_CHECK(cudaFree(d_tmp_b_idx));
    return false;
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> got_values(output_count);
  std::vector<int> got_indices(output_count);
  CUDA_CHECK(cudaMemcpy(got_values.data(), d_values, output_count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(got_indices.data(), d_indices, output_count * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_values));
  CUDA_CHECK(cudaFree(d_indices));

  int bad_count = 0;
  size_t worst_idx = 0;
  float max_abs = 0.0f;
  for (size_t i = 0; i < output_count; ++i) {
    const float abs_err = std::fabs(got_values[i] - ref_values[i]);
    const bool bad = abs_err > 0.0f || got_indices[i] != ref_indices[i];
    if (bad) {
      ++bad_count;
      if (abs_err >= max_abs) {
        max_abs = abs_err;
        worst_idx = i;
      }
    }
  }

  log << (bad_count == 0 ? "PASS " : "FAIL ") << "mode=" << mode_name(mode)
      << " case=" << c.name << " batch=" << c.batch << " n=" << c.n
      << " k=" << c.k << " bad_count=" << bad_count << " max_abs=" << max_abs
      << " worst_idx=" << worst_idx;
  if (bad_count != 0) {
    log << " got_value=" << got_values[worst_idx]
        << " ref_value=" << ref_values[worst_idx]
        << " got_index=" << got_indices[worst_idx]
        << " ref_index=" << ref_indices[worst_idx];
  }
  log << "\n";
  return bad_count == 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::system("mkdir -p reports/correctness");
  std::ofstream log("reports/correctness/latest.log");
  if (!log) {
    std::cerr << "failed to open reports/correctness/latest.log\n";
    return 1;
  }

  std::vector<TopKMode> modes;
  if (argc <= 1 || std::string(argv[1]) == "all") {
    modes = {TopKMode::kNaive, TopKMode::kV1, TopKMode::kV2, TopKMode::kV3};
  } else {
    TopKMode mode;
    if (!parse_mode(argv[1], &mode)) {
      std::cerr << "unknown mode: " << argv[1] << " (expected all, naive, v1, v2, v3; aliases: seme, warp, soa_warp)\n";
      return 2;
    }
    modes = {mode};
  }

  const std::string case_filter = argc > 2 ? argv[2] : "all";

  const std::vector<CaseDef> cases = {
      {"single", 1, 1, 1, 0},
      {"tiny_ties", 2, 7, 3, 0},
      {"block_minus_1", 3, 255, 8, 1},
      {"block", 3, 256, 8, 1},
      {"block_plus_1", 3, 257, 8, 1},
      {"non_divisible", 5, 1000, 16, 1},
      {"alternating", 4, 513, 32, 2},
      {"sparse_impulse", 4, 1024, 8, 3},
      {"many_ties", 6, 2049, 64, 4},
      {"max_k", 2, 4096, topk_lab::kMaxK, 1},
      {"large_segment_1m", 8, 1048576, 32, 1},
      {"fixed_b64_n262144_k32", 64, 262144, 32, 1},
      {"fixed_b64_n262144_k8", 64, 262144, 8, 4},
      {"irregular_b64_n100003_k32", 64, 100003, 32, 1},
      {"irregular_b64_n65521_k8", 64, 65521, 8, 4},
      {"fixed_b64_n262144_k100", 64, 262144, 100, 1},
      {"fixed_b64_n65536_k1", 64, 65536, 1, 1},
  };

  bool all_ok = true;
  log << "TopK correctness contract: float32 [B,N], largest K along N, tie by lower index.\n";
  for (const auto mode : modes) {
    log << "MODE " << mode_name(mode) << "\n";
    for (const auto& c : cases) {
      if (case_filter != "all" && case_filter != c.name) {
        continue;
      }
      if (!supports_case(mode, c)) {
        log << "SKIP mode=" << mode_name(mode) << " case=" << c.name
            << " reason=unsupported_k_gt_32 k=" << c.k << "\n";
        log.flush();
        std::cout << "SKIP mode=" << mode_name(mode) << " case=" << c.name
                  << " reason=unsupported_k_gt_32 k=" << c.k << "\n";
        continue;
      }
      log << "RUN mode=" << mode_name(mode) << " case=" << c.name << "\n";
      log.flush();
      std::cout << "RUN mode=" << mode_name(mode) << " case=" << c.name << "\n";
      all_ok = run_case(mode, c, log) && all_ok;
      log.flush();
    }
  }
  log << (all_ok ? "SUMMARY PASS\n" : "SUMMARY FAIL\n");
  std::cout << (all_ok ? "PASS" : "FAIL") << " reports/correctness/latest.log\n";
  return all_ok ? 0 : 1;
}
