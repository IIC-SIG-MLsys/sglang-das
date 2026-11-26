// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace dsv41_lt {

struct LtHandle {
  hipblasLtHandle_t value = nullptr;
  LtHandle() = default;
  LtHandle(const LtHandle&) = delete;
  LtHandle& operator=(const LtHandle&) = delete;
  ~LtHandle() {
    if (value) hipblasLtDestroy(value);
  }
};

struct MatmulDesc {
  hipblasLtMatmulDesc_t value = nullptr;
  MatmulDesc() = default;
  MatmulDesc(const MatmulDesc&) = delete;
  MatmulDesc& operator=(const MatmulDesc&) = delete;
  ~MatmulDesc() {
    if (value) hipblasLtMatmulDescDestroy(value);
  }
};

struct MatrixLayout {
  hipblasLtMatrixLayout_t value = nullptr;
  MatrixLayout() = default;
  MatrixLayout(const MatrixLayout&) = delete;
  MatrixLayout& operator=(const MatrixLayout&) = delete;
  ~MatrixLayout() {
    if (value) hipblasLtMatrixLayoutDestroy(value);
  }
};

struct MatmulPreference {
  hipblasLtMatmulPreference_t value = nullptr;
  MatmulPreference() = default;
  MatmulPreference(const MatmulPreference&) = delete;
  MatmulPreference& operator=(const MatmulPreference&) = delete;
  ~MatmulPreference() {
    if (value) hipblasLtMatmulPreferenceDestroy(value);
  }
};

inline void check(hipblasStatus_t status, const char* op) {
  if (status != HIPBLAS_STATUS_SUCCESS) {
    throw std::runtime_error(
        std::string("DSV41 INT8 hipBLASLt ") + op + " failed, status=" + std::to_string(static_cast<int>(status)));
  }
}

// Keep the request count fixed: ranks in configs are indices into this query,
// whose ordering can depend on both the requested count and workspace limit.
constexpr int kHeuristicCount = 64;

class UnpackedGemm {
 public:
  UnpackedGemm(
      hipblasLtHandle_t handle,
      int64_t m,
      int64_t n,
      int64_t cols,
      int64_t k,
      int64_t groups,
      bool group_major,
      uint64_t workspace_limit)
      : workspace_limit_(workspace_limit) {
    check(hipblasLtMatmulDescCreate(&desc_.value, HIPBLAS_COMPUTE_32I, HIP_R_32I), "descriptor");
    hipblasOperation_t ta = HIPBLAS_OP_T;
    hipblasOperation_t tb = HIPBLAS_OP_N;
    check(hipblasLtMatmulDescSetAttribute(desc_.value, HIPBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)), "transA");
    check(hipblasLtMatmulDescSetAttribute(desc_.value, HIPBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)), "transB");

    // The column-major view computes the transpose of the row-major output:
    // W[N,K] @ X.T[K,M].
    const int64_t ldx = group_major ? k : groups * k;
    const int64_t ldy = group_major ? n : groups * n;
    check(hipblasLtMatrixLayoutCreate(&a_.value, HIP_R_8I, k, cols, k), "A layout");
    check(hipblasLtMatrixLayoutCreate(&b_.value, HIP_R_8I, k, m, ldx), "B layout");
    check(hipblasLtMatrixLayoutCreate(&c_.value, HIP_R_32I, cols, m, ldy), "C layout");
    check(hipblasLtMatrixLayoutCreate(&d_.value, HIP_R_32I, cols, m, ldy), "D layout");
    if (groups > 1) {
      int32_t count = static_cast<int32_t>(groups);
      int64_t strides[] = {n * k, group_major ? m * k : k, group_major ? m * n : n, group_major ? m * n : n};
      hipblasLtMatrixLayout_t layouts[] = {a_.value, b_.value, c_.value, d_.value};
      for (int i = 0; i < 4; ++i) {
        check(
            hipblasLtMatrixLayoutSetAttribute(layouts[i], HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &count, sizeof(count)),
            "batch count");
        check(
            hipblasLtMatrixLayoutSetAttribute(
                layouts[i], HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &strides[i], sizeof(strides[i])),
            "batch stride");
      }
    }

    MatmulPreference pref;
    check(hipblasLtMatmulPreferenceCreate(&pref.value), "preference");
    check(
        hipblasLtMatmulPreferenceSetAttribute(
            pref.value, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit, sizeof(workspace_limit)),
        "workspace preference");
    check(
        hipblasLtMatmulAlgoGetHeuristic(
            handle,
            desc_.value,
            a_.value,
            b_.value,
            c_.value,
            d_.value,
            pref.value,
            kHeuristicCount,
            results_.data(),
            &count_),
        "heuristic");
  }

  int count() const {
    return count_;
  }

  const hipblasLtMatmulAlgo_t& algorithm(int rank) const {
    return results_[rank].algo;
  }

  int64_t workspace_bytes(int rank) const {
    if (rank < 0 || rank >= count_ || results_[rank].state != HIPBLAS_STATUS_SUCCESS ||
        results_[rank].workspaceSize > workspace_limit_) {
      return -1;
    }
    return results_[rank].workspaceSize;
  }

  hipblasStatus_t
  run(hipblasLtHandle_t handle,
      int rank,
      const void* q,
      const void* w,
      void* out,
      void* workspace,
      std::size_t workspace_bytes,
      hipStream_t stream) const {
    const int32_t alpha = 1;
    const int32_t beta = 0;
    return hipblasLtMatmul(
        handle,
        desc_.value,
        &alpha,
        w,
        a_.value,
        q,
        b_.value,
        &beta,
        out,
        c_.value,
        out,
        d_.value,
        &results_[rank].algo,
        workspace,
        workspace_bytes,
        stream);
  }

 private:
  MatmulDesc desc_;
  MatrixLayout a_, b_, c_, d_;
  std::array<hipblasLtMatmulHeuristicResult_t, kHeuristicCount> results_{};
  int count_ = 0;
  uint64_t workspace_limit_ = 0;
};

}  // namespace dsv41_lt
