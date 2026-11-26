// SPDX-License-Identifier: Apache-2.0
// Host-only shim for the offline tuner. Build against the serving DTK so
// Python need not guess the ABI of opaque hipBLASLt structs.
#include <cstdio>
#include <memory>

#include "int8_gemm_hipblaslt.h"

namespace {
thread_local std::array<char, 512> last_error{};

struct Probe {
  dsv41_lt::LtHandle handle;
  std::unique_ptr<dsv41_lt::UnpackedGemm> gemm;
};
}  // namespace

extern "C" const char* bw1000_error() {
  return last_error.data();
}

extern "C" Probe* bw1000_create(int64_t m, int64_t n, int64_t k, int64_t groups, uint64_t workspace_limit) {
  try {
    auto probe = std::make_unique<Probe>();
    dsv41_lt::check(hipblasLtCreate(&probe->handle.value), "create");
    probe->gemm =
        std::make_unique<dsv41_lt::UnpackedGemm>(probe->handle.value, m, n, n, k, groups, false, workspace_limit);
    return probe.release();
  } catch (const std::exception& e) {
    std::snprintf(last_error.data(), last_error.size(), "%s", e.what());
    return nullptr;
  }
}

extern "C" void bw1000_destroy(Probe* probe) {
  delete probe;
}

extern "C" int bw1000_count(const Probe* probe) {
  return probe->gemm->count();
}

extern "C" int64_t bw1000_workspace_bytes(const Probe* probe, int rank) {
  return probe->gemm->workspace_bytes(rank);
}

extern "C" const void* bw1000_algo_data(const Probe* probe, int rank, std::size_t* size) {
  const auto& algo = probe->gemm->algorithm(rank);
  *size = sizeof(algo.data);
  return algo.data;
}

extern "C" int bw1000_run(
    const Probe* probe,
    int rank,
    const void* q,
    const void* w,
    void* out,
    void* workspace,
    std::size_t workspace_bytes,
    hipStream_t stream) {
  return probe->gemm->run(probe->handle.value, rank, q, w, out, workspace, workspace_bytes, stream);
}
