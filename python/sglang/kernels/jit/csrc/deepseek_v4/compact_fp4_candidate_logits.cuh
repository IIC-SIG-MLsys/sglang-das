// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: Apache-2.0
// Adapted from LightOp's bf16_fp4_candidate_kernel. Keep the MMAC and FP32
// accumulation order, writing by candidate column instead of logical column.

#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>

namespace sglang {

#if defined(USE_ROCM)
namespace details {

constexpr int kCandidateBlock = 8;
constexpr int kThreads = 256;
constexpr int kHeads = 32;
constexpr int kHeadDim = 128;
constexpr int kTokensPerCTA = 256;

using CandidateBf16x4 = short __attribute__((ext_vector_type(4)));
using CandidateFloatx4 = float __attribute__((ext_vector_type(4)));

SGL_DEVICE uint16_t decode_candidate_fp4(unsigned code, unsigned scale) {
  const unsigned mag = code & 7u;
  const unsigned sign = (code & 8u) << 12;
  const unsigned base = mag >= 2 ? 0x3f00u + (mag << 6) : (mag == 1 ? 0x3f00u : 0u);
  const int exponent = static_cast<int>(base >> 7) + static_cast<int>(scale) - 127;
  if (scale < 255 && (mag == 0 || (exponent > 0 && exponent < 255))) {
    return sign | (mag == 0 ? 0u : (static_cast<unsigned>(exponent) << 7) | (base & 0x7fu));
  }
  const float factor = scale == 255 ? INFINITY : ldexpf(1.0f, static_cast<int>(scale) - 127);
  const uint32_t bits = __float_as_uint(__uint_as_float((sign | base) << 16) * factor);
  if ((bits & 0x7fffffffu) > 0x7f800000u) return (bits >> 16) | 0x40u;
  return (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
}

SGL_DEVICE CandidateBf16x4 decode_candidate_fp4x4(unsigned packed, unsigned scale) {
  CandidateBf16x4 result;
  if (scale >= 2 && scale <= 252) {
    const int adjustment = (static_cast<int>(scale) - 127) * 128;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned code = (packed >> (4 * j)) & 15u;
      const unsigned mag = code & 7u;
      const unsigned sign = (code & 8u) << 12;
      const int base = mag >= 2 ? 0x3f00 + (mag << 6) : 0x3f00;
      result[j] = sign | (mag == 0 ? 0 : base + adjustment);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      result[j] = decode_candidate_fp4((packed >> (4 * j)) & 15u, scale);
    }
  }
  return result;
}

__global__ __launch_bounds__(kThreads, 1) void compact_fp4_candidate_logits_kernel(
    float* __restrict__ out,
    const uint16_t* __restrict__ q,
    const uint8_t* __restrict__ k,
    const int32_t* __restrict__ scales,
    const float* __restrict__ weights,
    const int32_t* __restrict__ starts,
    const int32_t* __restrict__ ends,
    const int32_t* __restrict__ blocks,
    int64_t block_row_stride,
    int64_t num_keys,
    int32_t width) {
#if defined(__gfx936__)
  constexpr int kWaveSize = 64;
  constexpr int kTile = 16;
  constexpr int kTilesPerWave = 4;
  const int64_t query = blockIdx.y;
  const int lane = device::get_lane_id<kWaveSize>();
  const int key_in_tile = lane % kTile;
  const int group = lane / kTile;
  const int tile_start = blockIdx.x * kTokensPerCTA + threadIdx.x / kWaveSize * kWaveSize;
  if (tile_start >= width) return;

  const auto* row_blocks = blocks + query * block_row_stride;
  auto* row_scores = out + query * width;
  const int64_t start = starts[query];
  const int64_t end = ends[query];
  float row_weights[kHeads];
#pragma unroll
  for (int h = 0; h < kHeads; ++h)
    row_weights[h] = weights[query * kHeads + h];
  CandidateBf16x4 q_fragments[2][8];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int fragment = 0; fragment < 8; ++fragment) {
      const auto* src = q + (query * kHeads + h * kTile + key_in_tile) * kHeadDim + (fragment / 2) * 32 + group * 8 +
                        (fragment % 2) * 4;
      q_fragments[h][fragment] = {
          static_cast<short>(src[0]),
          static_cast<short>(src[1]),
          static_cast<short>(src[2]),
          static_cast<short>(src[3])};
    }
  }

#pragma unroll
  for (int tile = 0; tile < kTilesPerWave; ++tile) {
    const int column = tile_start + tile * kTile + key_in_tile;
    const int32_t block = column < width ? row_blocks[column / kCandidateBlock] : -1;
    const int64_t key = start + static_cast<int64_t>(block) * kCandidateBlock + column % kCandidateBlock;
    const bool live = block >= 0 && key >= 0 && key < end && key < num_keys;
    float score = -INFINITY;
    if (__ballot(live)) {
      const unsigned scale = live ? static_cast<unsigned>(scales[key]) : 0x7f7f7f7fu;
      CandidateFloatx4 dots[2] = {};
      // Preserve the order of the eight K16 fragments and both head tiles.
#pragma unroll
      for (int fragment = 0; fragment < 8; ++fragment) {
        const auto* src = live ? k + key * 64 + (fragment / 2) * 16 + group * 4 + (fragment % 2) * 2 : k;
        const auto packed = live ? *reinterpret_cast<const uint16_t*>(src) : 0;
        const unsigned group_scale = (scale >> ((fragment / 2) * 8)) & 255u;
        const auto k_fragment = decode_candidate_fp4x4(packed, group_scale);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          dots[h] = __builtin_amdgcn_mmac_f32_16x16x16bf16(k_fragment, q_fragments[h][fragment], dots[h]);
        }
      }
      float sum = 0.0f;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
          for (int g = 0; g < 4; ++g) {
            const float dot = __shfl(dots[h][j], key_in_tile + g * kTile, kWaveSize);
            float relu;
            // Match the original scorer's ReLU, including nonfinite inputs.
            asm volatile("v_max_f32_e32 %0, 0, %1" : "=v"(relu) : "v"(dot));
            sum = fmaf(relu, row_weights[h * kTile + j * 4 + g], sum);
          }
        }
      }
      if (live) score = sum;
    }
    if (lane < kTile && column < width) row_scores[column] = score;
  }
#else
  __builtin_trap();
#endif
}

}  // namespace details
#endif

/// Score request-local candidate blocks into contiguous FP32 candidate columns.
struct CompactFp4CandidateLogitsKernel {
  static void
  run(tvm::ffi::TensorView out,
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView k,
      tvm::ffi::TensorView scales,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView starts,
      tvm::ffi::TensorView ends,
      tvm::ffi::TensorView blocks) {
#if defined(USE_ROCM)
    using namespace host;
    using namespace sglang::details;
    const auto block_count = blocks.size(1);
    CHECK_HOST(block_count > 0 && block_count <= 4096) << "invalid candidate table width";
    const auto width = static_cast<int32_t>(block_count) * kCandidateBlock;
    auto rows = SymbolicSize{"queries"};
    auto keys = SymbolicSize{"keys"};
    auto block_stride = SymbolicSize{"block row stride"};
    auto device = SymbolicDevice{};
    TensorMatcher({rows, kHeads, kHeadDim}).with_dtype<bf16_t>().with_device<kDLGPU>(device).verify(q);
    TensorMatcher({keys, kHeadDim / 2}).with_dtype<int8_t>().with_device<kDLGPU>(device).ensure_alignment(2).verify(k);
    TensorMatcher({keys}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(scales);
    TensorMatcher({rows, kHeads}).with_dtype<float>().with_device<kDLGPU>(device).verify(weights);
    TensorMatcher({rows}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(starts).verify(ends);
    TensorMatcher({rows, block_count})
        .with_dtype<int32_t>()
        .with_device<kDLGPU>(device)
        .with_strides({block_stride, 1})
        .verify(blocks);
    TensorMatcher({rows, width}).with_dtype<float>().with_device<kDLGPU>(device).verify(out);
    CHECK_HOST(rows.unwrap() <= 65535) << "candidate rows exceed grid.y limit";
    if (rows.unwrap() == 0) return;
    LaunchKernel(
        {static_cast<uint32_t>(div_ceil(width, kTokensPerCTA)), static_cast<uint32_t>(rows.unwrap())},
        kThreads,
        device.unwrap())(
        compact_fp4_candidate_logits_kernel,
        static_cast<float*>(out.data_ptr()),
        static_cast<const uint16_t*>(q.data_ptr()),
        static_cast<const uint8_t*>(k.data_ptr()),
        static_cast<const int32_t*>(scales.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<const int32_t*>(starts.data_ptr()),
        static_cast<const int32_t*>(ends.data_ptr()),
        static_cast<const int32_t*>(blocks.data_ptr()),
        blocks.stride(0),
        keys.unwrap(),
        width);
#else
    host::Panic("compact FP4 candidate logits require gfx936");
#endif
  }
};

}  // namespace sglang
