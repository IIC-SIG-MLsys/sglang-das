// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// MMAC fragment order, even-wrap Q loads and vector FP4 decode adapted from
// HYGON-AI/lightop, csrc/mqa_logits/fp8_fp4/.

#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/utils.cuh>

#include <cmath>
#include <cstdint>
#include <type_traits>

namespace sglang {

#if defined(USE_ROCM)
namespace details::dsv41_fp4_mqa {

constexpr int kWave = 64;
constexpr int kThreads = 4 * kWave;
constexpr int kHeads = 32;
constexpr int kDim = 128;
constexpr int kBlock = 8;
constexpr int kQueries = 32;
constexpr int kKeys = 128;
constexpr int kMmacRows = 16;
constexpr int kCompactWaveKeys = 2 * kMmacRows;
constexpr int kCompactCtaKeys = (kThreads / kWave) * kCompactWaveKeys;

#if defined(__gfx936__)
using Bf16x4 = short __attribute__((ext_vector_type(4)));
using U32x2 = uint32_t __attribute__((ext_vector_type(2)));
using U32x4 = uint32_t __attribute__((ext_vector_type(4)));
using F32x2 = float __attribute__((ext_vector_type(2)));
using F32x4 = float __attribute__((ext_vector_type(4)));

SGL_DEVICE uint16_t decode_fp4(unsigned code, unsigned scale) {
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

// Each lane decodes one or more words of a UE8M0 group. For ordinary scales the
// BF16 exponent adjustment is exact, and v_perm looks up four entries per byte plane.
template <int Words>
SGL_DEVICE void decode_group(U32x4 packed, unsigned scale, U32x4 (&decoded)[Words]) {
  if (scale >= 2 && scale <= 252) {
    const uint32_t delta = static_cast<uint32_t>((static_cast<int>(scale) - 127) * 128);
    const uint32_t pair = delta * 0x10001u;
    const uint32_t p0 = 0x3f000000u + (delta << 16);
    const uint32_t p1 = 0x3fc03f80u + pair;
    const uint32_t p2 = 0x40404000u + pair;
    const uint32_t p3 = 0x40c04080u + pair;
    const uint32_t lo0 = __builtin_amdgcn_perm(p1, p0, 0x06040200u);
    const uint32_t lo1 = __builtin_amdgcn_perm(p3, p2, 0x06040200u);
    const uint32_t hi0 = __builtin_amdgcn_perm(p1, p0, 0x07050301u);
    const uint32_t hi1 = __builtin_amdgcn_perm(p3, p2, 0x07050301u);
#pragma unroll
    for (int i = 0; i < Words; ++i) {
      const uint32_t w = packed[i];
      // Put the nibbles in logical order before the lookups. Each lookup
      // now produces four consecutive bytes of the BF16 low or high plane.
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const uint32_t codes = __builtin_amdgcn_perm(w >> 4, w, half == 0 ? 0x05010400u : 0x07030602u);
        const uint32_t mag = codes & 0x07070707u;
        const uint32_t lo = __builtin_amdgcn_perm(lo1, lo0, mag);
        const uint32_t hi = __builtin_amdgcn_perm(hi1, hi0, mag) | ((codes << 4) & 0x80808080u);
        decoded[i][2 * half] = __builtin_amdgcn_perm(hi, lo, 0x05010400u);
        decoded[i][2 * half + 1] = __builtin_amdgcn_perm(hi, lo, 0x07030602u);
      }
    }
    return;
  }
#pragma unroll
  for (int i = 0; i < Words; ++i) {
    U32x4 values;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned byte = (packed[i] >> (j * 8)) & 255u;
      values[j] = decode_fp4(byte & 15u, scale) | (static_cast<uint32_t>(decode_fp4(byte >> 4, scale)) << 16);
    }
    decoded[i] = values;
  }
}

SGL_DEVICE Bf16x4 mmac_half(U32x4 fragment, int half) {
  const U32x2 pair = {fragment[half * 2], fragment[half * 2 + 1]};
  return __builtin_bit_cast(Bf16x4, pair);
}

SGL_DEVICE F32x4 mmac(Bf16x4 a, Bf16x4 b, F32x4 c) {
  return __builtin_amdgcn_mmac_f32_16x16x16bf16(a, b, c);
}

SGL_DEVICE float relu(float x) {
  float out;
  asm volatile("v_max_f32_e32 %0, 0, %1" : "=v"(out) : "v"(x));
  return out;
}

SGL_DEVICE float read_wave_float(float x, int lane) {
  return __int_as_float(__builtin_amdgcn_ds_bpermute(lane * 4, __float_as_int(x)));
}

SGL_DEVICE U32x4 buffer_resource(const void* ptr) {
  const auto address = reinterpret_cast<uint64_t>(ptr);
  return {
      static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address))),
      static_cast<uint32_t>(__builtin_amdgcn_readfirstlane(static_cast<uint32_t>(address >> 32))),
      0x80000000u,
      0x00020000u};
}

// One slot is 16 bytes, or eight BF16 values. Each Q panel holds 32 rows
// and each K panel holds one half-tile of 64 rows. The two uses share LDS.
union alignas(1024) Shared {
  U32x4 k[2][64 * 8];
  U32x4 q[2][2][kQueries * 8];
};
static_assert(sizeof(Shared) == 16 * 1024);

SGL_DEVICE int k_slot(int row, int slot, int panel) {
  const int wrap = row & 3;
  const int row_slot = (row & 31) >> 2;
  const int rotated = ((slot & 3) << 1) | (slot >> 2);
  return ((row >> 5) * 4 + wrap) * 64 + row_slot * 8 + ((wrap << 1) ^ (row_slot & 1) ^ rotated ^ (panel << 2));
}

template <bool Tail>
SGL_DEVICE void stage_keys(
    U32x4 (&lds)[2][64 * 8],
    const uint8_t* packed,
    const int32_t* scales,
    int tile,
    int num_keys,
    U32x4 (&fragments)[2][4]) {
  const int thread = threadIdx.x;
  const int row = thread / 4;
  const int group = thread % 4;
  const int lane = thread % kWave;
  const int wave = thread / kWave;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int key = tile + half * 64 + row;
    const bool live = !Tail || key < num_keys;
    const U32x4 payload = live ? *reinterpret_cast<const U32x4*>(packed + key * (kDim / 2) + group * 16) : U32x4{};
    const auto scale = live ? static_cast<uint32_t>(scales[key]) : 0x7f7f7f7fu;
    U32x4 decoded[4];
    decode_group(payload, (scale >> (group * 8)) & 255u, decoded);
    const int panel = group / 2;
#pragma unroll
    for (int slot = 0; slot < 4; ++slot) {
      lds[panel][k_slot(row, (group % 2) * 4 + slot, panel)] = decoded[slot];
    }
    __syncthreads();
#pragma unroll
    for (int step = 0; step < 4; ++step) {
      const int read_row = wave * 16 + lane % 16;
      const int slot = (step % 2) * 4 + lane / 16;
      fragments[half][step] = lds[step / 2][k_slot(read_row, slot, step / 2)];
    }
    __syncthreads();
  }
}

struct QueryHeadLoader {
  U32x4 q_resource;
  U32x4 w_resource;
  int q_offset;
  int w_offset;
  int lds_addr[2][2];

  SGL_DEVICE void init(Shared& lds, const uint16_t* q, const float* weights, int valid_rows, int lane, int wave) {
    q_resource = buffer_resource(q);
    w_resource = buffer_resource(weights);
    const int row = wave + (lane / 8) * 4;
    q_offset = row < valid_rows ? row * kHeads * kDim * 2 + (lane % 8) * 16 : -1;
    w_offset = lane < valid_rows ? lane * kHeads * 4 : -1;
#pragma unroll
    for (int stage = 0; stage < 2; ++stage) {
#pragma unroll
      for (int panel = 0; panel < 2; ++panel) {
        // m0 encodes the LDS address and gfx936's even-wrap rotation.
        lds_addr[stage][panel] = __builtin_amdgcn_readfirstlane(
            static_cast<int>(reinterpret_cast<uintptr_t>(lds.q[stage][panel])) + wave * 1024 + (wave << 17));
      }
    }
  }

  SGL_DEVICE void issue(int head, int stage, float& weight) const {
#pragma unroll
    for (int panel = 0; panel < 2; ++panel) {
      const int offset = head * kDim * 2 + panel * 128;
      asm volatile(
          "s_mov_b32 m0, %1\n\t"
          "buffer_load_dwordx4 %0, %2, %3 offen offset:0 lds"
          :
          : "v"(q_offset), "s"(lds_addr[stage][panel]), "s"(q_resource), "s"(offset)
          : "memory");
    }
    const int offset = head * 4;
    asm volatile("buffer_load_dword %0, %1, %2, %3 offen"
                 : "=&v"(weight)
                 : "v"(w_offset), "s"(w_resource), "s"(offset)
                 : "memory");
  }
};

SGL_DEVICE void wait_for_head(float& weight) {
  __builtin_amdgcn_sched_barrier(0);
  asm volatile("s_waitcnt vmcnt(0)" : "+v"(weight) : : "memory");
  __builtin_amdgcn_sched_barrier(0);
  __syncthreads();
}

SGL_DEVICE U32x4 q_fragment(const U32x4 (&panel)[2][kQueries * 8], int row, int quad, int step) {
  const int wrap = row % 4;
  const int slot = ((row / 4) * 8 + (step % 2) * 4 + quad + wrap * 2) % 64;
  return panel[step / 2][wrap * 64 + slot];
}

SGL_DEVICE void accumulate_head(F32x4 (&sum)[2][2], const F32x4 (&dot)[2][2], const float (&weights)[2]) {
#pragma unroll
  for (int m = 0; m < 2; ++m) {
#if __has_builtin(__builtin_elementwise_fma)
    const F32x2 weight = {weights[m], weights[m]};
#endif
#pragma unroll
    for (int n = 0; n < 2; ++n) {
      F32x4 activated;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        activated[j] = relu(dot[m][n][j]);
      }
#if __has_builtin(__builtin_elementwise_fma)
      sum[m][n].xy = __builtin_elementwise_fma(activated.xy, weight, sum[m][n].xy);
      sum[m][n].zw = __builtin_elementwise_fma(activated.zw, weight, sum[m][n].zw);
#else
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        sum[m][n][j] = fmaf(activated[j], weights[m], sum[m][n][j]);
      }
#endif
    }
  }
}

SGL_DEVICE uint32_t ordered_score(float x) {
  const uint32_t bits = __float_as_uint(x);
  return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
}

SGL_DEVICE float score_from_ordered(uint32_t x) {
  return __uint_as_float(x & 0x80000000u ? x ^ 0x80000000u : ~x);
}

SGL_DEVICE void store_block_max(float* dst, F32x4 scores, int first_key, int length, int lane) {
  // MMAC C has columns quad + 4*j. Two register pairs and two quad
  // exchanges reduce each eight-key block. No lane transposition or LDS.
  const int quad = lane / 16;
#pragma unroll
  for (int block = 0; block < 2; ++block) {
    const int base = first_key + block * kBlock;
    uint32_t value =
        max(ordered_score(base + quad < length ? scores[block * 2] : -INFINITY),
            ordered_score(base + quad + 4 < length ? scores[block * 2 + 1] : -INFINITY));
    value = max(value, static_cast<uint32_t>(__builtin_amdgcn_ds_bpermute((lane ^ 16) * 4, value)));
    value = max(value, static_cast<uint32_t>(__builtin_amdgcn_ds_bpermute((lane ^ 32) * 4, value)));
    if (quad == 0 && base < length) dst[base / kBlock] = score_from_ordered(value);
  }
}
#endif  // __gfx936__

template <bool KeyTail>
__global__ __launch_bounds__(kThreads, 1) void block_max_kernel(
    float* __restrict__ out,
    const uint16_t* __restrict__ q,
    const uint8_t* __restrict__ k,
    const int32_t* __restrict__ scales,
    const float* __restrict__ weights,
    const int32_t* __restrict__ lengths,
    int rows,
    int keys,
    int q_shards) {
#if defined(__gfx936__)
  __shared__ Shared lds;
  const int k_tiles = (keys + kKeys - 1) / kKeys;
  const int tile = (blockIdx.x % k_tiles) * kKeys;
  const int shard = blockIdx.x / k_tiles;
  const int lane = device::get_lane_id<kWave>();
  const int wave = threadIdx.x / kWave;
  const int row_in_tile = lane % 16;
  const int quad = lane / 16;
  const int64_t block_stride = (keys + kBlock - 1) / kBlock;
  U32x4 k_fragments[2][4];
  bool k_ready = false;
  for (int first_row = shard * kQueries; first_row < rows; first_row += q_shards * kQueries) {
    const int valid_rows = min(kQueries, rows - first_row);
    const int probe = lane % kQueries;
    // All waves vote on the same rows before any CTA barrier.
    if (!__ballot(probe < valid_rows && lengths[first_row + probe] > tile)) continue;
    if (!k_ready) {
      stage_keys<KeyTail>(lds.k, k, scales, tile, keys, k_fragments);
      k_ready = true;
    }

    QueryHeadLoader loader;
    loader.init(
        lds, q + static_cast<int64_t>(first_row) * kHeads * kDim, weights + first_row * kHeads, valid_rows, lane, wave);
    float weight;
    loader.issue(0, 0, weight);
    wait_for_head(weight);
    F32x4 sum[2][2] = {};
    F32x4 previous[2][2] = {};
    float previous_weights[2] = {};

    auto score_head = [&](auto prefetch_tag, int head, int stage) {
      constexpr bool prefetch = decltype(prefetch_tag)::value;
      const float head_weights[2] = {read_wave_float(weight, row_in_tile), read_wave_float(weight, row_in_tile + 16)};
      U32x4 a[2] = {
          q_fragment(lds.q[stage], row_in_tile, quad, 0), q_fragment(lds.q[stage], row_in_tile + 16, quad, 0)};
      __builtin_amdgcn_sched_barrier(0);
      float next_weight;
      if constexpr (prefetch) loader.issue(head + 1, stage ^ 1, next_weight);
      F32x4 dot[2][2] = {};
#pragma unroll
      for (int step = 0; step < 4; ++step) {
        U32x4 next[2];
        if (step < 3) {
          next[0] = q_fragment(lds.q[stage], row_in_tile, quad, step + 1);
          next[1] = q_fragment(lds.q[stage], row_in_tile + 16, quad, step + 1);
        }
        // Keep four independent MMAC chains and LightOp's K32 half order.
#pragma unroll
        for (int half = 0; half < 2; ++half) {
#pragma unroll
          for (int n = 0; n < 2; ++n) {
            const auto b = mmac_half(k_fragments[n][step], half);
#pragma unroll
            for (int m = 0; m < 2; ++m) {
              dot[m][n] = mmac(mmac_half(a[m], half), b, dot[m][n]);
            }
          }
        }
        if (step == 0) accumulate_head(sum, previous, previous_weights);
        if (step < 3) {
          a[0] = next[0];
          a[1] = next[1];
        }
      }
#pragma unroll
      for (int m = 0; m < 2; ++m) {
        previous_weights[m] = head_weights[m];
#pragma unroll
        for (int n = 0; n < 2; ++n)
          previous[m][n] = dot[m][n];
      }
      if constexpr (prefetch) {
        wait_for_head(next_weight);
        weight = next_weight;
      }
    };

    for (int head = 0; head < kHeads - 2; head += 2) {
      score_head(std::true_type{}, head, 0);
      score_head(std::true_type{}, head + 1, 1);
    }
    score_head(std::true_type{}, kHeads - 2, 0);
    score_head(std::false_type{}, kHeads - 1, 1);
    accumulate_head(sum, previous, previous_weights);

#pragma unroll
    for (int m = 0; m < 2; ++m) {
      const int row = row_in_tile + m * 16;
      if (row < valid_rows) {
        const int length = lengths[first_row + row];
        auto* dst = out + (first_row + row) * block_stride;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
          store_block_max(dst, sum[m][n], tile + wave * 16 + n * 64, length, lane);
        }
      }
    }
    __syncthreads();
  }
#else
  __builtin_trap();
#endif
}

#if defined(__gfx936__)
struct CompactQuery {
  U32x4 panels[kHeads][kDim / 8];
  F32x4 weights[kHeads / 4];
};

SGL_DEVICE int compact_q_slot(int head, int slot) {
  // Rotate the LDS bank group by the low bits of the MMAC column.
  return slot ^ (head / 8 + (head & 1) * 4);
}

SGL_DEVICE void stage_compact_query(CompactQuery& lds, const uint16_t* q, const float* weights) {
  const int thread = threadIdx.x;
  const int head = thread / 8;
  const auto* src = reinterpret_cast<const U32x2*>(q) + head * (kDim / 4);
#pragma unroll
  for (int panel = 0; panel < 2; ++panel) {
    const int slot = panel * 8 + thread % 8;
    const U32x2 lo = src[2 * slot];
    const U32x2 hi = src[2 * slot + 1];
    lds.panels[head][compact_q_slot(head, slot)] = {lo[0], lo[1], hi[0], hi[1]};
  }
  if (thread < kHeads / 4) {
    const auto* w = weights + thread * 4;
    lds.weights[thread] = {w[0], w[1], w[2], w[3]};
  }
  __syncthreads();
}

struct CompactKeyFragments {
  U32x4 packed;
  uint32_t scales;
  bool live;
};

SGL_DEVICE CompactKeyFragments load_compact_key(
    const uint8_t* k, const int32_t* scales, const int32_t* blocks, int column, int width, int length, int quad) {
  const int block = column < width ? blocks[column / kBlock] : -1;
  const int64_t key = static_cast<int64_t>(block) * kBlock + column % kBlock;
  if (key < 0 || key >= length) return {U32x4{}, 0x7f7f7f7fu, false};
  const auto* words = reinterpret_cast<const uint32_t*>(k + static_cast<int>(key) * (kDim / 2)) + quad;
  // Each FP4 dword holds the codes for both MMAC halves of one K32 group.
  return {{words[0], words[4], words[8], words[12]}, static_cast<uint32_t>(scales[key]), true};
}

SGL_DEVICE F32x2 compact_weighted_sum(const F32x4 (&dot)[2][2], const F32x4 (&weights)[2], int key_lane) {
  F32x2 activated[2][4];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      activated[h][j] = {relu(dot[0][h][j]), relu(dot[1][h][j])};
    }
  }

  // Each quad owns eight consecutive heads. Carry the FP32 sum to the
  // next quad before its FMA chain so the head order stays exactly 0..31.
  F32x2 sum = {};
#pragma unroll
  for (int quad = 0; quad < 4; ++quad) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float w = weights[h][j];
#if __has_builtin(__builtin_elementwise_fma)
        sum = __builtin_elementwise_fma(activated[h][j], F32x2{w, w}, sum);
#else
        sum = {fmaf(activated[h][j].x, w, sum.x), fmaf(activated[h][j].y, w, sum.y)};
#endif
      }
    }
    if (quad < 3) {
      const int source_lane = key_lane + quad * kMmacRows;
      sum = {read_wave_float(sum.x, source_lane), read_wave_float(sum.y, source_lane)};
    }
  }
  return sum;
}
#endif  // __gfx936__

__global__ __launch_bounds__(kThreads, 2) void compact_scores_kernel(
    float* __restrict__ out,
    const uint16_t* __restrict__ q,
    const uint8_t* __restrict__ k,
    const int32_t* __restrict__ scales,
    const float* __restrict__ weights,
    const int32_t* __restrict__ lengths,
    const int32_t* __restrict__ blocks,
    int block_stride,
    int width) {
#if defined(__gfx936__)
  __shared__ CompactQuery lds;
  const int query = blockIdx.y;
  const int length = lengths[query];
  if (length <= 0) return;
  const int lane = device::get_lane_id<kWave>();
  const int key_lane = lane % kMmacRows;
  const int quad = lane / kMmacRows;
  const int wave = __builtin_amdgcn_readfirstlane(static_cast<int>(threadIdx.x / kWave));
  const auto* row_blocks = blocks + static_cast<int64_t>(query) * block_stride;
  auto* row_out = out + static_cast<int64_t>(query) * width;

  stage_compact_query(lds, q + query * kHeads * kDim, weights + query * kHeads);
  U32x4 q_fragments[2][4];
  F32x4 row_weights[2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    // MMAC C column quad + 4*j maps to head 8*quad + 4*h + j.
    const int head = (key_lane % 4) * 8 + h * 4 + key_lane / 4;
    row_weights[h] = lds.weights[quad * 2 + h];
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      q_fragments[h][group] = lds.panels[head][compact_q_slot(head, group * 4 + quad)];
    }
  }

  const int start = blockIdx.x * kCompactCtaKeys + wave * kCompactWaveKeys;
  const int stride = gridDim.x * kCompactCtaKeys;
  if (start >= width) return;
  auto gather = [&](int column) { return load_compact_key(k, scales, row_blocks, column, width, length, quad); };
  CompactKeyFragments next[2] = {gather(start + key_lane), gather(start + key_lane + kMmacRows)};
  for (int first = start; first < width; first += stride) {
    const int column = first + key_lane;
    const CompactKeyFragments keys[2] = {next[0], next[1]};
    if (first + stride < width) {
      next[0] = gather(column + stride);
      next[1] = gather(column + stride + kMmacRows);
    }
    if (!__ballot(keys[0].live || keys[1].live)) continue;
    // Issue the next FP4 gathers before decoding and scoring this pair.
    __builtin_amdgcn_sched_barrier(0);
    F32x4 dot[2][2] = {};
#pragma unroll
    for (int group = 0; group < 4; ++group) {
      U32x4 decoded[2];
#pragma unroll
      for (int tile = 0; tile < 2; ++tile) {
        U32x4 fragment[1];
        const U32x4 word = {keys[tile].packed[group], 0u, 0u, 0u};
        decode_group(word, (keys[tile].scales >> (group * 8)) & 255u, fragment);
        decoded[tile] = fragment[0];
      }
      // Keep LightOp's transposed operands and its K32 half order.
#pragma unroll
      for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int tile = 0; tile < 2; ++tile) {
          const auto a = mmac_half(decoded[tile], half);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            dot[tile][h] = mmac(a, mmac_half(q_fragments[h][group], half), dot[tile][h]);
          }
        }
      }
    }
    const F32x2 sum = compact_weighted_sum(dot, row_weights, key_lane);
    if (quad == 3) {
      if (keys[0].live) row_out[column] = sum.x;
      if (keys[1].live) row_out[column + kMmacRows] = sum.y;
    }
  }
#else
  __builtin_trap();
#endif
}

struct Inputs {
  host::SymbolicSize rows{"queries"};
  host::SymbolicSize keys{"keys"};
  host::SymbolicDevice device{};

  Inputs(
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView k,
      tvm::ffi::TensorView scales,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView lengths) {
    using host::TensorMatcher;
    TensorMatcher({rows, kHeads, kDim}).with_dtype<bf16_t>().with_device<kDLGPU>(device).ensure_alignment(8).verify(q);
    TensorMatcher({keys, kDim / 2}).with_dtype<int8_t>().with_device<kDLGPU>(device).ensure_alignment(16).verify(k);
    TensorMatcher({keys}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(scales);
    TensorMatcher({rows, kHeads}).with_dtype<float>().with_device<kDLGPU>(device).verify(weights);
    TensorMatcher({rows}).with_dtype<int32_t>().with_device<kDLGPU>(device).verify(lengths);
    CHECK_HOST(rows.unwrap() <= 65535 && keys.unwrap() < (1LL << 31) / (kDim / 2)) << "FP4 scorer grid too large";
  }
};

}  // namespace details::dsv41_fp4_mqa
#endif  // USE_ROCM

struct Dsv41FusedFp4MqaKernel {
  /// Emit FP32 maxima of the visible, request-local groups of eight keys.
  static void block_max(
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView k,
      tvm::ffi::TensorView scales,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView lengths,
      int q_shards) {
#if defined(USE_ROCM)
    using namespace details::dsv41_fp4_mqa;
    auto in = Inputs(q, k, scales, weights, lengths);
    const int rows = static_cast<int>(in.rows.unwrap());
    const int keys = static_cast<int>(in.keys.unwrap());
    host::TensorMatcher({rows, host::div_ceil(keys, kBlock)})
        .with_dtype<float>()
        .with_device<kDLGPU>(in.device)
        .verify(out);
    if (!rows || !keys) return;
    CHECK_HOST(q_shards > 0 && q_shards <= host::div_ceil(rows, kQueries)) << "invalid FP4 scorer shard count";
    const auto kernel = keys % kKeys ? block_max_kernel<true> : block_max_kernel<false>;
    host::LaunchKernel(host::div_ceil(keys, kKeys) * q_shards, kThreads, in.device.unwrap())(
        kernel,
        static_cast<float*>(out.data_ptr()),
        static_cast<const uint16_t*>(q.data_ptr()),
        static_cast<const uint8_t*>(k.data_ptr()),
        static_cast<const int32_t*>(scales.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<const int32_t*>(lengths.data_ptr()),
        rows,
        keys,
        q_shards);
#else
    host::Panic("DSV4.1 fused FP4 MQA requires gfx936");
#endif
  }

  /// Recompute scores in ascending logical block order for token Top-512.
  static void compact_scores(
      tvm::ffi::TensorView out,
      tvm::ffi::TensorView q,
      tvm::ffi::TensorView k,
      tvm::ffi::TensorView scales,
      tvm::ffi::TensorView weights,
      tvm::ffi::TensorView lengths,
      tvm::ffi::TensorView blocks) {
#if defined(USE_ROCM)
    using namespace details::dsv41_fp4_mqa;
    auto in = Inputs(q, k, scales, weights, lengths);
    auto block_stride = host::SymbolicSize{"block stride"};
    const int rows = static_cast<int>(in.rows.unwrap());
    const int width = static_cast<int>(out.size(1));
    CHECK_HOST(width > 0 && width % kBlock == 0 && width <= 512 * kBlock) << "invalid compact FP4 score width";
    host::TensorMatcher({rows, width / kBlock})
        .with_dtype<int32_t>()
        .with_device<kDLGPU>(in.device)
        .with_strides({block_stride, 1})
        .verify(blocks);
    host::TensorMatcher({rows, width}).with_dtype<float>().with_device<kDLGPU>(in.device).verify(out);
    if (!rows) return;
    // A second CTA exposes more waves on small batches with wide shortlists.
    const uint32_t shards = rows < 256 && width > kCompactCtaKeys ? 2 : 1;
    host::LaunchKernel({shards, static_cast<uint32_t>(rows)}, kThreads, in.device.unwrap())(
        compact_scores_kernel,
        static_cast<float*>(out.data_ptr()),
        static_cast<const uint16_t*>(q.data_ptr()),
        static_cast<const uint8_t*>(k.data_ptr()),
        static_cast<const int32_t*>(scales.data_ptr()),
        static_cast<const float*>(weights.data_ptr()),
        static_cast<const int32_t*>(lengths.data_ptr()),
        static_cast<const int32_t*>(blocks.data_ptr()),
        static_cast<int>(blocks.stride(0)),
        width);
#else
    host::Panic("DSV4.1 fused FP4 MQA requires gfx936");
#endif
  }
};

}  // namespace sglang
