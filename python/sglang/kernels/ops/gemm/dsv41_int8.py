"""Single DSV4.1 INT8 projection entry point for the AOT hipBLASLt backend.

The existing HIP and private Triton computations are unchanged. Native operators
are provided by the matching sglang-kernel common_ops build, not a JIT extension.
"""
import json
import os
import threading
import weakref
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.quantization.dsv41_int8 import (
    _quantize_int8_rows,
    _supports_wo_a_inverse_rope,
    quantize_wo_a_inverse_rope,
)

_OPERATORS = (
    "dsv41_int8_quantize_out",
    "dsv41_int8_gemm_out",
    "dsv41_int8_scale_out",
    "dsv41_int8_prepare",
    "dsv41_int8_gemm_packed",
    "dsv41_int8_scale_packed",
    "dsv41_int8_scale_scheduled",
)

class _NativeOps:
    def __init__(self, ops):
        self._ops = ops

    def quantize_out(self, x, quantized, scales):
        return self._ops.dsv41_int8_quantize_out(x, quantized, scales)


    def gemm_out(self, q, weight, out, groups, tile_n, group_major, algo_rank=0):
        return self._ops.dsv41_int8_gemm_out(
            q, weight, out, groups, tile_n, group_major, algo_rank
        )


    def scale_out(self, raw, x_scale, weight_scale, out, groups, group_major):
        return self._ops.dsv41_int8_scale_out(
            raw, x_scale, weight_scale, out, groups, group_major
        )


    def prepare(self, packed, m, groups, rank, k, batches, lda, stride, pitch, sharded, algo):
        return self._ops.dsv41_int8_prepare(
            packed, m, groups, rank, k, batches, lda, stride, pitch, sharded, algo
        )


    def gemm(self, q, packed, raw, plan):
        return self._ops.dsv41_int8_gemm_packed(q, packed, raw, plan)


    def scale(self, raw, x_scale, weight_scale, out, groups, batches, sharded, mode):
        return self._ops.dsv41_int8_scale_packed(
            raw, x_scale, weight_scale, out, groups, batches, sharded, mode
        )


    def scale_scheduled(self, raw, x_scale, weight_scale, out, mode, tile, threads):
        return self._ops.dsv41_int8_scale_scheduled(
            raw, x_scale, weight_scale, out, mode, tile, threads
        )


@lru_cache(maxsize=1)
def backend():
    try:
        # Loads the installed common_ops registration.
        import sgl_kernel  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Rebuild the matching HCU sglang-kernel wheel with DSV4.1 INT8 support.") from exc
    ops = torch.ops.sgl_kernel
    missing = [name for name in _OPERATORS if not hasattr(ops, name)]
    if missing:
        raise RuntimeError("Rebuild the matching HCU sglang-kernel wheel; missing INT8 operators: " + ", ".join(missing))
    return _NativeOps(ops)

load_native = backend
_load = backend
_lock = threading.RLock()
_packs = {}
_plans = {}
_scale_seen = set()
_algo_rank_cache: OrderedDict = OrderedDict()
_failed_static_algo_keys: set = set()
_ALGO_RANK_CACHE_MAX_ENTRIES = 4096
_TRITON_SCALE_MIN_ELEMENTS = 4_194_304
_device_arch_cache: dict = {}

# Offline-benchmarked hipBLASLt heuristic algo_rank table for Hygon BW1000
# (gfx936, DTK 26.04.1) on unpacked DeepSeek-V4.1 INT8 projections.
# Key: (groups, rank, k, tile_n, group_major) -> (exact_m_map, bucket_m_map).
# Uncovered shapes or non-gfx936 devices fall back directly to algo_rank=0
# without online trial-and-error probing.
_DEFAULT_GFX936_UNPACKED_ALGO_RANKS = {
    # wo_a (target TP8/DP8): 2.2x-9.8x vs algo_rank=0
    (8, 1024, 4096, 0, False): (
        {
            1: 27, 2: 26, 4: 26, 5: 26, 6: 27, 8: 27, 10: 24, 12: 24, 16: 24,
            18: 13, 20: 29, 24: 17, 30: 17, 32: 18, 36: 18, 40: 18, 42: 18,
            48: 20, 64: 20,
        },
        {128: 1, 256: 5, 512: 6, 1024: 4, 2048: 2, 4096: 1, 8192: 1},
    ),
    # wo_a (draft): 1.9x-24.7x vs algo_rank=0
    (8, 512, 4096, 0, False): (
        {
            1: 4, 2: 4, 4: 3, 6: 4, 8: 4, 12: 2, 16: 3, 24: 31, 32: 11,
            48: 31, 64: 30,
        },
        {128: 6, 256: 15, 512: 23, 1024: 17, 2048: 9, 4096: 8},
    ),
    # lm_head (full vocab, DP lm_head): 2.2x-7.0x vs algo_rank=0
    (1, 129280, 5120, 0, False): (
        {
            1: 12, 2: 12, 4: 15, 5: 15, 6: 15, 8: 15, 10: 15, 12: 8, 16: 15,
            18: 6, 20: 12, 24: 12, 30: 12, 32: 12, 36: 3, 40: 5, 42: 3,
            48: 3, 64: 5,
        },
        {128: 14, 256: 2},
    ),
    # lm_head (TP8 sharded): 2.5x-8.0x vs algo_rank=0
    (1, 16160, 5120, 0, False): (
        {
            1: 15, 2: 15, 4: 12, 5: 15, 6: 15, 8: 15, 12: 15, 16: 15,
            24: 12, 32: 1, 48: 2, 64: 19,
        },
        {128: 3, 256: 3, 512: 6, 1024: 3, 2048: 3, 4096: 3},
    ),
    # DSPARK markov_head (TP8 sharded): 1.4x-3.0x vs algo_rank=0
    (1, 16160, 256, 0, False): (
        {
            1: 16, 2: 8, 4: 11, 5: 15, 6: 4, 8: 10, 12: 20, 16: 15,
            24: 11, 32: 15, 48: 11, 64: 3,
        },
        {128: 19, 256: 14},
    ),
    # DSPARK markov_head (full vocab): 1.5x-3.4x vs algo_rank=0
    (1, 129280, 256, 0, False): (
        {
            1: 18, 2: 18, 4: 16, 5: 16, 6: 1, 8: 16, 12: 1, 16: 1,
            24: 12, 32: 12, 48: 4, 64: 3,
        },
        {128: 4, 256: 19},
    ),
}

_DEFAULT_SERVICE_CONFIG = {'algorithms': {'wo_a:1:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:5:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:128:4160:8:8192:1024:0': '79220000000000000000000000000000',
                'wo_a:256:4160:8:8192:1024:0': 'd4220000000000000000000000000000',
                'wo_a:8:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:16:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:32:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:64:4160:8:8192:1024:0': '79220000000000000000000000000000',
                'wo_a:96:4160:8:8192:1024:0': '79220000000000000000000000000000',
                'lm_head:1:5120:101:1280:6619136:0': 'e7220000000000000000000000000000',
                'lm_head:5:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:8:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:16:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:32:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:64:5136:101:1280:6619136:1': 'cf220000000000000000000000000000',
                'lm_head:96:5248:101:1280:6619136:0': 'd4220000000000000000000000000000',
                'lm_head:128:5152:101:1280:6619136:0': 'cf220000000000000000000000000000',
                'lm_head:256:5376:101:1280:6619136:0': 'df220000000000000000000000000000',
                'markov_tp8:1:256:1:16160:4136960:0': '8b220000000000000000000000000000',
                'markov_tp8:5:256:1:16160:4136960:0': '8d220000000000000000000000000000',
                'markov_tp8:32:256:1:16160:4136960:0': '0d230000000000000000000000000000',
                'markov_tp8:128:256:1:16160:4136960:0': 'cf220000000000000000000000000000',
                'markov_tp8:256:256:4:4040:1034240:0': '83220000000000000000000000000000',
                'markov_full:1:256:101:1280:327680:0': '87220000000000000000000000000000',
                'markov_full:5:256:101:1280:327680:0': '87220000000000000000000000000000',
                'markov_full:32:256:101:1280:327680:0': 'e6220000000000000000000000000000',
                'markov_full:128:256:101:1280:327680:0': 'cf220000000000000000000000000000',
                'markov_full:256:256:101:1280:327680:0': 'cf220000000000000000000000000000',
                'markov_tp8:256:320:4:4040:1034240:0': 'e0220000000000000000000000000000',
                'wo_a:6:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:10:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:12:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:15:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:18:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:20:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:24:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:25:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:30:4096:8:8192:1024:0': '9d220000000000000000000000000000',
                'wo_a:35:4096:8:8192:1024:0': '79220000000000000000000000000000',
                'wo_a:36:4096:8:8192:1024:0': '79220000000000000000000000000000',
                'wo_a:40:4096:8:8192:1024:0': '79220000000000000000000000000000',
                'wo_a:42:4096:8:8192:1024:0': '79220000000000000000000000000000',
                'wo_a:48:4096:8:8192:1024:0': '79220000000000000000000000000000',
                'lm_head:6:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:10:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:12:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:15:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:18:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:20:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:24:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:25:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:30:5120:101:1280:6619136:1': 'e7220000000000000000000000000000',
                'lm_head:35:5120:101:1280:6619136:1': 'de220000000000000000000000000000',
                'lm_head:36:5120:101:1280:6619136:1': 'de220000000000000000000000000000',
                'lm_head:40:5120:101:1280:6619136:1': 'de220000000000000000000000000000',
                'lm_head:42:5120:101:1280:6619136:1': 'de220000000000000000000000000000',
                'lm_head:48:5120:101:1280:6619136:1': 'de220000000000000000000000000000',
                'markov_full:2:256:101:1280:327680:0': '88220000000000000000000000000000',
                'markov_full:3:256:101:1280:327680:0': '88220000000000000000000000000000',
                'markov_full:4:256:101:1280:327680:0': '87220000000000000000000000000000',
                'markov_full:6:256:101:1280:327680:0': '87220000000000000000000000000000',
                'markov_full:7:256:101:1280:327680:0': '87220000000000000000000000000000',
                'markov_full:8:256:101:1280:327680:0': '87220000000000000000000000000000',
                'wo_a:1:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:5:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:6:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:8:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:10:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:12:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:15:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:16:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:18:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:20:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:24:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:25:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:30:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:32:4096:8:1024:4325376:0': '9d220000000000000000000000000000',
                'wo_a:35:4096:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:36:4096:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:40:4096:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:42:4096:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:48:4096:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:64:4160:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:96:4160:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:128:4160:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:256:4160:8:1024:4325376:0': 'd4220000000000000000000000000000',
                'markov_full:1:256:1010:128:32768:0': '88220000000000000000000000000000',
                'markov_full:2:256:1010:128:32768:0': '87220000000000000000000000000000',
                'markov_full:3:256:1010:128:32768:0': '87220000000000000000000000000000',
                'markov_full:4:256:1010:128:32768:0': '87220000000000000000000000000000',
                'markov_full:5:256:1010:128:32768:0': '87220000000000000000000000000000',
                'markov_full:6:256:1010:128:32768:0': '87220000000000000000000000000000',
                'markov_full:7:256:1010:128:32768:0': '87220000000000000000000000000000',
                'markov_full:8:256:1010:128:32768:0': '87220000000000000000000000000000',
                'wo_a:36:4224:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:40:4224:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:42:4224:8:1024:4325376:0': '79220000000000000000000000000000',
                'wo_a:48:4224:8:1024:4325376:0': '79220000000000000000000000000000'},
 'scale_schedules': {'lm_head': {'10': [1, 1, 320],
                                 '18': [1, 1, 320],
                                 '24': [4, 2, 128],
                                 '48': [4, 2, 128]}}}

@lru_cache(maxsize=1)
def _service_config():
    path = os.environ.get("SGLANG_DSV41_INT8_SERVICE_ALGO_CONFIG")
    return json.loads(Path(path).read_text()) if path else _DEFAULT_SERVICE_CONFIG

@lru_cache(maxsize=1)
def _algorithms():
    return _service_config()["algorithms"]

@lru_cache(maxsize=1)
def _scale_schedules():
    return _service_config().get("scale_schedules", {}).get("lm_head", {})

@triton.jit
def _scale_int32(ACC, XS, WS, Y, T:tl.constexpr, G:tl.constexpr,
                 R:tl.constexpr, GROUP_MAJOR:tl.constexpr, B:tl.constexpr):
    j=tl.program_id(0)*B+tl.arange(0,B)
    valid=j<T*G*R
    row=j//R
    group=row%G
    token=row//G
    col=j%R
    if GROUP_MAJOR:
        raw=(group*T+token)*R+col
    else:
        raw=j
    acc=tl.load(ACC+raw,valid,other=0).to(tl.float32)
    xs=tl.load(XS+row,valid,other=0)
    ws=tl.load(WS+group*R+col,valid,other=0)
    tl.store(Y+j,acc*xs*ws,valid)


@triton.jit
def _scale_int32_dynamic(
    ACC,
    XS,
    WS,
    Y,
    T,
    TOTAL,
    G: tl.constexpr,
    R: tl.constexpr,
    GROUP_MAJOR: tl.constexpr,
    B: tl.constexpr,
):
    j = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B).to(tl.int64)
    valid = j < TOTAL
    row = j // R
    group = row % G
    token = row // G
    col = j % R
    if GROUP_MAJOR:
        raw = (group * T + token) * R + col
    else:
        raw = j
    acc = tl.load(ACC + raw, valid, other=0).to(tl.float32)
    xs = tl.load(XS + row, valid, other=0)
    ws = tl.load(WS + group * R + col, valid, other=0)
    tl.store(Y + j, acc * xs * ws, valid)


@triton.jit
def _scale_grid(ACC,XS,WS,Y,M:tl.constexpr,N:tl.constexpr,WIDTH:tl.constexpr,BLOCK:tl.constexpr):
 local=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);row=tl.program_id(1);group=tl.program_id(2);col=group*WIDTH+local;valid=local<WIDTH
 a=tl.load(ACC+(group*M+row)*WIDTH+local,valid,0).to(tl.float32);xs=tl.load(XS+row);ws=tl.load(WS+col,valid,0);tl.store(Y+row*N+col,(a*xs)*ws,valid)


# Measured on Hygon BW1000 (gfx936): below ~25M elements eager Triton launch
# overhead makes Triton slower than native C++ quantize_out (0.32x-0.97x);
# at >= 32M elements (e.g. prefill wo_a M >= 1024), Triton is 1.26x-1.42x faster.
_TRITON_QUANT_MIN_ELEMENTS = 33_554_432
_TRITON_QUANT_MAX_K = 8192
_TRITON_QUANT_WARMED = set()


def _triton_quantize_enabled() -> bool:
    raw = os.environ.get("SGLANG_DSV41_INT8_TRITON_QUANT")
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return torch.version.hip is not None


@triton.jit(do_not_specialize=("M",))
def _dsv41_int8_quantize_triton_kernel(
    X_ptr,
    Q_ptr,
    S_ptr,
    M,
    K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_K)
    row_mask = rows < M
    col_mask = cols < K
    mask = row_mask[:, None] & col_mask[None, :]
    offs = rows[:, None].to(tl.int64) * K + cols[None, :].to(tl.int64)

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    q, scale = _quantize_int8_rows(x, 1)

    tl.store(Q_ptr + offs, q, mask=mask)
    tl.store(S_ptr + rows, scale, mask=row_mask)


def _select_triton_quant_block_r(k: int) -> int:
    if k <= 512:
        return 8
    if k <= 1024:
        return 4
    return 1


def _triton_quant_compile_spec(x: torch.Tensor):
    """Exact Triton compilation signature for _dsv41_int8_quantize_triton_kernel.

    ``M`` is explicitly excluded from Triton specialization via
    ``do_not_specialize=("M",)`` so changing row counts (e.g. 1024 -> 1025)
    reuses the compiled binary without triggering JIT compilation during HIP
    Graph capture. All remaining specialization-affecting attributes (device,
    dtype, constexpr tile dimensions, and 16-byte input pointer alignment) are
    included in the warmup key.
    """
    k = x.shape[-1]
    block_r = _select_triton_quant_block_r(k)
    block_k = triton.next_power_of_2(k)
    x_aligned16 = (x.data_ptr() % 16) == 0
    return (x.device.index, x.dtype, k, block_r, block_k, x_aligned16)


def _can_use_triton_quantize(x: torch.Tensor) -> bool:
    if not (
        _triton_quantize_enabled()
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim >= 2
        and x.numel() >= _TRITON_QUANT_MIN_ELEMENTS
        and 1 <= x.shape[-1] <= _TRITON_QUANT_MAX_K
    ):
        return False
    spec = _triton_quant_compile_spec(x)
    if torch.cuda.is_current_stream_capturing() and spec not in _TRITON_QUANT_WARMED:
        return False
    return True


def _quantize_triton(x: torch.Tensor, q: torch.Tensor, scales: torch.Tensor) -> None:
    k = x.shape[-1]
    m = x.numel() // k
    block_r = _select_triton_quant_block_r(k)
    block_k = triton.next_power_of_2(k)
    spec = _triton_quant_compile_spec(x)
    grid = (triton.cdiv(m, block_r),)
    _dsv41_int8_quantize_triton_kernel[grid](
        x,
        q,
        scales,
        m,
        K=k,
        BLOCK_R=block_r,
        BLOCK_K=block_k,
        num_warps=4,
    )
    if not torch.cuda.is_current_stream_capturing():
        _TRITON_QUANT_WARMED.add(spec)


def quantize(x):
    if x.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("HIP INT8 quantization requires BF16/FP16/FP32 activations")
    x = x.contiguous()
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1] + (1,), dtype=torch.float32, device=x.device)
    if x.numel() == 0:
        return q, scales
    if _can_use_triton_quantize(x):
        _quantize_triton(x, q, scales)
        return q, scales
    ops = load_native()
    ops.quantize_out(x, q, scales)
    return q, scales


@lru_cache(maxsize=4)
def _algorithm_config(path):
    with open(path) as f:
        return json.load(f)["algorithms"]


def _blaslt_static_algo_enabled() -> bool:
    raw = os.environ.get("SGLANG_DSV41_INT8_BLASLT_STATIC_ALGO")
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return torch.version.hip is not None


def _blaslt_autotune_enabled() -> bool:
    # Online trial-and-error probing is opt-in (for offline tuning / explicit
    # warmup) so serving requests never pay cold-search spikes or trigger
    # failing Plan construction on older native wheels.
    raw = os.environ.get("SGLANG_DSV41_INT8_BLASLT_AUTOTUNE")
    if raw is not None:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return False


def _is_gfx936_device(device) -> bool:
    if device is None or getattr(device, "type", None) != "cuda":
        return False
    idx = device.index
    cached = _device_arch_cache.get(idx)
    if cached is not None:
        return cached
    try:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    except Exception:
        arch = ""
    is_match = arch == "gfx936"
    _device_arch_cache[idx] = is_match
    return is_match


def _lookup_default_unpacked_algo_rank(
    device, t: int, groups: int, rank: int, k: int, tile_n: int, group_major: bool
):
    if not _blaslt_static_algo_enabled() or not _is_gfx936_device(device):
        return None
    cache_key = (
        getattr(device, "index", None),
        _autotune_t_bucket(t),
        groups,
        rank,
        k,
        tile_n,
        bool(group_major),
    )
    if cache_key in _failed_static_algo_keys:
        return None
    spec = _DEFAULT_GFX936_UNPACKED_ALGO_RANKS.get(
        (groups, rank, k, tile_n, bool(group_major))
    )
    if spec is None:
        return None
    exact_map, bucket_map = spec
    if t <= 64:
        return exact_map.get(t)
    t_bucket = _autotune_t_bucket(t)
    return bucket_map.get(t_bucket)


def _autotune_t_bucket(t: int) -> int:
    # Keep exact M for small decode/graph-captured batches (<= 64) and bucket
    # larger variable-length prefill chunks by next power of two so arbitrary
    # chunk lengths reuse a handful of cached hipBLASLt heuristic ranks.
    return t if t <= 64 else 1 << (t - 1).bit_length()


def _select_unpacked_algo_rank(ops, q, w, raw, groups, rank, k, tile_n, group_major):
    t = q.shape[1] if group_major else q.shape[0]
    t_bucket = _autotune_t_bucket(t)
    cache_key = (q.device.index, t_bucket, groups, rank, k, tile_n, bool(group_major))
    cached = _algo_rank_cache.get(cache_key)
    if cached is not None:
        return cached
    if not q.is_cuda or torch.cuda.is_current_stream_capturing():
        return 0
    with _lock:
        cached = _algo_rank_cache.get(cache_key)
        if cached is not None:
            return cached
        if torch.cuda.is_current_stream_capturing():
            return 0
        max_ranks = max(
            1, int(os.environ.get("SGLANG_DSV41_INT8_BLASLT_AUTOTUNE_MAX_RANKS", "32"))
        )
        best_rank = 0
        best_ms = float("inf")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        iters = 2
        for cand in range(max_ranks):
            try:
                ops.gemm_out(q, w, raw, groups, tile_n, group_major, cand)
            except RuntimeError:
                if cand == 0:
                    raise
                break
            if cand == 0:
                ops.gemm_out(q, w, raw, groups, tile_n, group_major, cand)
                start.record()
                ops.gemm_out(q, w, raw, groups, tile_n, group_major, cand)
                end.record()
                end.synchronize()
                probe_ms = start.elapsed_time(end)
                if probe_ms < 0.3:
                    iters = 10
                elif probe_ms < 2.0:
                    iters = 5
                else:
                    iters = 2
            else:
                ops.gemm_out(q, w, raw, groups, tile_n, group_major, cand)
            start.record()
            for _ in range(iters):
                ops.gemm_out(q, w, raw, groups, tile_n, group_major, cand)
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end) / iters
            if elapsed < best_ms:
                best_ms = elapsed
                best_rank = cand
        _algo_rank_cache[cache_key] = best_rank
        if len(_algo_rank_cache) > _ALGO_RANK_CACHE_MAX_ENTRIES:
            _algo_rank_cache.popitem(last=False)
        return best_rank


def _project_unpacked(
    x,
    w,
    s,
    groups,
    rank,
    *,
    output_dtype=None,
    tile_n=None,
    group_major=False,
    algo_rank=None,
    input_quant_args=None,
):
    if (
        x.ndim != 3 or x.shape[1] != groups or groups <= 0 or rank <= 0
        or w.shape != (groups * rank, x.shape[-1])
        or w.dtype != torch.int8 or not w.is_contiguous()
        or s.shape != (groups * rank, 1)
        or s.dtype != torch.float32 or not s.is_contiguous()
        or x.device != w.device or x.device != s.device
    ):
        raise ValueError("Invalid channelwise INT8 projection tensors")
    output_dtype = output_dtype or x.dtype
    if output_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("HIP projection output requires BF16/FP16/FP32")
    if tile_n is None:
        tile_n = int(os.environ.get("SGLANG_DSV41_INT8_BLASLT_TILE_N", "0")) if groups == 1 else 0
    if tile_n < 0:
        raise ValueError("INT8 BLASLt tile size must be non-negative")
    ops = load_native()
    t = x.shape[0]
    k = x.shape[-1]
    use_autotune = False
    used_static = False
    if algo_rank is None:
        config = os.environ.get("SGLANG_DSV41_INT8_BLASLT_ALGO_CONFIG")
        if config:
            key = f"{t}:{groups}:{rank}:{k}:{tile_n}:{int(group_major)}"
            algo_rank = int(_algorithm_config(config).get(key, 0))
        elif _blaslt_autotune_enabled():
            use_autotune = True
        else:
            static_rank = _lookup_default_unpacked_algo_rank(
                x.device, t, groups, rank, k, tile_n, group_major
            )
            if static_rank is not None:
                algo_rank = static_rank
                used_static = True
            else:
                algo_rank = 0
    y = torch.empty((t, groups, rank), device=x.device, dtype=output_dtype)
    if not t:
        return y
    q, xs = quantize(x) if input_quant_args is None else input_quant_args
    if group_major:
        q = q.transpose(0, 1).contiguous()
    raw = torch.empty((groups, t, rank) if group_major else (t, groups, rank),
                      device=x.device, dtype=torch.int32)
    if use_autotune:
        algo_rank = _select_unpacked_algo_rank(
            ops, q, w, raw, groups, rank, k, tile_n, group_major
        )
    try:
        ops.gemm_out(q, w, raw, groups, tile_n, group_major, algo_rank)
    except RuntimeError:
        if (use_autotune or used_static) and algo_rank != 0:
            fail_key = (
                getattr(x.device, "index", None),
                _autotune_t_bucket(t),
                groups,
                rank,
                k,
                tile_n,
                bool(group_major),
            )
            with _lock:
                if used_static:
                    _failed_static_algo_keys.add(fail_key)
                if use_autotune:
                    _algo_rank_cache[fail_key] = 0
            ops.gemm_out(q, w, raw, groups, tile_n, group_major, 0)
        else:
            raise
    numel = t * groups * rank
    if y.is_cuda and torch.version.hip is not None and numel >= _TRITON_SCALE_MIN_ELEMENTS:
        sb = 2048
        _scale_int32_dynamic[(triton.cdiv(numel, sb),)](
            raw, xs, s, y, t, numel, groups, rank, bool(group_major), sb,
            num_warps=8, enable_fp_fusion=False,
        )
    else:
        ops.scale_out(raw, xs, s, y, groups, group_major)
    return y


def _scheduled_scale(raw,xs,ws,out,m,schedule):
 mode,tile,threads=map(int,schedule);key=(raw.device.index,m,mode,tile,threads)
 if key not in _scale_seen:
  with _lock:
   if key not in _scale_seen:
    print('BLASLT_OPT_SCALE',json.dumps(dict(role='lm_head',m=m,mode=mode,tile=tile,threads=threads,output_dtype=str(out.dtype),device=raw.device.index)),flush=True);_scale_seen.add(key)
 _load().scale_scheduled(raw,xs,ws,out,mode,tile,threads)


def _version(w):
 try:return w._version
 except RuntimeError:return -1


def _role(g,n,k):
 if (g,n,k)==(8,1024,4096):return 'wo_a'
 if (g,n,k)==(1,129280,5120):return 'lm_head'
 if g==1 and k==256 and n in (16160,129280):return 'markov_tp8' if n==16160 else 'markov_full'
 return None


def _layout(role,m,g,n,k):
 if role=='wo_a':return g,n,k*n+131072,4224 if m in (36,40,42,48) else (k+64 if m>=64 else k),False
 if role=='lm_head':
  pitch=5120 if m<64 else (5136 if m==64 else (5248 if m<=96 else (5152 if m<=128 else 5376)))
  return 101,1280,6619136,pitch,1<m<=64
 batches=(1010 if m<=8 else 101) if role=='markov_full' else (4 if m>128 else 1)
 return batches,n//batches,k*(n//batches),320 if role=='markov_tp8' and m==256 else k,False


def _pack(w,g,n,k,batches,lda,astride):
 key=(w.device.index,w.data_ptr(),tuple(w.shape),_version(w),g,n,k,batches,lda,astride)
 with _lock:
  found=_packs.get(key)
  if found is not None:return found['packed']
  if torch.cuda.is_current_stream_capturing():raise RuntimeError('INT8 packed weights must be prepared before graph capture')
  width=g*n//batches;span=(batches-1)*astride+(k-1)*lda+width
  storage=torch.zeros(span,device=w.device,dtype=torch.int8)
  if g>1:storage.as_strided((k,g,n),(lda,astride,1)).copy_(w.reshape(g,n,k).permute(2,0,1))
  else:storage.as_strided((batches,k,width),(astride,lda,1)).copy_(w.reshape(batches,width,k).transpose(1,2))
  packed=storage.view(1,1,-1);_packs[key]=dict(owner=w,packed=packed)
  print('BLASLT_OPT_PACK',json.dumps(dict(groups=g,rank=n,k=k,batches=batches,lda=lda,stride=astride,bytes=span)),flush=True)
  return packed


def _prepare_packed_weight(w):
    if not _use_packed_service_projection() or not w.is_cuda:
        return
    if w.dtype != torch.int8 or not w.is_contiguous() or w.ndim != 2:
        return
    n0, k = w.shape
    g = 8 if (n0, k) == (8192, 4096) else 1
    n = n0 // g
    role = _role(g, n, k)
    if role is None:
        return
    for m in [1, 256] if role in ("markov_tp8", "markov_full") else [1]:
        batches, lda, astride, _, _ = _layout(role, m, g, n, k)
        _pack(w, g, n, k, batches, lda, astride)


def release_weight_cache(weight):
 """Drop only this full weight's packed copies before compact VP releases it.

 The native plans own descriptors, not tensors. Other projection weights and
 their packed copies remain resident. Call only before graph capture.
 """
 if weight.is_cuda and torch.cuda.is_current_stream_capturing():
  raise RuntimeError('Cannot release INT8 weight cache during graph capture')
 with _lock:
  keys=[key for key,value in _packs.items() if value['owner'] is weight]
  for key in keys:del _packs[key]
 return len(keys)


def _project_packed(x,w,s,groups,rank,*,output_dtype=None,tile_n=None,group_major=False,algo_rank=None):
 from sglang.kernels.ops.quantization import int8_kernel as quant
 k=x.shape[-1];m=x.shape[0];g=groups;n=rank;role=_role(g,n,k);dt=output_dtype or x.dtype
 if role is None or tile_n not in (None,0) or group_major or algo_rank is not None:
  return _project_unpacked(x,w,s,g,n,output_dtype=dt,tile_n=tile_n,group_major=group_major,algo_rank=algo_rank)
 if x.ndim!=3 or x.shape[1]!=g or w.shape!=(g*n,k) or w.dtype!=torch.int8 or s.shape!=(g*n,1) or s.dtype!=torch.float32 or not w.is_contiguous() or not s.is_contiguous() or x.device!=w.device or x.device!=s.device:raise ValueError('Invalid optimized INT8 projection tensors')
 if x.dtype not in (torch.bfloat16,torch.float16,torch.float32) or dt not in (torch.bfloat16,torch.float16,torch.float32):raise ValueError('Floating activations/output required')
 # Preserve caller-requested output dtype; default follows activation dtype.
 y=torch.empty((m,g,n),device=x.device,dtype=dt)
 if m==0:return y
 x=x.contiguous();batches,lda,astride,pitch,shard=_layout(role,m,g,n,k);packed=_pack(w,g,n,k,batches,lda,astride)
 ops=_load();aid=_algorithms().get('%s:%s:%s:%s:%s:%s:%s'%(role,m,pitch,batches,lda,astride,int(shard)),'')
 key=(w.device.index,m,g,n,k,batches,lda,astride,pitch,shard,aid)
 with _lock:
  plan=_plans.get(key)
  if plan is None:
   plan=ops.prepare(packed,m,g,n,k,batches,lda,astride,pitch,shard,aid);_plans[key]=plan
   print('BLASLT_OPT_PLAN',json.dumps(dict(role=role,m=m,groups=g,rank=n,k=k,batches=batches,lda=lda,stride=astride,q_pitch=pitch,shard=shard,algorithm=aid or 'heuristic',input_dtype=str(x.dtype),output_dtype=str(dt))),flush=True)
 qstorage=torch.empty((m,g,pitch),device=x.device,dtype=torch.int8);q=qstorage[:,:,:k];xs=torch.empty((m,g,1),device=x.device,dtype=torch.float32);raw=torch.empty((m,g,n),device=x.device,dtype=torch.int32)
 block=triton.next_power_of_2(k)
 quant._per_token_quant_int8[(m*g,)](x,q,xs,None,stride_x=k,stride_xq=pitch,N=k,CAL_SUM=False,BLOCK=block,IS_HIP=quant._is_hip,num_warps=min(max(block//256,1),8),num_stages=1)
 ops.gemm(q,packed,raw,plan)
 if shard and dt!=torch.float32:
  _scale_grid[(triton.cdiv(n//batches,2048),m,batches)](raw,xs,s,y,m,n,n//batches,2048,num_warps=4,enable_fp_fusion=False)
 elif shard:
  schedule=_scale_schedules().get(str(m))
  if schedule is not None:_scheduled_scale(raw,xs,s,y,m,schedule)
  elif m<=5:_scale_grid[(triton.cdiv(n//batches,2048),m,batches)](raw,xs,s,y,m,n,n//batches,2048,num_warps=4,enable_fp_fusion=False)
  else:ops.scale(raw,xs,s,y,g,batches,True,1)
 elif role=='wo_a' or dt==torch.float16 or (role=='markov_tp8') or (role=='markov_full' and m<=5) or (role=='lm_head' and m>=256):
  sb=2048 if ((role=='wo_a' and m in (16,32,96,128)) or (role=='lm_head' and m>=256) or (role=='markov_tp8' and m>128)) else 1024
  _scale_int32[(triton.cdiv(m*g*n,sb),)](raw,xs,s,y,m,g,n,False,sb,num_warps=8 if sb==2048 else 4,enable_fp_fusion=False)
 else:ops.scale(raw,xs,s,y,g,batches,False,0)
 return y


def _use_packed_service_projection() -> bool:
    return os.environ.get("SGLANG_DSV41_INT8_SERVICE_OPT") == "1"


def prepare_weight(w):
    if _use_packed_service_projection():
        _prepare_packed_weight(w)


def project(x, w, s, groups, rank, **kwargs):
    if _use_packed_service_projection():
        return _project_packed(x, w, s, groups, rank, **kwargs)
    return _project_unpacked(x, w, s, groups, rank, **kwargs)


def try_project_wo_a_with_inverse_rope(x, w, s, groups, rank, freqs_cis, positions):
    # Keep packed service GEMM, small/decode shapes, and graph capture on
    # their existing paths. Large prefill already uses Triton for quantization.
    if not (
        x.is_cuda
        and x.numel() >= _TRITON_QUANT_MIN_ELEMENTS
        and _supports_wo_a_inverse_rope(x, freqs_cis, positions)
        and _triton_quantize_enabled()
        and _is_gfx936_device(x.device)
        and not _use_packed_service_projection()
        and not torch.cuda.is_current_stream_capturing()
    ):
        return None
    return _project_unpacked(
        x,
        w,
        s,
        groups,
        rank,
        input_quant_args=quantize_wo_a_inverse_rope(x, freqs_cis, positions),
    )


@lru_cache(maxsize=4)
def _read_tuning(path):
    return json.loads(Path(path).read_text())


def _vp_tuning(device, n, k, vp_size):
    path = os.environ.get("SGLANG_DSV41_INT8_VP_ALGO_CONFIG")
    if not path:
        return None
    config = _read_tuning(path)
    props = torch.cuda.get_device_properties(device)
    identity = (props.gcnArchName.split(":")[0], props.multi_processor_count,
                str(torch.__version__), n, k, vp_size)
    expected = (config['arch'], config['compute_units'], config['torch_version'],
                config['n'], config['k'], config['vp_size'])
    if config.get('format_version') != 1 or identity != expected:
        raise ValueError("INT8 VP tuning does not match device, runtime or matrix shape.")
    lda = config['weight_lda']
    if not isinstance(lda, int) or lda < n or config['quant_warps'] not in (2, 4, 8):
        raise ValueError("Invalid INT8 VP weight layout or quantization configuration.")
    for rows, item in config['shapes'].items():
        pitch, algorithm = item['q_pitch'], item['algorithm']
        if (int(rows) <= 0 or not isinstance(pitch, int) or pitch < k or pitch % 4
                or not isinstance(algorithm, str) or len(algorithm) != 32):
            raise ValueError("Invalid INT8 VP row shape, pitch or algorithm.")
        try:
            bytes.fromhex(algorithm)
        except ValueError as exc:
            raise ValueError("Invalid INT8 VP algorithm encoding.") from exc
    return config


def require_int8_vp_backend():
    if os.environ.get("SGLANG_DSV41_INT8_PROJECTION_BACKEND", "blaslt") != "blaslt":
        raise ValueError("INT8 LM-head VP requires the blaslt projection backend.")


class Int8VPProjection:
    """Own only the vocabulary shard, with fixed quantization/INT32 scratch.

    Activations have already been gathered across VP owners. Quantize each
    complete hidden row, never a hidden-dimension slice. Preserve the caller's
    floating output dtype and the existing FP32 scale multiplication order.
    """

    def __init__(self, weight, scale, *, vp_rank, vp_size, input_dtype):
        require_int8_vp_backend()
        if (
            weight.dtype != torch.int8 or weight.ndim != 2 or not weight.is_cuda
            or not weight.is_contiguous() or scale.dtype != torch.float32
            or scale.shape != (weight.shape[0], 1) or not scale.is_contiguous()
            or scale.device != weight.device or vp_size <= 1
            or weight.shape[0] % vp_size or not 0 <= vp_rank < vp_size
            or weight.shape[1] % 4
            or input_dtype not in (torch.bfloat16, torch.float16, torch.float32)
        ):
            raise ValueError("INT8 VP needs CUDA INT8 [N,K], FP32 [N,1] scales and floating activations.")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Initialize INT8 VP before graph capture.")

        self.device = weight.device
        self.input_dtype = input_dtype
        self.k = weight.shape[1]
        self.n = weight.shape[0] // vp_size
        self.rank, self.size = vp_rank, vp_size
        self._tuning = _vp_tuning(self.device, self.n, self.k, vp_size)
        self.lda = self._tuning['weight_lda'] if self._tuning else self.n
        self._max_pitch = max([self.k] + [item['q_pitch'] for item in self._tuning['shapes'].values()]) if self._tuning else self.k
        # Weak references allow the upstream compact target+draft VP path to
        # release the complete LM-head after both projections are constructed.
        self._weight_ref = weakref.ref(weight)
        self._scale_ref = weakref.ref(scale)
        self._weight_version = self._version(weight)
        self._scale_version = self._version(scale)
        start = vp_rank * self.n
        shard = weight.detach().narrow(0, start, self.n)
        if self.lda == self.n:
            self.packed = shard.T.clone(memory_format=torch.contiguous_format).view(1, 1, -1)
        else:
            storage = torch.zeros((self.k - 1) * self.lda + self.n,
                                  device=self.device, dtype=torch.int8)
            storage.as_strided((self.k, self.n), (self.lda, 1)).copy_(shard.T)
            self.packed = storage.view(1, 1, -1)
        self.scales = scale.detach().narrow(0, start, self.n).clone(
            memory_format=torch.contiguous_format
        )
        self.ops = _load()
        self.capacity = 0
        self.used = False
        self.plans = {}

    @staticmethod
    def _version(tensor):
        try:
            return tensor._version
        except RuntimeError:
            return -1

    def validate_shared(self, weight, scale, *, vp_rank, vp_size, input_dtype):
        if (
            weight is not self._weight_ref() or scale is not self._scale_ref()
            or (vp_rank, vp_size, input_dtype) != (self.rank, self.size, self.input_dtype)
            or self._version(weight) != self._weight_version
            or self._version(scale) != self._scale_version
        ):
            raise ValueError("Shared INT8 VP requires the same unchanged weight, scale, shard and dtype.")

    def reserve(self, rows):
        if rows <= self.capacity:
            return
        if self.used or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("INT8 VP scratch cannot grow after use or during capture.")
        self._q_storage = torch.empty(rows * self._max_pitch, device=self.device, dtype=torch.int8)
        self.q = self._q_storage[:rows * self.k].view(rows, 1, self.k)
        self.xs = torch.empty((rows, 1, 1), device=self.device, dtype=torch.float32)
        self.raw = torch.empty((rows, 1, self.n), device=self.device, dtype=torch.int32)
        self.capacity = rows

    def project(self, x, out):
        from sglang.kernels.ops.quantization import int8_kernel as quant

        m = x.shape[0]
        if (
            x.shape != (m, self.k) or x.dtype != self.input_dtype
            or x.device != self.device or not x.is_contiguous()
            or out.shape != (m, self.n) or out.dtype != self.input_dtype
            or out.device != self.device or not out.is_contiguous() or m > self.capacity
        ):
            raise ValueError("INT8 VP activation/output layout or capacity mismatch.")
        if not m:
            return
        self.used = True
        shape = self._tuning['shapes'].get(str(m)) if self._tuning else None
        pitch = shape['q_pitch'] if shape else self.k
        plan = self.plans.get(m)
        if plan is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up each INT8 VP row shape before graph capture.")
            plan = self.ops.prepare(self.packed, m, 1, self.n, self.k,
                                    1, self.lda, self.k * self.lda, pitch, False,
                                    shape['algorithm'] if shape else "")
            self.plans[m] = plan
        q = self._q_storage[:m * pitch].view(m, 1, pitch)[:, :, :self.k]
        xs, raw = self.xs[:m], self.raw[:m]
        block = triton.next_power_of_2(self.k)
        quant._per_token_quant_int8[(m,)](
            x, q, xs, None, stride_x=self.k, stride_xq=pitch, N=self.k,
            CAL_SUM=False, BLOCK=block, IS_HIP=quant._is_hip,
            num_warps=self._tuning['quant_warps'] if self._tuning else min(max(block // 256, 1), 8), num_stages=1,
        )
        self.ops.gemm(q, self.packed, raw, plan)
        if out.dtype in (torch.bfloat16, torch.float32):
            self.ops.scale(raw, xs, self.scales, out.view(m, 1, self.n), 1, 1, False, 0)
        else:

            _scale_int32[(triton.cdiv(out.numel(), 1024),)](
                raw, xs, self.scales, out, m, 1, self.n, False, 1024,
                num_warps=4, enable_fp_fusion=False,
            )
