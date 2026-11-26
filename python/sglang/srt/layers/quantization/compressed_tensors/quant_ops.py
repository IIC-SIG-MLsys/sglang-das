# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import os
from typing import Dict, Optional, Tuple

import torch

from sglang.srt.utils import get_bool_env_var, is_hcu

_is_hcu = is_hcu()

_W8A8_SMALL_M_AUTOTUNE_MAX_M = 64
_W8A8_SMALL_M_ASM_TUNES: Tuple[int, ...] = (1, 2, 3, 4, 9)
_W8A8_SMALL_M_AUTOTUNE_CACHE: Dict[
    Tuple[Optional[int], int, int, int, torch.dtype], Tuple[str, int]
] = {}
_W8A8_FAILED_STATIC_KEYS: set = set()
_W8A8_FAILED_CONFIG_KEYS: set = set()
_W8A8_DEVICE_ARCH_CACHE: Dict[Optional[int], bool] = {}

# Offline-benchmarked small-M (M <= 64) W8A8 GEMM kernel table for Hygon BW1000
# (gfx936, DTK 26.04.1, BF16 output) covering DeepSeek-V4.1 dense projections.
# Key: (n_i, k_i) -> {m_bucket: (kind, tune)} where m_bucket in {8, 16, 32, 64}.
# Uncovered shapes and non-gfx936 devices fall back directly to
# _default_w8a8_smooth_asm_tune without online synchronization or timing.
_DEFAULT_GFX936_W8A8_SMALL_M_TABLE: Dict[
    Tuple[int, int], Dict[int, Tuple[str, int]]
] = {
    (1792, 5120): {8: ("smooth_asm", 1), 16: ("smooth_asm", 1), 32: ("smooth_asm", 1), 64: ("smooth_asm", 2)},
    (2048, 5120): {8: ("smooth_asm", 1), 16: ("smooth_asm", 1), 32: ("smooth_asm", 1), 64: ("smooth_asm", 2)},
    (2112, 5120): {8: ("smooth_asm", 1), 16: ("smooth_asm", 1), 32: ("smooth_asm", 1), 64: ("smooth_asm", 9)},
    (2304, 5120): {8: ("smooth_asm", 1), 16: ("smooth_asm", 1), 32: ("smooth_asm", 1), 64: ("smooth_asm", 9)},
    (2560, 5120): {8: ("smooth_asm", 1), 16: ("smooth_asm", 1), 32: ("smooth_asm", 1), 64: ("smooth_asm", 9)},
    (3200, 6144): {8: ("smooth_asm", 2), 16: ("smooth_asm", 2), 32: ("smooth_asm", 2), 64: ("smooth_asm", 3)},
    (4096, 1280): {8: ("smooth_asm", 2), 16: ("smooth_asm", 9), 32: ("smooth_asm", 9), 64: ("smooth_asm", 9)},
    (4096, 5120): {8: ("smooth_asm", 2), 16: ("smooth_asm", 2), 32: ("smooth_asm", 2), 64: ("smooth_asm", 9)},
    (4608, 5120): {8: ("smooth_asm", 9), 16: ("smooth_asm", 9), 32: ("smooth_asm", 2), 64: ("smooth_asm", 9)},
    (5120, 2048): {8: ("smooth_asm", 2), 16: ("smooth_asm", 9), 32: ("smooth_asm", 2), 64: ("smooth_asm", 9)},
    (5120, 2304): {8: ("smooth_asm", 9), 16: ("smooth_asm", 9), 32: ("smooth_asm", 2), 64: ("smooth_asm", 9)},
    (5120, 4096): {8: ("smooth_asm", 9), 16: ("smooth_asm", 9), 32: ("smooth_asm", 9), 64: ("smooth_asm", 3)},
    (5120, 5120): {8: ("lightop_channel", 0), 16: ("lightop_channel", 0), 32: ("smooth_asm", 1), 64: ("smooth_asm", 9)},
    (5120, 8192): {8: ("lightop_channel", 0), 16: ("smooth_asm", 1), 32: ("smooth_asm", 1), 64: ("smooth_asm", 3)},
    (8192, 4096): {8: ("lightop_channel", 0), 16: ("smooth_asm", 2), 32: ("smooth_asm", 2), 64: ("smooth_asm", 3)},
    (8192, 5120): {8: ("smooth_asm", 2), 16: ("smooth_asm", 2), 32: ("smooth_asm", 2), 64: ("smooth_asm", 9)},
    (16160, 5120): {8: ("smooth_asm", 1), 16: ("smooth_asm", 1), 32: ("smooth_asm", 1), 64: ("smooth_asm", 9)},
    (16384, 5120): {8: ("smooth_asm", 1), 16: ("smooth_asm", 1), 32: ("smooth_asm", 1), 64: ("smooth_asm", 9)},
    (25600, 6144): {8: ("smooth_asm", 1), 16: ("smooth_asm", 1), 32: ("smooth_asm", 1), 64: ("smooth_asm", 9)},
    (32768, 5120): {8: ("smooth_asm", 1), 16: ("smooth_asm", 1), 32: ("smooth_asm", 1), 64: ("smooth_asm", 9)},
}

try:
    from lightop import gemm_ops as quant_ops
except Exception:
    print(
        "INFO: Please install lightop if you want to infer gptq or awq or w8a8 model.\n"
    )


# lightop's `gemm_w8a8_smooth_asm` loads an arch-specific .co through
# hipModuleLoad inside its C++ `HIP_CALL` macro, which calls `exit(0)` on
# failure; the Python `try/except` below can never rescue that. gfx928 (K100)
# images do not ship the w8a8 smooth asm kernels at all (lightop resolves
# LIGHTOP_ASM_DIR to .../third_party/hsa/gfx928, which is absent), so any call
# there kills the scheduler process during graph capture instead of raising.
# Keep the fast path permanently disabled on those archs and always use the
# tuned Triton path.
_W8A8_SMOOTH_ASM_UNSUPPORTED_ARCHS = ("gfx928",)


@functools.lru_cache(maxsize=8)
def _w8a8_smooth_asm_supported(device: torch.device) -> bool:
    """Whether `lightop.gemm_w8a8_smooth_asm` can run on `device`.

    The asm kernels are precompiled per arch and lightop exits the process
    instead of raising when the matching .co is missing, so a wrong answer here
    is fatal rather than just slow. Disable the fast path on known-unsupported
    archs and stay conservative when the arch cannot be determined.
    """
    try:
        props = torch.cuda.get_device_properties(device)
        arch = getattr(props, "gcnArchName", "").split(":")[0]
    except Exception:
        return False
    return bool(arch) and arch not in _W8A8_SMOOTH_ASM_UNSUPPORTED_ARCHS


def _w8a8_small_m_bucket(m_i: int) -> int:
    """Coarse power-of-2 bucket for M <= 64 so variable-length prefill tail
    chunks reuse the tuned kernel instead of re-probing every distinct M."""
    if m_i <= 8:
        return 8
    if m_i <= 16:
        return 16
    if m_i <= 32:
        return 32
    return 64


def _default_w8a8_smooth_asm_tune(m_i: int, n_i: int) -> Optional[int]:
    if n_i <= 1024 and m_i <= 256:
        return 1
    if n_i <= 2048 and m_i <= 128:
        return 1
    if m_i <= 64:
        return 9
    return None


def w8a8_triton_config(configs, m, n, k):
    if not configs:
        return None
    exact_key = f"{m}_{n}_{k}"
    if exact_key in configs:
        return configs[exact_key]
    if m <= 16:
        return None
    if m <= 64:
        bucket = (m + 3) & -4
    elif m <= 160:
        bucket = (m + 7) & -8
    elif m < 200:
        bucket = 160
    elif m < 480:
        bucket = 256
    elif m < 960:
        bucket = 512
    elif m < 2048:
        bucket = 1024
    elif m < 4096:
        bucket = 2048
    elif m < 6000:
        bucket = 4096
    else:
        bucket = 8192
    return configs.get(f"{bucket}_{n}_{k}")


def _hcu_w8a8_small_m_static_enabled() -> bool:
    return get_bool_env_var("SGLANG_HCU_W8A8_SMALL_M_STATIC", "true")


def _hcu_w8a8_small_m_autotune_enabled() -> bool:
    # Online synchronous timing in forward is strictly opt-in (for offline
    # tuning or explicit startup warmup) so serving requests never pay
    # per-bucket probing/module-load stalls.
    return get_bool_env_var("SGLANG_HCU_W8A8_SMALL_M_AUTOTUNE", "false")


def _is_gfx936_device(device: torch.device) -> bool:
    if device is None or getattr(device, "type", None) != "cuda":
        return False
    idx = device.index
    cached = _W8A8_DEVICE_ARCH_CACHE.get(idx)
    if cached is not None:
        return cached
    try:
        props = torch.cuda.get_device_properties(device)
        arch = getattr(props, "gcnArchName", "").split(":")[0]
    except Exception:
        arch = ""
    match = arch == "gfx936"
    _W8A8_DEVICE_ARCH_CACHE[idx] = match
    return match


def _lookup_default_w8a8_small_m_kernel(
    device: torch.device,
    m_i: int,
    n_i: int,
    k_i: int,
    out_dtype: torch.dtype,
) -> Optional[Tuple[str, int]]:
    if (
        out_dtype != torch.bfloat16
        or not _hcu_w8a8_small_m_static_enabled()
        or not _is_gfx936_device(device)
    ):
        return None
    m_bucket = _w8a8_small_m_bucket(m_i)
    fail_key = (device.index, m_bucket, n_i, k_i, out_dtype)
    if fail_key in _W8A8_FAILED_STATIC_KEYS:
        return None
    per_shape = _DEFAULT_GFX936_W8A8_SMALL_M_TABLE.get((n_i, k_i))
    if per_shape is None:
        return None
    return per_shape.get(m_bucket)


def _try_lightop_channel_int8_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    lmslim = getattr(torch.ops, "lmslim", None)
    fn = getattr(lmslim, "lightop_channel_int8_mm", None) if lmslim is not None else None
    if fn is None:
        return None
    status, out = fn(a, b, scale_a, scale_b, out_dtype, None)
    if (
        status is True
        and out is not None
        and out.shape == (a.shape[0], b.shape[1])
        and out.dtype == out_dtype
    ):
        return out
    return None


def _run_w8a8_small_m_candidate(
    kind: str,
    tune: int,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    m_i: int,
    n_i: int,
    k_i: int,
) -> Optional[torch.Tensor]:
    if kind == "lightop_channel":
        return _try_lightop_channel_int8_mm(a, b, scale_a, scale_b, out_dtype)
    from lightop import op as _lop

    return _lop.gemm_w8a8_smooth_asm(
        a,
        b,
        scale_a,
        scale_b,
        None,
        out_dtype,
        m_i,
        n_i,
        k_i,
        True,
        tune,
    )


def try_configured_w8a8_scaled_mm(a, b, scale_a, scale_b, out_dtype, method):
    path = os.environ.get("SGLANG_DSV41_INT8_BLASLT_ALGO_CONFIG")
    if (
        not path
        or not _is_hcu
        or method not in (1, 3)
        or not a.is_cuda
        or out_dtype != torch.bfloat16
        or a.ndim != 2
        or b.ndim != 2
        or a.dtype != torch.int8
        or b.dtype != torch.int8
        or not a.is_contiguous()
        or a.device != b.device
        or a.device != scale_a.device
        or a.device != scale_b.device
    ):
        return None
    from sglang.kernels.ops.gemm import dsv41_int8
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled

    if is_batch_invariant_mode_enabled():
        return None
    m, k = a.shape
    weight = b.t() if method == 1 else b
    n = weight.shape[0]
    if (
        not m
        or k % 4
        or weight.shape[1] != k
        or not weight.is_contiguous()
        or scale_a.numel() != m
        or scale_b.numel() != n
        or scale_a.dtype != torch.float32
        or scale_b.dtype != torch.float32
        or not scale_a.is_contiguous()
        or not scale_b.is_contiguous()
    ):
        return None
    selected = dsv41_int8.configured_w8a8_algo(path, a.device, method, m, n, k)
    if selected is None:
        return None
    kind, tune = selected
    if kind == "hipblaslt":
        return dsv41_int8.try_w8a8_blaslt(a, weight, scale_a, scale_b, out_dtype, tune)
    if kind == "smooth_asm" and (k % 256 or not _w8a8_smooth_asm_supported(a.device)):
        return None
    key = (a.device, m, n, k, kind, tune)
    if key in _W8A8_FAILED_CONFIG_KEYS:
        return None
    try:
        out = _run_w8a8_small_m_candidate(
            kind, tune, a, weight.t(), scale_a, scale_b, out_dtype, m, n, k
        )
    except (RuntimeError, ImportError):
        out = None
    if (
        not isinstance(out, torch.Tensor)
        or out.shape != (m, n)
        or out.dtype != out_dtype
    ):
        _W8A8_FAILED_CONFIG_KEYS.add(key)
        return None
    return out


def _select_hcu_w8a8_small_m_kernel(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    m_i: int,
    n_i: int,
    k_i: int,
) -> Optional[Tuple[str, int]]:
    m_bucket = _w8a8_small_m_bucket(m_i)
    cache_key = (a.device.index, m_bucket, n_i, k_i, out_dtype)
    cached = _W8A8_SMALL_M_AUTOTUNE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        return None

    candidates = [("lightop_channel", 0)] + [
        ("smooth_asm", t) for t in _W8A8_SMALL_M_ASM_TUNES
    ]
    best: Optional[Tuple[str, int]] = None
    best_ms = float("inf")
    stream = torch.cuda.current_stream(a.device)
    warmup = 2
    iters = 6

    for kind, tune in candidates:
        try:
            out = _run_w8a8_small_m_candidate(
                kind, tune, a, b, scale_a, scale_b, out_dtype, m_i, n_i, k_i
            )
            if out is None:
                continue
            for _ in range(warmup - 1):
                _run_w8a8_small_m_candidate(
                    kind, tune, a, b, scale_a, scale_b, out_dtype, m_i, n_i, k_i
                )
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            for _ in range(iters):
                _run_w8a8_small_m_candidate(
                    kind, tune, a, b, scale_a, scale_b, out_dtype, m_i, n_i, k_i
                )
            end.record(stream)
            end.synchronize()
            elapsed = start.elapsed_time(end)
            if elapsed < best_ms:
                best_ms = elapsed
                best = (kind, tune)
        except Exception:
            continue

    if best is not None:
        _W8A8_SMALL_M_AUTOTUNE_CACHE[cache_key] = best
    return best


def triton_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
    best_config: Optional[list] = None,
) -> torch.Tensor:
    # w8a8 decode with small M is not covered by the smooth GEMM tuning table;
    # autotune per shape (or fall back to explicit slide values) to avoid
    # excessive padding in the default heuristic.
    if (
        _is_hcu
        and best_config is None
        and bias is None
        and a.dim() == 2
        and a.is_contiguous()
        and a.dtype == torch.int8
        and b.dtype == torch.int8
        and b.dim() == 2
        and b.stride() == (1, b.shape[0])
        and a.shape[-1] == b.shape[0]
        and b.shape[0] >= 128
    ):
        m_i = a.shape[0]
        k_i = b.shape[0]
        n_i = b.shape[1]
        # The tuned slide values are valid for complete K=256 tiles.
        if k_i % 256 == 0 and _w8a8_smooth_asm_supported(a.device):
            if m_i <= _W8A8_SMALL_M_AUTOTUNE_MAX_M:
                used_static = False
                if _hcu_w8a8_small_m_autotune_enabled():
                    selected = _select_hcu_w8a8_small_m_kernel(
                        a, b, scale_a, scale_b, out_dtype, m_i, n_i, k_i
                    )
                else:
                    selected = _lookup_default_w8a8_small_m_kernel(
                        a.device, m_i, n_i, k_i, out_dtype
                    )
                    used_static = selected is not None
                if selected is not None:
                    try:
                        out = _run_w8a8_small_m_candidate(
                            selected[0],
                            selected[1],
                            a,
                            b,
                            scale_a,
                            scale_b,
                            out_dtype,
                            m_i,
                            n_i,
                            k_i,
                        )
                        if out is not None:
                            return out
                    except Exception:
                        pass
                    if used_static:
                        _W8A8_FAILED_STATIC_KEYS.add(
                            (
                                a.device.index,
                                _w8a8_small_m_bucket(m_i),
                                n_i,
                                k_i,
                                out_dtype,
                            )
                        )
            tune = _default_w8a8_smooth_asm_tune(m_i, n_i)
            if tune is not None:
                try:
                    from lightop import op as _lop

                    return _lop.gemm_w8a8_smooth_asm(
                        a,
                        b,
                        scale_a,
                        scale_b,
                        None,
                        out_dtype,
                        m_i,
                        n_i,
                        k_i,
                        True,
                        tune,
                    )
                except Exception:
                    pass

    return quant_ops.triton_scaled_mm(
        a, b, scale_a, scale_b, out_dtype, bias, best_config
    )


def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    `cutlass_scaled_mm` implements a fused version of
        `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
    where scale_a * a and scale_b * b are implemented using numpy-style
    broadcasting.

    In order to support blockwise scaling like found in DeepSeek V3 we also
    support extended "group" broadcast rules. We extend the numpy-style
    broadcasting rules with the following rule:
        "if the extent of a dimension in the source shape is between 1 and
        corresponding extent in the target shape we repeat each element along
        that dimension  src_shape[dim] // target_shape[dim] times consecutively"
    example if we have:
          a = [[1, 2], and target_shape = (2, 4)
               [3, 4]]
    then we would expand a to:
          a = [[1, 1, 2, 2],
               [3, 3, 4, 4]]
    currently we only support the case:
        scale_a.shape * [1, 128] == a.shape
        scale_b.shape * [128, 128] == b.shape
    """
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.shape[0] == b.shape[1] and bias.dtype == out_dtype

    # m = a.shape[0]
    # n = b.shape[1]

    # cutlass_compatible_b = (b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0)
    # if current_platform.is_rocm() or not cutlass_compatible_b:
    #     from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa
    #         triton_scaled_mm)
    #     return triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)

    # out = torch.empty((m, n), dtype=out_dtype, device=a.device)

    # torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    # return out
    # return quant_ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    return quant_ops.rocblas_scaled_mm_nn(a, b, scale_a, scale_b, out_dtype, bias)


def rocblas_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    return quant_ops.rocblas_scaled_mm_nn(a, b, scale_a, scale_b, out_dtype, bias)


def blaslt_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """W8A8 scaled GEMM via hipBLASLt.

    Preferred path on KME/gfx928: lmslim ``hipblaslt_gemm_channelwise_kme``
    (swapped TN, bf16 C/D, fp32 compute) with weight layout ``[K, N]``
    col-major stride ``(1, K)``.

    Fallback: legacy ``hipblaslt_w8a8_gemm`` expecting contiguous weight
    ``[N, K]`` (NT logical / TN API).
    """
    m, k = a.shape[-2], a.shape[-1]
    # KME packed weight: [K, N] with stride (1, K)
    if b.dim() == 2 and b.shape[0] == k and b.stride(0) == 1 and b.stride(1) == k:
        n = b.shape[1]
        return _blaslt_scaled_mm_kme(a, b, scale_a, scale_b, m, n, k, out_dtype, bias)

    # Legacy contiguous [N, K]
    n = b.shape[0]
    _, out = quant_ops.hipblaslt_w8a8_gemm(
        a, b, scale_a, scale_b, m, n, k, "NT", out_dtype
    )
    return out


def _blaslt_scaled_mm_kme(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    m: int,
    n: int,
    k: int,
    out_dtype: torch.dtype,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Logical O[M,N]=X[M,K]@W[K,N] via hipblaslt_gemm_channelwise_kme TN."""
    import lmslimquant

    if out_dtype is not torch.bfloat16:
        raise ValueError(
            "hipblaslt_gemm_channelwise_kme currently requires bfloat16 out"
        )
    if not a.is_contiguous():
        a = a.contiguous()
    scale_a = scale_a.reshape(-1).contiguous()
    scale_b = scale_b.reshape(-1).contiguous()
    bias_arg = None
    if bias is not None:
        if bias.dtype != out_dtype:
            bias = bias.to(out_dtype)
        bias_arg = bias.reshape(-1).contiguous()
    alpha = torch.tensor(1, dtype=torch.int32)
    beta = torch.tensor(0, dtype=torch.int32)
    # API (m,n,k)=(N,M,K), A=W, B=X, trans=TN
    out = lmslimquant.hipblaslt_gemm_channelwise_kme(
        b,
        a,
        scale_b,
        scale_a,
        out_dtype,
        n,
        m,
        k,
        1,
        "TN",
        alpha,
        beta,
        bias_arg,
    )
    if out.dim() == 3:
        out = out.squeeze(0)
    return out


def triton_int8_gemm_helper(
    m: int,
    n: int,
    k: int,
    per_token_act_quant: bool,
    per_out_channel_weight_quant: bool,
    use_bias: bool,
    out_dtype: type[torch.dtype] = torch.float16,
    device: str = "cuda:0",
    best_config: Optional[list] = None,
    repeat: Optional[int] = 2,
):
    return quant_ops.triton_int8_gemm_helper(
        m,
        n,
        k,
        per_token_act_quant,
        per_out_channel_weight_quant,
        use_bias,
        out_dtype,
        device,
        best_config,
        repeat,
    )
