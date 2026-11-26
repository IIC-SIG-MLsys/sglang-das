"""INT8 activation quantization for DeepSeek V4.1 projections."""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _quantize_int8_rows(x, AXIS: tl.constexpr):
    absmax = tl.maximum(tl.max(tl.abs(x), axis=AXIS), 1.0e-10)
    scale = tl.div_rn(absmax, 127.0)
    inv_scale = tl.div_rn(127.0, absmax)
    q = libdevice.round(x * tl.expand_dims(inv_scale, AXIS))
    return tl.clamp(q, -127.0, 127.0).to(tl.int8), scale


@triton.jit
def _quantize_wo_a_inverse_rope_kernel(
    X,
    FREQS,
    POSITIONS,
    Q,
    S,
    STRIDE_TOKEN: tl.constexpr,
    STRIDE_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // GROUPS
    col = tl.arange(0, BLOCK)
    x_off = token.to(tl.int64) * STRIDE_TOKEN + (row % GROUPS) * STRIDE_GROUP
    x = tl.load(X + x_off + col, col < K, 0).to(tl.float32)

    # Each 512-wide head has a 64-wide, interleaved RoPE tail.
    rope_col = col % 512 - 448
    is_rope = (rope_col >= 0) & (col < K)
    pos = tl.load(POSITIONS + token).to(tl.int64)
    freq_off = pos * 64 + (rope_col // 2) * 2
    cos = tl.load(FREQS + freq_off, is_rope, 0.0)
    sin = tl.load(FREQS + freq_off + 1, is_rope, 0.0)
    x_sin = x * sin
    x_sin = tl.where(col % 2 == 0, -x_sin, x_sin)
    partner = tl.reshape(tl.flip(tl.reshape(x_sin, (BLOCK // 2, 2)), 1), (BLOCK,))
    # Match the standalone inverse RoPE's FP32 FMA and BF16 store before
    # computing the per-group absmax, including columns outside the RoPE tail.
    rotated = tl.fma(x, cos, partner).to(tl.bfloat16).to(tl.float32)
    q, scale = _quantize_int8_rows(tl.where(is_rope, rotated, x), 0)
    tl.store(Q + row.to(tl.int64) * K + col, q, col < K)
    tl.store(S + row, scale)


def _supports_wo_a_inverse_rope(
    x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor
) -> bool:
    return (
        x.ndim == 3
        and x.dtype == torch.bfloat16
        and x.stride(-1) == 1
        and x.shape[1] > 0
        and 512 <= x.shape[-1] <= 8192
        and x.shape[-1] % 512 == 0
        and freqs_cis.device == positions.device == x.device
        and freqs_cis.dtype == torch.complex64
        and freqs_cis.ndim == 2
        and freqs_cis.shape[1] == 32
        and freqs_cis.is_contiguous()
        and positions.shape == (x.shape[0],)
        and positions.dtype in (torch.int32, torch.int64)
        and positions.is_contiguous()
    )


def quantize_wo_a_inverse_rope(
    x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate the RoPE tail and quantize each [token, group] BF16 row to INT8."""
    assert _supports_wo_a_inverse_rope(x, freqs_cis, positions)
    tokens, groups, width = x.shape
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty((tokens, groups, 1), dtype=torch.float32, device=x.device)
    if tokens:
        _quantize_wo_a_inverse_rope_kernel[(tokens * groups,)](
            x,
            torch.view_as_real(freqs_cis),
            positions,
            q,
            scales,
            x.stride(0),
            x.stride(1),
            groups,
            width,
            triton.next_power_of_2(width),
            num_warps=4,
        )
    return q, scales
