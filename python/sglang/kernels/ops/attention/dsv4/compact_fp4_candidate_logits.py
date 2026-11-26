import logging
from functools import cache

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

from .utils import make_name

logger = logging.getLogger(__name__)


@cache
def compact_fp4_candidate_logits_supported(device: torch.device) -> bool:
    return (
        device.type == "cuda"
        and torch.version.hip is not None
        and torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
        == "gfx936"
    )


@cache_once
def _jit_compact_fp4_candidate_logits():
    return load_jit(
        make_name("compact_fp4_candidate_logits"),
        cuda_files=["deepseek_v4/compact_fp4_candidate_logits.cuh"],
        cuda_wrappers=[("run", "CompactFp4CandidateLogitsKernel::run")],
    )


@cache
def _compact_kernel_loadable() -> bool:
    try:
        _jit_compact_fp4_candidate_logits()
    except (ImportError, OSError, RuntimeError) as exc:
        logger.warning("Compact FP4 candidate scorer unavailable: %s", exc)
        return False
    return True


def compact_fp4_candidate_logits_available(device: torch.device) -> bool:
    return (
        compact_fp4_candidate_logits_supported(device)
        and not torch.cuda.is_current_stream_capturing()
        and _compact_kernel_loadable()
    )


def compact_fp4_candidate_logits(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    blocks: torch.Tensor,
) -> torch.Tensor:
    """Score BF16 Q against packed FP4 K, in request-local candidate order.

    Column j corresponds to K[starts[row] + 8 * blocks[row, j // 8] + j % 8].
    Negative IDs and keys beyond a row's causal end produce -inf.
    """
    if not compact_fp4_candidate_logits_supported(q.device):
        raise ValueError("compact FP4 candidate logits require gfx936")
    rows, block_count = blocks.shape
    out = torch.empty((rows, block_count * 8), dtype=torch.float32, device=q.device)
    _jit_compact_fp4_candidate_logits().run(out, q, *kv, weights, starts, ends, blocks)
    return out
