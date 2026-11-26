"""gfx936 dense BF16-Q/FP4-K prefill with exact block-pruned token selection."""

from __future__ import annotations

import logging
from functools import cache

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, load_jit

from .topk import topk_transform_ragged_v2
from .utils import make_name

logger = logging.getLogger(__name__)

BLOCK_SIZE = 8
TOKEN_TOPK = 512
SOURCE_TOPK = 2048
MAX_QUERY_ROWS = 65535


@cache
def fused_fp4_mqa_supported(device: torch.device) -> bool:
    return (
        device.type == "cuda"
        and torch.version.hip is not None
        and getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(
            ":"
        )[0]
        == "gfx936"
    )


@cache_once
def _jit_fused_fp4_mqa():
    return load_jit(
        make_name("fused_fp4_mqa"),
        cuda_files=["deepseek_v4/fused_fp4_mqa.cuh"],
        cuda_wrappers=[
            ("block_max", "Dsv41FusedFp4MqaKernel::block_max"),
            ("compact_scores", "Dsv41FusedFp4MqaKernel::compact_scores"),
        ],
    )


def _check_lightop_ties(device: torch.device) -> bool:
    # The proof needs the same logical-ID tie break on long and compact rows.
    # Probe the installed wheel once, on eager activation, before using pruning.
    columns = torch.arange(32768, dtype=torch.int32, device=device)
    scores = torch.stack(
        (
            columns.new_zeros(columns.shape),
            columns.new_zeros(columns.shape),
            (columns % 2 == 0).int(),
            (columns % 2 == 0).int(),
        )
    ).float()
    lengths = torch.tensor([4096, 32768, 8192, 32768], dtype=torch.int32, device=device)
    offsets = torch.zeros_like(lengths)
    checks = []
    for k in (TOKEN_TOPK, SOURCE_TOPK):
        out = torch.empty((4, k), dtype=torch.int32, device=device)
        topk_transform_ragged_v2(scores, lengths, out_offsets=offsets, out_indices=out)
        expected = columns[:k] * columns.new_tensor([[1], [1], [2], [2]])
        checks.append((out.sort(dim=1).values == expected).all())
    return bool(torch.stack(checks).all().item())


@cache
def _fused_fp4_mqa_loadable(device: torch.device) -> bool:
    try:
        from lightop import op

        if not hasattr(op, "topk_transform_ragged_v2_interface"):
            return False
        with torch.cuda.device(device):
            if not _check_lightop_ties(device):
                logger.warning("Fused FP4 MQA requires LightOp's logical-ID tie order")
                return False
            _jit_fused_fp4_mqa()
    except (AttributeError, ImportError, OSError, RuntimeError) as exc:
        logger.warning("Fused FP4 MQA unavailable: %s", exc)
        return False
    logger.info("Using gfx936 fused FP4 MQA for dense DSV4.1 prefill")
    return True


def fused_fp4_mqa_available(device: torch.device) -> bool:
    return (
        fused_fp4_mqa_supported(device)
        and not torch.cuda.is_current_stream_capturing()
        and _fused_fp4_mqa_loadable(device)
    )


def _block_maxima(q, kv, weights, lengths):
    rows = q.shape[0]
    keys = kv[0].shape[0]
    out = torch.empty(
        (rows, triton.cdiv(keys, BLOCK_SIZE)), dtype=torch.float32, device=q.device
    )
    # Match LightOp's persistent traversal. Split Q only until the K grid
    # exposes enough CTAs. Each additional shard stages the same K tile again.
    key_tiles = triton.cdiv(keys, 128)
    target_ctas = 48 * torch.cuda.get_device_properties(q.device).multi_processor_count
    shards = min(
        16,
        triton.cdiv(rows, 32),
        triton.next_power_of_2(triton.cdiv(target_ctas, key_tiles)),
    )
    _jit_fused_fp4_mqa().block_max(out, q, *kv, weights, lengths, shards)
    return out


@triton.jit
def _prepare_block_lengths(
    Lengths,
    BlockLengths,
    ZeroOffsets,
    Maxima,
    Newest,
    rows,
    maxima_stride,
    PUBLISH: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0) * 256 + tl.arange(0, 256)
    length = tl.load(Lengths + row, row < rows, 0)
    nblocks = tl.cdiv(length, B)
    tl.store(BlockLengths + row, nblocks, row < rows)
    tl.store(ZeroOffsets + row, 0, row < rows)
    if PUBLISH:
        dst = Maxima + row.to(tl.int64) * maxima_stride + nblocks - 1
        score = tl.load(dst, nblocks > 0, -float("inf"))
        tl.store(Newest + row, score, row < rows)
        tl.store(dst, float("inf"), nblocks > 0)


@triton.jit
def _sort_prune_blocks(
    Blocks,
    Lengths,
    CompactLengths,
    SourceBlocks,
    FROM_SOURCE: tl.constexpr,
    SOURCE_K: tl.constexpr,
    K: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    lanes = tl.arange(0, K)
    ids = tl.load(Blocks + row * K + lanes)
    if FROM_SOURCE:
        ids = tl.load(
            SourceBlocks + row * SOURCE_K + ids, (ids >= 0) & (ids < SOURCE_K), -1
        )
    sentinel = 2147483647
    ordered = tl.sort(tl.where(ids >= 0, ids, sentinel), descending=False)
    tl.store(Blocks + row * K + lanes, tl.where(ordered < sentinel, ordered, -1))
    length = tl.load(Lengths + row)
    nblocks = tl.cdiv(length, B)
    last = nblocks - 1
    largest = tl.max(tl.where((ids >= 0) & (ids < sentinel), ids, -2), axis=0)
    padding = tl.where(largest == last, nblocks * B - length, 0)
    tl.store(CompactLengths + row, tl.minimum(nblocks, K) * B - padding)


@triton.jit
def _gather_source_maxima(
    Blocks,
    Maxima,
    Newest,
    BlockLengths,
    Scores,
    ScoreLengths,
    maxima_stride,
    K: tl.constexpr,
):
    row = tl.program_id(0)
    dst = Blocks + row * K + tl.arange(0, K)
    ids = tl.load(dst)
    sentinel = 2147483647
    ordered = tl.sort(tl.where(ids >= 0, ids, sentinel), descending=False)
    live = ordered < sentinel
    tl.store(dst, tl.where(live, ordered, -1))
    nblocks = tl.load(BlockLengths + row)
    newest = tl.load(Newest + row)
    scores = tl.load(
        Maxima + row.to(tl.int64) * maxima_stride + ordered, live, -float("inf")
    )
    tl.store(
        Scores + row * K + tl.arange(0, K),
        tl.where(ordered == nblocks - 1, newest, scores),
    )
    tl.store(ScoreLengths + row, tl.minimum(nblocks, K))


@triton.jit
def _remap_shortlist(Selected, Blocks, Offsets, K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    dst = Selected + row * K + tl.arange(0, K)
    cols = tl.load(dst)
    live = (cols >= 0) & (cols < K * B)
    block = tl.load(Blocks + row * K + cols // B, live, other=0)
    offset = tl.load(Offsets + row)
    mapped = offset + block * B + cols % B
    tl.store(dst, tl.where(live, mapped, -1))


def _score_shortlist(q, kv, weights, lengths, blocks):
    out = torch.empty(
        (q.shape[0], blocks.shape[1] * BLOCK_SIZE),
        dtype=torch.float32,
        device=q.device,
    )
    _jit_fused_fp4_mqa().compact_scores(out, q, *kv, weights, lengths, blocks)
    return out


def fused_fp4_mqa_topk(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    lengths: torch.Tensor,
    *,
    offsets: torch.Tensor | None = None,
    publish_blocks: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Select exact Top-512 tokens of one request's shared FP4 key buffer.

    Q is contiguous BF16 [rows, 32, 128], weights FP32 [rows, 32], and KV is
    (packed int8 [keys, 64], UE8M0 int32 [keys]). Each length is the visible
    prefix for that query. Lengths and optional output offsets are contiguous
    int32 [rows]. Offsets are nonnegative and mapped IDs fit in int32.
    Scores follow LightOp's ragged TopK contract (no NaNs).

    Returns unordered int32 token IDs, with offsets added if supplied, and
    optionally the ascending int32 Top-2048 logical candidate blocks. Both
    outputs use -1 padding. The source table follows LightOp's newest-block
    promotion. Callers with multiple requests slice the gathered KV per
    request to keep block boundaries request-local.
    """
    if not fused_fp4_mqa_supported(q.device):
        raise ValueError("DSV4.1 fused FP4 MQA requires gfx936")
    rows = q.shape[0]
    keys = kv[0].shape[0]
    if offsets is not None and (
        offsets.shape != (rows,)
        or offsets.dtype != torch.int32
        or offsets.device != q.device
        or not offsets.is_contiguous()
    ):
        raise ValueError(
            "FP4 MQA offsets must be contiguous int32 [rows] on Q's device"
        )
    selected = torch.empty((rows, TOKEN_TOPK), dtype=torch.int32, device=q.device)
    source = (
        torch.empty((rows, SOURCE_TOPK), dtype=torch.int32, device=q.device)
        if publish_blocks
        else None
    )
    if not rows or not keys:
        selected.fill_(-1)
        if source is not None:
            source.fill_(-1)
        return selected, source

    # Each TopK reads its row's visible prefix, so neither score buffer needs
    # a fill for causal padding.
    maxima = _block_maxima(q, kv, weights, lengths)
    num_blocks = min(maxima.shape[1], TOKEN_TOPK)
    zero = torch.empty_like(lengths)
    block_lens = torch.empty_like(lengths)
    newest = (
        torch.empty((rows,), dtype=torch.float32, device=q.device)
        if source is not None
        else None
    )
    _prepare_block_lengths[(triton.cdiv(rows, 256),)](
        lengths,
        block_lens,
        zero,
        maxima,
        newest,
        rows,
        maxima.stride(0),
        PUBLISH=publish_blocks,
        B=BLOCK_SIZE,
    )
    prune = torch.empty_like(selected)
    if source is None:
        topk_transform_ragged_v2(
            maxima, block_lens, out_offsets=zero, out_indices=prune
        )
    else:
        # Promoting one block displaces at most one true top block. The
        # Top-2048 source table therefore contains every true Top-512
        # block. Recover their scores there without another wide TopK.
        topk_transform_ragged_v2(
            maxima, block_lens, out_offsets=zero, out_indices=source
        )
        source_scores = torch.empty(
            (rows, SOURCE_TOPK), dtype=torch.float32, device=q.device
        )
        source_lens = torch.empty_like(lengths)
        _gather_source_maxima[(rows,)](
            source,
            maxima,
            newest,
            block_lens,
            source_scores,
            source_lens,
            maxima.stride(0),
            K=SOURCE_TOPK,
        )
        topk_transform_ragged_v2(
            source_scores, source_lens, out_offsets=zero, out_indices=prune
        )
        del source_scores
    del maxima

    compact_lens = torch.empty_like(lengths)
    _sort_prune_blocks[(rows,)](
        prune,
        lengths,
        compact_lens,
        source,
        FROM_SOURCE=publish_blocks,
        SOURCE_K=SOURCE_TOPK,
        K=TOKEN_TOPK,
        B=BLOCK_SIZE,
    )

    # Block order is the order of each block's best (score, -token_id).
    # Any omitted block has at least TOKEN_TOPK better tokens from distinct
    # blocks. Ascending block IDs preserve logical token order on equal scores.
    compact = _score_shortlist(q, kv, weights, lengths, prune[:, :num_blocks])
    topk_transform_ragged_v2(
        compact, compact_lens, out_offsets=zero, out_indices=selected
    )
    _remap_shortlist[(rows,)](
        selected,
        prune,
        zero if offsets is None else offsets,
        K=TOKEN_TOPK,
        B=BLOCK_SIZE,
    )
    return selected, source
