"""Time dense LightOp vs. block-pruned DSV4.1 prefill on BW1000 with HIP events."""

from __future__ import annotations

import argparse
import logging
from functools import partial

import torch

from sglang.kernels.ops.attention.dsv4.fused_fp4_mqa import (
    BLOCK_SIZE,
    SOURCE_TOPK,
    TOKEN_TOPK,
    _block_maxima,
    _score_shortlist,
    _sort_prune_blocks,
    fused_fp4_mqa_supported,
    fused_fp4_mqa_topk,
)
from sglang.kernels.ops.attention.dsv4.lightop_indexer import candidate_block_table
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2


def time_ms(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


def run_case(chunk, ratio, warmup, iters, verify):
    from lightop.attention import fp8_fp4_mqa_logits

    q_rows, chunk_tokens, cp = 4096, 32768, 8
    context = chunk * chunk_tokens
    keys = context // ratio
    device = torch.device("cuda")
    # CP8 rank zero. Lengths use the global positions and floor by the
    # layer's compression ratio. All shapes retain 32 heads per rank.
    lengths = (
        context
        - chunk_tokens
        + 1
        + torch.arange(q_rows, device=device, dtype=torch.int32) * cp
    ) // ratio
    starts = torch.zeros_like(lengths)
    magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=device)
    codes = torch.randint(0, 16, (q_rows, 32, 128), device=device, dtype=torch.int32)
    q = (magnitudes[(codes & 7).long()] * torch.where(codes < 8, 1, -1)).bfloat16()
    del codes
    weights = torch.randn((q_rows, 32), dtype=torch.bfloat16, device=device).float()
    weights.mul_(1 / 64)
    packed = torch.randint(-128, 128, (keys, 64), dtype=torch.int8, device=device)
    sf = torch.randint(123, 132, (keys, 4), dtype=torch.int32, device=device)
    kv = (packed, sf[:, 0] | (sf[:, 1] << 8) | (sf[:, 2] << 16) | (sf[:, 3] << 24))

    # One token block per logical page. LightOp permits a zero row stride
    # for the identity table, so this fixture adds no [Q, L/8] input copy.
    identity = torch.arange((keys + 7) // 8, dtype=torch.int32, device=device)
    pages = identity.expand(q_rows, -1)

    def score():
        return fp8_fp4_mqa_logits((q, None), kv, weights, starts, lengths, False, keys)

    def block_topk(logits):
        return candidate_block_table(logits, lengths, pages, 8, 2048, 8)

    def token_topk(logits):
        out = torch.empty((q_rows, 512), dtype=torch.int32, device=device)
        topk_transform_ragged_v2(logits, lengths, out_offsets=starts, out_indices=out)
        return out

    def baseline(publish):
        logits = score()
        blocks = block_topk(logits) if publish else None
        return token_topk(logits), blocks

    def fused(publish):
        return fused_fp4_mqa_topk(q, kv, weights, lengths, publish_blocks=publish)

    if verify:
        tokens, (ids, valid_lens) = baseline(True)
        count = ((valid_lens + 7) // 8)[:, None]
        column = torch.arange(2048, dtype=torch.int32, device=device)
        ids.masked_fill_(column >= count, -1)
        new_tokens, new_ids = fused(True)
        expected_tokens = tokens.sort().values
        torch.testing.assert_close(new_tokens.sort().values, expected_tokens)
        torch.testing.assert_close(new_ids, ids)
        torch.testing.assert_close(fused(False)[0].sort().values, expected_tokens)
        del tokens, ids, valid_lens, new_tokens, new_ids, expected_tokens

    score_ms = time_ms(score, warmup, iters)
    logits = score()
    blocks_ms = time_ms(partial(block_topk, logits), warmup, iters)
    tokens_ms = time_ms(partial(token_topk, logits), warmup, iters)
    del logits
    max_ms = time_ms(lambda: _block_maxima(q, kv, weights, lengths), warmup, iters)
    maxima = _block_maxima(q, kv, weights, lengths)
    prune = torch.empty((q_rows, TOKEN_TOPK), dtype=torch.int32, device=device)
    block_lens = (lengths + BLOCK_SIZE - 1) // BLOCK_SIZE
    topk_transform_ragged_v2(maxima, block_lens, out_offsets=starts, out_indices=prune)
    del maxima
    _sort_prune_blocks[(q_rows,)](
        prune,
        lengths,
        torch.empty_like(lengths),
        None,
        FROM_SOURCE=False,
        SOURCE_K=SOURCE_TOPK,
        K=TOKEN_TOPK,
        B=BLOCK_SIZE,
    )
    rescore_ms = time_ms(
        partial(_score_shortlist, q, kv, weights, lengths, prune), warmup, iters
    )
    del prune
    dense_ms = time_ms(lambda: baseline(False), warmup, iters)
    fused_dense_ms = time_ms(lambda: fused(False), warmup, iters)
    source_ms = time_ms(lambda: baseline(True), warmup, iters)
    fused_source_ms = time_ms(lambda: fused(True), warmup, iters)
    tflop = lengths.sum(dtype=torch.int64).item() * (2 * 32 * 128) / 1e12
    print(
        f"{chunk:>2}  C{ratio}  {q_rows:>5}  {keys:>7}  "
        f"{score_ms:>8.2f}  {blocks_ms:>7.2f}  {tokens_ms:>7.2f}  {max_ms:>8.2f}  {rescore_ms:>9.2f}  "
        f"{dense_ms:>8.2f}  {fused_dense_ms:>8.2f}  {source_ms:>8.2f}  "
        f"{fused_source_ms:>8.2f}  {tflop * 1000 / score_ms:>7.1f}  "
        f"{tflop * 1000 / max_ms:>7.1f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--chunks", type=int, nargs="+", choices=range(1, 13), default=(1, 6, 12)
    )
    parser.add_argument("--ratios", type=int, nargs="+", choices=(1, 2), default=(1, 2))
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--verify", action="store_true", help="check every timed shape against LightOp"
    )
    args = parser.parse_args()
    if args.iters < 1 or args.warmup < 0:
        parser.error("--iters must be positive and --warmup nonnegative")
    if not torch.cuda.is_available() or not fused_fp4_mqa_supported(
        torch.device("cuda", 0)
    ):
        parser.error("a gfx936 GPU and HIP PyTorch are required")
    import lightop

    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(936)
    print(
        f"{torch.cuda.get_device_name()} | HIP {torch.version.hip} | LightOp {getattr(lightop, 'git_hash', '?')}"
    )
    print("Times in ms, rates in useful BF16 TFLOP/s. Source adds block selection.")
    print(
        "ch  cr      Q        L     score   blocks   tok-k     amax8  rescore4K  "
        "    dense     fused    source   fused-s    score    amax8",
        flush=True,
    )
    for chunk in args.chunks:
        for ratio in args.ratios:
            run_case(chunk, ratio, args.warmup, args.iters, args.verify)


if __name__ == "__main__":
    main()
