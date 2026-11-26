"""BW1000 parity with LightOp's window-local BF16/FP4 prefill indexer."""

import itertools
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.compact_fp4_candidate_logits import (
    compact_fp4_candidate_logits,
    compact_fp4_candidate_logits_supported,
)
from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2
from sglang.srt.layers.attention.dsv4.candidate_indexer import select_candidate_blocks
from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: gfx936 BF16 MMAC and the HCU LightOp reference.
register_hcu_ci(est_time=30, suite="stage-b-test-1-hcu-small")


@unittest.skipUnless(
    torch.cuda.is_available()
    and compact_fp4_candidate_logits_supported(torch.device("cuda")),
    "gfx936 and LightOp required",
)
class TestDsv41CompactFp4CandidateLogits(CustomTestCase):
    def _check_prefill(self, ratio, cp_size, layer_split, multi_request):
        from lightop import dsv41_mqa_prefill_metadata
        from lightop.attention import (
            dsv41_ragged_topk_postprocess,
            fp8_fp4_mqa_logits,
        )

        from sglang.srt.layers.attention import deepseek_v4_backend as dsv4
        from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
            CpCacheLayerSplitDeepSeekV4TokenToKVPool,
        )
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DeepSeekV4IndexerPool,
            DeepSeekV4TokenToKVPool,
        )

        torch.manual_seed(247)
        seq_lens = [40003 * ratio]
        extend_lens = [16]
        if multi_request:
            seq_lens += [9 * ratio, 1]
            extend_lens += [8, 1]
        # Rank zero's rows from the concatenated chunk, retaining their full
        # request positions. In C2 the one-token request has an empty K window.
        positions_per_request = [
            [seq - extend + row for row in range(extend) if (base + row) % cp_size == 0]
            for seq, extend, base in zip(
                seq_lens, extend_lens, itertools.accumulate([0, *extend_lens])
            )
        ]
        counts = [len(p) for p in positions_per_request]
        lengths = [length // ratio for length in seq_lens]
        device = torch.device("cuda")
        positions = torch.tensor(
            sum(positions_per_request, []), device=device, dtype=torch.int64
        )
        rows = positions.numel()
        width = (max(lengths) + 3) // 4 * 4
        page = 64
        pages = (sum(lengths) + page - 1) // page
        physical_slots = torch.randperm(pages * page, device=device)[: sum(lengths)]
        req_to_token = torch.zeros(
            (len(lengths), max(seq_lens)), dtype=torch.int32, device=device
        )
        request_ids = torch.arange(
            len(lengths) - 1, -1, -1, dtype=torch.int64, device=device
        )
        for request, (length, start) in enumerate(
            zip(lengths, itertools.accumulate([0, *lengths]))
        ):
            req_to_token[len(lengths) - request - 1, : length * ratio : ratio] = (
                physical_slots[start : start + length] * ratio
            )
        ks, ke, lens, slots = dsv41_mqa_prefill_metadata(
            torch.tensor(seq_lens, dtype=torch.int32, device=device),
            torch.tensor(counts, dtype=torch.int32, device=device),
            positions,
            request_ids,
            req_to_token,
            ratio,
            sum(lengths),
            max(lengths),
        )
        self.assertTrue(torch.equal(ke - ks, (positions + 1) // ratio))
        self.assertTrue(torch.equal(ke - ks, lens))
        self.assertTrue(torch.equal(slots, physical_slots))

        payload = torch.randint(
            -128, 128, (pages, page * 64), dtype=torch.int8, device=device
        )
        scales = torch.randint(
            122, 130, (pages, page * 4), dtype=torch.uint8, device=device
        )
        index_pool = object.__new__(DeepSeekV4IndexerPool)
        index_pool.use_fp4_indexer = True
        index_pool.start_layer = 0
        index_pool.page_size = page
        index_pool.index_k_with_scale_buffer = [
            torch.cat((payload.view(torch.uint8), scales), dim=1)
        ]
        pool = object.__new__(
            CpCacheLayerSplitDeepSeekV4TokenToKVPool
            if layer_split
            else DeepSeekV4TokenToKVPool
        )
        pool.layer_transfer_counter = None
        pool.compression_ratios = [ratio]
        pool.layer_mapping = {0: (ratio, 0, None)}
        pool.index_pools = {ratio: index_pool}
        kv = pool.get_low_ratio_index_k_fp4(0, slots, use_hcu_fused_gather=True)
        self.assertIs(
            pool.get_index_k_with_scale_buffer(0),
            index_pool.index_k_with_scale_buffer[0],
        )

        q = torch.randn(rows, 32, 128, dtype=torch.bfloat16, device=device)
        weights = torch.randn(rows, 32, dtype=torch.float32, device=device)
        weights[0].zero_()  # More than 512 exact ties, across a gapped block table.
        dense = fp8_fp4_mqa_logits((q, None), kv, weights, ks, ke, False, width)
        tail = torch.arange(width, device=device)[None, :] >= lens[:, None]
        self.assertTrue(torch.isneginf(dense[tail]).all().item())

        requests = [
            slice(start, start + count)
            for start, count in zip(itertools.accumulate([0, *counts]), counts)
        ]
        blocks = [
            select_candidate_blocks(
                dense[query, :length], lens[query], 2048, 8, return_mask=False
            )
            if count and length
            else torch.empty(count, 0, dtype=torch.int32, device=device)
            for query, count, length in zip(requests, counts, lengths)
        ]
        dense = fp8_fp4_mqa_logits(
            (q, None),
            kv,
            weights,
            ks,
            ke,
            False,
            width,
            candidate_blocks=blocks,
            candidate_query_lens=counts,
        )
        for query, block_table in zip(requests, blocks):
            if not block_table.numel():
                continue
            # A column slice models the row-padded block-table view on the CP tail.
            padded = torch.nn.functional.pad(block_table, (0, 3), value=-1)[:, :-3]
            logits = compact_fp4_candidate_logits(
                q[query], kv, weights[query], ks[query], ke[query], padded
            )
            logical = (
                block_table.long()[:, :, None] * 8 + torch.arange(8, device=device)
            ).flatten(1)
            expected = dense[query].gather(1, logical.clamp(0, width - 1))
            expected.masked_fill_(
                (logical < 0) | (logical >= lens[query, None]), -torch.inf
            )
            self.assertTrue(
                torch.equal(logits.view(torch.int32), expected.view(torch.int32))
            )

        topk = 512
        expected_pages = torch.full(
            (rows + 2, topk + 1), -1, dtype=torch.int32, device=device
        )
        expected_raw = torch.full_like(expected_pages, -1)
        actual_pages, actual_raw = expected_pages.clone(), expected_raw.clone()
        selected = torch.empty((rows, topk), dtype=torch.int32, device=device)
        topk_transform_ragged_v2(dense, lens, out_offsets=ks, out_indices=selected)
        dsv41_ragged_topk_postprocess(
            dense,
            selected,
            ks,
            slots,
            expected_pages[:rows, :topk],
            expected_raw[:rows, :topk],
            True,
        )
        backend = object.__new__(dsv4.DeepseekV4AttnBackend)
        backend.req_to_token = req_to_token
        backend.token_to_kv_pool = pool
        backend.forward_metadata = dsv4.DSV4Metadata(
            SimpleNamespace(
                sparse_page_indices=lambda _: actual_pages,
                sparse_raw_indices=lambda _: actual_raw,
            ),
            None,
            candidate_metadata=dsv4.CandidateMasks(request_block_ids=blocks),
        )
        q_lens = torch.tensor(counts, device=device, dtype=torch.int32)
        backend.forward_metadata.low_ratio_query_lens = q_lens
        backend.forward_metadata.low_ratio_dense_metadata[ratio] = (
            dsv4._LowRatioDenseMetadata(ks, ke, lens, slots)
        )
        layer = SimpleNamespace(
            compress_ratio=ratio,
            layer_id=0,
            freqs_cis=torch.empty(max(seq_lens), device=device),
            indexer=SimpleNamespace(
                uses_candidates=True,
                is_candidate_source=False,
                candidate_block_size=8,
                candidate_topk_blocks=2048,
                n_local_heads=32,
                index_topk=topk,
                queries=lambda q, _: q,
                head_weights=lambda x: x,
            ),
        )
        batch = SimpleNamespace(
            seq_lens_cpu=seq_lens, forward_mode=dsv4.ForwardMode.EXTEND
        )
        with (
            patch.object(
                dsv4,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="prefill"),
            ),
            patch.object(
                dsv4,
                "_hcu_dense_fp4_mqa_logits",
                side_effect=AssertionError("consumer used dense-layout logits"),
            ),
        ):
            backend._low_ratio_index_topk_dense(
                layer,
                torch.nn.functional.pad(weights, (0, 0, 0, 2)),
                torch.nn.functional.pad(q, (0, 0, 0, 0, 0, 2)),
                torch.nn.functional.pad(positions, (0, 2)),
                batch,
                q_lens,
                counts,
            )
        self.assertTrue(torch.equal(actual_pages, expected_pages))
        self.assertTrue(torch.equal(actual_raw, expected_raw))

    def test_ragged_chunked_prefill_matches_dense_logits_and_topk(self):
        for ratio, cp_size, layer_split, multi_request in (
            (1, 1, False, False),
            (2, 1, False, True),
            (1, 8, False, True),
            (1, 8, True, True),
            (2, 8, False, True),
            (2, 8, True, True),
        ):
            with (
                self.subTest(ratio=ratio, cp_size=cp_size, layer_split=layer_split),
                patch.dict(
                    os.environ,
                    {
                        "SGLANG_HCU_PREFILL_PAGED_SPARSE": "0",
                        "SGLANG_HCU_OPT_DSV41_CANDIDATE_LOGITS": "1",
                        "SGLANG_USE_LIGHTOP_DSV41_CANDIDATE_BLOCKS": "1",
                    },
                ),
            ):
                self._check_prefill(ratio, cp_size, layer_split, multi_request)


if __name__ == "__main__":
    unittest.main()
