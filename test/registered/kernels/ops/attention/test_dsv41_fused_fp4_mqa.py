"""BW1000 parity for the dense DSV4.1 block-pruned FP4 indexer."""

import random
import unittest

import torch

from sglang.test.ci.ci_register import register_cpu_ci, register_hcu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
# backend-specific: the FP4 scorer uses gfx936 MMAC and the LightOp ragged ABI.
register_hcu_ci(est_time=30, suite="stage-b-test-1-hcu-small")


def _is_gfx936():
    return (
        torch.version.hip is not None
        and torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx936"
    )


def _inputs(rows, keys):
    device = "cuda"
    q = torch.randn((rows, 32, 128), dtype=torch.bfloat16, device=device)
    weights = torch.randn((rows, 32), dtype=torch.float32, device=device)
    packed = torch.randint(-128, 128, (keys, 64), dtype=torch.int8, device=device)
    sf = torch.randint(120, 132, (keys, 4), dtype=torch.int32, device=device)
    scales = sf[:, 0] | (sf[:, 1] << 8) | (sf[:, 2] << 16) | (sf[:, 3] << 24)
    return q, (packed, scales), weights


class TestBlockPruning(CustomTestCase):
    def test_block_order_preserves_token_order_at_ties(self):
        k, b = 512, 8
        width = 16405
        rng = random.Random(936)
        for scores in (
            [0] * width,
            [idx % 17 for idx in range(width)],
            [rng.randrange(-32, 32) for _ in range(width)],
        ):

            def rank(idx):
                return -scores[idx], idx

            ids = range(len(scores))
            blocks = [ids[start : start + b] for start in range(0, len(ids), b)]
            ranked = sorted(blocks, key=lambda block: min(map(rank, block)))
            shortlist = [idx for block in ranked[:k] for idx in block]
            self.assertEqual(sorted(shortlist, key=rank)[:k], sorted(ids, key=rank)[:k])
            newest = blocks[-1]
            source = [newest] + [block for block in ranked if block != newest][:2047]
            self.assertEqual(
                sorted(source, key=lambda block: min(map(rank, block)))[:k], ranked[:k]
            )


@unittest.skipUnless(_is_gfx936(), "requires a gfx936 HIP device")
class TestDsv41FusedFp4Mqa(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.attention.dsv4 import fused_fp4_mqa

        super().setUpClass()
        cls.fused = fused_fp4_mqa
        torch.manual_seed(936)
        cls.fused._jit_fused_fp4_mqa()
        if not cls.fused._check_lightop_ties(torch.device("cuda", 0)):
            raise AssertionError("installed LightOp fails the logical-ID tie check")

    @staticmethod
    def _dense(q, kv, weights, starts, lengths, width):
        from lightop.attention import fp8_fp4_mqa_logits

        return fp8_fp4_mqa_logits(
            (q, None), kv, weights, starts, starts + lengths, False, width
        )

    def test_score_bits_and_persistent_query_tail(self):
        rows, keys = 257, 267
        q, kv, weights = _inputs(rows, keys)
        lengths = (torch.arange(rows, device=q.device, dtype=torch.int32) * 7).clamp(
            max=keys
        )
        starts = torch.zeros_like(lengths)
        # Include the uncommon low UE8M0 exponents and a partial K tile.
        kv[1][::23] = 0x01000100
        dense = self._dense(q, kv, weights, starts, lengths, 272)
        out = torch.empty((rows, 34), dtype=torch.float32, device=q.device)
        # Force one shard so every CTA can reuse its K fragments across Q tiles.
        self.fused._jit_fused_fp4_mqa().block_max(out, q, *kv, weights, lengths, 1)
        valid = torch.arange(34, device=q.device)[None, :] * 8 < lengths[:, None]
        expected = dense.reshape(rows, 34, 8).amax(dim=-1)
        torch.testing.assert_close(
            out[valid].view(torch.int32),
            expected[valid].view(torch.int32),
            rtol=0,
            atol=0,
        )

        # Both launch sizes revisit a row past its first 256 columns. The
        # strided block table includes a partially visible block and padding.
        ids = torch.arange(1, 36, dtype=torch.int32, device=q.device)
        ids[33:] = -1
        blocks = ids.expand(rows, -1).contiguous()[:, :34]
        cols = (blocks[:, :, None] * 8 + torch.arange(8, device=q.device)).flatten(1)
        valid = (cols >= 0) & (cols < lengths[:, None])
        expected = dense.gather(1, cols.clamp_min(0))
        for count in (65, rows):
            with self.subTest(query_rows=count):
                compact = self.fused._score_shortlist(
                    q[:count], kv, weights[:count], lengths[:count], blocks[:count]
                )
                torch.testing.assert_close(
                    compact[valid[:count]].view(torch.int32),
                    expected[:count][valid[:count]].view(torch.int32),
                    rtol=0,
                    atol=0,
                )

    def test_ragged_cp_offsets_and_empty_requests(self):
        from sglang.kernels.ops.attention.dsv4.lightop_indexer import (
            candidate_block_mask,
        )
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2

        # The 11-key request has no queries on this rank. The live request
        # after it starts at a packed-K offset that is not a block boundary.
        contexts = [11, 0, 4109, 32785]
        per_request = [
            [],
            [0, 0],
            [0, 1, 7, 8, 9, 511, 512, 513, 4095, 4096, 4097, 4109],
            list(range(32785 - 32 * 8, 32785, 8)) + [32785],
        ]
        lengths = torch.tensor(
            [n for req in per_request for n in req], dtype=torch.int32, device="cuda"
        )
        starts = torch.tensor(
            [sum(contexts[:i]) for i, req in enumerate(per_request) for _ in req],
            dtype=torch.int32,
            device="cuda",
        )
        q, kv, weights = _inputs(lengths.numel(), sum(contexts))
        dense = self._dense(q, kv, weights, starts, lengths, 32788)
        expected_tokens = torch.empty(
            (q.shape[0], 512), dtype=torch.int32, device=q.device
        )
        topk_transform_ragged_v2(
            dense, lengths, out_offsets=starts, out_indices=expected_tokens
        )
        row_start = key_start = 0
        for length, req in zip(contexts, per_request):
            with self.subTest(keys=length, rows=len(req)):
                rows = slice(row_start, row_start + len(req))
                keys = slice(key_start, key_start + length)
                tokens, source = self.fused.fused_fp4_mqa_topk(
                    q[rows],
                    (kv[0][keys], kv[1][keys]),
                    weights[rows],
                    lengths[rows],
                    offsets=starts[rows],
                    publish_blocks=True,
                )
                self.assertTrue(
                    torch.equal(
                        tokens.sort().values, expected_tokens[rows].sort().values
                    )
                )
                if length and req:
                    _, ref = candidate_block_mask(
                        dense[rows, :length], lengths[rows], 2048
                    )
                    self.assertTrue(torch.equal(source, ref))
                else:
                    self.assertTrue((source == -1).all().item())
            row_start += len(req)
            key_start += length

    def test_cutoff_ties_and_forced_newest(self):
        from sglang.kernels.ops.attention.dsv4.lightop_indexer import (
            candidate_block_mask,
        )
        from sglang.kernels.ops.attention.dsv4.topk import topk_transform_ragged_v2

        keys = 32773
        q, kv, weights = _inputs(3, keys)
        # The first row ties at zero. The other rows exercise a newest
        # block that is strictly worse and one that ties at the cutoff.
        q.fill_(1)
        q[0].zero_()
        weights.fill_(0.25)
        kv[0].fill_(0x11)
        kv[0][32760:].zero_()
        kv[1].fill_(0x7F7F7F7F)
        lengths = torch.tensor([keys, keys, 16385], dtype=torch.int32, device=q.device)
        starts = torch.zeros_like(lengths)
        dense = self._dense(q, kv, weights, starts, lengths, 32776)
        expected_tokens = torch.empty((3, 512), dtype=torch.int32, device=q.device)
        topk_transform_ragged_v2(
            dense, lengths, out_offsets=starts, out_indices=expected_tokens
        )
        _, expected_blocks = candidate_block_mask(dense, lengths, 2048)
        for publish_blocks in (False, True):
            with self.subTest(source=publish_blocks):
                tokens, blocks = self.fused.fused_fp4_mqa_topk(
                    q, kv, weights, lengths, publish_blocks=publish_blocks
                )
                self.assertTrue(
                    torch.equal(tokens.sort().values, expected_tokens.sort().values)
                )
                if publish_blocks:
                    self.assertTrue(torch.equal(blocks, expected_blocks))
                    self.assertTrue(torch.equal(blocks[:, -1], (lengths - 1) // 8))
                else:
                    self.assertIsNone(blocks)


if __name__ == "__main__":
    unittest.main()
