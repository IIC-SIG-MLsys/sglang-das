"""CPU coverage for the HCU LayerSplit CP layout and projection payload."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend as backend
from sglang.srt.layers.attention.dsa import utils as dsa_utils
from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Compressor
from sglang.srt.layers.cp.interleave import InterleaveContextParallelMetadata
from sglang.srt.layers.utils.hcu_cp_utils import ContextParallelMetadata
from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_helpers import (
    can_optimize_hcu_cp_gather,
)
from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
    CpCacheLayerSplitDeepSeekV4TokenToKVPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


class TestDsv41CPCollectives(CustomTestCase):
    def setUp(self):
        self.pool = CpCacheLayerSplitDeepSeekV4TokenToKVPool.__new__(
            CpCacheLayerSplitDeepSeekV4TokenToKVPool
        )
        self.pool.cp_size = 8
        self.pool.request_window = None
        self.batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            attn_cp_metadata=ContextParallelMetadata(),
            encoder_swa_replay=False,
            extend_seq_lens_cpu=[13, 27],
            extend_seq_lens=torch.tensor([13, 27]),
            req_pool_indices=torch.tensor([7, 3]),
            positions=torch.cat((torch.arange(31, 44), torch.arange(91, 118))),
        )
        self.attn = backend.DeepseekV4AttnBackend.__new__(backend.DeepseekV4AttnBackend)
        self.attn.token_to_kv_pool = self.pool
        self.attn.forward_metadata = backend.DSV4Metadata(None, None)
        self.enterContext(
            patch.object(
                dsa_utils, "is_dsa_prefill_cp_round_robin_split", return_value=True
            )
        )
        self.enterContext(
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
        )

    def test_swa_store_preserves_slots_and_rope_positions(self):
        rows = self.batch.positions.numel()
        loc = (torch.arange(rows, dtype=torch.int32) * 17) % rows
        loc[5] = -1
        self.batch.out_cache_loc = loc
        self.attn.forward_metadata.core_attn_metadata = SimpleNamespace(
            swa_out_cache_loc=loc
        )

        payload = torch.arange(rows * 8, dtype=torch.int16).view(rows, 8)
        rank_major = torch.cat([payload[r::8] for r in range(8)])

        def scatter(values, slots, positions):
            # Include the RoPE positions in the output so a location-only
            # permutation cannot accidentally pass this check.
            live = slots >= 0
            cache = values.new_full((rows, 9), -1)
            cache[slots[live].long(), :8] = values[live]
            cache[slots[live].long(), 8] = positions[live].to(values.dtype)
            return cache

        slots, positions = self.attn.get_cp_rank_major_swa_store(self.batch, rows // 8)
        self.assertTrue(
            torch.equal(
                scatter(rank_major, slots, positions),
                scatter(payload, loc, self.batch.positions),
            )
        )
        self.assertIs(
            slots, self.attn.get_cp_rank_major_swa_store(self.batch, rows // 8)[0]
        )

    def test_attention_passes_rank_order_payload_and_metadata_to_store(self):
        from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import env_gate
        from sglang.srt.models import deepseek_v4

        slots = torch.arange(1, 41, dtype=torch.int32)
        self.batch.out_cache_loc = slots
        self.attn.forward_metadata.core_attn_metadata = SimpleNamespace(
            swa_out_cache_loc=slots
        )
        payload = torch.arange(40 * 8, dtype=torch.bfloat16).view(40, 8)
        rank_major = torch.cat([payload[r::8] for r in range(8)])
        local = payload[3::8].contiguous()
        qkv = torch.cat([local, local], dim=1)
        layer = SimpleNamespace(
            fuse_wqa_wkv=True,
            q_lora_rank=8,
            wqkv_a=lambda x: (qkv, None),
            _normalize_q_lora=lambda q: (q, q),
            _compute_q_b=lambda q, *_: q,
            is_dsv41=True,
            layer_id=0,
            kv_norm=SimpleNamespace(weight=torch.ones(8)),
            eps=1e-6,
            freqs_cis=None,
            compress_ratio=0,
            compressor=None,
            indexer=None,
        )
        store = Mock()
        with (
            envs.SGLANG_OPT_DSV41_CP_SWA_RANK_MAJOR_STORE.override(True),
            patch.object(deepseek_v4, "_is_hcu", True),
            patch.object(deepseek_v4, "_is_hip", False),
            patch.object(deepseek_v4, "dsa_use_prefill_cp", return_value=True),
            patch.object(deepseek_v4, "get_token_to_kv_pool", return_value=self.pool),
            patch.object(env_gate, "is_unified_kv_triton", return_value=False),
            patch.object(env_gate, "is_unified_kv_fp8", return_value=False),
            patch.object(self.pool, "set_swa_key_buffer_radix_fused_norm_rope", store),
            patch.object(
                deepseek_v4, "attn_cp_interleave_gather", return_value=rank_major
            ) as gather,
        ):
            _, kv = deepseek_v4.MQALayer._forward_prepare(
                layer, local, self.batch.positions[3::8], self.batch, self.attn
            )
        self.assertIsNone(kv)
        gather.assert_called_once()
        self.assertTrue(torch.equal(gather.call_args.args[0], local))
        store.assert_called_once()
        values = store.call_args.kwargs
        self.assertIs(values["kv"], rank_major)
        self.assertTrue(torch.equal(values["swa_loc"], slots.view(-1, 8).T.flatten()))
        self.assertTrue(
            torch.equal(
                values["positions"], self.batch.positions.view(-1, 8).T.flatten()
            )
        )

    def test_layout_and_execution_mode_fallbacks(self):
        self.assertTrue(can_optimize_hcu_cp_gather(self.pool, self.batch, 5))
        self.assertFalse(can_optimize_hcu_cp_gather(self.pool, self.batch, 6))
        for field, value in (
            ("encoder_swa_replay", True),
            ("forward_mode", ForwardMode.TARGET_VERIFY),
            (
                "attn_cp_metadata",
                InterleaveContextParallelMetadata(gather_index=torch.tensor([0, 2])),
            ),
        ):
            with self.subTest(field=field), patch.object(self.batch, field, value):
                self.assertFalse(can_optimize_hcu_cp_gather(self.pool, self.batch, 5))
        with patch.object(self.attn.forward_metadata, "late_layer_tail", object()):
            self.assertIsNone(self.attn.get_cp_rank_major_swa_store(self.batch, 5))

        with envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.override(True):
            self.assertFalse(can_optimize_hcu_cp_gather(self.pool, self.batch, 5))
        for name in (
            "torch.cuda.is_current_stream_capturing",
            "sglang.srt.batch_invariant_ops.is_batch_invariant_mode_enabled",
            "sglang.srt.model_executor.runner_utils.capture_mode.get_is_capture_mode",
        ):
            with self.subTest(name=name), patch(name, return_value=True):
                self.assertFalse(can_optimize_hcu_cp_gather(self.pool, self.batch, 5))

    @torch.inference_mode()
    def test_compressor_gathers_one_projected_payload_in_token_order(self):
        gen = torch.Generator().manual_seed(721)
        x = torch.randint(-7, 8, (40, 32), generator=gen).to(torch.bfloat16)
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                compressor = DeepseekV41Compressor(
                    32, 16, ratio, 1e-6, fused_compress=True
                )
                weight = (compressor.wkv if ratio == 1 else compressor.wkv_gate).weight
                weight.copy_(torch.randint(-7, 8, weight.shape, generator=gen))
                expected = compressor.project(x)
                shards = [x[r::8].contiguous() for r in range(8)]
                sends, writes = [], []

                def payload(shard):
                    if ratio == 1:
                        return compressor.project(shard)[0]
                    return compressor.project_fused(shard)

                def gather(local):
                    sends.append(tuple(local.shape))
                    self.assertTrue(torch.equal(local, payload(shards[3])))
                    return torch.cat([payload(shard) for shard in shards])

                def compress(layer, hidden, req, pos, *, projected, **_):
                    writes.append((projected, req, pos))

                with (
                    envs.SGLANG_OPT_DSV41_CP_LOCAL_COMPRESSOR_PROJECTION.override(True),
                    get_parallel().override(attn_cp_size=8, attn_cp_rank=3),
                    patch.object(backend, "_is_hcu", True),
                    patch.object(backend, "attn_cp_interleave_gather", gather),
                    patch.object(self.attn, "_low_ratio_compress_torch", compress),
                ):
                    self.attn._forward_low_ratio_sources_cp(
                        layer=SimpleNamespace(
                            compress_ratio=ratio, compressor=compressor
                        ),
                        x=shards[3],
                        q_lora=None,
                        positions=self.batch.positions[3::8],
                        forward_batch=self.batch,
                        run_compressor=True,
                        run_indexer=False,
                    )
                [((kv, score), req, pos)] = writes
                self.assertEqual(sends, [(5, 16 * ratio)])
                self.assertEqual(req.tolist(), [7] * 13 + [3] * 27)
                self.assertTrue(torch.equal(pos, self.batch.positions))
                self.assertTrue(
                    torch.equal(kv.view(torch.uint8), expected[0].view(torch.uint8))
                )
                if ratio == 1:
                    self.assertIsNone(score)
                else:
                    self.assertTrue(
                        torch.equal(
                            score.view(torch.uint8), expected[1].view(torch.uint8)
                        )
                    )


if __name__ == "__main__":
    unittest.main()
