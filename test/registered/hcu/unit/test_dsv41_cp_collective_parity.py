"""BW1000 CP8 parity with RCCL, the real SWA store, and C1/C2 GEMMs."""

import os
import unittest
from contextlib import ExitStack
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from sglang.kernels.ops.attention.dsv4 import fused_k_norm_rope_flashmla
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.gemm import bf16_fp32
from sglang.srt.distributed.parallel_state import init_model_parallel_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend as backend
from sglang.srt.layers.attention.dsa import utils as dsa_utils
from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Compressor
from sglang.srt.layers.cp.interleave import attn_cp_interleave_gather
from sglang.srt.layers.cp.utils import cp_materialize_global_token_order
from sglang.srt.layers.utils.hcu_cp_utils import ContextParallelMetadata
from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
    CpCacheLayerSplitDeepSeekV4TokenToKVPool,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_hcu
from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.kernels.utils import multigpu_pytest_main
from sglang.test.test_utils import CustomTestCase

CP_SIZE = 8
PAGE_SIZE = 256

register_hcu_ci(est_time=90, suite="nightly-hcu-8", nightly=True)


class TestDsv41CPCollectiveParity(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if int(os.environ.get("WORLD_SIZE", "1")) != CP_SIZE:
            raise unittest.SkipTest("run the script with eight HCU devices")
        cls.rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(cls.rank)
        if not is_hcu() or not torch.cuda.get_device_properties(
            cls.rank
        ).gcnArchName.startswith("gfx936"):
            raise unittest.SkipTest("requires Hygon BW1000 (gfx936)")

        dist.init_process_group("nccl", timeout=timedelta(seconds=120))
        cls.addClassCleanup(dist.destroy_process_group)
        cls.cp_group = init_model_parallel_group(
            [list(range(CP_SIZE))],
            cls.rank,
            "nccl",
            use_pynccl=False,
            use_custom_allreduce=False,
            use_mscclpp=False,
            use_torch_symm_mem_allreduce=False,
            group_name="attn_cp",
        )
        cls.addClassCleanup(cls.cp_group.destroy)

    def setUp(self):
        self._stack = ExitStack()
        self.addCleanup(self._stack.close)
        self._stack.enter_context(
            get_parallel().override(
                attn_cp_size=CP_SIZE,
                attn_cp_rank=self.rank,
                attn_cp_group=self.cp_group,
            )
        )
        self._stack.enter_context(envs.SGLANG_ENABLE_CP_V2.override(False))
        self._stack.enter_context(
            patch.object(
                dsa_utils, "is_dsa_prefill_cp_round_robin_split", return_value=True
            )
        )
        self.gen = torch.Generator(device="cuda").manual_seed(21025)

    def _batch(self, rows):
        # The second request starts on a different CP rank, with its own RoPE origin.
        first = rows // 2 - 3
        positions = torch.cat(
            (
                torch.arange(32768, 32768 + first, device="cuda"),
                torch.arange(4096, 4096 + rows - first, device="cuda"),
            )
        )
        return SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            attn_cp_metadata=ContextParallelMetadata(),
            encoder_swa_replay=False,
            extend_seq_lens_cpu=[first, rows - first],
            positions=positions,
        )

    @torch.inference_mode()
    def test_rank_major_store_matches_legacy_cache_bytes(self):
        pool = CpCacheLayerSplitDeepSeekV4TokenToKVPool.__new__(
            CpCacheLayerSplitDeepSeekV4TokenToKVPool
        )
        pool.cp_size = CP_SIZE
        pool.request_window = None
        attn = backend.DeepseekV4AttnBackend.__new__(backend.DeepseekV4AttnBackend)
        attn.token_to_kv_pool = pool
        weight = torch.randn(
            512, device="cuda", generator=self.gen, dtype=torch.bfloat16
        )
        angles = torch.randn(65536, 32, device="cuda", generator=self.gen)
        freqs = torch.polar(torch.ones_like(angles), angles)

        for rows in (32768, 16384, 6784):
            with self.subTest(rows=rows):
                batch = self._batch(rows)
                loc = (
                    torch.arange(rows, device="cuda", dtype=torch.int32) * 17
                ) % rows + 1
                loc[::29] = -1
                batch.out_cache_loc = loc
                attn.forward_metadata = backend.DSV4Metadata(
                    SimpleNamespace(swa_out_cache_loc=loc), None
                )
                # Keep the WQKV column slice strided through the collective boundary.
                qkv = torch.randn(
                    rows // CP_SIZE,
                    1280 + 512,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=self.gen,
                )
                local_kv = qkv[:, 1280:] + self.rank
                local_kv = torch.cat((qkv[:, :1280], local_kv), dim=1)[:, 1280:]
                token_kv = cp_materialize_global_token_order(
                    local_kv.contiguous(), batch
                )
                rank_kv = attn_cp_interleave_gather(local_kv)
                rank_loc, rank_pos = attn.get_cp_rank_major_swa_store(
                    batch, rows // CP_SIZE
                )

                def store(kv, slots, positions):
                    cache = torch.zeros(
                        rows // PAGE_SIZE + 1,
                        KVLayout.V4.page_bytes(PAGE_SIZE),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    fused_k_norm_rope_flashmla(
                        kv,
                        weight,
                        1e-6,
                        freqs,
                        positions,
                        slots,
                        cache,
                        PAGE_SIZE,
                    )
                    return cache

                self.assertTrue(
                    torch.equal(
                        store(rank_kv, rank_loc, rank_pos),
                        store(token_kv, loc, batch.positions),
                    )
                )

    @torch.inference_mode()
    def test_local_projection_matches_global_gemm_bits(self):
        for rows in (32768, 16384, 6784):
            batch = self._batch(rows)
            local = torch.randn(
                rows // CP_SIZE,
                5120,
                device="cuda",
                dtype=torch.bfloat16,
                generator=self.gen,
            ).add_(self.rank / 16)
            global_x = cp_materialize_global_token_order(local, batch)
            for ratio in (1, 2):
                compressor = DeepseekV41Compressor(5120, 512, ratio, 1e-6).cuda()
                weight = (compressor.wkv if ratio == 1 else compressor.wkv_gate).weight
                weight.normal_(std=5120**-0.5, generator=self.gen)
                # rocBLAS NN can choose different tiles for M=848 and M=6784.
                # Rerun these same operands with NN off to isolate the collective.
                tail_c2 = rows == 6784 and ratio == 2
                for nn_enabled in (True, False) if tail_c2 else (True,):
                    with (
                        self.subTest(rows=rows, ratio=ratio, rocblas_nn=nn_enabled),
                        patch.object(bf16_fp32, "_HCU_ROCBLAS_NN_ENABLED", nn_enabled),
                    ):
                        expected = compressor.project(global_x)
                        actual = backend._gather_cp_compressor_projection(
                            compressor, local
                        )
                        for value, reference in zip(actual, expected, strict=True):
                            if reference is None:
                                self.assertIsNone(value)
                            elif tail_c2 and nn_enabled:
                                # NN returns BF16 rounded values widened to FP32.
                                # nextafter takes at most one BF16 step toward value;
                                # small atol covers K=5120 near-zero cancellation.
                                adjacent = torch.nextafter(
                                    reference.bfloat16(), value.bfloat16()
                                )
                                torch.testing.assert_close(
                                    value, adjacent.float(), rtol=0, atol=1e-4
                                )
                            else:
                                self.assertTrue(
                                    torch.equal(
                                        value.view(torch.uint8),
                                        reference.view(torch.uint8),
                                    )
                                )
            del global_x


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        unittest.main()
    else:
        multigpu_pytest_main(__name__, __file__, num_gpus=[CP_SIZE])
