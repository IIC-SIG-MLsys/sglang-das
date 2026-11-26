"""Do not pass padded EP/TP rows as requests to the native dense indexer."""

from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention import deepseek_v4_backend as backend
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")


def test_dense_indexer_excludes_execution_padding():
    for lengths in ([1], [2, 1]):
        live = sum(lengths)
        padded = 8
        pages = torch.zeros((padded, 2), dtype=torch.int32)
        raw = pages.clone()
        core = SimpleNamespace(
            sparse_page_indices=lambda ratio: pages,
            sparse_raw_indices=lambda ratio: raw,
        )
        indexer = SimpleNamespace(
            queries=lambda q, freqs: torch.zeros(
                (q.shape[0], 32, 128), dtype=torch.bfloat16
            ),
            head_weights=lambda x: torch.zeros((x.shape[0], 32)),
            is_candidate_source=False,
            uses_candidates=False,
            index_topk=2,
        )
        layer = SimpleNamespace(
            layer_id=20,
            compress_ratio=1,
            indexer=indexer,
            freqs_cis=torch.zeros((8, 64)),
        )
        batch = SimpleNamespace(
            seq_lens_cpu=lengths,
            req_pool_indices=torch.arange(len(lengths)),
            forward_mode=backend.ForwardMode.EXTEND,
        )

        def gather(layer_id, slots, *, use_hcu_fused_gather):
            assert not use_hcu_fused_gather
            assert slots.numel() == live
            return torch.zeros((live, 64), dtype=torch.int8), torch.zeros(
                live, dtype=torch.int32
            )

        obj = SimpleNamespace(
            token_to_kv_pool=SimpleNamespace(get_low_ratio_index_k_fp4=gather),
            dsv41_hcu_indexer_kernels=False,
            forward_metadata=SimpleNamespace(core_metadata=core),
            req_to_token=torch.arange(16).reshape(2, 8),
            _hcu_prefill_candidate_blocks=lambda *args: None,
        )

        def logits(q, k, weights, starts, ends, width, **kwargs):
            assert q[0].shape[0] == live
            assert weights.shape[0] == live
            assert starts.shape == ends.shape == (live,)
            assert bool(((starts >= 0) & (ends <= live)).all())
            return torch.zeros((live, width))

        def topk(scores, lens, *, out_offsets, out_indices):
            out_indices.fill_(-1)
            out_indices[:, 0] = out_offsets

        positions = torch.tensor(
            [p for n in lengths for p in range(n)] + [0] * (padded - live)
        )
        with (
            envs.SGLANG_HCU_OPT_DSV41_FUSED_FP4_MQA.override(False),
            patch.object(backend, "_is_hcu", True),
            patch.object(
                backend,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="prefill"),
            ),
            patch.dict(
                "sys.modules", {"lightop.attention": ModuleType("lightop.attention")}
            ),
            patch.object(
                backend, "_hcu_dense_fp4_mqa_logits", side_effect=logits
            ) as native,
            patch.object(backend, "topk_transform_ragged_v2", side_effect=topk),
        ):
            backend.DeepseekV4AttnBackend._low_ratio_index_topk_dense(
                obj,
                layer,
                torch.zeros((padded, 4)),
                torch.zeros((padded, 4)),
                positions,
                batch,
                torch.tensor(lengths),
                lengths,
            )
        native.assert_called_once()
        assert bool((pages[live:] == -1).all())
        assert bool((raw[live:] == -1).all())
        assert bool((pages[:live, 0] >= 0).all())
