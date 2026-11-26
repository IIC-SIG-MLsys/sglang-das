"""DeepSeek V4.1 wrappers around the LayerSplit pool API."""

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_pool import (
    CpCacheLayerSplitDeepSeekV4TokenToKVPool,
)
from sglang.srt.model_executor.forward_context import get_attn_backend


def is_cp_cache_layer_split_deepseek_v4_pool(pool) -> bool:
    return isinstance(pool, CpCacheLayerSplitDeepSeekV4TokenToKVPool)


def can_optimize_hcu_cp_gather(pool, forward_batch, local_rows: int) -> bool:
    from sglang.srt.layers.attention.dsa.utils import (
        is_dsa_prefill_cp_round_robin_split,
    )
    from sglang.srt.layers.utils.hcu_cp_utils import ContextParallelMetadata
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    # All CP ranks derive eligibility from the same request lengths and the
    # equal physical shard size. Padded, indexed, and replay rows keep the
    # original path, which owns their distinct ordering and write masks.
    metadata = forward_batch.attn_cp_metadata
    if not (
        is_cp_cache_layer_split_deepseek_v4_pool(pool)
        and forward_batch.forward_mode == ForwardMode.EXTEND
        and isinstance(metadata, ContextParallelMetadata)
        and getattr(metadata, "gather_index", None) is None
        and is_dsa_prefill_cp_round_robin_split()
        and not forward_batch.encoder_swa_replay
        and forward_batch.extend_seq_lens_cpu is not None
        and local_rows > 0
        and local_rows * pool.cp_size == sum(forward_batch.extend_seq_lens_cpu)
        and not envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()
    ):
        return False

    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

    return (
        not is_batch_invariant_mode_enabled()
        and not get_is_capture_mode()
        and not torch.cuda.is_current_stream_capturing()
    )


def _should_sync_cp_cache_layer_split(pool, forward_batch) -> bool:
    if not is_cp_cache_layer_split_deepseek_v4_pool(pool):
        return False
    # Even short extend/warmup batches without attn_cp_metadata still read the
    # layer-sharded cache. All CP ranks participate in the same extend step.
    return (
        forward_batch is not None
        and forward_batch.forward_mode.is_context_parallel_extend()
    )


def maybe_prepare_cp_cache_layer_split_forward(pool, forward_batch) -> None:
    if is_cp_cache_layer_split_deepseek_v4_pool(pool):
        pool.prepare_forward(
            require_prefetched_reads=forward_batch.forward_mode.is_context_parallel_extend()
        )


def _get_core_attn_metadata():
    attn_backend = get_attn_backend()
    metadata = getattr(attn_backend, "forward_metadata", None)
    core_metadata = getattr(metadata, "core_attn_metadata", None)
    if core_metadata is None:
        raise RuntimeError("CP Cache LayerSplit requires DSV4 core attention metadata")
    return core_metadata


def maybe_prefetch_cp_kv_swa(pool, layer_id: int, forward_batch=None) -> None:
    if _should_sync_cp_cache_layer_split(pool, forward_batch):
        metadata = get_attn_backend().forward_metadata
        cache = getattr(metadata, "sparse_prefill_cache", None)
        read_token_capacity = None
        num_reqs = cache.num_reqs if cache is not None else None
        if cache is None:
            tail = metadata.late_layer_tail
            if tail is not None:
                # Replay floors exclude SWA history before each request's tail.
                extend_lens = tail.extend_seq_lens_cpu
                read_token_capacity = sum(extend_lens)
            else:
                # Before the first sparse attention builds its cache, bound the
                # full CP union, including the history before this extend chunk.
                extend_lens = forward_batch.extend_seq_lens_cpu
                read_token_capacity = sum(
                    min(seq_len, extend_len + pool.sliding_window - 1)
                    for seq_len, extend_len in zip(
                        forward_batch.seq_lens_cpu.tolist(), extend_lens, strict=True
                    )
                )
            num_reqs = len(extend_lens)
        # Sparse prefill dequantizes the full gathered SWA span, not only the
        # per-query attention indices.
        pool.prefetch_swa_layer(
            layer_id,
            _get_core_attn_metadata().swa_page_indices,
            read_indices=cache.swa_token_ids if cache is not None else None,
            num_reqs=num_reqs,
            read_token_capacity=read_token_capacity,
        )


def maybe_wait_cp_kv_swa_prefetch(pool, layer_id: int, forward_batch=None) -> None:
    if _should_sync_cp_cache_layer_split(pool, forward_batch):
        pool.wait_swa_prefetch(layer_id)


def maybe_prefetch_cp_kv_extra(pool, layer_id: int, forward_batch=None) -> None:
    if _should_sync_cp_cache_layer_split(pool, forward_batch):
        ratio = pool.compression_ratios[layer_id]
        if ratio in (1, 2):
            metadata = get_attn_backend().forward_metadata
            cache = getattr(metadata, "sparse_prefill_cache", None)
            core = _get_core_attn_metadata()
            read_indices = None
            read_token_capacity = None
            cache_full_read = False
            num_reqs = cache.num_reqs if cache is not None else None
            if cache is not None:
                # The workspace dequantizes every compressed position before
                # TopK selects rows; broadcasting only TopK pages is unsafe.
                read_indices = cache.ensure_compressed(
                    ratio, core.page_table, pool.kv_pools[ratio].page_size
                ).flat_token_ids
                # With explicit sparse prefill and nonempty original CP shards,
                # every rank builds the full-history gather before this source.
                # Tail metadata may later drop that cache, but not the KV data.
                cache_full_read = (
                    ratio in (1, 2)
                    and envs.SGLANG_OPT_FLASHMLA_SPARSE_PREFILL.get()
                    and min(forward_batch.extend_seq_lens_cpu, default=0) >= pool.cp_size
                )
            elif metadata.late_layer_tail is not None:
                # Tail queries can select keys anywhere in each full history.
                # Bound their page union without materializing a chunk cache.
                seq_lens = forward_batch.seq_lens_cpu.tolist()
                read_token_capacity = sum(
                    (seq_len + ratio - 1) // ratio for seq_len in seq_lens
                )
                num_reqs = len(seq_lens)
            pool.prefetch_low_ratio_extra_layer(
                layer_id,
                core.sparse_page_indices(ratio),
                read_indices=read_indices,
                num_reqs=num_reqs,
                read_token_capacity=read_token_capacity,
                cache_full_read=cache_full_read,
            )
