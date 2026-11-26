# BW1000 DeepSeek-V4.1-Flash INT8 GEMM

Roadmap #25 reports 1,370.4 ms of dense and projection INT8 kernels across 13
chunks, or 15.9% of the summed kernel times. The sweep measures single-rank
projection costs for that workload. [Roadmap #25](https://github.com/IIC-SIG-MLsys/sglang-das/issues/25)

## Run on DCU-2 GPU 7

Run from the checkout root in the DTK serving environment.

```sh
env -u ROCR_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES HIP_VISIBLE_DEVICES=7 PD_VENDOR_TUNING=0 python3 benchmark/kernels/quantization/tune_bw1000_dsv41_int8.py --device 0 --output-dir bw1000-int8-gpu7
```

The sweep uses the installed ROCm PyTorch, Triton, LightOp, and `sgl_kernel`
plus the DTK compiler and library already on the host. It uses synthetic
tensors and builds a small host-only hipBLASLt probe at startup. It can sweep
zero-workspace ranks through the existing wheel if the compiler is
unavailable. The default matrix includes serving and TP8-only controls. Use
`--deployment-only`, `--phase prefill`, or `--projection wo_a` to focus a run.
Use `--list-shapes` to print the exact CSV manifest with only Python installed.
`--model-config` and `--draft-config` read local checkpoint geometry. Use
`--verify-width` for an explicit `--speculative-num-draft-tokens` setting and
`--unfused-qkv` if the server disables Q/KV fusion. The sweep inherits
`W8A8_SUPPORT_METHODS` for the dense baseline and accepts `--w8a8-method 1` or
`3` to select it directly.

## Projection audit

Attention TP is `TP / (attention DP × attention CP)`, which is 1 for both
TP8/CP8 prefill and TP8/DP8 decode. A 32,768-row CP8 chunk therefore runs the
main attention projections with 4,096 token rows and the full attention
head width on each rank. With attention TP8 and CP/DP off, Q-B shards 64 heads
to eight and O-A shards eight output groups to one. Q-A, KV-A, the indexer
Q projection, and Engram WKV are replicated linears. RoPE rotates 64
channels within each 512-channel head and has no separate projection.
[Parallel width derivation](../../../python/sglang/srt/runtime_context.py) ·
[Attention construction and CP split](../../../python/sglang/srt/models/deepseek_v4.py)

The table uses logical output width `N`. O-A performs `G` strided-batched
GEMMs of `(M, N/G, K)` with a distinct activation and weight operand for
each group. Its rank key uses `N/G` and `G`.

| Projection | CP8 / DP8 `(N,K)` per rank | Pure TP8 `(N,K)` per rank | Runtime and built-in coverage |
| --- | ---: | ---: | --- |
| Target fused Q-A + KV-A `wqkv_a` | 1,792 × 5,120 | same | Compressed W8A8. PR #14 small-`M` table covers it. |
| Unfused `wq_a` and `wkv` | 1,280 × 5,120 and 512 × 5,120 | same | Compressed W8A8 when the checkpoint supplies the split INT8 targets. Draft always constructs separate linears. PR #14 uses its generic fallback for these widths. |
| Target and draft `wq_b` | 32,768 × 1,280 | 4,096 × 1,280 | Compressed W8A8. PR #14 covers the TP8 shape. Its 32,768 entry has `K=5120`. |
| Target `wo_a` | 8,192 × 4,096, `G=8` | 1,024 × 4,096, `G=1` | `dsv41_int8`. The eight-group entry covers prefill M=512, 1024, 2048, 4096, 8192 and buckets up to 8192. M=16384, 32768 and the one-group geometry use rank 0. |
| Target `wo_b` | 5,120 × 8,192 | 5,120 × 1,024 | Compressed W8A8. PR #14 covers the full-width case for M ≤ 64. |
| Shared expert `gate_up_proj` / `down_proj` | 4,608 × 5,120 / 5,120 × 2,304 | 576 × 5,120 / 5,120 × 288 | Compressed W8A8. PR #14 covers both replicated pairs at M ≤ 64. The TP8 control shards the 2,304 intermediate columns in eight. |
| `indexer.wq_b` | 4,096 × 1,280 | same | Compressed W8A8 on layers 2, 8, 14, 20, 24, 28, 32, 36. PR #14 covers this geometry. |
| `engram.wkv` | 25,600 × 6,144 | same | Compressed W8A8 on layers 1 and 14. `N=(4+1)×5120`, `K=(4-1)×8×256`. PR #14 covers this geometry. |
| Draft `wo_a` / `wo_b` for rank 512 | 4,096 × 4,096, `G=8` / 5,120 × 4,096 | 512 × 4,096, `G=1` / 5,120 × 512 | `wo_a` eight-group static buckets end at M=4096. `wo_b` uses compressed W8A8. Use `--draft-o-rank 1024` for the shared-geometry draft. |
| Target and draft LM head | 129,280 × 5,120 with DP LM head | 16,160 × 5,120 with TP LM head | `dsv41_int8` in the INT8-head checkpoint. Target TP LM head under DP may gather up to eight ranks' token rows before this GEMM. Draft base logits use the sampler's proposal rows. |
| Draft `markov_w2` | 129,280 × 256 | 16,160 × 256 | `dsv41_int8` in the INT8-head checkpoint. Runs on `B` rows once per proposal step. |

The Flash decoder constructs a MoE at every layer. Layers 0 and 1 are the
first two SWA layers, so their dense FFN GEMMs are the shared expert pair above.
With a MoE all-to-all backend such as `megamoe`, the shared expert is
replicated. A shared expert folded into the routed kernel has no standalone
GEMM. The INT4 routed experts belong to the W4A8 bucket. The BF16/FP32
MoE router and mHC gates use floating-point GEMMs, and text requests skip
the vision tower.
[Model geometry](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/config.json) ·
[Shared expert construction](../../../python/sglang/srt/models/deepseek_v2.py)

The public Hygon W4A8 config lists INT8 main/draft Q and KV projections, output
B, shared experts, indexer Q, and Engram WKV. Its ignore list includes O-A, LM
head, Markov, and draft `main_proj`. PR #6 measured a W4A8 deployment with
INT8 O-A and heads. The manifest retains those measured deployment paths and
includes optional draft `main_proj` geometry `(5120, 15360)` for an INT8
draft variant. The public config runs the draft feature projection in floating
point. [Hygon checkpoint config](https://www.modelscope.cn/models/hygon/DeepSeek-V4.1-Flash-Channel-INT4-w4a8/file/view/master/config.json) ·
[PR #6 operator measurements](https://github.com/IIC-SIG-MLsys/sglang-das/pull/6)

- **Prefill M.** The manifest includes 512, 1024, 2048, 4096, 8192, 16384, and
  32768. The 400 Ki-token estimate uses twelve 4,096-row rank-local chunks and
  a 2,048-row tail. Full-row cases cover non-CP execution. The low-ratio
  compressor really does gather the full 32,768 rows during CP, but its
  BF16 projections have `(N,K)=(1024,5120)` at ratio 2 and `(512,5120)` at
  ratio 1. The indexer Q projection runs locally, including at live-row sizes
  inside the attention backend. Indexer `weights_proj` (32 × 5,120) and
  `wk` (128 × 512) are BF16. Ordinary prefill logits select the last token
  per request. Input logprobs and other requested logits can add rows.
  [Attention backend](../../../python/sglang/srt/layers/attention/deepseek_v4_backend.py) ·
  [Compressor and indexer weights](../../../python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py) ·
  [Logits pruning and gather](../../../python/sglang/srt/layers/logits_processor.py)
- **Decode M.** For every rank-local `B=1…64`, the manifest includes decode
  rows `B` and the requested five-row blocks `5B`, including 5, 10, 20, 40,
  80, 160, and 320. The code defaults to `gamma=5` proposal rows and
  `gamma+1=6` target verify rows, so the suite also includes `6B`. Explicit
  verify width 5 uses four proposed rows. A padded graph or ragged verify
  can change the realized row count. The draft also projects accepted
  context rows into its SWA cache, using its INT8 `wkv` per stage.
  [DSPARK runtime widths](../../../python/sglang/srt/speculative/dspark_components/dspark_config.py) ·
  [Draft context and Markov dispatch](../../../python/sglang/srt/models/deepseek_v4_dspark.py)
- **Existing rank coverage.** The target eight-group O-A and full LM head
  have exact entries at 5, 10, 20, and 40. Their M=80/160 entries come from
  the 128/256 buckets. At M=320, O-A has a 512 bucket and the full LM head
  falls back to rank 0. The TP8 LM head and both Markov variants cover M=5
  and use default rank 0 at 10, 20, and 40. Their 128/256 buckets cover
  80/160. Only the TP8 LM head has a 512 bucket. Draft eight-group O-A
  misses exact M=5, 10, 20, and 40, and uses its 128/256/512 buckets at
  80/160/320. Regular W8A8 shapes use PR #14's small-M LightOp/ASM table,
  the generic small-M ASM fallback, or LightOp Triton. W8A8 method 3 uses
  the legacy LightOp hipBLASLt path on gfx936.
  [Unpacked ranks and dispatch](../../../python/sglang/kernels/ops/gemm/dsv41_int8.py) ·
  [PR #14](https://github.com/IIC-SIG-MLsys/sglang-das/pull/14)
- **Epilogues and quantization.** Unpacked hipBLASLt writes INT32 and one
  epilogue multiplies both FP32 scales before casting. Triton handles
  outputs at or above 4,194,304 elements. PR #22's BF16 quantizer requires
  33,554,432 input elements, so full-width O-A reaches it at M=1024
  and the M=512 case uses native HIP. The HCU O-A producer can emit
  token-major rows directly. Group-major conversion and genuinely strided
  inputs still require a copy. Main Q-B can consume the fused RMSNorm A8
  operand, and shared down can consume the fused SwiGLU A8 operand. On
  HCU, the indexer Q projection gets the BF16 Q-LoRA tensor and quantizes it
  separately. O-A writes BF16 before O-B's token quantization. Keeping that
  rounding point matters if those two stages are fused later.
  [HCU attention and FFN operand selection](../../../python/sglang/srt/models/deepseek_common/amd/deepseek_v4_hip.py) ·
  [HIP and Triton projection kernels](../../../python/sglang/kernels/ops/gemm/dsv41_int8.py)

## Results and deployment

- `shapes.csv` compares the installed default and fastest verified choice
  per geometry. It reports quantization separately. Each GEMM timing
  includes the applicable scale epilogue. The estimate multiplies
  the exported improvement by per-rank call counts (40 target layers,
  eight indexer sources, and two Engram layers for the default manifest).
  It estimates projection time before stream overlap or communication.
  Packed service mode, LM-head VP, early layer skipping, and variant heads
  have their own call counts or descriptors.
- `candidates.csv` includes failed and unsupported candidates. Native
  ranks use the same TN descriptors and fixed 64-entry heuristic request
  as serving. The sweep tests returned ranks at 0, 4, and 32 MiB caps.
  Identical hipBLASLt algorithm bytes with the same scratch requirement
  share their measured result. Reported latency is the median per-call
  HIP event time across repeated graph replays after warmup. Each native
  result is checked against exact INT32 sums formed from FP32 K-block
  products, and checked again after graph replay. Verification uses every
  output column of up to 64 sampled rows. LightOp BF16 candidates use
  a BF16 tolerance. LightOp is probed on supported single-group layouts.
  [hipBLASLt heuristic workspace fields](https://rocm.docs.amd.com/projects/hipBLASLt/en/docs-6.1.0/datatypes.html)
- Point `SGLANG_DSV41_INT8_BLASLT_ALGO_CONFIG` at `wo_a_algo.json` for the
  unpacked route. Historical integer and decimal-string rank entries still
  load. The script exports measured improvements only. Nonzero workspace
  entries need the updated `common_ops` wheel for the serving process.
  With the original wheel or an unavailable rank the request falls back
  to the normal dispatch. Tuned W8A8 entries use the exact method and
  geometry on gfx936 and reuse the already quantized operand. Both the
  compressed-tensors channelwise scheme and the symmetric `w8a8_int8`
  linear can consume them.
- `gfx936_static_ranks.py` is a mergeable update for the built-in
  zero-workspace table. It retains the existing bucket maps and adds
  measured exact M values. A rejection at one M does not invalidate
  another M in the same bucket. Keep results with the same DTK, HIP,
  PyTorch, and CU configuration because heuristic rank ordering depends
  on the installed library and workspace preference.

## Measured BW1000 (`gfx936`, DTK 26.04.1) sweep highlights

A full 2,478-shape deployment sweep on a dedicated Hygon BW1000 GPU (`PD_VENDOR_TUNING=0`,
bitwise INT32 verified before and after HIP graph replay) produced the built-in
zero-workspace static ranks baked into `_DEFAULT_GFX936_UNPACKED_ALGO_RANKS`:

| Projection | Shape `(G, N/G, K)` | `M` | Previous default | Baked zero-workspace winner | Speedup / 400K impact |
| --- | --- | ---: | --- | --- | --- |
| Target `wo_a` | `(8, 1024, 4096)` | `4096` | rank `1` (`1,356.1 µs`) | rank `33` (`986.3 µs`) | `1.37×` (`-177.51 ms` / 400K) |
| Target `wo_a` | `(8, 1024, 4096)` | `2048` | rank `2` (`792.5 µs`) | rank `42` (`562.3 µs`) | `1.41×` (`-9.21 ms` / 400K) |
| Target `wo_a` | `(8, 1024, 4096)` | `8192` | rank `1` (`2,650.5 µs`) | rank `33` (`1,876.2 µs`) | `1.41×` |
| Target `wo_a` | `(8, 1024, 4096)` | `16384` | rank `0` (`24,726.4 µs`) | rank `42` (`3,736.6 µs`) | `6.62×` |
| Target `wo_a` (decode) | `(8, 1024, 4096)` | `35–63` (uncovered) | rank `0` (`167–299 µs`) | rank `18–31` (`55–59 µs`) | `3.0×–5.1×` |
| Full `lm_head` (decode) | `(1, 129280, 5120)` | `258–384` | rank `0` (`14.3–20.3 ms`) | rank `31` (`1.49–1.59 ms`) | `9.7×–14.1×` |
| Full `lm_head` (decode) | `(1, 129280, 5120)` | `33–63` (uncovered) | rank `0` (`3.2–4.0 ms`) | rank `3–5` (`~0.85 ms`) | `~4.0×` |
| Draft `wo_a` | `(8, 512, 4096)` | `1–320` | rank `0/6/15/23` | exact ranks (`29–84 µs`) | up to `4.4×` |
| `markov_w2` | `(1, 129280/16160, 256)` | `1–64` | rank `0` on uncovered `M` | exact ranks (`11–110 µs`) | up to `2.9×` |
