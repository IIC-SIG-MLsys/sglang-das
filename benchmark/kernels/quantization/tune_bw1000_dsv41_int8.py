#!/usr/bin/env python3
"""Tune the DeepSeek-V4.1-Flash INT8 projections with the local BW1000 DTK.

--list-shapes needs only Python. Live runs use the serving environment and
synthetic tensors. See bw1000_dsv41_int8.md for the dispatch and workload audit.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import pprint
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FLASH = {
    "hidden_size": 5120,
    "num_hidden_layers": 40,
    "num_attention_heads": 64,
    "head_dim": 512,
    "q_lora_rank": 1280,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "moe_intermediate_size": 2304,
    "n_shared_experts": 1,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_source_layer_ids": (2, 8, 14, 20, 24, 28, 32, 36),
    "vocab_size": 129280,
    "hc_mult": 4,
    "engram_layer_ids": (1, 14),
    "engram_n_heads": 8,
    "engram_head_dim": 256,
    "engram_max_ngram_size": 4,
    "dspark_block_size": 5,
    "dspark_target_layer_ids": (37, 38, 39),
    "dspark_markov_rank": 256,
}
TP = 8
CHUNK_TOKENS = 32768
PREFILL_ROWS = (512, 1024, 2048, 4096, 8192, 16384, 32768)
MIB = 1024**2


@dataclass(frozen=True, order=True)
class Shape:
    m: int
    n: int  # per group
    k: int
    groups: int = 1
    route: str = "w8a8"
    dtype: str = "bfloat16"

    @property
    def total_n(self):
        return self.n * self.groups

    @property
    def key(self):
        return f"{self.m}:{self.groups}:{self.n}:{self.k}:0:0"


def model_geometry(config_path):
    raw = json.loads(config_path.read_text()) if config_path else {}
    config = raw | raw.get("text_config", {})
    dflash = raw.get("dflash_config") or {}
    dspark = dflash | (raw.get("dspark_config") or {})
    for name, nested in (
        ("block_size", dflash),
        ("target_layer_ids", dflash),
        ("markov_rank", dspark),
    ):
        prefixed = f"dspark_{name}"
        value = config.pop(prefixed, None)
        if value is None:
            value = nested.get(name, config.get(name))
        if value is not None:
            config[prefixed] = value
    return FLASH | config


def shape_suite(args):
    config = model_geometry(args.model_config)
    draft = model_geometry(args.draft_config) if args.draft_config else config
    hidden = config["hidden_size"]
    layers = config["num_hidden_layers"]
    vocab = config["vocab_size"]
    # ParallelLMHead aligns the vocabulary to 64 and, when needed, 64 * TP.
    full_vocab = (vocab + 63) // 64 * 64
    shard_vocab = (
        (vocab + 64 * TP - 1) // (64 * TP) * 64 if full_vocab % TP else full_vocab // TP
    )
    block = draft["dspark_block_size"]
    verify_width = args.verify_width or block + 1
    draft_width = verify_width - 1
    draft_attn_width = draft_width + (not draft.get("sample_from_anchor", True))
    draft_stages = len(draft["dspark_target_layer_ids"])
    draft_rank = args.draft_o_rank or (
        draft["o_lora_rank"] if args.draft_config else 512
    )
    target_dtype = "float32" if args.target_head_fp32 else "bfloat16"
    draft_dtype = "float32" if args.draft_head_fp32 else "bfloat16"
    markov_dtype = "float32" if args.markov_fp32 else "bfloat16"
    engram_k = (
        (config["engram_max_ngram_size"] - 1)
        * config["engram_n_heads"]
        * config["engram_head_dim"]
    )
    cases = defaultdict(set)
    contributions = defaultdict(Counter)

    def add(rows, role, n, k, *, groups=1, route="w8a8", dtype="bfloat16", calls=0):
        if args.projection and not any(term in role for term in args.projection):
            return
        for m in sorted(set(rows)):
            shape = Shape(m, n, k, groups, route, dtype)
            cases[shape].add(role)
            if calls and m in prefill_calls:
                contributions[shape]["calls_400k"] += calls * prefill_calls[m]
                if m == CHUNK_TOKENS // TP:
                    contributions[shape]["calls_32k"] += calls

    def attention(rows, dimensions, suffix, attn_tp, output_rank, *, fused, calls=0):
        h = dimensions["hidden_size"]
        head = dimensions["head_dim"]
        q_rank = dimensions["q_lora_rank"]
        heads = dimensions["num_attention_heads"]
        groups = dimensions["o_groups"]
        if fused:
            add(rows, "wqkv_a" + suffix, q_rank + head, h, calls=calls)
        else:
            add(rows, "wq_a" + suffix, q_rank, h, calls=calls)
            add(rows, "wkv" + suffix, head, h, calls=calls)
        add(rows, "wq_b" + suffix, heads // attn_tp * head, q_rank, calls=calls)
        add(
            rows,
            "wo_a" + suffix,
            output_rank,
            heads * head // groups,
            groups=groups // attn_tp,
            route="dsv41",
            calls=calls,
        )
        add(rows, "wo_b" + suffix, h, groups // attn_tp * output_rank, calls=calls)

    def shared(rows, dimensions, suffix, tp, calls=0):
        intermediate = (
            dimensions["moe_intermediate_size"] * dimensions["n_shared_experts"]
        )
        if intermediate:
            h = dimensions["hidden_size"]
            add(rows, "shared_gate_up" + suffix, 2 * intermediate // tp, h, calls=calls)
            add(rows, "shared_down" + suffix, h, intermediate // tp, calls=calls)

    chunks, tail = divmod(args.request_tokens, CHUNK_TOKENS)
    prefill_calls = Counter({CHUNK_TOKENS // TP: chunks})
    if tail:
        prefill_calls[(tail + TP - 1) // TP] += 1
    rows = set(args.prefill_rows) | prefill_calls.keys() | {CHUNK_TOKENS}
    if tail:
        rows.add(tail)
    if args.phase in ("all", "prefill"):
        attention(
            rows,
            config,
            "",
            1,
            config["o_lora_rank"],
            fused=not args.unfused_qkv,
            calls=layers,
        )
        if not args.unfused_qkv:
            # Alternate checkpoints/launches can disable the fused Q/KV load.
            add(rows, "wq_a (unfused)", config["q_lora_rank"], hidden)
            add(rows, "wkv (unfused)", config["head_dim"], hidden)
        shared(rows, config, "", 1, calls=layers)
        add(
            rows,
            "indexer.wq_b",
            config["index_n_heads"] * config["index_head_dim"],
            config["q_lora_rank"],
            calls=len(config["index_source_layer_ids"]),
        )
        if config["engram_layer_ids"]:
            add(
                rows,
                "engram.wkv",
                hidden * (config["hc_mult"] + 1),
                engram_k,
                calls=len(config["engram_layer_ids"]),
            )
        # Context import on the draft side can project the accepted tail. The
        # count depends on SWA replay and PD transfer, so it has no layer weight.
        context_rows = rows | {1, 128, config.get("sliding_window", 128)}
        add(
            context_rows,
            "draft.main_proj (context)",
            draft["hidden_size"],
            hidden * draft_stages,
        )
        add(
            context_rows, "draft.wkv (context)", draft["head_dim"], draft["hidden_size"]
        )
        for n in (full_vocab, shard_vocab):
            add(
                (1,),
                "target.lm_head (last token)",
                n,
                hidden,
                route="dsv41",
                dtype=target_dtype,
            )
        if not args.deployment_only:
            attention(
                rows,
                config,
                " (attention TP8)",
                TP,
                config["o_lora_rank"],
                fused=not args.unfused_qkv,
            )
            shared(rows, config, " (TP8)", TP)

    if args.phase in ("all", "decode"):
        batches = set(args.decode_batches)
        target_rows = batches | {
            m * width for m in batches for width in {block, verify_width}
        }
        draft_rows = {m * width for m in batches for width in {block, draft_attn_width}}
        attention(
            target_rows,
            config,
            " (target decode)",
            1,
            config["o_lora_rank"],
            fused=not args.unfused_qkv,
        )
        shared(target_rows, config, " (target decode)", 1)
        add(
            target_rows,
            "indexer.wq_b (decode)",
            config["index_n_heads"] * config["index_head_dim"],
            config["q_lora_rank"],
        )
        if config["engram_layer_ids"]:
            add(
                target_rows,
                "engram.wkv (decode)",
                hidden * (config["hc_mult"] + 1),
                engram_k,
            )
        attention(draft_rows, draft, " (draft)", 1, draft_rank, fused=False)
        shared(draft_rows, draft, " (draft)", 1)
        add(
            target_rows,
            "draft.main_proj (commit)",
            draft["hidden_size"],
            hidden * draft_stages,
        )
        add(target_rows, "draft.wkv (commit)", draft["head_dim"], draft["hidden_size"])
        if not args.deployment_only:
            attention(
                target_rows,
                config,
                " (attention TP8 decode)",
                TP,
                config["o_lora_rank"],
                fused=not args.unfused_qkv,
            )
            attention(
                draft_rows, draft, " (attention TP8 draft)", TP, draft_rank, fused=False
            )
            shared(target_rows | draft_rows, config, " (TP8 decode)", TP)
        for n in (full_vocab, shard_vocab):
            target_head_rows = (
                target_rows | {m * TP for m in target_rows}
                if n == shard_vocab
                else target_rows
            )
            add(
                target_head_rows,
                "target.lm_head (decode)",
                n,
                hidden,
                route="dsv41",
                dtype=target_dtype,
            )
            add(
                {m * draft_width for m in batches},
                "draft.lm_head",
                n,
                hidden,
                route="dsv41",
                dtype=draft_dtype,
            )
            if draft["dspark_markov_rank"]:
                add(
                    batches,
                    "draft.markov_w2 (per step)",
                    n,
                    draft["dspark_markov_rank"],
                    route="dsv41",
                    dtype=markov_dtype,
                )
    if not cases:
        raise ValueError("No shape matches the requested filters.")
    return cases, contributions


def integer_set(value):
    values = set()
    try:
        for part in value.split(","):
            if "-" in part:
                first, last = map(int, part.split("-"))
                values.update(range(first, last + 1))
            else:
                values.add(int(part))
        if not values or min(values) < 0:
            raise ValueError
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated integers or ranges"
        ) from exc
    return sorted(values)


def env_flag(name, default=""):
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("bw1000-int8-tuning"))
    parser.add_argument("--list-shapes", action="store_true")
    parser.add_argument(
        "--device", type=int, default=0, help="Logical HIP device index"
    )
    parser.add_argument("--phase", choices=("all", "prefill", "decode"), default="all")
    parser.add_argument(
        "--projection", action="append", help="Restrict projection names by substring"
    )
    parser.add_argument("--prefill-rows", type=integer_set, default=PREFILL_ROWS)
    parser.add_argument("--decode-batches", type=integer_set, default=range(1, 65))
    parser.add_argument("--request-tokens", type=int, default=400 * 1024)
    parser.add_argument(
        "--verify-width", type=int, help="Explicit --speculative-num-draft-tokens"
    )
    parser.add_argument(
        "--model-config", type=Path, help="Local target config.json for geometry"
    )
    parser.add_argument(
        "--draft-config", type=Path, help="Local draft config.json for geometry"
    )
    parser.add_argument(
        "--draft-o-rank",
        type=int,
        help="Override draft wo_a rank (default: 512 without a draft config)",
    )
    parser.add_argument("--target-head-fp32", action="store_true")
    parser.add_argument(
        "--draft-head-fp32",
        action="store_true",
        default=env_flag("SGLANG_DSPARK_FP32_LM_HEAD"),
    )
    parser.add_argument(
        "--markov-fp32",
        action="store_true",
        default=not env_flag("SGLANG_DSPARK_OPT_MARKOV_W2_BF16", "1"),
    )
    parser.add_argument(
        "--deployment-only",
        action="store_true",
        help="Omit non-CP attention-TP8 and TP8 shared expert controls",
    )
    parser.add_argument("--unfused-qkv", action="store_true")
    parser.add_argument(
        "--w8a8-method",
        type=int,
        choices=(1, 3),
        default=os.environ.get("W8A8_SUPPORT_METHODS", "1"),
    )
    parser.add_argument("--workspace-mib", type=integer_set, default=(0, 4, 32))
    parser.add_argument("--max-ranks", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--verify-rows", type=int, default=64)
    parser.add_argument(
        "--min-gain",
        type=float,
        default=0.03,
        help="Minimum fractional improvement for config export",
    )
    parser.add_argument(
        "--no-graph", action="store_true", help="Time eager launches with HIP events"
    )
    parser.add_argument("--skip-smooth-asm", action="store_true")
    parser.add_argument("--asm-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (
        min(
            *args.prefill_rows,
            *args.decode_batches,
            args.request_tokens,
            args.warmup,
            args.iters,
            args.repeats,
            args.verify_rows,
        )
        < 1
        or not 1 <= args.max_ranks <= 64
        or max(args.workspace_mib) >= 4096
        or not 0 <= args.min_gain < 1
        or (args.verify_width is not None and args.verify_width < 2)
        or (args.draft_o_rank is not None and args.draft_o_rank < 1)
    ):
        parser.error("Invalid row count, sweep limits, verify width, or minimum gain.")
    return args


def build_probe(output_dir):
    dtk = Path(os.environ.get("DTK_HOME", "/opt/dtk"))
    hipcc = shutil.which("hipcc")
    if hipcc is None:
        roots = (
            os.environ.get("ROCM_HOME"),
            os.environ.get("ROCM_PATH"),
            dtk,
            "/opt/rocm",
        )
        hipcc = next(
            (
                str(p)
                for root in roots
                if root and (p := Path(root) / "bin/hipcc").is_file()
            ),
            None,
        )
    if hipcc is None:
        print(
            "hipcc unavailable. Sweeping the installed wheel's zero-workspace TN ranks.",
            file=sys.stderr,
        )
        return None
    with tempfile.TemporaryDirectory(dir=output_dir) as scratch:
        library = Path(scratch) / "probe.so"
        include = ROOT / "python/sglang/kernels/aot/csrc/gemm"
        command = [
            hipcc,
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-shared",
            f"-I{include}",
            f"-I{dtk / 'include'}",
            f"-L{dtk / 'lib'}",
            f"-Wl,-rpath,{dtk / 'lib'}",
            str(HERE / "bw1000_hipblaslt_probe.cpp"),
            "-lhipblaslt",
            "-o",
            str(library),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
            return HipProbe(ctypes.CDLL(str(library)))
        except (subprocess.CalledProcessError, OSError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            print(
                f"hipBLASLt probe unavailable. Sweeping zero-workspace TN ranks.\n{detail}",
                file=sys.stderr,
            )
            return None


class HipProbe:
    def __init__(self, lib):
        self.lib = lib
        pointer = ctypes.c_void_p
        i64 = ctypes.c_int64
        signatures = (
            ("error", ctypes.c_char_p, ()),
            ("create", pointer, (i64, i64, i64, i64, ctypes.c_uint64)),
            ("destroy", None, (pointer,)),
            ("count", ctypes.c_int, (pointer,)),
            ("workspace_bytes", i64, (pointer, ctypes.c_int)),
            (
                "algo_data",
                pointer,
                (pointer, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)),
            ),
            (
                "run",
                ctypes.c_int,
                (
                    pointer,
                    ctypes.c_int,
                    pointer,
                    pointer,
                    pointer,
                    pointer,
                    ctypes.c_size_t,
                    pointer,
                ),
            ),
        )
        for name, result, parameters in signatures:
            function = getattr(lib, f"bw1000_{name}")
            function.restype = result
            function.argtypes = parameters

    @contextmanager
    def plan(self, shape, workspace_limit):
        handle = self.lib.bw1000_create(
            shape.m, shape.n, shape.k, shape.groups, workspace_limit
        )
        if not handle:
            raise RuntimeError(self.lib.bw1000_error().decode())
        try:
            yield handle
        finally:
            self.lib.bw1000_destroy(handle)

    def run(self, plan, rank, q, w, raw, workspace, required, stream):
        status = self.lib.bw1000_run(
            plan,
            rank,
            q.data_ptr(),
            w.data_ptr(),
            raw.data_ptr(),
            workspace.data_ptr() if required else None,
            required,
            stream,
        )
        if status != 0:
            raise RuntimeError(f"hipBLASLt matmul returned status {status}")

    def algorithm_bytes(self, plan, rank):
        size = ctypes.c_size_t()
        data = self.lib.bw1000_algo_data(plan, rank, ctypes.byref(size))
        return ctypes.string_at(data, size.value)


@dataclass(frozen=True)
class Trial:
    backend: str
    rank_or_tune: int | None = None
    limit_bytes: int = 0
    scratch_bytes: int = 0
    ms: float | None = None
    status: str = "ok"

    @property
    def label(self):
        if self.backend == "hipblaslt":
            return f"hipBLASLt rank {self.rank_or_tune}, limit {self.limit_bytes // MIB} MiB"
        if self.backend == "smooth_asm":
            return f"smooth ASM tune {self.rank_or_tune}"
        return self.backend

    def algorithm(self):
        rank = self.rank_or_tune
        return (
            {"rank": rank, "workspace_bytes": self.limit_bytes}
            if self.limit_bytes
            else rank
        )

    @property
    def timing(self):
        return (
            f"{self.label} {self.ms * 1000:.2f} us"
            if self.ms is not None
            else "unavailable"
        )


class Sweep:
    def __init__(
        self,
        args,
        torch,
        dsv,
        dense,
        dense_quantize,
        triton_cache,
        device,
        probe,
        asm_tunes,
    ):
        self.args = args
        self.torch = torch
        self.dsv = dsv
        self.ops = dsv.load_native()
        self.dense = dense
        self.dense_quantize = dense_quantize
        self.triton_cache = triton_cache
        self.device = device
        self.probe = probe
        self.asm_tunes = asm_tunes
        self.gen = torch.Generator(device=device).manual_seed(25)
        self.stream = torch.cuda.current_stream(device)
        self.method = args.w8a8_method

    @lru_cache(maxsize=None)
    def triton_configs(self, n, k):
        cache = self.triton_cache
        return cache.get_triton_cache(cache.get_w8a8json_name(n, k), n, k) or {}

    def time(self, run, verify=None):
        torch = self.torch
        for _ in range(self.args.warmup):
            run()
        self.stream.synchronize()

        def repeat():
            for _ in range(self.args.iters):
                out = run()
            return out

        launch = repeat
        if not self.args.no_graph:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.stream):
                captured = repeat()
            launch = graph.replay

        timings = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        for _ in range(self.args.repeats):
            start.record(self.stream)
            launch()
            end.record(self.stream)
            end.synchronize()
            timings.append(start.elapsed_time(end) / self.args.iters)
        if not self.args.no_graph and verify is not None:
            verify(captured)
        return statistics.median(timings)

    def reference(self, shape, q, weight, xs, ws, out_dtype):
        torch = self.torch
        # Cover both ends and interior rows. Use every
        # column of each sampled row, including vocabulary/shape tails.
        check_rows = min(self.args.verify_rows, shape.m)
        rows = torch.linspace(0, shape.m - 1, check_rows, device=self.device).long()
        if check_rows == 1:
            rows[0] = shape.m - 1
        q_sample = q.index_select(0, rows)
        # Split K so integer products and their FP32 partial sums remain exact.
        ref = torch.zeros(
            (check_rows, shape.groups, shape.n), dtype=torch.int32, device=self.device
        )
        for g in range(shape.groups):
            w_g = weight[g * shape.n : (g + 1) * shape.n]
            for start in range(0, shape.k, 1024):
                stop = start + 1024
                part = (
                    q_sample[:, g, start:stop].float() @ w_g[:, start:stop].float().t()
                )
                ref[:, g].add_(part.to(torch.int32))
        scaled = (
            (ref.float() * xs.view(shape.m, shape.groups, 1).index_select(0, rows))
            * ws.view(1, shape.groups, shape.n)
        ).to(out_dtype)
        return rows, ref, scaled

    def shape(self, shape):
        torch = self.torch
        m, n, k, groups = shape.m, shape.n, shape.k, shape.groups
        dsv_route = shape.route == "dsv41"
        out_dtype = getattr(torch, shape.dtype)
        x = torch.randn(
            (m, groups, k), dtype=torch.bfloat16, device=self.device, generator=self.gen
        )
        if out_dtype == torch.float32:
            x = x.float()
        w = torch.randint(
            -127,
            128,
            (groups * n, k),
            dtype=torch.int8,
            device=self.device,
            generator=self.gen,
        )
        ws = (
            torch.rand((groups * n, 1), device=self.device, generator=self.gen) + 0.1
        ) / 127

        def quantize(activation=x):
            if dsv_route:
                return self.dsv.quantize(activation)
            q, xs = self.dense_quantize(activation.view(m, k))
            return q.view(m, 1, k), xs

        q, xs = quantize()
        quant_ms = self.time(quantize)
        del quantize
        del x
        rows, raw_ref, ref = self.reference(shape, q, w, xs, ws, out_dtype)
        raw = torch.empty((m, groups, n), dtype=torch.int32, device=self.device)
        out = torch.empty_like(raw, dtype=out_dtype)
        aq = q.view(m, k) if groups == 1 else None
        light_weight = w.t() if groups == 1 and out_dtype == torch.bfloat16 else None
        trials = []

        def scale():
            self.dsv._scale_unpacked(self.ops, raw, xs, ws, out, groups, False)
            return out

        def check_run(run, trial, *, exact=False):
            def verify(result):
                if result is None:
                    raise ValueError("backend rejected the shape")
                if result.dtype != out_dtype or result.shape not in (
                    (m, groups, n),
                    (m, n),
                ):
                    raise ValueError(
                        f"unexpected result layout {tuple(result.shape)} {result.dtype}"
                    )
                sampled = result.index_select(0, rows).reshape_as(ref)
                if exact:
                    if not (
                        torch.equal(raw.index_select(0, rows), raw_ref)
                        and torch.equal(sampled, ref)
                    ):
                        raise ValueError(
                            "INT32 or scaled output differs from FP32 reference"
                        )
                else:
                    torch.testing.assert_close(sampled, ref, rtol=0.015625, atol=1e-4)

            try:
                if exact:
                    raw.index_fill_(0, rows, -(2**31))
                verify(run())
                return replace(trial, ms=self.time(run, verify))
            except (RuntimeError, ValueError, AssertionError, ImportError) as exc:
                detail = str(exc).strip().splitlines()
                return replace(
                    trial, status=detail[0] if detail else type(exc).__name__
                )

        def installed_rank(rank):
            self.ops.gemm_out(q, w, raw, groups, 0, False, rank)
            return scale()

        if dsv_route:
            rank = self.dsv._lookup_default_unpacked_algo_rank(
                self.device, m, groups, n, k, 0, False
            )
            baseline = check_run(
                lambda: installed_rank(rank or 0),
                Trial("hipblaslt", rank or 0),
                exact=True,
            )
            if baseline.ms is None and rank:
                trials.append(baseline)
                baseline = check_run(
                    lambda: installed_rank(0), Trial("hipblaslt", 0), exact=True
                )
        elif self.method == 3:
            baseline = check_run(
                lambda: self.dense.blaslt_scaled_mm(aq, w, xs, ws, out_dtype),
                Trial("w8a8 method 3 (LightOp hipBLASLt)"),
            )
        else:
            triton_config = self.dense.w8a8_triton_config(
                self.triton_configs(n, k), m, n, k
            )
            if triton_config is not None:
                current = "w8a8 method 1 (local Triton config)"
            elif self.asm_tunes:
                selected = None
                if m <= self.dense._W8A8_SMALL_M_AUTOTUNE_MAX_M and k % 256 == 0:
                    selected = self.dense._lookup_default_w8a8_small_m_kernel(
                        self.device, m, n, k, out_dtype
                    )
                default_asm = (
                    self.dense._default_w8a8_smooth_asm_tune(m, n)
                    if k % 256 == 0
                    else None
                )
                expected = (
                    Trial(*selected).label
                    if selected
                    else (
                        Trial("smooth_asm", default_asm).label
                        if default_asm is not None
                        else "LightOp Triton"
                    )
                )
                current = f"w8a8 method 1 ({expected} first)"
            else:
                current = "w8a8 method 1 (LightOp Triton, ASM disabled)"
            fn = (
                self.dense.triton_scaled_mm
                if self.asm_tunes
                else self.dense.quant_ops.triton_scaled_mm
            )
            baseline = check_run(
                lambda: fn(aq, light_weight, xs, ws, out_dtype, None, triton_config),
                Trial(current),
            )

        if self.probe is not None:
            measured = {}
            for limit in sorted(set(self.args.workspace_mib) | {0}):
                limit *= MIB
                try:
                    with self.probe.plan(shape, limit) as plan:
                        count = min(
                            self.probe.lib.bw1000_count(plan), self.args.max_ranks
                        )
                        needs = [
                            self.probe.lib.bw1000_workspace_bytes(plan, rank)
                            for rank in range(count)
                        ]
                        workspace = torch.empty(
                            max([0, *needs]), dtype=torch.uint8, device=self.device
                        )
                        if (
                            dsv_route
                            and limit == 0
                            and baseline.ms is not None
                            and baseline.rank_or_tune < count
                            and needs[baseline.rank_or_tune] == 0
                        ):
                            identity = (
                                self.probe.algorithm_bytes(plan, baseline.rank_or_tune),
                                0,
                            )
                            measured[identity] = baseline
                        stream = self.stream.cuda_stream
                        for rank, required in enumerate(needs):
                            trial = Trial("hipblaslt", rank, limit, max(required, 0))
                            if required < 0:
                                trials.append(
                                    replace(trial, status="unusable heuristic result")
                                )
                                continue
                            identity = (
                                self.probe.algorithm_bytes(plan, rank),
                                required,
                            )
                            previous = measured.get(identity)
                            if previous is not None:
                                trials.append(
                                    replace(
                                        trial, ms=previous.ms, status=previous.status
                                    )
                                )
                                continue

                            def run():
                                self.probe.run(
                                    plan, rank, q, w, raw, workspace, required, stream
                                )
                                return scale()

                            measured[identity] = check_run(run, trial, exact=True)
                            trials.append(measured[identity])
                except RuntimeError as exc:
                    trials.append(
                        Trial("hipblaslt", limit_bytes=limit, status=str(exc))
                    )
        else:
            for rank in range(self.args.max_ranks):
                if dsv_route and rank == baseline.rank_or_tune:
                    trial = replace(baseline)
                else:
                    trial = check_run(
                        lambda: installed_rank(rank),
                        Trial("hipblaslt", rank),
                        exact=True,
                    )
                trials.append(trial)

        if light_weight is not None:
            candidates = [("lightop_channel", 0)]
            default_asm = (
                self.dense._default_w8a8_smooth_asm_tune(m, n) if k % 256 == 0 else None
            )
            if default_asm is not None:
                candidates.extend(
                    ("smooth_asm", tune)
                    for tune in self.asm_tunes
                    if m <= 64 or tune == default_asm
                )
            for kind, tune in candidates:
                trial = check_run(
                    lambda: self.dense._run_w8a8_small_m_candidate(
                        kind, tune, aq, light_weight, xs, ws, out_dtype, m, n, k
                    ),
                    Trial(kind, tune),
                )
                trials.append(trial)
        return quant_ms, baseline, trials


def probe_smooth_asm(torch, dense, device):
    m, n, k = 5, 1792, 5120
    a = torch.zeros((m, k), device=device, dtype=torch.int8)
    b = torch.zeros((n, k), device=device, dtype=torch.int8).t()
    sa = torch.ones((m, 1), device=device, dtype=torch.float32)
    sb = torch.ones((n, 1), device=device, dtype=torch.float32)
    for tune in dense._W8A8_SMALL_M_ASM_TUNES:
        try:
            result = dense._run_w8a8_small_m_candidate(
                "smooth_asm", tune, a, b, sa, sb, torch.bfloat16, m, n, k
            )
            torch.cuda.synchronize(device)
            if result is not None:
                print(f"ASM_OK={tune}", flush=True)
        except (RuntimeError, ImportError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
    print("ASM_PROBE_COMPLETE", flush=True)


def available_asm_tunes(args):
    if args.skip_smooth_asm:
        return ()
    # A missing LightOp .co can exit the process with status 0. Isolate the
    # module loads and require a marker from Python after the calls return.
    command = [
        sys.executable,
        str(__file__),
        "--asm-probe",
        "--device",
        str(args.device),
        "--w8a8-method",
        str(args.w8a8_method),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print("LightOp smooth ASM probe timed out.", file=sys.stderr)
        return ()
    lines = result.stdout.splitlines()
    if result.returncode or "ASM_PROBE_COMPLETE" not in lines:
        print(
            f"LightOp smooth ASM unavailable.\n{(result.stderr or result.stdout)[-2000:]}",
            file=sys.stderr,
        )
        return ()
    return tuple(
        int(line.split("=")[1]) for line in lines if line.startswith("ASM_OK=")
    )


def manifest_row(shape, roles):
    return {
        "M": shape.m,
        "N": shape.total_n,
        "K": shape.k,
        "groups": shape.groups,
        "N_per_group": shape.n,
        "route": shape.route,
        "output_dtype": shape.dtype,
        "projections": ", ".join(sorted(roles)),
    }


def faster_than(baseline, options, min_gain):
    if baseline.ms is None:
        return None
    best = min(
        (trial for trial in options if trial.ms is not None),
        key=lambda t: t.ms,
        default=None,
    )
    return best if best is not None and best.ms < baseline.ms * (1 - min_gain) else None


def summarize_shape(
    shape, trials, baseline, config, static, method, min_gain, typed_key=False
):
    verified = [trial for trial in (baseline, *trials) if trial.ms is not None]
    best = min(verified, key=lambda trial: trial.ms, default=None)
    native = [trial for trial in trials if trial.backend == "hipblaslt"]
    if shape.route == "dsv41":
        selected = faster_than(baseline, native, min_gain)
        if selected:
            key = f"{shape.key}:{shape.dtype}" if typed_key else shape.key
            config["algorithms"][key] = selected.algorithm()
        zero = faster_than(
            baseline, (trial for trial in native if not trial.limit_bytes), min_gain
        )
        if zero and not typed_key:
            static[(shape.groups, shape.n, shape.k, 0, False)][shape.m] = (
                zero.rank_or_tune
            )
    else:
        selected = faster_than(baseline, trials, min_gain)
        if selected:
            key = f"{method}:{shape.m}:{shape.total_n}:{shape.k}"
            value = {"backend": selected.backend}
            if selected.backend == "hipblaslt":
                value.update(
                    rank=selected.rank_or_tune, workspace_bytes=selected.limit_bytes
                )
            if selected.backend == "smooth_asm":
                value["tune"] = selected.rank_or_tune
            config["w8a8"][key] = value
    return best, selected


def write_exports(output_dir, config, static, dsv):
    config_path = output_dir / "wo_a_algo.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    table = {}
    for key, entries in static.items():
        exact, buckets = dsv._DEFAULT_GFX936_UNPACKED_ALGO_RANKS.get(key, ({}, {}))
        table[key] = (exact | entries, buckets)
    (output_dir / "gfx936_static_ranks.py").write_text(
        "# Measured zero-workspace entries. Merge the keys into the existing table.\n"
        "_DEFAULT_GFX936_UNPACKED_ALGO_RANKS.update(\n"
        f"{pprint.pformat(table, sort_dicts=True)}\n)\n"
    )


def run_sweep(args, cases, contributions, sweep, config):
    output_dir = args.output_dir
    planned = sum(contributions.values(), Counter())
    with (
        (output_dir / "shapes.csv").open("w", newline="") as table_file,
        (output_dir / "candidates.csv").open("w", newline="") as trial_file,
    ):
        manifest_fields = tuple(manifest_row(next(iter(cases)), ()))
        table = csv.DictWriter(
            table_file,
            manifest_fields
            + (
                "current",
                "current_status",
                "quant_us",
                "current_us",
                "best",
                "best_us",
                "config_choice",
                "config_us",
                "calls_32k",
                "calls_400k",
                "saved_ms_32k",
                "saved_ms_400k",
            ),
        )
        trials_out = csv.DictWriter(
            trial_file, manifest_fields + ("current",) + tuple(asdict(Trial("")))
        )
        table.writeheader()
        trials_out.writeheader()
        static = defaultdict(dict)
        totals = Counter()
        covered = Counter()
        verified_shapes = 0
        dsv_keys = Counter(shape.key for shape in cases if shape.route == "dsv41")
        for shape in sorted(
            cases,
            key=lambda s: (
                s not in contributions,
                s.route,
                s.n,
                s.k,
                s.groups,
                s.dtype,
                s.m,
            ),
        ):
            info = manifest_row(shape, cases[shape])
            try:
                quant_ms, current, trials = sweep.shape(shape)
            except sweep.torch.OutOfMemoryError as exc:
                current = Trial("unavailable", status=str(exc).splitlines()[0])
                quant_ms, trials = None, []
            best, selected = summarize_shape(
                shape,
                trials,
                current,
                config,
                static,
                args.w8a8_method,
                args.min_gain,
                typed_key=dsv_keys[shape.key] > 1,
            )
            calls = contributions.get(shape, Counter())
            saved = (current.ms - selected.ms) if selected is not None else 0
            chunk_ms, request_ms = (
                saved * calls["calls_32k"],
                saved * calls["calls_400k"],
            )
            totals.update(ms_32k=chunk_ms, ms_400k=request_ms)
            if current.ms is not None:
                verified_shapes += 1
                covered.update(calls)
            table.writerow(
                dict(
                    info,
                    current=current.label,
                    current_status=current.status,
                    quant_us=None if quant_ms is None else quant_ms * 1000,
                    current_us=None if current.ms is None else current.ms * 1000,
                    best=best.label if best else None,
                    best_us=best.ms * 1000 if best else None,
                    config_choice=selected.label if selected else None,
                    config_us=selected.ms * 1000 if selected else None,
                    calls_32k=calls["calls_32k"],
                    calls_400k=calls["calls_400k"],
                    saved_ms_32k=chunk_ms,
                    saved_ms_400k=request_ms,
                )
            )
            for trial in (current, *trials):
                trials_out.writerow(
                    dict(info, current=trial is current, **asdict(trial))
                )
            table_file.flush()
            trial_file.flush()

            print(
                f"{shape.m:>5} {shape.total_n:>6} {shape.k:>5} G={shape.groups} "
                f"{shape.route:<5}  {current.timing} -> {best.timing if best else 'unavailable'}  {info['projections']}",
                flush=True,
            )
        config["metadata"]["prefill_estimate"] = {
            "request_tokens": args.request_tokens,
            "ms_32k": totals["ms_32k"],
            "ms_request": totals["ms_400k"],
            "covered_calls": dict(covered),
            "planned_calls": dict(planned),
        }
        write_exports(output_dir, config, static, sweep.dsv)
    print(
        f"32K CP8 chunk saves {totals['ms_32k']:.2f} ms of projection time "
        f"({covered['calls_32k']}/{planned['calls_32k']} launches with a verified baseline)."
    )
    print(
        f"{args.request_tokens:,} token request saves {totals['ms_400k']:.2f} ms of projection time "
        f"({covered['calls_400k']}/{planned['calls_400k']} launches with a verified baseline)."
    )
    print(f"Reports and tuning files written to {output_dir}")
    if verified_shapes == 0:
        raise SystemExit("No baseline passed numerical and timing checks.")


def main():
    args = parse_args()
    if args.list_shapes:
        cases, _ = shape_suite(args)
        writer = csv.DictWriter(sys.stdout, tuple(manifest_row(next(iter(cases)), ())))
        writer.writeheader()
        writer.writerows(manifest_row(shape, cases[shape]) for shape in sorted(cases))
        return
    if args.w8a8_method not in (1, 3):
        raise SystemExit("Select the deployed W8A8 method with --w8a8-method 1 or 3.")

    sys.path.insert(0, str(ROOT / "python"))
    import torch

    if torch.version.hip is None or not torch.cuda.is_available():
        raise SystemExit(
            "This sweep requires ROCm/DTK PyTorch and a visible BW1000 GPU."
        )
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device)
    arch = props.gcnArchName.split(":")[0]
    if arch != "gfx936":
        raise SystemExit(f"Expected gfx936 on {device}, found {arch}.")

    from lightop.quant import per_token_quant_int8

    from sglang.kernels.ops.gemm import dsv41_int8
    from sglang.srt.layers.quantization.compressed_tensors import quant_ops
    from sglang.srt.utils import W8a8GetCacheJSON

    if args.asm_probe:
        probe_smooth_asm(torch, quant_ops, device)
        return
    cases, contributions = shape_suite(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Benchmark the out-of-box dispatch with PD_VENDOR_TUNING disabled.
    os.environ.pop("SGLANG_DSV41_INT8_BLASLT_ALGO_CONFIG", None)
    os.environ["SGLANG_DSV41_INT8_BLASLT_STATIC_ALGO"] = "1"
    os.environ["SGLANG_HCU_W8A8_SMALL_M_STATIC"] = "1"
    os.environ["SGLANG_HCU_W8A8_SMALL_M_AUTOTUNE"] = "0"
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    asm_tunes = available_asm_tunes(args)
    dsv41_int8.load_native()
    hip_probe = build_probe(args.output_dir)
    config = {
        "algorithms": {},
        "w8a8": {},
        "metadata": {
            "arch": arch,
            "compute_units": props.multi_processor_count,
            "torch": str(torch.__version__),
            "hip": torch.version.hip,
            "heuristic_count": 64,
            "w8a8_method": args.w8a8_method,
            "workspace_limits_mib": sorted(set(args.workspace_mib) | {0})
            if hip_probe
            else [0],
            "hip_graph": not args.no_graph,
            "ref_rows": args.verify_rows,
        },
    }
    print(
        f"Sweeping {len(cases)} shapes on {device} ({arch}, {props.multi_processor_count} CUs).",
        flush=True,
    )
    stream = torch.cuda.Stream(device=device)
    with torch.inference_mode(), torch.cuda.stream(stream):
        sweep = Sweep(
            args,
            torch,
            dsv41_int8,
            quant_ops,
            per_token_quant_int8,
            W8a8GetCacheJSON(),
            device,
            hip_probe,
            asm_tunes,
        )
        run_sweep(args, cases, contributions, sweep, config)


if __name__ == "__main__":
    main()
