"""CPU checks for the consolidated API and installed kernel boundary."""
import builtins
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from sglang.kernels.ops.gemm import dsv41_int8
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestInt8KernelBridge(CustomTestCase):
    def tearDown(self):
        for function in [dsv41_int8.backend, dsv41_int8._service_config, dsv41_int8._algorithms, dsv41_int8._scale_schedules]:
            function.cache_clear()
        dsv41_int8._algo_rank_cache.clear()
        dsv41_int8._failed_static_algo_keys.clear()
        dsv41_int8._device_arch_cache.clear()

    def test_unpacked_projection_consumes_rotated_quant_pair(self):
        x = torch.zeros(2, 1, 2, dtype=torch.bfloat16)
        q = torch.tensor([[[2, -3]], [[-4, 1]]], dtype=torch.int8)
        xs = torch.tensor([[[0.25]], [[0.5]]])
        weight = torch.tensor([[3, -2]], dtype=torch.int8)
        ws = torch.tensor([[0.5]])

        def gemm(operand, weights, result, *args):
            result.copy_(operand.int() @ weights.int().T)

        def scale(raw, input_scale, weight_scale, result, *args):
            result.copy_(raw * input_scale * weight_scale.T)

        ops = types.SimpleNamespace(gemm_out=gemm, scale_out=scale)
        with patch.object(dsv41_int8, "load_native", return_value=ops):
            actual = dsv41_int8._project_unpacked(
                x, weight, ws, 1, 1, input_quant_args=(q, xs), algo_rank=0
            )
        expected = torch.tensor([[[1.5]], [[-3.5]]], dtype=x.dtype)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_inverse_rope_projection_respects_fast_path_gates(self):
        minimum = dsv41_int8._TRITON_QUANT_MIN_ELEMENTS
        x = Mock(is_cuda=True, numel=Mock(return_value=minimum))
        w, s, freqs, positions, pair, output = [object() for _ in range(6)]
        with (
            patch.object(dsv41_int8, "_supports_wo_a_inverse_rope", return_value=True),
            patch.object(dsv41_int8, "_triton_quantize_enabled", return_value=True),
            patch.object(dsv41_int8, "_is_gfx936_device", return_value=True),
            patch.object(dsv41_int8, "_use_packed_service_projection") as packed,
            patch.object(torch.cuda, "is_current_stream_capturing") as capturing,
            patch.object(
                dsv41_int8, "quantize_wo_a_inverse_rope", return_value=pair
            ) as quantize,
            patch.object(
                dsv41_int8, "_project_unpacked", return_value=output
            ) as project,
        ):
            args = (x, w, s, 8, 1024, freqs, positions)
            for size, is_packed, is_capturing in (
                (minimum - 1, False, False),
                (minimum, True, False),
                (minimum, False, True),
            ):
                x.numel.return_value = size
                packed.return_value = is_packed
                capturing.return_value = is_capturing
                self.assertIsNone(dsv41_int8.try_project_wo_a_with_inverse_rope(*args))
            quantize.assert_not_called()
            project.assert_not_called()
            capturing.return_value = False
            self.assertIs(dsv41_int8.try_project_wo_a_with_inverse_rope(*args), output)
            quantize.assert_called_once_with(x, freqs, positions)
            project.assert_called_once_with(x, w, s, 8, 1024, input_quant_args=pair)

    def test_backend_lazily_binds_installed_kernel_once(self):
        ops = types.SimpleNamespace(**{name: Mock() for name in dsv41_int8._OPERATORS})
        with patch.dict(sys.modules, {"sgl_kernel": types.ModuleType("sgl_kernel")}), patch.object(torch.ops, "sgl_kernel", ops):
            first = dsv41_int8.backend()
            self.assertIs(first, dsv41_int8.backend())
            q, weight, raw = object(), object(), object()
            first.gemm(q, weight, raw, 17)
        ops.dsv41_int8_gemm_packed.assert_called_once_with(q, weight, raw, 17)

    def test_old_kernel_wheel_reports_required_upgrade(self):
        with patch.dict(sys.modules, {"sgl_kernel": types.ModuleType("sgl_kernel")}), patch.object(torch.ops, "sgl_kernel", types.SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "Rebuild the matching HCU"):
                dsv41_int8.backend()

    def test_native_bridge_keeps_plan_and_scale_arguments(self):
        ops = types.SimpleNamespace(dsv41_int8_prepare=Mock(return_value=17), dsv41_int8_gemm_packed=Mock(), dsv41_int8_scale_scheduled=Mock())
        api = dsv41_int8._NativeOps(ops)
        weight, q, raw, xs, ws, out = [object() for _ in range(6)]
        plan = api.prepare(weight, 40, 1, 16160, 5120, 1, 16160, 82739200, 5120, False, "")
        api.gemm(q, weight, raw, plan)
        api.scale_scheduled(raw, xs, ws, out, 2, 4, 128)
        ops.dsv41_int8_prepare.assert_called_once_with(weight, 40, 1, 16160, 5120, 1, 16160, 82739200, 5120, False, "")
        ops.dsv41_int8_gemm_packed.assert_called_once_with(q, weight, raw, 17)
        ops.dsv41_int8_scale_scheduled.assert_called_once_with(raw, xs, ws, out, 2, 4, 128)

    def test_embedded_defaults_and_explicit_configuration(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_DSV41_INT8_SERVICE_ALGO_CONFIG", None)
            self.assertIn("wo_a:1:4096:8:1024:4325376:0", dsv41_int8._algorithms())
        for function in [dsv41_int8._service_config, dsv41_int8._algorithms, dsv41_int8._scale_schedules]:
            function.cache_clear()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "algorithms.json"
            config = {"algorithms": {"custom": "1234"}, "scale_schedules": {"lm_head": {"4": [1, 2, 128]}}}
            path.write_text(json.dumps(config))
            with patch.dict(os.environ, {"SGLANG_DSV41_INT8_SERVICE_ALGO_CONFIG": str(path)}):
                self.assertEqual(dsv41_int8._algorithms(), config["algorithms"])
                self.assertEqual(dsv41_int8._scale_schedules(), config["scale_schedules"]["lm_head"])

    def test_import_does_not_load_native_backend(self):
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == "sgl_kernel":
                raise AssertionError("Importing the API must not load native operators")
            return original(name, *args, **kwargs)
        spec = importlib.util.spec_from_file_location("int8_import_probe", dsv41_int8.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch("builtins.__import__", side_effect=guarded):
            spec.loader.exec_module(module)
        self.assertEqual(module.backend.cache_info().currsize, 0)

    def test_autotune_selects_fastest_rank_caches_and_stops_on_exhaustion(self):
        latencies = {0: 5.0, 1: 1.2, 2: 2.5}
        calls = []

        def fake_gemm_out(q, w, raw, groups, tile_n, group_major, algo_rank=0):
            calls.append(algo_rank)
            if algo_rank not in latencies:
                raise RuntimeError("invalid heuristic rank")

        class FakeEvent:
            last_rank = 0

            def __init__(self, enable_timing=False):
                self.rank = 0

            def record(self):
                self.rank = calls[-1] if calls else 0

            def synchronize(self):
                pass

            def elapsed_time(self, other):
                return latencies[other.rank]

        ops = types.SimpleNamespace(gemm_out=fake_gemm_out)
        fake_q = types.SimpleNamespace(
            shape=(4096, 8, 4096),
            is_cuda=True,
            device=types.SimpleNamespace(index=0),
        )
        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.cuda, "Event", FakeEvent),
        ):
            best = dsv41_int8._select_unpacked_algo_rank(
                ops, fake_q, object(), object(), 8, 1024, 4096, 0, False
            )
            self.assertEqual(best, 1)
            probe_calls = len(calls)
            # Second call hits cache with zero extra probes.
            best_cached = dsv41_int8._select_unpacked_algo_rank(
                ops, fake_q, object(), object(), 8, 1024, 4096, 0, False
            )
            self.assertEqual(best_cached, 1)
            self.assertEqual(len(calls), probe_calls)

    def test_autotune_skipped_during_graph_capture_without_poisoning_cache(self):
        ops = types.SimpleNamespace(gemm_out=Mock())
        fake_q = types.SimpleNamespace(
            shape=(6, 8, 4096),
            is_cuda=True,
            device=types.SimpleNamespace(index=0),
        )
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            rank = dsv41_int8._select_unpacked_algo_rank(
                ops, fake_q, object(), object(), 8, 1024, 4096, 0, False
            )
        self.assertEqual(rank, 0)
        self.assertEqual(len(dsv41_int8._algo_rank_cache), 0)
        ops.gemm_out.assert_not_called()

    def test_autotune_defaults_off_and_respects_opt_in_env(self):
        with patch.dict(os.environ):
            os.environ.pop("SGLANG_DSV41_INT8_BLASLT_AUTOTUNE", None)
            self.assertFalse(dsv41_int8._blaslt_autotune_enabled())
        with patch.dict(os.environ, {"SGLANG_DSV41_INT8_BLASLT_AUTOTUNE": "0"}):
            self.assertFalse(dsv41_int8._blaslt_autotune_enabled())
        with patch.dict(os.environ, {"SGLANG_DSV41_INT8_BLASLT_AUTOTUNE": "1"}):
            self.assertTrue(dsv41_int8._blaslt_autotune_enabled())

    def test_static_gfx936_algo_lookup_and_uncovered_fallback(self):
        dev = types.SimpleNamespace(type="cuda", index=0)
        gfx936_props = types.SimpleNamespace(gcnArchName="gfx936:sramecc+:xnack-")
        gfx942_props = types.SimpleNamespace(gcnArchName="gfx942:sramecc+:xnack-")

        with (
            patch.dict(os.environ, {"SGLANG_DSV41_INT8_BLASLT_STATIC_ALGO": "1"}),
            patch.object(torch.cuda, "get_device_properties", return_value=gfx936_props),
        ):
            # Covered exact decode M and bucketed prefill M on gfx936.
            self.assertEqual(
                dsv41_int8._lookup_default_unpacked_algo_rank(dev, 6, 8, 1024, 4096, 0, False),
                27,
            )
            self.assertEqual(
                dsv41_int8._lookup_default_unpacked_algo_rank(dev, 4096, 8, 1024, 4096, 0, False),
                1,
            )
            self.assertEqual(
                dsv41_int8._lookup_default_unpacked_algo_rank(dev, 3000, 8, 1024, 4096, 0, False),
                1,
            )
            self.assertEqual(
                dsv41_int8._lookup_default_unpacked_algo_rank(dev, 512, 8, 1024, 4096, 0, False),
                6,
            )
            # Uncovered M / shape returns None (caller uses algo_rank=0 without probing).
            self.assertIsNone(
                dsv41_int8._lookup_default_unpacked_algo_rank(dev, 13, 8, 1024, 4096, 0, False)
            )
            self.assertIsNone(
                dsv41_int8._lookup_default_unpacked_algo_rank(dev, 16384, 8, 1024, 4096, 0, False)
            )
            self.assertIsNone(
                dsv41_int8._lookup_default_unpacked_algo_rank(dev, 4096, 4, 256, 512, 0, False)
            )

        # Disabled via env -> returns None.
        dsv41_int8._device_arch_cache.clear()
        with (
            patch.dict(os.environ, {"SGLANG_DSV41_INT8_BLASLT_STATIC_ALGO": "0"}),
            patch.object(torch.cuda, "get_device_properties", return_value=gfx936_props),
        ):
            self.assertIsNone(
                dsv41_int8._lookup_default_unpacked_algo_rank(dev, 4096, 8, 1024, 4096, 0, False)
            )

        # Non-gfx936 GPU -> returns None.
        dsv41_int8._device_arch_cache.clear()
        with (
            patch.dict(os.environ, {"SGLANG_DSV41_INT8_BLASLT_STATIC_ALGO": "1"}),
            patch.object(torch.cuda, "get_device_properties", return_value=gfx942_props),
        ):
            self.assertIsNone(
                dsv41_int8._lookup_default_unpacked_algo_rank(dev, 4096, 8, 1024, 4096, 0, False)
            )

    def test_project_unpacked_default_uses_static_rank_and_caches_fallback_to_zero(self):
        calls = []

        def fake_gemm_out(q, w, raw, groups, tile_n, group_major, algo_rank=0):
            calls.append(algo_rank)
            if algo_rank == 1:
                raise RuntimeError("rank unavailable on this build")

        ops = types.SimpleNamespace(
            quantize_out=Mock(),
            gemm_out=fake_gemm_out,
            scale_out=Mock(),
        )
        x = torch.zeros(4096, 8, 4096, dtype=torch.bfloat16)
        w = torch.zeros(8192, 4096, dtype=torch.int8)
        s = torch.ones(8192, 1, dtype=torch.float32)
        with (
            patch.dict(
                os.environ, {"SGLANG_DSV41_INT8_BLASLT_STATIC_ALGO": "1"}, clear=False
            ),
            patch.object(dsv41_int8, "load_native", return_value=ops),
            patch.object(dsv41_int8, "_is_gfx936_device", return_value=True),
            patch.object(dsv41_int8, "_select_unpacked_algo_rank") as mock_autotune,
        ):
            os.environ.pop("SGLANG_DSV41_INT8_BLASLT_AUTOTUNE", None)
            os.environ.pop("SGLANG_DSV41_INT8_BLASLT_ALGO_CONFIG", None)
            # First call tries static rank 1, catches RuntimeError, caches failure, falls back to 0.
            dsv41_int8._project_unpacked(x, w, s, 8, 1024)
            # Second call goes directly to rank 0 without retrying the rejected static rank.
            dsv41_int8._project_unpacked(x, w, s, 8, 1024)
        mock_autotune.assert_not_called()
        self.assertEqual(calls, [1, 0, 0])

    def test_autotune_buckets_large_prefill_t_but_keeps_small_decode_exact(self):
        self.assertEqual(dsv41_int8._autotune_t_bucket(6), 6)
        self.assertEqual(dsv41_int8._autotune_t_bucket(48), 48)
        self.assertEqual(dsv41_int8._autotune_t_bucket(64), 64)
        self.assertEqual(dsv41_int8._autotune_t_bucket(65), 128)
        self.assertEqual(dsv41_int8._autotune_t_bucket(3652), 4096)
        self.assertEqual(dsv41_int8._autotune_t_bucket(3653), 4096)
        self.assertEqual(dsv41_int8._autotune_t_bucket(4096), 4096)

    @unittest.skipUnless(
        torch.cuda.is_available() and torch.version.hip is not None,
        "HIP/ROCm GPU required",
    )
    def test_gpu_static_algo_rank_matches_rank_zero_bitwise(self):
        device = torch.device("cuda:0")
        torch.manual_seed(0)
        x = torch.randn(4096, 8, 4096, device=device, dtype=torch.bfloat16) * 0.25
        w = torch.randint(-64, 64, (8192, 4096), device=device, dtype=torch.int8)
        s = torch.randn(8192, 1, device=device, dtype=torch.float32).abs() * 0.01
        for out_dtype in (torch.bfloat16, torch.float32):
            y_ref = dsv41_int8._project_unpacked(
                x, w, s, 8, 1024, output_dtype=out_dtype, algo_rank=0
            )
            y_opt = dsv41_int8._project_unpacked(
                x, w, s, 8, 1024, output_dtype=out_dtype
            )
            self.assertTrue(torch.equal(y_ref, y_opt))

if __name__ == "__main__":
    unittest.main()
