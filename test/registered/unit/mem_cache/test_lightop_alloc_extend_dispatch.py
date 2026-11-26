"""HCU extend routing and capacity checks; GPU kernel equality is tested separately."""

import os
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.allocator import paged
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLightOpAllocExtendDispatch(unittest.TestCase):
    def call(
        self,
        bs,
        ext,
        *,
        enabled=True,
        available=None,
        lightop_missing=False,
        triton=False,
    ):
        p = 256
        need = bs * ((ext + p - 1) // p)
        n = need + bs + 2 if available is None else available
        flag = paged.envs.SGLANG_LIGHTOP_KVALLOC_KERNEL
        with flag.override(enabled) if enabled is not None else nullcontext():
            a = paged.PagedTokenToKVPoolAllocator(
                n * p, p, torch.bfloat16, "cpu", None, False
            )
        a.sglang_kvalloc_kernel = True
        a.use_triton_kernels = triton
        a.free_pages = torch.arange(100, 100 + n, dtype=torch.int64)
        original = a.free_pages.clone()
        pre = torch.zeros(bs, dtype=torch.int64)
        seq = pre + ext
        last = pre - 1

        def reference(**k):
            paged.alloc_extend_naive(
                k["pre_lens_ptr"],
                k["seq_lens_ptr"],
                k["last_loc_ptr"],
                k["free_page_ptr"],
                k["out_indices"],
                k["page_size"],
                "cpu",
            )

        def single(prefix, last_loc, pages, out, page_size):
            paged.alloc_extend_naive(
                prefix, seq, last_loc, pages, out, page_size, "cpu"
            )

        with (
            patch.object(
                paged,
                "sgl_kernel_alloc_extend_kernel",
                side_effect=reference,
                create=True,
            ) as old,
            patch.object(paged, "_get_lightop_alloc_extend_kernel") as get,
            patch.object(
                paged, "alloc_extend_single_request", side_effect=single
            ) as parallel,
        ):
            fast = Mock(side_effect=reference)
            get.return_value = None if lightop_missing else fast
            out = a.alloc_extend(pre, pre, seq, seq, last, bs * ext)
            if need > n:
                self.assertIsNone(out)
                old.assert_not_called()
                get.assert_not_called()
                self.assertTrue(torch.equal(a.free_pages, original))
            else:
                expect = torch.empty(bs * ext, dtype=torch.int64)
                paged.alloc_extend_naive(pre, seq, last, original, expect, p, "cpu")
                self.assertTrue(torch.equal(out, expect))
                self.assertTrue(torch.equal(a.free_pages, original[need:]))
                self.assertEqual(
                    parallel.call_count + old.call_count + fast.call_count, 1
                )
            return (old.call_count, fast.call_count, get.call_count)

    def test_supported_long_extend(self):
        for bs, ext in [(1, 8192), (1, 32768), (64, 8192)]:
            self.assertEqual(self.call(bs, ext), (0, 1, 1))

    def test_tiny_extend_falls_back(self):
        self.assertEqual(self.call(1, 127), (1, 0, 0))

    def test_short_request_batch_falls_back(self):
        for bs, ext in [(1, 8191), (64, 4096)]:
            self.assertEqual(self.call(bs, ext), (1, 0, 0))

    def test_large_batch_falls_back(self):
        for bs in [65, 128]:
            self.assertEqual(self.call(bs, 8192), (1, 0, 0))

    def test_explicit_disable(self):
        self.assertEqual(self.call(1, 32768, enabled=False), (1, 0, 0))

    def test_missing_lightop(self):
        self.assertEqual(self.call(1, 32768, lightop_missing=True), (1, 0, 1))

    def test_parallel_singleton_and_explicit_lightop_priority(self):
        with (
            patch.dict(os.environ),
            patch.object(torch.Tensor, "is_cuda", True),
            patch.object(paged, "_is_hcu", True),
            paged.envs.SGLANG_HCU_PARALLEL_KV_ALLOC_EXTEND.override(True),
        ):
            flag = paged.envs.SGLANG_LIGHTOP_KVALLOC_KERNEL
            os.environ.pop(flag.name, None)
            self.assertEqual(self.call(1, 16384, enabled=None, triton=True), (0, 0, 0))
            self.assertEqual(self.call(1, 8192, enabled=None, triton=True), (0, 1, 1))
            self.assertEqual(self.call(2, 16384, enabled=None, triton=True), (0, 1, 1))
            with flag.override(True):
                self.assertEqual(
                    self.call(1, 16384, enabled=None, triton=True), (0, 1, 1)
                )
                self.assertEqual(
                    self.call(
                        1, 16384, enabled=None, triton=True, lightop_missing=True
                    ),
                    (0, 0, 1),
                )

    def test_short_pool_before_dispatch(self):
        self.assertEqual(self.call(1, 32768, available=127), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
