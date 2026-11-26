"""HCU correctness for parallel long paged extend and PD page translation."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.memory.virtual_slot import translate_token_ids
from sglang.srt.disaggregation.utils import translate_transfer_pages
from sglang.srt.environ import envs
from sglang.srt.mem_cache.allocator import paged
from sglang.srt.utils import is_hcu
from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.test_utils import CustomTestCase

register_hcu_ci(est_time=10, suite="stage-b-test-1-hcu-small")


@unittest.skipUnless(is_hcu(), "requires an HCU GPU")
class TestLongSingleRequestAlloc(CustomTestCase):
    def test_extend_reuses_prefix_and_consumes_fragmented_pages(self):
        page_size = 256
        with (
            envs.SGLANG_LIGHTOP_KVALLOC_KERNEL.override(False),
            envs.SGLANG_HCU_PARALLEL_KV_ALLOC_EXTEND.override(True),
            patch.object(
                paged,
                "alloc_extend_single_request",
                wraps=paged.alloc_extend_single_request,
            ) as expand,
        ):
            allocator = paged.PagedTokenToKVPoolAllocator(
                size=5000 * page_size,
                page_size=page_size,
                dtype=torch.bfloat16,
                device="cuda",
                kvcache=None,
                need_sort=False,
            )
            pages = torch.arange(4900, 500, -2, dtype=torch.int64)
            for prefix, end in ((0, 409600), (4096, 409600), (4099, 409619)):
                with self.subTest(prefix=prefix, end=end):
                    allocator.free_pages = pages.cuda()
                    prefix_lens = torch.tensor([prefix])
                    seq_lens = torch.tensor([end])
                    tail = -(-prefix // page_size) * page_size - prefix
                    last = 123 * page_size + (prefix - 1) % page_size
                    num_pages = -(-(end - prefix - tail) // page_size)
                    actual = allocator.alloc_extend(
                        prefix_lens.cuda(),
                        prefix_lens,
                        seq_lens.cuda(),
                        seq_lens,
                        torch.tensor([last if prefix else -1], device="cuda"),
                        end - prefix,
                    )
                    new = (
                        pages[:num_pages, None] * page_size + torch.arange(page_size)
                    ).flatten()[: end - prefix - tail]
                    expected = torch.cat((torch.arange(last + 1, last + 1 + tail), new))
                    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                    torch.testing.assert_close(
                        allocator.free_pages.cpu(), pages[num_pages:]
                    )
            self.assertEqual(expand.call_count, 3)

    def test_translation_on_gpu_uses_the_same_physical_page_order(self):
        page_size = 256
        virtual = torch.arange(1, 1601, device="cuda", dtype=torch.int32)
        offsets = torch.arange(page_size, device="cuda", dtype=torch.int32)
        row = (virtual[:, None] * page_size + offsets).flatten()[:-13]
        v2p = torch.arange(3200, 1599, -1, device="cuda", dtype=torch.int64)

        def translate(indices):
            return translate_token_ids(loc=indices, v2p=v2p, page_size=page_size)

        actual = translate_transfer_pages(row, page_size, translate)
        expected = translate(row)[::page_size] // page_size
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
