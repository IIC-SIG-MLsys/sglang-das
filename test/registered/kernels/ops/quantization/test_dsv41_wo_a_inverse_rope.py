"""Bitwise parity for the gfx936 INT8 wo_a inverse-RoPE operand."""

import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import deepseek_v4_rope
from sglang.kernels.ops.gemm import dsv41_int8
from sglang.kernels.ops.quantization.dsv41_int8 import quantize_wo_a_inverse_rope
from sglang.test.ci.ci_register import register_hcu_ci
from sglang.test.test_utils import CustomTestCase

register_hcu_ci(est_time=15, suite="stage-b-test-1-hcu-small")


@unittest.skipUnless(
    torch.version.hip is not None
    and torch.cuda.is_available()
    and dsv41_int8._is_gfx936_device(torch.device("cuda", 0)),
    "requires a gfx936 GPU",
)
class TestDsv41WoAInverseRope(CustomTestCase):
    def test_quantization_matches_materialized_bf16_rope_bitwise(self):
        generator = torch.Generator(device="cuda").manual_seed(107)
        for tokens, groups, width, pad in ((65, 2, 1536, 512), (1024, 8, 4096, 0)):
            with self.subTest(tokens=tokens, width=width, pad=pad):
                x = torch.randn(
                    (tokens, groups, width + pad),
                    dtype=torch.bfloat16,
                    device="cuda",
                    generator=generator,
                )[..., :width]
                x[0, 0].zero_()
                x[1, 1] *= 1.0e-13
                # Include exact half-way quantization ties on an unrotated row.
                x[2, 0].zero_()
                x[2, 0, :3] = torch.tensor([-0.5, 0.5, 127], device="cuda")
                phases = torch.randn(8192, 32, device="cuda", generator=generator)
                freqs_cis = torch.polar(torch.ones_like(phases), phases)
                positions = torch.randint(
                    0, 8192, (tokens,), device="cuda", generator=generator
                ).to(torch.int32 if pad else torch.int64)
                rotated = x.clone(memory_format=torch.contiguous_format).view(
                    tokens, -1, 512
                )
                with patch.object(deepseek_v4_rope, "_USE_BATCHED_ROPE", True):
                    deepseek_v4_rope.apply_rotary_emb_triton(
                        rotated[..., -64:], freqs_cis, positions, inverse=True
                    )
                with patch.object(
                    dsv41_int8, "_triton_quantize_enabled", return_value=True
                ):
                    ref_q, ref_scales = dsv41_int8.quantize(rotated.view(x.shape))
                q, scales = quantize_wo_a_inverse_rope(x, freqs_cis, positions)
                torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
                torch.testing.assert_close(scales, ref_scales, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
