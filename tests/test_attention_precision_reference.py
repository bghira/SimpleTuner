import unittest
from decimal import Decimal, localcontext

import torch

from scripts.benchmark_attention_precision import fp64_reference, metrics


class AttentionPrecisionReferenceTests(unittest.TestCase):
    def test_matches_autograd_for_normal_logits_masks_gqa_and_signed_scale(self):
        for scale in (0.0, -0.3, 0.3):
            for causal in (False, True):
                with self.subTest(scale=scale, causal=causal):
                    torch.manual_seed(95)
                    q = torch.randn(2, 4, 5, 3, dtype=torch.float64, requires_grad=True)
                    k, v = [torch.randn(2, 2, 7, 3, dtype=torch.float64, requires_grad=True) for _ in range(2)]
                    dout = torch.randn_like(q)
                    mask = torch.rand(1, 1, 5, 7) > 0.3
                    mask[..., 0, :] = False
                    effective_mask = mask & (torch.arange(7)[None, :] <= torch.arange(5)[:, None]) if causal else mask
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        expected = torch.nn.functional.scaled_dot_product_attention(
                            q, k, v, attn_mask=effective_mask, scale=scale, enable_gqa=True
                        )
                    gradients = torch.autograd.grad(expected, (q, k, v), dout)
                    for actual, want in zip(fp64_reference((q, k, v), dout, mask, causal, scale), (expected, *gradients)):
                        torch.testing.assert_close(actual, want, atol=1e-12, rtol=1e-12)

    def test_saturated_softmax_matches_decimal_finite_difference(self):
        q = torch.tensor([[[[10.0]]]], dtype=torch.float64)
        k = torch.tensor([[[[10.0], [5.0]]]], dtype=torch.float64)
        v = torch.tensor([[[[1.0], [-1.0]]]], dtype=torch.float64)
        _, dq, dk, dv = fp64_reference((q, k, v), torch.ones_like(q), scale=1.0)
        with localcontext() as context:
            context.prec = 100

            def output(query):
                tail = (-Decimal(5) * query).exp()
                return (1 - tail) / (1 + tail)

            step = Decimal("0.000001")
            expected = float((output(Decimal(10) + step) - output(Decimal(10) - step)) / (2 * step))
        self.assertGreater(dq.item(), 0)
        self.assertLess(abs(dq.item() / expected - 1), 1e-10)
        self.assertEqual(dk.sum().item(), 0)
        self.assertEqual(dv.sum().item(), 1)

    def test_metrics_preserve_tiny_reference_norms_and_cosines(self):
        reference = torch.tensor([1e-250, -2e-250], dtype=torch.float64)
        result = metrics(reference * 2, reference)
        self.assertGreater(result["reference_norm"], 0)
        self.assertAlmostEqual(result["relative_error"], 1)
        self.assertAlmostEqual(result["norm_ratio"], 2)
        self.assertAlmostEqual(result["cosine"], 1)
