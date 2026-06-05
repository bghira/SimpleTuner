import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from simpletuner.helpers.models.ideogram import quantized_loading
from simpletuner.helpers.models.ideogram.quantized_loading import Fp8Linear, quantize_weight_to_fp8


class IdeogramFp8LinearTests(unittest.TestCase):
    def _make_layer(self, in_features=4, out_features=3, dtype=torch.bfloat16):
        torch.manual_seed(123)
        layer = Fp8Linear(in_features, out_features, bias=True, compute_dtype=dtype)
        weight = torch.randn(out_features, in_features, dtype=dtype)
        weight_fp8, weight_scale = quantize_weight_to_fp8(weight)
        layer.weight = weight_fp8
        layer.weight_scale = weight_scale
        layer.bias = torch.randn(out_features, dtype=dtype)
        return layer

    def test_cpu_forward_uses_dequantized_linear_fallback(self):
        layer = self._make_layer()
        x = torch.randn(2, 5, layer.in_features, dtype=torch.bfloat16)

        out = layer(x)
        dequant_weight = layer.weight.to(x.dtype) * layer.weight_scale.to(x.dtype).unsqueeze(1)
        expected = F.linear(x, dequant_weight, layer.bias)

        self.assertEqual(out.dtype, x.dtype)
        torch.testing.assert_close(out, expected)

    def test_scaled_mm_path_preserves_input_gradient(self):
        layer = self._make_layer(dtype=torch.float32)
        x = torch.randn(2, 5, layer.in_features, dtype=torch.float32, requires_grad=True)
        x_expected = x.detach().clone().requires_grad_(True)

        def fake_scaled_mm(a, b, *, scale_a, scale_b, bias=None, out_dtype=None, use_fast_accum=None):
            out = a.to(out_dtype) * scale_a.to(out_dtype)
            out = out.matmul(b.to(out_dtype) * scale_b.to(out_dtype))
            if bias is not None:
                out = out + bias
            return out

        with (
            patch.object(quantized_loading, "_scaled_mm_supported", return_value=True),
            patch.object(quantized_loading.torch, "_scaled_mm", side_effect=fake_scaled_mm),
        ):
            out = layer(x)
            out.sum().backward()

        dequant_weight = layer.weight.to(x_expected.dtype) * layer.weight_scale.to(x_expected.dtype).unsqueeze(1)
        expected = F.linear(x_expected, dequant_weight, layer.bias)
        expected.sum().backward()

        self.assertEqual(out.shape, expected.shape)
        torch.testing.assert_close(x.grad, x_expected.grad)

    def test_scaled_mm_support_can_be_forced_or_disabled(self):
        class FakeCudaTensor:
            is_cuda = True
            device = torch.device("cuda:0")

        x = FakeCudaTensor()

        with patch.dict("os.environ", {"SIMPLETUNER_IDEOGRAM_FP8_SCALED_MM": "1"}):
            with (
                patch.object(quantized_loading.torch.cuda, "get_device_capability", return_value=(9, 0)),
            ):
                self.assertTrue(quantized_loading._scaled_mm_supported(x))

        with patch.dict("os.environ", {"SIMPLETUNER_IDEOGRAM_FP8_SCALED_MM": "0"}):
            self.assertFalse(quantized_loading._scaled_mm_supported(x))

        with patch.dict("os.environ", {"SIMPLETUNER_IDEOGRAM_FP8_SCALED_MM": "auto"}):
            with (
                patch.object(quantized_loading.torch.cuda, "get_device_capability", return_value=(8, 9)),
            ):
                self.assertTrue(quantized_loading._scaled_mm_supported(x))
            with (
                patch.object(quantized_loading.torch.cuda, "get_device_capability", return_value=(9, 0)),
            ):
                self.assertFalse(quantized_loading._scaled_mm_supported(x))


if __name__ == "__main__":
    unittest.main()
