import unittest
from unittest import mock

import torch

from simpletuner.helpers.models.ideogram import quantized_loading
from simpletuner.helpers.models.ideogram.quantized_loading import (
  FP8_WEIGHT_DTYPE,
  Fp8Linear,
  quantize_weight_to_fp8,
)


class IdeogramFp8LinearTests(unittest.TestCase):
  def test_to_dtype_preserves_fp8_weight_and_float_scale(self):
    layer = Fp8Linear(4, 3, bias=True, compute_dtype=torch.bfloat16)
    weight = layer.weight
    weight_scale = layer.weight_scale

    layer.to(dtype=torch.bfloat16)

    self.assertEqual(layer.weight.dtype, FP8_WEIGHT_DTYPE)
    self.assertEqual(layer.weight_scale.dtype, torch.float32)
    self.assertEqual(layer.bias.dtype, torch.bfloat16)
    self.assertIs(layer.weight, weight)
    self.assertIs(layer.weight_scale, weight_scale)

  def test_forward_requires_fp8_weight(self):
    layer = Fp8Linear(4, 3, bias=False, compute_dtype=torch.bfloat16)
    layer.weight = layer.weight.to(dtype=torch.bfloat16)
    x = torch.randn(2, 4, dtype=torch.bfloat16)

    with mock.patch.object(quantized_loading, "_scaled_mm_supported", return_value=True):
      with mock.patch.object(quantized_loading._Fp8LinearScaledMm, "apply") as scaled_mm:
        with self.assertRaisesRegex(RuntimeError, "Fp8Linear weight must be"):
          layer(x)

    scaled_mm.assert_not_called()

  def test_scaled_mm_uses_supported_kernel_dtype_for_float32_input(self):
    x = torch.randn(2, 4, dtype=torch.float32)
    weight = torch.zeros(3, 4, dtype=FP8_WEIGHT_DTYPE)
    weight_scale = torch.ones(3, dtype=torch.float32)
    bias = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    calls = []

    def fake_scaled_mm(*args, **kwargs):
      calls.append(kwargs)
      return torch.zeros(2, 3, dtype=kwargs["out_dtype"])

    with mock.patch.object(torch, "_scaled_mm", side_effect=fake_scaled_mm):
      out = quantized_loading._Fp8LinearScaledMm.apply(x, weight, weight_scale, bias, 3)

    self.assertEqual(calls[0]["bias"].dtype, torch.bfloat16)
    self.assertEqual(calls[0]["out_dtype"], torch.bfloat16)
    self.assertEqual(calls[0]["scale_a"].shape, (2, 1))
    self.assertEqual(calls[0]["scale_a"].stride(0), 1)
    self.assertTrue(calls[0]["scale_a"].is_contiguous())
    self.assertEqual(out.dtype, torch.float32)

  def test_scaled_mm_layout_error_falls_back_to_dequantized_linear(self):
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    weight = torch.zeros(3, 4, dtype=FP8_WEIGHT_DTYPE)
    weight_scale = torch.ones(3, dtype=torch.float32)

    with mock.patch.object(torch, "_scaled_mm", side_effect=RuntimeError("Expected scale_a.stride(0) == 1")):
      out = quantized_loading._Fp8LinearScaledMm.apply(x, weight, weight_scale, None, 3)

    self.assertEqual(out.shape, (2, 3))
    self.assertEqual(out.dtype, torch.bfloat16)

  def test_scaled_mm_runtime_error_falls_back_to_dequantized_linear(self):
    x = torch.randn(2, 4, dtype=torch.bfloat16)
    weight = torch.zeros(3, 4, dtype=FP8_WEIGHT_DTYPE)
    weight_scale = torch.ones(3, dtype=torch.float32)

    with mock.patch.object(torch, "_scaled_mm", side_effect=RuntimeError("unsupported scaled_mm kernel")):
      out = quantized_loading._Fp8LinearScaledMm.apply(x, weight, weight_scale, None, 3)

    self.assertEqual(out.shape, (2, 3))
    self.assertEqual(out.dtype, torch.bfloat16)

  @unittest.skipUnless(torch.cuda.is_available() and hasattr(torch, "_scaled_mm"), "CUDA _scaled_mm required")
  def test_scaled_mm_cuda_forward_supports_float32_input_with_bias(self):
    layer = Fp8Linear(16, 16, bias=True, compute_dtype=torch.float32).cuda()
    weight = torch.randn(16, 16, dtype=torch.bfloat16)
    quantized_weight, weight_scale = quantize_weight_to_fp8(weight)
    layer.weight = quantized_weight.cuda()
    layer.weight_scale = weight_scale.cuda()
    layer.bias = torch.randn(16, device="cuda", dtype=torch.float32)
    x = torch.randn(8, 16, device="cuda", dtype=torch.float32, requires_grad=True)

    with mock.patch.object(quantized_loading, "_scaled_mm_supported", return_value=True):
      out = layer(x)
      out.sum().backward()

    self.assertEqual(out.shape, (8, 16))
    self.assertEqual(out.dtype, torch.float32)
    self.assertEqual(x.grad.shape, x.shape)


if __name__ == "__main__":
  unittest.main()
