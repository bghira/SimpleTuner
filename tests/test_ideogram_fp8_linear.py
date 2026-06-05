import unittest
from unittest import mock

import torch

from simpletuner.helpers.models.ideogram import quantized_loading
from simpletuner.helpers.models.ideogram.quantized_loading import (
  FP8_WEIGHT_DTYPE,
  Fp8Linear,
  _Fp8LinearScaledMm,
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

  def test_scaled_mm_adds_float32_bias_outside_kernel(self):
    x = torch.randn(2, 4, dtype=torch.float32)
    weight = torch.zeros(3, 4, dtype=FP8_WEIGHT_DTYPE)
    weight_scale = torch.ones(3, dtype=torch.float32)
    bias = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    calls = []

    def fake_scaled_mm(*args, **kwargs):
      calls.append(kwargs)
      return torch.zeros(2, 3, dtype=torch.float32)

    with mock.patch.object(torch, "_scaled_mm", side_effect=fake_scaled_mm):
      out = _Fp8LinearScaledMm.apply(x, weight, weight_scale, bias, 3)

    self.assertIsNone(calls[0]["bias"])
    self.assertEqual(calls[0]["out_dtype"], torch.float32)
    torch.testing.assert_close(out, bias.to(torch.float32).expand_as(out))


if __name__ == "__main__":
  unittest.main()
