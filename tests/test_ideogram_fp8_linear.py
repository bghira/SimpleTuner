import unittest
from unittest import mock

import torch

from simpletuner.helpers.models.ideogram import quantized_loading
from simpletuner.helpers.models.ideogram.quantized_loading import (
  FP8_WEIGHT_DTYPE,
  Fp8Linear,
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


if __name__ == "__main__":
  unittest.main()
