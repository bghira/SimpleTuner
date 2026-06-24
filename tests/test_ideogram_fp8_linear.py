import unittest
from unittest import mock

import torch

from simpletuner.helpers.models.ideogram import quantized_loading
from simpletuner.helpers.models.ideogram.quantized_loading import (
    FP8_WEIGHT_DTYPE,
    Fp8Linear,
    dequantize_fp8_state_dict,
    device_supports_float8,
    is_fp8_state_dict,
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

    def test_fp8_linear_is_linear_subclass_for_adapter_discovery(self):
        layer = Fp8Linear(4, 3, bias=True, compute_dtype=torch.bfloat16)

        self.assertIsInstance(layer, torch.nn.Linear)

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

    def test_lycoris_algorithms_wrap_fp8_linear_in_bypass_mode(self):
        try:
            from lycoris import LycorisNetwork, create_lycoris
        except ImportError:
            self.skipTest("LyCORIS is not installed")

        class ToyBlock(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = Fp8Linear(16, 16, bias=False, compute_dtype=torch.bfloat16)
                weight = torch.randn(16, 16, dtype=torch.bfloat16)
                quantized_weight, weight_scale = quantize_weight_to_fp8(weight)
                self.proj.weight.copy_(quantized_weight)
                self.proj.weight_scale.copy_(weight_scale)

            def forward(self, x):
                return self.proj(x)

        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.block = ToyBlock()

            def forward(self, x):
                return self.block(x)

        target_module_attr = (
            "UNET_TARGET_REPLACE_MODULE"
            if hasattr(LycorisNetwork, "UNET_TARGET_REPLACE_MODULE")
            else "TARGET_REPLACE_MODULE"
        )
        target_name_attr = (
            "UNET_TARGET_REPLACE_NAME" if hasattr(LycorisNetwork, "UNET_TARGET_REPLACE_NAME") else "TARGET_REPLACE_NAME"
        )
        original_target_modules = list(getattr(LycorisNetwork, target_module_attr))
        original_target_names = list(getattr(LycorisNetwork, target_name_attr))
        try:
            for algo in ("lokr", "loha"):
                with self.subTest(algo=algo):
                    LycorisNetwork.apply_preset({"target_module": ["ToyBlock"], "target_name": []})
                    model = ToyModel()
                    network = create_lycoris(model, 1.0, 4, 1, algo=algo, bypass_mode=True)

                    self.assertEqual(len(network.loras), 1)
                    self.assertIs(network.loras[0].org_module[0], model.block.proj)
                    self.assertTrue(network.loras[0].bypass_mode)

                    network.apply_to()
                    network.to(dtype=torch.bfloat16)
                    x = torch.randn(2, 16, dtype=torch.bfloat16, requires_grad=True)
                    out = model(x)
                    out.float().sum().backward()

                    self.assertEqual(out.shape, (2, 16))
                    self.assertIsNotNone(x.grad)
        finally:
            setattr(LycorisNetwork, target_module_attr, original_target_modules)
            setattr(LycorisNetwork, target_name_attr, original_target_names)


class IdeogramFp8DequantTests(unittest.TestCase):
    def test_dequantize_folds_scales_and_drops_scale_keys(self):
        weight = torch.randn(8, 16, dtype=torch.float32)
        quantized_weight, weight_scale = quantize_weight_to_fp8(weight)
        bias = torch.randn(8, dtype=torch.bfloat16)
        norm = torch.randn(16, dtype=torch.bfloat16)
        state_dict = {
            "layer.weight": quantized_weight,
            "layer.weight_scale": weight_scale,
            "layer.bias": bias,
            "norm.weight": norm,
        }
        self.assertTrue(is_fp8_state_dict(state_dict))

        dequantized = dequantize_fp8_state_dict(state_dict, torch.bfloat16)

        self.assertFalse(is_fp8_state_dict(dequantized))
        self.assertNotIn("layer.weight_scale", dequantized)
        self.assertEqual(dequantized["layer.weight"].dtype, torch.bfloat16)
        # Non-fp8 tensors pass through untouched.
        self.assertIs(dequantized["layer.bias"], bias)
        self.assertIs(dequantized["norm.weight"], norm)
        # Scale is folded in float32 (full precision) before the final cast.
        expected = (quantized_weight.to(torch.float32) * weight_scale.unsqueeze(1)).to(torch.bfloat16)
        self.assertTrue(torch.equal(dequantized["layer.weight"], expected))

    def test_dequantized_state_dict_loads_into_plain_linear(self):
        model = torch.nn.Sequential()
        model.add_module("proj", torch.nn.Linear(4, 3, bias=True))
        weight = torch.randn(3, 4, dtype=torch.float32)
        quantized_weight, weight_scale = quantize_weight_to_fp8(weight)
        state_dict = {
            "proj.weight": quantized_weight,
            "proj.weight_scale": weight_scale,
            "proj.bias": torch.randn(3),
        }

        dequantized = dequantize_fp8_state_dict(state_dict, torch.float32)
        missing, unexpected = model.load_state_dict(dequantized, strict=False)

        self.assertEqual(unexpected, [])
        self.assertEqual(missing, [])
        self.assertEqual(model.proj.weight.dtype, torch.float32)

    def test_device_supports_float8_cpu(self):
        # CPU can cast float8 to the compute dtype, so the fast FP8 path stays active.
        self.assertTrue(device_supports_float8(torch.device("cpu")))


if __name__ == "__main__":
    unittest.main()
