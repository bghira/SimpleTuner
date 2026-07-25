from __future__ import annotations

import sys
import types
import unittest
from inspect import signature
from unittest.mock import patch

import torch

from simpletuner.helpers.training.sdnq_compat import apply_sdnq_checkpointed_backward_fix

MODULE_NAMES = (
    "sdnq.training.layers.linear.linear_int8.linear_int8_ckpt",
    "sdnq.training.layers.linear.linear_int8.linear_int8_dynamic_ckpt",
    "sdnq.training.layers.linear.linear_uint8.linear_uint8_ckpt",
    "sdnq.training.layers.linear.linear_uint8.linear_uint8_dynamic_ckpt",
    "sdnq.training.layers.linear.linear_fp8.linear_fp8_ckpt",
    "sdnq.training.layers.linear.linear_fp8.linear_fp8_dynamic_ckpt",
    "sdnq.training.layers.linear.linear_fp16.linear_fp16_ckpt",
    "sdnq.training.layers.linear.linear_fp16.linear_fp16_dynamic_ckpt",
)


def _install_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    sys.modules[name] = module
    return module


def _matmul(input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
    output = input.flatten(0, -2).to(dtype=torch.float32).matmul(weight.to(dtype=torch.float32).t())
    output_shape = kwargs.get("output_shape")
    if output_shape is not None:
        output = output.reshape(output_shape)
    return output.to(dtype=input.dtype)


def _dynamic_matmul(input: torch.Tensor, weight: torch.Tensor, **kwargs) -> torch.Tensor:
    output = input.flatten(0, -2).to(dtype=torch.float32).matmul(weight.to(dtype=torch.float32))
    output_shape = kwargs.get("output_shape")
    if output_shape is not None:
        output = output.reshape(output_shape)
    return output.to(dtype=input.dtype)


def _quantize_int_mm(input: torch.Tensor, **kwargs):
    return input.round().to(torch.int8), torch.ones((input.shape[0], 1), dtype=torch.float32, device=input.device)


def _quantize_uint_mm(input: torch.Tensor, **kwargs):
    return (
        input.round().clamp_min(0).to(torch.uint8),
        torch.ones((input.shape[0], 1), dtype=torch.float32, device=input.device),
        torch.zeros((input.shape[0], 1), dtype=torch.float32, device=input.device),
    )


def _quantize_fp_mm(input: torch.Tensor, **kwargs):
    return input.to(torch.float16), torch.ones((input.shape[0], 1), dtype=torch.float32, device=input.device)


def _install_fake_sdnq_modules(*, upstream_fixed: bool = False) -> None:
    for package_name in (
        "sdnq",
        "sdnq.training",
        "sdnq.training.layers",
        "sdnq.training.layers.linear",
        "sdnq.training.layers.linear.linear_int8",
        "sdnq.training.layers.linear.linear_uint8",
        "sdnq.training.layers.linear.linear_fp8",
        "sdnq.training.layers.linear.linear_fp16",
    ):
        _install_module(package_name)

    for module_name in MODULE_NAMES:
        module = _install_module(module_name)
        module.compile_func = lambda func: func
        module.SDNQTensor = torch.Tensor
        module.get_hadamard = lambda group_size, dtype, device: torch.eye(group_size, dtype=dtype, device=device)
        module.dequantize_symmetric_compiled = lambda weight, scale: weight.to(dtype=torch.float32)
        module.dequantize_asymmetric_compiled = lambda weight, scale, zero_point: weight.to(dtype=torch.float32)

    int8_static = sys.modules["sdnq.training.layers.linear.linear_int8.linear_int8_ckpt"]
    int8_static.int8_matmul = _matmul
    int8_static.int8_matmul_dynamic = _dynamic_matmul
    int8_static.get_int8_matmul_backward_inputs = lambda input, hadamard: _quantize_int_mm(input.flatten(0, -2))
    if upstream_fixed:
        int8_static.int8_matmul_backward_ckpt = lambda grad_output, input, weight, input_scale, scale, input_shape=None: None
    else:
        int8_static.int8_matmul_backward_ckpt = lambda grad_output, input, weight, input_scale, scale: None

    int8_dynamic = sys.modules["sdnq.training.layers.linear.linear_int8.linear_int8_dynamic_ckpt"]
    int8_dynamic.quantize_int_mm = _quantize_int_mm
    int8_dynamic.int8_matmul = _matmul
    int8_dynamic.int8_matmul_dynamic = _dynamic_matmul
    int8_dynamic.get_int8_matmul_dynamic_backward_inputs = lambda input, weight, hadamard: (
        *_quantize_int_mm(input.flatten(0, -2)),
        *_quantize_int_mm(weight),
    )
    int8_dynamic.int8_matmul_dynamic_backward_ckpt = lambda grad_output, input, weight, input_scale, weight_scale: None

    uint8_static = sys.modules["sdnq.training.layers.linear.linear_uint8.linear_uint8_ckpt"]
    uint8_static.uint8_matmul = _matmul
    uint8_static.uint8_matmul_dynamic = _dynamic_matmul
    uint8_static.get_uint8_matmul_backward_inputs = lambda input, hadamard: _quantize_uint_mm(input.flatten(0, -2))
    uint8_static.uint8_matmul_backward_ckpt = (
        lambda grad_output, input, weight, input_scale, scale, input_zero_point, zero_point: None
    )

    uint8_dynamic = sys.modules["sdnq.training.layers.linear.linear_uint8.linear_uint8_dynamic_ckpt"]
    uint8_dynamic.quantize_uint_mm = _quantize_uint_mm
    uint8_dynamic.uint8_matmul = _matmul
    uint8_dynamic.uint8_matmul_dynamic = _dynamic_matmul
    uint8_dynamic.get_uint8_matmul_dynamic_backward_inputs = lambda input, weight, hadamard: (
        *_quantize_uint_mm(input.flatten(0, -2)),
        *_quantize_uint_mm(weight),
    )
    uint8_dynamic.uint8_matmul_dynamic_backward_ckpt = (
        lambda grad_output, input, weight, input_scale, weight_scale, input_zero_point, weight_zero_point: None
    )

    for dtype_name in ("fp8", "fp16"):
        static = sys.modules[f"sdnq.training.layers.linear.linear_{dtype_name}.linear_{dtype_name}_ckpt"]
        setattr(static, f"{dtype_name}_matmul", _matmul)
        setattr(static, f"{dtype_name}_matmul_dynamic", _dynamic_matmul)
        setattr(static, f"{dtype_name}_matmul_backward_ckpt", lambda grad_output, input, weight, input_scale, scale: None)
        static.quantize_fp_mm = _quantize_fp_mm

        dynamic = sys.modules[f"sdnq.training.layers.linear.linear_{dtype_name}.linear_{dtype_name}_dynamic_ckpt"]
        setattr(dynamic, f"{dtype_name}_matmul", _matmul)
        setattr(dynamic, f"{dtype_name}_matmul_dynamic", _dynamic_matmul)
        setattr(
            dynamic,
            f"get_{dtype_name}_matmul_dynamic_backward_inputs",
            lambda input, weight, hadamard: (*_quantize_fp_mm(input.flatten(0, -2)), *_quantize_fp_mm(weight)),
        )
        setattr(
            dynamic,
            f"{dtype_name}_matmul_dynamic_backward_ckpt",
            lambda grad_output, input, weight, input_scale, weight_scale: None,
        )
        dynamic.quantize_fp_mm = _quantize_fp_mm


class SDNQCompatTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_modules = {name: sys.modules.get(name) for name in MODULE_NAMES}

    def tearDown(self) -> None:
        for name in list(sys.modules):
            if name == "sdnq" or name.startswith("sdnq."):
                sys.modules.pop(name)
        for name, module in self.original_modules.items():
            if module is not None:
                sys.modules[name] = module

    def test_version_gate_skips_old_sdnq(self):
        _install_fake_sdnq_modules()
        with patch("simpletuner.helpers.training.sdnq_compat.metadata.version", return_value="0.2.1"):
            self.assertFalse(apply_sdnq_checkpointed_backward_fix())

    def test_upstream_fixed_sdnq_is_left_unpatched(self):
        _install_fake_sdnq_modules(upstream_fixed=True)
        int8_static = sys.modules["sdnq.training.layers.linear.linear_int8.linear_int8_ckpt"]
        original = int8_static.int8_matmul_backward_ckpt
        with patch("simpletuner.helpers.training.sdnq_compat.metadata.version", return_value="0.2.2"):
            self.assertFalse(apply_sdnq_checkpointed_backward_fix())
        self.assertIs(int8_static.int8_matmul_backward_ckpt, original)

    def test_patch_is_idempotent_and_updates_checkpoint_signatures(self):
        _install_fake_sdnq_modules()
        with patch("simpletuner.helpers.training.sdnq_compat.metadata.version", return_value="0.2.2"):
            self.assertTrue(apply_sdnq_checkpointed_backward_fix())
            self.assertFalse(apply_sdnq_checkpointed_backward_fix())

        function_names = {
            "sdnq.training.layers.linear.linear_int8.linear_int8_ckpt": "int8_matmul_backward_ckpt",
            "sdnq.training.layers.linear.linear_int8.linear_int8_dynamic_ckpt": "int8_matmul_dynamic_backward_ckpt",
            "sdnq.training.layers.linear.linear_uint8.linear_uint8_ckpt": "uint8_matmul_backward_ckpt",
            "sdnq.training.layers.linear.linear_uint8.linear_uint8_dynamic_ckpt": "uint8_matmul_dynamic_backward_ckpt",
            "sdnq.training.layers.linear.linear_fp8.linear_fp8_ckpt": "fp8_matmul_backward_ckpt",
            "sdnq.training.layers.linear.linear_fp8.linear_fp8_dynamic_ckpt": "fp8_matmul_dynamic_backward_ckpt",
            "sdnq.training.layers.linear.linear_fp16.linear_fp16_ckpt": "fp16_matmul_backward_ckpt",
            "sdnq.training.layers.linear.linear_fp16.linear_fp16_dynamic_ckpt": "fp16_matmul_dynamic_backward_ckpt",
        }
        for module_name, func_name in function_names.items():
            module = sys.modules[module_name]
            self.assertIn("input_shape", signature(getattr(module, func_name)).parameters)

    def test_frozen_static_int8_weight_skips_backward_input_quantization(self):
        _install_fake_sdnq_modules()
        with patch("simpletuner.helpers.training.sdnq_compat.metadata.version", return_value="0.2.2"):
            self.assertTrue(apply_sdnq_checkpointed_backward_fix())

        int8_static = sys.modules["sdnq.training.layers.linear.linear_int8.linear_int8_ckpt"]
        calls = {"backward_inputs": 0}

        def fail_if_called(input, hadamard):
            calls["backward_inputs"] += 1
            raise AssertionError("input quantization should be skipped for frozen weights")

        int8_static.get_int8_matmul_backward_inputs = fail_if_called

        weight = torch.randn(4, 8)
        weight.weight = weight
        weight.scale = torch.ones(4, 1)
        weight.zero_point = None
        weight.svd_up = None
        weight.svd_down = None
        weight.sdnq_dequantizer = types.SimpleNamespace(use_hadamard=False, hadamard_group_size=128)

        input = torch.randn(2, 8, requires_grad=True)
        output = int8_static.INT8MatmulBackwardCKPT.apply(input, weight, None)
        output.sum().backward()

        self.assertEqual(0, calls["backward_inputs"])
        self.assertEqual(input.shape, input.grad.shape)


if __name__ == "__main__":
    unittest.main()
