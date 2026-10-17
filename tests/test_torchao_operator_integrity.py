import unittest
from types import SimpleNamespace

import torch
from torch.fx.experimental.proxy_tensor import make_fx

from simpletuner.helpers.training.quantisation import torchao_workarounds


class TorchAOOperatorIntegrityTests(unittest.TestCase):
    def test_frozen_float8_bias_gradient_preserves_unbatched_output_shape(self):
        context = SimpleNamespace(
            saved_tensors=(torch.empty(0), torch.empty(0)),
            needs_input_grad=(False, False, True),
            has_bias=True,
        )
        gradient = torch.randn(64)
        grad_input, grad_weight, grad_bias = torchao_workarounds._FrozenFloat8AddmmGrad.backward(context, gradient)
        self.assertIsNone(grad_input)
        self.assertIsNone(grad_weight)
        torch.testing.assert_close(grad_bias, gradient)

    def test_scaled_mm_retains_native_dispatch_and_schema_metadata(self):
        operator = torch.ops.aten._scaled_mm.default
        self.assertIsInstance(operator, torch._ops.OpOverload)
        self.assertEqual(operator._schema.name, "aten::_scaled_mm")
        self.assertTrue(hasattr(operator, "tags"))
        self.assertTrue(hasattr(operator, "_dispatch_cache"))

    def test_scaled_mm_can_be_captured_with_meta_tensors(self):
        def scaled_mm(left, right, left_scale, right_scale):
            return torch.ops.aten._scaled_mm.default(left, right, left_scale, right_scale, out_dtype=torch.bfloat16)

        inputs = (
            torch.empty(16, 16, device="meta", dtype=torch.float8_e4m3fn),
            torch.empty(16, 16, device="meta", dtype=torch.float8_e4m3fn).t(),
            torch.ones(1, device="meta"),
            torch.ones(1, device="meta"),
        )
        graph = make_fx(scaled_mm)(*inputs)
        operators = [node.target for node in graph.graph.nodes if node.op == "call_function"]
        self.assertIn(torch.ops.aten._scaled_mm.default, operators)
        self.assertEqual(graph(*inputs).shape, (16, 16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA FP8 kernels required")
    def test_compiled_frozen_float8_linear_preserves_input_gradients(self):
        from simpletuner.helpers.training.quantisation import _torchao_model

        if torch.cuda.get_device_capability() < (8, 9):
            self.skipTest("Native FP8 scaled matmul requires compute capability 8.9 or newer")
        with torch._dynamo.config.patch(cache_size_limit=64):
            for precision in ("fp8-torchao", "fp8wo-torchao"):
                for shape in ((16, 64), (2, 8, 64)):
                    for bias in (False, True):
                        with self.subTest(precision=precision, shape=shape, bias=bias):
                            torch.manual_seed(321)
                            model = torch.nn.Sequential(torch.nn.Linear(64, 64, bias=bias)).to(
                                device="cuda", dtype=torch.bfloat16
                            )
                            model.requires_grad_(False)
                            _torchao_model(model, precision, base_model_precision=precision)
                            self.assertNotEqual(type(model[0].weight), torch.nn.Parameter)
                            if bias:
                                model[0].bias.requires_grad_(True)
                            inputs = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                            with torch.no_grad():
                                expected = torch.ones_like(inputs).matmul(model[0].weight.dequantize())
                            compiled = torch.compile(model, backend="eager", fullgraph=True)
                            compiled(inputs).sum().backward()
                            self.assertIsNotNone(inputs.grad)
                            torch.testing.assert_close(inputs.grad, expected, rtol=0.03, atol=0.005)
                            if bias:
                                expected_bias = torch.full_like(model[0].bias, inputs.numel() // inputs.shape[-1])
                                torch.testing.assert_close(model[0].bias.grad, expected_bias)
