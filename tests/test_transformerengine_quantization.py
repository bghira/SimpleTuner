import contextlib
import types
import unittest

import torch


class _StubTELinear(torch.nn.Linear):
    def __init__(self, in_features, out_features, bias=True, params_dtype=None, device=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=params_dtype)


class TestTransformerEnginePrecisionRegistration(unittest.TestCase):
    def test_transformerengine_preset_is_manual(self):
        from simpletuner.helpers.training.quantisation import MANUAL_QUANTIZATION_PRESETS, MANUAL_TRANSFORMERENGINE_PRESETS

        self.assertEqual(MANUAL_TRANSFORMERENGINE_PRESETS, {"fp8-transformerengine"})
        self.assertIn("fp8-transformerengine", MANUAL_QUANTIZATION_PRESETS)

    def test_get_quant_fn_returns_transformerengine_model(self):
        from simpletuner.helpers.training.quantisation import _transformerengine_model, get_quant_fn

        self.assertEqual(get_quant_fn("fp8-transformerengine"), _transformerengine_model)


class TestTransformerEngineQuantizationHelpers(unittest.TestCase):
    def test_replace_linears_preserves_parameters_and_filters_unsupported_shapes(self):
        from simpletuner.helpers.training.quantisation import _replace_linears_with_transformerengine

        model = torch.nn.Sequential(
            torch.nn.Linear(16, 32, bias=True),
            torch.nn.Linear(15, 32, bias=False),
        )
        expected_weight = model[0].weight.detach().clone()
        expected_bias = model[0].bias.detach().clone()
        model[0].weight.requires_grad_(False)

        te = types.SimpleNamespace(Linear=_StubTELinear)
        converted = _replace_linears_with_transformerengine(model, te, torch.bfloat16)

        self.assertEqual(converted, 1)
        self.assertIsInstance(model[0], _StubTELinear)
        self.assertIsInstance(model[1], torch.nn.Linear)
        self.assertEqual(model[0].weight.dtype, torch.bfloat16)
        self.assertFalse(model[0].weight.requires_grad)
        torch.testing.assert_close(model[0].weight.float(), expected_weight, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(model[0].bias.float(), expected_bias, atol=5e-3, rtol=5e-3)

    def test_forward_wrapper_uses_transformerengine_autocast(self):
        from simpletuner.helpers.training.quantisation import _wrap_transformerengine_fp8_forward

        calls = []

        @contextlib.contextmanager
        def autocast(enabled, recipe):
            calls.append((enabled, recipe))
            yield

        te = types.SimpleNamespace(autocast=autocast)
        recipe = object()
        model = torch.nn.Linear(16, 16)
        _wrap_transformerengine_fp8_forward(model, te, recipe)

        result = model(torch.ones(1, 16))

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][0])
        self.assertIs(calls[0][1], recipe)
        self.assertEqual(result.shape, (1, 16))


if __name__ == "__main__":
    unittest.main()
