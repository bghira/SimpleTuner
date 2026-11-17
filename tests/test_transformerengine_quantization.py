import contextlib
import importlib.util
import os
import types
import unittest
from collections import OrderedDict
from unittest import mock

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
    def test_mark_transformerengine_ddp_ignore_params_marks_frozen_float8_tensors(self):
        from simpletuner.helpers.training.quantisation import mark_transformerengine_ddp_ignore_params

        _FakeParam = type(
            "Float8Tensor",
            (),
            {
                "__module__": "transformer_engine.pytorch.tensor.float8_tensor",
                "requires_grad": False,
            },
        )
        _FakeTrainableParam = type(
            "Float8Tensor",
            (),
            {
                "__module__": "transformer_engine.pytorch.tensor.float8_tensor",
                "requires_grad": True,
            },
        )

        class _FakeModule:
            def named_parameters(self):
                return [
                    ("frozen.weight", _FakeParam()),
                    ("wrapped.weight", types.SimpleNamespace(data=_FakeParam(), requires_grad=False)),
                    ("trainable.weight", _FakeTrainableParam()),
                    ("regular.weight", torch.nn.Parameter(torch.ones(1))),
                ]

            def named_buffers(self):
                return [("frozen.scale", _FakeParam())]

        model = _FakeModule()

        ignored = mark_transformerengine_ddp_ignore_params(model)

        self.assertEqual(ignored, 3)
        ignore_set = getattr(model, "_ddp_params_and_buffers_to_ignore")
        self.assertIn("frozen.weight", ignore_set)
        self.assertIn("wrapped.weight", ignore_set)
        self.assertIn("frozen.scale", ignore_set)
        self.assertNotIn("trainable.weight", ignore_set)

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

    def test_replace_linears_initializes_fp8_weights_with_the_forward_recipe(self):
        from simpletuner.helpers.training.quantisation import _replace_linears_with_transformerengine

        calls = []

        @contextlib.contextmanager
        def quantized_model_init(enabled=True, recipe=None):
            calls.append((enabled, recipe))
            yield

        model = torch.nn.Sequential(torch.nn.Linear(16, 32, bias=False))
        te = types.SimpleNamespace(Linear=_StubTELinear, quantized_model_init=quantized_model_init)
        recipe = object()

        converted = _replace_linears_with_transformerengine(model, te, torch.bfloat16, recipe, fp8_model_init_enabled=True)

        self.assertEqual(converted, 1)
        self.assertEqual(calls, [(True, recipe)])

    def test_replace_linears_does_not_wrap_converted_linear_forward_in_fp8_autocast(self):
        from simpletuner.helpers.training.quantisation import _replace_linears_with_transformerengine

        calls = []

        @contextlib.contextmanager
        def fp8_autocast(enabled, fp8_recipe):
            calls.append((enabled, fp8_recipe))
            yield

        recipe = object()
        model = torch.nn.Sequential(torch.nn.Linear(16, 32, bias=False))
        te = types.SimpleNamespace(Linear=_StubTELinear, fp8_autocast=fp8_autocast)

        converted = _replace_linears_with_transformerengine(model, te, torch.bfloat16, recipe=recipe)
        result = model(torch.ones(1, 16, dtype=torch.bfloat16))

        self.assertEqual(converted, 1)
        self.assertEqual(calls, [])
        self.assertEqual(result.shape, (1, 32))

    def test_checkpoint_context_attaches_transformerengine_autocast_to_module_tree(self):
        from simpletuner.helpers.training.quantisation import _attach_transformerengine_checkpoint_context

        calls = []

        @contextlib.contextmanager
        def fp8_autocast(enabled, fp8_recipe):
            calls.append((enabled, fp8_recipe))
            yield

        te = types.SimpleNamespace(fp8_autocast=fp8_autocast)
        recipe = object()
        model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU())

        _attach_transformerengine_checkpoint_context(model, te, recipe)

        context_fn = getattr(model[0], "_simpletuner_te_checkpoint_context_fn")
        forward_context, recompute_context = context_fn()
        with forward_context:
            pass
        with recompute_context:
            pass

        self.assertIs(getattr(model, "_simpletuner_te_checkpoint_context_fn"), context_fn)
        self.assertEqual(len(calls), 2)
        self.assertTrue(calls[0][0])
        self.assertIs(calls[0][1], recipe)
        self.assertTrue(calls[1][0])
        self.assertIs(calls[1][1], recipe)

    def test_frozen_transformerengine_linear_uses_weight_cache_microbatch_hint(self):
        from simpletuner.helpers.training.quantisation import _replace_linears_with_transformerengine

        calls = []

        class _CacheTELinear(_StubTELinear):
            def forward(self, inp, is_first_microbatch=None, fp8_output=False, fp8_grad=False):
                calls.append(is_first_microbatch)
                return super().forward(inp)

        model = torch.nn.Sequential(torch.nn.Linear(16, 32, bias=False))
        model[0].weight.requires_grad_(False)
        te = types.SimpleNamespace(Linear=_CacheTELinear)
        previous_weight_cache = os.environ.get("SIMPLETUNER_TE_FROZEN_WEIGHT_CACHE")

        try:
            os.environ["SIMPLETUNER_TE_FROZEN_WEIGHT_CACHE"] = "1"
            converted = _replace_linears_with_transformerengine(model, te, torch.bfloat16)
            model(torch.ones(1, 16, dtype=torch.bfloat16))
            model(torch.ones(1, 16, dtype=torch.bfloat16))
        finally:
            if previous_weight_cache is None:
                os.environ.pop("SIMPLETUNER_TE_FROZEN_WEIGHT_CACHE", None)
            else:
                os.environ["SIMPLETUNER_TE_FROZEN_WEIGHT_CACHE"] = previous_weight_cache

        self.assertEqual(converted, 1)
        self.assertEqual(calls, [True, False])

    def test_attention_qkv_transformerengine_linears_request_fp8_output_when_enabled(self):
        from simpletuner.helpers.training.quantisation import _replace_linears_with_transformerengine
        from simpletuner.helpers.training.state_tracker import StateTracker

        calls = []

        class _Fp8OutputTELinear(_StubTELinear):
            def forward(self, inp, is_first_microbatch=None, fp8_output=False, fp8_grad=False):
                calls.append(fp8_output)
                return super().forward(inp)

        model = torch.nn.Module()
        model.to_q = torch.nn.Linear(16, 32, bias=False)
        model.to_out = torch.nn.ModuleList([torch.nn.Linear(16, 32, bias=False)])
        previous_args = StateTracker.get_args()
        previous_fp8_output = os.environ.get("SIMPLETUNER_TE_FP8_ATTENTION_OUTPUT")
        try:
            StateTracker.set_args(types.SimpleNamespace(model_type="full"))
            os.environ["SIMPLETUNER_TE_FP8_ATTENTION_OUTPUT"] = "1"
            te = types.SimpleNamespace(Linear=_Fp8OutputTELinear)

            converted = _replace_linears_with_transformerengine(model, te, torch.bfloat16, prefix="blocks.0.attn")
            model.to_q(torch.ones(1, 16, dtype=torch.bfloat16))
            model.to_out[0](torch.ones(1, 16, dtype=torch.bfloat16))
        finally:
            StateTracker.set_args(previous_args)
            if previous_fp8_output is None:
                os.environ.pop("SIMPLETUNER_TE_FP8_ATTENTION_OUTPUT", None)
            else:
                os.environ["SIMPLETUNER_TE_FP8_ATTENTION_OUTPUT"] = previous_fp8_output

        self.assertEqual(converted, 2)
        self.assertEqual(calls, [True, False])

    def test_transformerengine_filter_skips_ltx_paths_that_are_not_regular_video_linears(self):
        from simpletuner.helpers.training.quantisation import _transformerengine_filter_fn
        from simpletuner.helpers.training.state_tracker import StateTracker

        with mock.patch.object(StateTracker, "get_args", return_value=types.SimpleNamespace(model_type="lora")):
            self.assertFalse(_transformerengine_filter_fn(torch.nn.Linear(128, 4096), "audio_proj_in"))
            self.assertFalse(
                _transformerengine_filter_fn(torch.nn.Linear(4096, 4096), "transformer_blocks.0.audio_attn1.to_q")
            )
            self.assertFalse(
                _transformerengine_filter_fn(torch.nn.Linear(256, 4096), "prompt_adaln.emb.timestep_embedder.linear_1")
            )
            self.assertFalse(_transformerengine_filter_fn(torch.nn.Linear(4096, 8192), "audio_prompt_adaln.linear"))
            self.assertFalse(_transformerengine_filter_fn(torch.nn.Linear(4096, 4096), "proj_in"))
            self.assertTrue(_transformerengine_filter_fn(torch.nn.Linear(4096, 4096), "transformer_blocks.0.attn1.to_q"))
            self.assertTrue(_transformerengine_filter_fn(torch.nn.Linear(4096, 4096), "transformer_blocks.0.attn2.to_out.0"))

    def test_transformerengine_filter_can_expand_lora_scope_for_benchmarks(self):
        from simpletuner.helpers.training.quantisation import _transformerengine_filter_fn
        from simpletuner.helpers.training.state_tracker import StateTracker

        previous_override = os.environ.get("SIMPLETUNER_TE_LORA_CONVERT_ALL")
        try:
            with mock.patch.object(StateTracker, "get_args", return_value=types.SimpleNamespace(model_type="lora")):
                self.assertTrue(
                    _transformerengine_filter_fn(torch.nn.Linear(4096, 4096), "transformer_blocks.0.attn1.to_qkv")
                )
                self.assertTrue(
                    _transformerengine_filter_fn(torch.nn.Linear(4096, 4096), "transformer_blocks.0.attn2.to_kv")
                )
                self.assertTrue(
                    _transformerengine_filter_fn(torch.nn.Linear(4096, 4096), "transformer_blocks.0.attn.to_added_qkv")
                )
                os.environ["SIMPLETUNER_TE_LORA_CONVERT_ALL"] = "1"
                self.assertTrue(
                    _transformerengine_filter_fn(torch.nn.Linear(4096, 4096), "transformer_blocks.0.attn1.to_qkv")
                )
                self.assertTrue(
                    _transformerengine_filter_fn(torch.nn.Linear(4096, 4096), "transformer_blocks.0.attn1.to_out.0")
                )
                self.assertTrue(_transformerengine_filter_fn(torch.nn.Linear(16384, 4096), "transformer_blocks.0.ff.net.2"))
                self.assertTrue(_transformerengine_filter_fn(torch.nn.Linear(4096, 512), "proj_out"))
        finally:
            if previous_override is None:
                os.environ.pop("SIMPLETUNER_TE_LORA_CONVERT_ALL", None)
            else:
                os.environ["SIMPLETUNER_TE_LORA_CONVERT_ALL"] = previous_override

    def test_forward_wrapper_uses_transformerengine_autocast(self):
        from simpletuner.helpers.training.quantisation import _wrap_transformerengine_fp8_forward

        calls = []

        @contextlib.contextmanager
        def fp8_autocast(enabled, fp8_recipe):
            calls.append((enabled, fp8_recipe))
            yield

        te = types.SimpleNamespace(fp8_autocast=fp8_autocast)
        recipe = object()
        model = torch.nn.Linear(16, 16)
        _wrap_transformerengine_fp8_forward(model, te, recipe)

        result = model(torch.ones(1, 16))

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0][0])
        self.assertIs(calls[0][1], recipe)
        self.assertEqual(result.shape, (1, 16))

    def test_forward_wrapper_falls_back_to_transformerengine_autocast(self):
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


@unittest.skipUnless(
    torch.cuda.is_available()
    and torch.cuda.get_device_capability() >= (8, 9)
    and importlib.util.find_spec("transformer_engine") is not None,
    "TransformerEngine and an Ada or newer CUDA GPU are required",
)
class TestTransformerEngineCuda(unittest.TestCase):
    def test_fp8_backward_with_trainable_and_frozen_weights(self):
        from simpletuner.helpers.training.quantisation import _transformerengine_model
        from simpletuner.helpers.training.state_tracker import StateTracker

        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                model = (
                    torch.nn.Sequential(OrderedDict(attn=torch.nn.Sequential(OrderedDict(to_q=torch.nn.Linear(128, 128)))))
                    .cuda()
                    .bfloat16()
                )
                model.requires_grad_(not frozen)
                args = types.SimpleNamespace(model_type="lora" if frozen else "full")
                with mock.patch.object(StateTracker, "get_args", return_value=args):
                    _transformerengine_model(model, "fp8-transformerengine")
                inputs = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
                output = model(inputs)
                output.float().square().mean().backward()
                self.assertTrue(torch.isfinite(output).all())
                self.assertIsNotNone(inputs.grad)
                self.assertTrue(torch.isfinite(inputs.grad).all())
                if frozen:
                    self.assertIsNone(model.attn.to_q.weight.grad)
                else:
                    self.assertIsNotNone(model.attn.to_q.weight.grad)
                    self.assertTrue(torch.isfinite(model.attn.to_q.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
