import unittest

import torch

from simpletuner.helpers.training.quantisation import (
    _torchao_filter_fn,
    _torchao_model,
    get_pipeline_quantization_builder,
    get_quant_fn,
    mark_torchao_ddp_ignore_params,
)
from simpletuner.helpers.training.quantisation.fp8_native import Fp8NativeLinear, patch_peft_fp8_native_dispatcher


class TestTorchAoPipelineQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from torchao.quantization.quant_api import (
                AOBaseConfig,
                Float8DynamicActivationFloat8WeightConfig,
                Float8DynamicActivationInt4WeightConfig,
                Float8WeightOnlyConfig,
                Int4WeightOnlyConfig,
                Int8DynamicActivationIntxWeightConfig,
                Int8WeightOnlyConfig,
            )
        except ImportError as exc:
            raise unittest.SkipTest(f"torchao config classes are unavailable: {exc}") from exc

        cls.AOBaseConfig = AOBaseConfig
        cls.Float8DynamicActivationFloat8WeightConfig = Float8DynamicActivationFloat8WeightConfig
        cls.Float8DynamicActivationInt4WeightConfig = Float8DynamicActivationInt4WeightConfig
        cls.Float8WeightOnlyConfig = Float8WeightOnlyConfig
        cls.Int4WeightOnlyConfig = Int4WeightOnlyConfig
        cls.Int8DynamicActivationIntxWeightConfig = Int8DynamicActivationIntxWeightConfig
        cls.Int8WeightOnlyConfig = Int8WeightOnlyConfig

    def test_torchao_pipeline_presets_build_ao_config_instances(self):
        expected_quant_types = {
            "int4-torchao": self.Int4WeightOnlyConfig,
            "int8-torchao": self.Int8WeightOnlyConfig,
            "fp8-torchao": self.Float8DynamicActivationFloat8WeightConfig,
            "fp8wo-torchao": self.Float8WeightOnlyConfig,
        }

        for preset, expected_quant_type in expected_quant_types.items():
            with self.subTest(preset=preset):
                builder = get_pipeline_quantization_builder(preset)
                config = builder(component_type="diffusers")

                self.assertIsInstance(config.quant_type, self.AOBaseConfig)
                self.assertIsInstance(config.quant_type, expected_quant_type)

    def test_torchao_pipeline_quant_type_kwargs_apply_to_inner_config(self):
        builder = get_pipeline_quantization_builder("int8-torchao")
        config = builder(
            overrides={
                "quant_type": "int4_weight_only",
                "quant_type_kwargs": {"group_size": 64},
                "modules_to_not_convert": ["proj_out"],
            },
            component_type="diffusers",
        )

        self.assertIsInstance(config.quant_type, self.Int4WeightOnlyConfig)
        self.assertEqual(config.quant_type.group_size, 64)
        self.assertEqual(config.modules_to_not_convert, ["proj_out"])

    def test_torchao_pipeline_top_level_quant_kwargs_are_preserved(self):
        builder = get_pipeline_quantization_builder("int8-torchao")
        config = builder(
            overrides={
                "group_size": 64,
                "include_input_output_embeddings": True,
            },
            component_type="transformers",
        )

        self.assertIsInstance(config.quant_type, self.Int8WeightOnlyConfig)
        self.assertEqual(config.quant_type.group_size, 64)
        self.assertTrue(config.include_input_output_embeddings)

    def test_torchao_pipeline_supports_dynamic_activation_int4_weight_configs(self):
        cases = {
            "float8_dynamic_activation_int4_weight": self.Float8DynamicActivationInt4WeightConfig,
            "int8_dynamic_activation_intx_weight": self.Int8DynamicActivationIntxWeightConfig,
        }

        for quant_type, expected_type in cases.items():
            with self.subTest(quant_type=quant_type):
                builder = get_pipeline_quantization_builder("int4-torchao")
                config = builder(
                    overrides={
                        "quant_type": quant_type,
                        "quant_type_kwargs": {"weight_dtype": "int4"} if quant_type.startswith("int8_") else {},
                    },
                    component_type="diffusers",
                )

                self.assertIsInstance(config.quant_type, expected_type)
                if quant_type.startswith("int8_"):
                    self.assertEqual(str(config.quant_type.weight_dtype), "torch.int4")

    def test_torchao_dynamic_presets_build_expected_configs(self):
        from torchao.quantization.quant_api import (
            Float8DynamicActivationFloat8WeightConfig,
            Int8DynamicActivationInt8WeightConfig,
        )

        expected_quant_types = {
            "int8dq-torchao": Int8DynamicActivationInt8WeightConfig,
            "int8dq-int4-torchao": self.Int8DynamicActivationIntxWeightConfig,
            "fp8-int4-torchao": self.Float8DynamicActivationInt4WeightConfig,
        }

        for preset, expected_quant_type in expected_quant_types.items():
            with self.subTest(preset=preset):
                builder = get_pipeline_quantization_builder(preset)
                config = builder(component_type="diffusers")

                self.assertIsInstance(config.quant_type, expected_quant_type)
                if preset == "int8dq-int4-torchao":
                    self.assertEqual(str(config.quant_type.weight_dtype), "torch.int4")
                if preset == "int8dq-torchao":
                    self.assertEqual(config.quant_type.version, 2)

    def test_torchao_filter_only_matches_linear_modules(self):
        self.assertFalse(_torchao_filter_fn(torch.nn.Sequential(torch.nn.Linear(16, 16)), ""))
        self.assertTrue(_torchao_filter_fn(torch.nn.Linear(16, 16), "to_q"))
        self.assertTrue(_torchao_filter_fn(torch.nn.Linear(16, 16), "proj_out"))
        self.assertFalse(_torchao_filter_fn(torch.nn.Linear(15, 16), "to_q"))

    def test_fp8_native_is_manual_only(self):
        self.assertIsNotNone(get_quant_fn("fp8-native"))
        self.assertIsNone(get_pipeline_quantization_builder("fp8-native"))

    @unittest.skipUnless(torch.cuda.is_available(), "native fp8 test requires CUDA")
    def test_manual_fp8_native_replaces_linear_with_scaled_mm_module(self):
        model = torch.nn.Sequential(torch.nn.Linear(32, 32, bias=False, device="cuda", dtype=torch.bfloat16))

        get_quant_fn("fp8-native")(model, "fp8-native", base_model_precision="fp8-native")

        self.assertIsInstance(model[0], Fp8NativeLinear)
        self.assertEqual(model[0].weight.dtype, torch.float8_e4m3fn)
        x = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = model(x).sum()
        y.backward()
        self.assertIsNotNone(x.grad)

    @unittest.skipUnless(torch.cuda.is_available(), "native fp8 PEFT test requires CUDA")
    def test_peft_lora_can_wrap_fp8_native_linear(self):
        from peft import LoraConfig, get_peft_model

        patch_peft_fp8_native_dispatcher()
        linear = torch.nn.Linear(32, 32, bias=False, device="cuda", dtype=torch.bfloat16)
        model = torch.nn.Sequential(
            Fp8NativeLinear(32, 32, None, linear.weight, torch.bfloat16).to("cuda"),
            torch.nn.LayerNorm(32, device="cuda", dtype=torch.bfloat16),
        )

        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=4, target_modules=["0"]))
        x = torch.randn(2, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        y = model(x).sum()
        y.backward()

        self.assertIsNotNone(x.grad)
        self.assertIsInstance(model.base_model.model[0].base_layer, Fp8NativeLinear)

    @unittest.skipUnless(torch.cuda.is_available(), "TorchAO DDP ignore test requires CUDA")
    def test_mark_torchao_ddp_ignore_params_marks_frozen_tensor_subclasses(self):
        from torchao.quantization import quantize_
        from torchao.quantization.quant_api import Int8DynamicActivationInt8WeightConfig

        model = torch.nn.Sequential(torch.nn.Linear(32, 32, device="cuda", dtype=torch.bfloat16))
        quantize_(model, Int8DynamicActivationInt8WeightConfig())
        model.requires_grad_(False)

        ignored = mark_torchao_ddp_ignore_params(model)

        self.assertGreater(ignored, 0)
        self.assertIn("0.weight", getattr(model, "_ddp_params_and_buffers_to_ignore"))

    @unittest.skipUnless(torch.cuda.is_available(), "TorchAO manual fp8 test requires CUDA")
    def test_manual_fp8wo_torchao_uses_persistent_float8_weight_tensor(self):
        from torchao.quantization import Float8Tensor

        model = torch.nn.Sequential(torch.nn.Linear(32, 32, bias=False, device="cuda", dtype=torch.bfloat16))

        _torchao_model(model, "fp8wo-torchao", base_model_precision="fp8wo-torchao")

        self.assertIsInstance(model[0].weight, Float8Tensor)


if __name__ == "__main__":
    unittest.main()
