import unittest

from simpletuner.helpers.training.quantisation import get_pipeline_quantization_builder


class TestTorchAoPipelineQuantization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from torchao.quantization.quant_api import (
                AOBaseConfig,
                Float8DynamicActivationInt4WeightConfig,
                Float8WeightOnlyConfig,
                Int4WeightOnlyConfig,
                Int8DynamicActivationIntxWeightConfig,
                Int8WeightOnlyConfig,
            )
        except ImportError as exc:
            raise unittest.SkipTest(f"torchao config classes are unavailable: {exc}") from exc

        cls.AOBaseConfig = AOBaseConfig
        cls.Float8DynamicActivationInt4WeightConfig = Float8DynamicActivationInt4WeightConfig
        cls.Float8WeightOnlyConfig = Float8WeightOnlyConfig
        cls.Int4WeightOnlyConfig = Int4WeightOnlyConfig
        cls.Int8DynamicActivationIntxWeightConfig = Int8DynamicActivationIntxWeightConfig
        cls.Int8WeightOnlyConfig = Int8WeightOnlyConfig

    def test_torchao_pipeline_presets_build_ao_config_instances(self):
        expected_quant_types = {
            "int4-torchao": self.Int4WeightOnlyConfig,
            "int8-torchao": self.Int8WeightOnlyConfig,
            "fp8-torchao": self.Float8WeightOnlyConfig,
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


if __name__ == "__main__":
    unittest.main()
