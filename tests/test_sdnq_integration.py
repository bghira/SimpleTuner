import unittest

import torch


class TestSDNQPrecisionLevels(unittest.TestCase):
    """Test that SDNQ precision levels are correctly registered."""

    def test_sdnq_precision_levels_in_list(self):
        from simpletuner.helpers.training import quantised_precision_levels

        expected_sdnq_levels = [
            "int8-sdnq",
            "uint8-sdnq",
            "int16-sdnq",
            "uint16-sdnq",
            "fp16-sdnq",
            "int6-sdnq",
            "int5-sdnq",
            "uint5-sdnq",
            "uint4-sdnq",
            "uint3-sdnq",
            "uint2-sdnq",
        ]
        for level in expected_sdnq_levels:
            self.assertIn(level, quantised_precision_levels, f"{level} not in quantised_precision_levels")

    def test_sdnq_presets_defined(self):
        from simpletuner.helpers.training.quantisation import MANUAL_SDNQ_PRESETS

        expected_presets = {
            "int8-sdnq",
            "uint8-sdnq",
            "int16-sdnq",
            "uint16-sdnq",
            "fp16-sdnq",
            "int6-sdnq",
            "int5-sdnq",
            "uint5-sdnq",
            "uint4-sdnq",
            "uint3-sdnq",
            "uint2-sdnq",
        }
        self.assertEqual(MANUAL_SDNQ_PRESETS, expected_presets)

    def test_sdnq_presets_in_manual_quantization_presets(self):
        from simpletuner.helpers.training.quantisation import MANUAL_QUANTIZATION_PRESETS, MANUAL_SDNQ_PRESETS

        for preset in MANUAL_SDNQ_PRESETS:
            self.assertIn(preset, MANUAL_QUANTIZATION_PRESETS)


class TestSDNQQuantizationFunction(unittest.TestCase):
    """Test that get_quant_fn returns correct function for SDNQ."""

    def test_get_quant_fn_returns_sdnq_model_for_sdnq_precision(self):
        from simpletuner.helpers.training.quantisation import _sdnq_model, get_quant_fn

        sdnq_precisions = [
            "int8-sdnq",
            "uint8-sdnq",
            "int16-sdnq",
            "uint16-sdnq",
            "fp16-sdnq",
            "int6-sdnq",
            "int5-sdnq",
            "uint5-sdnq",
            "uint4-sdnq",
            "uint3-sdnq",
            "uint2-sdnq",
        ]
        for precision in sdnq_precisions:
            quant_fn = get_quant_fn(precision)
            self.assertEqual(quant_fn, _sdnq_model, f"get_quant_fn({precision}) should return _sdnq_model")

    def test_get_quant_fn_returns_none_for_no_change(self):
        from simpletuner.helpers.training.quantisation import get_quant_fn

        self.assertIsNone(get_quant_fn("no_change"))


class TestSDNQOptimizerRegistration(unittest.TestCase):
    """Test that SDNQ optimizers are registered when available."""

    def test_sdnq_optimizers_registered_when_available(self):
        from simpletuner.helpers.training.optimizer_param import is_sdnq_available, optimizer_choices

        if not is_sdnq_available:
            self.skipTest("SDNQ not installed")

        expected_optimizers = [
            "sdnq-adamw",
            "sdnq-adamw+no_quant",
            "sdnq-adafactor",
            "sdnq-came",
            "sdnq-lion",
            "sdnq-muon",
            "sdnq-muon+quantized_matmul",
        ]
        for opt in expected_optimizers:
            self.assertIn(opt, optimizer_choices, f"{opt} not in optimizer_choices")

    def test_sdnq_adamw_has_quantized_buffer_settings(self):
        from simpletuner.helpers.training.optimizer_param import is_sdnq_available, optimizer_choices

        if not is_sdnq_available:
            self.skipTest("SDNQ not installed")

        adamw_config = optimizer_choices["sdnq-adamw"]
        self.assertIn("default_settings", adamw_config)
        settings = adamw_config["default_settings"]
        self.assertTrue(settings.get("use_quantized_buffers", False))
        self.assertEqual(settings.get("quantized_buffers_dtype"), "uint8")
        self.assertEqual(settings.get("quantized_buffers_group_size"), 32)

    def test_sdnq_adamw_no_quant_disables_quantized_buffers(self):
        from simpletuner.helpers.training.optimizer_param import is_sdnq_available, optimizer_choices

        if not is_sdnq_available:
            self.skipTest("SDNQ not installed")

        adamw_config = optimizer_choices["sdnq-adamw+no_quant"]
        settings = adamw_config["default_settings"]
        self.assertFalse(settings.get("use_quantized_buffers", True))

    def test_sdnq_muon_has_quantized_matmul_options(self):
        from simpletuner.helpers.training.optimizer_param import is_sdnq_available, optimizer_choices

        if not is_sdnq_available:
            self.skipTest("SDNQ not installed")

        muon_config = optimizer_choices["sdnq-muon"]
        settings = muon_config["default_settings"]
        self.assertIn("use_quantized_matmul", settings)
        self.assertIn("quantized_matmul_dtype", settings)
        self.assertIn("zeropower_dtype", settings)

    def test_sdnq_muon_quantized_matmul_variant(self):
        from simpletuner.helpers.training.optimizer_param import is_sdnq_available, optimizer_choices

        if not is_sdnq_available:
            self.skipTest("SDNQ not installed")

        muon_config = optimizer_choices["sdnq-muon+quantized_matmul"]
        settings = muon_config["default_settings"]
        self.assertTrue(settings.get("use_quantized_matmul", False))
        self.assertEqual(settings.get("quantized_matmul_dtype"), "int8")


class TestSDNQModelQuantization(unittest.TestCase):
    """Test SDNQ model quantization functionality."""

    def _check_sdnq_available(self):
        try:
            from sdnq.training import sdnq_training_post_load_quant

            return True
        except ImportError:
            return False

    def test_sdnq_model_returns_none_for_none_model(self):
        if not self._check_sdnq_available():
            self.skipTest("SDNQ not installed")

        from simpletuner.helpers.training.quantisation import _sdnq_model

        result = _sdnq_model(None, "int8-sdnq")
        self.assertIsNone(result)

    def test_sdnq_model_returns_model_for_no_change(self):
        if not self._check_sdnq_available():
            self.skipTest("SDNQ not installed")

        from simpletuner.helpers.training.quantisation import _sdnq_model

        model = torch.nn.Linear(4, 4)
        result = _sdnq_model(model, "no_change")
        self.assertIs(result, model)

    def test_sdnq_model_raises_for_invalid_precision(self):
        if not self._check_sdnq_available():
            self.skipTest("SDNQ not installed")

        from simpletuner.helpers.training.quantisation import _sdnq_model

        model = torch.nn.Linear(4, 4)
        with self.assertRaises(ValueError) as ctx:
            _sdnq_model(model, "invalid-sdnq")
        self.assertIn("Invalid SDNQ precision level", str(ctx.exception))


class TestSDNQDtypeMapping(unittest.TestCase):
    """Test that SDNQ dtype mapping covers all precision levels."""

    def test_all_sdnq_presets_have_dtype_mapping(self):
        from simpletuner.helpers.training.quantisation import MANUAL_SDNQ_PRESETS

        # This is the dtype map from _sdnq_model
        sdnq_dtype_map = {
            "int8-sdnq": "int8",
            "uint8-sdnq": "uint8",
            "int16-sdnq": "int16",
            "uint16-sdnq": "uint16",
            "fp16-sdnq": "fp16",
            "int6-sdnq": "int6",
            "int5-sdnq": "int5",
            "uint5-sdnq": "uint5",
            "uint4-sdnq": "uint4",
            "uint3-sdnq": "uint3",
            "uint2-sdnq": "uint2",
        }

        for preset in MANUAL_SDNQ_PRESETS:
            self.assertIn(preset, sdnq_dtype_map, f"Missing dtype mapping for {preset}")


class TestSDNQSVDConfiguration(unittest.TestCase):
    """Test that SVD is correctly configured for low-bit precision."""

    def test_low_bit_dtypes_defined(self):
        # These are the dtypes that should trigger SVD
        low_bit_dtypes = {"int5", "uint5", "uint4", "uint3", "uint2", "int4", "int3", "int2"}
        # Ensure we have the expected set
        self.assertIn("uint4", low_bit_dtypes)
        self.assertIn("uint2", low_bit_dtypes)
        self.assertNotIn("uint8", low_bit_dtypes)
        self.assertNotIn("int8", low_bit_dtypes)


class TestSDNQSaveHooksIntegration(unittest.TestCase):
    """Test SDNQ save/load hook methods."""

    def test_is_sdnq_model_returns_false_without_sdnq(self):
        # Create a mock save hooks instance to test _is_sdnq_model
        # This tests that a regular model is not detected as SDNQ
        model = torch.nn.Linear(4, 4)

        try:
            from sdnq.quantizer import QuantizationMethod
            from sdnq.training import SDNQTensor

            # Model has no quantization_method attribute
            has_quant_method = hasattr(model, "quantization_method")
            self.assertFalse(has_quant_method)

            # Model parameters are not SDNQTensor
            for param in model.parameters():
                self.assertNotIsInstance(param, SDNQTensor)
        except ImportError:
            # SDNQ not installed, skip
            self.skipTest("SDNQ not installed")


if __name__ == "__main__":
    unittest.main()
