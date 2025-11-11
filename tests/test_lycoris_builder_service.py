import unittest

from lycoris.config_sdk import PresetValidationError

from simpletuner.simpletuner_sdk.server.services.lycoris_builder_service import LycorisBuilderService


class LycorisBuilderServiceTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.service = LycorisBuilderService()

    def test_metadata_contains_algorithms_presets_and_defaults(self) -> None:
        metadata = self.service.get_metadata(force_refresh=True)

        self.assertIn("algorithms", metadata)
        self.assertIn("defaults", metadata)
        self.assertIn("presets", metadata)
        self.assertIn("suggestions", metadata)
        self.assertTrue(any(algo["name"] == "lora" for algo in metadata["algorithms"]))
        self.assertIn("lora", metadata["defaults"])
        self.assertTrue(any(preset["name"] == "attn-only" for preset in metadata["presets"]))
        self.assertIn("target_module", metadata["suggestions"])

    def test_validate_preset_detects_invalid_payloads(self) -> None:
        valid_preset = {
            "target_module": ["Attention"],
            "module_algo_map": {"Attention": {"factor": 4}},
        }
        # Should not raise
        self.service.validate_preset(valid_preset)

        with self.assertRaises(PresetValidationError):
            self.service.validate_preset({"unknown_key": "value"})


if __name__ == "__main__":
    unittest.main()
