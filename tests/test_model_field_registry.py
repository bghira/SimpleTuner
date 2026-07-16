"""Tests for model-owned field registry modules."""

import unittest

from simpletuner.simpletuner_sdk.server.services.field_registry.registry import FieldRegistry


class TestModelFieldRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = FieldRegistry()

    def test_model_registry_modules_are_discovered(self):
        expected_fields = [
            "deepfloyd_validation_pipeline_mode",
            "wan_validation_load_other_stage",
            "sdxl_validation_pipeline_mode",
            "validation_lyrics",
            "flux_lora_target",
            "ltx_train_mode",
            "ltx2_intrinsic_conditioning",
            "ideogram_auto_json",
            "sana_complex_human_instruction",
            "hidream_use_load_balancing_loss",
            "sd3_clip_uncond_behaviour",
            "krea2_reference_latents",
        ]

        for field_name in expected_fields:
            with self.subTest(field=field_name):
                self.assertIsNotNone(self.registry.get_field(field_name))

    def test_model_specific_fields_keep_context_filtering(self):
        flux_fields = {
            field.name
            for field in self.registry.get_fields_for_tab(
                "model",
                context={
                    "model_family": "flux",
                    "model_type": "lora",
                    "i_know_what_i_am_doing": True,
                },
            )
        }
        self.assertIn("flux_lora_target", flux_fields)
        self.assertNotIn("acestep_lora_target", flux_fields)

        ace_fields = {
            field.name
            for field in self.registry.get_fields_for_tab(
                "model",
                context={
                    "model_family": "ace_step",
                    "model_type": "lora",
                },
            )
        }
        self.assertIn("acestep_lora_target", ace_fields)
        self.assertNotIn("flux_lora_target", ace_fields)


if __name__ == "__main__":
    unittest.main()
