"""Tests for ModelsService validation preview capability metadata."""

import unittest


class TestModelsServiceValidationPreview(unittest.TestCase):
    """Ensure model capability metadata exposes preview support correctly."""

    @classmethod
    def setUpClass(cls):
        # Import models package to register all families with the ModelRegistry.
        import simpletuner.helpers.models  # noqa: F401
        from simpletuner.simpletuner_sdk.server.services.models_service import ModelsService

        cls.service = ModelsService()

    def test_flux_supports_validation_preview(self):
        details = self.service.get_model_details("flux")
        capabilities = details["capabilities"]
        self.assertTrue(capabilities.get("supports_validation_preview"))

    def test_deepfloyd_lacks_validation_preview(self):
        details = self.service.get_model_details("deepfloyd")
        capabilities = details["capabilities"]
        self.assertFalse(capabilities.get("supports_validation_preview"))

    def test_ace_step_supports_lyrics(self):
        """ACE Step should advertise lyrics support via capabilities."""
        details = self.service.get_model_details("ace_step")
        capabilities = details["capabilities"]
        self.assertTrue(capabilities.get("supports_lyrics"))
        self.assertTrue(capabilities.get("is_audio_model"))

    def test_sdxl_does_not_support_lyrics(self):
        """SDXL should not advertise lyrics support."""
        details = self.service.get_model_details("sdxl")
        capabilities = details["capabilities"]
        self.assertFalse(capabilities.get("supports_lyrics"))
        self.assertFalse(capabilities.get("is_audio_model"))

    def test_ltxvideo2_supports_audio_inputs(self):
        """LTX-Video 2 should support audio inputs for audio conditioning."""
        details = self.service.get_model_details("ltxvideo2")
        capabilities = details["capabilities"]
        self.assertTrue(capabilities.get("supports_audio_inputs"))
        # LTX-2 supports audio but doesn't require S2V datasets
        self.assertFalse(capabilities.get("requires_s2v_datasets"))

    def test_wan_s2v_requires_s2v_datasets(self):
        """Wan S2V should require S2V datasets and support audio inputs."""
        details = self.service.get_model_details("wan_s2v")
        capabilities = details["capabilities"]
        self.assertTrue(capabilities.get("supports_audio_inputs"))
        self.assertTrue(capabilities.get("requires_s2v_datasets"))

    def test_sdxl_does_not_support_audio_inputs(self):
        """SDXL should not support audio inputs."""
        details = self.service.get_model_details("sdxl")
        capabilities = details["capabilities"]
        self.assertFalse(capabilities.get("supports_audio_inputs"))
        self.assertFalse(capabilities.get("requires_s2v_datasets"))

    def test_flux_does_not_support_audio_inputs(self):
        """Flux should not support audio inputs."""
        details = self.service.get_model_details("flux")
        capabilities = details["capabilities"]
        self.assertFalse(capabilities.get("supports_audio_inputs"))
        self.assertFalse(capabilities.get("requires_s2v_datasets"))

    # -- Multi-stage validation capability tests --

    def test_multistage_capability_present(self):
        """The supports_multistage_validation key must appear in capabilities."""
        details = self.service.get_model_details("flux")
        capabilities = details["capabilities"]
        self.assertIn("supports_multistage_validation", capabilities)

    def test_sdxl_no_multistage(self):
        """SDXL does not support multi-stage validation."""
        details = self.service.get_model_details("sdxl")
        self.assertFalse(details["capabilities"]["supports_multistage_validation"])

    def test_flux_no_multistage(self):
        """Flux does not support multi-stage validation."""
        details = self.service.get_model_details("flux")
        self.assertFalse(details["capabilities"]["supports_multistage_validation"])

    def test_ltxvideo2_no_multistage(self):
        """LTX-Video 2 does not yet support multi-stage validation."""
        details = self.service.get_model_details("ltxvideo2")
        self.assertFalse(details["capabilities"]["supports_multistage_validation"])

    def test_deepfloyd_no_multistage(self):
        """DeepFloyd does not yet support multi-stage validation."""
        details = self.service.get_model_details("deepfloyd")
        self.assertFalse(details["capabilities"]["supports_multistage_validation"])

    def test_evaluate_requirements_includes_multistage(self):
        """evaluate_requirements() must include supports_multistage_validation."""
        result = self.service.evaluate_requirements("flux")
        self.assertIn("supports_multistage_validation", result)
        self.assertFalse(result["supports_multistage_validation"])


if __name__ == "__main__":
    unittest.main()
