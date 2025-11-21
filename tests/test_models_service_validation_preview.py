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


if __name__ == "__main__":
    unittest.main()
