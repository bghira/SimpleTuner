import types
import unittest

from simpletuner.helpers.models.flux2.model import Flux2


class Flux2KleinValidationGuidanceTestCase(unittest.TestCase):
    """Test that Klein flavours move validation_guidance to validation_guidance_real."""

    def _make_flux2_instance(self, flavour: str, validation_guidance=None):
        """Create a minimal Flux2 instance for testing check_user_config."""
        flux2 = Flux2.__new__(Flux2)
        flux2.config = types.SimpleNamespace(
            model_flavour=flavour,
            aspect_bucket_alignment=16,
            tokenizer_max_length=512,
            validation_guidance=validation_guidance,
            validation_guidance_real=None,
            flux_guidance_mode=None,
            flux_guidance_value=None,
        )
        return flux2

    def test_klein_9b_moves_validation_guidance_to_real(self):
        """Klein-9b should move validation_guidance to validation_guidance_real."""
        flux2 = self._make_flux2_instance("klein-9b", validation_guidance=3.5)
        flux2.check_user_config()

        self.assertIsNone(flux2.config.validation_guidance)
        self.assertEqual(flux2.config.validation_guidance_real, 3.5)

    def test_klein_4b_moves_validation_guidance_to_real(self):
        """Klein-4b should move validation_guidance to validation_guidance_real."""
        flux2 = self._make_flux2_instance("klein-4b", validation_guidance=2.0)
        flux2.check_user_config()

        self.assertIsNone(flux2.config.validation_guidance)
        self.assertEqual(flux2.config.validation_guidance_real, 2.0)

    def test_klein_no_validation_guidance_unchanged(self):
        """Klein without validation_guidance set should not create validation_guidance_real."""
        flux2 = self._make_flux2_instance("klein-9b", validation_guidance=None)
        flux2.check_user_config()

        self.assertIsNone(flux2.config.validation_guidance)
        self.assertIsNone(flux2.config.validation_guidance_real)

    def test_dev_keeps_validation_guidance(self):
        """Dev flavour should not move validation_guidance."""
        flux2 = self._make_flux2_instance("dev", validation_guidance=3.5)
        flux2.check_user_config()

        self.assertEqual(flux2.config.validation_guidance, 3.5)
        self.assertIsNone(flux2.config.validation_guidance_real)

    def test_klein_distilled_keeps_validation_guidance(self):
        """Distilled Klein flavours should not move validation_guidance."""
        flux2 = self._make_flux2_instance("klein-9b-distilled", validation_guidance=3.5)
        # Add distilled flavour to KLEIN_FLAVOURS temporarily for test
        from simpletuner.helpers.models.flux2.model import KLEIN_FLAVOURS

        original_flavours = KLEIN_FLAVOURS.copy()
        KLEIN_FLAVOURS.add("klein-9b-distilled")
        try:
            flux2.check_user_config()
            self.assertEqual(flux2.config.validation_guidance, 3.5)
            self.assertIsNone(flux2.config.validation_guidance_real)
        finally:
            KLEIN_FLAVOURS.clear()
            KLEIN_FLAVOURS.update(original_flavours)


if __name__ == "__main__":
    unittest.main()
