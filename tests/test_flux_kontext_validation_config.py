import types
import unittest

from simpletuner.helpers.models.common import PipelineTypes
from simpletuner.helpers.models.flux.model import Flux, FluxKontextPipeline


class FluxKontextValidationConfigTestCase(unittest.TestCase):
    def _make_flux_instance(self, flavour: str, validation_using_datasets: bool) -> Flux:
        flux = Flux.__new__(Flux)
        flux.config = types.SimpleNamespace(
            unet_attention_slice=False,
            aspect_bucket_alignment=64,
            prediction_type=None,
            tokenizer_max_length=None,
            model_flavour=flavour,
            validation_num_inference_steps=28,
            validation_using_datasets=validation_using_datasets,
        )
        flux.PIPELINE_CLASSES = Flux.PIPELINE_CLASSES.copy()
        return flux

    def test_kontext_disables_validation_using_datasets(self):
        flux = self._make_flux_instance("kontext", True)

        flux.check_user_config()

        self.assertFalse(flux.config.validation_using_datasets)
        self.assertIs(flux.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG], FluxKontextPipeline)

    def test_non_kontext_keeps_validation_using_datasets(self):
        flux = self._make_flux_instance("dev", True)

        flux.check_user_config()

        self.assertTrue(flux.config.validation_using_datasets)


if __name__ == "__main__":
    unittest.main()
