from types import SimpleNamespace
from unittest.mock import patch

from diffusers import DDIMScheduler
from transformers import CLIPImageProcessor, CLIPVisionModel

from tests.test_pipelines._common import (
    PipelineTestCase,
    PreprocessMixin,
    RescaleNoiseCfgMixin,
    RetrieveLatentsMixin,
    RetrieveTimestepsMixin,
)


class TestSD1xPipeline(
    RetrieveTimestepsMixin,
    RetrieveLatentsMixin,
    RescaleNoiseCfgMixin,
    PreprocessMixin,
    PipelineTestCase,
):
    module_name = "sd1x"

    def test_lightweight_pipeline_registers_feature_extractor(self):
        from simpletuner.helpers.models.sd1x.model import StableDiffusion1

        model = StableDiffusion1.__new__(StableDiffusion1)
        model.config = SimpleNamespace(
            model_family="sd1x",
            pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5",
            controlnet=False,
        )
        model.pipelines = {}
        model.model = None
        model.text_encoders = None
        model.noise_schedule = DDIMScheduler(steps_offset=1, clip_sample=False)
        processor = CLIPImageProcessor()

        with (
            patch.object(model, "get_vae", return_value=None),
            patch.object(CLIPVisionModel, "from_pretrained", side_effect=OSError("No image encoder weights")),
            patch.object(CLIPImageProcessor, "from_pretrained", return_value=processor),
        ):
            pipeline = model._load_pipeline(load_base_model=False)

        self.assertIs(pipeline.feature_extractor, processor)
