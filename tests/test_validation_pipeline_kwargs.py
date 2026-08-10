import unittest
from types import SimpleNamespace

import torch

from simpletuner.helpers.training.validation import Validation


class _StrictPipeline:
    def __call__(self, prompt_embeds=None, height=None):
        return prompt_embeds, height


class _ForwardingPipeline:
    def __call__(self, *args, **kwargs):
        return kwargs


class _VideoPipeline:
    def __call__(self, num_videos_per_prompt=None):
        return num_videos_per_prompt


class _TensorLike:
    dtype = torch.long

    def __init__(self):
        self.to_kwargs = None

    def to(self, **kwargs):
        self.to_kwargs = kwargs
        return self


class ValidationPipelineKwargsTests(unittest.TestCase):
    def setUp(self):
        self.validation = Validation.__new__(Validation)

    def test_strict_pipeline_drops_unknown_kwargs(self):
        result = self.validation._filter_pipeline_kwargs_for_call(
            _StrictPipeline(),
            {
                "prompt_embeds": "embeds",
                "height": 512,
                "unknown": "drop",
            },
        )

        self.assertEqual(result, {"prompt_embeds": "embeds", "height": 512})

    def test_kwargs_pipeline_preserves_forwarded_kwargs(self):
        result = self.validation._filter_pipeline_kwargs_for_call(
            _ForwardingPipeline(),
            {
                "prompt_embeds": "embeds",
                "height": 512,
                "unknown": "kept",
            },
        )

        self.assertEqual(
            result,
            {
                "prompt_embeds": "embeds",
                "height": 512,
                "unknown": "kept",
            },
        )

    def test_image_count_kwarg_maps_to_explicit_video_count(self):
        result = self.validation._filter_pipeline_kwargs_for_call(
            _VideoPipeline(),
            {
                "num_images_per_prompt": 2,
            },
        )

        self.assertEqual(result, {"num_videos_per_prompt": 2})

    def test_minimaxh3_token_tags_move_to_inference_device(self):
        self.validation.config = SimpleNamespace(model_family="minimaxh3", weight_dtype=torch.bfloat16)
        self.validation.inference_device = torch.device("cpu")
        value = _TensorLike()

        result = self.validation._prepare_pipeline_kwarg_for_inference("text_token_tags", value)

        self.assertIs(result, value)
        self.assertEqual(value.to_kwargs, {"device": torch.device("cpu")})

    def test_pipeline_media_extraction_accepts_videos_field(self):
        videos = [["frame-1", "frame-2"]]
        result = self.validation._extract_pipeline_media(SimpleNamespace(videos=videos))

        self.assertIs(result, videos)


if __name__ == "__main__":
    unittest.main()
