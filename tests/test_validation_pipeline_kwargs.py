import unittest

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


if __name__ == "__main__":
    unittest.main()
