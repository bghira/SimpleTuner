import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.common import ImageModelFoundation
from simpletuner.helpers.models.pixart.model import PixartSigma


class _PipelineResult:
    def __init__(self, images):
        self.images = images


class _RecordingPipeline:
    def __init__(self, name):
        self.name = name
        self.calls = []
        self.scheduler = None

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        return _PipelineResult([f"{self.name}-latents"])

    def to(self, *args, **kwargs):
        return self

    def set_progress_bar_config(self, *args, **kwargs):
        return None


class PixArtMultistageValidationTests(unittest.TestCase):
    def _model(
        self,
        *,
        mode="full-pipeline",
        refiner_training=True,
        invert_schedule=True,
        validation_using_datasets=False,
    ):
        model = PixartSigma.__new__(PixartSigma)
        model.config = SimpleNamespace(
            model_flavour="900M-1024-v0.7-stage1",
            pretrained_model_name_or_path="terminusresearch/pixart-900m-1024-ft-v0.7-stage1",
            refiner_training=refiner_training,
            refiner_training_invert_schedule=invert_schedule,
            refiner_training_strength=0.35,
            pixart_validation_pipeline_mode=mode,
            pixart_validation_stage1_model=None,
            pixart_validation_stage2_model=None,
            validation_using_datasets=validation_using_datasets,
            controlnet=False,
            control=False,
            weight_dtype=torch.float32,
        )
        model.pipelines = {}
        model.pipeline = _RecordingPipeline("trained")
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.get_vae = MagicMock(return_value=None)
        return model

    def _pipeline_kwargs(self):
        return {
            "prompt_embeds": torch.zeros(1, 4, 8),
            "prompt_attention_mask": torch.ones(1, 4),
            "negative_prompt_embeds": torch.zeros(1, 4, 8),
            "negative_prompt_attention_mask": torch.ones(1, 4),
            "_validation_prompt_text": "an astronaut riding a green horse",
            "_validation_negative_prompt_text": "blur",
            "num_images_per_prompt": 2,
            "num_inference_steps": 40,
            "guidance_scale": 4.5,
            "width": 1024,
            "height": 1024,
        }

    def test_trained_stage_mode_does_not_use_multistage_validation(self):
        model = self._model(mode="trained-stage")

        self.assertFalse(model.supports_multistage_validation())

    def test_dataset_validation_keeps_trained_stage_path(self):
        model = self._model(mode="full-pipeline", validation_using_datasets=True)

        self.assertFalse(model.supports_multistage_validation())

    def test_stage1_training_runs_trained_stage1_then_fixed_stage2(self):
        model = self._model(invert_schedule=True)
        stage2 = _RecordingPipeline("stage2")
        model._pixart_stage2_pipeline = MagicMock(return_value=stage2)

        result = model.run_multistage_validation(
            self._pipeline_kwargs(),
            lambda pipeline, kwargs: pipeline(**kwargs),
        )

        self.assertEqual(result.images, ["stage2-latents"])
        stage1_call = model.pipeline.calls[0]
        stage2_call = stage2.calls[0]
        self.assertEqual(stage1_call["denoising_end"], 0.65)
        self.assertEqual(stage1_call["output_type"], "latent")
        self.assertEqual(stage1_call["num_images_per_prompt"], 2)
        self.assertIn("prompt_embeds", stage1_call)
        self.assertEqual(stage2_call["image"], ["trained-latents"])
        self.assertEqual(stage2_call["denoising_start"], 0.65)
        self.assertEqual(stage2_call["output_type"], "pil")
        self.assertEqual(stage2_call["num_images_per_prompt"], 2)
        self.assertIn("prompt_embeds", stage2_call)

    def test_stage2_training_runs_fixed_stage1_then_trained_stage2(self):
        model = self._model(invert_schedule=False)
        stage1 = _RecordingPipeline("stage1")
        stage2 = _RecordingPipeline("stage2")
        model._pixart_stage1_pipeline = MagicMock(return_value=stage1)
        model._pixart_stage2_pipeline = MagicMock(return_value=stage2)

        result = model.run_multistage_validation(
            self._pipeline_kwargs(),
            lambda pipeline, kwargs: pipeline(**kwargs),
        )

        self.assertEqual(result.images, ["stage2-latents"])
        stage1_call = stage1.calls[0]
        stage2_call = stage2.calls[0]
        self.assertEqual(stage1_call["denoising_end"], 0.65)
        self.assertEqual(stage1_call["output_type"], "latent")
        self.assertEqual(stage2_call["image"], ["stage1-latents"])
        self.assertEqual(stage2_call["denoising_start"], 0.65)
        self.assertIn("prompt_embeds", stage1_call)
        self.assertIn("prompt_embeds", stage2_call)

    def test_prompt_text_is_used_when_embeds_are_missing(self):
        model = self._model()
        kwargs = {
            "_validation_prompt_text": "a painted castle",
            "_validation_negative_prompt_text": "blur",
        }

        self.assertEqual(
            model._pixart_prompt_kwargs(kwargs),
            {"prompt": "a painted castle", "negative_prompt": "blur"},
        )

    def test_stage_model_defaults_use_pixart_v07_split(self):
        model = self._model()

        self.assertEqual(
            model._pixart_validation_stage_model(1),
            "terminusresearch/pixart-900m-1024-ft-v0.7-stage1",
        )
        self.assertEqual(
            model._pixart_validation_stage_model(2),
            "terminusresearch/pixart-900m-1024-ft-v0.7-stage2",
        )

    def test_unload_validation_models_clears_peer_pipelines(self):
        model = self._model()
        model.pipelines["pixart_validation_stage1"] = _RecordingPipeline("stage1")
        model.pipelines["pixart_validation_stage2"] = _RecordingPipeline("stage2")

        with patch.object(ImageModelFoundation, "unload_validation_models", autospec=True) as super_unload:
            PixartSigma.unload_validation_models(model)

        super_unload.assert_called_once_with(model)
        self.assertNotIn("pixart_validation_stage1", model.pipelines)
        self.assertNotIn("pixart_validation_stage2", model.pipelines)


if __name__ == "__main__":
    unittest.main()
