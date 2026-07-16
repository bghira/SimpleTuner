import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.common import ImageModelFoundation, PipelineTypes
from simpletuner.helpers.models.sdxl.model import SDXL


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
        return _PipelineResult([f"{self.name}-image"])

    def to(self, *args, **kwargs):
        return self

    def set_progress_bar_config(self, *args, **kwargs):
        return None


class SDXLMultistageValidationTests(unittest.TestCase):
    def _model(
        self,
        *,
        mode="full-pipeline",
        refiner_training=True,
        invert_schedule=True,
        validation_using_datasets=False,
    ):
        model = SDXL.__new__(SDXL)
        model.config = SimpleNamespace(
            model_flavour="base-1.0",
            pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
            refiner_training=refiner_training,
            refiner_training_invert_schedule=invert_schedule,
            refiner_training_strength=0.35,
            sdxl_validation_pipeline_mode=mode,
            sdxl_validation_stage1_model=None,
            sdxl_validation_stage2_model=None,
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
            "negative_prompt_embeds": torch.zeros(1, 4, 8),
            "pooled_prompt_embeds": torch.zeros(1, 8),
            "negative_pooled_prompt_embeds": torch.zeros(1, 8),
            "_validation_prompt_text": "an astronaut riding a green horse",
            "_validation_negative_prompt_text": "blur",
            "num_images_per_prompt": 2,
            "num_inference_steps": 40,
            "guidance_scale": 9.2,
            "guidance_rescale": 0.7,
            "width": 1024,
            "height": 1024,
        }

    def test_trained_stage_mode_does_not_use_multistage_validation(self):
        model = self._model(mode="trained-stage")

        self.assertFalse(model.supports_multistage_validation())

    def test_dataset_validation_keeps_trained_stage_path(self):
        model = self._model(mode="full-pipeline", validation_using_datasets=True)

        self.assertFalse(model.supports_multistage_validation())

    def test_stage1_training_runs_trained_base_then_fixed_refiner(self):
        model = self._model(invert_schedule=True)
        stage2 = _RecordingPipeline("stage2")
        model._sdxl_stage2_pipeline = MagicMock(return_value=stage2)

        result = model.run_multistage_validation(
            self._pipeline_kwargs(),
            lambda pipeline, kwargs, target_stage=None: pipeline(**kwargs),
        )

        self.assertEqual(result.images, ["stage2-image"])
        stage1_call = model.pipeline.calls[0]
        stage2_call = stage2.calls[0]
        self.assertEqual(stage1_call["denoising_end"], 0.65)
        self.assertEqual(stage1_call["output_type"], "latent")
        self.assertEqual(stage1_call["num_images_per_prompt"], 2)
        self.assertIn("prompt_embeds", stage1_call)
        self.assertEqual(stage2_call["image"], ["trained-image"])
        self.assertEqual(stage2_call["denoising_start"], 0.65)
        self.assertEqual(stage2_call["output_type"], "pil")
        self.assertEqual(stage2_call["num_images_per_prompt"], 1)
        self.assertEqual(stage2_call["prompt"], ["an astronaut riding a green horse"])
        self.assertEqual(stage2_call["negative_prompt"], ["blur"])

    def test_stage2_training_runs_fixed_base_then_trained_refiner(self):
        model = self._model(invert_schedule=False)
        stage1 = _RecordingPipeline("stage1")
        stage2 = _RecordingPipeline("stage2")
        model._sdxl_stage1_pipeline = MagicMock(return_value=stage1)
        model._sdxl_stage2_pipeline = MagicMock(return_value=stage2)

        result = model.run_multistage_validation(
            self._pipeline_kwargs(),
            lambda pipeline, kwargs, target_stage=None: pipeline(**kwargs),
        )

        self.assertEqual(result.images, ["stage2-image"])
        stage1_call = stage1.calls[0]
        stage2_call = stage2.calls[0]
        self.assertEqual(stage1_call["prompt"], "an astronaut riding a green horse")
        self.assertEqual(stage1_call["negative_prompt"], "blur")
        self.assertNotIn("prompt_embeds", stage1_call)
        self.assertEqual(stage1_call["denoising_end"], 0.65)
        self.assertEqual(stage2_call["image"], ["stage1-image"])
        self.assertEqual(stage2_call["denoising_start"], 0.65)
        self.assertEqual(stage2_call["num_images_per_prompt"], 2)
        self.assertIn("prompt_embeds", stage2_call)

    def test_stage2_full_pipeline_setup_uses_img2img_pipeline(self):
        model = self._model(invert_schedule=False)
        pipeline = _RecordingPipeline("img2img")

        with patch.object(ImageModelFoundation, "get_pipeline", autospec=True, return_value=pipeline) as get_pipeline:
            result = SDXL.get_pipeline(model, PipelineTypes.TEXT2IMG, load_base_model=False)

        self.assertIs(result, pipeline)
        get_pipeline.assert_called_once_with(model, pipeline_type=PipelineTypes.IMG2IMG, load_base_model=False)

    def test_stage_model_defaults_follow_sdxl_version(self):
        model = self._model()
        model.config.model_flavour = "refiner-0.9"

        self.assertEqual(model._sdxl_validation_stage_model(1), "stabilityai/stable-diffusion-xl-base-0.9")
        self.assertEqual(model._sdxl_validation_stage_model(2), "stabilityai/stable-diffusion-xl-refiner-0.9")

    def test_unload_validation_models_clears_extra_img2img_pipeline(self):
        model = self._model()
        model.pipelines[PipelineTypes.IMG2IMG] = _RecordingPipeline("img2img")
        model.pipelines["sdxl_validation_stage2"] = _RecordingPipeline("stage2")

        with patch.object(ImageModelFoundation, "unload_validation_models", autospec=True) as super_unload:
            SDXL.unload_validation_models(model)

        super_unload.assert_called_once_with(model)
        self.assertNotIn(PipelineTypes.IMG2IMG, model.pipelines)
        self.assertNotIn("sdxl_validation_stage2", model.pipelines)


if __name__ == "__main__":
    unittest.main()
