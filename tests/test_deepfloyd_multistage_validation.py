import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.deepfloyd.model import DeepFloydIF


class _PipelineResult:
    def __init__(self, images):
        self.images = images


class _RecordingPipeline:
    def __init__(self, name):
        self.name = name
        self.calls = []

    def __call__(
        self,
        prompt=None,
        negative_prompt=None,
        image=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        num_inference_steps=None,
        generator=None,
        guidance_scale=None,
        output_type=None,
        width=None,
        height=None,
        num_images_per_prompt=None,
        noise_level=None,
    ):
        self.calls.append(
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": image,
                "prompt_embeds": prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
                "guidance_scale": guidance_scale,
                "output_type": output_type,
                "width": width,
                "height": height,
                "num_images_per_prompt": num_images_per_prompt,
                "noise_level": noise_level,
            }
        )
        return _PipelineResult([f"{self.name}-image"])


class DeepFloydConfigurationTests(unittest.TestCase):
    def test_stage_one_keeps_small_pixel_sizes_for_lora_and_full_training(self):
        from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args
        from simpletuner.helpers.data_backend.config.image import ImageBackendConfig
        from simpletuner.helpers.data_backend.factory import init_backend_config

        for model_type in ("lora", "full"):
            with self.subTest(model_type=model_type), TemporaryDirectory() as output_dir:
                args = parse_cmdline_args(
                    [
                        "--model_family=deepfloyd",
                        "--model_flavour=i-medium-400m",
                        f"--model_type={model_type}",
                        f"--output_dir={output_dir}",
                        "--optimizer=torch-adamw",
                        "--data_backend_config=dataset.json",
                        "--resolution_type=pixel",
                        "--resolution=64",
                        "--validation_resolution=64",
                        "--maximum_image_size=128",
                        "--target_downsample_size=64",
                        "--aspect_bucket_alignment=32",
                    ]
                )
                self.assertEqual(int(args.validation_resolution), 64)
                self.assertEqual(args.aspect_bucket_alignment, 8)
                backend = {
                    "id": "stage-one",
                    "type": "local",
                    "resolution_type": "pixel",
                    "maximum_image_size": 128,
                    "target_downsample_size": 64,
                }
                ImageBackendConfig.from_dict(backend, vars(args)).validate(vars(args))
                configured = init_backend_config(backend, args, accelerator=None)["config"]
                self.assertEqual(configured["maximum_image_size"], 128)
                self.assertEqual(configured["target_downsample_size"], 64)

    def test_stage_two_uses_a_256_pixel_resolution_floor(self):
        from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args

        with TemporaryDirectory() as output_dir:
            args = parse_cmdline_args(
                [
                    "--model_family=deepfloyd",
                    "--model_flavour=ii-medium-450m",
                    "--model_type=lora",
                    f"--output_dir={output_dir}",
                    "--optimizer=torch-adamw",
                    "--data_backend_config=dataset.json",
                    "--resolution=128",
                ]
            )
        self.assertEqual(args.resolution, 256)
        self.assertEqual(args.resolution_type, "pixel")
        self.assertEqual(args.aspect_bucket_alignment, 64)


class DeepFloydMultistageValidationTests(unittest.TestCase):
    def _model(self, *, flavour="ii-medium-450m", mode="auto", stage3_mode="none", validation_using_datasets=False):
        model = DeepFloydIF.__new__(DeepFloydIF)
        model.config = SimpleNamespace(
            model_flavour=flavour,
            deepfloyd_validation_pipeline_mode=mode,
            deepfloyd_validation_stage3_mode=stage3_mode,
            deepfloyd_validation_stage1_num_inference_steps=None,
            deepfloyd_validation_stage2_num_inference_steps=None,
            deepfloyd_validation_stage1_guidance=None,
            deepfloyd_validation_stage2_guidance=None,
            deepfloyd_validation_stage3_guidance=None,
            deepfloyd_validation_stage3_noise_level=100,
            validation_using_datasets=validation_using_datasets,
        )
        model.pipelines = {}
        model.pipeline = _RecordingPipeline("trained")
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        return model

    def test_auto_mode_uses_full_pipeline_for_prompt_validation(self):
        model = self._model(mode="auto", validation_using_datasets=False)

        self.assertTrue(model.supports_multistage_validation())

    def test_auto_mode_uses_trained_stage_for_dataset_validation(self):
        model = self._model(mode="auto", validation_using_datasets=True)

        self.assertFalse(model.supports_multistage_validation())

    def test_stage2_training_runs_fixed_stage1_then_trained_stage2(self):
        model = self._model(flavour="ii-medium-450m")
        stage1 = _RecordingPipeline("stage1")
        stage2 = _RecordingPipeline("stage2")
        model._deepfloyd_stage1_pipeline = MagicMock(return_value=stage1)
        model._deepfloyd_stage2_pipeline = MagicMock(return_value=stage2)

        stages = []

        result = model.run_multistage_validation(
            {
                "prompt_embeds": torch.zeros(1, 4, 8),
                "negative_prompt_embeds": torch.zeros(1, 4, 8),
                "num_inference_steps": 20,
                "guidance_scale": 5.5,
                "width": 256,
                "height": 256,
                "num_images_per_prompt": 1,
            },
            lambda pipeline, kwargs, target_stage=None: stages.append(target_stage) or pipeline(**kwargs),
        )

        self.assertEqual(result.images, ["stage2-image"])
        self.assertEqual(stages, ["stage1", "stage2"])
        self.assertEqual(stage1.calls[0]["width"], 64)
        self.assertEqual(stage1.calls[0]["height"], 64)
        self.assertEqual(stage1.calls[0]["output_type"], "pt")
        self.assertEqual(stage2.calls[0]["image"], ["stage1-image"])
        self.assertEqual(stage2.calls[0]["width"], 256)
        self.assertEqual(stage2.calls[0]["height"], 256)
        self.assertEqual(stage2.calls[0]["output_type"], "pil")

    def test_stage3_sd_x4_upscaler_receives_prompt_text(self):
        model = self._model(flavour="i-large-900m", stage3_mode="sd-x4-upscaler")
        stage1 = _RecordingPipeline("stage1")
        stage2 = _RecordingPipeline("stage2")
        stage3 = _RecordingPipeline("stage3")
        model._deepfloyd_stage1_pipeline = MagicMock(return_value=stage1)
        model._deepfloyd_stage2_pipeline = MagicMock(return_value=stage2)
        model._deepfloyd_stage3_pipeline = MagicMock(return_value=stage3)

        stages = []

        result = model.run_multistage_validation(
            {
                "prompt_embeds": torch.zeros(1, 4, 8),
                "negative_prompt_embeds": torch.zeros(1, 4, 8),
                "num_inference_steps": 20,
                "guidance_scale": 5.5,
                "width": 1024,
                "height": 1024,
                "num_images_per_prompt": 1,
                "_validation_prompt_text": "a painted castle",
                "_validation_negative_prompt_text": "blur",
            },
            lambda pipeline, kwargs, target_stage=None: stages.append(target_stage) or pipeline(**kwargs),
        )

        self.assertEqual(result.images, ["stage3-image"])
        self.assertEqual(stages, ["stage1", "stage2", "stage3"])
        self.assertEqual(stage1.calls[0]["width"], 64)
        self.assertEqual(stage2.calls[0]["width"], 256)
        self.assertEqual(stage3.calls[0]["prompt"], ["a painted castle"])
        self.assertEqual(stage3.calls[0]["negative_prompt"], ["blur"])
        self.assertEqual(stage3.calls[0]["image"], ["stage2-image"])
        self.assertEqual(stage3.calls[0]["noise_level"], 100)


if __name__ == "__main__":
    unittest.main()
