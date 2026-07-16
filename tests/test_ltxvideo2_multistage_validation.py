import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.common import VideoModelFoundation
from simpletuner.helpers.models.ltxvideo2.model import LTXVideo2


class _LTX2Result:
    def __init__(self, frames, audio=None):
        self.frames = frames
        self.audio = audio


class _RecordingLTX2Pipeline:
    def __init__(self, name):
        self.name = name
        self.calls = []
        self.vae_temporal_compression_ratio = 8
        self.vae_spatial_compression_ratio = 32
        self.transformer_spatial_patch_size = 1
        self.transformer_temporal_patch_size = 1

    def __call__(self, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs.get("output_type") == "latent":
            return _LTX2Result(frames=torch.zeros(1, 1008, 128), audio=torch.ones(1, 256, 128))
        return _LTX2Result(frames=[f"{self.name}-frames"], audio=f"{self.name}-audio")

    def _unpack_latents(self, latents, num_frames, height, width, patch_size=1, patch_size_t=1):
        return torch.zeros(1, 128, num_frames, height, width)

    def _pack_latents(self, latents, patch_size=1, patch_size_t=1):
        return "packed-upscaled-latents"


class LTXVideo2MultistageValidationTests(unittest.TestCase):
    def _model(self, *, mode="spatial-upscale", validation_audio_only=False):
        model = LTXVideo2.__new__(LTXVideo2)
        model.config = SimpleNamespace(
            ltx2_validation_pipeline_mode=mode,
            ltx2_validation_spatial_upsampler_model="Lightricks/LTX-2.3",
            ltx2_validation_spatial_upsampler_filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            validation_audio_only=validation_audio_only,
            validation_num_video_frames=49,
            revision=None,
            weight_dtype=torch.float32,
        )
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.pipeline = _RecordingLTX2Pipeline("stage1")
        model.stage2 = _RecordingLTX2Pipeline("stage2")
        model.get_pipeline = MagicMock(return_value=model.stage2)
        model.get_vae = MagicMock(return_value=SimpleNamespace())
        model._ltx2_spatial_upsampler = MagicMock(return_value=object())
        return model

    def _pipeline_kwargs(self):
        return {
            "prompt_embeds": torch.zeros(1, 4, 8),
            "prompt_attention_mask": torch.ones(1, 4),
            "negative_prompt_embeds": torch.zeros(1, 4, 8),
            "negative_prompt_attention_mask": torch.ones(1, 4),
            "num_videos_per_prompt": 1,
            "num_inference_steps": 30,
            "guidance_scale": 3.0,
            "width": 1024,
            "height": 768,
            "num_frames": 49,
            "frame_rate": 24,
            "image": "stage1-image-conditioning",
            "output_type": "np",
        }

    def test_trained_stage_mode_does_not_use_multistage_validation(self):
        self.assertFalse(self._model(mode="trained-stage").supports_multistage_validation())

    def test_audio_only_validation_does_not_use_spatial_upscale(self):
        self.assertFalse(self._model(validation_audio_only=True).supports_multistage_validation())

    def test_spatial_upscale_runs_half_resolution_stage1_then_full_resolution_stage2(self):
        model = self._model()
        upscaled_latents = torch.ones(1, 128, 7, 24, 32)

        with patch(
            "simpletuner.helpers.models.ltxvideo2.model.upsample_ltx2_video_latents",
            return_value=upscaled_latents,
        ) as upsample:
            result = model.run_multistage_validation(
                self._pipeline_kwargs(),
                lambda pipeline, kwargs: pipeline(**kwargs),
            )

        self.assertEqual(result.frames, ["stage2-frames"])
        stage1_call = model.pipeline.calls[0]
        stage2_call = model.stage2.calls[0]
        self.assertEqual(stage1_call["width"], 512)
        self.assertEqual(stage1_call["height"], 384)
        self.assertEqual(stage1_call["output_type"], "latent")
        self.assertEqual(stage2_call["width"], 1024)
        self.assertEqual(stage2_call["height"], 768)
        self.assertEqual(stage2_call["latents"], "packed-upscaled-latents")
        self.assertTrue(torch.equal(stage2_call["audio_latents"], torch.ones(1, 256, 128)))
        self.assertNotIn("image", stage2_call)
        self.assertEqual(stage2_call["sigmas"], [0.909375, 0.725, 0.421875])
        self.assertEqual(stage2_call["latent_noise_scale"], 0.909375)
        self.assertEqual(stage2_call["audio_latent_noise_scale"], 0.909375)
        self.assertEqual(stage2_call["output_type"], "np")
        upsample.assert_called_once()

    def test_stage1_resolution_requires_target_divisible_by_64(self):
        model = self._model()
        kwargs = self._pipeline_kwargs()
        kwargs["width"] = 800

        with self.assertRaisesRegex(ValueError, "divisible by 64"):
            model.run_multistage_validation(kwargs, lambda pipeline, call_kwargs: pipeline(**call_kwargs))

    def test_unload_validation_models_clears_upsampler(self):
        model = self._model()
        model._ltx2_validation_spatial_upsampler = object()

        with patch.object(VideoModelFoundation, "unload_validation_models", autospec=True) as super_unload:
            LTXVideo2.unload_validation_models(model)

        super_unload.assert_called_once_with(model)
        self.assertFalse(hasattr(model, "_ltx2_validation_spatial_upsampler"))


if __name__ == "__main__":
    unittest.main()
