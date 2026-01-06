import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.ltxvideo2.model import LTXVideo2
from simpletuner.helpers.training.state_tracker import StateTracker


class TestLTXVideo2Pipeline(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            model_family="ltxvideo2",
            pretrained_model_name_or_path="dummy_path",
            pretrained_vae_model_name_or_path=None,
            vae_path=None,
            model_flavour=None,
            weight_dtype=torch.float32,
            validation_num_video_frames=42,
            framerate=24,
            revision=None,
            variant=None,
            scheduled_sampling_max_step_offset=0,
            twinflow_enabled=False,
            use_ema=False,
        )
        self.accelerator = MagicMock()
        self.accelerator.device = torch.device("cpu")

        self._schedule_patcher = patch("simpletuner.helpers.models.common.ModelFoundation.setup_training_noise_schedule")
        self._schedule_patcher.start()
        self.addCleanup(self._schedule_patcher.stop)

        self.model = LTXVideo2(self.config, self.accelerator)

    def test_update_pipeline_call_kwargs_injects_audio_latents(self):
        expected = torch.randn(1, 2)
        with patch.object(self.model, "_load_audio_latents_for_validation", return_value=expected):
            kwargs = {"_s2v_conditioning": {"audio_path": "/data/audio/sample.wav"}}
            updated = self.model.update_pipeline_call_kwargs(kwargs)

        self.assertEqual(updated["num_frames"], 42)
        self.assertEqual(updated["frame_rate"], 24)
        self.assertTrue(torch.equal(updated["audio_latents"], expected))
        self.assertNotIn("_s2v_conditioning", updated)

    def test_load_audio_latents_for_validation_uses_cache(self):
        latents = torch.randn(1, 3)
        cache = MagicMock()
        cache.retrieve_from_cache.return_value = {"latents": latents}
        backends = {"audio-backend": {"config": {"instance_data_dir": "/data/audio"}}}

        with (
            patch.object(StateTracker, "get_data_backends", return_value=backends),
            patch.object(StateTracker, "get_vaecache", return_value=cache),
        ):
            result = self.model._load_audio_latents_for_validation("/data/audio/sample.wav")

        self.assertTrue(torch.equal(result.cpu(), latents))
        self.assertEqual(result.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
