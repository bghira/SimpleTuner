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


class TestLTXVideo2TransformerLoading(unittest.TestCase):
    """Test LTXVideo2 transformer can be imported and configured."""

    def test_transformer_has_tread_support(self):
        """Test that the transformer has TREAD router support."""
        from simpletuner.helpers.models.ltxvideo2.transformer import LTX2VideoTransformer3DModel

        self.assertTrue(hasattr(LTX2VideoTransformer3DModel, "_tread_router"))
        self.assertTrue(hasattr(LTX2VideoTransformer3DModel, "_tread_routes"))

    def test_transformer_has_set_router_method(self):
        """Test that the transformer has the set_router method for TREAD."""
        from simpletuner.helpers.models.ltxvideo2.transformer import LTX2VideoTransformer3DModel

        self.assertTrue(hasattr(LTX2VideoTransformer3DModel, "set_router"))

    def test_transformer_has_musubi_block_swap(self):
        """Test that the transformer has Musubi block swap support."""
        import inspect

        from simpletuner.helpers.models.ltxvideo2.transformer import LTX2VideoTransformer3DModel

        sig = inspect.signature(LTX2VideoTransformer3DModel.__init__)
        params = list(sig.parameters.keys())
        self.assertIn("musubi_blocks_to_swap", params)
        self.assertIn("musubi_block_swap_device", params)

    def test_transformer_has_time_sign_embed_flag(self):
        """Test that the transformer exposes TwinFlow sign embedding support."""
        import inspect

        from simpletuner.helpers.models.ltxvideo2.transformer import LTX2VideoTransformer3DModel

        sig = inspect.signature(LTX2VideoTransformer3DModel.__init__)
        params = list(sig.parameters.keys())
        self.assertIn("enable_time_sign_embed", params)


if __name__ == "__main__":
    unittest.main()
