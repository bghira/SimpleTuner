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

    def test_resolve_ltx23_flavour(self):
        self.model.config.model_flavour = "2.3-distilled"
        self.assertEqual(self.model._resolve_ltx2_version(), "2.3")
        self.assertEqual(self.model._resolve_ltx2_combined_filename(), "ltx-2.3-22b-distilled.safetensors")

    def test_model_config_path_uses_ltx23_repo_for_single_file(self):
        self.model.config.model_flavour = "2.3"
        self.model.config.pretrained_model_name_or_path = "/tmp/ltx-2.3-22b-dev.safetensors"

        self.assertEqual(self.model._model_config_path(), "Lightricks/LTX-2.3")


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

    def test_transformer_supports_ltx23_flags(self):
        from simpletuner.helpers.models.ltxvideo2.transformer import LTX2VideoTransformer3DModel

        transformer = LTX2VideoTransformer3DModel(
            in_channels=4,
            out_channels=4,
            patch_size=1,
            patch_size_t=1,
            num_attention_heads=2,
            attention_head_dim=8,
            cross_attention_dim=16,
            vae_scale_factors=(1, 1, 1),
            pos_embed_max_pos=4,
            base_height=32,
            base_width=32,
            gated_attn=True,
            cross_attn_mod=True,
            audio_in_channels=4,
            audio_out_channels=4,
            audio_patch_size=1,
            audio_patch_size_t=1,
            audio_num_attention_heads=2,
            audio_attention_head_dim=4,
            audio_cross_attention_dim=8,
            audio_scale_factor=1,
            audio_pos_embed_max_pos=4,
            audio_gated_attn=True,
            audio_cross_attn_mod=True,
            num_layers=1,
            caption_channels=16,
            use_prompt_embeddings=False,
            perturbed_attn=True,
        )

        self.assertFalse(hasattr(transformer, "caption_projection"))
        self.assertTrue(transformer.prompt_modulation)
        self.assertIsNotNone(transformer.prompt_adaln)
        self.assertIsNotNone(transformer.audio_prompt_adaln)
        self.assertIsNotNone(transformer.transformer_blocks[0].attn1.to_gate_logits)


if __name__ == "__main__":
    unittest.main()
