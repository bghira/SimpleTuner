import json
import unittest
from pathlib import Path
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

    def test_model_config_path_uses_ltx23_dev_repo_for_single_file(self):
        self.model.config.model_flavour = "2.3-dev"
        self.model.config.pretrained_model_name_or_path = "/tmp/ltx-2.3-22b-dev.safetensors"

        self.assertEqual(self.model._model_config_path(), "dg845/LTX-2.3-Diffusers")

    def test_model_config_path_uses_ltx23_distilled_repo_for_single_file(self):
        self.model.config.model_flavour = "2.3-distilled"
        self.model.config.pretrained_model_name_or_path = "/tmp/ltx-2.3-22b-distilled.safetensors"

        self.assertEqual(self.model._model_config_path(), "dg845/LTX-2.3-Distilled-Diffusers")

    def test_legacy_ltx23_aliases_are_rejected(self):
        for flavour in ("2.3", "distilled"):
            with self.subTest(flavour=flavour):
                self.model.config.model_flavour = flavour
                with self.assertRaisesRegex(ValueError, "Unsupported LTX-2 model flavour"):
                    self.model._resolve_ltx2_version()

    def test_load_video_vae_from_diffusers_repo_uses_corrected_ltx23_config(self):
        self.model.config.model_flavour = "2.3-dev"
        self.model.config.pretrained_model_name_or_path = "dg845/LTX-2.3-Diffusers"
        self.model.config.pretrained_vae_model_name_or_path = "dg845/LTX-2.3-Diffusers"
        state_dict = {"dummy": torch.zeros(1)}
        fake_vae = MagicMock()

        with (
            patch("simpletuner.helpers.models.ltxvideo2.model.hf_hub_download", return_value="/tmp/vae.safetensors"),
            patch("simpletuner.helpers.models.ltxvideo2.model.safetensors.torch.load_file", return_value=state_dict),
            patch.object(self.model.AUTOENCODER_CLASS, "from_config", return_value=fake_vae) as mock_from_config,
        ):
            self.model._load_video_vae_from_diffusers_repo()

        self.assertEqual(
            mock_from_config.call_args.args[0]["upsample_type"],
            ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
        )
        self.assertEqual(
            mock_from_config.call_args.args[0]["upsample_residual"],
            (True, True, True, True),
        )
        self.assertEqual(
            mock_from_config.call_args.args[0]["decoder_spatial_padding_mode"],
            "reflect",
        )
        fake_vae.load_state_dict.assert_called_once_with(state_dict, strict=True, assign=True)
        fake_vae.register_to_config.assert_called_once_with(_name_or_path="dg845/LTX-2.3-Diffusers")
        self.assertIs(self.model.vae, fake_vae)

    def test_intrinsic_first_frame_conditioning_replaces_tokens_and_masks_loss(self):
        self.model.config.ltx2_intrinsic_conditioning = [{"type": "first_frame", "probability": 1.0}]
        packed_noisy = torch.zeros(2, 8, 1)
        packed_clean = torch.arange(16, dtype=torch.float32).view(2, 8, 1)
        timesteps = torch.ones(2, 8)

        conditioned, conditioned_timesteps, loss_mask = self.model._apply_ltx2_intrinsic_conditioning(
            packed_noisy=packed_noisy,
            packed_clean=packed_clean,
            target_timesteps=timesteps,
            prepared_batch={},
            num_frames=2,
            height=2,
            width=2,
            patch_size=1,
            patch_size_t=1,
        )

        self.assertTrue(torch.equal(conditioned[:, :4], packed_clean[:, :4]))
        self.assertTrue(torch.equal(conditioned[:, 4:], packed_noisy[:, 4:]))
        self.assertTrue(torch.equal(conditioned_timesteps[:, :4], torch.zeros(2, 4)))
        self.assertTrue(torch.equal(conditioned_timesteps[:, 4:], torch.ones(2, 4)))
        self.assertTrue(torch.equal(loss_mask[:, :4], torch.zeros(2, 4, dtype=torch.bool)))
        self.assertTrue(torch.equal(loss_mask[:, 4:], torch.ones(2, 4, dtype=torch.bool)))

    def test_intrinsic_conditioning_accepts_json_string_config(self):
        self.model.config.ltx2_intrinsic_conditioning = '[{"type":"prefix","probability":0.5,"temporal_boundary":2}]'

        specs = self.model._ltx2_intrinsic_condition_specs()

        self.assertEqual(specs, [{"type": "prefix", "probability": 0.5, "temporal_boundary": 2}])

    def test_mask_conditioning_uses_one_as_clean_no_loss(self):
        mask_pixels = torch.full((1, 1, 2, 2), -1.0)
        mask_pixels[:, :, 0, :] = 1.0

        token_mask = self.model._ltx2_mask_condition_mask(
            mask_pixels,
            batch_size=1,
            post_patch_frames=2,
            post_patch_height=2,
            post_patch_width=2,
            device=torch.device("cpu"),
        )

        self.assertTrue(torch.equal(token_mask, torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]])))

    def test_reference_coords_scale_to_target_space(self):
        self.model.config.ltx2_reference_temporal_scale_factor = 2
        ref_coords = torch.tensor([[[[0.0, 0.5], [1.0, 1.5]], [[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0]]]])
        target_coords = torch.tensor([[[[0.0, 0.25], [0.25, 0.5]], [[0.0, 1.0], [1.0, 2.0]], [[0.0, 1.0], [1.0, 2.0]]]])

        scaled = self.model._scale_ltx2_reference_coords(
            ref_coords=ref_coords,
            target_coords=target_coords,
            ref_height=4,
            ref_width=4,
            target_height=8,
            target_width=8,
        )

        self.assertTrue(torch.equal(scaled[:, 1], ref_coords[:, 1] * 2))
        self.assertTrue(torch.equal(scaled[:, 2], ref_coords[:, 2] * 2))
        self.assertTrue(torch.equal(scaled[:, 0], torch.clamp(ref_coords[:, 0] - 0.25, min=0)))

    def test_masked_video_loss_ignores_intrinsic_condition_tokens(self):
        self.model.model = SimpleNamespace(config=SimpleNamespace(patch_size=1, patch_size_t=1))
        self.model.config.loss_type = "l2"
        prepared_batch = {
            "latents": torch.zeros(1, 1, 1, 1, 2),
            "noise": torch.ones(1, 1, 1, 1, 2),
            "video_loss_mask": torch.tensor([[False, True]]),
        }
        model_output = {"model_prediction": torch.tensor([[[[[99.0, 1.0]]]]])}

        loss = self.model._compute_ltx2_masked_video_loss(prepared_batch, model_output)

        self.assertEqual(loss.item(), 0.0)


class TestLTXVideo2Metadata(unittest.TestCase):
    def test_model_metadata_exposes_only_named_ltx23_flavours(self):
        metadata_path = Path(__file__).parent.parent.parent / "simpletuner/helpers/models/model_metadata.json"
        with open(metadata_path) as handle:
            metadata = json.load(handle)

        self.assertEqual(
            metadata["ltxvideo2"]["flavour_choices"],
            ["dev", "dev-fp4", "dev-fp8", "2.3-dev", "2.3-distilled"],
        )


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

    def test_transformer_forward_exposes_self_attention_masks(self):
        import inspect

        from simpletuner.helpers.models.ltxvideo2.transformer import LTX2VideoTransformer3DModel

        sig = inspect.signature(LTX2VideoTransformer3DModel.forward)
        self.assertIn("self_attention_mask", sig.parameters)
        self.assertIn("audio_self_attention_mask", sig.parameters)
        self.assertIn("a2v_cross_attention_mask", sig.parameters)
        self.assertIn("v2a_cross_attention_mask", sig.parameters)

    def test_ltx2_conditioning_config_fields_are_parseable(self):
        from simpletuner.helpers.configuration.cmd_args import get_argument_parser

        parser = get_argument_parser()
        args = parser.parse_args(
            [
                "--model_family",
                "ltxvideo2",
                "--output_dir",
                "/tmp/simpletuner-test",
                "--model_type",
                "lora",
                "--optimizer",
                "adamw_bf16",
                "--data_backend_config",
                "/tmp/backend.json",
                "--ltx2_intrinsic_conditioning",
                '[{"type":"first_frame","probability":1.0}]',
                "--ltx2_reference_temporal_scale_factor",
                "2",
            ]
        )

        self.assertEqual(args.ltx2_intrinsic_conditioning, '[{"type":"first_frame","probability":1.0}]')
        self.assertEqual(args.ltx2_reference_temporal_scale_factor, 2)


if __name__ == "__main__":
    unittest.main()
