import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import safetensors.torch
import torch
import torch.nn.functional as F

from simpletuner.helpers.models.ltxvideo2.autoencoder import AutoencoderKLLTX2Video
from simpletuner.helpers.models.ltxvideo2.checkpoint_loader import (
    _apply_remap_rules,
    _convert_ltx2_3_vocoder_upsamplers,
    _extract_audio_vae_config_from_metadata,
    _get_ltx2_connectors_config,
    _get_ltx2_transformer_config,
    _get_ltx2_video_vae_config,
    _get_ltx2_vocoder_config,
    _infer_ltx2_connectors_config_overrides,
    _infer_ltx2_transformer_config_overrides,
    convert_ltx2_video_vae,
    get_model_state_dict_from_combined_ckpt,
    is_ltx2_diffusion_video_vae_state_dict,
    load_ltx2_state_dict_from_checkpoint,
)
from simpletuner.helpers.models.ltxvideo2.na_kernels import na3d, rms_rope_


class TestLTX2CheckpointLoader(unittest.TestCase):
    def test_extract_audio_vae_config_from_metadata(self):
        metadata_config = {
            "audio_vae": {
                "preprocessing": {
                    "audio": {"sampling_rate": 22050},
                    "stft": {"hop_length": 256, "filter_length": 2048, "causal": False},
                    "mel": {"n_mel_channels": 80},
                },
                "model": {
                    "params": {
                        "sampling_rate": 16000,
                        "ddconfig": {
                            "ch": 64,
                            "out_ch": 1,
                            "ch_mult": [1, 2],
                            "num_res_blocks": 3,
                            "attn_resolutions": [16],
                            "in_channels": 1,
                            "resolution": 128,
                            "z_channels": 4,
                            "double_z": False,
                            "norm_type": "group",
                            "causality_axis": "width",
                            "dropout": 0.1,
                            "mid_block_add_attention": True,
                            "mel_bins": 80,
                        },
                    }
                },
            }
        }

        config = _extract_audio_vae_config_from_metadata(metadata_config)

        self.assertIsNotNone(config)
        self.assertEqual(config["base_channels"], 64)
        self.assertEqual(config["output_channels"], 1)
        self.assertEqual(config["ch_mult"], (1, 2))
        self.assertEqual(config["num_res_blocks"], 3)
        self.assertEqual(config["attn_resolutions"], (16,))
        self.assertEqual(config["in_channels"], 1)
        self.assertEqual(config["resolution"], 128)
        self.assertEqual(config["latent_channels"], 4)
        self.assertFalse(config["double_z"])
        self.assertEqual(config["norm_type"], "group")
        self.assertEqual(config["causality_axis"], "width")
        self.assertAlmostEqual(config["dropout"], 0.1)
        self.assertTrue(config["mid_block_add_attention"])
        self.assertEqual(config["sample_rate"], 16000)
        self.assertEqual(config["mel_hop_length"], 256)
        self.assertEqual(config["n_fft"], 2048)
        self.assertFalse(config["is_causal"])
        self.assertEqual(config["mel_bins"], 80)

    def test_apply_remap_rules_removes_keys(self):
        state_dict = {
            "remove.me": torch.zeros(1),
            "keep.me": torch.ones(1),
        }

        _apply_remap_rules(state_dict, rename_dict={}, special_keys_remap={"remove.me": None})

        self.assertNotIn("remove.me", state_dict)
        self.assertIn("keep.me", state_dict)

    def test_load_state_dict_from_safetensors_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "ltx2.safetensors")
            state_dict = {
                "model.diffusion_model.block.weight": torch.zeros(1),
                "model.diffusion_model.block.bias": torch.ones(1),
                "text_embedding_projection.aggregate_embed.weight": torch.full((1,), 2.0),
                "text_embedding_projection.video_aggregate_embed.weight": torch.full((1,), 4.0),
                "text_embedding_projection.audio_aggregate_embed.weight": torch.full((1,), 5.0),
                "unrelated.weight": torch.full((1,), 3.0),
            }
            safetensors.torch.save_file(state_dict, ckpt_path)

            loaded = load_ltx2_state_dict_from_checkpoint(ckpt_path, "model.diffusion_model")

        self.assertIn("block.weight", loaded)
        self.assertIn("block.bias", loaded)
        self.assertIn("text_embedding_projection.aggregate_embed.weight", loaded)
        self.assertIn("text_embedding_projection.video_aggregate_embed.weight", loaded)
        self.assertIn("text_embedding_projection.audio_aggregate_embed.weight", loaded)
        self.assertNotIn("unrelated.weight", loaded)

    def test_get_model_state_dict_from_combined_ckpt_includes_all_text_projection_keys(self):
        combined_ckpt = {
            "model.diffusion_model.block.weight": torch.zeros(1),
            "text_embedding_projection.aggregate_embed.weight": torch.ones(1),
            "text_embedding_projection.video_aggregate_embed.weight": torch.full((1,), 2.0),
            "text_embedding_projection.audio_aggregate_embed.weight": torch.full((1,), 3.0),
        }

        loaded = get_model_state_dict_from_combined_ckpt(combined_ckpt, "model.diffusion_model")

        self.assertIn("block.weight", loaded)
        self.assertIn("text_embedding_projection.aggregate_embed.weight", loaded)
        self.assertIn("text_embedding_projection.video_aggregate_embed.weight", loaded)
        self.assertIn("text_embedding_projection.audio_aggregate_embed.weight", loaded)

    def test_ltx2_3_configs_expose_required_flags(self):
        transformer_config = _get_ltx2_transformer_config("2.3")
        connectors_config = _get_ltx2_connectors_config("2.3")
        vocoder_config = _get_ltx2_vocoder_config("2.3")

        self.assertTrue(transformer_config["gated_attn"])
        self.assertTrue(transformer_config["cross_attn_mod"])
        self.assertFalse(transformer_config["use_prompt_embeddings"])
        self.assertTrue(connectors_config["per_modality_projections"])
        self.assertTrue(connectors_config["video_gated_attn"])
        self.assertEqual(vocoder_config["output_sampling_rate"], 48000)
        self.assertEqual(vocoder_config["bwe_hidden_channels"], 512)

    def test_ltx2_5_configs_alias_ltx23_with_comfy_flags(self):
        transformer_config = _get_ltx2_transformer_config("2.5")
        connectors_config = _get_ltx2_connectors_config("2.5")
        vae_config = _get_ltx2_video_vae_config("2.5")
        vocoder_config = _get_ltx2_vocoder_config("2.5")

        self.assertTrue(transformer_config["gated_attn"])
        self.assertTrue(transformer_config["cross_attn_mod"])
        self.assertTrue(transformer_config["ff_bias"])
        self.assertTrue(transformer_config["audio_ff_bias"])
        self.assertTrue(transformer_config["use_prompt_adaln_single"])
        self.assertFalse(transformer_config["use_keyframes_abs_pos_embedding"])
        self.assertTrue(connectors_config["connector_ff_bias"])
        self.assertEqual(vae_config["decoder_spatial_padding_mode"], "reflect")
        self.assertEqual(vocoder_config["output_sampling_rate"], 48000)

    def test_ltx2_transformer_state_dict_infers_comfy_capability_flags(self):
        state_dict = {
            "keyframes_abs_pos_embedding": torch.zeros(1, 16),
            "transformer_blocks.0.ff.net.0.proj.weight": torch.zeros(1),
            "transformer_blocks.0.ff.net.2.weight": torch.zeros(1),
            "transformer_blocks.0.audio_ff.net.0.proj.weight": torch.zeros(1),
            "transformer_blocks.0.audio_ff.net.2.weight": torch.zeros(1),
            "transformer_blocks.0.prompt_scale_shift_table": torch.zeros(1),
            "transformer_blocks.0.audio_prompt_scale_shift_table": torch.zeros(1),
        }

        overrides = _infer_ltx2_transformer_config_overrides(state_dict)

        self.assertTrue(overrides["use_keyframes_abs_pos_embedding"])
        self.assertFalse(overrides["ff_bias"])
        self.assertFalse(overrides["audio_ff_bias"])
        self.assertFalse(overrides["use_prompt_adaln_single"])

    def test_ltx2_connectors_state_dict_infers_ff_bias(self):
        state_dict = {
            "video_connector.transformer_blocks.0.ff.net.0.proj.weight": torch.zeros(1),
            "video_connector.transformer_blocks.0.ff.net.2.weight": torch.zeros(1),
            "video_connector.transformer_blocks.0.ff.net.2.bias": torch.zeros(1),
        }

        overrides = _infer_ltx2_connectors_config_overrides(state_dict)

        self.assertTrue(overrides["connector_ff_bias"])

    def test_ltx2_diffusion_video_vae_is_detected_and_loaded_with_vendored_decoder(self):
        state_dict = {
            "decoder.conv_in_x_t.weight": torch.zeros(1),
            "per_channel_statistics.mean-of-means": torch.zeros(128),
            "per_channel_statistics.std-of-means": torch.ones(128),
            "per_channel_statistics.channel": torch.arange(128),
            "per_channel_statistics.mean-of-stds": torch.zeros(128),
        }
        fake_vae = MagicMock()

        self.assertTrue(is_ltx2_diffusion_video_vae_state_dict(state_dict))
        with patch(
            "simpletuner.helpers.models.ltxvideo2.checkpoint_loader.AutoencoderKLLTX2VideoDiffusionDecoder",
            return_value=fake_vae,
        ) as vae_cls:
            vae = convert_ltx2_video_vae(state_dict, version="2.5")

        self.assertIs(vae, fake_vae)
        vae_cls.assert_called_once()
        _, kwargs = vae_cls.call_args
        self.assertIn("diffusion_decoder_config", kwargs)
        loaded_state_dict = fake_vae.load_state_dict.call_args.args[0]
        self.assertIn("decoder.conv_in_x_t.weight", loaded_state_dict)
        self.assertIn("latents_mean", loaded_state_dict)
        self.assertIn("latents_std", loaded_state_dict)
        self.assertNotIn("per_channel_statistics.mean-of-means", loaded_state_dict)
        self.assertNotIn("per_channel_statistics.std-of-means", loaded_state_dict)
        self.assertNotIn("per_channel_statistics.channel", loaded_state_dict)
        self.assertNotIn("per_channel_statistics.mean-of-stds", loaded_state_dict)
        fake_vae.load_state_dict.assert_called_once_with(loaded_state_dict, strict=True, assign=True)

    def test_ltx2_diffusion_video_vae_requires_ltx25_config(self):
        state_dict = {"decoder.conv_in_x_t.weight": torch.zeros(1)}

        with self.assertRaisesRegex(ValueError, "LTX-2.5"):
            convert_ltx2_video_vae(state_dict, version="2.3")

    def test_vendored_na3d_singleton_kernel_returns_value_tensor(self):
        q = torch.randn(1, 2, 2, 2, 1, 3)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        out = na3d(q, k, v, [1, 1, 1], None, 1.0)

        self.assertTrue(torch.equal(out, v))

    def test_vendored_rms_rope_applies_identity_rope_in_place(self):
        q = torch.randn(1, 3, 2, 4)
        k = torch.randn(1, 3, 2, 4)
        original_q = q.clone()
        original_k = k.clone()
        freqs = torch.eye(2).expand(1, 3, 1, 2, 2, 2).contiguous()
        q_weight = torch.ones(4)
        k_weight = torch.full((4,), 2.0)

        q_out, k_out = rms_rope_(q, k, freqs, q_weight, k_weight)

        self.assertIs(q_out, q)
        self.assertIs(k_out, k)
        self.assertTrue(torch.allclose(q, F.rms_norm(original_q, (4,), weight=q_weight, eps=1e-6)))
        self.assertTrue(torch.allclose(k, F.rms_norm(original_k, (4,), weight=k_weight, eps=1e-6)))

    def test_ltx2_3_video_vae_config_matches_decoder_checkpoint_layout(self):
        config = _get_ltx2_video_vae_config("2.3")
        vae = AutoencoderKLLTX2Video.from_config(config)

        self.assertEqual(config["upsample_type"], ("spatial", "temporal", "spatiotemporal", "spatiotemporal"))
        self.assertEqual(config["upsample_residual"], (True, True, True, True))
        self.assertEqual(config["decoder_spatial_padding_mode"], "reflect")
        self.assertEqual(
            [block.upsamplers[0].stride for block in vae.decoder.up_blocks],
            [(2, 2, 2), (2, 2, 2), (2, 1, 1), (1, 2, 2)],
        )
        self.assertEqual(
            [block.upsamplers[0].residual for block in vae.decoder.up_blocks],
            [True, True, True, True],
        )
        self.assertEqual(
            [tuple(block.upsamplers[0].conv.conv.weight.shape) for block in vae.decoder.up_blocks],
            [
                (4096, 1024, 3, 3, 3),
                (4096, 512, 3, 3, 3),
                (512, 512, 3, 3, 3),
                (512, 256, 3, 3, 3),
            ],
        )

    def test_vocoder_upsampler_remap_handles_root_level_keys(self):
        state_dict = {
            "ups.0.weight": torch.ones(1),
            "ups.0.bias": torch.zeros(1),
        }

        _apply_remap_rules(state_dict, rename_dict={}, special_keys_remap={"ups.": _convert_ltx2_3_vocoder_upsamplers})

        self.assertIn("upsamplers.0.weight", state_dict)
        self.assertIn("upsamplers.0.bias", state_dict)
        self.assertNotIn("ups.0.weight", state_dict)
        self.assertNotIn("ups.0.bias", state_dict)


if __name__ == "__main__":
    unittest.main()
