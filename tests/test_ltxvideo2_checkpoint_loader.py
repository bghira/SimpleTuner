import os
import tempfile
import unittest

import safetensors.torch
import torch

from simpletuner.helpers.models.ltxvideo2.checkpoint_loader import (
    _apply_remap_rules,
    _extract_audio_vae_config_from_metadata,
    load_ltx2_state_dict_from_checkpoint,
)


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
                "unrelated.weight": torch.full((1,), 3.0),
            }
            safetensors.torch.save_file(state_dict, ckpt_path)

            loaded = load_ltx2_state_dict_from_checkpoint(ckpt_path, "model.diffusion_model")

        self.assertIn("block.weight", loaded)
        self.assertIn("block.bias", loaded)
        self.assertIn("text_embedding_projection.aggregate_embed.weight", loaded)
        self.assertNotIn("unrelated.weight", loaded)


if __name__ == "__main__":
    unittest.main()
