import os
import tempfile
import unittest

import safetensors.torch
import torch

from simpletuner.helpers.models.ltxvideo2.checkpoint_loader import load_ltx2_state_dict_from_checkpoint


class TestLTX2CheckpointLoader(unittest.TestCase):
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
