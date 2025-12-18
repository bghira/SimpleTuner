import unittest

import torch

from simpletuner.helpers.models.flux2.transformer import Flux2Transformer2DModel


def _tiny_flux2_transformer(**extra_kwargs) -> Flux2Transformer2DModel:
    return Flux2Transformer2DModel(
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=1,
        joint_attention_dim=16,
        timestep_guidance_channels=4,
        mlp_ratio=2.0,
        axes_dims_rope=(2, 2, 2, 2),
        rope_theta=2000,
        eps=1e-6,
        musubi_blocks_to_swap=0,
        musubi_block_swap_device="cpu",
        **extra_kwargs,
    )


class Flux2TimeSignEmbedGatingTestCase(unittest.TestCase):
    def test_time_sign_embed_disabled_by_default(self):
        model = _tiny_flux2_transformer()
        self.assertTrue(hasattr(model, "time_sign_embed"))
        self.assertIsNone(model.time_sign_embed)
        self.assertNotIn("time_sign_embed.weight", model.state_dict())

    def test_time_sign_embed_enabled_when_requested(self):
        model = _tiny_flux2_transformer(enable_time_sign_embed=True)
        self.assertIsNotNone(model.time_sign_embed)
        self.assertTrue(isinstance(model.time_sign_embed, torch.nn.Embedding))
        self.assertIn("time_sign_embed.weight", model.state_dict())


if __name__ == "__main__":
    unittest.main()
