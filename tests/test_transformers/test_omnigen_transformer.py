import unittest

import torch

from simpletuner.helpers.models.omnigen.transformer import OmniGenTransformer2DModel


class TestOmniGenTransformer(unittest.TestCase):
    def _tiny_transformer(self):
        model = OmniGenTransformer2DModel(
            in_channels=4,
            patch_size=1,
            hidden_size=32,
            rms_norm_eps=1e-5,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=64,
            num_layers=1,
            pad_token_id=0,
            vocab_size=32,
            max_position_embeddings=64,
            original_max_position_embeddings=32,
            rope_base=10000,
            rope_scaling={"short_factor": [1.0, 1.0, 1.0, 1.0], "long_factor": [1.0, 1.0, 1.0, 1.0]},
            pos_embed_max_size=8,
            time_step_dim=8,
        )
        model.gradient_checkpointing = False
        return model

    def test_forward_accepts_tokenwise_timesteps(self):
        transformer = self._tiny_transformer()
        latents = torch.randn(1, 4, 2, 2)
        timestep = torch.tensor([[0.2, 0.8, 0.2, 0.8]], dtype=torch.float32)
        position_ids = torch.arange(5).view(1, 5)

        with torch.no_grad():
            out = transformer(
                hidden_states=latents,
                timestep=timestep,
                input_ids=None,
                input_img_latents=[],
                input_image_sizes={},
                attention_mask=None,
                position_ids=position_ids,
                return_dict=False,
            )[0]

        self.assertEqual(out.shape, latents.shape)

    def test_forward_rejects_wrong_tokenwise_length(self):
        transformer = self._tiny_transformer()
        latents = torch.randn(1, 4, 2, 2)
        timestep = torch.tensor([[0.2, 0.8, 0.2]], dtype=torch.float32)
        position_ids = torch.arange(5).view(1, 5)

        with self.assertRaisesRegex(ValueError, "expected tokenwise timesteps with sequence length 4"):
            transformer(
                hidden_states=latents,
                timestep=timestep,
                input_ids=None,
                input_img_latents=[],
                input_image_sizes={},
                attention_mask=None,
                position_ids=position_ids,
                return_dict=False,
            )


if __name__ == "__main__":
    unittest.main()
