import unittest

import torch

from simpletuner.helpers.models.flux.transformer import FluxTransformer2DModel


def _tiny_flux_transformer(**extra_kwargs) -> FluxTransformer2DModel:
    return FluxTransformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=8,
        num_attention_heads=1,
        joint_attention_dim=16,
        pooled_projection_dim=8,
        guidance_embeds=False,
        axes_dims_rope=(2, 2, 4),
        musubi_blocks_to_swap=0,
        musubi_block_swap_device="cpu",
        **extra_kwargs,
    )


class FluxTimeSignEmbedGatingTestCase(unittest.TestCase):
    def test_forward_accepts_tokenwise_timesteps(self):
        model = _tiny_flux_transformer()
        hidden_states = torch.randn(1, 4, 4)
        encoder_hidden_states = torch.randn(1, 2, 16)
        pooled_projections = torch.randn(1, 8)
        timestep = torch.tensor([[0.1, 0.3, 0.5, 0.7]], dtype=torch.float32)
        img_ids = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
            ],
            dtype=torch.int64,
        )
        txt_ids = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
            ],
            dtype=torch.int64,
        )

        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[0]

        self.assertEqual(output.shape, (1, 4, 4))

    def test_forward_rejects_wrong_tokenwise_timestep_length(self):
        model = _tiny_flux_transformer()
        with self.assertRaisesRegex(ValueError, "tokenwise timesteps"):
            model(
                hidden_states=torch.randn(1, 4, 4),
                encoder_hidden_states=torch.randn(1, 2, 16),
                pooled_projections=torch.randn(1, 8),
                timestep=torch.tensor([[0.1, 0.3, 0.5]], dtype=torch.float32),
                img_ids=torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=torch.int64),
                txt_ids=torch.tensor([[0, 0, 0], [0, 0, 1]], dtype=torch.int64),
                return_dict=False,
            )


if __name__ == "__main__":
    unittest.main()
