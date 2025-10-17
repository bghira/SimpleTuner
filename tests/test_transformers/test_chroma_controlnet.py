import unittest

import torch

from simpletuner.helpers.models.chroma.controlnet import ChromaControlNetModel
from simpletuner.helpers.models.chroma.transformer import ChromaTransformer2DModel


def _build_tiny_transformer():
    return ChromaTransformer2DModel(
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=4,
        num_attention_heads=2,
        joint_attention_dim=8,
        axes_dims_rope=(2, 2, 2),
        approximator_num_channels=8,
        approximator_hidden_dim=16,
        approximator_layers=1,
    )


class ChromaControlNetTests(unittest.TestCase):
    @torch.no_grad()
    def test_from_transformer_copies_structure(self):
        transformer = _build_tiny_transformer()
        controlnet = ChromaControlNetModel.from_transformer(transformer)

        self.assertEqual(controlnet.inner_dim, transformer.inner_dim)
        self.assertEqual(controlnet.x_embedder.weight.shape, transformer.x_embedder.weight.shape)
        self.assertTrue(
            torch.allclose(
                controlnet.controlnet_x_embedder.weight,
                torch.zeros_like(controlnet.controlnet_x_embedder.weight),
            )
        )
        self.assertEqual(len(controlnet.transformer_blocks), len(transformer.transformer_blocks))
        self.assertEqual(len(controlnet.single_transformer_blocks), len(transformer.single_transformer_blocks))

    @torch.no_grad()
    def test_forward_emits_residuals_with_expected_shapes(self):
        controlnet = ChromaControlNetModel(
            patch_size=1,
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=4,
            num_attention_heads=2,
            joint_attention_dim=8,
            axes_dims_rope=(2, 2, 2),
            approximator_num_channels=8,
            approximator_hidden_dim=16,
            approximator_layers=1,
        )

        batch_size = 2
        seq_len = 8
        hidden_states = torch.randn(batch_size, seq_len, 4)
        controlnet_cond = torch.randn(batch_size, seq_len, 4)
        encoder_hidden_states = torch.randn(batch_size, 3, 8)
        timestep = torch.ones(batch_size)

        txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3)
        img_ids = torch.zeros(seq_len, 3)

        block_samples, single_block_samples = controlnet(
            hidden_states=hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=0.5,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            joint_attention_kwargs=None,
            attention_mask=None,
            return_dict=False,
        )

        self.assertIsInstance(block_samples, tuple)
        self.assertIsInstance(single_block_samples, tuple)
        self.assertEqual(len(block_samples), len(controlnet.transformer_blocks))
        self.assertEqual(len(single_block_samples), len(controlnet.single_transformer_blocks))
        self.assertEqual(block_samples[0].shape, (batch_size, seq_len, controlnet.inner_dim))
        self.assertEqual(single_block_samples[0].shape, (batch_size, seq_len, controlnet.inner_dim))


if __name__ == "__main__":
    unittest.main()
