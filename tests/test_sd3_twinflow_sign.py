import unittest

import torch

from simpletuner.helpers.models.sd3.transformer import SD3Transformer2DModel


class SD3TwinFlowSignEmbeddingTest(unittest.TestCase):
    def test_signed_time_embedding_changes_output(self):
        # Small config to keep test lightweight
        model = SD3Transformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=4,
            num_layers=2,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=32,
            caption_projection_dim=16,
            pooled_projection_dim=8,
            out_channels=4,
            enable_time_sign_embed=True,
        )
        hidden_states = torch.randn(1, 4, 8, 8)
        encoder_hidden_states = torch.randn(1, 4, 32)
        pooled_projections = torch.randn(1, 8)
        timestep = torch.tensor([0.5])

        # Make the sign embedding effect observable: zero row 0, ones row 1
        with torch.no_grad():
            model.time_sign_embed.weight.zero_()
            model.time_sign_embed.weight[1].fill_(1.0)

        out_pos = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            timestep_sign=torch.tensor([1.0]),
            return_dict=False,
        )[0]
        out_neg = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            timestep_sign=torch.tensor([-1.0]),
            return_dict=False,
        )[0]

        self.assertFalse(torch.allclose(out_pos, out_neg), "Sign embedding should alter the transformer output.")

    def test_signed_time_embedding_requires_enable_flag(self):
        model = SD3Transformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=4,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=32,
            caption_projection_dim=16,
            pooled_projection_dim=8,
            out_channels=4,
        )
        hidden_states = torch.randn(1, 4, 8, 8)
        encoder_hidden_states = torch.randn(1, 4, 32)
        pooled_projections = torch.randn(1, 8)
        timestep = torch.tensor([0.5])

        with self.assertRaises(ValueError):
            _ = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                timestep_sign=torch.tensor([-1.0]),
                return_dict=False,
            )

    def test_tokenwise_timesteps_are_accepted(self):
        model = SD3Transformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=4,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=32,
            caption_projection_dim=16,
            pooled_projection_dim=8,
            out_channels=4,
        )
        hidden_states = torch.randn(1, 4, 8, 8)
        encoder_hidden_states = torch.randn(1, 4, 32)
        pooled_projections = torch.randn(1, 8)
        timestep = torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.15, 0.85, 0.25, 0.75, 0.35, 0.65, 0.45, 0.55]])

        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            return_dict=False,
        )[0]

        self.assertEqual(output.shape, (1, 4, 8, 8))

    def test_tokenwise_timestep_length_is_validated(self):
        model = SD3Transformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=4,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=32,
            caption_projection_dim=16,
            pooled_projection_dim=8,
            out_channels=4,
        )
        hidden_states = torch.randn(1, 4, 8, 8)
        encoder_hidden_states = torch.randn(1, 4, 32)
        pooled_projections = torch.randn(1, 8)

        with self.assertRaisesRegex(ValueError, "tokenwise timesteps expected sequence length"):
            _ = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=torch.tensor([[0.1, 0.9, 0.2]]),
                return_dict=False,
            )

    def test_tokenwise_hidden_state_capture_uses_image_tokens(self):
        model = SD3Transformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=4,
            num_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=32,
            caption_projection_dim=16,
            pooled_projection_dim=8,
            out_channels=4,
        )
        hidden_states = torch.randn(1, 4, 8, 8)
        encoder_hidden_states = torch.randn(1, 4, 32)
        pooled_projections = torch.randn(1, 8)
        hidden_states_buffer = {}

        _ = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=torch.tensor(
                [[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6, 0.15, 0.85, 0.25, 0.75, 0.35, 0.65, 0.45, 0.55]]
            ),
            return_dict=False,
            hidden_states_buffer=hidden_states_buffer,
        )[0]

        self.assertIn("layer_0", hidden_states_buffer)
        self.assertEqual(hidden_states_buffer["layer_0"].shape, (1, 16, 16))


if __name__ == "__main__":
    unittest.main()
