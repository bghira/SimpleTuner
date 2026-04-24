import unittest

import torch

from simpletuner.helpers.models.sanavideo.transformer import SanaVideoTransformer3DModel


class TestSanaVideoTransformer3DModel(unittest.TestCase):
    def setUp(self):
        self.model = SanaVideoTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=1,
            num_cross_attention_heads=2,
            cross_attention_head_dim=8,
            cross_attention_dim=16,
            caption_channels=16,
            mlp_ratio=2.0,
            sample_size=2,
            patch_size=(1, 1, 1),
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
        )

    def test_forward_with_tokenwise_timesteps(self):
        hidden_states = torch.randn(2, 4, 2, 2, 2)
        encoder_hidden_states = torch.randn(2, 5, 16)
        timestep = torch.randint(0, 1000, (2, 8))
        encoder_attention_mask = torch.ones(2, 5)

        with torch.no_grad():
            output = self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
            )

        output_tensor = output.sample if hasattr(output, "sample") else output
        self.assertEqual(output_tensor.shape, hidden_states.shape)
        self.assertFalse(torch.isnan(output_tensor).any())
        self.assertFalse(torch.isinf(output_tensor).any())

    def test_forward_rejects_wrong_tokenwise_timestep_length(self):
        hidden_states = torch.randn(1, 4, 2, 2, 2)
        encoder_hidden_states = torch.randn(1, 5, 16)
        timestep = torch.randint(0, 1000, (1, 7))

        with self.assertRaisesRegex(ValueError, "does not match token count"):
            self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
            )

    def test_forward_with_tokenwise_timesteps_and_guidance_embeddings(self):
        model = SanaVideoTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=1,
            num_cross_attention_heads=2,
            cross_attention_head_dim=8,
            cross_attention_dim=16,
            caption_channels=16,
            mlp_ratio=2.0,
            sample_size=2,
            patch_size=(1, 1, 1),
            guidance_embeds=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
        )

        hidden_states = torch.randn(1, 4, 2, 2, 2)
        encoder_hidden_states = torch.randn(1, 5, 16)
        timestep = torch.randint(0, 1000, (1, 8))
        guidance = torch.randint(0, 1000, (1,))

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                guidance=guidance,
            )

        output_tensor = output.sample if hasattr(output, "sample") else output
        self.assertEqual(output_tensor.shape, hidden_states.shape)
        self.assertFalse(torch.isnan(output_tensor).any())

    def test_forward_rejects_wrong_tokenwise_timestep_sign_length(self):
        model = SanaVideoTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=1,
            num_cross_attention_heads=2,
            cross_attention_head_dim=8,
            cross_attention_dim=16,
            caption_channels=16,
            mlp_ratio=2.0,
            sample_size=2,
            patch_size=(1, 1, 1),
            enable_time_sign_embed=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
        )

        hidden_states = torch.randn(1, 4, 2, 2, 2)
        encoder_hidden_states = torch.randn(1, 5, 16)
        timestep = torch.randint(0, 1000, (1, 8))
        timestep_sign = torch.ones(1, 7)

        with self.assertRaisesRegex(ValueError, "tokenwise timestep_sign expected sequence length 8"):
            model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                timestep_sign=timestep_sign,
            )

    def test_guidance_embeddings_reject_wrong_tokenwise_timestep_sign_length(self):
        model = SanaVideoTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=8,
            num_layers=1,
            num_cross_attention_heads=2,
            cross_attention_head_dim=8,
            cross_attention_dim=16,
            caption_channels=16,
            mlp_ratio=2.0,
            sample_size=2,
            patch_size=(1, 1, 1),
            guidance_embeds=True,
            enable_time_sign_embed=True,
            qk_norm="rms_norm_across_heads",
            rope_max_seq_len=32,
        )

        hidden_states = torch.randn(1, 4, 2, 2, 2)
        encoder_hidden_states = torch.randn(1, 5, 16)
        timestep = torch.randint(0, 1000, (1, 8))
        guidance = torch.randint(0, 1000, (1,))
        timestep_sign = torch.ones(1, 7)

        with self.assertRaisesRegex(ValueError, "tokenwise timestep_sign expected sequence length 8"):
            model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                guidance=guidance,
                timestep_sign=timestep_sign,
            )


if __name__ == "__main__":
    unittest.main()
