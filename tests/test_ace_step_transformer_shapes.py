import unittest

import torch

from simpletuner.helpers.models.ace_step.transformer import ACEStepTransformer2DModel


class ACEStepTransformerShapeTest(unittest.TestCase):
    def setUp(self):
        # Create a smaller model for testing
        self.config = {
            "in_channels": 8,
            "num_layers": 2,
            "inner_dim": 64,
            "attention_head_dim": 16,
            "num_attention_heads": 4,
            "out_channels": 8,
            "max_position": 128,
            "speaker_embedding_dim": 32,
            "text_embedding_dim": 32,
            "ssl_encoder_depths": [1],
            "ssl_names": ["mert"],
            "ssl_latent_dims": [32],
            "lyric_encoder_vocab_size": 100,
            "lyric_hidden_size": 1024,
            "patch_size": [16, 1],
            "max_height": 16,
            "max_width": 64,
        }
        self.model = ACEStepTransformer2DModel(**self.config)
        self.model.eval()

    def test_encode_decode_mask_shapes(self):
        batch_size = 2
        channels = self.config["in_channels"]
        height = self.config["max_height"]
        width = self.config["max_width"]

        # Latents: N x C x H x W (Note: PatchEmbed expects this, but transformer works on patches)
        # The transformer's forward expects hidden_states in latent shape?
        # Looking at PatchEmbed: forward(latent) -> early_conv_layers(latent) -> flatten -> transpose
        # So input should be N x C x H x W.
        hidden_states = torch.randn(batch_size, channels, height, width)

        text_dim = self.config["text_embedding_dim"]
        seq_len = 16
        encoder_text_hidden_states = torch.randn(batch_size, seq_len, text_dim)
        text_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.float32)

        speaker_dim = self.config["speaker_embedding_dim"]
        speaker_embeds = torch.zeros(batch_size, speaker_dim)

        lyric_token_idx = torch.zeros(batch_size, 1, dtype=torch.long)
        lyric_mask = torch.zeros(batch_size, 1, dtype=torch.long)

        timestep = torch.rand(batch_size)

        with torch.no_grad():
            enc_states, enc_mask = self.model.encode(
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask,
                speaker_embeds=speaker_embeds,
                lyric_token_idx=lyric_token_idx,
                lyric_mask=lyric_mask,
            )

            # Check shapes
            # We expect enc_states to be (batch_size, total_seq_len, inner_dim)
            # total_seq_len = 1 (speaker) + seq_len (text) + 1 (lyric, roughly, depends on LyricEncoder)
            self.assertEqual(enc_states.ndim, 3)
            self.assertEqual(enc_mask.ndim, 2)
            self.assertEqual(enc_states.shape[0], batch_size)
            self.assertEqual(enc_mask.shape[0], batch_size)
            self.assertEqual(enc_states.shape[1], enc_mask.shape[1])
            self.assertEqual(enc_states.shape[2], self.config["inner_dim"])

            out = self.model.decode(
                hidden_states=hidden_states,
                attention_mask=None,
                encoder_hidden_states=enc_states,
                encoder_hidden_mask=enc_mask,
                timestep=timestep,
                output_length=width,
                return_dict=True,
            )

        self.assertIn("sample", out)
        # Output should be same spatial shape as input?
        # T2IFinalLayer unpatchifies.
        # If width was provided, it tries to match it.
        self.assertEqual(out["sample"].shape, (batch_size, self.config["out_channels"], height, width))

    def test_forward_pass(self):
        batch_size = 1
        channels = self.config["in_channels"]
        height = self.config["max_height"]
        width = self.config["max_width"]

        hidden_states = torch.randn(batch_size, channels, height, width)

        text_dim = self.config["text_embedding_dim"]
        seq_len = 8
        encoder_text_hidden_states = torch.randn(batch_size, seq_len, text_dim)

        speaker_dim = self.config["speaker_embedding_dim"]
        speaker_embeds = torch.randn(batch_size, speaker_dim)

        timestep = torch.tensor([500.0] * batch_size)

        output = self.model(
            hidden_states=hidden_states,
            attention_mask=None,
            encoder_text_hidden_states=encoder_text_hidden_states,
            speaker_embeds=speaker_embeds,
            lyric_token_idx=torch.zeros(batch_size, 1, dtype=torch.long),
            lyric_mask=torch.zeros(batch_size, 1, dtype=torch.long),
            timestep=timestep,
        )

        self.assertIsNotNone(output.sample)
        self.assertEqual(output.sample.shape, (batch_size, self.config["out_channels"], height, width))


if __name__ == "__main__":
    unittest.main()
