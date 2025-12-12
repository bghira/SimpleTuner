import unittest

import torch

from simpletuner.helpers.models.longcat_video.transformer import LongCatVideoTransformer3DModel


class TestLongCatVideoTransformerShapes(unittest.TestCase):
    def test_forward_preserves_5d_shape(self):
        transformer = LongCatVideoTransformer3DModel(
            in_channels=16,
            out_channels=16,
            hidden_size=64,
            depth=1,
            num_heads=8,
            caption_channels=64,
            mlp_ratio=2,
            adaln_tembed_dim=32,
            frequency_embedding_size=16,
            patch_size=(1, 2, 2),
            enable_flashattn3=False,
            enable_flashattn2=False,
            enable_xformers=False,
            enable_bsa=False,
            cp_split_hw=None,
            text_tokens_zero_pad=False,
        )
        transformer.gradient_checkpointing = False

        latents = torch.randn(1, 16, 1, 4, 4)
        encoder_hidden_states = torch.randn(1, 4, 64)
        encoder_attention_mask = torch.ones(1, 4)
        timestep = torch.randint(0, 1000, (1,))

        with torch.no_grad():
            out = transformer(
                latents,
                timestep,
                encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=False,
            )[0]

        self.assertEqual(out.shape, latents.shape)


if __name__ == "__main__":
    unittest.main()
