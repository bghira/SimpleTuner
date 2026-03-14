import unittest

import torch

from simpletuner.helpers.models.hunyuanvideo.transformer import HunyuanVideo15Transformer3DModel


def make_tiny_hunyuan_transformer() -> HunyuanVideo15Transformer3DModel:
    return HunyuanVideo15Transformer3DModel(
        in_channels=5,
        out_channels=2,
        num_attention_heads=2,
        attention_head_dim=8,
        num_layers=1,
        num_refiner_layers=1,
        mlp_ratio=2.0,
        patch_size=1,
        patch_size_t=1,
        text_embed_dim=8,
        text_embed_2_dim=4,
        image_embed_dim=6,
        rope_axes_dim=(2, 2, 4),
        musubi_blocks_to_swap=0,
    )


class HunyuanVideoTransformerTests(unittest.TestCase):
    def _forward_inputs(self):
        return {
            "hidden_states": torch.randn(1, 5, 2, 2, 2),
            "encoder_hidden_states": torch.randn(1, 3, 8),
            "encoder_attention_mask": torch.ones(1, 3, dtype=torch.bool),
            "encoder_hidden_states_2": torch.randn(1, 2, 4),
            "encoder_attention_mask_2": torch.ones(1, 2, dtype=torch.bool),
            "image_embeds": torch.zeros(1, 4, 6),
        }

    def test_forward_accepts_tokenwise_timesteps(self):
        transformer = make_tiny_hunyuan_transformer()
        hidden_states_buffer = {}

        output = transformer(
            **self._forward_inputs(),
            timestep=torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]], dtype=torch.float32),
            return_dict=False,
            hidden_states_buffer=hidden_states_buffer,
        )[0]

        self.assertEqual(output.shape, (1, 2, 2, 2, 2))
        self.assertIn("layer_0", hidden_states_buffer)
        self.assertEqual(hidden_states_buffer["layer_0"].shape, (1, 2, 4, 16))

    def test_forward_rejects_wrong_tokenwise_timestep_length(self):
        transformer = make_tiny_hunyuan_transformer()

        with self.assertRaisesRegex(ValueError, "tokenwise timesteps expected shape"):
            transformer(
                **self._forward_inputs(),
                timestep=torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4]], dtype=torch.float32),
                return_dict=False,
            )

    def test_gradient_checkpointing_preserves_framewise_capture_shape(self):
        transformer = make_tiny_hunyuan_transformer()
        transformer.gradient_checkpointing = True
        hidden_states_buffer = {}
        inputs = self._forward_inputs()
        inputs["hidden_states"] = inputs["hidden_states"].requires_grad_(True)
        inputs["encoder_hidden_states"] = inputs["encoder_hidden_states"].requires_grad_(True)
        inputs["encoder_hidden_states_2"] = inputs["encoder_hidden_states_2"].requires_grad_(True)

        output = transformer(
            **inputs,
            timestep=torch.tensor([[0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]], dtype=torch.float32),
            return_dict=False,
            hidden_states_buffer=hidden_states_buffer,
        )[0]
        output.sum().backward()

        self.assertIn("layer_0", hidden_states_buffer)
        self.assertEqual(hidden_states_buffer["layer_0"].shape, (1, 2, 4, 16))


if __name__ == "__main__":
    unittest.main()
