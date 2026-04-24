import unittest

import torch

from simpletuner.helpers.models.longcat_image.transformer import LongCatImageTransformer2DModel


class TestLongCatImageTransformer(unittest.TestCase):
    def _tiny_transformer(self):
        model = LongCatImageTransformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            pooled_projection_dim=16,
            axes_dims_rope=[2, 2, 4],
        )
        model.gradient_checkpointing = False
        return model

    def test_forward_accepts_tokenwise_timesteps(self):
        transformer = self._tiny_transformer()
        hidden_states = torch.randn(1, 4, 4)
        encoder_hidden_states = torch.randn(1, 2, 16)
        timestep = torch.tensor([[100.0, 900.0, 100.0, 900.0]], dtype=torch.float32)
        txt_ids = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        img_ids = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )

        with torch.no_grad():
            out = transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False,
            )[0]

        self.assertEqual(out.shape, hidden_states.shape)

    def test_forward_rejects_wrong_tokenwise_length(self):
        transformer = self._tiny_transformer()
        hidden_states = torch.randn(1, 4, 4)
        encoder_hidden_states = torch.randn(1, 2, 16)
        timestep = torch.tensor([[100.0, 900.0, 100.0]], dtype=torch.float32)
        txt_ids = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
        img_ids = torch.tensor(
            [[1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            dtype=torch.float32,
        )

        with self.assertRaisesRegex(ValueError, "expected tokenwise timesteps with sequence length 4"):
            transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                txt_ids=txt_ids,
                img_ids=img_ids,
                return_dict=False,
            )


if __name__ == "__main__":
    unittest.main()
