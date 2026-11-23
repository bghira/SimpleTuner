import unittest
from unittest.mock import patch

import torch

from simpletuner.helpers.models.pixart.transformer import PixArtTransformer2DModel


class TestPixArtTransformer2DModel(unittest.TestCase):
    def make_model(self, num_layers):
        return PixArtTransformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=4,
            out_channels=8,
            attention_head_dim=8,
            num_attention_heads=2,
            num_layers=num_layers,
            cross_attention_dim=16,
            caption_channels=16,
            num_embeds_ada_norm=1000,
            use_additional_conditions=False,
        )

    def test_forward_with_controlnet_residual(self):
        model = self.make_model(num_layers=2)
        inputs = dict(
            hidden_states=torch.randn(1, 4, 8, 8),
            timestep=torch.tensor([100]),
            encoder_hidden_states=torch.randn(1, 3, 16),
            encoder_attention_mask=torch.ones(1, 3),
        )
        residual = torch.randn(1, 16, 16)
        with torch.no_grad():
            baseline = model(**inputs).sample
            (zero,) = model(**inputs, controlnet_block_samples=[torch.zeros_like(residual)], return_dict=False)
            conditioned = model(**inputs, controlnet_block_samples=[residual]).sample
        self.assertEqual(baseline.shape, (1, 8, 8, 8))
        torch.testing.assert_close(zero, baseline)
        self.assertTrue(torch.isfinite(conditioned).all())
        self.assertFalse(torch.allclose(conditioned, baseline))

    def test_invalid_adaptive_norm_config(self):
        with self.assertRaisesRegex(NotImplementedError, "norm_type"):
            PixArtTransformer2DModel(norm_type="layer_norm")
        with self.assertRaisesRegex(ValueError, "num_embeds_ada_norm"):
            PixArtTransformer2DModel(num_embeds_ada_norm=None)

    def test_forward_accepts_tokenwise_timesteps(self):
        model = self.make_model(num_layers=1)

        output = model(
            hidden_states=torch.randn(1, 4, 8, 8),
            encoder_hidden_states=torch.randn(1, 3, 16),
            timestep=torch.tensor([[100, 900, 250, 750, 100, 900, 250, 750, 100, 900, 250, 750, 100, 900, 250, 750]]),
            encoder_attention_mask=torch.ones(1, 3),
            return_dict=False,
        )[0]

        self.assertEqual(output.shape, (1, 8, 8, 8))

    def test_forward_rejects_wrong_tokenwise_timestep_length(self):
        model = self.make_model(num_layers=1)

        with self.assertRaisesRegex(ValueError, "expected shape"):
            model(
                hidden_states=torch.randn(1, 4, 8, 8),
                encoder_hidden_states=torch.randn(1, 3, 16),
                timestep=torch.tensor([[100, 900]]),
                encoder_attention_mask=torch.ones(1, 3),
                return_dict=False,
            )

    def test_segmented_checkpointing_uses_sequential_state_helper(self):
        model = self.make_model(num_layers=4)
        model.train()
        model.gradient_checkpointing = True
        model.gradient_checkpointing_interval = 2
        model.gradient_checkpointing_segment_stride = 4

        def fake_checkpoint_sequential_state(
            blocks,
            segment_size,
            state,
            run_block,
            checkpoint_fn,
            checkpoint_kwargs=None,
            segment_stride=None,
        ):
            return state

        with patch(
            "simpletuner.helpers.models.pixart.transformer.checkpoint_sequential_state",
            side_effect=fake_checkpoint_sequential_state,
        ) as checkpoint_sequential:
            output = model(
                hidden_states=torch.randn(1, 4, 8, 8, requires_grad=True),
                encoder_hidden_states=torch.randn(1, 3, 16),
                timestep=torch.tensor([100]),
                encoder_attention_mask=torch.ones(1, 3),
                return_dict=False,
            )[0]

        self.assertEqual(output.shape, (1, 8, 8, 8))
        checkpoint_sequential.assert_called_once()
        args, kwargs = checkpoint_sequential.call_args
        self.assertIs(args[0], model.transformer_blocks)
        self.assertEqual(args[1], 2)
        self.assertEqual(kwargs, {"segment_stride": 4})
