import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from simpletuner.helpers.models.sd3.transformer import SD3Transformer2DModel


class TestSD3Transformer2DModel(unittest.TestCase):
    def make_model(self, num_layers):
        return SD3Transformer2DModel(
            sample_size=8,
            patch_size=2,
            in_channels=4,
            out_channels=4,
            attention_head_dim=8,
            num_attention_heads=2,
            num_layers=num_layers,
            joint_attention_dim=32,
            caption_projection_dim=16,
            pooled_projection_dim=16,
            pos_embed_max_size=8,
        )

    def test_forward_with_controlnet_residual(self):
        model = self.make_model(num_layers=2)
        inputs = dict(
            hidden_states=torch.randn(1, 4, 8, 8),
            timestep=torch.tensor([100]),
            encoder_hidden_states=torch.randn(1, 3, 32),
            pooled_projections=torch.randn(1, 16),
        )
        residual = torch.randn(1, 16, 16)
        with torch.no_grad():
            baseline = model(**inputs).sample
            (zero,) = model(**inputs, block_controlnet_hidden_states=[torch.zeros_like(residual)], return_dict=False)
            conditioned = model(**inputs, block_controlnet_hidden_states=[residual]).sample
        self.assertEqual(baseline.shape, (1, 4, 8, 8))
        torch.testing.assert_close(zero, baseline)
        self.assertTrue(torch.isfinite(conditioned).all())
        self.assertFalse(torch.allclose(conditioned, baseline))

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
            "simpletuner.helpers.models.sd3.transformer.checkpoint_sequential_state",
            side_effect=fake_checkpoint_sequential_state,
        ) as checkpoint_sequential:
            output = model(
                hidden_states=torch.randn(1, 4, 8, 8, requires_grad=True),
                encoder_hidden_states=torch.randn(1, 3, 32),
                pooled_projections=torch.randn(1, 16),
                timestep=torch.tensor([100]),
                return_dict=False,
            )[0]

        self.assertEqual(output.shape, (1, 4, 8, 8))
        checkpoint_sequential.assert_called_once()
        args, kwargs = checkpoint_sequential.call_args
        self.assertIs(args[0], model.transformer_blocks)
        self.assertEqual(args[1], 2)
        self.assertEqual(kwargs, {"segment_stride": 4})

    def test_attention_offload_context_is_used(self):
        model = self.make_model(num_layers=1)
        model.train()
        model.set_gradient_checkpointing_offload_attention(True)

        with patch(
            "simpletuner.helpers.models.sd3.transformer.activation_offload_context",
            side_effect=lambda *args, **kwargs: nullcontext(),
        ) as offload_context:
            output = model(
                hidden_states=torch.randn(1, 4, 8, 8, requires_grad=True),
                encoder_hidden_states=torch.randn(1, 3, 32),
                pooled_projections=torch.randn(1, 16),
                timestep=torch.tensor([100]),
                return_dict=False,
            )[0]

        self.assertEqual(output.shape, (1, 4, 8, 8))
        self.assertTrue(model.gradient_checkpointing_offload_attention)
        offload_context.assert_called()
        self.assertTrue(any(call.args and call.args[0] is True for call in offload_context.call_args_list))
