"""Model-level segmented checkpointing capability coverage."""

import unittest


def assert_checkpointing_controls(
    test_case,
    model_cls,
    *,
    backend=False,
    interval=False,
    stride=False,
    checkpoint_attention_offload=False,
    ffn=False,
    attention_offload=False,
):
    if backend:
        test_case.assertTrue(hasattr(model_cls, "set_gradient_checkpointing_backend"))
    if interval:
        test_case.assertTrue(hasattr(model_cls, "set_gradient_checkpointing_interval"))
    if stride:
        test_case.assertTrue(hasattr(model_cls, "set_gradient_checkpointing_segment_stride"))
    if checkpoint_attention_offload:
        test_case.assertTrue(hasattr(model_cls, "set_gradient_checkpointing_offload_attention"))
    if ffn:
        test_case.assertTrue(getattr(model_cls, "_supports_ffn_gradient_checkpointing", False))
    if attention_offload:
        test_case.assertTrue(getattr(model_cls, "_supports_attention_activation_offload", False))


class AceStepSegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.ace_step.transformer import ACEStepTransformer2DModel

        assert_checkpointing_controls(
            self,
            ACEStepTransformer2DModel,
            backend=True,
            interval=True,
            stride=True,
        )


class AuraFlowSegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.auraflow.transformer import AuraFlowTransformer2DModel

        assert_checkpointing_controls(
            self,
            AuraFlowTransformer2DModel,
            backend=True,
            interval=True,
            stride=True,
            checkpoint_attention_offload=False,
            ffn=False,
            attention_offload=False,
        )


class BooguImageSegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.boogu_image.transformer import BooguImageTransformer2DModel

        assert_checkpointing_controls(
            self,
            BooguImageTransformer2DModel,
            backend=False,
            interval=True,
            stride=True,
            checkpoint_attention_offload=False,
            ffn=False,
            attention_offload=False,
        )


class ChromaSegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.chroma.transformer import ChromaTransformer2DModel

        assert_checkpointing_controls(
            self,
            ChromaTransformer2DModel,
            backend=True,
            interval=True,
            stride=True,
            checkpoint_attention_offload=True,
            ffn=True,
            attention_offload=True,
        )

    def test_ffn_checkpointing_uses_non_reentrant_checkpoint(self):
        import torch

        from simpletuner.helpers.models.chroma.transformer import ChromaTransformerBlock

        checkpoint_kwargs = {}

        def checkpoint_fn(function, *args, **kwargs):
            checkpoint_kwargs.update(kwargs)
            return function(*args)

        block = ChromaTransformerBlock(dim=16, num_attention_heads=2, attention_head_dim=8).train()
        encoder_hidden_states, hidden_states = block(
            hidden_states=torch.randn(2, 4, 16, requires_grad=True),
            encoder_hidden_states=torch.randn(2, 3, 16, requires_grad=True),
            image_temb=torch.randn(2, 6, 16),
            text_temb=torch.randn(2, 6, 16),
            checkpoint_ffn=True,
            checkpoint_fn=checkpoint_fn,
        )

        self.assertEqual(checkpoint_kwargs, {"use_reentrant": False})
        self.assertEqual(encoder_hidden_states.shape, (2, 3, 16))
        self.assertEqual(hidden_states.shape, (2, 4, 16))


class CosmosSegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.cosmos.transformer import CosmosTransformer3DModel

        assert_checkpointing_controls(
            self,
            CosmosTransformer3DModel,
            backend=False,
            interval=True,
            stride=True,
            checkpoint_attention_offload=False,
            ffn=False,
            attention_offload=False,
        )


class Cosmos3SegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.cosmos3.transformer import Cosmos3OmniTransformer

        assert_checkpointing_controls(
            self,
            Cosmos3OmniTransformer,
            backend=False,
            interval=True,
            stride=True,
            checkpoint_attention_offload=False,
            ffn=False,
            attention_offload=False,
        )


class ErnieSegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.ernie.transformer import ErnieImageTransformer2DModel

        assert_checkpointing_controls(
            self,
            ErnieImageTransformer2DModel,
            backend=True,
            interval=True,
            stride=True,
            checkpoint_attention_offload=False,
            ffn=False,
            attention_offload=False,
        )


class FluxSegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.flux.transformer import FluxTransformer2DModel

        assert_checkpointing_controls(
            self,
            FluxTransformer2DModel,
            backend=True,
            interval=True,
            stride=True,
            offload=True,
            ffn=True,
            attention_offload=True,
        )


class FluxBlockCheckpointingScopeTests(unittest.TestCase):
    def test_blocks_accept_ffn_checkpoint_and_attention_offload_scope(self):
        import torch

        from simpletuner.helpers.models.flux.transformer import FluxSingleTransformerBlock, FluxTransformerBlock

        double_block = FluxTransformerBlock(dim=16, num_attention_heads=2, attention_head_dim=8).train()
        hidden = torch.randn(2, 4, 16, requires_grad=True)
        encoder_hidden = torch.randn(2, 3, 16, requires_grad=True)
        temb = torch.randn(2, 16)

        expected_encoder, expected_hidden = double_block(hidden, encoder_hidden, temb)
        actual_encoder, actual_hidden = double_block(
            hidden,
            encoder_hidden,
            temb,
            checkpoint_ffn=True,
            checkpoint_fn=torch.utils.checkpoint.checkpoint,
            offload_attention=True,
        )
        self.assertTrue(torch.allclose(expected_encoder, actual_encoder, atol=1e-6))
        self.assertTrue(torch.allclose(expected_hidden, actual_hidden, atol=1e-6))

        single_block = FluxSingleTransformerBlock(dim=16, num_attention_heads=2, attention_head_dim=8).train()
        hidden = torch.randn(2, 7, 16, requires_grad=True)
        temb = torch.randn(2, 16)

        expected_hidden = single_block(hidden, temb)
        actual_hidden = single_block(
            hidden, temb, checkpoint_ffn=True, checkpoint_fn=torch.utils.checkpoint.checkpoint, offload_attention=True
        )
        self.assertTrue(torch.allclose(expected_hidden, actual_hidden, atol=1e-6))
