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
            checkpoint_attention_offload=True,
            ffn=True,
            attention_offload=True,
        )

    def test_checkpointing_forwards_attention_offload_to_block_wrappers(self):
        from unittest.mock import patch

        import torch
        import torch.nn as nn

        from simpletuner.helpers.models.flux.transformer import FluxTransformer2DModel

        class RecordingDoubleBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.received_offload_attention = None

            def forward(
                self,
                hidden_states,
                encoder_hidden_states,
                temb,
                context_temb=None,
                image_rotary_emb=None,
                attention_mask=None,
                checkpoint_ffn=False,
                checkpoint_fn=None,
                offload_attention=False,
            ):
                self.received_offload_attention = offload_attention
                return encoder_hidden_states + hidden_states.mean() * 0, hidden_states + temb.mean() * 0

        class RecordingSingleBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.received_offload_attention = None

            def forward(
                self,
                hidden_states,
                temb,
                image_rotary_emb=None,
                attention_mask=None,
                checkpoint_ffn=False,
                checkpoint_fn=None,
                offload_attention=False,
            ):
                self.received_offload_attention = offload_attention
                return hidden_states + temb.mean() * 0

        model = FluxTransformer2DModel(
            patch_size=1,
            in_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=6,
            num_attention_heads=2,
            joint_attention_dim=12,
            pooled_projection_dim=12,
            axes_dims_rope=(2, 2, 2),
        )
        double_block = RecordingDoubleBlock()
        single_block = RecordingSingleBlock()
        model.transformer_blocks[0] = double_block
        model.single_transformer_blocks[0] = single_block
        model.train()
        model.gradient_checkpointing = True
        model.set_gradient_checkpointing_offload_attention(True)

        def fake_checkpoint(function, *args, **kwargs):
            return function(*args)

        with patch("simpletuner.helpers.models.flux.transformer.simpletuner_checkpoint", side_effect=fake_checkpoint):
            model(
                hidden_states=torch.randn(1, 2, 4, requires_grad=True),
                encoder_hidden_states=torch.randn(1, 3, 12),
                pooled_projections=torch.randn(1, 12),
                timestep=torch.tensor([1.0]),
                img_ids=torch.zeros(2, 3),
                txt_ids=torch.zeros(3, 3),
                return_dict=True,
            )

        self.assertTrue(double_block.received_offload_attention)
        self.assertTrue(single_block.received_offload_attention)


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


class Flux2SegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.flux2.transformer import Flux2Transformer2DModel

        assert_checkpointing_controls(
            self,
            Flux2Transformer2DModel,
            backend=True,
            interval=True,
            stride=True,
            checkpoint_attention_offload=True,
            ffn=False,
            attention_offload=True,
        )


class HiDreamSegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.hidream.transformer import HiDreamImageTransformer2DModel

        assert_checkpointing_controls(
            self,
            HiDreamImageTransformer2DModel,
            backend=True,
            interval=True,
            stride=True,
            checkpoint_attention_offload=False,
            ffn=False,
            attention_offload=False,
        )


class HunyuanVideoSegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.hunyuanvideo.transformer import HunyuanVideo15Transformer3DModel

        assert_checkpointing_controls(
            self,
            HunyuanVideo15Transformer3DModel,
            backend=True,
            interval=True,
            stride=True,
            checkpoint_attention_offload=True,
            ffn=False,
            attention_offload=True,
        )


class IdeogramSegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.ideogram.transformer import Ideogram4Transformer

        assert_checkpointing_controls(
            self,
            Ideogram4Transformer,
            backend=True,
            interval=True,
            stride=True,
            checkpoint_attention_offload=False,
            ffn=False,
            attention_offload=False,
        )


class Kandinsky5SegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.kandinsky5_video.transformer_kandinsky5 import Kandinsky5Transformer3DModel

        assert_checkpointing_controls(
            self,
            Kandinsky5Transformer3DModel,
            backend=True,
            interval=True,
            stride=True,
            checkpoint_attention_offload=True,
            ffn=False,
            attention_offload=True,
        )


class KolorsControlNetCheckpointingCompatibilityTests(unittest.TestCase):
    def test_gradient_checkpointing_signature_accepts_diffusers_kwargs(self):
        import inspect

        from simpletuner.helpers.models.kolors.controlnet import ControlNetModel

        parameters = inspect.signature(ControlNetModel._set_gradient_checkpointing).parameters
        self.assertIn("enable", parameters)
        self.assertIn("gradient_checkpointing_func", parameters)


class Krea2SegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.krea2.transformer import Krea2Transformer2DModel

        assert_checkpointing_controls(
            self,
            Krea2Transformer2DModel,
            backend=True,
            interval=True,
            stride=True,
            checkpoint_attention_offload=True,
            ffn=True,
            attention_offload=True,
        )


class LongCatImageSegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.longcat_image.transformer import LongCatImageTransformer2DModel

        assert_checkpointing_controls(
            self,
            LongCatImageTransformer2DModel,
            backend=False,
            interval=True,
            stride=True,
            checkpoint_attention_offload=True,
            ffn=False,
            attention_offload=True,
        )


class LongCatVideoSegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.longcat_video.transformer import LongCatVideoTransformer3DModel

        assert_checkpointing_controls(
            self,
            LongCatVideoTransformer3DModel,
            backend=True,
            interval=True,
            stride=True,
            checkpoint_attention_offload=True,
            ffn=False,
            attention_offload=True,
        )


class LTXVideoSegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.ltxvideo.transformer import LTXVideoTransformer3DModel

        assert_checkpointing_controls(
            self,
            LTXVideoTransformer3DModel,
            backend=True,
            interval=True,
            stride=True,
            checkpoint_attention_offload=False,
            ffn=False,
            attention_offload=False,
        )


class LTXVideo2SegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.ltxvideo2.transformer import LTX2VideoTransformer3DModel

        assert_checkpointing_controls(
            self,
            LTX2VideoTransformer3DModel,
            backend=True,
            interval=True,
            stride=True,
            checkpoint_attention_offload=True,
            ffn=True,
            attention_offload=True,
        )


class Lumina2SegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.lumina2.transformer import Lumina2Transformer2DModel

        assert_checkpointing_controls(
            self,
            Lumina2Transformer2DModel,
            backend=True,
            interval=True,
            stride=True,
            checkpoint_attention_offload=False,
            ffn=False,
            attention_offload=False,
        )


class MageFlowSegmentedCheckpointingSupportTests(unittest.TestCase):
    def test_checkpointing_controls(self):
        from simpletuner.helpers.models.mageflow.transformer import MageFlowTransformer2DModel

        assert_checkpointing_controls(
            self,
            MageFlowTransformer2DModel,
            backend=True,
            interval=True,
            stride=True,
            checkpoint_attention_offload=True,
            ffn=True,
            attention_offload=True,
        )
