import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers.models.attention_processor import Attention

from simpletuner.helpers.models.chroma.transformer import ChromaTransformer2DModel
from simpletuner.helpers.models.flux.attention import FluxFusedFlashAttnProcessor3
from simpletuner.helpers.models.ltxvideo2.transformer import LTX2Attention, LTX2VideoTransformer3DModel
from simpletuner.helpers.models.lumina2.transformer import Lumina2PackedAttnProcessor2_0
from simpletuner.helpers.training.packed_attention_processors import (
    PackedAuraFlowAttnProcessor2_0,
    PackedFusedAttnProcessor2_0,
    PackedJointAttnProcessor2_0,
)


class _FakePackedBackend:
    def __init__(self):
        self.calls = []
        self.capabilities = SimpleNamespace(fixed_qkvpacked=True, varlen_qkvpacked=True, varlen_unpacked=True)

    def qkvpacked(self, qkv, attention_mask=None, causal=False, softmax_scale=None):
        self.calls.append(
            {
                "shape": tuple(qkv.shape),
                "mask": attention_mask,
                "causal": causal,
                "softmax_scale": softmax_scale,
            }
        )
        return qkv[:, :, 0]


class PackedAttentionProcessorTests(unittest.TestCase):
    def _patch_backend(self, backend):
        return patch(
            "simpletuner.helpers.training.packed_attention_processors.get_packed_attention_backend",
            return_value=backend,
        )

    def test_packed_fused_self_attention_dispatches_qkvpacked(self):
        backend = _FakePackedBackend()
        attn = Attention(query_dim=8, heads=2, dim_head=4, out_dim=8, bias=True, out_bias=True)
        attn.fuse_projections(fuse=True)
        processor = PackedFusedAttnProcessor2_0(preferred_backend="flash2-hub")
        hidden_states = torch.randn(2, 5, 8)

        with self._patch_backend(backend):
            output = processor(attn, hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(backend.calls[0]["shape"], (2, 5, 3, 2, 4))
        self.assertIsNone(backend.calls[0]["mask"])

    def test_packed_joint_attention_concatenates_context_and_sample(self):
        backend = _FakePackedBackend()
        attn = Attention(
            query_dim=8,
            added_kv_proj_dim=8,
            heads=2,
            dim_head=4,
            out_dim=8,
            context_pre_only=False,
            bias=True,
            out_bias=True,
        )
        attn.fuse_projections(fuse=True)
        processor = PackedJointAttnProcessor2_0(preferred_backend="flash2-hub")
        hidden_states = torch.randn(2, 5, 8)
        encoder_hidden_states = torch.randn(2, 3, 8)

        with self._patch_backend(backend):
            sample, context = processor(attn, hidden_states, encoder_hidden_states)

        self.assertEqual(sample.shape, hidden_states.shape)
        self.assertEqual(context.shape, encoder_hidden_states.shape)
        self.assertEqual(backend.calls[0]["shape"], (2, 8, 3, 2, 4))

    def test_packed_auraflow_attention_preserves_context_first_order(self):
        backend = _FakePackedBackend()
        attn = Attention(
            query_dim=8,
            added_kv_proj_dim=8,
            heads=2,
            dim_head=4,
            out_dim=8,
            context_pre_only=False,
            bias=False,
            out_bias=False,
        )
        attn.fuse_projections(fuse=True)
        processor = PackedAuraFlowAttnProcessor2_0(preferred_backend="flash2-hub")
        hidden_states = torch.randn(2, 5, 8)
        encoder_hidden_states = torch.randn(2, 3, 8)

        with self._patch_backend(backend):
            sample, context = processor(attn, hidden_states, encoder_hidden_states)

        self.assertEqual(sample.shape, hidden_states.shape)
        self.assertEqual(context.shape, encoder_hidden_states.shape)
        self.assertEqual(backend.calls[0]["shape"], (2, 8, 3, 2, 4))

    def test_lumina2_packed_processor_handles_gqa_and_bool_mask(self):
        backend = _FakePackedBackend()
        attn = Attention(
            query_dim=16,
            cross_attention_dim=None,
            heads=4,
            kv_heads=2,
            dim_head=4,
            qk_norm="rms_norm",
            bias=False,
            out_bias=False,
            out_dim=16,
        )
        attn.fuse_projections(fuse=True)
        processor = Lumina2PackedAttnProcessor2_0(preferred_backend="flash2-hub")
        hidden_states = torch.randn(2, 5, 16)
        attention_mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])

        with self._patch_backend(backend):
            output = processor(attn, hidden_states, hidden_states, attention_mask=attention_mask)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(backend.calls[0]["shape"], (2, 5, 3, 4, 4))
        self.assertTrue(torch.equal(backend.calls[0]["mask"], attention_mask))

    def test_ltx2_self_attention_uses_packed_backend_when_enabled(self):
        backend = _FakePackedBackend()
        attn = LTX2Attention(query_dim=8, heads=2, kv_heads=2, dim_head=4, bias=True, out_bias=True)
        attn.fuse_projections()
        attn.processor._packed_attention_backend = "flash2-hub"
        hidden_states = torch.randn(2, 5, 8)

        with self._patch_backend(backend):
            output = attn(hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(backend.calls[0]["shape"], (2, 5, 3, 2, 4))

    def test_chroma_transformer_fuse_installs_flux_packed_processor(self):
        backend = _FakePackedBackend()
        model = ChromaTransformer2DModel(
            in_channels=4,
            out_channels=4,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=8,
            num_attention_heads=2,
            joint_attention_dim=16,
            axes_dims_rope=(2, 2, 4),
            approximator_num_channels=16,
            approximator_hidden_dim=32,
            approximator_layers=1,
        )

        with patch("simpletuner.helpers.models.flux.attention.get_packed_attention_backend", return_value=backend):
            model.fuse_qkv_projections(preferred_backend="flash2-hub")

        self.assertIsInstance(model.transformer_blocks[0].attn.processor, FluxFusedFlashAttnProcessor3)
        self.assertIsInstance(model.single_transformer_blocks[0].attn.processor, FluxFusedFlashAttnProcessor3)
        model.unfuse_qkv_projections()
        self.assertNotIsInstance(model.transformer_blocks[0].attn.processor, FluxFusedFlashAttnProcessor3)
        self.assertNotIsInstance(model.single_transformer_blocks[0].attn.processor, FluxFusedFlashAttnProcessor3)

    def test_ltx2_transformer_fuse_enables_packed_self_attention_processors(self):
        model = LTX2VideoTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=4,
            cross_attention_dim=8,
            audio_in_channels=4,
            audio_out_channels=4,
            audio_num_attention_heads=2,
            audio_attention_head_dim=4,
            audio_cross_attention_dim=8,
            caption_channels=8,
            num_layers=1,
        )

        model.fuse_qkv_projections(preferred_backend="flash2-hub")
        block = model.transformer_blocks[0]
        self.assertEqual(block.attn1.processor._packed_attention_backend, "flash2-hub")
        self.assertEqual(block.audio_attn1.processor._packed_attention_backend, "flash2-hub")
        self.assertEqual(block.attn2.processor._packed_attention_backend, "flash2-hub")
        model.unfuse_qkv_projections()
        self.assertIsNone(block.attn1.processor._packed_attention_backend)
        self.assertIsNone(block.audio_attn1.processor._packed_attention_backend)
        self.assertIsNone(block.attn2.processor._packed_attention_backend)


if __name__ == "__main__":
    unittest.main()
