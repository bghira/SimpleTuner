import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers.models.attention_processor import Attention

from simpletuner.helpers.models.chroma.transformer import ChromaTransformer2DModel
from simpletuner.helpers.models.flux2.transformer import Flux2Attention, Flux2ParallelSelfAttention
from simpletuner.helpers.models.flux.attention import FluxFusedFlashAttnProcessor3
from simpletuner.helpers.models.ltxvideo2.transformer import (
    LTX2Attention,
    LTX2VideoTransformer3DModel,
    _ltx2_prepare_attention_mask,
)
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

    def test_flux_fused_flash_varlen_backend_creates_all_valid_mask(self):
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
        hidden_states = torch.randn(2, 5, 8)
        encoder_hidden_states = torch.randn(2, 3, 8)

        with patch("simpletuner.helpers.models.flux.attention.get_packed_attention_backend", return_value=backend):
            processor = FluxFusedFlashAttnProcessor3(preferred_backend="flash-attn-3-varlen-hub")
            sample, context = processor(attn, hidden_states, encoder_hidden_states)

        self.assertEqual(sample.shape, hidden_states.shape)
        self.assertEqual(context.shape, encoder_hidden_states.shape)
        self.assertEqual(backend.calls[0]["shape"], (2, 8, 3, 2, 4))
        self.assertEqual(backend.calls[0]["mask"].shape, (2, 8))
        self.assertEqual(backend.calls[0]["mask"].dtype, torch.bool)
        self.assertTrue(backend.calls[0]["mask"].all())

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

    def test_packed_joint_attention_splits_4d_sample_by_sequence_length(self):
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
        hidden_states = torch.randn(2, 8, 4, 5)
        encoder_hidden_states = torch.randn(2, 3, 8)

        with self._patch_backend(backend):
            sample, context = processor(attn, hidden_states, encoder_hidden_states)

        self.assertEqual(sample.shape, hidden_states.shape)
        self.assertEqual(context.shape, encoder_hidden_states.shape)
        self.assertEqual(backend.calls[0]["shape"], (2, 23, 3, 2, 4))

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

    def test_ltx2_self_attention_uses_fused_projection_after_split_layers_removed(self):
        backend = _FakePackedBackend()
        attn = LTX2Attention(query_dim=8, heads=2, kv_heads=2, dim_head=4, bias=True, out_bias=True)
        attn.fuse_projections()
        delattr(attn, "to_q")
        delattr(attn, "to_k")
        delattr(attn, "to_v")
        attn.processor._packed_attention_backend = "flash2-hub"
        hidden_states = torch.randn(2, 5, 8)

        with self._patch_backend(backend):
            output = attn(hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertEqual(backend.calls[0]["shape"], (2, 5, 3, 2, 4))

    def test_ltx2_cross_attention_fuses_kv_for_different_context_dim(self):
        attn = LTX2Attention(
            query_dim=8,
            cross_attention_dim=12,
            heads=2,
            kv_heads=2,
            dim_head=4,
            bias=True,
            out_bias=True,
        )
        attn.fuse_projections()
        delattr(attn, "to_k")
        delattr(attn, "to_v")
        hidden_states = torch.randn(2, 5, 8)
        encoder_hidden_states = torch.randn(2, 3, 12)

        output = attn(hidden_states, encoder_hidden_states=encoder_hidden_states)

        self.assertEqual(output.shape, hidden_states.shape)
        self.assertTrue(hasattr(attn, "to_kv"))
        self.assertFalse(hasattr(attn, "to_qkv"))

    def test_ltx2_ulysses_attention_mask_uses_post_exchange_key_length(self):
        attn = LTX2Attention(query_dim=8, heads=4, kv_heads=4, dim_head=2)
        attention_mask = torch.tensor([[-10000, -10000, 0, 0, 0, 0, 0, 0]], dtype=torch.float32)
        parallel_config = SimpleNamespace(context_parallel_config=SimpleNamespace(ulysses_anything=True, ulysses_degree=2))

        prepared = _ltx2_prepare_attention_mask(attn, attention_mask, 3, 1, parallel_config)

        self.assertEqual(prepared.shape, (1, 1, 1, 6))
        self.assertTrue(torch.equal(prepared, torch.zeros_like(prepared)))

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

    def test_flux2_double_stream_context_parallel_uses_distributed_dispatch(self):
        parallel_config = SimpleNamespace(context_parallel_config=SimpleNamespace(ulysses_degree=2))
        attn = Flux2Attention(
            query_dim=8,
            heads=2,
            dim_head=4,
            added_kv_proj_dim=8,
            out_dim=8,
        )
        attn.processor._parallel_config = parallel_config
        attn.processor._packed_attention_backend = "flash2-hub"
        hidden_states = torch.randn(1, 2, 8)
        encoder_hidden_states = torch.randn(1, 2, 8)
        attention_mask = torch.ones(1, 8, dtype=torch.bool)

        with (
            patch("simpletuner.helpers.models.flux2.transformer._run_packed_qkv_attention") as packed,
            patch(
                "simpletuner.helpers.models.flux2.transformer.dispatch_attention_fn",
                side_effect=lambda query, *_args, **_kwargs: torch.zeros_like(query),
            ) as dispatch,
        ):
            sample, context = attn.processor(
                attn,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )

        packed.assert_not_called()
        self.assertEqual(sample.shape, hidden_states.shape)
        self.assertEqual(context.shape, encoder_hidden_states.shape)
        self.assertIs(dispatch.call_args.kwargs["parallel_config"], parallel_config)
        self.assertEqual(dispatch.call_args.kwargs["attn_mask"].shape, (1, 1, 1, 8))

    def test_flux2_single_stream_context_parallel_uses_distributed_dispatch(self):
        parallel_config = SimpleNamespace(context_parallel_config=SimpleNamespace(ulysses_degree=2))
        attn = Flux2ParallelSelfAttention(query_dim=8, heads=2, dim_head=4, out_dim=8, mlp_ratio=1.0)
        attn.processor._parallel_config = parallel_config
        attn.processor._packed_attention_backend = "flash2-hub"
        hidden_states = torch.randn(1, 3, 8)
        attention_mask = torch.ones(1, 6, dtype=torch.bool)

        with (
            patch("simpletuner.helpers.models.flux2.transformer._run_packed_qkv_attention") as packed,
            patch(
                "simpletuner.helpers.models.flux2.transformer.dispatch_attention_fn",
                side_effect=lambda query, *_args, **_kwargs: torch.zeros_like(query),
            ) as dispatch,
        ):
            output = attn.processor(attn, hidden_states, attention_mask=attention_mask)

        packed.assert_not_called()
        self.assertEqual(output.shape, hidden_states.shape)
        self.assertIs(dispatch.call_args.kwargs["parallel_config"], parallel_config)
        self.assertEqual(dispatch.call_args.kwargs["attn_mask"].shape, (1, 1, 1, 6))

    def test_ltx2_transformer_fuse_enables_packed_self_attention_processors(self):
        model = LTX2VideoTransformer3DModel(
            in_channels=4,
            out_channels=4,
            num_attention_heads=2,
            attention_head_dim=4,
            cross_attention_dim=12,
            audio_in_channels=4,
            audio_out_channels=4,
            audio_num_attention_heads=1,
            audio_attention_head_dim=4,
            audio_cross_attention_dim=10,
            caption_channels=12,
            num_layers=1,
        )

        model.fuse_qkv_projections(preferred_backend="flash2-hub")
        block = model.transformer_blocks[0]
        self.assertEqual(block.attn1.processor._packed_attention_backend, "flash2-hub")
        self.assertEqual(block.audio_attn1.processor._packed_attention_backend, "flash2-hub")
        self.assertEqual(block.attn2.processor._packed_attention_backend, "flash2-hub")
        self.assertTrue(hasattr(block.attn1, "to_qkv"))
        self.assertTrue(hasattr(block.audio_attn1, "to_qkv"))
        self.assertTrue(hasattr(block.attn2, "to_kv"))
        self.assertTrue(hasattr(block.audio_attn2, "to_kv"))
        self.assertTrue(hasattr(block.audio_to_video_attn, "to_kv"))
        self.assertTrue(hasattr(block.video_to_audio_attn, "to_kv"))
        model.unfuse_qkv_projections()
        self.assertIsNone(block.attn1.processor._packed_attention_backend)
        self.assertIsNone(block.audio_attn1.processor._packed_attention_backend)
        self.assertIsNone(block.attn2.processor._packed_attention_backend)


if __name__ == "__main__":
    unittest.main()
