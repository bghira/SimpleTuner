"""
Integration tests for GLIGEN grounding token injection and forward pass.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.training.grounding.gligen_layers import (
    _extract_block_dims,
    apply_grounding_fuser,
    enable_all_fusers,
    get_gligen_trainable_parameters,
    inject_gligen_layers,
    make_grounding_step_callback,
)
from simpletuner.helpers.training.grounding.types import GroundingBatch


class TestInjectGligenLayers(unittest.TestCase):
    """Test inject_gligen_layers adds position_net and fuser to a UNet."""

    def _make_mock_unet(self, cross_attention_dim=768, n_blocks=4, n_heads=8):
        """Build a minimal mock UNet with BasicTransformerBlocks."""
        from diffusers.models.attention import BasicTransformerBlock

        blocks = []
        for _ in range(n_blocks):
            block = BasicTransformerBlock(
                dim=cross_attention_dim,
                num_attention_heads=n_heads,
                attention_head_dim=cross_attention_dim // n_heads,
                cross_attention_dim=cross_attention_dim,
            )
            blocks.append(block)

        unet = torch.nn.Module()
        unet.blocks = torch.nn.ModuleList(blocks)
        unet.config = MagicMock()
        unet.config.cross_attention_dim = cross_attention_dim
        return unet

    def test_position_net_created(self):
        unet = self._make_mock_unet(cross_attention_dim=768)
        inject_gligen_layers(unet, positive_len=768, cross_attention_dim=768)
        self.assertTrue(hasattr(unet, "position_net"))

    def test_fuser_added_to_blocks(self):
        unet = self._make_mock_unet(cross_attention_dim=768, n_blocks=3)
        inject_gligen_layers(unet, positive_len=768, cross_attention_dim=768)
        from diffusers.models.attention import BasicTransformerBlock

        for module in unet.modules():
            if isinstance(module, BasicTransformerBlock):
                self.assertTrue(hasattr(module, "fuser"))

    def test_text_only_feature_type(self):
        unet = self._make_mock_unet(cross_attention_dim=768)
        inject_gligen_layers(unet, positive_len=768, cross_attention_dim=768, feature_type="text-only")
        self.assertTrue(hasattr(unet.position_net, "linears"))
        self.assertFalse(hasattr(unet.position_net, "linears_text"))

    def test_text_image_feature_type(self):
        unet = self._make_mock_unet(cross_attention_dim=768)
        inject_gligen_layers(unet, positive_len=768, cross_attention_dim=768, feature_type="text-image")
        self.assertTrue(hasattr(unet.position_net, "linears_text"))
        self.assertTrue(hasattr(unet.position_net, "linears_image"))


class TestInjectGligenLayersCustomBlocks(unittest.TestCase):
    """Test inject_gligen_layers with non-BasicTransformerBlock blocks."""

    def _make_custom_block(self, dim=768, n_heads=8):
        """Build a minimal custom block with an attn1 sub-module."""
        from diffusers.models.attention_processor import Attention

        block = torch.nn.Module()
        block.attn1 = Attention(query_dim=dim, heads=n_heads, dim_head=dim // n_heads)
        block.ff = torch.nn.Linear(dim, dim)
        return block

    def test_auto_detect_custom_blocks(self):
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList([self._make_custom_block() for _ in range(3)])
        inject_gligen_layers(model, positive_len=768, cross_attention_dim=768)
        self.assertTrue(hasattr(model, "position_net"))
        for block in model.blocks:
            self.assertTrue(hasattr(block, "fuser"))

    def test_explicit_block_types(self):
        """When block_types is set, only matching blocks get fusers."""
        from diffusers.models.attention import BasicTransformerBlock

        model = torch.nn.Module()
        model.custom = self._make_custom_block()
        model.basic = BasicTransformerBlock(dim=768, num_attention_heads=8, attention_head_dim=96, cross_attention_dim=768)
        inject_gligen_layers(model, positive_len=768, cross_attention_dim=768, block_types=(BasicTransformerBlock,))
        self.assertTrue(hasattr(model.basic, "fuser"))
        self.assertFalse(hasattr(model.custom, "fuser"))

    def test_skip_blocks_with_existing_fuser(self):
        model = torch.nn.Module()
        block = self._make_custom_block()
        block.fuser = torch.nn.Identity()  # pre-existing fuser
        model.blocks = torch.nn.ModuleList([block])
        inject_gligen_layers(model, positive_len=768, cross_attention_dim=768)
        # Should not overwrite the existing fuser
        self.assertIsInstance(block.fuser, torch.nn.Identity)


class TestExtractBlockDims(unittest.TestCase):
    """Test _extract_block_dims helper."""

    def test_standard_attention(self):
        from diffusers.models.attention_processor import Attention

        block = torch.nn.Module()
        block.attn1 = Attention(query_dim=512, heads=8, dim_head=64)
        dims = _extract_block_dims(block)
        self.assertEqual(dims, (512, 8))

    def test_attn_attr_override(self):
        from diffusers.models.attention_processor import Attention

        block = torch.nn.Module()
        block.my_attn = Attention(query_dim=1024, heads=16, dim_head=64)
        # Default candidates won't find 'my_attn'
        self.assertIsNone(_extract_block_dims(block))
        # With explicit attr
        dims = _extract_block_dims(block, attn_attr="my_attn")
        self.assertEqual(dims, (1024, 16))

    def test_no_attention_returns_none(self):
        block = torch.nn.Module()
        block.ff = torch.nn.Linear(768, 768)
        self.assertIsNone(_extract_block_dims(block))


class TestGetGligenTrainableParameters(unittest.TestCase):
    """Test that get_gligen_trainable_parameters selects only GLIGEN params."""

    def test_only_gligen_params_trainable(self):
        from diffusers.models.attention import BasicTransformerBlock

        unet = torch.nn.Module()
        unet.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
        block = BasicTransformerBlock(
            dim=768,
            num_attention_heads=8,
            attention_head_dim=96,
            cross_attention_dim=768,
        )
        unet.block = block
        inject_gligen_layers(unet, positive_len=768, cross_attention_dim=768)

        params = get_gligen_trainable_parameters(unet)
        self.assertGreater(len(params), 0)

        # All returned params should require grad
        for p in params:
            self.assertTrue(p.requires_grad)

        # Conv params should not require grad
        for p in unet.conv.parameters():
            self.assertFalse(p.requires_grad)


def _make_model_stub(text_image_mode=False, model_type=None):
    """Create a stub that exposes grounding builder methods as bound methods."""
    from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes

    stub = MagicMock(spec=ImageModelFoundation)
    component = MagicMock()
    if text_image_mode:
        component.position_net = MagicMock(spec=["null_text_feature", "null_image_feature"])
    else:
        component.position_net = MagicMock(spec=["null_positive_feature"])
    stub.get_trained_component = MagicMock(return_value=component)
    if model_type is not None:
        stub.MODEL_TYPE = model_type
    # Bind the real methods to the mock
    for method_name in (
        "_build_gligen_cross_attention_kwargs",
        "_build_grounding_position_net_kwargs",
        "_build_validation_grounding_pipeline_kwargs",
    ):
        setattr(stub, method_name, getattr(ImageModelFoundation, method_name).__get__(stub))
    stub._cfg_double_gligen_dict = ImageModelFoundation._cfg_double_gligen_dict
    return stub


class TestBuildGligenCrossAttentionKwargs(unittest.TestCase):
    """Test _build_gligen_cross_attention_kwargs."""

    def test_none_grounding_batch(self):
        stub = _make_model_stub()
        result = stub._build_gligen_cross_attention_kwargs(None)
        self.assertIsNone(result)

    def test_text_only_mode(self):
        stub = _make_model_stub(text_image_mode=False)
        batch = GroundingBatch(
            boxes=torch.zeros(1, 2, 4),
            validity_mask=torch.ones(1, 2),
            spatial_masks=torch.zeros(1, 2, 8, 8),
            text_embeds=torch.randn(1, 2, 768),
            image_embeds=None,
            text_masks=torch.ones(1, 2),
            image_masks=torch.zeros(1, 2),
            max_entities=2,
        )
        result = stub._build_gligen_cross_attention_kwargs(batch)
        self.assertIn("gligen", result)
        self.assertIn("positive_embeddings", result["gligen"])
        self.assertNotIn("phrases_embeddings", result["gligen"])

    def test_text_image_mode(self):
        stub = _make_model_stub(text_image_mode=True)
        batch = GroundingBatch(
            boxes=torch.zeros(1, 2, 4),
            validity_mask=torch.ones(1, 2),
            spatial_masks=torch.zeros(1, 2, 8, 8),
            text_embeds=torch.randn(1, 2, 768),
            image_embeds=torch.randn(1, 2, 768),
            text_masks=torch.ones(1, 2),
            image_masks=torch.ones(1, 2),
            max_entities=2,
        )
        result = stub._build_gligen_cross_attention_kwargs(batch)
        self.assertIn("gligen", result)
        self.assertIn("phrases_embeddings", result["gligen"])
        self.assertIn("image_embeddings", result["gligen"])
        self.assertNotIn("positive_embeddings", result["gligen"])

    def test_text_image_mode_without_image_embeds(self):
        """When position_net is text-image but image_embeds is None, should
        use text-image format with zero image embeddings."""
        stub = _make_model_stub(text_image_mode=True)
        batch = GroundingBatch(
            boxes=torch.zeros(1, 2, 4),
            validity_mask=torch.ones(1, 2),
            spatial_masks=torch.zeros(1, 2, 8, 8),
            text_embeds=torch.randn(1, 2, 768),
            image_embeds=None,
            text_masks=torch.ones(1, 2),
            image_masks=torch.zeros(1, 2),
            max_entities=2,
        )
        result = stub._build_gligen_cross_attention_kwargs(batch)
        self.assertIn("gligen", result)
        self.assertIn("phrases_embeddings", result["gligen"])
        self.assertIn("image_embeddings", result["gligen"])
        self.assertNotIn("positive_embeddings", result["gligen"])
        # image_embeddings should be zeros, image_masks should be zeros
        self.assertTrue(torch.all(result["gligen"]["image_embeddings"] == 0))
        self.assertTrue(torch.all(result["gligen"]["image_masks"] == 0))


class TestBuildGroundingPositionNetKwargs(unittest.TestCase):
    """Test _build_grounding_position_net_kwargs for transformer models."""

    def test_none_returns_none(self):
        stub = _make_model_stub()
        result = stub._build_grounding_position_net_kwargs(None)
        self.assertIsNone(result)

    def test_image_text_only(self):
        stub = _make_model_stub(text_image_mode=False)
        batch = GroundingBatch(
            boxes=torch.zeros(2, 4, 4),
            validity_mask=torch.ones(2, 4),
            spatial_masks=torch.zeros(2, 4, 8, 8),
            text_embeds=torch.randn(2, 4, 768),
            image_embeds=None,
            text_masks=torch.ones(2, 4),
            image_masks=torch.zeros(2, 4),
            max_entities=4,
        )
        result = stub._build_grounding_position_net_kwargs(batch)
        self.assertIn("boxes", result)
        self.assertIn("masks", result)
        self.assertIn("positive_embeddings", result)
        self.assertNotIn("phrases_embeddings", result)
        self.assertEqual(result["boxes"].shape, (2, 4, 4))

    def test_image_text_image_mode(self):
        stub = _make_model_stub(text_image_mode=True)
        batch = GroundingBatch(
            boxes=torch.zeros(2, 4, 4),
            validity_mask=torch.ones(2, 4),
            spatial_masks=torch.zeros(2, 4, 8, 8),
            text_embeds=torch.randn(2, 4, 768),
            image_embeds=torch.randn(2, 4, 1024),
            text_masks=torch.ones(2, 4),
            image_masks=torch.ones(2, 4),
            max_entities=4,
        )
        result = stub._build_grounding_position_net_kwargs(batch)
        self.assertIn("phrases_embeddings", result)
        self.assertIn("image_embeddings", result)
        self.assertNotIn("positive_embeddings", result)

    def test_video_flatten_temporal(self):
        stub = _make_model_stub(text_image_mode=False)
        B, T, N = 2, 4, 3
        batch = GroundingBatch(
            boxes=torch.zeros(B, T, N, 4),
            validity_mask=torch.ones(B, T, N),
            spatial_masks=torch.zeros(B, N, 8, 8),
            text_embeds=torch.randn(B, T, N, 768),
            image_embeds=None,
            text_masks=torch.ones(B, T, N),
            image_masks=torch.zeros(B, T, N),
            max_entities=N,
            num_frames=T,
        )
        result = stub._build_grounding_position_net_kwargs(batch, flatten_temporal=True)
        # (B, T, N, ...) -> (B*T, N, ...)
        self.assertEqual(result["boxes"].shape, (B * T, N, 4))
        self.assertEqual(result["masks"].shape, (B * T, N))
        self.assertEqual(result["positive_embeddings"].shape, (B * T, N, 768))

    def test_video_no_flatten(self):
        stub = _make_model_stub(text_image_mode=False)
        B, T, N = 2, 4, 3
        batch = GroundingBatch(
            boxes=torch.zeros(B, T, N, 4),
            validity_mask=torch.ones(B, T, N),
            spatial_masks=torch.zeros(B, N, 8, 8),
            text_embeds=torch.randn(B, T, N, 768),
            image_embeds=None,
            text_masks=torch.ones(B, T, N),
            image_masks=torch.zeros(B, T, N),
            max_entities=N,
            num_frames=T,
        )
        result = stub._build_grounding_position_net_kwargs(batch, flatten_temporal=False)
        # No flattening: shapes remain (B, T, N, ...)
        self.assertEqual(result["boxes"].shape, (B, T, N, 4))


class TestApplyGroundingFuser(unittest.TestCase):
    """Test apply_grounding_fuser utility."""

    def _make_fuser(self, query_dim=768, context_dim=768, n_heads=8):
        from diffusers.models.attention import GatedSelfAttentionDense

        return GatedSelfAttentionDense(
            query_dim=query_dim,
            context_dim=context_dim,
            n_heads=n_heads,
            d_head=query_dim // n_heads,
        )

    def test_image_only(self):
        fuser = self._make_fuser()
        B, S, D, N = 2, 16, 768, 4
        hidden = torch.randn(B, S, D)
        objs = torch.randn(B, N, D)
        result = apply_grounding_fuser(fuser, hidden, objs)
        self.assertEqual(result.shape, (B, S, D))

    def test_text_image_split(self):
        fuser = self._make_fuser()
        B, txt_len, S_img, D, N = 2, 10, 16, 768, 4
        hidden = torch.randn(B, txt_len + S_img, D)
        objs = torch.randn(B, N, D)
        result = apply_grounding_fuser(fuser, hidden, objs, txt_len=txt_len)
        self.assertEqual(result.shape, (B, txt_len + S_img, D))
        # Text portion should be unchanged (fuser alpha init to 0 so
        # the fuser output is the identity, but let's check shapes)

    def test_video_temporal_fold(self):
        fuser = self._make_fuser()
        B, T, S_frame, D, N = 2, 4, 16, 768, 3
        hidden = torch.randn(B, T * S_frame, D)
        objs = torch.randn(B * T, N, D)
        result = apply_grounding_fuser(
            fuser,
            hidden,
            objs,
            tokens_per_frame=S_frame,
            num_frames=T,
        )
        self.assertEqual(result.shape, (B, T * S_frame, D))

    def test_video_with_text_split(self):
        fuser = self._make_fuser()
        B, T, txt_len, S_frame, D, N = 1, 3, 5, 8, 768, 2
        hidden = torch.randn(B, txt_len + T * S_frame, D)
        objs = torch.randn(B * T, N, D)
        result = apply_grounding_fuser(
            fuser,
            hidden,
            objs,
            txt_len=txt_len,
            tokens_per_frame=S_frame,
            num_frames=T,
        )
        self.assertEqual(result.shape, (B, txt_len + T * S_frame, D))

    def test_video_requires_tokens_per_frame(self):
        fuser = self._make_fuser()
        hidden = torch.randn(2, 64, 768)
        objs = torch.randn(8, 4, 768)
        with self.assertRaises(AssertionError):
            apply_grounding_fuser(fuser, hidden, objs, num_frames=4)

    def test_video_shape_mismatch(self):
        fuser = self._make_fuser()
        hidden = torch.randn(2, 60, 768)  # 60 != 4 * 16
        objs = torch.randn(8, 4, 768)
        with self.assertRaises(AssertionError):
            apply_grounding_fuser(fuser, hidden, objs, tokens_per_frame=16, num_frames=4)


class TestPositionNetForward(unittest.TestCase):
    """Test GLIGENTextBoundingboxProjection forward pass with grounding data."""

    def test_text_only_forward(self):
        from diffusers.models.embeddings import GLIGENTextBoundingboxProjection

        pos_net = GLIGENTextBoundingboxProjection(positive_len=768, out_dim=768, feature_type="text-only")
        B, N = 2, 4
        boxes = torch.rand(B, N, 4)
        masks = torch.ones(B, N)
        positive_embeddings = torch.randn(B, N, 768)

        objs = pos_net(boxes=boxes, masks=masks, positive_embeddings=positive_embeddings)
        self.assertEqual(objs.shape, (B, N, 768))

    def test_text_image_forward(self):
        from diffusers.models.embeddings import GLIGENTextBoundingboxProjection

        pos_net = GLIGENTextBoundingboxProjection(positive_len=768, out_dim=768, feature_type="text-image")
        B, N = 2, 4
        boxes = torch.rand(B, N, 4)
        masks = torch.ones(B, N)
        phrases_masks = torch.ones(B, N)
        image_masks = torch.ones(B, N)
        phrases_embeddings = torch.randn(B, N, 768)
        image_embeddings = torch.randn(B, N, 768)

        objs = pos_net(
            boxes=boxes,
            masks=masks,
            phrases_masks=phrases_masks,
            image_masks=image_masks,
            phrases_embeddings=phrases_embeddings,
            image_embeddings=image_embeddings,
        )
        # text-image mode concatenates text and image objects: (B, 2N, out_dim)
        self.assertEqual(objs.shape, (B, 2 * N, 768))


class TestEnableAllFusers(unittest.TestCase):
    """Test enable_all_fusers utility."""

    def _make_model_with_fusers(self, n_blocks=4, dim=768, n_heads=8):
        from diffusers.models.attention import BasicTransformerBlock

        model = torch.nn.Module()
        blocks = []
        for _ in range(n_blocks):
            block = BasicTransformerBlock(
                dim=dim,
                num_attention_heads=n_heads,
                attention_head_dim=dim // n_heads,
                cross_attention_dim=dim,
            )
            blocks.append(block)
        model.blocks = torch.nn.ModuleList(blocks)
        inject_gligen_layers(model, positive_len=dim, cross_attention_dim=dim)
        return model

    def test_disable_all_fusers(self):
        model = self._make_model_with_fusers(n_blocks=3)
        count = enable_all_fusers(model, enabled=False)
        self.assertEqual(count, 3)
        from diffusers.models.attention import GatedSelfAttentionDense

        for m in model.modules():
            if isinstance(m, GatedSelfAttentionDense):
                self.assertFalse(m.enabled)

    def test_re_enable_all_fusers(self):
        model = self._make_model_with_fusers(n_blocks=3)
        enable_all_fusers(model, enabled=False)
        count = enable_all_fusers(model, enabled=True)
        self.assertEqual(count, 3)
        from diffusers.models.attention import GatedSelfAttentionDense

        for m in model.modules():
            if isinstance(m, GatedSelfAttentionDense):
                self.assertTrue(m.enabled)

    def test_no_fusers_returns_zero(self):
        model = torch.nn.Module()
        model.linear = torch.nn.Linear(10, 10)
        count = enable_all_fusers(model, enabled=False)
        self.assertEqual(count, 0)


class TestMakeGroundingStepCallback(unittest.TestCase):
    """Test make_grounding_step_callback factory."""

    def _make_model_with_fusers(self):
        from diffusers.models.attention import BasicTransformerBlock

        model = torch.nn.Module()
        block = BasicTransformerBlock(dim=768, num_attention_heads=8, attention_head_dim=96, cross_attention_dim=768)
        model.blocks = torch.nn.ModuleList([block])
        inject_gligen_layers(model, positive_len=768, cross_attention_dim=768)
        return model

    def test_callback_disables_at_cutoff(self):
        model = self._make_model_with_fusers()
        # beta=0.3, steps=10 -> cutoff at step 3
        callback = make_grounding_step_callback(model, num_inference_steps=10, scheduled_sampling_beta=0.3)
        from diffusers.models.attention import GatedSelfAttentionDense

        # Before cutoff: fusers should remain enabled
        for step in range(3):
            result = callback(None, step, None, {})
            self.assertIsInstance(result, dict)
            for m in model.modules():
                if isinstance(m, GatedSelfAttentionDense):
                    self.assertTrue(m.enabled, f"Fuser should be enabled at step {step}")

        # At cutoff step 3: fusers should be disabled
        callback(None, 3, None, {})
        for m in model.modules():
            if isinstance(m, GatedSelfAttentionDense):
                self.assertFalse(m.enabled)

    def test_callback_returns_kwargs(self):
        model = self._make_model_with_fusers()
        callback = make_grounding_step_callback(model, num_inference_steps=10)
        result = callback(None, 0, None, {"latents": torch.zeros(1)})
        self.assertIn("latents", result)


class TestGroundingBatchTo(unittest.TestCase):
    """Test GroundingBatch.to() method."""

    def _make_batch(self, device="cpu", dtype=torch.float32):
        return GroundingBatch(
            boxes=torch.zeros(1, 2, 4, device=device, dtype=dtype),
            validity_mask=torch.ones(1, 2, device=device, dtype=dtype),
            spatial_masks=torch.zeros(1, 2, 8, 8, device=device, dtype=dtype),
            text_embeds=torch.randn(1, 2, 768, device=device, dtype=dtype),
            image_embeds=torch.randn(1, 2, 512, device=device, dtype=dtype),
            text_masks=torch.ones(1, 2, device=device, dtype=dtype),
            image_masks=torch.ones(1, 2, device=device, dtype=dtype),
            max_entities=2,
            num_frames=1,
        )

    def test_to_preserves_shapes(self):
        batch = self._make_batch()
        moved = batch.to(device=torch.device("cpu"), dtype=torch.float16)
        self.assertEqual(moved.boxes.shape, batch.boxes.shape)
        self.assertEqual(moved.text_embeds.shape, batch.text_embeds.shape)
        self.assertEqual(moved.image_embeds.shape, batch.image_embeds.shape)

    def test_to_changes_dtype(self):
        batch = self._make_batch(dtype=torch.float32)
        moved = batch.to(device=torch.device("cpu"), dtype=torch.float16)
        self.assertEqual(moved.boxes.dtype, torch.float16)
        self.assertEqual(moved.text_embeds.dtype, torch.float16)

    def test_to_preserves_scalars(self):
        batch = self._make_batch()
        moved = batch.to(device=torch.device("cpu"))
        self.assertEqual(moved.max_entities, batch.max_entities)
        self.assertEqual(moved.num_frames, batch.num_frames)

    def test_to_with_none_image_embeds(self):
        batch = GroundingBatch(
            boxes=torch.zeros(1, 2, 4),
            validity_mask=torch.ones(1, 2),
            spatial_masks=torch.zeros(1, 2, 8, 8),
            text_embeds=torch.randn(1, 2, 768),
            image_embeds=None,
            text_masks=torch.ones(1, 2),
            image_masks=torch.zeros(1, 2),
            max_entities=2,
        )
        moved = batch.to(device=torch.device("cpu"), dtype=torch.float16)
        self.assertIsNone(moved.image_embeds)
        self.assertEqual(moved.text_embeds.dtype, torch.float16)


class TestStateTrackerGroundingCache(unittest.TestCase):
    """Test StateTracker grounding image embed cache methods."""

    def test_set_and_get_cache(self):
        from simpletuner.helpers.training.state_tracker import StateTracker

        mock_cache = MagicMock()
        StateTracker.set_grounding_image_embed_cache("test_backend", mock_cache)
        result = StateTracker.get_grounding_image_embed_cache("test_backend")
        self.assertIs(result, mock_cache)

        # Cleanup
        StateTracker._grounding_image_embed_caches.pop("test_backend", None)

    def test_get_nonexistent_returns_none(self):
        from simpletuner.helpers.training.state_tracker import StateTracker

        result = StateTracker.get_grounding_image_embed_cache("nonexistent_backend")
        self.assertIsNone(result)


class TestCfgDoubleGligenDict(unittest.TestCase):
    """Test _cfg_double_gligen_dict doubles tensors and zeros unconditional masks."""

    def test_text_only_doubling(self):
        from simpletuner.helpers.models.common import ImageModelFoundation

        gligen = {
            "boxes": torch.ones(1, 2, 4),
            "masks": torch.ones(1, 2),
            "positive_embeddings": torch.randn(1, 2, 768),
        }
        result = ImageModelFoundation._cfg_double_gligen_dict(gligen)
        self.assertEqual(result["boxes"].shape[0], 2)
        self.assertEqual(result["masks"].shape[0], 2)
        self.assertEqual(result["positive_embeddings"].shape[0], 2)
        # Unconditional half (first) should have masks zeroed
        self.assertTrue(torch.all(result["masks"][0] == 0))
        # Conditional half (second) should retain masks
        self.assertTrue(torch.all(result["masks"][1] == 1))

    def test_text_image_doubling(self):
        from simpletuner.helpers.models.common import ImageModelFoundation

        gligen = {
            "boxes": torch.ones(1, 2, 4),
            "masks": torch.ones(1, 2),
            "phrases_masks": torch.ones(1, 2),
            "image_masks": torch.ones(1, 2),
            "phrases_embeddings": torch.randn(1, 2, 768),
            "image_embeddings": torch.randn(1, 2, 768),
        }
        result = ImageModelFoundation._cfg_double_gligen_dict(gligen)
        self.assertEqual(result["boxes"].shape[0], 2)
        # Unconditional half masks zeroed
        self.assertTrue(torch.all(result["masks"][0] == 0))
        self.assertTrue(torch.all(result["phrases_masks"][0] == 0))
        self.assertTrue(torch.all(result["image_masks"][0] == 0))
        # Conditional half masks retained
        self.assertTrue(torch.all(result["masks"][1] == 1))
        self.assertTrue(torch.all(result["phrases_masks"][1] == 1))
        self.assertTrue(torch.all(result["image_masks"][1] == 1))


class TestBuildValidationGroundingPipelineKwargs(unittest.TestCase):
    """Test _build_validation_grounding_pipeline_kwargs with CFG doubling."""

    def _make_batch(self):
        return GroundingBatch(
            boxes=torch.zeros(1, 2, 4),
            validity_mask=torch.ones(1, 2),
            spatial_masks=torch.zeros(1, 2, 8, 8),
            text_embeds=torch.randn(1, 2, 768),
            image_embeds=None,
            text_masks=torch.ones(1, 2),
            image_masks=torch.zeros(1, 2),
            max_entities=2,
        )

    def test_unet_no_cfg(self):
        from simpletuner.helpers.models.common import ModelTypes

        stub = _make_model_stub(text_image_mode=False, model_type=ModelTypes.UNET)
        batch = self._make_batch()
        result = stub._build_validation_grounding_pipeline_kwargs(batch, do_cfg=False)
        self.assertIn("cross_attention_kwargs", result)
        gligen = result["cross_attention_kwargs"]["gligen"]
        self.assertEqual(gligen["boxes"].shape[0], 1)

    def test_unet_with_cfg(self):
        from simpletuner.helpers.models.common import ModelTypes

        stub = _make_model_stub(text_image_mode=False, model_type=ModelTypes.UNET)
        batch = self._make_batch()
        result = stub._build_validation_grounding_pipeline_kwargs(batch, do_cfg=True)
        self.assertIn("cross_attention_kwargs", result)
        gligen = result["cross_attention_kwargs"]["gligen"]
        # Batch should be doubled
        self.assertEqual(gligen["boxes"].shape[0], 2)
        self.assertEqual(gligen["masks"].shape[0], 2)
        # Unconditional half masks zeroed
        self.assertTrue(torch.all(gligen["masks"][0] == 0))
        # Conditional half masks retained
        self.assertTrue(torch.all(gligen["masks"][1] == 1))

    def test_transformer_with_cfg(self):
        from simpletuner.helpers.models.common import ModelTypes

        stub = _make_model_stub(text_image_mode=False, model_type=ModelTypes.TRANSFORMER)
        stub.ATTENTION_KWARG_NAME = "joint_attention_kwargs"
        batch = self._make_batch()
        result = stub._build_validation_grounding_pipeline_kwargs(batch, do_cfg=True)
        self.assertIn("joint_attention_kwargs", result)
        gk = result["joint_attention_kwargs"]["_grounding_kwargs"]
        # Batch should be doubled
        self.assertEqual(gk["boxes"].shape[0], 2)
        self.assertEqual(gk["masks"].shape[0], 2)
        # Unconditional half masks zeroed
        self.assertTrue(torch.all(gk["masks"][0] == 0))


class TestGligenProperty(unittest.TestCase):
    """Test ModelFoundation.gligen property wraps supports_grounding()."""

    def _make_stub(self, **config_attrs):
        from simpletuner.helpers.models.common import ModelFoundation

        stub = MagicMock(spec=ModelFoundation)
        stub.config = MagicMock(**config_attrs)
        # Wire up the real property via the real methods
        stub.supports_grounding = lambda: ModelFoundation.supports_grounding(stub)
        type(stub).gligen = ModelFoundation.gligen

        return stub

    def test_gligen_true_when_grounding_enabled(self):
        model = self._make_stub(max_grounding_entities=4)
        self.assertTrue(model.gligen)

    def test_gligen_false_when_grounding_disabled(self):
        model = self._make_stub(max_grounding_entities=0)
        self.assertFalse(model.gligen)

    def test_gligen_false_when_no_attribute(self):
        from simpletuner.helpers.models.common import ModelFoundation

        stub = MagicMock(spec=ModelFoundation)
        stub.config = MagicMock(spec=[])
        stub.supports_grounding = lambda: ModelFoundation.supports_grounding(stub)
        type(stub).gligen = ModelFoundation.gligen
        self.assertFalse(stub.gligen)


if __name__ == "__main__":
    unittest.main()
