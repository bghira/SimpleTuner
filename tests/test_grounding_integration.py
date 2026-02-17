"""
Integration tests for GLIGEN grounding token injection and forward pass.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.training.grounding.gligen_layers import (
    _extract_block_dims,
    apply_grounding_fuser,
    get_gligen_trainable_parameters,
    inject_gligen_layers,
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


class TestBuildGligenCrossAttentionKwargs(unittest.TestCase):
    """Test _build_gligen_cross_attention_kwargs static method."""

    def test_none_grounding_batch(self):
        from simpletuner.helpers.models.common import ImageModelFoundation

        result = ImageModelFoundation._build_gligen_cross_attention_kwargs(None)
        self.assertIsNone(result)

    def test_text_only_mode(self):
        from simpletuner.helpers.models.common import ImageModelFoundation

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
        result = ImageModelFoundation._build_gligen_cross_attention_kwargs(batch)
        self.assertIn("gligen", result)
        self.assertIn("positive_embeddings", result["gligen"])
        self.assertNotIn("phrases_embeddings", result["gligen"])

    def test_text_image_mode(self):
        from simpletuner.helpers.models.common import ImageModelFoundation

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
        result = ImageModelFoundation._build_gligen_cross_attention_kwargs(batch)
        self.assertIn("gligen", result)
        self.assertIn("phrases_embeddings", result["gligen"])
        self.assertIn("image_embeddings", result["gligen"])
        self.assertNotIn("positive_embeddings", result["gligen"])


class TestBuildGroundingPositionNetKwargs(unittest.TestCase):
    """Test _build_grounding_position_net_kwargs for transformer models."""

    def test_none_returns_none(self):
        from simpletuner.helpers.models.common import ImageModelFoundation

        result = ImageModelFoundation._build_grounding_position_net_kwargs(None)
        self.assertIsNone(result)

    def test_image_text_only(self):
        from simpletuner.helpers.models.common import ImageModelFoundation

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
        result = ImageModelFoundation._build_grounding_position_net_kwargs(batch)
        self.assertIn("boxes", result)
        self.assertIn("masks", result)
        self.assertIn("positive_embeddings", result)
        self.assertNotIn("phrases_embeddings", result)
        self.assertEqual(result["boxes"].shape, (2, 4, 4))

    def test_image_text_image_mode(self):
        from simpletuner.helpers.models.common import ImageModelFoundation

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
        result = ImageModelFoundation._build_grounding_position_net_kwargs(batch)
        self.assertIn("phrases_embeddings", result)
        self.assertIn("image_embeddings", result)
        self.assertNotIn("positive_embeddings", result)

    def test_video_flatten_temporal(self):
        from simpletuner.helpers.models.common import ImageModelFoundation

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
        result = ImageModelFoundation._build_grounding_position_net_kwargs(batch, flatten_temporal=True)
        # (B, T, N, ...) -> (B*T, N, ...)
        self.assertEqual(result["boxes"].shape, (B * T, N, 4))
        self.assertEqual(result["masks"].shape, (B * T, N))
        self.assertEqual(result["positive_embeddings"].shape, (B * T, N, 768))

    def test_video_no_flatten(self):
        from simpletuner.helpers.models.common import ImageModelFoundation

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
        result = ImageModelFoundation._build_grounding_position_net_kwargs(batch, flatten_temporal=False)
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


if __name__ == "__main__":
    unittest.main()
