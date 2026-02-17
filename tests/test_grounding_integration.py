"""
Integration tests for GLIGEN grounding token injection and UNet forward pass.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.training.grounding.gligen_layers import get_gligen_trainable_parameters, inject_gligen_layers
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
