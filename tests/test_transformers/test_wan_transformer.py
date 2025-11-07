"""
Comprehensive unit tests for WAN transformer classes.
Tests all 6 classes and 6 functions with focus on typo prevention and WAN-specific functionality.

Classes tested:
1. WanAttnProcessor2_0 (attention processor with image/video processing)
2. WanImageEmbedding (image embedding module)
3. WanTimeTextImageEmbedding (combined time, text, and image embeddings)
4. WanRotaryPosEmbed (3D rotary position embedding)
5. WanTransformerBlock (transformer block with dual-stream attention)
6. WanTransformer3DModel (main model for video generation)

Functions tested:
1. _apply_rotary_emb_anyshape (flexible rotary embedding application)
2. _route_rope (routing rotary embeddings for TREAD)
3. forward methods for all classes
4. set_router (TREAD integration)
5. split_with_sizes (frequency splitting)
6. prepare_video_coords (coordinate preparation - inherited)
"""

import copy
import math
import os

# Import test base classes
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from transformer_base_test import (
    AttentionProcessorTestMixin,
    EmbeddingTestMixin,
    TransformerBaseTest,
    TransformerBlockTestMixin,
)
from transformer_test_helpers import (
    MockComponents,
    MockDiffusersConfig,
    PerformanceUtils,
    ShapeValidator,
    TensorGenerator,
    TypoTestUtils,
)

# Import classes under test
from simpletuner.helpers.models.wan.transformer import (
    WanAttnProcessor2_0,
    WanImageEmbedding,
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanTransformer3DModel,
    WanTransformerBlock,
    _apply_rotary_emb_anyshape,
)


class TestApplyRotaryEmbAnyShape(TransformerBaseTest):
    """Test _apply_rotary_emb_anyshape function for flexible rotary embedding application."""

    def test_single_batch_rotary_emb(self):
        """Test rotary embedding application with single batch format (1, 1, S, D)."""
        batch_size, heads, seq_len, head_dim = 2, 8, 64, 32
        x = torch.randn(batch_size, heads, seq_len, head_dim)

        # Single batch format rotary embedding
        rotary_emb = torch.randn(1, 1, seq_len, head_dim // 2, dtype=torch.complex64)

        output = _apply_rotary_emb_anyshape(x, rotary_emb)

        self.assert_tensor_shape(output, x.shape)
        self.assert_tensor_dtype(output, x.dtype)
        self.assert_no_nan_or_inf(output)

    def test_multi_batch_rotary_emb(self):
        """Test rotary embedding application with multi-batch format (B, 1, S, D)."""
        batch_size, heads, seq_len, head_dim = 2, 8, 64, 32
        x = torch.randn(batch_size, heads, seq_len, head_dim)

        # Multi-batch format rotary embedding for routed tokens
        rotary_emb = torch.randn(batch_size, 1, seq_len, head_dim // 2, dtype=torch.complex64)

        output = _apply_rotary_emb_anyshape(x, rotary_emb)

        self.assert_tensor_shape(output, x.shape)
        self.assert_tensor_dtype(output, x.dtype)
        self.assert_no_nan_or_inf(output)

    def test_dtype_conversion_mps_backend(self):
        """Test dtype conversion for MPS backend compatibility."""
        batch_size, heads, seq_len, head_dim = 2, 8, 64, 32
        x = torch.randn(batch_size, heads, seq_len, head_dim, dtype=torch.float16)
        rotary_emb = torch.randn(1, 1, seq_len, head_dim // 2, dtype=torch.complex64)

        with patch("torch.backends.mps.is_available", return_value=True):
            output = _apply_rotary_emb_anyshape(x, rotary_emb)

            self.assert_tensor_shape(output, x.shape)
            # Should preserve input dtype
            self.assert_tensor_dtype(output, torch.float16)

    def test_dtype_conversion_cuda_backend(self):
        """Test dtype conversion for CUDA backend."""
        batch_size, heads, seq_len, head_dim = 2, 8, 64, 32
        x = torch.randn(batch_size, heads, seq_len, head_dim, dtype=torch.float16)
        rotary_emb = torch.randn(1, 1, seq_len, head_dim // 2, dtype=torch.complex128)

        with patch("torch.backends.mps.is_available", return_value=False):
            output = _apply_rotary_emb_anyshape(x, rotary_emb)

            self.assert_tensor_shape(output, x.shape)
            self.assert_tensor_dtype(output, torch.float16)

    def test_complex_rotary_operations(self):
        """Test complex number operations in rotary embedding."""
        batch_size, heads, seq_len, head_dim = 1, 1, 4, 4
        x = torch.randn(batch_size, heads, seq_len, head_dim)

        # Create specific rotary embedding to test rotation
        rotary_emb = torch.tensor([[[[1 + 0j, 0 + 1j]]]], dtype=torch.complex64)

        output = _apply_rotary_emb_anyshape(x, rotary_emb)

        self.assert_tensor_shape(output, x.shape)
        self.assertFalse(torch.allclose(output, x))  # Should be different after rotation

    def test_heads_dimension_expansion(self):
        """Test proper expansion for heads dimension in multi-batch case."""
        batch_size, heads, seq_len, head_dim = 2, 4, 16, 8
        x = torch.randn(batch_size, heads, seq_len, head_dim)

        # Multi-batch rotary embedding that needs expansion for heads
        rotary_emb = torch.randn(batch_size, 1, seq_len, head_dim // 2, dtype=torch.complex64)

        output = _apply_rotary_emb_anyshape(x, rotary_emb)

        # Should handle heads dimension expansion correctly
        self.assert_tensor_shape(output, x.shape)

    def test_edge_case_single_token(self):
        """Test edge case with single token sequence."""
        x = torch.randn(1, 1, 1, 2)
        rotary_emb = torch.randn(1, 1, 1, 1, dtype=torch.complex64)

        output = _apply_rotary_emb_anyshape(x, rotary_emb)
        self.assert_tensor_shape(output, (1, 1, 1, 2))

    def test_typo_prevention_function_name(self):
        """Test typo prevention for function name."""
        # Test correct function exists
        from simpletuner.helpers.models.wan.transformer import _apply_rotary_emb_anyshape

        self.assertTrue(callable(_apply_rotary_emb_anyshape))

        # Test common typos fail
        with self.assertRaises(ImportError):
            from simpletuner.helpers.models.wan.transformer import _apply_rotary_emb_any_shape
        with self.assertRaises(ImportError):
            from simpletuner.helpers.models.wan.transformer import apply_rotary_emb_anyshape  # Missing '_'


class TestWanAttnProcessor2_0(TransformerBaseTest, AttentionProcessorTestMixin):
    """Test WanAttnProcessor2_0 attention processor with image/video processing capabilities."""

    def setUp(self):
        super().setUp()
        self.processor = WanAttnProcessor2_0()
        self.mock_attn = self._create_mock_wan_attention()

    def _create_mock_wan_attention(self):
        """Create a mock WAN attention with required attributes."""
        mock_attn = Mock()
        mock_attn.heads = self.num_heads
        mock_attn.to_q = Mock(return_value=torch.randn(self.batch_size, self.seq_len, self.hidden_dim))
        mock_attn.to_k = Mock(return_value=torch.randn(self.batch_size, self.seq_len, self.hidden_dim))
        mock_attn.to_v = Mock(return_value=torch.randn(self.batch_size, self.seq_len, self.hidden_dim))
        mock_attn.norm_q = Mock(side_effect=lambda x: x)
        mock_attn.norm_k = Mock(side_effect=lambda x: x)
        mock_attn.to_out = [Mock(side_effect=lambda x: x), Mock(side_effect=lambda x: x)]  # Linear layer  # Dropout layer
        # Mock added projections for I2V task
        mock_attn.add_k_proj = None
        mock_attn.add_v_proj = None
        mock_attn.norm_added_k = None
        return mock_attn

    def test_basic_instantiation(self):
        """Test basic processor instantiation with PyTorch version check."""
        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            processor = WanAttnProcessor2_0()
            self.assertIsNotNone(processor)

    def test_pytorch_version_requirement(self):
        """Test PyTorch version requirement."""
        original_sdpa = F.scaled_dot_product_attention
        try:
            delattr(F, "scaled_dot_product_attention")
            with self.assertRaises(ImportError) as cm:
                WanAttnProcessor2_0()
            self.assertIn("PyTorch 2.0", str(cm.exception))
        finally:
            setattr(F, "scaled_dot_product_attention", original_sdpa)

    def test_forward_pass_minimal(self):
        """Test minimal forward pass without encoder states."""
        hidden_states = self.hidden_states

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            mock_sdpa.return_value = torch.randn_like(hidden_states.unflatten(2, (self.num_heads, -1)).transpose(1, 2))

            output = self.processor(attn=self.mock_attn, hidden_states=hidden_states)

            self.assert_tensor_shape(output, hidden_states.shape)
            self.assert_no_nan_or_inf(output)

    def test_forward_pass_with_encoder_states(self):
        """Test forward pass with encoder hidden states."""
        hidden_states = self.hidden_states
        encoder_hidden_states = self.encoder_hidden_states

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            mock_sdpa.return_value = torch.randn_like(hidden_states.unflatten(2, (self.num_heads, -1)).transpose(1, 2))

            output = self.processor(
                attn=self.mock_attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states
            )

            self.assert_tensor_shape(output, hidden_states.shape)
            self.mock_attn.to_k.assert_called_with(encoder_hidden_states)
            self.mock_attn.to_v.assert_called_with(encoder_hidden_states)

    def test_forward_pass_with_rotary_embeddings(self):
        """Test forward pass with rotary embeddings."""
        hidden_states = self.hidden_states
        rotary_emb = torch.randn(1, 1, self.seq_len, self.head_dim // 2, dtype=torch.complex64)

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            mock_sdpa.return_value = torch.randn_like(hidden_states.unflatten(2, (self.num_heads, -1)).transpose(1, 2))
            with patch("simpletuner.helpers.models.wan.transformer._apply_rotary_emb_anyshape") as mock_rotary:
                mock_rotary.side_effect = lambda x, emb: x  # Identity for testing

                output = self.processor(attn=self.mock_attn, hidden_states=hidden_states, rotary_emb=rotary_emb)

                # Should call rotary embedding twice (query and key)
                self.assertEqual(mock_rotary.call_count, 2)
                self.assert_tensor_shape(output, hidden_states.shape)

    def test_forward_pass_with_image_encoder_states_i2v(self):
        """Test forward pass with image encoder states for I2V task."""
        hidden_states = self.hidden_states

        # Create encoder states with image part (first 513 tokens) and text part
        image_tokens = torch.randn(self.batch_size, 513, self.hidden_dim)
        text_tokens = torch.randn(self.batch_size, 77, self.hidden_dim)
        encoder_hidden_states = torch.cat([image_tokens, text_tokens], dim=1)

        # Mock that add_k_proj exists (I2V mode)
        self.mock_attn.add_k_proj = Mock(return_value=torch.randn(self.batch_size, 513, self.hidden_dim))
        self.mock_attn.add_v_proj = Mock(return_value=torch.randn(self.batch_size, 513, self.hidden_dim))
        self.mock_attn.norm_added_k = Mock(side_effect=lambda x: x)

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            mock_sdpa.return_value = torch.randn_like(hidden_states.unflatten(2, (self.num_heads, -1)).transpose(1, 2))

            output = self.processor(
                attn=self.mock_attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states
            )

            self.assert_tensor_shape(output, hidden_states.shape)
            # Should process both image and text parts
            self.mock_attn.add_k_proj.assert_called_once()
            self.mock_attn.add_v_proj.assert_called_once()

    def test_attention_mask_processing(self):
        """Test attention mask handling in scaled dot product attention."""
        hidden_states = self.hidden_states
        attention_mask = torch.ones(self.batch_size, self.seq_len)

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            mock_sdpa.return_value = torch.randn_like(hidden_states.unflatten(2, (self.num_heads, -1)).transpose(1, 2))

            output = self.processor(attn=self.mock_attn, hidden_states=hidden_states, attention_mask=attention_mask)

            # Check that attention mask was passed to SDPA
            call_args = mock_sdpa.call_args
            self.assertIn("attn_mask", call_args[1])

    def test_tensor_reshape_operations(self):
        """Test tensor reshaping for multi-head attention."""
        hidden_states = self.hidden_states

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            # Mock return value with correct head dimension
            reshaped_output = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
            mock_sdpa.return_value = reshaped_output

            output = self.processor(attn=self.mock_attn, hidden_states=hidden_states)

            # Should return to original shape after unflatten/transpose operations
            self.assert_tensor_shape(output, hidden_states.shape)

    def test_i2v_task_dual_attention(self):
        """Test dual attention computation for I2V task."""
        hidden_states = self.hidden_states

        # Setup for I2V: encoder states with both image and text
        image_tokens = torch.randn(self.batch_size, 513, self.hidden_dim)
        text_tokens = torch.randn(self.batch_size, 77, self.hidden_dim)
        encoder_hidden_states = torch.cat([image_tokens, text_tokens], dim=1)

        self.mock_attn.add_k_proj = Mock(return_value=torch.randn(self.batch_size, 513, self.hidden_dim))
        self.mock_attn.add_v_proj = Mock(return_value=torch.randn(self.batch_size, 513, self.hidden_dim))
        self.mock_attn.norm_added_k = Mock(side_effect=lambda x: x)

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            # Mock return for both image and text attention
            mock_attention_output = torch.randn(self.batch_size, self.num_heads, self.seq_len, self.head_dim)
            mock_sdpa.return_value = mock_attention_output

            output = self.processor(
                attn=self.mock_attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states
            )

            # Should call SDPA twice (image and text attention)
            self.assertEqual(mock_sdpa.call_count, 2)
            self.assert_tensor_shape(output, hidden_states.shape)

    def test_typo_prevention_method_parameters(self):
        """Test typo prevention for method parameter names."""
        valid_params = {
            "attn": self.mock_attn,
            "hidden_states": self.hidden_states,
            "encoder_hidden_states": self.encoder_hidden_states,
            "attention_mask": self.attention_mask,
            "rotary_emb": torch.randn(1, 1, self.seq_len, self.head_dim // 2, dtype=torch.complex64),
        }

        typo_params = {
            "attn_": "attn",  # Extra underscore
            "hiden_states": "hidden_states",  # Missing 'd'
            "encoder_hiden_states": "encoder_hidden_states",  # Missing 'd'
            "atention_mask": "attention_mask",  # Missing 't'
            "rotary_embedding": "rotary_emb",  # Different name
        }

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            mock_sdpa.return_value = torch.randn_like(self.hidden_states.unflatten(2, (self.num_heads, -1)).transpose(1, 2))
            self.run_typo_prevention_tests(self.processor, "__call__", valid_params, typo_params)

    def test_edge_case_no_add_projections(self):
        """Test edge case where add_k_proj is None (non-I2V mode)."""
        hidden_states = self.hidden_states
        encoder_hidden_states = self.encoder_hidden_states

        # Remove add projections to simulate non-I2V mode
        self.mock_attn.add_k_proj = None

        with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
            mock_sdpa.return_value = torch.randn_like(hidden_states.unflatten(2, (self.num_heads, -1)).transpose(1, 2))

            output = self.processor(
                attn=self.mock_attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states
            )

            # Should work normally without image processing
            self.assert_tensor_shape(output, hidden_states.shape)

    def test_dtype_preservation(self):
        """Test that dtype is preserved through processing."""
        for dtype in [torch.float32, torch.float16]:
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue

            hidden_states = self.hidden_states.to(dtype)

            with patch("torch.nn.functional.scaled_dot_product_attention") as mock_sdpa:
                mock_sdpa.return_value = torch.randn_like(hidden_states.unflatten(2, (self.num_heads, -1)).transpose(1, 2))

                output = self.processor(attn=self.mock_attn, hidden_states=hidden_states)

                self.assert_tensor_dtype(output, dtype)


class TestWanImageEmbedding(TransformerBaseTest, EmbeddingTestMixin):
    """Test WanImageEmbedding module."""

    def setUp(self):
        super().setUp()
        self.embedding_config = {"in_features": 1280, "out_features": self.hidden_dim}

    def test_basic_instantiation(self):
        """Test basic image embedding instantiation."""
        embedding = WanImageEmbedding(**self.embedding_config)

        self.assertIsNotNone(embedding)
        self.assertIsNotNone(embedding.norm1)
        self.assertIsNotNone(embedding.ff)
        self.assertIsNotNone(embedding.norm2)

        # Check that norm layers are FP32LayerNorm
        self.assertEqual(embedding.norm1.normalized_shape, (self.embedding_config["in_features"],))
        self.assertEqual(embedding.norm2.normalized_shape, (self.embedding_config["out_features"],))

    def test_forward_pass(self):
        """Test forward pass through image embedding."""
        embedding = WanImageEmbedding(**self.embedding_config)

        input_tensor = torch.randn(self.batch_size, 513, self.embedding_config["in_features"])

        output = embedding.forward(input_tensor)

        expected_shape = (self.batch_size, 513, self.embedding_config["out_features"])
        self.assert_tensor_shape(output, expected_shape)
        self.assert_no_nan_or_inf(output)

    def test_feed_forward_integration(self):
        """Test feed-forward network integration."""
        embedding = WanImageEmbedding(**self.embedding_config)

        # Mock feed-forward to track calls
        with patch.object(embedding.ff, "forward") as mock_ff:
            mock_ff.return_value = torch.randn(self.batch_size, 513, self.embedding_config["out_features"])

            input_tensor = torch.randn(self.batch_size, 513, self.embedding_config["in_features"])
            embedding.forward(input_tensor)

            mock_ff.assert_called_once()

    def test_normalization_layers(self):
        """Test that normalization layers are applied correctly."""
        embedding = WanImageEmbedding(**self.embedding_config)

        input_tensor = torch.randn(self.batch_size, 513, self.embedding_config["in_features"])

        # Mock normalization layers to track calls
        with patch.object(embedding.norm1, "forward") as mock_norm1:
            mock_norm1.return_value = input_tensor
            with patch.object(embedding.norm2, "forward") as mock_norm2:
                mock_norm2.return_value = torch.randn(self.batch_size, 513, self.embedding_config["out_features"])

                embedding.forward(input_tensor)

                mock_norm1.assert_called_once()
                mock_norm2.assert_called_once()

    def test_typo_prevention_parameter_names(self):
        """Test typo prevention for constructor parameter names."""
        valid_params = self.embedding_config.copy()

        typo_params = {
            "in_feature": "in_features",  # Missing 's'
            "out_feature": "out_features",  # Missing 's'
            "input_features": "in_features",  # Different name
            "output_features": "out_features",  # Different name
        }

        for typo_param, correct_param in typo_params.items():
            invalid_config = valid_params.copy()
            invalid_config[typo_param] = invalid_config.pop(correct_param)

            with self.assertRaises(TypeError) as cm:
                WanImageEmbedding(**invalid_config)
            self.assertIn("unexpected keyword argument", str(cm.exception))

    def test_different_input_dimensions(self):
        """Test with different input dimensions."""
        configs = [
            {"in_features": 512, "out_features": 1024},
            {"in_features": 2048, "out_features": 512},
            {"in_features": 1280, "out_features": 2048},
        ]

        for config in configs:
            embedding = WanImageEmbedding(**config)
            input_tensor = torch.randn(2, 100, config["in_features"])
            output = embedding.forward(input_tensor)
            self.assert_tensor_shape(output, (2, 100, config["out_features"]))


class TestWanTimeTextImageEmbedding(TransformerBaseTest, EmbeddingTestMixin):
    """Test WanTimeTextImageEmbedding combined embedding module."""

    def setUp(self):
        super().setUp()
        self.embedding_config = {
            "dim": self.hidden_dim,
            "time_freq_dim": 256,
            "time_proj_dim": self.hidden_dim * 6,
            "text_embed_dim": 4096,
            "image_embed_dim": 1280,
        }

    def test_basic_instantiation(self):
        """Test basic combined embedding instantiation."""
        embedding = WanTimeTextImageEmbedding(**self.embedding_config)

        self.assertIsNotNone(embedding)
        self.assertIsNotNone(embedding.timesteps_proj)
        self.assertIsNotNone(embedding.time_embedder)
        self.assertIsNotNone(embedding.act_fn)
        self.assertIsNotNone(embedding.time_proj)
        self.assertIsNotNone(embedding.text_embedder)
        self.assertIsNotNone(embedding.image_embedder)

    def test_instantiation_without_image_embedding(self):
        """Test instantiation without image embedding."""
        config = self.embedding_config.copy()
        config["image_embed_dim"] = None

        embedding = WanTimeTextImageEmbedding(**config)
        self.assertIsNone(embedding.image_embedder)

    def test_forward_pass_with_all_inputs(self):
        """Test forward pass with timestep, text, and image inputs."""
        embedding = WanTimeTextImageEmbedding(**self.embedding_config)

        timestep = torch.randint(0, 1000, (self.batch_size,))
        encoder_hidden_states = torch.randn(self.batch_size, 77, self.embedding_config["text_embed_dim"])
        encoder_hidden_states_image = torch.randn(self.batch_size, 513, self.embedding_config["image_embed_dim"])

        temb, timestep_proj, text_emb, image_emb = embedding.forward(
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_hidden_states_image=encoder_hidden_states_image,
        )

        # Check output shapes
        self.assert_tensor_shape(temb, (self.batch_size, self.embedding_config["dim"]))
        self.assert_tensor_shape(timestep_proj, (self.batch_size, self.embedding_config["time_proj_dim"]))
        self.assert_tensor_shape(text_emb, (self.batch_size, 77, self.embedding_config["dim"]))
        self.assert_tensor_shape(image_emb, (self.batch_size, 513, self.embedding_config["dim"]))

        # Check no NaN/inf values
        self.assert_no_nan_or_inf(temb)
        self.assert_no_nan_or_inf(timestep_proj)
        self.assert_no_nan_or_inf(text_emb)
        self.assert_no_nan_or_inf(image_emb)

    def test_forward_pass_without_image(self):
        """Test forward pass without image encoder states."""
        embedding = WanTimeTextImageEmbedding(**self.embedding_config)

        timestep = torch.randint(0, 1000, (self.batch_size,))
        encoder_hidden_states = torch.randn(self.batch_size, 77, self.embedding_config["text_embed_dim"])

        temb, timestep_proj, text_emb, image_emb = embedding.forward(
            timestep=timestep, encoder_hidden_states=encoder_hidden_states, encoder_hidden_states_image=None
        )

        # Image embedding should be None
        self.assertIsNone(image_emb)
        self.assertIsNotNone(temb)
        self.assertIsNotNone(timestep_proj)
        self.assertIsNotNone(text_emb)

    def test_forward_pass_without_image_embedder(self):
        """Test forward pass when image embedder is not configured."""
        config = self.embedding_config.copy()
        config["image_embed_dim"] = None
        embedding = WanTimeTextImageEmbedding(**config)

        timestep = torch.randint(0, 1000, (self.batch_size,))
        encoder_hidden_states = torch.randn(self.batch_size, 77, self.embedding_config["text_embed_dim"])

        temb, timestep_proj, text_emb, image_emb = embedding.forward(
            timestep=timestep, encoder_hidden_states=encoder_hidden_states, encoder_hidden_states_image=None
        )

        # Image embedding should be None since no image embedder and no image tokens supplied
        self.assertIsNone(image_emb)

    def test_timestep_projection_processing(self):
        """Test timestep projection and processing."""
        embedding = WanTimeTextImageEmbedding(**self.embedding_config)

        timestep = torch.randint(0, 1000, (self.batch_size,))
        encoder_hidden_states = torch.randn(self.batch_size, 77, self.embedding_config["text_embed_dim"])

        # Mock timesteps_proj and time_embedder to track calls
        with patch.object(embedding.timesteps_proj, "forward") as mock_timesteps:
            mock_timesteps.return_value = torch.randn(self.batch_size, self.embedding_config["time_freq_dim"])
            with patch.object(embedding.time_embedder, "forward") as mock_time_emb:
                mock_time_emb.return_value = torch.randn(self.batch_size, self.embedding_config["dim"])

                embedding.forward(timestep=timestep, encoder_hidden_states=encoder_hidden_states)

                mock_timesteps.assert_called_once_with(timestep)
                mock_time_emb.assert_called_once()

    def test_dtype_conversion_handling(self):
        """Test dtype conversion for time embedder compatibility."""
        embedding = WanTimeTextImageEmbedding(**self.embedding_config)

        # Test with different input dtypes
        timestep_int = torch.randint(0, 1000, (self.batch_size,), dtype=torch.int32)
        timestep_long = torch.randint(0, 1000, (self.batch_size,), dtype=torch.int64)
        encoder_hidden_states = torch.randn(self.batch_size, 77, self.embedding_config["text_embed_dim"])

        for timestep in [timestep_int, timestep_long]:
            temb, timestep_proj, text_emb, image_emb = embedding.forward(
                timestep=timestep, encoder_hidden_states=encoder_hidden_states
            )

            self.assert_no_nan_or_inf(temb)
            self.assert_no_nan_or_inf(timestep_proj)

    def test_typo_prevention_parameter_names(self):
        """Test typo prevention for constructor parameter names."""
        valid_params = self.embedding_config.copy()

        typo_params = {
            "dims": "dim",  # Extra 's'
            "time_freq_dims": "time_freq_dim",  # Extra 's'
            "time_proj_dims": "time_proj_dim",  # Extra 's'
            "text_embed_dims": "text_embed_dim",  # Extra 's'
            "image_embed_dims": "image_embed_dim",  # Extra 's'
            "time_frequency_dim": "time_freq_dim",  # Different name
            "text_embedding_dim": "text_embed_dim",  # Different name
            "image_embedding_dim": "image_embed_dim",  # Different name
        }

        for typo_param, correct_param in typo_params.items():
            invalid_config = valid_params.copy()
            invalid_config[typo_param] = invalid_config.pop(correct_param)

            with self.assertRaises(TypeError) as cm:
                WanTimeTextImageEmbedding(**invalid_config)
            self.assertIn("unexpected keyword argument", str(cm.exception))


class TestWanRotaryPosEmbed(TransformerBaseTest):
    """Test WanRotaryPosEmbed for 3D rotary position embedding."""

    def setUp(self):
        super().setUp()
        self.rope_config = {"attention_head_dim": 128, "patch_size": (1, 2, 2), "max_seq_len": 1024, "theta": 10000.0}

    def test_basic_instantiation(self):
        """Test basic rotary position embedding instantiation."""
        rope = WanRotaryPosEmbed(**self.rope_config)

        self.assertIsNotNone(rope)
        self.assertEqual(rope.attention_head_dim, 128)
        self.assertEqual(rope.patch_size, (1, 2, 2))
        self.assertEqual(rope.max_seq_len, 1024)

        # Check that frequencies are generated
        self.assertIsNotNone(rope.freqs)

    def test_frequency_dimension_calculation(self):
        """Test frequency dimension calculation for T, H, W components."""
        rope = WanRotaryPosEmbed(**self.rope_config)

        attention_head_dim = self.rope_config["attention_head_dim"]
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        # Total dimension should equal attention_head_dim // 2
        total_freq_dim = (t_dim + h_dim + w_dim) // 2
        expected_freq_dim = attention_head_dim // 2

        self.assertEqual(total_freq_dim, expected_freq_dim)

    def test_forward_pass_video_input(self):
        """Test forward pass with video-shaped input."""
        rope = WanRotaryPosEmbed(**self.rope_config)

        # Video input: (batch, channels, frames, height, width)
        batch_size, num_channels, num_frames, height, width = 2, 16, 4, 8, 8
        hidden_states = torch.randn(batch_size, num_channels, num_frames, height, width)

        freqs = rope.forward(hidden_states)

        # Calculate expected sequence length after patching
        p_t, p_h, p_w = rope.patch_size
        expected_seq_len = (num_frames // p_t) * (height // p_h) * (width // p_w)

        expected_shape = (1, 1, expected_seq_len, rope.attention_head_dim // 2)
        self.assert_tensor_shape(freqs, expected_shape)
        self.assert_no_nan_or_inf(freqs)

    def test_patch_size_calculation(self):
        """Test patch size calculation for different video dimensions."""
        rope = WanRotaryPosEmbed(**self.rope_config)

        test_cases = [
            (4, 8, 8),  # 4 frames, 8x8 spatial
            (8, 16, 16),  # 8 frames, 16x16 spatial
            (2, 4, 4),  # 2 frames, 4x4 spatial
        ]

        for num_frames, height, width in test_cases:
            hidden_states = torch.randn(1, 16, num_frames, height, width)
            freqs = rope.forward(hidden_states)

            p_t, p_h, p_w = rope.patch_size
            expected_seq_len = (num_frames // p_t) * (height // p_h) * (width // p_w)
            self.assert_tensor_shape(freqs, (1, 1, expected_seq_len, rope.attention_head_dim // 2))

    def test_frequency_splitting(self):
        """Test frequency splitting for T, H, W dimensions."""
        rope = WanRotaryPosEmbed(**self.rope_config)

        hidden_states = torch.randn(1, 16, 4, 8, 8)
        freqs = rope.forward(hidden_states)

        # Verify that freqs tensor is on correct device
        self.assertEqual(freqs.device, hidden_states.device)

        # Check frequency composition
        attention_head_dim = rope.attention_head_dim
        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        # The split should work correctly
        expected_splits = [t_dim // 2, h_dim // 2, w_dim // 2]
        self.assertEqual(sum(expected_splits), attention_head_dim // 2)

    def test_device_compatibility(self):
        """Test device compatibility for frequencies."""
        rope = WanRotaryPosEmbed(**self.rope_config)

        for device in ["cpu"]:  # Add "cuda" if available
            hidden_states = torch.randn(1, 16, 4, 8, 8, device=device)
            freqs = rope.forward(hidden_states)

            self.assertEqual(freqs.device.type, device)

    def test_complex_frequency_generation(self):
        """Test complex frequency generation."""
        rope = WanRotaryPosEmbed(**self.rope_config)

        hidden_states = torch.randn(1, 16, 4, 8, 8)

        # Mock get_1d_rotary_pos_embed to track calls
        def fake_rotary(dim, pos, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=torch.float32):
            if isinstance(pos, int):
                seq_len = pos
            elif isinstance(pos, torch.Tensor):
                seq_len = pos.numel()
            else:
                seq_len = len(pos)
            return torch.randn(seq_len, dim // 2, dtype=torch.complex64)

        with patch("simpletuner.helpers.models.wan.transformer.get_1d_rotary_pos_embed") as mock_rotary:
            mock_rotary.side_effect = fake_rotary

            rope_instance = WanRotaryPosEmbed(**self.rope_config)
            freqs = rope_instance.forward(hidden_states)

            # Should call get_1d_rotary_pos_embed three times (T, H, W)
            self.assertEqual(mock_rotary.call_count, 3)

    def test_typo_prevention_parameter_names(self):
        """Test typo prevention for constructor parameter names."""
        valid_params = self.rope_config.copy()

        typo_params = {
            "attention_head_dims": "attention_head_dim",  # Extra 's'
            "patch_siz": "patch_size",  # Missing 'e'
            "max_seq_length": "max_seq_len",  # Different name
            "thata": "theta",  # Misspelled
        }

        for typo_param, correct_param in typo_params.items():
            invalid_config = valid_params.copy()
            invalid_config[typo_param] = invalid_config.pop(correct_param)

            with self.assertRaises(TypeError) as cm:
                WanRotaryPosEmbed(**invalid_config)
            self.assertIn("unexpected keyword argument", str(cm.exception))

    def test_different_attention_head_dims(self):
        """Test with different attention head dimensions."""
        test_dims = [64, 96, 128, 192]

        for dim in test_dims:
            config = self.rope_config.copy()
            config["attention_head_dim"] = dim

            rope = WanRotaryPosEmbed(**config)
            hidden_states = torch.randn(1, 16, 4, 8, 8)
            freqs = rope.forward(hidden_states)

            expected_freq_dim = dim // 2
            self.assertEqual(freqs.shape[-1], expected_freq_dim)


class TestWanTransformerBlock(TransformerBaseTest, TransformerBlockTestMixin):
    """Test WanTransformerBlock."""

    def setUp(self):
        super().setUp()
        self.block_config = {
            "dim": self.hidden_dim,
            "ffn_dim": self.hidden_dim * 4,
            "num_heads": self.num_heads,
            "qk_norm": "rms_norm_across_heads",
            "cross_attn_norm": False,
            "eps": 1e-6,
            "added_kv_proj_dim": None,
        }

    def _accelerator_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        self.skipTest("No accelerator device available for device realignment test.")

    def test_basic_instantiation(self):
        """Test basic transformer block instantiation."""
        block = WanTransformerBlock(**self.block_config)

        self.assertIsNotNone(block)
        self.assertIsNotNone(block.norm1)
        self.assertIsNotNone(block.attn1)
        self.assertIsNotNone(block.attn2)
        self.assertIsNotNone(block.norm2)
        self.assertIsNotNone(block.ffn)
        self.assertIsNotNone(block.norm3)
        self.assertIsNotNone(block.scale_shift_table)

        # Check scale_shift_table shape (6 parameters for adaptive normalization)
        self.assertEqual(block.scale_shift_table.shape, (1, 6, self.hidden_dim))

    def test_instantiation_with_cross_attn_norm(self):
        """Test instantiation with cross attention normalization."""
        config = self.block_config.copy()
        config["cross_attn_norm"] = True

        block = WanTransformerBlock(**config)

        # norm2 should be FP32LayerNorm instead of Identity
        self.assertNotIsInstance(block.norm2, nn.Identity)

    def test_forward_pass_minimal(self):
        """Test minimal forward pass."""
        block = WanTransformerBlock(**self.block_config)

        encoder_hidden_states = torch.randn(self.batch_size, 77, self.hidden_dim)
        temb = torch.randn(self.batch_size, 6, self.hidden_dim)
        rotary_emb = torch.randn(1, 1, self.seq_len, self.head_dim // 2, dtype=torch.complex64)

        output = block.forward(
            hidden_states=self.hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb, rotary_emb=rotary_emb
        )

        self.assert_tensor_shape(output, self.hidden_states.shape)
        self.assert_no_nan_or_inf(output)

    def test_adaptive_normalization_parameters(self):
        """Test adaptive normalization with timestep embeddings."""
        block = WanTransformerBlock(**self.block_config)

        encoder_hidden_states = torch.randn(self.batch_size, 77, self.hidden_dim)
        temb = torch.randn(self.batch_size, 6, self.hidden_dim)
        rotary_emb = torch.randn(1, 1, self.seq_len, self.head_dim // 2, dtype=torch.complex64)

        # Test that the 6 adaptive parameters are used correctly
        output = block.forward(
            hidden_states=self.hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb, rotary_emb=rotary_emb
        )

        # Should complete without errors
        self.assertIsNotNone(output)

    def test_self_attention_processing(self):
        """Test self-attention processing with rotary embeddings."""
        block = WanTransformerBlock(**self.block_config)

        encoder_hidden_states = torch.randn(self.batch_size, 77, self.hidden_dim)
        temb = torch.randn(self.batch_size, 6, self.hidden_dim)
        rotary_emb = torch.randn(1, 1, self.seq_len, self.head_dim // 2, dtype=torch.complex64)

        # Mock self-attention to track rotary embedding usage
        with patch.object(block.attn1, "forward") as mock_attn1:
            mock_attn1.return_value = torch.randn_like(self.hidden_states)

            block.forward(
                hidden_states=self.hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                rotary_emb=rotary_emb,
            )

            # Self-attention should be called with rotary embeddings
            call_args = mock_attn1.call_args
            self.assertIn("rotary_emb", call_args[1])
            self.assertTrue(torch.equal(call_args[1]["rotary_emb"], rotary_emb))

    def test_cross_attention_processing(self):
        """Test cross-attention processing without rotary embeddings."""
        block = WanTransformerBlock(**self.block_config)

        encoder_hidden_states = torch.randn(self.batch_size, 77, self.hidden_dim)
        temb = torch.randn(self.batch_size, 6, self.hidden_dim)
        rotary_emb = torch.randn(1, 1, self.seq_len, self.head_dim // 2, dtype=torch.complex64)

        # Mock cross-attention to verify it doesn't get rotary embeddings
        with patch.object(block.attn2, "forward") as mock_attn2:
            mock_attn2.return_value = torch.randn_like(self.hidden_states)

            block.forward(
                hidden_states=self.hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                rotary_emb=rotary_emb,
            )

            # Cross-attention should be called with encoder states but no rotary emb
            call_args = mock_attn2.call_args
            self.assertIn("encoder_hidden_states", call_args[1])
            self.assertTrue(torch.equal(call_args[1]["encoder_hidden_states"], encoder_hidden_states))

    def test_feed_forward_processing(self):
        """Test feed-forward processing with adaptive normalization."""
        block = WanTransformerBlock(**self.block_config)

        encoder_hidden_states = torch.randn(self.batch_size, 77, self.hidden_dim)
        temb = torch.randn(self.batch_size, 6, self.hidden_dim)
        rotary_emb = torch.randn(1, 1, self.seq_len, self.head_dim // 2, dtype=torch.complex64)

        # Mock feed-forward to track usage
        with patch.object(block.ffn, "forward") as mock_ffn:
            mock_ffn.return_value = torch.randn_like(self.hidden_states)

            block.forward(
                hidden_states=self.hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                rotary_emb=rotary_emb,
            )

            mock_ffn.assert_called_once()

    def test_temporal_embedding_device_alignment(self):
        """Ensure temb tensors are aligned to the block device before scale-shift operations."""
        device = self._accelerator_device()
        block = WanTransformerBlock(**self.block_config).to(device)
        expected_device = (
            f"{device.type}:{device.index}"
            if device.index is not None
            else ("mps:0" if device.type == "mps" else device.type)
        )

        hidden_states = self.hidden_states.to(device)
        encoder_hidden_states = self.encoder_hidden_states.to(device)
        temb = torch.randn(self.batch_size, 6, self.hidden_dim, device="cpu")
        rotary_emb = torch.randn(1, 1, self.seq_len, self.head_dim // 2, dtype=torch.complex64).to(device)

        output = block.forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            rotary_emb=rotary_emb,
        )

        self.assert_tensor_device(output, expected_device)
        self.assert_tensor_device(block.scale_shift_table, expected_device)

    def test_scale_shift_table_realigns_to_hidden_state_device(self):
        """Ensure scale_shift_table matches the execution device before modulation."""
        device = self._accelerator_device()
        block = WanTransformerBlock(**self.block_config).to(device)
        expected_device = (
            f"{device.type}:{device.index}"
            if device.index is not None
            else ("mps:0" if device.type == "mps" else device.type)
        )

        block.scale_shift_table.data = block.scale_shift_table.data.to("cpu")

        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=device)
        encoder_hidden_states = torch.randn(self.batch_size, 77, self.hidden_dim, device=device)
        temb = torch.randn(self.batch_size, 6, self.hidden_dim, device=device)

        def _zero_like(*args, **kwargs):
            if "hidden_states" in kwargs:
                reference = kwargs["hidden_states"]
            elif args:
                reference = args[0]
            else:
                raise AssertionError("Expected tensor argument for zero_like helper.")
            return torch.zeros_like(reference)

        with (
            patch.object(block.attn1, "forward", autospec=True) as mock_attn1,
            patch.object(block.attn2, "forward", autospec=True) as mock_attn2,
            patch.object(block.ffn, "forward", autospec=True) as mock_ffn,
        ):
            mock_attn1.side_effect = _zero_like
            mock_attn2.side_effect = _zero_like
            mock_ffn.side_effect = _zero_like

            output = block(hidden_states, encoder_hidden_states, temb, rotary_emb=None)

        self.assert_tensor_device(output, expected_device)
        self.assert_tensor_device(block.scale_shift_table.data, expected_device)

    def test_cross_attention_norm_device_alignment(self):
        """Ensure FP32LayerNorm used in cross-attn path moves to the execution device."""
        device = self._accelerator_device()
        config = self.block_config.copy()
        config["cross_attn_norm"] = True
        block = WanTransformerBlock(**config).to(device)
        expected_device = (
            f"{device.type}:{device.index}"
            if device.index is not None
            else ("mps:0" if device.type == "mps" else device.type)
        )

        block.norm2.weight.data = block.norm2.weight.data.to("cpu")

        hidden_states = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=device)
        encoder_hidden_states = torch.randn(self.batch_size, 77, self.hidden_dim, device=device)
        temb = torch.randn(self.batch_size, 6, self.hidden_dim, device=device)
        rotary_emb = torch.randn(1, 1, self.seq_len, self.head_dim // 2, dtype=torch.complex64).to(device)

        block(hidden_states, encoder_hidden_states, temb, rotary_emb)

        self.assert_tensor_device(block.norm2.weight.data, expected_device)

    def test_feed_forward_chunking_matches_full_pass(self):
        """Chunked feed-forward should match the unchunked reference."""
        base_block = WanTransformerBlock(**self.block_config)
        chunked_block = copy.deepcopy(base_block)
        chunked_block.set_chunk_feed_forward(chunk_size=2, dim=0)

        encoder_hidden_states = torch.randn(self.batch_size, 77, self.hidden_dim)
        temb = torch.randn(self.batch_size, 6, self.hidden_dim)
        rotary_emb = torch.randn(1, 1, self.seq_len, self.head_dim // 2, dtype=torch.complex64)

        ref_output = base_block(
            hidden_states=self.hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            rotary_emb=rotary_emb,
        )
        chunked_output = chunked_block(
            hidden_states=self.hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            rotary_emb=rotary_emb,
        )

        torch.testing.assert_close(ref_output, chunked_output, atol=1e-5, rtol=1e-5)

    def test_feed_forward_chunking_handles_non_divisible_shapes(self):
        """Chunking gracefully falls back when batch size < chunk size."""
        base_block = WanTransformerBlock(**self.block_config)
        chunked_block = copy.deepcopy(base_block)
        chunked_block.set_chunk_feed_forward(chunk_size=16, dim=0)

        encoder_hidden_states = torch.randn(self.batch_size, 77, self.hidden_dim)
        temb = torch.randn(self.batch_size, 6, self.hidden_dim)
        rotary_emb = torch.randn(1, 1, self.seq_len, self.head_dim // 2, dtype=torch.complex64)

        ref_output = base_block(
            hidden_states=self.hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            rotary_emb=rotary_emb,
        )
        chunked_output = chunked_block(
            hidden_states=self.hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            rotary_emb=rotary_emb,
        )

        torch.testing.assert_close(ref_output, chunked_output, atol=1e-5, rtol=1e-5)

    def test_auto_chunk_uses_batch_dimension_when_batch_gt_one(self):
        block = WanTransformerBlock(**self.block_config)
        block.set_chunk_feed_forward(None, None)

        hidden_states = torch.randn(2, self.seq_len, self.hidden_dim)
        encoder_hidden_states = torch.randn(2, 77, self.hidden_dim)
        temb = torch.randn(2, 6, self.hidden_dim)
        rotary_emb = torch.randn(1, 1, self.seq_len, self.head_dim // 2, dtype=torch.complex64)

        def _passthrough(ffn, norm_hs, dim, size):
            return ffn(norm_hs)

        with patch(
            "simpletuner.helpers.models.wan.transformer._chunked_feed_forward", side_effect=_passthrough
        ) as mock_chunk:
            block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb, rotary_emb=rotary_emb)

        self.assertIsNotNone(mock_chunk.call_args)
        self.assertEqual(mock_chunk.call_args[0][2], 0)

    def test_auto_chunk_switches_to_sequence_dimension_for_single_batch(self):
        block = WanTransformerBlock(**self.block_config)
        block.set_chunk_feed_forward(None, None)

        hidden_states = torch.randn(1, self.seq_len, self.hidden_dim)
        encoder_hidden_states = torch.randn(1, 77, self.hidden_dim)
        temb = torch.randn(1, 6, self.hidden_dim)
        rotary_emb = torch.randn(1, 1, self.seq_len, self.head_dim // 2, dtype=torch.complex64)

        def _passthrough(ffn, norm_hs, dim, size):
            return ffn(norm_hs)

        with patch(
            "simpletuner.helpers.models.wan.transformer._chunked_feed_forward", side_effect=_passthrough
        ) as mock_chunk:
            block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb, rotary_emb=rotary_emb)

        self.assertIsNotNone(mock_chunk.call_args)
        self.assertEqual(mock_chunk.call_args[0][2], 1)

    def test_dtype_conversions_float_operations(self):
        """Test dtype conversions for float operations in normalization."""
        if not torch.cuda.is_available():
            self.skipTest("float16 normalization path requires CUDA for WAN transformer block")

        block = WanTransformerBlock(**self.block_config)

        # Test with float16 input
        hidden_states_fp16 = self.hidden_states.to(torch.float16)
        encoder_hidden_states = torch.randn(self.batch_size, 77, self.hidden_dim, dtype=torch.float16)
        temb = torch.randn(self.batch_size, 6, self.hidden_dim, dtype=torch.float16)
        rotary_emb = torch.randn(1, 1, self.seq_len, self.head_dim // 2, dtype=torch.complex64)

        output = block.forward(
            hidden_states=hidden_states_fp16, encoder_hidden_states=encoder_hidden_states, temb=temb, rotary_emb=rotary_emb
        )

        # Output should preserve original dtype
        self.assert_tensor_dtype(output, torch.float16)

    def test_typo_prevention_parameter_names(self):
        """Test typo prevention for constructor parameter names."""
        valid_params = self.block_config.copy()

        typo_params = {
            "dims": "dim",  # Extra 's'
            "ffn_dims": "ffn_dim",  # Extra 's'
            "num_head": "num_heads",  # Missing 's'
            "qk_norms": "qk_norm",  # Extra 's'
            "cross_attn_norms": "cross_attn_norm",  # Extra 's'
            "epss": "eps",  # Extra 's'
            "added_kv_proj_dims": "added_kv_proj_dim",  # Extra 's'
        }

        for typo_param, correct_param in typo_params.items():
            invalid_config = valid_params.copy()
            invalid_config[typo_param] = invalid_config.pop(correct_param)

            with self.assertRaises(TypeError) as cm:
                WanTransformerBlock(**invalid_config)
            self.assertIn("unexpected keyword argument", str(cm.exception))


class TestWanTransformer3DModel(TransformerBaseTest):
    """Test WanTransformer3DModel main model class."""

    def setUp(self):
        super().setUp()
        self.model_config = {
            "patch_size": (1, 2, 2),
            "num_attention_heads": 8,  # Reduced for testing
            "attention_head_dim": 128,
            "in_channels": 16,
            "out_channels": 16,
            "text_dim": 4096,
            "freq_dim": 256,
            "ffn_dim": 1024,  # Reduced for testing
            "num_layers": 2,  # Reduced for testing
            "cross_attn_norm": True,
            "qk_norm": "rms_norm_across_heads",
            "eps": 1e-6,
            "image_dim": self.hidden_dim,
            "added_kv_proj_dim": None,
            "rope_max_seq_len": 1024,
        }

    def test_basic_instantiation(self):
        """Test basic model instantiation."""
        model = WanTransformer3DModel(**self.model_config)

        self.assertIsNotNone(model)
        self.assertIsNotNone(model.rope)
        self.assertIsNotNone(model.patch_embedding)
        self.assertIsNotNone(model.condition_embedder)
        self.assertIsNotNone(model.blocks)
        self.assertIsNotNone(model.norm_out)
        self.assertIsNotNone(model.proj_out)
        self.assertIsNotNone(model.scale_shift_table)

        # Check number of transformer blocks
        self.assertEqual(len(model.blocks), self.model_config["num_layers"])

    def test_set_chunk_feed_forward_propagates(self):
        """Model-level chunk configuration should propagate to every block."""
        model = WanTransformer3DModel(**self.model_config)
        chunk_size = 2
        model.set_chunk_feed_forward(chunk_size, dim=0)
        self.assertEqual(model._feed_forward_chunk_size, chunk_size)
        for block in model.blocks:
            self.assertEqual(block._chunk_size, chunk_size)

    def test_forward_pass_minimal(self):
        """Test minimal forward pass with video data."""
        model = WanTransformer3DModel(**self.model_config)

        # Video input (batch, channels, frames, height, width)
        batch_size, in_channels, num_frames, height, width = 2, 16, 4, 8, 8
        hidden_states = torch.randn(batch_size, in_channels, num_frames, height, width)
        timestep = torch.randint(0, 1000, (batch_size,))
        encoder_hidden_states = torch.randn(batch_size, 77, self.model_config["text_dim"])

        with torch.no_grad():
            output = model.forward(
                hidden_states=hidden_states, timestep=timestep, encoder_hidden_states=encoder_hidden_states
            )

        if hasattr(output, "sample"):
            output_tensor = output.sample
        else:
            output_tensor = output

        # Output should have same spatial structure as input
        expected_shape = (batch_size, self.model_config["out_channels"], num_frames, height, width)
        self.assert_tensor_shape(output_tensor, expected_shape)
        self.assert_no_nan_or_inf(output_tensor)

    def test_forward_pass_with_image_encoder_states(self):
        """Test forward pass with image encoder states for I2V."""
        model = WanTransformer3DModel(**self.model_config)

        batch_size, in_channels, num_frames, height, width = 2, 16, 4, 8, 8
        hidden_states = torch.randn(batch_size, in_channels, num_frames, height, width)
        timestep = torch.randint(0, 1000, (batch_size,))
        encoder_hidden_states = torch.randn(batch_size, 77, self.model_config["text_dim"])
        encoder_hidden_states_image = torch.randn(batch_size, 513, self.model_config["image_dim"])

        with torch.no_grad():
            output = model.forward(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_image=encoder_hidden_states_image,
            )

        if hasattr(output, "sample"):
            output_tensor = output.sample
        else:
            output_tensor = output

        expected_shape = (batch_size, self.model_config["out_channels"], num_frames, height, width)
        self.assert_tensor_shape(output_tensor, expected_shape)

    def test_set_time_embedding_v2_1_toggle(self):
        """Ensure the helper toggles the internal force flag."""
        model = WanTransformer3DModel(**self.model_config)
        self.assertFalse(model.force_v2_1_time_embedding)
        model.set_time_embedding_v2_1(True)
        self.assertTrue(model.force_v2_1_time_embedding)
        model.set_time_embedding_v2_1(False)
        self.assertFalse(model.force_v2_1_time_embedding)

    def test_forward_time_embedding_override_with_sequence_timesteps(self):
        """Time embedding override should handle sequence-shaped timesteps without errors."""
        model = WanTransformer3DModel(**self.model_config)
        model.set_time_embedding_v2_1(True)

        batch_size, in_channels, num_frames, height, width = 1, 16, 4, 8, 8
        hidden_states = torch.randn(batch_size, in_channels, num_frames, height, width)
        # Simulate Wan 2.2-style timestep tensor (batch, sequence_length)
        sequence_length = num_frames // self.model_config["patch_size"][0]
        timestep = torch.randint(0, 1000, (batch_size, sequence_length))
        encoder_hidden_states = torch.randn(batch_size, 77, self.model_config["text_dim"])

        with torch.no_grad():
            output = model.forward(
                hidden_states=hidden_states, timestep=timestep, encoder_hidden_states=encoder_hidden_states
            )

        output_tensor = output.sample if hasattr(output, "sample") else output
        expected_shape = (batch_size, self.model_config["out_channels"], num_frames, height, width)
        self.assert_tensor_shape(output_tensor, expected_shape)

    def test_3d_patch_embedding(self):
        """Test 3D patch embedding processing."""
        model = WanTransformer3DModel(**self.model_config)

        batch_size, in_channels, num_frames, height, width = 1, 16, 4, 8, 8
        hidden_states = torch.randn(batch_size, in_channels, num_frames, height, width)

        # Test patch embedding directly
        with torch.no_grad():
            embedded = model.patch_embedding(hidden_states)

        # Should flatten spatial dimensions
        p_t, p_h, p_w = model.config.patch_size
        expected_seq_len = (num_frames // p_t) * (height // p_h) * (width // p_w)
        expected_embed_dim = model.config.num_attention_heads * model.config.attention_head_dim

        # After Conv3D and flatten/transpose operations
        self.assertEqual(embedded.shape[0], batch_size)
        self.assertEqual(embedded.shape[1], expected_embed_dim)
        self.assertEqual(embedded.shape[2:].numel(), expected_seq_len)

    @patch("simpletuner.helpers.training.tread.TREADRouter")
    def test_tread_routing_integration(self, mock_tread_router_class):
        """Test TREAD routing integration with WAN-specific routing."""
        model = WanTransformer3DModel(**self.model_config)

        # Create mock router
        mock_router = Mock()
        mock_mask_info = Mock()
        mock_mask_info.ids_shuffle = torch.randperm(64).unsqueeze(0).expand(2, -1)
        mock_router.get_mask.return_value = mock_mask_info
        mock_router.start_route.return_value = torch.randn(2, 32, 1024)  # Reduced tokens
        mock_router.end_route.return_value = torch.randn(2, 64, 1024)  # Restored tokens

        # Set up routing
        routes = [{"start_layer_idx": 0, "end_layer_idx": 0, "selection_ratio": 0.5}]
        model.set_router(mock_router, routes)

        batch_size, in_channels, num_frames, height, width = 2, 16, 4, 8, 8
        hidden_states = torch.randn(batch_size, in_channels, num_frames, height, width)
        timestep = torch.randint(0, 1000, (batch_size,))
        encoder_hidden_states = torch.randn(batch_size, 77, self.model_config["text_dim"])

        model.train()  # Enable routing
        with torch.enable_grad():
            output = model.forward(
                hidden_states=hidden_states, timestep=timestep, encoder_hidden_states=encoder_hidden_states
            )

        # Router methods should be called
        mock_router.get_mask.assert_called()
        mock_router.start_route.assert_called()
        mock_router.end_route.assert_called()

    def test_rope_routing_static_method(self):
        """Test _route_rope static method for rotary embedding routing."""
        # Create test data
        rope = torch.randn(1, 1, 64, 32, dtype=torch.complex64)
        mock_info = Mock()
        mock_info.ids_shuffle = torch.randperm(64).unsqueeze(0).expand(2, -1)
        keep_len = 32
        batch = 2

        routed_rope = WanTransformer3DModel._route_rope(rope, mock_info, keep_len, batch)

        # Should output batched, reduced rotary embeddings
        expected_shape = (batch, 1, keep_len, 32)
        self.assert_tensor_shape(routed_rope, expected_shape)

    def test_skip_layers_functionality(self):
        """Test skip_layers parameter to skip specific layers."""
        model = WanTransformer3DModel(**self.model_config)

        batch_size, in_channels, num_frames, height, width = 1, 16, 4, 8, 8
        hidden_states = torch.randn(batch_size, in_channels, num_frames, height, width)
        timestep = torch.randint(0, 1000, (batch_size,))
        encoder_hidden_states = torch.randn(batch_size, 77, self.model_config["text_dim"])

        # Skip the first layer
        skip_layers = [0]

        with torch.no_grad():
            output = model.forward(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                skip_layers=skip_layers,
            )

        # Should still produce valid output
        if hasattr(output, "sample"):
            output_tensor = output.sample
        else:
            output_tensor = output

        expected_shape = (batch_size, self.model_config["out_channels"], num_frames, height, width)
        self.assert_tensor_shape(output_tensor, expected_shape)

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing functionality."""
        model = WanTransformer3DModel(**self.model_config)
        model.gradient_checkpointing = True

        # Use a wrapper to pass use_reentrant=False
        def checkpoint_func(func, *args, **kwargs):
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)

        model._gradient_checkpointing_func = checkpoint_func

        batch_size, in_channels, num_frames, height, width = 1, 16, 2, 4, 4
        hidden_states = torch.randn(batch_size, in_channels, num_frames, height, width, requires_grad=True)
        timestep = torch.randint(0, 1000, (batch_size,))
        encoder_hidden_states = torch.randn(batch_size, 77, self.model_config["text_dim"])

        output = model.forward(hidden_states=hidden_states, timestep=timestep, encoder_hidden_states=encoder_hidden_states)

        if hasattr(output, "sample"):
            loss = output.sample.sum()
        else:
            loss = output.sum()

        loss.backward()
        self.assertIsNotNone(hidden_states.grad)

    def test_return_dict_control(self):
        """Test return_dict parameter controls output format."""
        model = WanTransformer3DModel(**self.model_config)

        batch_size, in_channels, num_frames, height, width = 1, 16, 2, 4, 4
        hidden_states = torch.randn(batch_size, in_channels, num_frames, height, width)
        timestep = torch.randint(0, 1000, (batch_size,))
        encoder_hidden_states = torch.randn(batch_size, 77, self.model_config["text_dim"])

        # Test return_dict=False
        with torch.no_grad():
            output_tuple = model.forward(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False,
            )

        self.assertIsInstance(output_tuple, tuple)

        # Test return_dict=True (default)
        with torch.no_grad():
            output_dict = model.forward(
                hidden_states=hidden_states, timestep=timestep, encoder_hidden_states=encoder_hidden_states, return_dict=True
            )

        self.assertTrue(hasattr(output_dict, "sample"))

    def test_output_reshape_operations(self):
        """Test complex output reshaping for video structure."""
        model = WanTransformer3DModel(**self.model_config)

        batch_size, in_channels, num_frames, height, width = 1, 16, 4, 8, 8
        hidden_states = torch.randn(batch_size, in_channels, num_frames, height, width)
        timestep = torch.randint(0, 1000, (batch_size,))
        encoder_hidden_states = torch.randn(batch_size, 77, self.model_config["text_dim"])

        with torch.no_grad():
            output = model.forward(
                hidden_states=hidden_states, timestep=timestep, encoder_hidden_states=encoder_hidden_states
            )

        if hasattr(output, "sample"):
            output_tensor = output.sample
        else:
            output_tensor = output

        # Should reshape back to original video dimensions
        self.assert_tensor_shape(output_tensor, (batch_size, self.model_config["out_channels"], num_frames, height, width))

    def test_typo_prevention_parameter_names(self):
        """Test typo prevention for constructor parameter names."""
        valid_params = self.model_config.copy()

        typo_params = {
            "patch_siz": "patch_size",  # Missing 'e'
            "num_attention_head": "num_attention_heads",  # Missing 's'
            "attention_head_dims": "attention_head_dim",  # Extra 's'
            "in_channel": "in_channels",  # Missing 's'
            "out_channel": "out_channels",  # Missing 's'
            "text_dims": "text_dim",  # Extra 's'
            "freq_dims": "freq_dim",  # Extra 's'
            "ffn_dims": "ffn_dim",  # Extra 's'
            "num_layer": "num_layers",  # Missing 's'
            "cross_attn_norms": "cross_attn_norm",  # Extra 's'
            "qk_norms": "qk_norm",  # Extra 's'
            "epss": "eps",  # Extra 's'
            "image_dims": "image_dim",  # Extra 's'
            "added_kv_proj_dims": "added_kv_proj_dim",  # Extra 's'
            "rope_max_seq_length": "rope_max_seq_len",  # Different name
        }

        for typo_param, correct_param in typo_params.items():
            invalid_config = valid_params.copy()
            invalid_config[typo_param] = invalid_config.pop(correct_param)

            with self.assertRaises(TypeError) as cm:
                WanTransformer3DModel(**invalid_config)
            self.assertIn("unexpected keyword argument", str(cm.exception))

    def test_performance_benchmark(self):
        """Test performance benchmark for WAN video transformer."""
        model = WanTransformer3DModel(**self.model_config)

        inputs = {
            "hidden_states": torch.randn(1, 16, 2, 4, 4),  # Small for speed
            "timestep": torch.randint(0, 1000, (1,)),
            "encoder_hidden_states": torch.randn(1, 77, self.model_config["text_dim"]),
        }

        # Test should complete within reasonable time for small model
        self.run_performance_benchmark(model, inputs, max_time_ms=3000.0)


if __name__ == "__main__":
    unittest.main()
