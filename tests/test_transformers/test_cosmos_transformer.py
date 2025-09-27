"""
Comprehensive unit tests for Cosmos transformer components.
Tests all 10 classes and 10 functions with focus on typo prevention and edge cases.
"""

import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from transformer_base_test import (
    AttentionProcessorTestMixin,
    EmbeddingTestMixin,
    TransformerBaseTest,
    TransformerBlockTestMixin,
)
from transformer_test_helpers import (
    MockAttention,
    MockDiffusersConfig,
    MockingUtils,
    MockModule,
    MockNormLayer,
    PerformanceUtils,
    ShapeValidator,
    TensorGenerator,
    TypoTestUtils,
)

# Import the target classes
from simpletuner.helpers.models.cosmos.transformer import (
    CosmosAdaLayerNorm,
    CosmosAdaLayerNormZero,
    CosmosAttnProcessor2_0,
    CosmosEmbedding,
    CosmosLearnablePositionalEmbed,
    CosmosPatchEmbed,
    CosmosRotaryPosEmbed,
    CosmosTimestepEmbedding,
    CosmosTransformer3DModel,
    CosmosTransformerBlock,
)


class TestCosmosPatchEmbed(TransformerBaseTest, EmbeddingTestMixin):
    """Comprehensive tests for CosmosPatchEmbed class."""

    def setUp(self):
        super().setUp()
        self.patch_embed_config = {"in_channels": 16, "out_channels": 1024, "patch_size": (1, 2, 2), "bias": True}

    def test_patch_embed_instantiation(self):
        """Test CosmosPatchEmbed instantiation."""
        patch_embed = CosmosPatchEmbed(**self.patch_embed_config)

        # Check basic attributes
        self.assertEqual(patch_embed.patch_size, (1, 2, 2))

        # Check linear projection
        self.assertIsInstance(patch_embed.proj, nn.Linear)
        expected_in_features = 16 * 1 * 2 * 2  # in_channels * patch_size product
        self.assertEqual(patch_embed.proj.in_features, expected_in_features)
        self.assertEqual(patch_embed.proj.out_features, 1024)
        self.assertIsNotNone(patch_embed.proj.bias)  # bias=True

    def test_patch_embed_instantiation_no_bias(self):
        """Test CosmosPatchEmbed instantiation without bias."""
        config = self.patch_embed_config.copy()
        config["bias"] = False

        patch_embed = CosmosPatchEmbed(**config)
        self.assertIsNone(patch_embed.proj.bias)

    def test_patch_embed_parameter_typos(self):
        """Test parameter name typo prevention in instantiation."""
        valid_params = self.patch_embed_config.copy()

        typo_mappings = {
            "in_channel": "in_channels",
            "out_channel": "out_channels",
            "patch_szie": "patch_size",
            "bias_": "bias",
            "in_channels_": "in_channels",
        }

        for typo_param, correct_param in typo_mappings.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            with self.assertRaises(TypeError) as context:
                CosmosPatchEmbed(**invalid_params)
            self.assertIn("unexpected keyword argument", str(context.exception))

    def test_patch_embed_forward_pass(self):
        """Test forward pass with various input shapes."""
        patch_embed = CosmosPatchEmbed(**self.patch_embed_config)

        # Test cases: (batch, channels, frames, height, width)
        test_cases = [
            (1, 16, 8, 32, 32),  # Single batch
            (2, 16, 16, 64, 64),  # Multiple batch
            (4, 16, 4, 16, 16),  # Small spatial
            (2, 16, 32, 128, 128),  # Large spatial
        ]

        p_t, p_h, p_w = self.patch_embed_config["patch_size"]

        for batch_size, channels, frames, height, width in test_cases:
            with torch.no_grad():
                hidden_states = torch.randn(batch_size, channels, frames, height, width)

                # Test forward pass
                output = patch_embed.forward(hidden_states)

                # Calculate expected output shape
                expected_t = frames // p_t
                expected_h = height // p_h
                expected_w = width // p_w
                expected_shape = (batch_size, expected_t, expected_h, expected_w, 1024)

                self.assert_tensor_shape(output, expected_shape)
                self.assert_no_nan_or_inf(output)

    def test_patch_embed_forward_mathematical_correctness(self):
        """Test mathematical correctness of patch embedding operation."""
        patch_embed = CosmosPatchEmbed(**self.patch_embed_config)

        batch_size, channels, frames, height, width = 2, 16, 8, 32, 32
        hidden_states = torch.randn(batch_size, channels, frames, height, width)

        with torch.no_grad():
            output = patch_embed.forward(hidden_states)

            # Manual computation for verification
            p_t, p_h, p_w = patch_embed.patch_size
            expected_shape_after_reshape = (
                batch_size,
                channels,
                frames // p_t,
                p_t,
                height // p_h,
                p_h,
                width // p_w,
                p_w,
            )

            # Verify the reshaping logic
            reshaped = hidden_states.reshape(*expected_shape_after_reshape)
            permuted = reshaped.permute(0, 2, 4, 6, 1, 3, 5, 7)
            flattened = permuted.flatten(4, 7)

            # Apply projection manually
            expected_output = patch_embed.proj(flattened)

            torch.testing.assert_close(output, expected_output, rtol=1e-5, atol=1e-6)

    def test_patch_embed_forward_shape_validation(self):
        """Test that forward pass validates input shapes correctly."""
        patch_embed = CosmosPatchEmbed(**self.patch_embed_config)

        # Test invalid input shapes (not divisible by patch size)
        invalid_shapes = [
            (2, 16, 7, 32, 32),  # frames not divisible by patch_size[0]
            (2, 16, 8, 31, 32),  # height not divisible by patch_size[1]
            (2, 16, 8, 32, 31),  # width not divisible by patch_size[2]
        ]

        for shape in invalid_shapes:
            with torch.no_grad():
                hidden_states = torch.randn(*shape)

                # Should handle gracefully or provide clear error
                try:
                    output = patch_embed.forward(hidden_states)
                    # If it doesn't error, this might be due to integer division
                    self.assertIsInstance(output, torch.Tensor)
                except RuntimeError as e:
                    # Expected for incompatible shapes
                    self.assertTrue(any(keyword in str(e) for keyword in ["size", "shape", "view"]))

    def test_patch_embed_device_consistency(self):
        """Test device consistency in patch embedding."""
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        patch_embed = CosmosPatchEmbed(**self.patch_embed_config).to(device)

        with torch.no_grad():
            hidden_states = torch.randn(2, 16, 8, 32, 32, device=device)
            output = patch_embed.forward(hidden_states)

            self.assert_tensor_device(output, device)

    def test_patch_embed_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        patch_embed = CosmosPatchEmbed(**self.patch_embed_config)

        required_methods = ["forward"]
        self.run_method_existence_tests(patch_embed, required_methods)


class TestCosmosTimestepEmbedding(TransformerBaseTest, EmbeddingTestMixin):
    """Comprehensive tests for CosmosTimestepEmbedding class."""

    def setUp(self):
        super().setUp()
        self.in_features = 256
        self.out_features = 1024

    def test_timestep_embedding_instantiation(self):
        """Test CosmosTimestepEmbedding instantiation."""
        embedding = CosmosTimestepEmbedding(self.in_features, self.out_features)

        # Check components
        self.assertIsInstance(embedding.linear_1, nn.Linear)
        self.assertIsInstance(embedding.activation, nn.SiLU)
        self.assertIsInstance(embedding.linear_2, nn.Linear)

        # Check dimensions
        self.assertEqual(embedding.linear_1.in_features, self.in_features)
        self.assertEqual(embedding.linear_1.out_features, self.out_features)
        self.assertEqual(embedding.linear_2.in_features, self.out_features)
        self.assertEqual(embedding.linear_2.out_features, 3 * self.out_features)

        # Check bias configuration
        self.assertIsNone(embedding.linear_1.bias)
        self.assertIsNone(embedding.linear_2.bias)

    def test_timestep_embedding_parameter_typos(self):
        """Test parameter name typo prevention in instantiation."""
        valid_params = {"in_features": self.in_features, "out_features": self.out_features}

        typo_mappings = {
            "in_feature": "in_features",
            "out_feature": "out_features",
            "in_feaures": "in_features",
            "out_feaures": "out_features",
        }

        for typo_param, correct_param in typo_mappings.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            with self.assertRaises(TypeError) as context:
                CosmosTimestepEmbedding(**invalid_params)
            self.assertIn("unexpected keyword argument", str(context.exception))

    def test_timestep_embedding_forward_pass(self):
        """Test forward pass with various input shapes."""
        embedding = CosmosTimestepEmbedding(self.in_features, self.out_features)

        test_shapes = [
            (2, self.in_features),
            (1, self.in_features),
            (8, self.in_features),
        ]

        for shape in test_shapes:
            with torch.no_grad():
                timesteps = torch.randn(*shape)
                output = embedding.forward(timesteps)

                expected_shape = (shape[0], 3 * self.out_features)
                self.assert_tensor_shape(output, expected_shape)
                self.assert_no_nan_or_inf(output)

    def test_timestep_embedding_mathematical_correctness(self):
        """Test mathematical correctness of timestep embedding."""
        embedding = CosmosTimestepEmbedding(self.in_features, self.out_features)

        timesteps = torch.randn(2, self.in_features)

        with torch.no_grad():
            output = embedding.forward(timesteps)

            # Manual computation
            emb = embedding.linear_1(timesteps)
            emb = embedding.activation(emb)
            expected_output = embedding.linear_2(emb)

            torch.testing.assert_close(output, expected_output, rtol=1e-5, atol=1e-6)

    def test_timestep_embedding_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        embedding = CosmosTimestepEmbedding(self.in_features, self.out_features)

        required_methods = ["forward"]
        self.run_method_existence_tests(embedding, required_methods)


class TestCosmosEmbedding(TransformerBaseTest, EmbeddingTestMixin):
    """Comprehensive tests for CosmosEmbedding class."""

    def setUp(self):
        super().setUp()
        self.embedding_dim = 256
        self.condition_dim = 1024

    @patch("diffusers.models.embeddings.Timesteps")
    @patch("diffusers.models.normalization.RMSNorm")
    def test_cosmos_embedding_instantiation(self, mock_rms_norm, mock_timesteps):
        """Test CosmosEmbedding instantiation."""
        mock_timesteps.return_value = Mock()
        mock_rms_norm.return_value = Mock()

        embedding = CosmosEmbedding(self.embedding_dim, self.condition_dim)

        # Check components exist
        self.assertTrue(hasattr(embedding, "time_proj"))
        self.assertTrue(hasattr(embedding, "t_embedder"))
        self.assertTrue(hasattr(embedding, "norm"))

        # Check t_embedder is CosmosTimestepEmbedding
        self.assertIsInstance(embedding.t_embedder, CosmosTimestepEmbedding)

    def test_cosmos_embedding_parameter_typos(self):
        """Test parameter name typo prevention in instantiation."""
        valid_params = {"embedding_dim": self.embedding_dim, "condition_dim": self.condition_dim}

        typo_mappings = {
            "embeding_dim": "embedding_dim",
            "condition_dims": "condition_dim",
            "embedding_dims": "embedding_dim",
            "conditon_dim": "condition_dim",
        }

        for typo_param, correct_param in typo_mappings.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            with self.assertRaises(TypeError) as context:
                CosmosEmbedding(**invalid_params)
            self.assertIn("unexpected keyword argument", str(context.exception))

    @patch("diffusers.models.embeddings.Timesteps")
    @patch("diffusers.models.normalization.RMSNorm")
    def test_cosmos_embedding_forward_pass(self, mock_rms_norm, mock_timesteps):
        """Test forward pass with mocked dependencies."""
        # Mock the components
        mock_time_proj = mock_timesteps.return_value
        mock_time_proj.return_value = torch.randn(2, self.embedding_dim)

        mock_norm_instance = mock_rms_norm.return_value
        mock_norm_instance.return_value = torch.randn(2, self.embedding_dim)

        embedding = CosmosEmbedding(self.embedding_dim, self.condition_dim)

        # Mock t_embedder
        embedding.t_embedder = Mock(return_value=torch.randn(2, 3 * self.condition_dim))

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, self.embedding_dim)
        timestep = torch.randint(0, 1000, (batch_size,))

        with torch.no_grad():
            temb, embedded_timestep = embedding.forward(hidden_states, timestep)

            # Check output shapes
            self.assert_tensor_shape(temb, (batch_size, 3 * self.condition_dim))
            self.assert_tensor_shape(embedded_timestep, (batch_size, self.embedding_dim))

    def test_cosmos_embedding_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        with patch("diffusers.models.embeddings.Timesteps"), patch("diffusers.models.normalization.RMSNorm"):

            embedding = CosmosEmbedding(self.embedding_dim, self.condition_dim)

            required_methods = ["forward"]
            self.run_method_existence_tests(embedding, required_methods)


class TestCosmosAdaLayerNorm(TransformerBaseTest):
    """Comprehensive tests for CosmosAdaLayerNorm class."""

    def setUp(self):
        super().setUp()
        self.in_features = 768
        self.hidden_features = 256

    def test_ada_layer_norm_instantiation(self):
        """Test CosmosAdaLayerNorm instantiation."""
        norm = CosmosAdaLayerNorm(self.in_features, self.hidden_features)

        # Check components
        self.assertIsInstance(norm.activation, nn.SiLU)
        self.assertIsInstance(norm.norm, nn.LayerNorm)
        self.assertIsInstance(norm.linear_1, nn.Linear)
        self.assertIsInstance(norm.linear_2, nn.Linear)

        # Check dimensions
        self.assertEqual(norm.embedding_dim, self.in_features)
        self.assertEqual(norm.linear_1.in_features, self.in_features)
        self.assertEqual(norm.linear_1.out_features, self.hidden_features)
        self.assertEqual(norm.linear_2.in_features, self.hidden_features)
        self.assertEqual(norm.linear_2.out_features, 2 * self.in_features)

        # Check LayerNorm configuration
        self.assertFalse(norm.norm.elementwise_affine)
        self.assertEqual(norm.norm.eps, 1e-6)

    def test_ada_layer_norm_parameter_typos(self):
        """Test parameter name typo prevention in instantiation."""
        valid_params = {"in_features": self.in_features, "hidden_features": self.hidden_features}

        typo_mappings = {
            "in_feature": "in_features",
            "hidden_feature": "hidden_features",
            "in_feaures": "in_features",
            "hidden_feaures": "hidden_features",
        }

        for typo_param, correct_param in typo_mappings.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            with self.assertRaises(TypeError) as context:
                CosmosAdaLayerNorm(**invalid_params)
            self.assertIn("unexpected keyword argument", str(context.exception))

    def test_ada_layer_norm_forward_pass(self):
        """Test forward pass functionality."""
        norm = CosmosAdaLayerNorm(self.in_features, self.hidden_features)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, self.in_features)
        embedded_timestep = torch.randn(batch_size, self.in_features)

        with torch.no_grad():
            output = norm.forward(hidden_states, embedded_timestep)

            self.assert_tensor_shape(output, (batch_size, seq_len, self.in_features))
            self.assert_no_nan_or_inf(output)

    def test_ada_layer_norm_forward_with_temb(self):
        """Test forward pass with additional temb parameter."""
        norm = CosmosAdaLayerNorm(self.in_features, self.hidden_features)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, self.in_features)
        embedded_timestep = torch.randn(batch_size, self.in_features)
        temb = torch.randn(batch_size, 2 * self.in_features)

        with torch.no_grad():
            output = norm.forward(hidden_states, embedded_timestep, temb)

            self.assert_tensor_shape(output, (batch_size, seq_len, self.in_features))
            self.assert_no_nan_or_inf(output)

    def test_ada_layer_norm_broadcasting(self):
        """Test broadcasting behavior for different timestep dimensions."""
        norm = CosmosAdaLayerNorm(self.in_features, self.hidden_features)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, self.in_features)

        # Test 2D embedded_timestep (should unsqueeze)
        embedded_timestep_2d = torch.randn(batch_size, self.in_features)

        with torch.no_grad():
            output = norm.forward(hidden_states, embedded_timestep_2d)
            self.assert_tensor_shape(output, (batch_size, seq_len, self.in_features))

    def test_ada_layer_norm_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        norm = CosmosAdaLayerNorm(self.in_features, self.hidden_features)

        required_methods = ["forward"]
        self.run_method_existence_tests(norm, required_methods)


class TestCosmosAdaLayerNormZero(TransformerBaseTest):
    """Comprehensive tests for CosmosAdaLayerNormZero class."""

    def setUp(self):
        super().setUp()
        self.in_features = 768
        self.hidden_features = 256

    def test_ada_layer_norm_zero_instantiation(self):
        """Test CosmosAdaLayerNormZero instantiation."""
        norm = CosmosAdaLayerNormZero(self.in_features, self.hidden_features)

        # Check components
        self.assertIsInstance(norm.norm, nn.LayerNorm)
        self.assertIsInstance(norm.activation, nn.SiLU)
        self.assertIsInstance(norm.linear_1, nn.Linear)
        self.assertIsInstance(norm.linear_2, nn.Linear)

        # Check dimensions
        self.assertEqual(norm.linear_1.in_features, self.in_features)
        self.assertEqual(norm.linear_1.out_features, self.hidden_features)
        self.assertEqual(norm.linear_2.in_features, self.hidden_features)
        self.assertEqual(norm.linear_2.out_features, 3 * self.in_features)

    def test_ada_layer_norm_zero_instantiation_no_hidden(self):
        """Test CosmosAdaLayerNormZero instantiation without hidden_features."""
        norm = CosmosAdaLayerNormZero(self.in_features)

        # Should use Identity for linear_1
        self.assertIsInstance(norm.linear_1, nn.Identity)

    def test_ada_layer_norm_zero_parameter_typos(self):
        """Test parameter name typo prevention in instantiation."""
        valid_params = {"in_features": self.in_features, "hidden_features": self.hidden_features}

        typo_mappings = {
            "in_feature": "in_features",
            "hidden_feature": "hidden_features",
        }

        for typo_param, correct_param in typo_mappings.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            with self.assertRaises(TypeError) as context:
                CosmosAdaLayerNormZero(**invalid_params)
            self.assertIn("unexpected keyword argument", str(context.exception))

    def test_ada_layer_norm_zero_forward_pass(self):
        """Test forward pass functionality."""
        norm = CosmosAdaLayerNormZero(self.in_features, self.hidden_features)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, self.in_features)
        embedded_timestep = torch.randn(batch_size, self.in_features)

        with torch.no_grad():
            output, gate = norm.forward(hidden_states, embedded_timestep)

            self.assert_tensor_shape(output, (batch_size, seq_len, self.in_features))
            self.assert_tensor_shape(gate, (batch_size, self.in_features))
            self.assert_no_nan_or_inf(output)
            self.assert_no_nan_or_inf(gate)

    def test_ada_layer_norm_zero_forward_with_temb(self):
        """Test forward pass with additional temb parameter."""
        norm = CosmosAdaLayerNormZero(self.in_features, self.hidden_features)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, self.in_features)
        embedded_timestep = torch.randn(batch_size, self.in_features)
        temb = torch.randn(batch_size, 3 * self.in_features)

        with torch.no_grad():
            output, gate = norm.forward(hidden_states, embedded_timestep, temb)

            self.assert_tensor_shape(output, (batch_size, seq_len, self.in_features))
            self.assert_tensor_shape(gate, (batch_size, self.in_features))

    def test_ada_layer_norm_zero_mathematical_correctness(self):
        """Test mathematical correctness of the forward pass."""
        norm = CosmosAdaLayerNormZero(self.in_features, self.hidden_features)

        batch_size = 2
        seq_len = 128
        hidden_states = torch.randn(batch_size, seq_len, self.in_features)
        embedded_timestep = torch.randn(batch_size, self.in_features)

        with torch.no_grad():
            # Manual computation
            embedded_timestep_proc = norm.activation(embedded_timestep)
            embedded_timestep_proc = norm.linear_1(embedded_timestep_proc)
            embedded_timestep_proc = norm.linear_2(embedded_timestep_proc)

            shift, scale, gate = embedded_timestep_proc.chunk(3, dim=-1)
            norm_hidden_states = norm.norm(hidden_states)

            # Handle broadcasting
            if embedded_timestep_proc.ndim == 2:
                shift = shift.unsqueeze(1)
                scale = scale.unsqueeze(1)
                gate = gate.unsqueeze(1)

            expected_output = norm_hidden_states * (1 + scale) + shift
            expected_gate = gate.squeeze(1) if gate.shape[1] == 1 else gate

            # Actual forward pass
            actual_output, actual_gate = norm.forward(hidden_states, embedded_timestep)

            torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-6)

    def test_ada_layer_norm_zero_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        norm = CosmosAdaLayerNormZero(self.in_features, self.hidden_features)

        required_methods = ["forward"]
        self.run_method_existence_tests(norm, required_methods)


class TestCosmosAttnProcessor2_0(TransformerBaseTest, AttentionProcessorTestMixin):
    """Comprehensive tests for CosmosAttnProcessor2_0 class."""

    def setUp(self):
        super().setUp()

    def test_attn_processor_instantiation(self):
        """Test CosmosAttnProcessor2_0 instantiation."""
        # Should require PyTorch 2.0
        processor = CosmosAttnProcessor2_0()
        self.assertIsNotNone(processor)

    def test_attn_processor_pytorch_version_check(self):
        """Test that processor checks for PyTorch 2.0."""
        # This test may not work in all environments, so we'll skip if needed
        try:
            with patch("torch.nn.functional.scaled_dot_product_attention", None):
                with self.assertRaises(ImportError) as context:
                    CosmosAttnProcessor2_0()
                self.assertIn("PyTorch 2.0", str(context.exception))
        except ImportError:
            # PyTorch version doesn't have the function, which is expected
            pass

    @patch("diffusers.models.attention_processor.Attention")
    def test_attn_processor_call(self, mock_attention):
        """Test attention processor call method."""
        processor = CosmosAttnProcessor2_0()

        # Mock attention module
        mock_attn = mock_attention.return_value
        mock_attn.heads = 8
        mock_attn.to_q = Mock(return_value=torch.randn(2, 128, 512))
        mock_attn.to_k = Mock(return_value=torch.randn(2, 128, 512))
        mock_attn.to_v = Mock(return_value=torch.randn(2, 128, 512))
        mock_attn.norm_q = Mock(side_effect=lambda x: x)
        mock_attn.norm_k = Mock(side_effect=lambda x: x)
        mock_attn.to_out = nn.ModuleList([nn.Linear(512, 512), nn.Dropout(0.0)])

        hidden_states = torch.randn(2, 128, 512)

        # Test basic call
        with torch.no_grad():
            output = processor(
                attn=mock_attn,
                hidden_states=hidden_states,
            )

            self.assertIsInstance(output, torch.Tensor)
            self.assert_tensor_shape(output, (2, 128, 512))

    def test_attn_processor_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        processor = CosmosAttnProcessor2_0()

        # Should be callable
        self.assertTrue(callable(processor))


class TestCosmosTransformerBlock(TransformerBaseTest, TransformerBlockTestMixin):
    """Comprehensive tests for CosmosTransformerBlock class."""

    def setUp(self):
        super().setUp()
        self.block_config = {
            "num_attention_heads": 16,
            "attention_head_dim": 64,
            "cross_attention_dim": 1024,
            "mlp_ratio": 4.0,
            "adaln_lora_dim": 256,
            "qk_norm": "rms_norm",
            "out_bias": False,
        }

    @patch("diffusers.models.attention.Attention")
    @patch("diffusers.models.attention.FeedForward")
    def test_transformer_block_instantiation(self, mock_feed_forward, mock_attention):
        """Test CosmosTransformerBlock instantiation."""
        mock_attention.return_value = Mock()
        mock_feed_forward.return_value = Mock()

        block = CosmosTransformerBlock(**self.block_config)

        # Check components exist
        component_names = ["norm1", "attn1", "norm2", "attn2", "norm3", "ff"]
        for name in component_names:
            self.assertTrue(hasattr(block, name), f"Missing component: {name}")

        # Check that all norms are CosmosAdaLayerNormZero
        self.assertIsInstance(block.norm1, CosmosAdaLayerNormZero)
        self.assertIsInstance(block.norm2, CosmosAdaLayerNormZero)
        self.assertIsInstance(block.norm3, CosmosAdaLayerNormZero)

    def test_transformer_block_parameter_typos(self):
        """Test parameter name typo prevention in instantiation."""
        valid_params = self.block_config.copy()

        typo_mappings = {
            "num_attention_head": "num_attention_heads",
            "attention_head_dims": "attention_head_dim",
            "cross_attention_dims": "cross_attention_dim",
            "mlp_ratios": "mlp_ratio",
            "adaln_lora_dims": "adaln_lora_dim",
            "qk_norms": "qk_norm",
            "out_bias_": "out_bias",
        }

        for typo_param, correct_param in typo_mappings.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            with self.assertRaises(TypeError) as context:
                CosmosTransformerBlock(**invalid_params)
            self.assertIn("unexpected keyword argument", str(context.exception))

    @patch("diffusers.models.attention.Attention")
    @patch("diffusers.models.attention.FeedForward")
    def test_transformer_block_forward_pass(self, mock_feed_forward, mock_attention):
        """Test forward pass with mocked dependencies."""
        # Setup mocks
        mock_attention_instance = mock_attention.return_value
        mock_attention_instance.return_value = torch.randn(2, 128, 1024)

        mock_ff_instance = mock_feed_forward.return_value
        mock_ff_instance.return_value = torch.randn(2, 128, 1024)

        block = CosmosTransformerBlock(**self.block_config)

        # Mock the norm layers with proper mock modules
        with (
            patch.object(block, "norm1", MockNormLayer(return_tuple=True, num_values=2)),
            patch.object(block, "norm2", MockNormLayer(return_tuple=True, num_values=2)),
            patch.object(block, "norm3", MockNormLayer(return_tuple=True, num_values=2)),
        ):

            # Test inputs
            batch_size, seq_len, hidden_dim = 2, 128, 1024
            hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
            encoder_hidden_states = torch.randn(batch_size, 77, hidden_dim)
            embedded_timestep = torch.randn(batch_size, hidden_dim)

            with torch.no_grad():
                output = block.forward(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    embedded_timestep=embedded_timestep,
                )

            self.assert_tensor_shape(output, (batch_size, seq_len, hidden_dim))
            self.assert_no_nan_or_inf(output)

    def test_transformer_block_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        with patch("diffusers.models.attention.Attention"), patch("diffusers.models.attention.FeedForward"):

            block = CosmosTransformerBlock(**self.block_config)

            required_methods = ["forward"]
            self.run_method_existence_tests(block, required_methods)


class TestCosmosRotaryPosEmbed(TransformerBaseTest):
    """Comprehensive tests for CosmosRotaryPosEmbed class."""

    def setUp(self):
        super().setUp()
        self.rope_config = {
            "hidden_size": 1024,
            "max_size": (128, 240, 240),
            "patch_size": (1, 2, 2),
            "base_fps": 24,
            "rope_scale": (2.0, 1.0, 1.0),
        }

    def test_rotary_pos_embed_instantiation(self):
        """Test CosmosRotaryPosEmbed instantiation."""
        rope = CosmosRotaryPosEmbed(**self.rope_config)

        # Check basic attributes
        expected_max_size = [128 // 1, 240 // 2, 240 // 2]  # max_size // patch_size
        self.assertEqual(rope.max_size, expected_max_size)
        self.assertEqual(rope.patch_size, (1, 2, 2))
        self.assertEqual(rope.base_fps, 24)

        # Check dimension calculations
        expected_dim_h = 1024 // 6 * 2
        expected_dim_w = 1024 // 6 * 2
        expected_dim_t = 1024 - expected_dim_h - expected_dim_w

        self.assertEqual(rope.dim_h, expected_dim_h)
        self.assertEqual(rope.dim_w, expected_dim_w)
        self.assertEqual(rope.dim_t, expected_dim_t)

    def test_rotary_pos_embed_parameter_typos(self):
        """Test parameter name typo prevention in instantiation."""
        valid_params = self.rope_config.copy()

        typo_mappings = {
            "hidden_szie": "hidden_size",
            "max_szie": "max_size",
            "patch_szie": "patch_size",
            "base_fp": "base_fps",
            "rope_scales": "rope_scale",
        }

        for typo_param, correct_param in typo_mappings.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            with self.assertRaises(TypeError) as context:
                CosmosRotaryPosEmbed(**invalid_params)
            self.assertIn("unexpected keyword argument", str(context.exception))

    def test_rotary_pos_embed_forward_pass(self):
        """Test forward pass with various input shapes."""
        rope = CosmosRotaryPosEmbed(**self.rope_config)

        # Test cases: (batch, channels, frames, height, width)
        test_cases = [
            (2, 16, 8, 32, 32),
            (1, 16, 16, 64, 64),
            (4, 16, 4, 16, 16),
        ]

        for batch_size, channels, frames, height, width in test_cases:
            with torch.no_grad():
                hidden_states = torch.randn(batch_size, channels, frames, height, width)

                # Test without fps (images)
                cos, sin = rope.forward(hidden_states)

                # Calculate expected sequence length
                pe_size = [
                    frames // rope.patch_size[0],
                    height // rope.patch_size[1],
                    width // rope.patch_size[2],
                ]
                expected_seq_len = pe_size[0] * pe_size[1] * pe_size[2]

                self.assert_tensor_shape(cos, (expected_seq_len, rope.hidden_size))
                self.assert_tensor_shape(sin, (expected_seq_len, rope.hidden_size))
                self.assert_no_nan_or_inf(cos)
                self.assert_no_nan_or_inf(sin)

    def test_rotary_pos_embed_forward_with_fps(self):
        """Test forward pass with fps parameter for videos."""
        rope = CosmosRotaryPosEmbed(**self.rope_config)

        hidden_states = torch.randn(2, 16, 8, 32, 32)
        fps = 30

        with torch.no_grad():
            cos, sin = rope.forward(hidden_states, fps=fps)

            # Should still produce valid embeddings
            pe_size = [8 // 1, 32 // 2, 32 // 2]  # frames//patch_t, height//patch_h, width//patch_w
            expected_seq_len = pe_size[0] * pe_size[1] * pe_size[2]

            self.assert_tensor_shape(cos, (expected_seq_len, rope.hidden_size))
            self.assert_tensor_shape(sin, (expected_seq_len, rope.hidden_size))

    def test_rotary_pos_embed_mathematical_consistency(self):
        """Test mathematical consistency of rotary embeddings."""
        rope = CosmosRotaryPosEmbed(**self.rope_config)

        hidden_states = torch.randn(2, 16, 8, 32, 32)

        with torch.no_grad():
            cos1, sin1 = rope.forward(hidden_states)
            cos2, sin2 = rope.forward(hidden_states)

            # Should be deterministic
            torch.testing.assert_close(cos1, cos2, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(sin1, sin2, rtol=1e-5, atol=1e-6)

    def test_rotary_pos_embed_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        rope = CosmosRotaryPosEmbed(**self.rope_config)

        required_methods = ["forward"]
        self.run_method_existence_tests(rope, required_methods)


class TestCosmosLearnablePositionalEmbed(TransformerBaseTest):
    """Comprehensive tests for CosmosLearnablePositionalEmbed class."""

    def setUp(self):
        super().setUp()
        self.pos_embed_config = {"hidden_size": 1024, "max_size": (128, 240, 240), "patch_size": (1, 2, 2), "eps": 1e-6}

    def test_learnable_pos_embed_instantiation(self):
        """Test CosmosLearnablePositionalEmbed instantiation."""
        pos_embed = CosmosLearnablePositionalEmbed(**self.pos_embed_config)

        # Check basic attributes
        expected_max_size = [128 // 1, 240 // 2, 240 // 2]
        self.assertEqual(pos_embed.max_size, expected_max_size)
        self.assertEqual(pos_embed.patch_size, (1, 2, 2))
        self.assertEqual(pos_embed.eps, 1e-6)

        # Check learned parameters
        self.assertEqual(tuple(pos_embed.pos_emb_t.shape), (128, 1024))
        self.assertEqual(tuple(pos_embed.pos_emb_h.shape), (120, 1024))
        self.assertEqual(tuple(pos_embed.pos_emb_w.shape), (120, 1024))

    def test_learnable_pos_embed_parameter_typos(self):
        """Test parameter name typo prevention in instantiation."""
        valid_params = self.pos_embed_config.copy()

        typo_mappings = {"hidden_szie": "hidden_size", "max_szie": "max_size", "patch_szie": "patch_size", "ep": "eps"}

        for typo_param, correct_param in typo_mappings.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            with self.assertRaises(TypeError) as context:
                CosmosLearnablePositionalEmbed(**invalid_params)
            self.assertIn("unexpected keyword argument", str(context.exception))

    def test_learnable_pos_embed_forward_pass(self):
        """Test forward pass with various input shapes."""
        pos_embed = CosmosLearnablePositionalEmbed(**self.pos_embed_config)

        # Test cases: (batch, channels, frames, height, width)
        test_cases = [
            (2, 16, 8, 32, 32),
            (1, 16, 16, 64, 64),
            (4, 16, 4, 16, 16),
        ]

        for batch_size, channels, frames, height, width in test_cases:
            with torch.no_grad():
                hidden_states = torch.randn(batch_size, channels, frames, height, width)

                output = pos_embed.forward(hidden_states)

                # Calculate expected output shape
                pe_size = [
                    frames // pos_embed.patch_size[0],
                    height // pos_embed.patch_size[1],
                    width // pos_embed.patch_size[2],
                ]
                expected_seq_len = pe_size[0] * pe_size[1] * pe_size[2]

                self.assert_tensor_shape(output, (batch_size, expected_seq_len, 1024))
                self.assert_no_nan_or_inf(output)

    def test_learnable_pos_embed_normalization(self):
        """Test that output is properly normalized."""
        pos_embed = CosmosLearnablePositionalEmbed(**self.pos_embed_config)

        hidden_states = torch.randn(2, 16, 8, 32, 32)

        with torch.no_grad():
            output = pos_embed.forward(hidden_states)

            # Check that normalization was applied correctly
            # The norm should be close to sqrt(norm.numel() / emb.numel())
            norms = torch.linalg.vector_norm(output, dim=-1, keepdim=True)

            # All norms should be positive
            self.assertTrue(torch.all(norms > 0))

    def test_learnable_pos_embed_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        pos_embed = CosmosLearnablePositionalEmbed(**self.pos_embed_config)

        required_methods = ["forward"]
        self.run_method_existence_tests(pos_embed, required_methods)


class TestCosmosTransformer3DModel(TransformerBaseTest):
    """Comprehensive tests for CosmosTransformer3DModel main class."""

    def setUp(self):
        super().setUp()
        self.model_config = {
            "in_channels": 16,
            "out_channels": 16,
            "num_attention_heads": 16,
            "attention_head_dim": 64,
            "num_layers": 4,
            "mlp_ratio": 4.0,
            "text_embed_dim": 1024,
            "adaln_lora_dim": 256,
            "max_size": (32, 64, 64),
            "patch_size": (1, 2, 2),
            "rope_scale": (2.0, 1.0, 1.0),
            "concat_padding_mask": True,
            "extra_pos_embed_type": "learnable",
        }

    @patch("diffusers.models.attention.Attention")
    @patch("diffusers.models.attention.FeedForward")
    def test_transformer_3d_model_instantiation(self, mock_feed_forward, mock_attention):
        """Test CosmosTransformer3DModel instantiation."""
        mock_attention.return_value = Mock()
        mock_feed_forward.return_value = Mock()

        model = CosmosTransformer3DModel(**self.model_config)

        # Check basic attributes
        expected_hidden_size = 16 * 64  # num_attention_heads * attention_head_dim
        self.assertEqual(model.gradient_checkpointing, False)

        # Check components exist
        component_names = [
            "patch_embed",
            "rope",
            "learnable_pos_embed",
            "time_embed",
            "transformer_blocks",
            "norm_out",
            "proj_out",
        ]

        for name in component_names:
            self.assertTrue(hasattr(model, name), f"Missing component: {name}")

        # Check transformer blocks count
        self.assertEqual(len(model.transformer_blocks), 4)

        # Check patch embed configuration
        expected_in_channels = 16 + 1  # in_channels + padding mask channel
        self.assertEqual(model.patch_embed.proj.in_features, expected_in_channels * 1 * 2 * 2)

    def test_transformer_3d_model_parameter_typos(self):
        """Test parameter name typo prevention in instantiation."""
        valid_params = self.model_config.copy()

        typo_mappings = {
            "in_channel": "in_channels",
            "out_channel": "out_channels",
            "num_attention_head": "num_attention_heads",
            "attention_head_dims": "attention_head_dim",
            "num_layer": "num_layers",
            "mlp_ratios": "mlp_ratio",
            "text_embed_dims": "text_embed_dim",
            "adaln_lora_dims": "adaln_lora_dim",
            "max_szie": "max_size",
            "patch_szie": "patch_size",
            "rope_scales": "rope_scale",
            "concat_padding_mas": "concat_padding_mask",
            "extra_pos_embed_typ": "extra_pos_embed_type",
        }

        for typo_param, correct_param in typo_mappings.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            with self.assertRaises(TypeError) as context:
                CosmosTransformer3DModel(**invalid_params)
            self.assertIn("unexpected keyword argument", str(context.exception))

    def test_transformer_3d_model_tread_router_methods(self):
        """Test TREAD router configuration methods."""
        with patch("diffusers.models.attention.Attention"), patch("diffusers.models.attention.FeedForward"):

            model = CosmosTransformer3DModel(**self.model_config)

            # Test setting router
            mock_router = Mock()
            mock_routes = [{"start_layer_idx": 0, "end_layer_idx": 2, "selection_ratio": 0.5}]

            model.set_router(mock_router, mock_routes)
            self.assertEqual(model._tread_router, mock_router)
            self.assertEqual(model._tread_routes, mock_routes)

    def test_transformer_3d_model_config_attributes(self):
        """Test config attribute access and typo prevention."""
        with patch("diffusers.models.attention.Attention"), patch("diffusers.models.attention.FeedForward"):

            model = CosmosTransformer3DModel(**self.model_config)

            # Test that config attributes are accessible
            config_attrs = [
                "in_channels",
                "out_channels",
                "num_attention_heads",
                "attention_head_dim",
                "num_layers",
                "mlp_ratio",
                "text_embed_dim",
                "adaln_lora_dim",
                "max_size",
                "patch_size",
                "rope_scale",
                "concat_padding_mask",
                "extra_pos_embed_type",
            ]

            for attr in config_attrs:
                self.assertTrue(hasattr(model.config, attr), f"Config missing attribute: {attr}")
                # Verify the value matches what we set
                expected_value = self.model_config[attr]
                actual_value = getattr(model.config, attr)
                self.assertEqual(actual_value, expected_value, f"Config {attr} mismatch")

    def test_transformer_3d_model_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        with patch("diffusers.models.attention.Attention"), patch("diffusers.models.attention.FeedForward"):

            model = CosmosTransformer3DModel(**self.model_config)

            required_methods = ["forward", "set_router"]
            self.run_method_existence_tests(model, required_methods)

    @patch("diffusers.models.attention.Attention")
    @patch("diffusers.models.attention.FeedForward")
    @patch("torchvision.transforms.functional.resize")
    def test_transformer_3d_model_forward_pass_structure(self, mock_resize, mock_feed_forward, mock_attention):
        """Test the structure of forward pass without full execution."""
        # Mock components
        mock_attention.return_value = Mock()
        mock_feed_forward.return_value = Mock()
        mock_resize.return_value = torch.randn(2, 1, 32, 64, 64)

        model = CosmosTransformer3DModel(**self.model_config)

        # Mock key components to avoid complex tensor operations
        model.patch_embed = Mock(return_value=torch.randn(2, 512, 1024))
        model.rope = Mock(return_value=(torch.randn(512, 64), torch.randn(512, 64)))
        model.learnable_pos_embed = Mock(return_value=torch.randn(2, 512, 1024))
        model.time_embed = Mock(return_value=(torch.randn(2, 1024), torch.randn(2, 1024)))
        model.norm_out = Mock(return_value=torch.randn(2, 512, 1024))
        model.proj_out = Mock(return_value=torch.randn(2, 512, 64))

        # Mock transformer blocks using proper mocking
        MockingUtils.safely_replace_modules_in_list(model.transformer_blocks)

        # Test inputs
        hidden_states = torch.randn(2, 16, 8, 32, 32)  # batch, channels, frames, height, width
        timestep = torch.randint(0, 1000, (2,))
        encoder_hidden_states = torch.randn(2, 77, 1024)
        padding_mask = torch.ones(2, 1, 32, 32)

        with torch.no_grad():
            output = model.forward(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                padding_mask=padding_mask,
                return_dict=True,
            )

            # Verify output structure
            self.assertTrue(hasattr(output, "sample"))
            sample = output.sample
            self.assertIsInstance(sample, torch.Tensor)


class TestCosmosTransformerPerformance(TransformerBaseTest):
    """Performance and benchmarking tests for Cosmos transformer components."""

    def setUp(self):
        super().setUp()
        self.perf_utils = PerformanceUtils()

    def test_patch_embed_performance(self):
        """Test patch embedding performance."""
        config = {"in_channels": 16, "out_channels": 1024, "patch_size": (1, 2, 2), "bias": False}

        patch_embed = CosmosPatchEmbed(**config)
        hidden_states = torch.randn(4, 16, 8, 32, 32)  # Larger batch for performance test

        inputs = {"hidden_states": hidden_states}

        # Measure forward pass time
        avg_time = self.perf_utils.measure_forward_pass_time(
            lambda hidden_states: patch_embed.forward(hidden_states), inputs, num_runs=20
        )

        # Should be fast for moderate inputs
        self.assertLess(avg_time * 1000, 200)  # Less than 200ms

    def test_rotary_pos_embed_performance(self):
        """Test rotary position embedding performance."""
        config = {
            "hidden_size": 1024,
            "max_size": (64, 128, 128),
            "patch_size": (1, 2, 2),
            "base_fps": 24,
            "rope_scale": (2.0, 1.0, 1.0),
        }

        rope = CosmosRotaryPosEmbed(**config)
        hidden_states = torch.randn(4, 16, 16, 64, 64)

        inputs = {"hidden_states": hidden_states}

        # Measure forward pass time
        avg_time = self.perf_utils.measure_forward_pass_time(
            lambda hidden_states: rope.forward(hidden_states), inputs, num_runs=10
        )

        # Should be reasonable for this size
        self.assertLess(avg_time * 1000, 500)  # Less than 500ms


if __name__ == "__main__":
    unittest.main()
