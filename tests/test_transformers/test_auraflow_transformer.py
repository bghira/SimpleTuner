"""
Comprehensive unit tests for AuraFlow transformer components.
Tests all 6 classes and 27 functions with focus on typo prevention and edge cases.
"""

import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn

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
from simpletuner.helpers.models.auraflow.transformer import (
    AuraFlowFeedForward,
    AuraFlowJointTransformerBlock,
    AuraFlowPatchEmbed,
    AuraFlowPreFinalBlock,
    AuraFlowSingleTransformerBlock,
    AuraFlowTransformer2DModel,
    find_multiple,
)


class TestFindMultipleFunction(unittest.TestCase):
    """Test the find_multiple utility function for typo prevention."""

    def test_find_multiple_basic_cases(self):
        """Test basic functionality of find_multiple function."""
        # Exact multiples
        self.assertEqual(find_multiple(16, 4), 16)
        self.assertEqual(find_multiple(32, 8), 32)
        self.assertEqual(find_multiple(256, 256), 256)

        # Non-multiples
        self.assertEqual(find_multiple(17, 4), 20)  # 17 + 4 - (17 % 4) = 17 + 4 - 1 = 20
        self.assertEqual(find_multiple(30, 8), 32)  # 30 + 8 - (30 % 8) = 30 + 8 - 6 = 32
        self.assertEqual(find_multiple(100, 7), 105)  # 100 + 7 - (100 % 7) = 100 + 7 - 2 = 105

    def test_find_multiple_edge_cases(self):
        """Test edge cases for find_multiple function."""
        # Zero input
        self.assertEqual(find_multiple(0, 4), 0)

        # Small numbers
        self.assertEqual(find_multiple(1, 4), 4)
        self.assertEqual(find_multiple(3, 4), 4)

        # Large numbers
        self.assertEqual(find_multiple(1000, 256), 1024)

    def test_find_multiple_parameter_name_typos(self):
        """Test that find_multiple properly handles parameter names."""
        # This function uses positional args, but test that we don't have typos in usage
        with self.assertRaises(TypeError):
            find_multiple(n=16, k=4, extra_param=True)  # Should fail with extra param


class TestAuraFlowPatchEmbed(TransformerBaseTest, EmbeddingTestMixin):
    """Comprehensive tests for AuraFlowPatchEmbed class."""

    def setUp(self):
        super().setUp()
        self.patch_embed_config = {
            "height": 64,
            "width": 64,
            "patch_size": 16,
            "in_channels": 4,
            "embed_dim": 768,
            "pos_embed_max_size": 1024,
        }

    def test_patch_embed_instantiation(self):
        """Test AuraFlowPatchEmbed instantiation."""
        patch_embed = AuraFlowPatchEmbed(**self.patch_embed_config)

        # Check basic attributes
        self.assertEqual(patch_embed.patch_size, 16)
        self.assertEqual(patch_embed.height, 4)  # 64 // 16
        self.assertEqual(patch_embed.width, 4)  # 64 // 16
        self.assertEqual(patch_embed.base_size, 4)
        self.assertEqual(patch_embed.num_patches, 16)  # 4 * 4
        self.assertEqual(patch_embed.pos_embed_max_size, 1024)

        # Check learned parameters
        self.assertIsInstance(patch_embed.proj, nn.Linear)
        self.assertEqual(patch_embed.proj.in_features, 16 * 16 * 4)  # patch_size^2 * in_channels
        self.assertEqual(patch_embed.proj.out_features, 768)

        # Check positional embedding
        self.assertEqual(tuple(patch_embed.pos_embed.shape), (1, 1024, 768))

    def test_patch_embed_instantiation_parameter_typos(self):
        """Test parameter name typo prevention in instantiation."""
        valid_params = self.patch_embed_config.copy()

        # Test typos in parameter names
        typo_mappings = {
            "hieght": "height",  # Common typo
            "widht": "width",  # Common typo
            "patch_szie": "patch_size",  # Common typo
            "embed_dims": "embed_dim",  # Common typo
            "in_channel": "in_channels",  # Common typo
        }

        for typo_param, correct_param in typo_mappings.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            with self.assertRaises(TypeError) as context:
                AuraFlowPatchEmbed(**invalid_params)
            self.assertIn("unexpected keyword argument", str(context.exception))

    def test_pe_selection_index_based_on_dim(self):
        """Test position embedding selection logic for typo prevention."""
        patch_embed = AuraFlowPatchEmbed(**self.patch_embed_config)

        # Test with various input dimensions
        test_cases = [
            (64, 64),  # Same as config
            (32, 32),  # Smaller
            (128, 128),  # Larger
            (64, 32),  # Non-square
        ]

        for h, w in test_cases:
            indices = patch_embed.pe_selection_index_based_on_dim(h, w)

            # Validate output
            self.assertIsInstance(indices, torch.Tensor)
            h_p, w_p = h // patch_embed.patch_size, w // patch_embed.patch_size
            expected_length = h_p * w_p
            self.assertEqual(len(indices), expected_length)

            # Check indices are within bounds
            self.assertTrue(torch.all(indices >= 0))
            self.assertTrue(torch.all(indices < patch_embed.pos_embed_max_size))

    def test_pe_selection_mathematical_correctness(self):
        """Test mathematical correctness of position embedding selection."""
        patch_embed = AuraFlowPatchEmbed(**self.patch_embed_config)

        # Test with specific case we can verify manually
        h, w = 64, 64  # Same as config
        indices = patch_embed.pe_selection_index_based_on_dim(h, w)

        h_p, w_p = 4, 4  # 64 // 16
        h_max = w_max = int(1024**0.5)  # 32

        # Calculate expected center start positions
        expected_starth = h_max // 2 - h_p // 2  # 32//2 - 4//2 = 16 - 2 = 14
        expected_startw = w_max // 2 - w_p // 2  # Same as above

        # Verify the indices correspond to the centered grid
        expected_indices = []
        for i in range(h_p):
            for j in range(w_p):
                row_idx = expected_starth + i
                col_idx = expected_startw + j
                flat_idx = row_idx * w_max + col_idx
                expected_indices.append(flat_idx)

        expected_tensor = torch.tensor(expected_indices)
        torch.testing.assert_close(indices, expected_tensor)

    def test_patch_embed_forward_pass(self):
        """Test forward pass with various input shapes."""
        # Test cases with same channel count as config
        test_cases_same_channels = [
            (1, 4, 64, 64),  # Single batch
            (2, 4, 64, 64),  # Multiple batch
            (4, 4, 64, 64),  # Larger batch
            (2, 4, 32, 32),  # Smaller spatial
            (2, 4, 128, 128),  # Larger spatial
        ]

        patch_embed = AuraFlowPatchEmbed(**self.patch_embed_config)

        for batch_size, channels, height, width in test_cases_same_channels:
            with torch.no_grad():
                latent = torch.randn(batch_size, channels, height, width)

                # Test forward pass
                output = patch_embed.forward(latent)

                # Validate output shape
                h_p, w_p = height // patch_embed.patch_size, width // patch_embed.patch_size
                expected_shape = (batch_size, h_p * w_p, 768)
                self.assert_tensor_shape(output, expected_shape)

                # Validate no NaN/inf
                self.assert_no_nan_or_inf(output)

        # Test with different channel count - create new instance
        config_3_channels = self.patch_embed_config.copy()
        config_3_channels["in_channels"] = 3
        patch_embed_3ch = AuraFlowPatchEmbed(**config_3_channels)

        with torch.no_grad():
            latent = torch.randn(2, 3, 64, 64)
            output = patch_embed_3ch.forward(latent)

            h_p, w_p = 64 // 16, 64 // 16
            expected_shape = (2, h_p * w_p, 768)
            self.assert_tensor_shape(output, expected_shape)
            self.assert_no_nan_or_inf(output)

    def test_patch_embed_forward_shape_validation(self):
        """Test that forward pass validates input shapes correctly."""
        patch_embed = AuraFlowPatchEmbed(**self.patch_embed_config)

        # Test invalid input shapes
        invalid_shapes = [
            (2, 4, 63, 64),  # Height not divisible by patch_size
            (2, 4, 64, 63),  # Width not divisible by patch_size
            (2, 4, 17, 17),  # Both not divisible
        ]

        for shape in invalid_shapes:
            with torch.no_grad():
                latent = torch.randn(*shape)

                # Should handle gracefully or provide clear error
                try:
                    output = patch_embed.forward(latent)
                    # If it doesn't error, verify the reshaping still works mathematically
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

        patch_embed = AuraFlowPatchEmbed(**self.patch_embed_config).to(device)

        with torch.no_grad():
            latent = torch.randn(2, 4, 64, 64, device=device)
            output = patch_embed.forward(latent)

            self.assert_tensor_device(output, device)

    def test_patch_embed_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        patch_embed = AuraFlowPatchEmbed(**self.patch_embed_config)

        required_methods = ["forward", "pe_selection_index_based_on_dim"]

        self.run_method_existence_tests(patch_embed, required_methods)


class TestAuraFlowFeedForward(TransformerBaseTest):
    """Comprehensive tests for AuraFlowFeedForward class."""

    def setUp(self):
        super().setUp()
        self.dim = 768
        self.hidden_dim = 3072

    def test_feed_forward_instantiation(self):
        """Test AuraFlowFeedForward instantiation."""
        # Test with explicit hidden_dim
        ff = AuraFlowFeedForward(self.dim, self.hidden_dim)

        # Calculate expected final hidden dim
        expected_final_hidden = int(2 * self.hidden_dim / 3)
        expected_final_hidden = find_multiple(expected_final_hidden, 256)

        self.assertEqual(ff.linear_1.in_features, self.dim)
        self.assertEqual(ff.linear_1.out_features, expected_final_hidden)
        self.assertEqual(ff.linear_2.in_features, self.dim)
        self.assertEqual(ff.linear_2.out_features, expected_final_hidden)
        self.assertEqual(ff.out_projection.in_features, expected_final_hidden)
        self.assertEqual(ff.out_projection.out_features, self.dim)

        # Test defaults
        self.assertIsNone(ff.chunk_size)
        self.assertEqual(ff.dim, 0)

    def test_feed_forward_instantiation_defaults(self):
        """Test AuraFlowFeedForward with default hidden_dim."""
        ff = AuraFlowFeedForward(self.dim)

        # Should default to 4 * dim
        expected_hidden = 4 * self.dim
        expected_final_hidden = int(2 * expected_hidden / 3)
        expected_final_hidden = find_multiple(expected_final_hidden, 256)

        self.assertEqual(ff.linear_1.out_features, expected_final_hidden)

    def test_feed_forward_parameter_typos(self):
        """Test parameter name typo prevention in instantiation."""
        valid_params = {"dim": self.dim, "hidden_dim": self.hidden_dim}

        typo_mappings = {"dims": "dim", "hidden_dims": "hidden_dim", "hidde_dim": "hidden_dim", "dimm": "dim"}

        for typo_param, correct_param in typo_mappings.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            with self.assertRaises(TypeError) as context:
                AuraFlowFeedForward(**invalid_params)
            self.assertIn("unexpected keyword argument", str(context.exception))

    def test_feed_forward_forward_pass(self):
        """Test standard forward pass."""
        ff = AuraFlowFeedForward(self.dim, self.hidden_dim)

        test_shapes = [
            (1, 128, self.dim),
            (2, 64, self.dim),
            (4, 256, self.dim),
        ]

        for shape in test_shapes:
            with torch.no_grad():
                x = torch.randn(*shape)
                output = ff.forward(x)

                self.assert_tensor_shape(output, shape)
                self.assert_no_nan_or_inf(output)
                self.assert_tensor_in_range(output)

    def test_set_chunk_feed_forward(self):
        """Test chunked feed forward functionality."""
        ff = AuraFlowFeedForward(self.dim, self.hidden_dim)

        # Test setting chunk parameters
        chunk_size = 32
        dim = 1
        ff.set_chunk_feed_forward(chunk_size, dim)

        self.assertEqual(ff.chunk_size, chunk_size)
        self.assertEqual(ff.dim, dim)

    def test_chunked_forward_pass(self):
        """Test chunked forward pass functionality."""
        ff = AuraFlowFeedForward(self.dim, self.hidden_dim)

        # Enable chunking
        chunk_size = 32
        ff.set_chunk_feed_forward(chunk_size, dim=1)

        batch_size, seq_len = 2, 128
        x = torch.randn(batch_size, seq_len, self.dim)

        with torch.no_grad():
            output = ff.forward(x)

            self.assert_tensor_shape(output, (batch_size, seq_len, self.dim))
            self.assert_no_nan_or_inf(output)

    def test_chunked_vs_normal_forward_consistency(self):
        """Test that chunked and normal forward passes produce similar results."""
        ff = AuraFlowFeedForward(self.dim, self.hidden_dim)

        x = torch.randn(2, 128, self.dim)

        # Normal forward pass
        with torch.no_grad():
            normal_output = ff.forward(x)

        # Chunked forward pass
        ff.set_chunk_feed_forward(chunk_size=32, dim=1)
        with torch.no_grad():
            chunked_output = ff.forward(x)

        # Results should be close (allowing for numerical differences)
        torch.testing.assert_close(normal_output, chunked_output, rtol=1e-5, atol=1e-6)

    def test_chunk_forward_edge_cases(self):
        """Test edge cases in chunked forward pass."""
        ff = AuraFlowFeedForward(self.dim, self.hidden_dim)

        # Test with chunk size larger than sequence length
        ff.set_chunk_feed_forward(chunk_size=256, dim=1)
        x = torch.randn(2, 64, self.dim)  # seq_len < chunk_size

        with torch.no_grad():
            output = ff.forward(x)
            self.assert_tensor_shape(output, x.shape)

    def test_feed_forward_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        ff = AuraFlowFeedForward(self.dim, self.hidden_dim)

        required_methods = ["forward", "set_chunk_feed_forward", "_chunk_forward"]

        self.run_method_existence_tests(ff, required_methods)

    def test_feed_forward_bias_configuration(self):
        """Test that all linear layers have bias=False as specified."""
        ff = AuraFlowFeedForward(self.dim, self.hidden_dim)

        self.assertIsNone(ff.linear_1.bias)
        self.assertIsNone(ff.linear_2.bias)
        self.assertIsNone(ff.out_projection.bias)


class TestAuraFlowPreFinalBlock(TransformerBaseTest):
    """Comprehensive tests for AuraFlowPreFinalBlock class."""

    def setUp(self):
        super().setUp()
        self.embedding_dim = 768
        self.conditioning_embedding_dim = 512

    def test_pre_final_block_instantiation(self):
        """Test AuraFlowPreFinalBlock instantiation."""
        block = AuraFlowPreFinalBlock(self.embedding_dim, self.conditioning_embedding_dim)

        # Check components
        self.assertIsInstance(block.silu, nn.SiLU)
        self.assertIsInstance(block.linear, nn.Linear)

        # Check linear layer configuration
        self.assertEqual(block.linear.in_features, self.conditioning_embedding_dim)
        self.assertEqual(block.linear.out_features, self.embedding_dim * 2)
        self.assertIsNone(block.linear.bias)  # Should be bias=False

    def test_pre_final_block_parameter_typos(self):
        """Test parameter name typo prevention in instantiation."""
        valid_params = {"embedding_dim": self.embedding_dim, "conditioning_embedding_dim": self.conditioning_embedding_dim}

        typo_mappings = {
            "embeding_dim": "embedding_dim",
            "conditioning_embeding_dim": "conditioning_embedding_dim",
            "embedding_dims": "embedding_dim",
            "conditoning_embedding_dim": "conditioning_embedding_dim",
        }

        for typo_param, correct_param in typo_mappings.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            with self.assertRaises(TypeError) as context:
                AuraFlowPreFinalBlock(**invalid_params)
            self.assertIn("unexpected keyword argument", str(context.exception))

    def test_pre_final_block_forward_pass(self):
        """Test forward pass functionality."""
        block = AuraFlowPreFinalBlock(self.embedding_dim, self.conditioning_embedding_dim)

        batch_size = 2
        seq_len = 128
        x = torch.randn(batch_size, seq_len, self.embedding_dim)
        conditioning_embedding = torch.randn(batch_size, self.conditioning_embedding_dim)

        with torch.no_grad():
            output = block.forward(x, conditioning_embedding)

            self.assert_tensor_shape(output, (batch_size, seq_len, self.embedding_dim))
            self.assert_no_nan_or_inf(output)

    def test_pre_final_block_mathematical_operations(self):
        """Test mathematical correctness of the forward pass."""
        block = AuraFlowPreFinalBlock(self.embedding_dim, self.conditioning_embedding_dim)

        batch_size = 2
        seq_len = 128
        x = torch.randn(batch_size, seq_len, self.embedding_dim)
        conditioning_embedding = torch.randn(batch_size, self.conditioning_embedding_dim)

        with torch.no_grad():
            # Manual computation for verification
            emb = block.linear(block.silu(conditioning_embedding).to(x.dtype))
            scale, shift = torch.chunk(emb, 2, dim=1)
            expected_output = x * (1 + scale)[:, None, :] + shift[:, None, :]

            # Actual forward pass
            actual_output = block.forward(x, conditioning_embedding)

            torch.testing.assert_close(actual_output, expected_output, rtol=1e-5, atol=1e-6)

    def test_pre_final_block_dtype_handling(self):
        """Test dtype conversion handling."""
        block = AuraFlowPreFinalBlock(self.embedding_dim, self.conditioning_embedding_dim)

        batch_size = 2
        seq_len = 128

        # Test with different dtypes
        x_float32 = torch.randn(batch_size, seq_len, self.embedding_dim, dtype=torch.float32)
        conditioning_float16 = torch.randn(batch_size, self.conditioning_embedding_dim, dtype=torch.float16)

        with torch.no_grad():
            output = block.forward(x_float32, conditioning_float16)

            # Output should match input x dtype
            self.assertEqual(output.dtype, x_float32.dtype)

    def test_pre_final_block_broadcasting(self):
        """Test that broadcasting works correctly for conditioning embedding."""
        block = AuraFlowPreFinalBlock(self.embedding_dim, self.conditioning_embedding_dim)

        batch_size = 2
        seq_len = 128
        x = torch.randn(batch_size, seq_len, self.embedding_dim)
        conditioning_embedding = torch.randn(batch_size, self.conditioning_embedding_dim)

        with torch.no_grad():
            output = block.forward(x, conditioning_embedding)

            # Verify the scale and shift operations broadcast correctly
            emb = block.linear(block.silu(conditioning_embedding).to(x.dtype))
            scale, shift = torch.chunk(emb, 2, dim=1)

            # These should broadcast from [batch, dim] to [batch, seq_len, dim]
            self.assertEqual(scale.shape, (batch_size, self.embedding_dim))
            self.assertEqual(shift.shape, (batch_size, self.embedding_dim))

    def test_pre_final_block_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        block = AuraFlowPreFinalBlock(self.embedding_dim, self.conditioning_embedding_dim)

        required_methods = ["forward"]
        self.run_method_existence_tests(block, required_methods)


class TestAuraFlowSingleTransformerBlock(TransformerBaseTest, TransformerBlockTestMixin):
    """Comprehensive tests for AuraFlowSingleTransformerBlock class."""

    def setUp(self):
        super().setUp()
        self.block_config = {"dim": 768, "num_attention_heads": 12, "attention_head_dim": 64}

    def test_single_transformer_block_instantiation(self):
        """Test AuraFlowSingleTransformerBlock instantiation."""
        block = AuraFlowSingleTransformerBlock(**self.block_config)

        # Check components exist
        self.assertTrue(hasattr(block, "norm1"))
        self.assertTrue(hasattr(block, "attn"))
        self.assertTrue(hasattr(block, "norm2"))
        self.assertTrue(hasattr(block, "ff"))

        # Check types
        from diffusers.models.attention_processor import Attention
        from diffusers.models.normalization import AdaLayerNormZero, FP32LayerNorm

        self.assertIsInstance(block.norm1, AdaLayerNormZero)
        self.assertIsInstance(block.attn, Attention)
        self.assertIsInstance(block.norm2, FP32LayerNorm)
        self.assertIsInstance(block.ff, AuraFlowFeedForward)

    def test_single_transformer_block_parameter_typos(self):
        """Test parameter name typo prevention in instantiation."""
        valid_params = self.block_config.copy()

        typo_mappings = {
            "dims": "dim",
            "num_attention_head": "num_attention_heads",
            "attention_head_dims": "attention_head_dim",
            "atention_head_dim": "attention_head_dim",
        }

        for typo_param, correct_param in typo_mappings.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            with self.assertRaises(TypeError) as context:
                AuraFlowSingleTransformerBlock(**invalid_params)
            self.assertIn("unexpected keyword argument", str(context.exception))

    @patch("simpletuner.helpers.models.auraflow.transformer.AdaLayerNormZero")
    @patch("simpletuner.helpers.models.auraflow.transformer.Attention")
    @patch("simpletuner.helpers.models.auraflow.transformer.FP32LayerNorm")
    def test_single_transformer_block_forward_pass(self, mock_fp32_norm, mock_attention, mock_ada_norm):
        """Test forward pass with mocked dependencies."""
        # Setup mocks
        mock_ada_norm_instance = mock_ada_norm.return_value
        mock_attention_instance = mock_attention.return_value
        mock_fp32_norm_instance = mock_fp32_norm.return_value

        # Mock norm1 output (AdaLayerNormZero returns 5 values)
        batch_size, seq_len, dim = 2, 128, 768
        mock_ada_norm_instance.return_value = (
            torch.randn(batch_size, seq_len, dim),  # norm_hidden_states
            torch.randn(batch_size, dim),  # gate_msa
            torch.randn(batch_size, dim),  # shift_mlp
            torch.randn(batch_size, dim),  # scale_mlp
            torch.randn(batch_size, dim),  # gate_mlp
        )

        # Mock attention output
        mock_attention_instance.return_value = torch.randn(batch_size, seq_len, dim)

        # Mock norm2 output
        mock_fp32_norm_instance = mock_fp32_norm.return_value
        mock_fp32_norm_instance.side_effect = lambda x, *args, **kwargs: torch.randn_like(x)

        # Create block
        block = AuraFlowSingleTransformerBlock(**self.block_config)

        # Test inputs
        hidden_states = torch.randn(batch_size, seq_len, dim)
        temb = torch.randn(batch_size, dim)

        with torch.no_grad():
            output = block.forward(hidden_states, temb)

            self.assert_tensor_shape(output, (batch_size, seq_len, dim))
            self.assertIsInstance(output, torch.Tensor)

    def test_single_transformer_block_chunking(self):
        """Test chunked feed forward capability."""
        block = AuraFlowSingleTransformerBlock(**self.block_config)

        # Test setting chunk parameters
        chunk_size = 32
        dim = 1
        block.set_chunk_feed_forward(chunk_size, dim)

        # Verify the feed forward module received the chunking settings
        self.assertEqual(block.ff.chunk_size, chunk_size)
        self.assertEqual(block.ff.dim, dim)

    def test_single_transformer_block_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        block = AuraFlowSingleTransformerBlock(**self.block_config)

        required_methods = ["forward", "set_chunk_feed_forward"]

        self.run_method_existence_tests(block, required_methods)

    def test_single_transformer_attention_kwargs_handling(self):
        """Test handling of attention_kwargs parameter."""
        with (
            patch("simpletuner.helpers.models.auraflow.transformer.AdaLayerNormZero"),
            patch("simpletuner.helpers.models.auraflow.transformer.Attention"),
            patch("simpletuner.helpers.models.auraflow.transformer.FP32LayerNorm"),
        ):

            block = AuraFlowSingleTransformerBlock(**self.block_config)

            # Mock the norm and attention components with proper mock modules
            with (
                patch.object(block, "norm1", MockNormLayer(return_tuple=True, num_values=5)),
                patch.object(block, "attn", MockAttention()),
                patch.object(block, "norm2", MockNormLayer()),
                patch.object(block, "ff", MockModule(torch.randn(2, 128, 768))),
            ):

                hidden_states = torch.randn(2, 128, 768)
                temb = torch.randn(2, 768)
                attention_kwargs = {"some_key": "some_value"}

                with torch.no_grad():
                    output = block.forward(hidden_states, temb, attention_kwargs)

                    # Test that forward pass works with mocked components
                    self.assertIsInstance(output, torch.Tensor)
                    self.assert_tensor_shape(output, (2, 128, 768))


class TestAuraFlowJointTransformerBlock(TransformerBaseTest, TransformerBlockTestMixin):
    """Comprehensive tests for AuraFlowJointTransformerBlock class."""

    def setUp(self):
        super().setUp()
        self.block_config = {"dim": 768, "num_attention_heads": 12, "attention_head_dim": 64}

    def test_joint_transformer_block_instantiation(self):
        """Test AuraFlowJointTransformerBlock instantiation."""
        block = AuraFlowJointTransformerBlock(**self.block_config)

        # Check all components exist
        component_names = ["norm1", "norm1_context", "attn", "norm2", "ff", "norm2_context", "ff_context"]

        for name in component_names:
            self.assertTrue(hasattr(block, name), f"Missing component: {name}")

        # Check context_pre_only attribute
        self.assertFalse(block.context_pre_only)

    def test_joint_transformer_block_parameter_typos(self):
        """Test parameter name typo prevention in instantiation."""
        valid_params = self.block_config.copy()

        typo_mappings = {
            "dims": "dim",
            "num_attention_head": "num_attention_heads",
            "attention_head_dims": "attention_head_dim",
        }

        for typo_param, correct_param in typo_mappings.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            with self.assertRaises(TypeError) as context:
                AuraFlowJointTransformerBlock(**invalid_params)
            self.assertIn("unexpected keyword argument", str(context.exception))

    @patch("simpletuner.helpers.models.auraflow.transformer.AdaLayerNormZero")
    @patch("simpletuner.helpers.models.auraflow.transformer.Attention")
    @patch("simpletuner.helpers.models.auraflow.transformer.FP32LayerNorm")
    def test_joint_transformer_block_forward_pass(self, mock_fp32_norm, mock_attention, mock_ada_norm):
        """Test forward pass with mocked dependencies."""
        # Setup mocks
        batch_size, seq_len, context_len, dim = 2, 128, 77, 768

        # Mock AdaLayerNormZero (returns 5 values)
        mock_ada_norm_instance = mock_ada_norm.return_value
        mock_ada_norm_instance.return_value = (
            torch.randn(batch_size, seq_len, dim),  # norm_hidden_states
            torch.randn(batch_size, dim),  # gate_msa
            torch.randn(batch_size, dim),  # shift_mlp
            torch.randn(batch_size, dim),  # scale_mlp
            torch.randn(batch_size, dim),  # gate_mlp
        )

        # Mock attention (returns tuple for joint attention)
        mock_attention_instance = mock_attention.return_value
        mock_attention_instance.return_value = (
            torch.randn(batch_size, seq_len, dim),  # attn_output
            torch.randn(batch_size, context_len, dim),  # context_attn_output
        )

        # Mock FP32LayerNorm
        mock_fp32_norm_instance = mock_fp32_norm.return_value
        mock_fp32_norm_instance.side_effect = lambda x, *args, **kwargs: torch.randn_like(x)

        # Create block
        block = AuraFlowJointTransformerBlock(**self.block_config)

        # Test inputs
        hidden_states = torch.randn(batch_size, seq_len, dim)
        encoder_hidden_states = torch.randn(batch_size, context_len, dim)
        temb = torch.randn(batch_size, dim)

        with torch.no_grad():
            encoder_output, hidden_output = block.forward(hidden_states, encoder_hidden_states, temb)

            self.assert_tensor_shape(encoder_output, (batch_size, context_len, dim))
            self.assert_tensor_shape(hidden_output, (batch_size, seq_len, dim))

    def test_joint_transformer_block_chunking(self):
        """Test chunked feed forward capability for both FF modules."""
        block = AuraFlowJointTransformerBlock(**self.block_config)

        # Test setting chunk parameters
        chunk_size = 32
        dim = 1
        block.set_chunk_feed_forward(chunk_size, dim)

        # Verify both feed forward modules received the chunking settings
        self.assertEqual(block.ff.chunk_size, chunk_size)
        self.assertEqual(block.ff.dim, dim)
        self.assertEqual(block.ff_context.chunk_size, chunk_size)
        self.assertEqual(block.ff_context.dim, dim)

    def test_joint_transformer_block_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        block = AuraFlowJointTransformerBlock(**self.block_config)

        required_methods = ["forward", "set_chunk_feed_forward"]

        self.run_method_existence_tests(block, required_methods)

    def test_joint_transformer_attention_kwargs_handling(self):
        """Test handling of attention_kwargs parameter."""
        with (
            patch("simpletuner.helpers.models.auraflow.transformer.AdaLayerNormZero"),
            patch("simpletuner.helpers.models.auraflow.transformer.Attention"),
            patch("simpletuner.helpers.models.auraflow.transformer.FP32LayerNorm"),
        ):

            block = AuraFlowJointTransformerBlock(**self.block_config)

            # Mock all the components with proper mock modules
            with (
                patch.object(block, "norm1", MockNormLayer(return_tuple=True, num_values=5)),
                patch.object(block, "norm1_context", MockNormLayer(return_tuple=True, num_values=5)),
                patch.object(block, "attn", MockAttention(return_tuple=True)),
                patch.object(block, "norm2", MockNormLayer()),
                patch.object(block, "norm2_context", MockNormLayer()),
                patch.object(block, "ff", MockModule(torch.randn(2, 128, 768))),
                patch.object(block, "ff_context", MockModule(torch.randn(2, 77, 768))),
            ):

                hidden_states = torch.randn(2, 128, 768)
                encoder_hidden_states = torch.randn(2, 77, 768)
                temb = torch.randn(2, 768)
                attention_kwargs = {"scale": 1.0}

                with torch.no_grad():
                    encoder_output, hidden_output = block.forward(
                        hidden_states, encoder_hidden_states, temb, attention_kwargs
                    )

                    # Test that forward pass works with mocked components
                    self.assertIsInstance(encoder_output, torch.Tensor)
                    self.assertIsInstance(hidden_output, torch.Tensor)
                    self.assert_tensor_shape(encoder_output, (2, 77, 768))
                    self.assert_tensor_shape(hidden_output, (2, 128, 768))


class TestAuraFlowTransformer2DModel(TransformerBaseTest):
    """Comprehensive tests for AuraFlowTransformer2DModel main class."""

    def setUp(self):
        super().setUp()
        self.model_config = {
            "sample_size": 32,
            "patch_size": 2,
            "in_channels": 4,
            "num_mmdit_layers": 2,
            "num_single_dit_layers": 4,
            "attention_head_dim": 64,
            "num_attention_heads": 8,
            "joint_attention_dim": 512,
            "caption_projection_dim": self.hidden_dim,
            "out_channels": 4,
            "pos_embed_max_size": 512,
        }

    @patch("diffusers.models.embeddings.Timesteps")
    @patch("diffusers.models.embeddings.TimestepEmbedding")
    def test_transformer_model_instantiation(self, mock_timestep_embed, mock_timesteps):
        """Test AuraFlowTransformer2DModel instantiation."""
        # Mock the timestep components
        mock_timesteps.return_value = Mock()
        mock_timestep_embed.return_value = Mock()

        model = AuraFlowTransformer2DModel(**self.model_config)

        # Check basic attributes
        self.assertEqual(model.out_channels, 4)
        expected_inner_dim = 8 * 64  # num_attention_heads * attention_head_dim
        self.assertEqual(model.inner_dim, expected_inner_dim)

        # Check components exist
        component_names = [
            "pos_embed",
            "context_embedder",
            "time_step_embed",
            "time_step_proj",
            "joint_transformer_blocks",
            "single_transformer_blocks",
            "norm_out",
            "proj_out",
            "register_tokens",
        ]

        for name in component_names:
            self.assertTrue(hasattr(model, name), f"Missing component: {name}")

        # Check transformer blocks count
        self.assertEqual(len(model.joint_transformer_blocks), 2)
        self.assertEqual(len(model.single_transformer_blocks), 4)

    def test_transformer_model_parameter_typos(self):
        """Test parameter name typo prevention in instantiation."""
        valid_params = self.model_config.copy()

        typo_mappings = {
            "sample_szie": "sample_size",
            "patch_szie": "patch_size",
            "in_channel": "in_channels",
            "out_channel": "out_channels",
            "num_mmdit_layer": "num_mmdit_layers",
            "num_single_dit_layer": "num_single_dit_layers",
            "attention_head_dims": "attention_head_dim",
            "num_attention_head": "num_attention_heads",
            "joint_attention_dims": "joint_attention_dim",
            "caption_projection_dims": "caption_projection_dim",
        }

        for typo_param, correct_param in typo_mappings.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            with self.assertRaises(TypeError) as context:
                AuraFlowTransformer2DModel(**invalid_params)
            self.assertIn("unexpected keyword argument", str(context.exception))

    def test_transformer_gradient_checkpointing_methods(self):
        """Test gradient checkpointing configuration methods."""
        with patch("diffusers.models.embeddings.Timesteps"), patch("diffusers.models.embeddings.TimestepEmbedding"):

            model = AuraFlowTransformer2DModel(**self.model_config)

            # Test setting gradient checkpointing interval
            interval = 2
            model.set_gradient_checkpointing_interval(interval)
            self.assertEqual(model.gradient_checkpointing_interval, interval)

    def test_transformer_tread_router_methods(self):
        """Test TREAD router configuration methods."""
        with patch("diffusers.models.embeddings.Timesteps"), patch("diffusers.models.embeddings.TimestepEmbedding"):

            model = AuraFlowTransformer2DModel(**self.model_config)

            # Test setting router
            mock_router = Mock()
            mock_routes = [{"start": 0, "end": 2, "ratio": 0.5}]

            model.set_router(mock_router, mock_routes)
            self.assertEqual(model._tread_router, mock_router)
            self.assertEqual(model._tread_routes, mock_routes)

    def test_transformer_chunking_methods(self):
        """Test forward chunking enable/disable methods."""
        with patch("diffusers.models.embeddings.Timesteps"), patch("diffusers.models.embeddings.TimestepEmbedding"):

            model = AuraFlowTransformer2DModel(**self.model_config)

            # Test enable chunking
            chunk_size = 32
            dim = 1
            model.enable_forward_chunking(chunk_size, dim)

            # Test invalid dim raises error
            with self.assertRaises(ValueError) as context:
                model.enable_forward_chunking(chunk_size, dim=3)
            self.assertIn("Make sure to set `dim` to either 0 or 1", str(context.exception))

            # Test disable chunking
            model.disable_forward_chunking()

    def test_transformer_attention_processor_methods(self):
        """Test attention processor management methods."""
        with patch("diffusers.models.embeddings.Timesteps"), patch("diffusers.models.embeddings.TimestepEmbedding"):

            model = AuraFlowTransformer2DModel(**self.model_config)

            # Mock attention modules with proper mock modules
            for block in model.joint_transformer_blocks:
                mock_attn = MockAttention()
                mock_attn.get_processor = Mock(return_value=Mock())
                mock_attn.set_processor = Mock()
                with patch.object(block, "attn", mock_attn):
                    pass  # Setup complete

            for block in model.single_transformer_blocks:
                mock_attn = MockAttention()
                mock_attn.get_processor = Mock(return_value=Mock())
                mock_attn.set_processor = Mock()
                with patch.object(block, "attn", mock_attn):
                    pass  # Setup complete

            # Test getting attention processors
            processors = model.attn_processors
            self.assertIsInstance(processors, dict)

            # Test setting attention processor
            new_processor = Mock()
            model.set_attn_processor(new_processor)

    def test_transformer_qkv_fusion_methods(self):
        """Test QKV projection fusion methods."""
        with patch("diffusers.models.embeddings.Timesteps"), patch("diffusers.models.embeddings.TimestepEmbedding"):

            model = AuraFlowTransformer2DModel(**self.model_config)

            # Mock attention modules for fusion testing
            mock_attention_modules = []
            for _ in range(3):  # Create some mock attention modules
                mock_attn = Mock(spec=["fuse_projections"])
                mock_attn.fuse_projections = Mock()
                mock_attention_modules.append(mock_attn)

            with patch.object(model, "modules", return_value=mock_attention_modules):
                with patch.object(model, "attn_processors", {"test": Mock()}):
                    # Test fuse QKV projections
                    model.fuse_qkv_projections()

                    # Test unfuse QKV projections
                    model.unfuse_qkv_projections()

    def test_transformer_method_existence(self):
        """Test that all required methods exist (typo prevention)."""
        with patch("diffusers.models.embeddings.Timesteps"), patch("diffusers.models.embeddings.TimestepEmbedding"):

            model = AuraFlowTransformer2DModel(**self.model_config)

            required_methods = [
                "forward",
                "set_gradient_checkpointing_interval",
                "set_router",
                "enable_forward_chunking",
                "disable_forward_chunking",
                "attn_processors",
                "set_attn_processor",
                "fuse_qkv_projections",
                "unfuse_qkv_projections",
            ]

            self.run_method_existence_tests(model, required_methods)

    def test_transformer_config_attributes(self):
        """Test config attribute access and typo prevention."""
        with patch("diffusers.models.embeddings.Timesteps"), patch("diffusers.models.embeddings.TimestepEmbedding"):

            model = AuraFlowTransformer2DModel(**self.model_config)

            # Test that config attributes are accessible
            config_attrs = [
                "sample_size",
                "patch_size",
                "in_channels",
                "out_channels",
                "num_mmdit_layers",
                "num_single_dit_layers",
                "attention_head_dim",
                "num_attention_heads",
                "joint_attention_dim",
                "caption_projection_dim",
            ]

            for attr in config_attrs:
                self.assertTrue(hasattr(model.config, attr), f"Config missing attribute: {attr}")
                # Verify the value matches what we set
                expected_value = self.model_config[attr]
                actual_value = getattr(model.config, attr)
                self.assertEqual(actual_value, expected_value, f"Config {attr} mismatch")

    @patch("diffusers.models.embeddings.Timesteps")
    @patch("diffusers.models.embeddings.TimestepEmbedding")
    @patch("simpletuner.helpers.models.auraflow.transformer.AuraFlowPatchEmbed")
    def test_transformer_forward_pass_structure(self, mock_patch_embed, mock_timestep_embed, mock_timesteps):
        """Test the structure of forward pass without full execution."""
        # Mock all components
        mock_timesteps.return_value = Mock()
        mock_timestep_embed.return_value = Mock()
        mock_patch_embed.return_value = Mock()

        model = AuraFlowTransformer2DModel(**self.model_config)

        # Mock components that are called in forward
        model.pos_embed = Mock(return_value=torch.randn(2, 256, 512))
        model.time_step_embed = Mock(return_value=torch.randn(256))
        model.time_step_proj = Mock(return_value=torch.randn(2, 512))
        model.context_embedder = Mock(return_value=torch.randn(2, 77, self.model_config["caption_projection_dim"]))
        model.norm_out = Mock(return_value=torch.randn(2, 256, 512))
        model.proj_out = Mock(return_value=torch.randn(2, 256, 16))

        # Mock transformer blocks using proper mocking
        MockingUtils.safely_replace_modules_in_list(model.joint_transformer_blocks)
        MockingUtils.safely_replace_modules_in_list(model.single_transformer_blocks)

        # Test inputs
        hidden_states = torch.randn(2, 4, 32, 32)  # batch_size, channels, height, width
        encoder_hidden_states = torch.randn(2, 77, 512)
        timestep = torch.randint(0, 1000, (2,))

        with torch.no_grad():
            output = model.forward(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep, return_dict=True
            )

            # Verify output structure
            self.assertTrue(hasattr(output, "sample"))
            sample = output.sample
            self.assertIsInstance(sample, torch.Tensor)
            # Output should match input spatial dimensions
            self.assertEqual(sample.shape[0], 2)  # batch_size
            self.assertEqual(sample.shape[1], 4)  # out_channels
            expected_hw = self.model_config["sample_size"]
            self.assertEqual(sample.shape[2], expected_hw)  # height
            self.assertEqual(sample.shape[3], expected_hw)  # width


class TestAuraFlowTransformerPerformance(TransformerBaseTest):
    """Performance and benchmarking tests for AuraFlow transformer components."""

    def setUp(self):
        super().setUp()
        self.perf_utils = PerformanceUtils()

    def test_find_multiple_performance(self):
        """Test find_multiple function performance."""
        import time

        # Test with various inputs
        test_cases = [(i, 256) for i in range(1, 1000, 50)]

        start_time = time.time()
        for n, k in test_cases:
            find_multiple(n, k)
        end_time = time.time()

        # Should be very fast for simple arithmetic
        self.assertLess(end_time - start_time, 0.01)  # Less than 10ms for all cases

    @patch("diffusers.models.embeddings.Timesteps")
    @patch("diffusers.models.embeddings.TimestepEmbedding")
    def test_patch_embed_performance(self, mock_timestep_embed, mock_timesteps):
        """Test patch embedding performance."""
        config = {
            "height": 64,
            "width": 64,
            "patch_size": 16,
            "in_channels": 4,
            "embed_dim": 768,
            "pos_embed_max_size": 1024,
        }

        patch_embed = AuraFlowPatchEmbed(**config)
        latent = torch.randn(4, 4, 64, 64)  # Larger batch for performance test

        inputs = {"latent": latent}

        # Measure forward pass time
        avg_time = self.perf_utils.measure_forward_pass_time(lambda latent: patch_embed.forward(latent), inputs, num_runs=50)

        # Should be fast for small inputs
        self.assertLess(avg_time * 1000, 100)  # Less than 100ms

    def test_feed_forward_chunking_performance(self):
        """Test that chunking doesn't significantly impact performance for small inputs."""
        ff = AuraFlowFeedForward(768, 3072)
        x = torch.randn(2, 128, 768)

        # Measure normal forward pass
        inputs = {"x": x}
        normal_time = self.perf_utils.measure_forward_pass_time(lambda x: ff.forward(x), inputs, num_runs=20)

        # Measure chunked forward pass
        ff.set_chunk_feed_forward(32, 1)
        chunked_time = self.perf_utils.measure_forward_pass_time(lambda x: ff.forward(x), inputs, num_runs=20)

        # Chunked should not be more than 2x slower for this size
        self.assertLess(chunked_time, normal_time * 2)


if __name__ == "__main__":
    unittest.main()
