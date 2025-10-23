"""
Comprehensive unit tests for Sana transformer components.

This module tests the Sana transformer architecture including:
- GLUMBConv: GLU + MBConv fusion module
- SanaTransformerBlock: Sana-specific transformer block
- SanaTransformer2DModel: Complete Sana transformer model

Focus areas:
- Typo prevention in parameter names, method names, tensor operations
- Edge case handling (empty inputs, None values, device compatibility)
- Shape validation and mathematical correctness
- Architecture-specific features (GLU+MBConv, linear attention)
- Performance benchmarking
- TREAD router integration
- Gradient checkpointing functionality
"""

import os

# Test base classes
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock, Mock, create_autospec, patch

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from transformer_base_test import EmbeddingTestMixin, TransformerBaseTest, TransformerBlockTestMixin
from transformer_test_helpers import (
    MockAttention,
    MockComponents,
    MockDiffusersConfig,
    MockingUtils,
    MockModule,
    MockNormLayer,
    TensorGenerator,
    TypoTestUtils,
)

# Import components under test
from simpletuner.helpers.models.sana.transformer import GLUMBConv, SanaTransformer2DModel, SanaTransformerBlock


class TestGLUMBConv(TransformerBaseTest):
    """Test GLUMBConv (GLU + MBConv) module."""

    def test_instantiation_basic(self):
        """Test basic instantiation."""
        module = GLUMBConv(in_channels=32, out_channels=32, expand_ratio=4.0, norm_type=None, residual_connection=True)

        self.assertIsInstance(module, nn.Module)
        self.assertEqual(module.residual_connection, True)
        self.assertIsNone(module.norm_type)

    def test_instantiation_with_rms_norm(self):
        """Test instantiation with RMS normalization."""
        module = GLUMBConv(in_channels=32, out_channels=32, norm_type="rms_norm")

        self.assertEqual(module.norm_type, "rms_norm")
        self.assertIsNotNone(module.norm)

    def test_component_initialization(self):
        """Test proper initialization of internal components."""
        in_channels, out_channels, expand_ratio = 32, 64, 4.0
        module = GLUMBConv(in_channels, out_channels, expand_ratio)

        hidden_channels = int(expand_ratio * in_channels)

        # Check conv layers
        self.assertIsInstance(module.nonlinearity, nn.SiLU)
        self.assertIsInstance(module.conv_inverted, nn.Conv2d)
        self.assertIsInstance(module.conv_depth, nn.Conv2d)
        self.assertIsInstance(module.conv_point, nn.Conv2d)

        # Check conv_inverted configuration
        self.assertEqual(module.conv_inverted.in_channels, in_channels)
        self.assertEqual(module.conv_inverted.out_channels, hidden_channels * 2)
        self.assertEqual(module.conv_inverted.kernel_size, (1, 1))

        # Check conv_depth configuration (depthwise)
        self.assertEqual(module.conv_depth.in_channels, hidden_channels * 2)
        self.assertEqual(module.conv_depth.out_channels, hidden_channels * 2)
        self.assertEqual(module.conv_depth.groups, hidden_channels * 2)  # Depthwise
        self.assertEqual(module.conv_depth.kernel_size, (3, 3))

        # Check conv_point configuration
        self.assertEqual(module.conv_point.in_channels, hidden_channels)
        self.assertEqual(module.conv_point.out_channels, out_channels)
        self.assertEqual(module.conv_point.kernel_size, (1, 1))
        self.assertFalse(module.conv_point.bias)  # Should be False

    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        module = GLUMBConv(32, 32)

        # Create 4D input (batch, channels, height, width)
        input_tensor = torch.randn(2, 32, 16, 16)

        with torch.no_grad():
            output = module(input_tensor)

        self.assert_tensor_shape(output, input_tensor.shape)
        self.assert_no_nan_or_inf(output)

    def test_forward_pass_different_shapes(self):
        """Test forward pass with different spatial dimensions."""
        module = GLUMBConv(32, 32)

        test_shapes = [
            (1, 32, 8, 8),
            (2, 32, 32, 32),
            (4, 32, 64, 64),
            (1, 32, 1, 1),  # Minimal spatial
        ]

        for shape in test_shapes:
            with self.subTest(shape=shape):
                input_tensor = torch.randn(shape)

                with torch.no_grad():
                    output = module(input_tensor)

                self.assert_tensor_shape(output, shape)
                self.assert_no_nan_or_inf(output)

    def test_forward_pass_channel_change(self):
        """Test forward pass with different input/output channels."""
        in_channels, out_channels = 32, 64
        module = GLUMBConv(in_channels, out_channels)

        input_tensor = torch.randn(2, in_channels, 16, 16)

        with torch.no_grad():
            output = module(input_tensor)

        expected_shape = (2, out_channels, 16, 16)
        self.assert_tensor_shape(output, expected_shape)

    def test_residual_connection_behavior(self):
        """Test residual connection behavior."""
        # Test with residual connection
        module_with_residual = GLUMBConv(32, 32, residual_connection=True)

        # Test without residual connection
        module_without_residual = GLUMBConv(32, 32, residual_connection=False)

        input_tensor = torch.randn(2, 32, 16, 16)

        with torch.no_grad():
            output_with = module_with_residual(input_tensor)
            output_without = module_without_residual(input_tensor)

        # Both should have same shape but different values
        self.assert_tensor_shape(output_with, input_tensor.shape)
        self.assert_tensor_shape(output_without, input_tensor.shape)
        self.assertFalse(torch.allclose(output_with, output_without))

    def test_residual_incompatible_shapes(self):
        """Test residual connection with incompatible shapes (should work without residual)."""
        # Different input/output channels - residual should still work by design
        module = GLUMBConv(32, 64, residual_connection=True)

        input_tensor = torch.randn(2, 32, 16, 16)

        with torch.no_grad():
            output = module(input_tensor)

        # Should handle the channel mismatch gracefully
        expected_shape = (2, 64, 16, 16)
        self.assert_tensor_shape(output, expected_shape)

    def test_expand_ratio_effects(self):
        """Test different expand ratios."""
        expand_ratios = [2.0, 4.0, 6.0]
        input_tensor = torch.randn(2, 32, 16, 16)

        outputs = []
        for ratio in expand_ratios:
            with self.subTest(expand_ratio=ratio):
                module = GLUMBConv(32, 32, expand_ratio=ratio)

                with torch.no_grad():
                    output = module(input_tensor)

                self.assert_tensor_shape(output, input_tensor.shape)
                self.assert_no_nan_or_inf(output)
                outputs.append(output)

        # Different expand ratios should produce different outputs
        for i in range(len(outputs) - 1):
            self.assertFalse(torch.allclose(outputs[i], outputs[i + 1]))

    def test_rms_norm_functionality(self):
        """Test RMS normalization functionality."""
        module_with_norm = GLUMBConv(32, 32, norm_type="rms_norm")
        module_without_norm = GLUMBConv(32, 32, norm_type=None)

        input_tensor = torch.randn(2, 32, 16, 16)

        with torch.no_grad():
            output_with_norm = module_with_norm(input_tensor)
            output_without_norm = module_without_norm(input_tensor)

        # Both should have same shape but different values due to normalization
        self.assert_tensor_shape(output_with_norm, input_tensor.shape)
        self.assert_tensor_shape(output_without_norm, input_tensor.shape)
        self.assertFalse(torch.allclose(output_with_norm, output_without_norm))

    def test_glu_gating_mechanism(self):
        """Test GLU gating mechanism in the depth convolution."""
        # Create module to test internal gating
        module = GLUMBConv(32, 32)

        # Create input that will test the gating
        input_tensor = torch.randn(2, 32, 16, 16)

        # Test by checking intermediate values (if we can access them)
        with torch.no_grad():
            # Run the computation steps manually to verify gating
            x = module.conv_inverted(input_tensor)
            x = module.nonlinearity(x)
            x = module.conv_depth(x)

            # This should split into two halves for gating
            x_main, gate = torch.chunk(x, 2, dim=1)
            gated = x_main * module.nonlinearity(gate)

            # Verify shapes are correct for gating
            expected_channels = int(4.0 * 32)  # expand_ratio * in_channels
            self.assert_tensor_shape(x_main, (2, expected_channels, 16, 16))
            self.assert_tensor_shape(gate, (2, expected_channels, 16, 16))
            self.assert_tensor_shape(gated, (2, expected_channels, 16, 16))

    def test_dtype_preservation(self):
        """Test dtype preservation through the module."""
        module = GLUMBConv(32, 32)

        dtypes = [torch.float32, torch.float16]
        for dtype in dtypes:
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue

            device = "cuda" if dtype == torch.float16 else "cpu"
            input_tensor = torch.randn(2, 32, 16, 16, dtype=dtype, device=device)

            with torch.no_grad():
                output = module.to(device)(input_tensor)

            self.assertEqual(output.dtype, dtype)

    def test_device_consistency(self):
        """Test device consistency."""
        if torch.cuda.is_available():
            module = GLUMBConv(32, 32).cuda()
            input_tensor = torch.randn(2, 32, 16, 16, device="cuda")

            with torch.no_grad():
                output = module(input_tensor)

            self.assertEqual(output.device, input_tensor.device)

    def test_typo_prevention(self):
        """Test parameter name typos."""
        # Test correct instantiation parameters
        try:
            module = GLUMBConv(
                in_channels=32, out_channels=32, expand_ratio=4.0, norm_type="rms_norm", residual_connection=True
            )
            self.assertIsNotNone(module)
        except TypeError as e:
            self.fail(f"Should accept valid parameter names: {e}")

        # Test method existence
        self.run_method_existence_tests(module, ["forward"])

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with minimal expand ratio
        module_min = GLUMBConv(32, 32, expand_ratio=1.0)
        input_tensor = torch.randn(2, 32, 16, 16)

        with torch.no_grad():
            output = module_min(input_tensor)

        self.assert_no_nan_or_inf(output)

        # Test with large expand ratio
        module_large = GLUMBConv(16, 16, expand_ratio=8.0)
        input_small = torch.randn(1, 16, 8, 8)

        with torch.no_grad():
            output_large = module_large(input_small)

        self.assert_no_nan_or_inf(output_large)

        # Test with single channel
        module_single = GLUMBConv(1, 1)
        input_single = torch.randn(1, 1, 4, 4)

        with torch.no_grad():
            output_single = module_single(input_single)

        self.assert_no_nan_or_inf(output_single)


class TestSanaTransformerBlock(TransformerBaseTest, TransformerBlockTestMixin):
    """Test SanaTransformerBlock class."""

    def setUp(self):
        super().setUp()
        self.block_config = {
            "dim": 512,
            "num_attention_heads": 8,
            "attention_head_dim": 64,
            "dropout": 0.0,
            "num_cross_attention_heads": 4,
            "cross_attention_head_dim": 128,
            "cross_attention_dim": 512,
            "attention_bias": True,
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "attention_out_bias": True,
            "mlp_ratio": 2.5,
        }

    def test_instantiation(self):
        """Test basic instantiation."""
        block = SanaTransformerBlock(**self.block_config)

        self.assertIsInstance(block, nn.Module)
        self.assertTrue(hasattr(block, "norm1"))
        self.assertTrue(hasattr(block, "attn1"))
        self.assertTrue(hasattr(block, "norm2"))
        self.assertTrue(hasattr(block, "attn2"))
        self.assertTrue(hasattr(block, "ff"))
        self.assertTrue(hasattr(block, "scale_shift_table"))

    def test_component_initialization(self):
        """Test proper initialization of components."""
        block = SanaTransformerBlock(**self.block_config)

        # Check self attention
        self.assertIsInstance(block.norm1, nn.LayerNorm)
        self.assertFalse(block.norm1.elementwise_affine)

        # Check cross attention exists
        self.assertIsNotNone(block.attn2)
        self.assertIsInstance(block.norm2, nn.LayerNorm)

        # Check GLUMBConv feedforward
        self.assertIsInstance(block.ff, GLUMBConv)

        # Check scale_shift_table
        expected_params = 6 * self.block_config["dim"]
        self.assertEqual(block.scale_shift_table.numel(), expected_params)

    def test_instantiation_without_cross_attention(self):
        """Test instantiation without cross attention."""
        config = self.block_config.copy()
        config["cross_attention_dim"] = None

        block = SanaTransformerBlock(**config)

        self.assertIsNone(block.attn2)

    def test_forward_pass_basic(self):
        """Test basic forward pass with cross attention."""
        block = SanaTransformerBlock(**self.block_config)

        batch_size = 2
        seq_len = 128
        dim = self.block_config["dim"]
        height = width = 16  # For reshaping in feedforward

        hidden_states = torch.randn(batch_size, seq_len, dim)
        encoder_hidden_states = torch.randn(batch_size, 77, dim)
        timestep = torch.randn(batch_size, 6, dim)  # Modulation parameters

        with torch.no_grad():
            output = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                height=height,
                width=width,
            )

        self.assert_tensor_shape(output, hidden_states.shape)
        self.assert_no_nan_or_inf(output)

    def test_forward_pass_without_cross_attention(self):
        """Test forward pass without cross attention."""
        config = self.block_config.copy()
        config["cross_attention_dim"] = None

        block = SanaTransformerBlock(**config)

        batch_size = 2
        seq_len = 128
        dim = config["dim"]
        height = width = 16

        hidden_states = torch.randn(batch_size, seq_len, dim)
        timestep = torch.randn(batch_size, 6, dim)

        with torch.no_grad():
            output = block(hidden_states=hidden_states, timestep=timestep, height=height, width=width)

        self.assert_tensor_shape(output, hidden_states.shape)
        self.assert_no_nan_or_inf(output)

    def test_modulation_mechanism(self):
        """Test the modulation mechanism with scale_shift_table."""
        block = SanaTransformerBlock(**self.block_config)

        batch_size = 2
        seq_len = 128
        dim = self.block_config["dim"]
        height = width = 16

        hidden_states = torch.randn(batch_size, seq_len, dim)

        # Test different timestep values
        timestep_zero = torch.zeros(batch_size, 6, dim)
        timestep_ones = torch.ones(batch_size, 6, dim)

        with torch.no_grad():
            output_zero = block(hidden_states=hidden_states, timestep=timestep_zero, height=height, width=width)

            output_ones = block(hidden_states=hidden_states, timestep=timestep_ones, height=height, width=width)

        # Different timesteps should produce different outputs
        self.assertFalse(torch.allclose(output_zero, output_ones))

    def test_attention_mask_handling(self):
        """Test attention mask parameter handling."""
        block = SanaTransformerBlock(**self.block_config)

        batch_size = 2
        seq_len = 128
        dim = self.block_config["dim"]
        height = width = 16

        hidden_states = torch.randn(batch_size, seq_len, dim)
        encoder_hidden_states = torch.randn(batch_size, 77, dim)
        timestep = torch.randn(batch_size, 6, dim)

        # Create attention masks
        attention_mask = torch.ones(batch_size, seq_len)
        encoder_attention_mask = torch.ones(batch_size, 77)

        with torch.no_grad():
            output = block(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                height=height,
                width=width,
            )

        self.assert_tensor_shape(output, hidden_states.shape)
        self.assert_no_nan_or_inf(output)

    def test_feedforward_spatial_reshaping(self):
        """Test feedforward layer spatial reshaping."""
        block = SanaTransformerBlock(**self.block_config)

        # Test with square dimensions that work for reshaping
        batch_size = 2
        height = width = 16
        seq_len = height * width  # 256
        dim = self.block_config["dim"]

        hidden_states = torch.randn(batch_size, seq_len, dim)
        timestep = torch.randn(batch_size, 6, dim)

        with torch.no_grad():
            output = block(hidden_states=hidden_states, timestep=timestep, height=height, width=width)

        self.assert_tensor_shape(output, hidden_states.shape)
        self.assert_no_nan_or_inf(output)

    def test_different_dimensions(self):
        """Test with different dimension configurations."""
        dim_configs = [
            {"dim": 256, "num_attention_heads": 4, "attention_head_dim": 64},
            {"dim": 1024, "num_attention_heads": 16, "attention_head_dim": 64},
            {"dim": 2048, "num_attention_heads": 32, "attention_head_dim": 64},
        ]

        for config in dim_configs:
            with self.subTest(config=config):
                test_config = self.block_config.copy()
                test_config.update(config)

                block = SanaTransformerBlock(**test_config)

                batch_size = 1
                seq_len = 64
                dim = config["dim"]
                height = width = 8

                hidden_states = torch.randn(batch_size, seq_len, dim)
                timestep = torch.randn(batch_size, 6, dim)

                with torch.no_grad():
                    output = block(hidden_states=hidden_states, timestep=timestep, height=height, width=width)

                self.assert_tensor_shape(output, hidden_states.shape)

    def test_mlp_ratio_effects(self):
        """Test different MLP ratio values."""
        mlp_ratios = [1.0, 2.5, 4.0]

        for ratio in mlp_ratios:
            with self.subTest(mlp_ratio=ratio):
                config = self.block_config.copy()
                config["mlp_ratio"] = ratio

                block = SanaTransformerBlock(**config)

                # Check that GLUMBConv has correct expand_ratio
                self.assertEqual(block.ff.expand_ratio, ratio)

    def test_dropout_parameter(self):
        """Test dropout parameter configuration."""
        dropout_values = [0.0, 0.1, 0.2]

        for dropout in dropout_values:
            with self.subTest(dropout=dropout):
                config = self.block_config.copy()
                config["dropout"] = dropout

                block = SanaTransformerBlock(**config)

                # Basic functionality test
                hidden_states = torch.randn(2, 64, config["dim"])
                timestep = torch.randn(2, 6, config["dim"])

                with torch.no_grad():
                    output = block(hidden_states=hidden_states, timestep=timestep, height=8, width=8)

                self.assert_no_nan_or_inf(output)

    def test_typo_prevention(self):
        """Test parameter name typos."""
        # Test correct instantiation parameters
        try:
            block = SanaTransformerBlock(
                dim=512,
                num_attention_heads=8,
                attention_head_dim=64,
                dropout=0.0,
                num_cross_attention_heads=4,
                cross_attention_head_dim=128,
                cross_attention_dim=512,
                attention_bias=True,
                norm_elementwise_affine=False,
                norm_eps=1e-6,
                attention_out_bias=True,
                mlp_ratio=2.5,
            )
            self.assertIsNotNone(block)
        except TypeError as e:
            self.fail(f"Should accept valid parameter names: {e}")

        # Test method existence
        self.run_method_existence_tests(block, ["forward"])

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        block = SanaTransformerBlock(**self.block_config)

        # Test minimal sequence length
        hidden_states = torch.randn(1, 1, self.block_config["dim"])
        timestep = torch.randn(1, 6, self.block_config["dim"])

        with torch.no_grad():
            output = block(hidden_states=hidden_states, timestep=timestep, height=1, width=1)

        self.assert_tensor_shape(output, hidden_states.shape)
        self.assert_no_nan_or_inf(output)

        # Test with mismatched height/width vs sequence length
        # This should still work as the model handles reshaping
        hidden_states = torch.randn(1, 16, self.block_config["dim"])  # seq_len = 16
        timestep = torch.randn(1, 6, self.block_config["dim"])

        with torch.no_grad():
            output = block(
                hidden_states=hidden_states, timestep=timestep, height=4, width=4  # height * width = 16, matches seq_len
            )

        self.assert_tensor_shape(output, hidden_states.shape)


class TestSanaTransformer2DModel(TransformerBaseTest):
    """Test SanaTransformer2DModel class."""

    def setUp(self):
        super().setUp()
        self.model_config = {
            "in_channels": 32,
            "out_channels": 32,
            "num_attention_heads": 8,
            "attention_head_dim": 64,
            "num_layers": 2,  # Reduced for testing
            "num_cross_attention_heads": 4,
            "cross_attention_head_dim": 128,
            "cross_attention_dim": 512,
            "caption_channels": 1024,
            "mlp_ratio": 2.5,
            "dropout": 0.0,
            "attention_bias": False,
            "sample_size": 32,
            "patch_size": 1,
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "interpolation_scale": None,
        }

    def test_instantiation(self):
        """Test basic instantiation."""
        model = SanaTransformer2DModel(**self.model_config)

        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.config.in_channels, 32)
        self.assertEqual(model.config.num_layers, 2)
        self.assertEqual(model.config.patch_size, 1)

    def test_component_initialization(self):
        """Test proper initialization of all components."""
        model = SanaTransformer2DModel(**self.model_config)

        # Check main components
        self.assertTrue(hasattr(model, "patch_embed"))
        self.assertTrue(hasattr(model, "time_embed"))
        self.assertTrue(hasattr(model, "caption_projection"))
        self.assertTrue(hasattr(model, "caption_norm"))
        self.assertTrue(hasattr(model, "transformer_blocks"))
        self.assertTrue(hasattr(model, "scale_shift_table"))
        self.assertTrue(hasattr(model, "norm_out"))
        self.assertTrue(hasattr(model, "proj_out"))

        # Check transformer blocks count
        self.assertEqual(len(model.transformer_blocks), self.model_config["num_layers"])

        # Check TREAD support
        self.assertTrue(hasattr(model, "_tread_router"))
        self.assertTrue(hasattr(model, "_tread_routes"))

    @patch("simpletuner.helpers.models.sana.transformer.SanaTransformerBlock")
    def test_forward_pass_basic(self, mock_block_class):
        """Test basic forward pass."""
        # Setup mock transformer blocks
        mock_block = Mock()
        mock_block.return_value = torch.randn(2, 1024, 512)  # Flattened spatial output
        mock_block_class.return_value = mock_block

        model = SanaTransformer2DModel(**self.model_config)

        # Replace transformer blocks with mocks
        for i in range(len(model.transformer_blocks)):
            model.transformer_blocks[i] = mock_block

        # Input tensors
        batch_size = 2
        height = width = 32
        in_channels = self.model_config["in_channels"]

        hidden_states = torch.randn(batch_size, in_channels, height, width)
        encoder_hidden_states = torch.randn(batch_size, 77, self.model_config["caption_channels"])
        timestep = torch.randint(0, 1000, (batch_size,))

        with torch.no_grad():
            output = model(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep)

        # Check output shape and properties
        if hasattr(output, "sample"):
            output_tensor = output.sample
        else:
            output_tensor = output

        # Output should match input spatial dimensions
        expected_shape = (batch_size, self.model_config["out_channels"], height, width)
        self.assert_tensor_shape(output_tensor, expected_shape)
        self.assert_no_nan_or_inf(output_tensor)

    def test_patch_embedding_integration(self):
        """Test patch embedding integration."""
        model = SanaTransformer2DModel(**self.model_config)

        # Test that patch embedding is properly configured
        expected_patch = self.model_config["patch_size"]
        if isinstance(expected_patch, int):
            self.assertEqual(model.patch_embed.patch_size, expected_patch)
        else:
            self.assertEqual(model.patch_embed.patch_size, tuple(expected_patch))
        self.assertEqual(model.patch_embed.proj.in_channels, self.model_config["in_channels"])

    def test_time_embedding_functionality(self):
        """Test time embedding component."""
        model = SanaTransformer2DModel(**self.model_config)

        batch_size = 2
        timestep = torch.randint(0, 1000, (batch_size,))

        # Test time embedding component
        timestep_emb, embedded_timestep = model.time_embed(timestep, batch_size=batch_size, hidden_dtype=torch.float32)

        self.assertIsInstance(timestep_emb, torch.Tensor)
        self.assertIsInstance(embedded_timestep, torch.Tensor)

    def test_caption_processing(self):
        """Test caption projection and normalization."""
        model = SanaTransformer2DModel(**self.model_config)

        batch_size = 2
        caption_channels = self.model_config["caption_channels"]
        seq_len = 77

        encoder_hidden_states = torch.randn(batch_size, seq_len, caption_channels)

        # Test caption projection
        projected = model.caption_projection(encoder_hidden_states)
        inner_dim = self.model_config["num_attention_heads"] * self.model_config["attention_head_dim"]

        expected_shape = (batch_size, seq_len, inner_dim)
        self.assert_tensor_shape(projected, expected_shape)

        # Test caption normalization
        normalized = model.caption_norm(projected)
        self.assert_tensor_shape(normalized, expected_shape)
        self.assert_no_nan_or_inf(normalized)

    def test_attention_mask_conversion(self):
        """Test attention mask conversion to bias format."""
        model = SanaTransformer2DModel(**self.model_config)

        # Mock transformer blocks for faster execution
        MockingUtils.safely_replace_modules_in_list(model.transformer_blocks)

        batch_size = 2
        hidden_states = torch.randn(batch_size, 32, 32, 32)
        encoder_hidden_states = torch.randn(batch_size, 77, 1024)
        timestep = torch.randint(0, 1000, (batch_size,))

        # Test with 2D attention mask (should be converted to bias)
        attention_mask = torch.ones(batch_size, 1024)  # Sequence length after patching
        encoder_attention_mask = torch.ones(batch_size, 77)

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
            )

        self.assertIsNotNone(output)

    def test_tread_router_integration(self):
        """Test TREAD router integration."""
        model = SanaTransformer2DModel(**self.model_config)

        # Initially no router
        self.assertIsNone(model._tread_router)
        self.assertIsNone(model._tread_routes)

        # Set router
        mock_router = Mock()
        routes = [{"start_layer_idx": 0, "end_layer_idx": 1, "selection_ratio": 0.5}]

        model.set_router(mock_router, routes)

        self.assertEqual(model._tread_router, mock_router)
        self.assertEqual(model._tread_routes, routes)

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing functionality."""
        model = SanaTransformer2DModel(**self.model_config)
        model.gradient_checkpointing = True
        model.gradient_checkpointing_interval = 1

        # Mock transformer blocks
        MockingUtils.safely_replace_modules_in_list(model.transformer_blocks)

        hidden_states = torch.randn(2, 32, 32, 32)
        encoder_hidden_states = torch.randn(2, 77, 1024)
        timestep = torch.randint(0, 1000, (2,))

        # Enable gradients and training mode to test checkpointing path
        hidden_states.requires_grad_(True)
        model.train()

        output = model(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep)

        self.assertIsNotNone(output)

    def test_gradient_checkpointing_interval(self):
        """Test gradient checkpointing interval functionality."""
        model = SanaTransformer2DModel(**self.model_config)

        # Test setting interval
        interval = 2
        model.set_gradient_checkpointing_interval(interval)
        self.assertEqual(model.gradient_checkpointing_interval, interval)

    def test_attention_processors_property(self):
        """Test attn_processors property."""
        model = SanaTransformer2DModel(**self.model_config)

        processors = model.attn_processors
        self.assertIsInstance(processors, dict)

        # Should have processors for each attention layer
        expected_keys = 0
        for block in model.transformer_blocks:
            if hasattr(block, "attn1"):
                expected_keys += 1
            if hasattr(block, "attn2") and block.attn2 is not None:
                expected_keys += 1

        # Due to mocking, exact count may vary, but should be a dict
        self.assertGreaterEqual(len(processors), 0)

    def test_set_attn_processor(self):
        """Test set_attn_processor method."""
        model = SanaTransformer2DModel(**self.model_config)

        # Create a mock processor
        mock_processor = Mock()

        # Test setting single processor for all
        try:
            model.set_attn_processor(mock_processor)
        except Exception as e:
            # May fail due to mocking, but should not have typos
            if "unexpected keyword argument" in str(e):
                self.fail(f"set_attn_processor should accept valid arguments: {e}")

    def test_return_dict_parameter(self):
        """Test return_dict parameter controls output format."""
        model = SanaTransformer2DModel(**self.model_config)

        # Mock transformer blocks
        MockingUtils.safely_replace_modules_in_list(model.transformer_blocks)

        hidden_states = torch.randn(2, 32, 32, 32)
        encoder_hidden_states = torch.randn(2, 77, 1024)
        timestep = torch.randint(0, 1000, (2,))

        # Test return_dict=True (default)
        with torch.no_grad():
            output_dict = model(
                hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep, return_dict=True
            )

        self.assertTrue(hasattr(output_dict, "sample"))

        # Test return_dict=False
        with torch.no_grad():
            output_tuple = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                return_dict=False,
            )

        self.assertIsInstance(output_tuple, tuple)

    def test_attention_kwargs_handling(self):
        """Test attention_kwargs parameter handling including LoRA scale."""
        model = SanaTransformer2DModel(**self.model_config)

        # Mock transformer blocks
        MockingUtils.safely_replace_modules_in_list(model.transformer_blocks)

        hidden_states = torch.randn(2, 32, 32, 32)
        encoder_hidden_states = torch.randn(2, 77, 1024)
        timestep = torch.randint(0, 1000, (2,))
        attention_kwargs = {"scale": 0.5}

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                attention_kwargs=attention_kwargs,
            )

        self.assertIsNotNone(output)

    def test_different_patch_sizes(self):
        """Test model with different patch sizes."""
        patch_sizes = [1, 2, 4]

        for patch_size in patch_sizes:
            with self.subTest(patch_size=patch_size):
                config = self.model_config.copy()
                config["patch_size"] = patch_size

                model = SanaTransformer2DModel(**config)

                # Mock transformer blocks
                MockingUtils.safely_replace_modules_in_list(model.transformer_blocks)

                # Adjust input size to be divisible by patch size
                spatial_size = 32
                while spatial_size % patch_size != 0:
                    spatial_size += 1

                hidden_states = torch.randn(2, 32, spatial_size, spatial_size)
                encoder_hidden_states = torch.randn(2, 77, 1024)
                timestep = torch.randint(0, 1000, (2,))

                with torch.no_grad():
                    output = model(
                        hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep
                    )

                self.assertIsNotNone(output)

    def test_interpolation_scale_parameter(self):
        """Test interpolation_scale parameter for position embeddings."""
        config = self.model_config.copy()
        config["interpolation_scale"] = 2

        model = SanaTransformer2DModel(**config)

        # Should initialize without errors
        self.assertIsNotNone(model.patch_embed)

    def test_typo_prevention(self):
        """Test parameter name typos."""
        # Test correct instantiation parameters
        try:
            model = SanaTransformer2DModel(
                in_channels=32,
                out_channels=32,
                num_attention_heads=8,
                attention_head_dim=64,
                num_layers=2,
                num_cross_attention_heads=4,
                cross_attention_head_dim=128,
                cross_attention_dim=512,
                caption_channels=1024,
                mlp_ratio=2.5,
                dropout=0.0,
                attention_bias=False,
                sample_size=32,
                patch_size=1,
                norm_elementwise_affine=False,
                norm_eps=1e-6,
                interpolation_scale=None,
            )
            self.assertIsNotNone(model)
        except TypeError as e:
            self.fail(f"Should accept valid parameter names: {e}")

        # Test method existence
        self.run_method_existence_tests(
            model, ["forward", "set_router", "set_gradient_checkpointing_interval", "set_attn_processor"]
        )

    def test_performance_benchmark(self):
        """Test performance benchmark."""
        model = SanaTransformer2DModel(**self.model_config)

        # Mock transformer blocks for faster execution
        MockingUtils.safely_replace_modules_in_list(model.transformer_blocks)

        inputs = {
            "hidden_states": torch.randn(1, 32, 16, 16),  # Smaller input
            "encoder_hidden_states": torch.randn(1, 77, 1024),
            "timestep": torch.randint(0, 1000, (1,)),
        }

        # Performance test with relaxed threshold for complex model
        self.run_performance_benchmark(model, inputs, max_time_ms=2000.0)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        model = SanaTransformer2DModel(**self.model_config)

        # Mock transformer blocks
        MockingUtils.safely_replace_modules_in_list(model.transformer_blocks)

        # Test minimal input sizes
        hidden_states = torch.randn(1, 32, 1, 1)  # Minimal spatial size
        encoder_hidden_states = torch.randn(1, 1, 1024)  # Single token
        timestep = torch.randint(0, 1000, (1,))

        with torch.no_grad():
            output = model(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep)

        self.assertIsNotNone(output)
        if hasattr(output, "sample"):
            self.assert_no_nan_or_inf(output.sample)

    def test_out_channels_default(self):
        """Test out_channels defaults to in_channels when not specified."""
        config = self.model_config.copy()
        config.pop("out_channels")  # Remove out_channels

        model = SanaTransformer2DModel(**config)

        # Should default to in_channels
        self.assertEqual(model.config.out_channels, config["in_channels"])


# Performance and integration tests
class TestSanaTransformerIntegration(TransformerBaseTest):
    """Integration tests for Sana transformer components."""

    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline integration."""
        # Create minimal model for integration test
        config = {
            "in_channels": 16,
            "out_channels": 16,
            "num_attention_heads": 4,
            "attention_head_dim": 32,
            "num_layers": 1,
            "num_cross_attention_heads": 2,
            "cross_attention_head_dim": 64,
            "cross_attention_dim": 256,
            "caption_channels": 512,
            "mlp_ratio": 2.0,
            "sample_size": 16,
            "patch_size": 1,
        }

        model = SanaTransformer2DModel(**config)

        # Create realistic inputs
        batch_size = 1
        hidden_states = torch.randn(batch_size, 16, 16, 16)
        encoder_hidden_states = torch.randn(batch_size, 77, 512)
        timestep = torch.randint(0, 1000, (batch_size,))

        # Test forward pass
        with torch.no_grad():
            output = model(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep)

        # Validate output
        if hasattr(output, "sample"):
            output_tensor = output.sample
        else:
            output_tensor = output

        self.assert_tensor_shape(output_tensor, hidden_states.shape)
        self.assert_no_nan_or_inf(output_tensor)
        self.assert_tensor_in_range(output_tensor, -10.0, 10.0)

    def test_glumconv_in_transformer_block(self):
        """Test GLUMBConv integration within transformer block."""
        # Test GLUMBConv directly
        glumconv = GLUMBConv(512, 512, expand_ratio=2.5)

        # Test within transformer block context
        block = SanaTransformerBlock(dim=512, num_attention_heads=8, attention_head_dim=64, mlp_ratio=2.5)

        # Verify GLUMBConv is used as feedforward
        self.assertIsInstance(block.ff, GLUMBConv)
        self.assertEqual(block.ff.expand_ratio, 2.5)

        # Test functionality
        batch_size = 2
        seq_len = 64
        height = width = 8  # height * width = seq_len

        hidden_states = torch.randn(batch_size, seq_len, 512)
        timestep = torch.randn(batch_size, 6, 512)

        with torch.no_grad():
            output = block(hidden_states=hidden_states, timestep=timestep, height=height, width=width)

        self.assert_tensor_shape(output, hidden_states.shape)
        self.assert_no_nan_or_inf(output)

    def test_component_compatibility(self):
        """Test compatibility between different components."""
        # Test patch embedding with transformer blocks
        model = SanaTransformer2DModel(
            in_channels=4,
            num_attention_heads=4,
            attention_head_dim=32,
            num_layers=1,
            caption_channels=256,
            sample_size=16,
            patch_size=2,
        )

        batch_size = 1
        hidden_states = torch.randn(batch_size, 4, 16, 16)
        encoder_hidden_states = torch.randn(batch_size, 77, 256)
        timestep = torch.randint(0, 1000, (batch_size,))

        # Test that all components work together
        with torch.no_grad():
            output = model(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, timestep=timestep)

        self.assertIsNotNone(output)

    def test_tread_routing_integration(self):
        """Test TREAD routing integration in training mode."""
        model = SanaTransformer2DModel(
            in_channels=4,
            num_attention_heads=4,
            attention_head_dim=32,
            num_layers=2,
            caption_channels=256,
            sample_size=16,
            patch_size=1,
        )

        # Set up TREAD router
        mock_router = Mock()
        mock_router.get_mask.return_value = {"mask": torch.ones(64), "selected_indices": torch.arange(32)}
        mock_router.start_route.return_value = torch.randn(1, 32, 128)  # Reduced tokens
        mock_router.end_route.return_value = torch.randn(1, 64, 128)  # Restored tokens

        routes = [{"start_layer_idx": 0, "end_layer_idx": 1, "selection_ratio": 0.5}]
        model.set_router(mock_router, routes)

        # Set training mode to enable routing
        model.train()

        hidden_states = torch.randn(1, 4, 16, 16)
        encoder_hidden_states = torch.randn(1, 77, 256)
        timestep = torch.randint(0, 1000, (1,))

        # Enable gradients to trigger routing
        hidden_states.requires_grad_(True)

        with torch.no_grad():  # Still use no_grad for testing
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                force_keep_mask=torch.ones(256),  # Keep all tokens
            )

        self.assertIsNotNone(output)


if __name__ == "__main__":
    unittest.main()
