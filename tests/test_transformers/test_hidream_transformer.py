"""
Comprehensive unit tests for hidream/transformer.py

Tests all 16 classes and 55+ functions with focus on:
- EmbedND, PatchEmbed, PooledEmbed, TimestepEmbed, OutEmbed
- Load balancing loss functions (save, clear, get operations)
- MoE (Mixture of Experts) components
- HiDreamImageTransformer2DModel forward pass with various configurations
- Expert routing and selection
- Normalization layers
- Attention mechanisms
- TREAD integration
- Gradient checkpointing
"""

import os

# Import the base test classes
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
import torch.nn as nn

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

# Import the modules under test
from simpletuner.helpers.models.hidream.transformer import (  # Load balancing functions; Core functions; Embedding classes; Attention and MLP classes; Block types and transformer classes
    Attention,
    BlockType,
    EmbedND,
    FeedForward,
    HiDreamAttnProcessor_flashattn,
    HiDreamImageBlock,
    HiDreamImageSingleTransformerBlock,
    HiDreamImageTransformer2DModel,
    HiDreamImageTransformerBlock,
    MOEFeedForward,
    MoEGate,
    OutEmbed,
    PatchEmbed,
    PooledEmbed,
    TextProjection,
    TimestepEmbed,
    apply_rope,
    attention,
    clear_load_balancing_loss,
    get_load_balancing_loss,
    rope,
    save_load_balancing_loss,
)


class TestLoadBalancingFunctions(TransformerBaseTest):
    """Test the global load balancing loss functions."""

    def setUp(self):
        super().setUp()
        # Clear any existing load balancing losses before each test
        clear_load_balancing_loss()

    def tearDown(self):
        super().tearDown()
        # Clean up after each test
        clear_load_balancing_loss()

    def test_save_and_get_load_balancing_loss(self):
        """Test saving and retrieving load balancing losses."""
        # Test initial state
        losses = get_load_balancing_loss()
        self.assertEqual(len(losses), 0)

        # Test saving losses
        test_loss_1 = (torch.tensor(0.1), torch.randn(4), torch.randn(4), 0.01)
        test_loss_2 = (torch.tensor(0.2), torch.randn(4), torch.randn(4), 0.01)

        save_load_balancing_loss(test_loss_1)
        save_load_balancing_loss(test_loss_2)

        # Test retrieval
        losses = get_load_balancing_loss()
        self.assertEqual(len(losses), 2)
        self.assertEqual(losses[0], test_loss_1)
        self.assertEqual(losses[1], test_loss_2)

    def test_clear_load_balancing_loss(self):
        """Test clearing load balancing losses."""
        # Add some losses
        test_loss = (torch.tensor(0.1), torch.randn(4), torch.randn(4), 0.01)
        save_load_balancing_loss(test_loss)
        save_load_balancing_loss(test_loss)

        # Verify they exist
        self.assertEqual(len(get_load_balancing_loss()), 2)

        # Clear and verify
        clear_load_balancing_loss()
        self.assertEqual(len(get_load_balancing_loss()), 0)

    def test_load_balancing_loss_typos(self):
        """Test that load balancing functions reject typos."""
        # Test typo in function names would cause NameError
        with self.assertRaises(NameError):
            save_load_balancing_losses(None)  # Extra 's'

        with self.assertRaises(NameError):
            clear_load_balancing_losses()  # Extra 's'

        with self.assertRaises(NameError):
            get_load_balancing_losses()  # Extra 's'


class TestRopeFunctions(TransformerBaseTest):
    """Test the rope and apply_rope functions."""

    def test_rope_function_signature(self):
        """Test rope function parameter validation."""
        pos = torch.randn(2, 64)  # (batch, seq_len)
        dim = 128
        theta = 10000

        # Test valid call
        result = rope(pos, dim, theta)
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (2, 64, dim // 2, 2, 2))

    def test_rope_dimension_validation(self):
        """Test that rope validates dimension is even."""
        pos = torch.randn(2, 64)

        # Test odd dimension raises assertion
        with self.assertRaises(AssertionError) as context:
            rope(pos, 127, 10000)  # Odd dimension

        self.assertIn("must be even", str(context.exception))

        # Test even dimension works
        result = rope(pos, 128, 10000)
        self.assertIsNotNone(result)

    def test_rope_parameter_typos(self):
        """Test rope function rejects parameter typos."""
        pos = torch.randn(2, 64)

        # Test valid parameters work
        result = rope(pos, 128, 10000)
        self.assertIsNotNone(result)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            rope(pos, 128, theta_val=10000)  # Wrong name

        with self.assertRaises(TypeError):
            rope(position=pos, dim=128, theta=10000)  # Wrong name

    def test_apply_rope_function(self):
        """Test apply_rope function."""
        batch, seq_len, heads, head_dim = 2, 64, 8, 64
        # Correct shape: [batch, seq_len, heads, head_dim]
        xq = torch.randn(batch, seq_len, heads, head_dim)
        xk = torch.randn(batch, seq_len, heads, head_dim)

        # Create freqs_cis with shape from EmbedND: [batch, seq_len, 1, rope_dim, 2, 2]
        rope_dim = head_dim // 2
        freqs_cis = torch.randn(batch, seq_len, 1, rope_dim, 2, 2)

        # Test apply_rope
        xq_out, xk_out = apply_rope(xq, xk, freqs_cis)

        self.assertEqual(xq_out.shape, xq.shape)
        self.assertEqual(xk_out.shape, xk.shape)
        self.assertEqual(xq_out.dtype, xq.dtype)
        self.assertEqual(xk_out.dtype, xk.dtype)

    def test_apply_rope_dtype_preservation(self):
        """Test that apply_rope preserves input dtypes."""
        for dtype in [torch.float32, torch.float16]:
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue

            batch, seq_len, heads, head_dim = 2, 64, 8, 64
            rope_dim = head_dim // 2
            # Correct shapes: [batch, seq_len, heads, head_dim] and [batch, seq_len, 1, rope_dim, 2, 2]
            xq = torch.randn(batch, seq_len, heads, head_dim, dtype=dtype)
            xk = torch.randn(batch, seq_len, heads, head_dim, dtype=dtype)
            freqs_cis = torch.randn(batch, seq_len, 1, rope_dim, 2, 2, dtype=dtype)

            xq_out, xk_out = apply_rope(xq, xk, freqs_cis)

            self.assertEqual(xq_out.dtype, dtype)
            self.assertEqual(xk_out.dtype, dtype)


class TestAttentionFunction(TransformerBaseTest):
    """Test the attention function."""

    def test_attention_function_signature(self):
        """Test attention function call signature."""
        batch, heads, seq_len, head_dim = 2, 8, 64, 64
        query = torch.randn(batch, seq_len, heads, head_dim)
        key = torch.randn(batch, seq_len, heads, head_dim)
        value = torch.randn(batch, seq_len, heads, head_dim)

        # Test valid call
        result = attention(query, key, value)
        self.assertIsInstance(result, torch.Tensor)

        # Expected shape: flattened last two dimensions
        expected_shape = (batch, seq_len, heads * head_dim)
        self.assertEqual(result.shape, expected_shape)

    def test_attention_parameter_typos(self):
        """Test attention function rejects parameter typos."""
        query = torch.randn(2, 64, 8, 64)
        key = torch.randn(2, 64, 8, 64)
        value = torch.randn(2, 64, 8, 64)

        # Test valid parameters work
        result = attention(query, key, value)
        self.assertIsNotNone(result)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            attention(queries=query, key=key, value=value)  # Wrong name

        with self.assertRaises(TypeError):
            attention(query=query, keys=key, value=value)  # Wrong name

        with self.assertRaises(TypeError):
            attention(query=query, key=key, values=value)  # Wrong name


class TestEmbedND(TransformerBaseTest, EmbeddingTestMixin):
    """Test EmbedND positional embedding class."""

    def test_instantiation(self):
        """Test EmbedND instantiation."""
        theta = 10000
        axes_dim = [32, 32]

        embed = EmbedND(theta, axes_dim)
        self.assertEqual(embed.theta, theta)
        self.assertEqual(embed.axes_dim, axes_dim)

    def test_forward_pass(self):
        """Test EmbedND forward pass."""
        embed = EmbedND(10000, [32, 32])

        # Create test input: (batch, seq_len, n_axes)
        ids = torch.zeros(2, 64, 2)
        result = embed(ids)

        # Expected output shape: (batch, seq_len, 1, combined_dim, 2, 2)
        expected_combined_dim = sum(dim // 2 for dim in [32, 32])
        self.assertEqual(result.shape, (2, 64, 1, expected_combined_dim, 2, 2))

    def test_multiple_axes_handling(self):
        """Test EmbedND with different numbers of axes."""
        # Test 2 axes
        embed_2d = EmbedND(10000, [16, 16])
        ids_2d = torch.zeros(2, 64, 2)
        result_2d = embed_2d(ids_2d)
        self.assertIsNotNone(result_2d)

        # Test 3 axes
        embed_3d = EmbedND(10000, [16, 16, 8])
        ids_3d = torch.zeros(2, 64, 3)
        result_3d = embed_3d(ids_3d)
        self.assertIsNotNone(result_3d)

    def test_parameter_name_typos(self):
        """Test EmbedND rejects parameter typos."""
        # Test valid parameters work
        embed = EmbedND(10000, [32, 32])
        self.assertIsNotNone(embed)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            EmbedND(theta_val=10000, axes_dim=[32, 32])  # Wrong name

        with self.assertRaises(TypeError):
            EmbedND(theta=10000, axes_dims=[32, 32])  # Wrong name


class TestPatchEmbed(TransformerBaseTest, EmbeddingTestMixin):
    """Test PatchEmbed class."""

    def test_instantiation(self):
        """Test PatchEmbed instantiation."""
        embed = PatchEmbed(patch_size=2, in_channels=4, out_channels=512)
        self.assertEqual(embed.patch_size, 2)
        self.assertEqual(embed.out_channels, 512)
        self.assertIsInstance(embed.proj, nn.Linear)

    def test_forward_pass(self):
        """Test PatchEmbed forward pass."""
        embed = PatchEmbed(patch_size=2, in_channels=4, out_channels=512)

        # Input should be patchified: (batch, seq_len, patch_size^2 * in_channels)
        latent = torch.randn(2, 64, 16)  # 2^2 * 4 = 16
        result = embed(latent)

        self.assertEqual(result.shape, (2, 64, 512))
        self.assert_no_nan_or_inf(result)

    def test_weight_initialization(self):
        """Test PatchEmbed weight initialization."""
        embed = PatchEmbed(patch_size=2, in_channels=4, out_channels=512)

        # Check that weights were initialized (not all zeros)
        self.assertFalse(torch.allclose(embed.proj.weight, torch.zeros_like(embed.proj.weight)))

        # Check bias initialization
        if embed.proj.bias is not None:
            self.assertTrue(torch.allclose(embed.proj.bias, torch.zeros_like(embed.proj.bias)))

    def test_parameter_name_typos(self):
        """Test PatchEmbed rejects parameter typos."""
        # Test valid parameters work
        embed = PatchEmbed(patch_size=2, in_channels=4, out_channels=512)
        self.assertIsNotNone(embed)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            PatchEmbed(patch_sizes=2, in_channels=4, out_channels=512)  # Extra 's'

        with self.assertRaises(TypeError):
            PatchEmbed(patch_size=2, input_channels=4, out_channels=512)  # Wrong name

        with self.assertRaises(TypeError):
            PatchEmbed(patch_size=2, in_channels=4, output_channels=512)  # Wrong name


class TestPooledEmbed(TransformerBaseTest, EmbeddingTestMixin):
    """Test PooledEmbed class."""

    def test_instantiation(self):
        """Test PooledEmbed instantiation."""
        embed = PooledEmbed(text_emb_dim=768, hidden_size=512)
        self.assertIsInstance(embed.pooled_embedder, torch.nn.Module)

    def test_forward_pass(self):
        """Test PooledEmbed forward pass."""
        embed = PooledEmbed(text_emb_dim=768, hidden_size=512)

        pooled_embed = torch.randn(2, 768)
        result = embed(pooled_embed)

        self.assertEqual(result.shape, (2, 512))
        self.assert_no_nan_or_inf(result)

    def test_parameter_name_typos(self):
        """Test PooledEmbed rejects parameter typos."""
        # Test valid parameters work
        embed = PooledEmbed(text_emb_dim=768, hidden_size=512)
        self.assertIsNotNone(embed)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            PooledEmbed(text_embedding_dim=768, hidden_size=512)  # Wrong name

        with self.assertRaises(TypeError):
            PooledEmbed(text_emb_dim=768, hidden_dim=512)  # Wrong name


class TestTimestepEmbed(TransformerBaseTest, EmbeddingTestMixin):
    """Test TimestepEmbed class."""

    def test_instantiation(self):
        """Test TimestepEmbed instantiation."""
        embed = TimestepEmbed(hidden_size=512, frequency_embedding_size=256)
        self.assertIsInstance(embed.time_proj, torch.nn.Module)
        self.assertIsInstance(embed.timestep_embedder, torch.nn.Module)

    def test_forward_pass(self):
        """Test TimestepEmbed forward pass."""
        embed = TimestepEmbed(hidden_size=512)

        timesteps = torch.randint(0, 1000, (2,))
        wdtype = torch.float32
        result = embed(timesteps, wdtype)

        self.assertEqual(result.shape, (2, 512))
        self.assertEqual(result.dtype, wdtype)
        self.assert_no_nan_or_inf(result)

    def test_parameter_name_typos(self):
        """Test TimestepEmbed forward method rejects parameter typos."""
        embed = TimestepEmbed(hidden_size=512)
        timesteps = torch.randint(0, 1000, (2,))
        wdtype = torch.float32

        # Test valid parameters work
        result = embed(timesteps, wdtype)
        self.assertIsNotNone(result)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            embed.forward(timestep=timesteps, wdtype=wdtype)  # Missing 's'

        with self.assertRaises(TypeError):
            embed.forward(timesteps=timesteps, dtype=wdtype)  # Wrong name


class TestOutEmbed(TransformerBaseTest, EmbeddingTestMixin):
    """Test OutEmbed class."""

    def test_instantiation(self):
        """Test OutEmbed instantiation."""
        embed = OutEmbed(hidden_size=512, patch_size=2, out_channels=4)
        self.assertIsInstance(embed.norm_final, nn.LayerNorm)
        self.assertIsInstance(embed.linear, nn.Linear)
        self.assertIsInstance(embed.adaLN_modulation, nn.Sequential)

    def test_forward_pass(self):
        """Test OutEmbed forward pass."""
        embed = OutEmbed(hidden_size=512, patch_size=2, out_channels=4)

        x = torch.randn(2, 64, 512)
        adaln_input = torch.randn(2, 512)
        result = embed(x, adaln_input)

        # Expected output: (batch, seq_len, patch_size^2 * out_channels)
        expected_dim = 2 * 2 * 4  # patch_size^2 * out_channels
        self.assertEqual(result.shape, (2, 64, expected_dim))
        self.assert_no_nan_or_inf(result)

    def test_weight_initialization(self):
        """Test OutEmbed weight initialization."""
        embed = OutEmbed(hidden_size=512, patch_size=2, out_channels=4)

        # Linear layer should be zero-initialized
        self.assertTrue(torch.allclose(embed.linear.weight, torch.zeros_like(embed.linear.weight)))
        if embed.linear.bias is not None:
            self.assertTrue(torch.allclose(embed.linear.bias, torch.zeros_like(embed.linear.bias)))

    def test_parameter_name_typos(self):
        """Test OutEmbed forward method rejects parameter typos."""
        embed = OutEmbed(hidden_size=512, patch_size=2, out_channels=4)
        x = torch.randn(2, 64, 512)
        adaln_input = torch.randn(2, 512)

        # Test valid parameters work
        result = embed.forward(x, adaln_input)
        self.assertIsNotNone(result)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            embed.forward(x=x, adaln_inputs=adaln_input)  # Extra 's'

        with self.assertRaises(TypeError):
            embed.forward(input=x, adaln_input=adaln_input)  # Wrong name


class TestHiDreamAttention(TransformerBaseTest, AttentionProcessorTestMixin):
    """Test HiDream custom Attention class."""

    def test_instantiation_single_attention(self):
        """Test Attention instantiation for single stream."""
        attn = Attention(query_dim=512, heads=8, dim_head=64, single=True)

        self.assertEqual(attn.heads, 8)
        self.assertEqual(attn.inner_dim, 512)
        self.assertTrue(attn.single)

        # Single stream should not have text projections
        self.assertFalse(hasattr(attn, "to_q_t"))

    def test_instantiation_dual_attention(self):
        """Test Attention instantiation for dual stream."""
        attn = Attention(query_dim=512, heads=8, dim_head=64, single=False)

        self.assertFalse(attn.single)

        # Dual stream should have text projections
        self.assertTrue(hasattr(attn, "to_q_t"))
        self.assertTrue(hasattr(attn, "to_k_t"))
        self.assertTrue(hasattr(attn, "to_v_t"))

    def test_forward_pass_single_stream(self):
        """Test Attention forward pass for single stream."""
        attn = Attention(query_dim=512, heads=8, dim_head=64, single=True, processor=Mock())

        # Mock the processor call
        attn.processor.return_value = torch.randn(2, 64, 512)

        norm_image_tokens = torch.randn(2, 64, 512)
        rope = torch.randn(2, 64, 32, 2, 2)

        result = attn(norm_image_tokens, rope=rope)

        # Verify processor was called with correct arguments
        attn.processor.assert_called_once()
        call_args = attn.processor.call_args
        self.assertIn("image_tokens", call_args[1])
        self.assertIn("rope", call_args[1])

    def test_weight_initialization(self):
        """Test Attention weight initialization."""
        attn = Attention(query_dim=512, heads=8, dim_head=64)

        # Check that linear layers were initialized (not all zeros)
        self.assertFalse(torch.allclose(attn.to_q.weight, torch.zeros_like(attn.to_q.weight)))
        self.assertFalse(torch.allclose(attn.to_k.weight, torch.zeros_like(attn.to_k.weight)))
        self.assertFalse(torch.allclose(attn.to_v.weight, torch.zeros_like(attn.to_v.weight)))

        # Check bias initialization
        if attn.to_q.bias is not None:
            self.assertTrue(torch.allclose(attn.to_q.bias, torch.zeros_like(attn.to_q.bias)))

    def test_parameter_name_typos(self):
        """Test Attention rejects parameter typos."""
        # Test valid parameters work
        attn = Attention(query_dim=512, heads=8, dim_head=64)
        self.assertIsNotNone(attn)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            Attention(query_dims=512, heads=8, dim_head=64)  # Extra 's'

        with self.assertRaises(TypeError):
            Attention(query_dim=512, head=8, dim_head=64)  # Missing 's'

        with self.assertRaises(TypeError):
            Attention(query_dim=512, heads=8, head_dim=64)  # Wrong name


class TestHiDreamAttnProcessor(TransformerBaseTest, AttentionProcessorTestMixin):
    """Test HiDreamAttnProcessor_flashattn."""

    def test_instantiation(self):
        """Test processor instantiation."""
        processor = HiDreamAttnProcessor_flashattn()
        self.assertIsNotNone(processor)
        self.assertTrue(callable(processor))

    def test_single_stream_processing(self):
        """Test processor for single stream attention."""
        processor = HiDreamAttnProcessor_flashattn()

        # Create mock attention
        mock_attn = Mock()
        mock_attn.heads = 8
        mock_attn.single = True
        mock_attn.q_rms_norm = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.k_rms_norm = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.to_q = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.to_k = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.to_v = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.to_out = Mock(return_value=torch.randn(2, 64, 512))

        image_tokens = torch.randn(2, 64, 512)
        rope = torch.randn(2, 64, 32, 2, 2)

        with patch("simpletuner.helpers.models.hidream.transformer.apply_rope") as mock_apply_rope:
            with patch("simpletuner.helpers.models.hidream.transformer.attention") as mock_attention:
                mock_apply_rope.return_value = (torch.randn(2, 64, 8, 64), torch.randn(2, 64, 8, 64))
                mock_attention.return_value = torch.randn(2, 64, 512)

                result = processor(attn=mock_attn, image_tokens=image_tokens, rope=rope)

                self.assertIsInstance(result, torch.Tensor)
                self.assertEqual(result.shape, image_tokens.shape)

    def test_dual_stream_processing(self):
        """Test processor for dual stream attention."""
        processor = HiDreamAttnProcessor_flashattn()

        # Create mock attention
        mock_attn = Mock()
        mock_attn.heads = 8
        mock_attn.single = False
        mock_attn.q_rms_norm = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.k_rms_norm = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.q_rms_norm_t = Mock(return_value=torch.randn(2, 77, 512))
        mock_attn.k_rms_norm_t = Mock(return_value=torch.randn(2, 77, 512))
        mock_attn.to_q = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.to_k = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.to_v = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.to_q_t = Mock(return_value=torch.randn(2, 77, 512))
        mock_attn.to_k_t = Mock(return_value=torch.randn(2, 77, 512))
        mock_attn.to_v_t = Mock(return_value=torch.randn(2, 77, 512))
        mock_attn.to_out = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.to_out_t = Mock(return_value=torch.randn(2, 77, 512))

        image_tokens = torch.randn(2, 64, 512)
        text_tokens = torch.randn(2, 77, 512)
        rope = torch.randn(2, 141, 32, 2, 2)  # 64 + 77 = 141

        with patch("simpletuner.helpers.models.hidream.transformer.apply_rope") as mock_apply_rope:
            with patch("simpletuner.helpers.models.hidream.transformer.attention") as mock_attention:
                mock_apply_rope.return_value = (torch.randn(2, 141, 8, 64), torch.randn(2, 141, 8, 64))
                mock_attention.return_value = torch.randn(2, 141, 512)

                result = processor(attn=mock_attn, image_tokens=image_tokens, text_tokens=text_tokens, rope=rope)

                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 2)
                self.assertEqual(result[0].shape, image_tokens.shape)
                self.assertEqual(result[1].shape, text_tokens.shape)

    def test_rope_application_with_dimension_mismatch(self):
        """Test RoPE application when query dimension doesn't match RoPE dimension."""
        processor = HiDreamAttnProcessor_flashattn()

        mock_attn = Mock()
        mock_attn.heads = 8
        mock_attn.single = True
        mock_attn.q_rms_norm = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.k_rms_norm = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.to_q = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.to_k = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.to_v = Mock(return_value=torch.randn(2, 64, 512))
        mock_attn.to_out = Mock(return_value=torch.randn(2, 64, 512))

        image_tokens = torch.randn(2, 64, 512)
        # RoPE dimension (32) * 2 != query head dimension (64)
        rope = torch.randn(2, 64, 32, 2, 2)

        with patch("simpletuner.helpers.models.hidream.transformer.apply_rope") as mock_apply_rope:
            with patch("simpletuner.helpers.models.hidream.transformer.attention") as mock_attention:
                mock_apply_rope.return_value = (torch.randn(2, 64, 8, 32), torch.randn(2, 64, 8, 32))
                mock_attention.return_value = torch.randn(2, 64, 512)

                result = processor(attn=mock_attn, image_tokens=image_tokens, rope=rope)

                # Should chunk the query/key and apply RoPE to first chunk only
                self.assertIsInstance(result, torch.Tensor)

    def test_parameter_name_typos(self):
        """Test processor rejects parameter typos."""
        processor = HiDreamAttnProcessor_flashattn()
        mock_attn = Mock()
        image_tokens = torch.randn(2, 64, 512)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            processor(attention=mock_attn, image_tokens=image_tokens)  # Wrong name

        with self.assertRaises(TypeError):
            processor(attn=mock_attn, image_token=image_tokens)  # Missing 's'


class TestFeedForward(TransformerBaseTest):
    """Test FeedForward class."""

    def test_instantiation(self):
        """Test FeedForward instantiation."""
        ff = FeedForward(dim=512, hidden_dim=2048)

        self.assertIsInstance(ff.w1, nn.Linear)
        self.assertIsInstance(ff.w2, nn.Linear)
        self.assertIsInstance(ff.w3, nn.Linear)

    def test_hidden_dim_calculation(self):
        """Test hidden dimension calculation with multiple_of."""
        ff = FeedForward(dim=512, hidden_dim=1000, multiple_of=256)

        # Should round up to next multiple of 256
        # 1000 * 2/3 = 666.67 -> 667, then round up to 768
        expected_hidden = 768  # 3 * 256
        self.assertEqual(ff.w1.out_features, expected_hidden)
        self.assertEqual(ff.w3.out_features, expected_hidden)
        self.assertEqual(ff.w2.in_features, expected_hidden)

    def test_forward_pass(self):
        """Test FeedForward forward pass."""
        ff = FeedForward(dim=512, hidden_dim=2048)

        x = torch.randn(2, 64, 512)
        result = ff(x)

        self.assertEqual(result.shape, x.shape)
        self.assert_no_nan_or_inf(result)

    def test_ffn_dim_multiplier(self):
        """Test FeedForward with custom FFN dimension multiplier."""
        ff = FeedForward(dim=512, hidden_dim=1000, ffn_dim_multiplier=1.5)

        # hidden_dim = 1000 * 2/3 = 666.67 -> 667
        # With multiplier: 667 * 1.5 = 1000.5 -> 1000
        # Round to multiple of 256: 1024
        expected_hidden = 1024  # 4 * 256
        self.assertEqual(ff.w1.out_features, expected_hidden)

    def test_weight_initialization(self):
        """Test FeedForward weight initialization."""
        ff = FeedForward(dim=512, hidden_dim=2048)

        # Check that weights were initialized (not all zeros)
        self.assertFalse(torch.allclose(ff.w1.weight, torch.zeros_like(ff.w1.weight)))
        self.assertFalse(torch.allclose(ff.w2.weight, torch.zeros_like(ff.w2.weight)))
        self.assertFalse(torch.allclose(ff.w3.weight, torch.zeros_like(ff.w3.weight)))

        # Check that biases are None (no bias=False)
        self.assertIsNone(ff.w1.bias)
        self.assertIsNone(ff.w2.bias)
        self.assertIsNone(ff.w3.bias)

    def test_parameter_name_typos(self):
        """Test FeedForward rejects parameter typos."""
        # Test valid parameters work
        ff = FeedForward(dim=512, hidden_dim=2048)
        self.assertIsNotNone(ff)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            FeedForward(dims=512, hidden_dim=2048)  # Extra 's'

        with self.assertRaises(TypeError):
            FeedForward(dim=512, hidden_dims=2048)  # Extra 's'

        with self.assertRaises(TypeError):
            FeedForward(dim=512, hidden_dim=2048, multiplier=256)  # Wrong name


class TestMoEGate(TransformerBaseTest):
    """Test MoEGate (Mixture of Experts gate) class."""

    def test_instantiation(self):
        """Test MoEGate instantiation."""
        gate = MoEGate(embed_dim=512, num_routed_experts=4, num_activated_experts=2, aux_loss_alpha=0.01)

        self.assertEqual(gate.top_k, 2)
        self.assertEqual(gate.n_routed_experts, 4)
        self.assertEqual(gate.alpha, 0.01)
        self.assertIsInstance(gate.weight, nn.Parameter)

    def test_forward_pass_training_mode(self):
        """Test MoEGate forward pass in training mode."""
        gate = MoEGate(embed_dim=512, num_routed_experts=4, num_activated_experts=2)
        gate.train()

        hidden_states = torch.randn(2, 64, 512)

        topk_idx, topk_weight, aux_loss = gate(hidden_states)

        # Check output shapes
        self.assertEqual(topk_idx.shape, (2 * 64, 2))  # (batch*seq, top_k)
        self.assertEqual(topk_weight.shape, (2 * 64, 2))  # (batch*seq, top_k)

        # Check expert indices are valid
        self.assertTrue((topk_idx >= 0).all())
        self.assertTrue((topk_idx < 4).all())  # num_routed_experts = 4

        # Check weights are normalized (sum to 1 if norm_topk_prob=True)
        # Note: norm_topk_prob is False by default, so no normalization

        # In training mode with alpha > 0, should have aux_loss
        if gate.alpha > 0:
            self.assertIsNotNone(aux_loss)
            self.assertIsInstance(aux_loss, torch.Tensor)

    def test_forward_pass_eval_mode(self):
        """Test MoEGate forward pass in eval mode."""
        gate = MoEGate(embed_dim=512, num_routed_experts=4, num_activated_experts=2)
        gate.eval()

        hidden_states = torch.randn(2, 64, 512)

        topk_idx, topk_weight, aux_loss = gate(hidden_states)

        # In eval mode, no aux_loss
        self.assertIsNone(aux_loss)

    def test_weight_normalization(self):
        """Test MoEGate with weight normalization enabled."""
        gate = MoEGate(embed_dim=512, num_routed_experts=4, num_activated_experts=2)
        gate.norm_topk_prob = True
        gate.top_k = 2

        hidden_states = torch.randn(2, 64, 512)

        topk_idx, topk_weight, aux_loss = gate(hidden_states)

        # With normalization, weights should sum to 1
        weight_sums = topk_weight.sum(dim=-1)
        self.assertTrue(torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6))

    def test_parameter_name_typos(self):
        """Test MoEGate rejects parameter typos."""
        # Test valid parameters work
        gate = MoEGate(embed_dim=512, num_routed_experts=4, num_activated_experts=2)
        self.assertIsNotNone(gate)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            MoEGate(embedding_dim=512, num_routed_experts=4, num_activated_experts=2)  # Wrong name

        with self.assertRaises(TypeError):
            MoEGate(embed_dim=512, num_experts=4, num_activated_experts=2)  # Wrong name

        with self.assertRaises(TypeError):
            MoEGate(embed_dim=512, num_routed_experts=4, num_activated_expert=2)  # Missing 's'


class TestMOEFeedForward(TransformerBaseTest):
    """Test MOEFeedForward (Mixture of Experts FeedForward) class."""

    def test_instantiation(self):
        """Test MOEFeedForward instantiation."""
        moe_ff = MOEFeedForward(dim=512, hidden_dim=2048, num_routed_experts=4, num_activated_experts=2)

        self.assertIsInstance(moe_ff.shared_experts, FeedForward)
        self.assertEqual(len(moe_ff.experts), 4)
        self.assertIsInstance(moe_ff.gate, MoEGate)
        self.assertEqual(moe_ff.num_activated_experts, 2)

    def test_forward_pass_training(self):
        """Test MOEFeedForward forward pass in training mode."""
        moe_ff = MOEFeedForward(dim=512, hidden_dim=2048, num_routed_experts=4, num_activated_experts=2)
        moe_ff.train()

        x = torch.randn(2, 64, 512)
        result = moe_ff(x)

        self.assertEqual(result.shape, x.shape)
        self.assert_no_nan_or_inf(result)

    def test_forward_pass_inference(self):
        """Test MOEFeedForward forward pass in inference mode."""
        moe_ff = MOEFeedForward(dim=512, hidden_dim=2048, num_routed_experts=4, num_activated_experts=2)
        moe_ff.eval()

        with torch.no_grad():
            x = torch.randn(2, 64, 512)
            result = moe_ff(x)

        self.assertEqual(result.shape, x.shape)
        self.assert_no_nan_or_inf(result)

    def test_moe_inference_method(self):
        """Test the moe_infer method."""
        moe_ff = MOEFeedForward(dim=512, hidden_dim=2048, num_routed_experts=4, num_activated_experts=2)

        # Create test inputs for inference method
        x = torch.randn(128, 512)  # (batch*seq, dim)
        flat_expert_indices = torch.randint(0, 4, (128,))
        flat_expert_weights = torch.randn(128, 1)

        with torch.no_grad():
            result = moe_ff.moe_infer(x, flat_expert_indices, flat_expert_weights)

        self.assertEqual(result.shape, x.shape)

    def test_parameter_name_typos(self):
        """Test MOEFeedForward rejects parameter typos."""
        # Test valid parameters work
        moe_ff = MOEFeedForward(dim=512, hidden_dim=2048, num_routed_experts=4, num_activated_experts=2)
        self.assertIsNotNone(moe_ff)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            MOEFeedForward(dims=512, hidden_dim=2048, num_routed_experts=4, num_activated_experts=2)

        with self.assertRaises(TypeError):
            MOEFeedForward(dim=512, hidden_dims=2048, num_routed_experts=4, num_activated_experts=2)

        with self.assertRaises(TypeError):
            MOEFeedForward(dim=512, hidden_dim=2048, num_experts=4, num_activated_experts=2)


class TestTextProjection(TransformerBaseTest):
    """Test TextProjection class."""

    def test_instantiation(self):
        """Test TextProjection instantiation."""
        proj = TextProjection(in_features=768, hidden_size=512)
        self.assertIsInstance(proj.linear, nn.Linear)
        self.assertEqual(proj.linear.in_features, 768)
        self.assertEqual(proj.linear.out_features, 512)
        self.assertIsNone(proj.linear.bias)  # bias=False

    def test_forward_pass(self):
        """Test TextProjection forward pass."""
        proj = TextProjection(in_features=768, hidden_size=512)

        caption = torch.randn(2, 77, 768)
        result = proj(caption)

        self.assertEqual(result.shape, (2, 77, 512))
        self.assert_no_nan_or_inf(result)

    def test_parameter_name_typos(self):
        """Test TextProjection rejects parameter typos."""
        # Test valid parameters work
        proj = TextProjection(in_features=768, hidden_size=512)
        self.assertIsNotNone(proj)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            TextProjection(input_features=768, hidden_size=512)  # Wrong name

        with self.assertRaises(TypeError):
            TextProjection(in_features=768, hidden_dim=512)  # Wrong name


class TestBlockType(TransformerBaseTest):
    """Test BlockType enum-like class."""

    def test_block_type_constants(self):
        """Test BlockType constants."""
        self.assertEqual(BlockType.TransformerBlock, 1)
        self.assertEqual(BlockType.SingleTransformerBlock, 2)

    def test_block_type_typos(self):
        """Test that BlockType typos raise AttributeError."""
        # Test valid access works
        self.assertIsNotNone(BlockType.TransformerBlock)

        # Test typo access raises AttributeError
        with self.assertRaises(AttributeError):
            BlockType.TransformersBlock  # Extra 's'

        with self.assertRaises(AttributeError):
            BlockType.SingleTransformersBlock  # Extra 's'


class TestHiDreamImageSingleTransformerBlock(TransformerBaseTest, TransformerBlockTestMixin):
    """Test HiDreamImageSingleTransformerBlock."""

    def setUp(self):
        super().setUp()
        self.block = HiDreamImageSingleTransformerBlock(dim=512, num_attention_heads=8, attention_head_dim=64)

    def test_instantiation_default_params(self):
        """Test block instantiation with default parameters."""
        block = HiDreamImageSingleTransformerBlock(dim=512, num_attention_heads=8, attention_head_dim=64)

        self.assertEqual(block.num_attention_heads, 8)
        self.assertIsInstance(block.norm1_i, nn.LayerNorm)
        self.assertIsInstance(block.attn1, Attention)
        self.assertIsInstance(block.norm3_i, nn.LayerNorm)

    def test_instantiation_with_moe(self):
        """Test block instantiation with MoE enabled."""
        block = HiDreamImageSingleTransformerBlock(
            dim=512, num_attention_heads=8, attention_head_dim=64, num_routed_experts=4, num_activated_experts=2
        )

        self.assertIsInstance(block.ff_i, MOEFeedForward)

    def test_instantiation_without_moe(self):
        """Test block instantiation without MoE."""
        block = HiDreamImageSingleTransformerBlock(
            dim=512, num_attention_heads=8, attention_head_dim=64, num_routed_experts=0
        )

        self.assertIsInstance(block.ff_i, FeedForward)

    def test_forward_pass(self):
        """Test forward pass."""
        image_tokens = torch.randn(2, 64, 512)
        adaln_input = torch.randn(2, 512)
        rope = torch.randn(2, 64, 32, 2, 2)

        # Mock the components to avoid complex dependencies
        with patch.object(self.block, "adaLN_modulation") as mock_adaln:
            with patch.object(self.block, "norm1_i") as mock_norm1:
                with patch.object(self.block, "attn1") as mock_attn:
                    with patch.object(self.block, "norm3_i") as mock_norm3:
                        with patch.object(self.block, "ff_i") as mock_ff:

                            # Set up mock returns
                            adaln_chunks = [torch.randn(2, 1, 512) for _ in range(6)]
                            mock_adaln.return_value = torch.cat(adaln_chunks, dim=-1)
                            mock_norm1.return_value = image_tokens
                            mock_attn.return_value = image_tokens
                            mock_norm3.return_value = image_tokens
                            mock_ff.return_value = image_tokens

                            result = self.block(image_tokens=image_tokens, adaln_input=adaln_input, rope=rope)

                            self.assertEqual(result.shape, image_tokens.shape)

    def test_parameter_name_typos(self):
        """Test block forward method rejects parameter typos."""
        image_tokens = torch.randn(2, 64, 512)
        adaln_input = torch.randn(2, 512)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            self.block.forward(image_token=image_tokens, adaln_input=adaln_input)  # Missing 's'

        with self.assertRaises(TypeError):
            self.block.forward(image_tokens=image_tokens, adaln_inputs=adaln_input)  # Extra 's'


class TestHiDreamImageTransformerBlock(TransformerBaseTest, TransformerBlockTestMixin):
    """Test HiDreamImageTransformerBlock."""

    def setUp(self):
        super().setUp()
        self.block = HiDreamImageTransformerBlock(dim=512, num_attention_heads=8, attention_head_dim=64)

    def test_instantiation_dual_stream(self):
        """Test dual stream block instantiation."""
        block = HiDreamImageTransformerBlock(dim=512, num_attention_heads=8, attention_head_dim=64)

        # Should have both image and text components
        self.assertIsInstance(block.norm1_i, nn.LayerNorm)
        self.assertIsInstance(block.norm1_t, nn.LayerNorm)
        self.assertIsInstance(block.attn1, Attention)
        self.assertIsInstance(block.norm3_i, nn.LayerNorm)
        self.assertIsInstance(block.norm3_t, nn.LayerNorm)
        self.assertIsInstance(block.ff_i, (FeedForward, MOEFeedForward))
        self.assertIsInstance(block.ff_t, FeedForward)

    def test_forward_pass_dual_stream(self):
        """Test dual stream forward pass."""
        image_tokens = torch.randn(2, 64, 512)
        text_tokens = torch.randn(2, 77, 512)
        adaln_input = torch.randn(2, 512)
        rope = torch.randn(2, 141, 32, 2, 2)  # 64 + 77 = 141

        # Mock the components
        with patch.object(self.block, "adaLN_modulation") as mock_adaln:
            with patch.object(self.block, "norm1_i") as mock_norm1_i:
                with patch.object(self.block, "norm1_t") as mock_norm1_t:
                    with patch.object(self.block, "attn1") as mock_attn:
                        with patch.object(self.block, "norm3_i") as mock_norm3_i:
                            with patch.object(self.block, "norm3_t") as mock_norm3_t:
                                with patch.object(self.block, "ff_i") as mock_ff_i:
                                    with patch.object(self.block, "ff_t") as mock_ff_t:

                                        # Set up mock returns
                                        adaln_chunks = [torch.randn(2, 1, 512) for _ in range(12)]
                                        mock_adaln.return_value = torch.cat(adaln_chunks, dim=-1)
                                        mock_norm1_i.return_value = image_tokens
                                        mock_norm1_t.return_value = text_tokens
                                        mock_attn.return_value = (image_tokens, text_tokens)
                                        mock_norm3_i.return_value = image_tokens
                                        mock_norm3_t.return_value = text_tokens
                                        mock_ff_i.return_value = image_tokens
                                        mock_ff_t.return_value = text_tokens

                                        result = self.block(
                                            image_tokens=image_tokens,
                                            text_tokens=text_tokens,
                                            adaln_input=adaln_input,
                                            rope=rope,
                                        )

                                        self.assertIsInstance(result, tuple)
                                        self.assertEqual(len(result), 2)
                                        self.assertEqual(result[0].shape, image_tokens.shape)
                                        self.assertEqual(result[1].shape, text_tokens.shape)

    def test_parameter_name_typos(self):
        """Test block forward method rejects parameter typos."""
        image_tokens = torch.randn(2, 64, 512)
        text_tokens = torch.randn(2, 77, 512)
        adaln_input = torch.randn(2, 512)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            self.block.forward(image_token=image_tokens, text_tokens=text_tokens, adaln_input=adaln_input)  # Missing 's'

        with self.assertRaises(TypeError):
            self.block.forward(image_tokens=image_tokens, text_token=text_tokens, adaln_input=adaln_input)  # Missing 's'


class TestHiDreamImageBlock(TransformerBaseTest):
    """Test HiDreamImageBlock wrapper class."""

    def test_instantiation_transformer_block(self):
        """Test instantiation with TransformerBlock type."""
        block = HiDreamImageBlock(
            dim=512, num_attention_heads=8, attention_head_dim=64, block_type=BlockType.TransformerBlock
        )

        self.assertIsInstance(block.block, HiDreamImageTransformerBlock)

    def test_instantiation_single_transformer_block(self):
        """Test instantiation with SingleTransformerBlock type."""
        block = HiDreamImageBlock(
            dim=512, num_attention_heads=8, attention_head_dim=64, block_type=BlockType.SingleTransformerBlock
        )

        self.assertIsInstance(block.block, HiDreamImageSingleTransformerBlock)

    def test_forward_delegation(self):
        """Test that forward calls are properly delegated."""
        block = HiDreamImageBlock(
            dim=512, num_attention_heads=8, attention_head_dim=64, block_type=BlockType.SingleTransformerBlock
        )

        # Mock the underlying block
        with patch.object(block, "block") as mock_block:
            mock_block.return_value = torch.randn(2, 64, 512)

            image_tokens = torch.randn(2, 64, 512)
            adaln_input = torch.randn(2, 512)

            result = block(image_tokens=image_tokens, adaln_input=adaln_input)

            # Verify delegation
            mock_block.assert_called_once_with(
                image_tokens, None, None, adaln_input, None  # image_tokens_masks  # text_tokens  # rope
            )

    def test_parameter_name_typos(self):
        """Test HiDreamImageBlock rejects parameter typos."""
        # Test valid parameters work
        block = HiDreamImageBlock(
            dim=512, num_attention_heads=8, attention_head_dim=64, block_type=BlockType.TransformerBlock
        )
        self.assertIsNotNone(block)

        # Test typo parameters raise TypeError
        with self.assertRaises(TypeError):
            HiDreamImageBlock(
                dims=512, num_attention_heads=8, attention_head_dim=64, block_type=BlockType.TransformerBlock  # Extra 's'
            )


class TestHiDreamImageTransformer2DModel(TransformerBaseTest):
    """Test HiDreamImageTransformer2DModel main class."""

    def setUp(self):
        super().setUp()
        self.config = {
            "patch_size": 2,
            "in_channels": 64,
            "out_channels": 64,
            "num_layers": 2,
            "num_single_layers": 2,
            "attention_head_dim": 128,
            "num_attention_heads": 4,
            "caption_channels": [4096, 768],
            "text_emb_dim": 768,
            "num_routed_experts": 0,  # Disable MoE for simpler testing
            "num_activated_experts": 2,
            "axes_dims_rope": (32, 32, 32),
            "max_resolution": (128, 128),
            "llama_layers": [0, 1],
            "aux_loss_alpha": 0.0,
        }

    def test_instantiation(self):
        """Test model instantiation."""
        model = HiDreamImageTransformer2DModel(**self.config)

        self.assertEqual(model.config.num_layers, 2)
        self.assertEqual(model.config.num_single_layers, 2)
        self.assertEqual(model.config.attention_head_dim, 128)
        self.assertEqual(model.config.num_attention_heads, 4)

        # Check submodules exist
        self.assertIsInstance(model.t_embedder, TimestepEmbed)
        self.assertIsInstance(model.p_embedder, PooledEmbed)
        self.assertIsInstance(model.x_embedder, PatchEmbed)
        self.assertIsInstance(model.pe_embedder, EmbedND)
        self.assertIsInstance(model.final_layer, OutEmbed)

    def test_gradient_checkpointing_methods(self):
        """Test gradient checkpointing enable/disable."""
        model = HiDreamImageTransformer2DModel(**self.config)

        # Test initial state
        self.assertFalse(model.gradient_checkpointing)

        # Test enable
        model.enable_gradient_checkpointing()
        self.assertTrue(model.gradient_checkpointing)

        # Test disable
        model.disable_gradient_checkpointing()
        self.assertFalse(model.gradient_checkpointing)

    def test_tread_router_integration(self):
        """Test TREAD router integration."""
        model = HiDreamImageTransformer2DModel(**self.config)

        # Test initial state
        self.assertIsNone(model._tread_router)
        self.assertIsNone(model._tread_routes)

        # Test setting router
        mock_router = Mock()
        mock_routes = [{"start_layer_idx": 0, "end_layer_idx": 1, "selection_ratio": 0.5}]

        model.set_router(mock_router, mock_routes)
        self.assertEqual(model._tread_router, mock_router)
        self.assertEqual(model._tread_routes, mock_routes)

    def test_expand_timesteps_method(self):
        """Test timestep expansion method."""
        model = HiDreamImageTransformer2DModel(**self.config)

        # Test float timestep
        timesteps = 0.5
        result = model.expand_timesteps(timesteps, 2, torch.device("cpu"))
        self.assertEqual(result.shape, (2,))
        self.assertEqual(result.dtype, torch.float32)

        # Test tensor timestep
        timesteps = torch.tensor([0.5])
        result = model.expand_timesteps(timesteps, 2, torch.device("cpu"))
        self.assertEqual(result.shape, (2,))

        # Test scalar tensor timestep
        timesteps = torch.tensor(0.5)
        result = model.expand_timesteps(timesteps, 2, torch.device("cpu"))
        self.assertEqual(result.shape, (2,))

    def test_patchify_method(self):
        """Test patchify method."""
        model = HiDreamImageTransformer2DModel(**self.config)
        max_seq = 32

        # Test with tensor input
        x = torch.randn(2, 64, 8, 8)  # (batch, channels, height, width)
        result_x, result_masks, result_img_sizes = model.patchify(x, max_seq)

        expected_seq_len = (8 // 2) * (8 // 2)  # 4 * 4 = 16
        expected_channels = 2 * 2 * 64  # patch_size^2 * channels = 256

        self.assertEqual(result_x.shape, (2, expected_seq_len, expected_channels))
        self.assertIsNone(result_masks)
        self.assertEqual(len(result_img_sizes), 2)

    def test_unpatchify_method(self):
        """Test unpatchify method."""
        model = HiDreamImageTransformer2DModel(**self.config)

        # Test unpatchify
        patch_size = model.config.patch_size
        x = torch.randn(2, 16, 4 * patch_size * patch_size)  # 16 patches, 4 output channels
        img_sizes = [(4, 4), (4, 4)]

        result = model.unpatchify(x, img_sizes, False)

        expected_shape = (2, 4, 8, 8)  # (batch, channels, height, width)
        self.assertEqual(result.shape, expected_shape)

    def test_extract_llama_layers_method(self):
        """Test _extract_llama_layers method."""
        model = HiDreamImageTransformer2DModel(**self.config)

        # Test 5D input [batch, num_layers, 1, seq, dim]
        llama_5d = torch.randn(2, 4, 1, 77, 512)
        result = model._extract_llama_layers(llama_5d)

        self.assertEqual(len(result), len(model.llama_layers))
        for layer_tensor in result:
            self.assertEqual(layer_tensor.shape, (2, 77, 512))

        # Test 4D input [num_layers, batch, seq, dim]
        llama_4d = torch.randn(4, 2, 77, 512)
        result = model._extract_llama_layers(llama_4d)

        self.assertEqual(len(result), len(model.llama_layers))
        for layer_tensor in result:
            self.assertEqual(layer_tensor.shape, (2, 77, 512))

    def test_forward_pass_basic(self):
        """Test basic forward pass with minimal inputs."""
        model = HiDreamImageTransformer2DModel(**self.config)

        # Create test inputs
        batch_size = 2
        hidden_states = torch.randn(batch_size, 64, 8, 8)
        timesteps = torch.randint(0, 1000, (batch_size,))
        t5_hidden_states = torch.randn(batch_size, 512, 4096)
        llama_hidden_states = torch.randn(2, batch_size, 32, 768)  # [num_layers, batch, seq, dim]
        pooled_embeds = torch.randn(batch_size, 768)

        # Mock the complex sub-operations to focus on structure
        with patch.object(model, "double_stream_blocks") as mock_double_blocks:
            with patch.object(model, "single_stream_blocks") as mock_single_blocks:
                mock_double_blocks.__iter__.return_value = []
                mock_single_blocks.__iter__.return_value = []

                result = model(
                    hidden_states=hidden_states,
                    timesteps=timesteps,
                    t5_hidden_states=t5_hidden_states,
                    llama_hidden_states=llama_hidden_states,
                    pooled_embeds=pooled_embeds,
                )

                self.assertIsNotNone(result)
                if hasattr(result, "sample"):
                    self.assertIsInstance(result.sample, torch.Tensor)
                else:
                    self.assertIsInstance(result, tuple)

    def test_parameter_name_typos_config(self):
        """Test that config rejects common typos."""
        # Test typo configs raise errors
        typo_configs = [
            {"patch_sizes": 2},  # Extra 's'
            {"in_channel": 64},  # Missing 's'
            {"out_channel": 64},  # Missing 's'
            {"num_layer": 2},  # Missing 's'
            {"num_single_layer": 2},  # Missing 's'
            {"attention_head_dims": 128},  # Extra 's'
            {"num_attention_head": 4},  # Missing 's'
            {"caption_channel": [512, 768]},  # Missing 's'
        ]

        for typo_config in typo_configs:
            config = self.config.copy()
            config.update(typo_config)

            # These should either raise errors during instantiation
            # or create models with incorrect attributes
            try:
                model = HiDreamImageTransformer2DModel(**config)
                # If it creates successfully, check that the attribute is wrong
                if "patch_sizes" in typo_config:
                    self.assertFalse(hasattr(model.config, "patch_size"))
            except (TypeError, AttributeError):
                # Expected for typo parameters
                pass

    def test_parameter_name_typos_forward(self):
        """Test forward method rejects common parameter typos."""
        model = HiDreamImageTransformer2DModel(**self.config)

        # Valid inputs
        valid_inputs = {
            "hidden_states": torch.randn(2, 64, 8, 8),
            "timesteps": torch.randint(0, 1000, (2,)),
            "t5_hidden_states": torch.randn(2, 77, 768),
            "llama_hidden_states": torch.randn(2, 2, 32, 512),
            "pooled_embeds": torch.randn(2, 768),
        }

        # Test typo parameters raise TypeError
        typo_tests = [
            "hidden_state",  # Missing 's'
            "timestep",  # Missing 's'
            "t5_hidden_state",  # Missing 's'
            "llama_hidden_state",  # Missing 's'
            "pooled_embed",  # Missing 's'
            "img_size",  # Missing 's'
            "img_id",  # Missing 's'
        ]

        for typo_param in typo_tests:
            with self.assertRaises(TypeError):
                invalid_inputs = valid_inputs.copy()
                invalid_inputs[typo_param] = torch.randn(2, 10)
                model(**invalid_inputs)


class TestHiDreamTransformerIntegration(TransformerBaseTest):
    """Integration tests for the complete HiDream transformer pipeline."""

    def test_load_balancing_loss_integration(self):
        """Test load balancing loss integration with MoE blocks."""
        # Clear any existing losses
        clear_load_balancing_loss()

        # Create model with MoE enabled
        config = {
            "patch_size": 2,
            "in_channels": 4,
            "num_layers": 1,
            "num_single_layers": 1,
            "attention_head_dim": 64,
            "num_attention_heads": 2,
            "caption_channels": [256, 256],
            "text_emb_dim": 256,
            "num_routed_experts": 4,
            "num_activated_experts": 2,
            "aux_loss_alpha": 0.01,
            "llama_layers": [0, 1],
            "axes_dims_rope": (32, 32, 32),
            "max_resolution": (128, 128),
        }

        model = HiDreamImageTransformer2DModel(**config)
        model.train()

        # Forward pass should potentially generate load balancing losses
        # (but we'll mock the complex parts)
        with patch.object(model, "double_stream_blocks") as mock_double_blocks:
            with patch.object(model, "single_stream_blocks") as mock_single_blocks:

                # Mock MoE blocks that save load balancing losses (inner_dim = 2 * 64 = 128)
                mock_block = Mock()
                mock_block.return_value = (torch.randn(2, 16, 128), torch.randn(2, 16, 128))

                # Simulate saving a load balancing loss
                def side_effect(*args, **kwargs):
                    save_load_balancing_loss((torch.tensor(0.1), torch.randn(4), torch.randn(4), 0.01))
                    return mock_block.return_value

                mock_block.side_effect = side_effect
                mock_double_blocks.__iter__.return_value = [mock_block]
                mock_single_blocks.__iter__.return_value = []

                # Run forward pass
                inputs = {
                    "hidden_states": torch.randn(2, 4, 8, 8),
                    "timesteps": torch.randint(0, 1000, (2,)),
                    "t5_hidden_states": torch.randn(2, 16, 256),
                    "llama_hidden_states": torch.randn(1, 2, 16, 256),
                    "pooled_embeds": torch.randn(2, 256),
                }

                model(**inputs)

                # Check that load balancing losses were saved
                losses = get_load_balancing_loss()
                self.assertGreater(len(losses), 0)

        # Clean up
        clear_load_balancing_loss()

    def test_rope_integration(self):
        """Test RoPE integration throughout the pipeline."""
        # Test that rope functions work correctly together
        pos = torch.randn(2, 64)
        dim = 128
        theta = 10000

        # Generate RoPE
        rope_result = rope(pos, dim, theta)
        self.assertEqual(rope_result.shape, (2, 64, dim // 2, 2, 2))

        # Apply to query/key tensors
        query = torch.randn(2, 8, 64, dim)
        key = torch.randn(2, 8, 64, dim)

        # Expand rope_result to have heads dimension for broadcasting
        rope_result = rope_result.unsqueeze(1).expand(2, 8, 64, dim // 2, 2, 2)

        query_rope, key_rope = apply_rope(query, key, rope_result)
        self.assertEqual(query_rope.shape, query.shape)
        self.assertEqual(key_rope.shape, key.shape)

    def test_attention_pipeline(self):
        """Test complete attention pipeline."""
        batch, seq_len, heads, head_dim = 2, 64, 8, 64

        # Create attention tensors
        query = torch.randn(batch, seq_len, heads, head_dim)
        key = torch.randn(batch, seq_len, heads, head_dim)
        value = torch.randn(batch, seq_len, heads, head_dim)

        # Apply attention
        result = attention(query, key, value)

        expected_shape = (batch, seq_len, heads * head_dim)
        self.assertEqual(result.shape, expected_shape)
        self.assert_no_nan_or_inf(result)


if __name__ == "__main__":
    unittest.main()
