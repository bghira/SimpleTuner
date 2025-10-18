"""
Comprehensive unit tests for Qwen Image transformer components.

This module tests the Qwen-Image transformer architecture including:
- QwenTimestepProjEmbeddings: Timestep projection and embedding
- QwenEmbedRope: Rotary position embeddings for Qwen architecture
- QwenDoubleStreamAttnProcessor2_0: Joint attention for text and image streams
- QwenImageTransformerBlock: Dual-stream transformer block
- QwenImageTransformer2DModel: Complete Qwen transformer model

Focus areas:
- Typo prevention in parameter names, method names, tensor operations
- Edge case handling (empty inputs, None values, device compatibility)
- Shape validation and mathematical correctness
- Architecture-specific features (double stream attention, rope embeddings)
- Performance benchmarking
- TREAD router integration
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

from transformer_base_test import (
    AttentionProcessorTestMixin,
    EmbeddingTestMixin,
    TransformerBaseTest,
    TransformerBlockTestMixin,
)
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
from simpletuner.helpers.models.qwen_image.transformer import (
    QwenDoubleStreamAttnProcessor2_0,
    QwenEmbedRope,
    QwenImageTransformer2DModel,
    QwenImageTransformerBlock,
    QwenTimestepProjEmbeddings,
    apply_rotary_emb_qwen,
    get_timestep_embedding,
)
from diffusers.utils import logging as diffusers_logging
from diffusers.utils.testing_utils import CaptureLogger


class TestGetTimestepEmbedding(TransformerBaseTest):
    """Test the get_timestep_embedding function."""

    def test_basic_functionality(self):
        """Test basic timestep embedding generation."""
        timesteps = torch.tensor([0, 100, 500], dtype=torch.long)
        embedding_dim = 128

        embeddings = get_timestep_embedding(timesteps, embedding_dim)

        # Validate output shape and properties
        self.assert_tensor_shape(embeddings, (3, 128))
        self.assert_no_nan_or_inf(embeddings)
        self.assertEqual(embeddings.dtype, torch.float32)

    def test_flip_sin_to_cos_parameter(self):
        """Test flip_sin_to_cos parameter functionality."""
        timesteps = torch.tensor([100], dtype=torch.long)
        embedding_dim = 64

        emb_normal = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False)
        emb_flipped = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=True)

        # Should be different when flipped
        self.assertFalse(torch.allclose(emb_normal, emb_flipped))

        # Shape should be the same
        self.assertEqual(emb_normal.shape, emb_flipped.shape)

    def test_odd_embedding_dim(self):
        """Test with odd embedding dimensions (should zero pad)."""
        timesteps = torch.tensor([100], dtype=torch.long)
        embedding_dim = 65  # Odd number

        embeddings = get_timestep_embedding(timesteps, embedding_dim)

        # Should pad to match requested dimension
        self.assert_tensor_shape(embeddings, (1, 65))
        self.assert_no_nan_or_inf(embeddings)

    def test_scale_parameter(self):
        """Test scale parameter affects output magnitude."""
        timesteps = torch.tensor([100], dtype=torch.long)
        embedding_dim = 64

        emb_scale_1 = get_timestep_embedding(timesteps, embedding_dim, scale=1.0)
        emb_scale_2 = get_timestep_embedding(timesteps, embedding_dim, scale=2.0)

        # Scaled version should have larger magnitude
        self.assertFalse(torch.allclose(emb_scale_2, emb_scale_1))

    def test_downscale_freq_shift(self):
        """Test downscale_freq_shift parameter."""
        timesteps = torch.tensor([100], dtype=torch.long)
        embedding_dim = 64

        emb_shift_0 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=0)
        emb_shift_1 = get_timestep_embedding(timesteps, embedding_dim, downscale_freq_shift=1)

        # Should produce different embeddings
        self.assertFalse(torch.allclose(emb_shift_0, emb_shift_1))

    def test_max_period_parameter(self):
        """Test max_period parameter affects frequency range."""
        timesteps = torch.tensor([100], dtype=torch.long)
        embedding_dim = 64

        emb_period_1000 = get_timestep_embedding(timesteps, embedding_dim, max_period=1000)
        emb_period_10000 = get_timestep_embedding(timesteps, embedding_dim, max_period=10000)

        # Different max_period should produce different embeddings
        self.assertFalse(torch.allclose(emb_period_1000, emb_period_10000))

    def test_shape_validation(self):
        """Test proper shape validation and error handling."""
        embedding_dim = 64

        # Test 1D timesteps (should work)
        timesteps_1d = torch.tensor([100], dtype=torch.long)
        embeddings = get_timestep_embedding(timesteps_1d, embedding_dim)
        self.assert_tensor_shape(embeddings, (1, 64))

        # Test 2D timesteps (should fail)
        timesteps_2d = torch.tensor([[100]], dtype=torch.long)
        with self.assertRaises(AssertionError):
            get_timestep_embedding(timesteps_2d, embedding_dim)

    def test_device_consistency(self):
        """Test device consistency between input and output."""
        if torch.cuda.is_available():
            timesteps = torch.tensor([100], dtype=torch.long, device="cuda")
            embeddings = get_timestep_embedding(timesteps, 64)
            self.assertEqual(embeddings.device, timesteps.device)

    def test_typo_prevention(self):
        """Test for common parameter name typos."""
        timesteps = torch.tensor([100], dtype=torch.long)

        # Test correct parameter names work
        try:
            get_timestep_embedding(
                timesteps=timesteps,
                embedding_dim=64,
                flip_sin_to_cos=True,
                downscale_freq_shift=1.0,
                scale=1.0,
                max_period=10000,
            )
        except TypeError as e:
            self.fail(f"Function should accept valid parameter names: {e}")

        # Test common typos should fail
        typo_tests = [
            {"timestep": timesteps, "embedding_dim": 64},  # timestep vs timesteps
            {"timesteps": timesteps, "embed_dim": 64},  # embed_dim vs embedding_dim
            {"timesteps": timesteps, "embedding_dim": 64, "flip_cos_sin": True},  # flip_cos_sin vs flip_sin_to_cos
        ]

        for invalid_kwargs in typo_tests:
            with self.assertRaises(TypeError):
                get_timestep_embedding(**invalid_kwargs)


class TestApplyRotaryEmbQwen(TransformerBaseTest):
    """Test the apply_rotary_emb_qwen function."""

    def test_real_mode_default(self):
        """Test real mode with default unbind dimension."""
        batch_size, seq_len, heads, head_dim = 2, 128, 8, 64
        x = torch.randn(batch_size, seq_len, heads, head_dim)
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)
        freqs_cis = (cos, sin)

        output = apply_rotary_emb_qwen(x, freqs_cis, use_real=True, use_real_unbind_dim=-1)

        self.assert_tensor_shape(output, x.shape)
        self.assertEqual(output.dtype, x.dtype)
        self.assert_no_nan_or_inf(output)

    def test_real_mode_unbind_dim_minus_2(self):
        """Test real mode with unbind dimension -2."""
        batch_size, seq_len, heads, head_dim = 2, 128, 8, 64
        x = torch.randn(batch_size, seq_len, heads, head_dim)
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)
        freqs_cis = (cos, sin)

        output = apply_rotary_emb_qwen(x, freqs_cis, use_real=True, use_real_unbind_dim=-2)

        self.assert_tensor_shape(output, x.shape)
        self.assertEqual(output.dtype, x.dtype)
        self.assert_no_nan_or_inf(output)

    def test_complex_mode(self):
        """Test complex mode operation."""
        batch_size, seq_len, heads, head_dim = 2, 128, 8, 64
        x = torch.randn(batch_size, seq_len, heads, head_dim)
        # Complex freqs_cis for complex mode
        freqs_cis = torch.randn(seq_len, head_dim // 2, dtype=torch.complex64)

        output = apply_rotary_emb_qwen(x, freqs_cis, use_real=False)

        self.assert_tensor_shape(output, x.shape)
        self.assertEqual(output.dtype, x.dtype)
        self.assert_no_nan_or_inf(output)

    def test_invalid_unbind_dim(self):
        """Test invalid unbind dimension raises error."""
        x = torch.randn(2, 128, 8, 64)
        freqs_cis = (torch.randn(128, 64), torch.randn(128, 64))

        with self.assertRaises(ValueError) as context:
            apply_rotary_emb_qwen(x, freqs_cis, use_real=True, use_real_unbind_dim=0)

        self.assertIn("use_real_unbind_dim", str(context.exception))

    def test_device_consistency(self):
        """Test device consistency for cos/sin tensors."""
        if torch.cuda.is_available():
            x = torch.randn(2, 128, 8, 64, device="cuda")
            cos = torch.randn(128, 64)  # CPU tensors
            sin = torch.randn(128, 64)
            freqs_cis = (cos, sin)

            # Should handle device mismatch by moving cos/sin to x.device
            output = apply_rotary_emb_qwen(x, freqs_cis, use_real=True)
            self.assertEqual(output.device, x.device)

    def test_dtype_preservation(self):
        """Test output dtype matches input dtype."""
        dtypes_to_test = [torch.float32, torch.float16]

        for dtype in dtypes_to_test:
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue  # Skip float16 on CPU

            device = "cuda" if dtype == torch.float16 else "cpu"
            x = torch.randn(2, 128, 8, 64, dtype=dtype, device=device)
            cos = torch.randn(128, 64, device=device)
            sin = torch.randn(128, 64, device=device)
            freqs_cis = (cos, sin)

            output = apply_rotary_emb_qwen(x, freqs_cis, use_real=True)
            self.assertEqual(output.dtype, dtype)

    def test_mathematical_correctness(self):
        """Test mathematical correctness of rotation."""
        # Simple test case to verify rotation mathematics
        x = torch.ones(1, 4, 1, 4)  # Simple tensor for easy verification
        cos = torch.ones(4, 4) * 0.5
        sin = torch.ones(4, 4) * 0.5
        freqs_cis = (cos, sin)

        output = apply_rotary_emb_qwen(x, freqs_cis, use_real=True)

        # Output should be finite and reasonable
        self.assertTrue(torch.isfinite(output).all())
        self.assertLess(output.abs().max(), 10.0)  # Reasonable magnitude

    def test_typo_prevention(self):
        """Test common parameter name typos."""
        x = torch.randn(2, 128, 8, 64)
        freqs_cis = (torch.randn(128, 64), torch.randn(128, 64))

        # Test correct parameters work
        try:
            apply_rotary_emb_qwen(x=x, freqs_cis=freqs_cis, use_real=True, use_real_unbind_dim=-1)
        except TypeError as e:
            self.fail(f"Function should accept valid parameter names: {e}")


class TestQwenTimestepProjEmbeddings(TransformerBaseTest, EmbeddingTestMixin):
    """Test QwenTimestepProjEmbeddings class."""

    def test_instantiation(self):
        """Test basic instantiation."""
        embedding_dim = 512
        module = QwenTimestepProjEmbeddings(embedding_dim)

        self.assertIsInstance(module, nn.Module)
        self.assertTrue(hasattr(module, "time_proj"))
        self.assertTrue(hasattr(module, "timestep_embedder"))

    def test_forward_pass(self):
        """Test forward pass functionality."""
        embedding_dim = 512
        module = QwenTimestepProjEmbeddings(embedding_dim)

        timestep = self.timestep
        hidden_states = self.hidden_states

        with torch.no_grad():
            output = module(timestep, hidden_states)

        # Output should have correct shape [batch_size, embedding_dim]
        self.assert_tensor_shape(output, (self.batch_size, embedding_dim))
        self.assert_no_nan_or_inf(output)
        self.assertEqual(output.dtype, hidden_states.dtype)

    def test_different_embedding_dims(self):
        """Test with different embedding dimensions."""
        test_dims = [256, 512, 1024, 2048]

        for embedding_dim in test_dims:
            with self.subTest(embedding_dim=embedding_dim):
                module = QwenTimestepProjEmbeddings(embedding_dim)
                output = module(self.timestep, self.hidden_states)
                self.assert_tensor_shape(output, (self.batch_size, embedding_dim))

    def test_dtype_consistency(self):
        """Test dtype consistency between input and output."""
        module = QwenTimestepProjEmbeddings(512)

        # Test with different dtypes
        dtypes = [torch.float32, torch.float16]
        for dtype in dtypes:
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue

            device = "cuda" if dtype == torch.float16 else "cpu"
            timestep = torch.randint(0, 1000, (2,), device=device)
            hidden_states = torch.randn(2, 128, 512, dtype=dtype, device=device)

            with torch.no_grad():
                output = module(timestep, hidden_states)

            self.assertEqual(output.dtype, dtype)

    def test_batch_size_variations(self):
        """Test with different batch sizes."""
        module = QwenTimestepProjEmbeddings(512)

        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                timestep = torch.randint(0, 1000, (batch_size,))
                hidden_states = torch.randn(batch_size, 128, 512)

                output = module(timestep, hidden_states)
                self.assert_tensor_shape(output, (batch_size, 512))

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        module = QwenTimestepProjEmbeddings(512)

        # Test with zero timesteps
        zero_timestep = torch.zeros(2, dtype=torch.long)
        output = module(zero_timestep, self.hidden_states)
        self.assert_no_nan_or_inf(output)

        # Test with maximum timesteps
        max_timestep = torch.full((2,), 1000, dtype=torch.long)
        output = module(max_timestep, self.hidden_states)
        self.assert_no_nan_or_inf(output)

    def test_typo_prevention(self):
        """Test parameter name typos."""
        module = QwenTimestepProjEmbeddings(512)

        # Test correct parameter names
        try:
            output = module.forward(timestep=self.timestep, hidden_states=self.hidden_states)
            self.assertIsNotNone(output)
        except TypeError as e:
            self.fail(f"Should accept valid parameter names: {e}")

        # Test method existence
        self.run_method_existence_tests(module, ["forward"])

    def test_component_attributes(self):
        """Test internal component attributes."""
        module = QwenTimestepProjEmbeddings(512)

        # Check time_proj configuration
        self.assertEqual(module.time_proj.num_channels, 256)
        self.assertTrue(module.time_proj.flip_sin_to_cos)
        self.assertEqual(module.time_proj.downscale_freq_shift, 0)

        # Check timestep_embedder configuration
        self.assertEqual(module.timestep_embedder.time_embed_dim, 512)


class TestQwenEmbedRope(TransformerBaseTest):
    """Test QwenEmbedRope class."""

    def setUp(self):
        super().setUp()
        self.theta = 10000
        self.axes_dim = [16, 56, 56]  # Typical dimensions for frame, height, width

    def test_instantiation(self):
        """Test basic instantiation."""
        module = QwenEmbedRope(self.theta, self.axes_dim)

        self.assertIsInstance(module, nn.Module)
        self.assertEqual(module.theta, self.theta)
        self.assertEqual(module.axes_dim, self.axes_dim)
        self.assertEqual(module._current_max_len, 1024)

    def test_rope_params_generation(self):
        """Test rope_params method generates correct frequencies."""
        module = QwenEmbedRope(self.theta, self.axes_dim)

        index = torch.arange(10)
        dim = 16

        freqs = module.rope_params(index, dim, self.theta)

        # Should return complex tensor with correct shape
        self.assert_tensor_shape(freqs, (10, dim // 2))
        self.assertTrue(freqs.dtype.is_complex)

    def test_forward_pass_single_video(self):
        """Test forward pass with single video configuration."""
        module = QwenEmbedRope(self.theta, self.axes_dim)

        # Single video: frame=4, height=32, width=32
        video_fhw = [(4, 32, 32)]
        txt_seq_lens = [77]
        device = "cpu"

        vid_freqs, txt_freqs = module(video_fhw, txt_seq_lens, device)

        # Check shapes
        expected_vid_len = 4 * 32 * 32  # frame * height * width
        expected_rope_dim = sum(self.axes_dim) // 2

        self.assert_tensor_shape(vid_freqs, (expected_vid_len, expected_rope_dim))
        self.assert_tensor_shape(txt_freqs, (77, expected_rope_dim))

    def test_forward_pass_multiple_videos(self):
        """Test forward pass with multiple video configurations."""
        module = QwenEmbedRope(self.theta, self.axes_dim)

        # Multiple videos with different sizes
        video_fhw = [(4, 32, 32), (2, 16, 16)]
        txt_seq_lens = [77, 128]
        device = "cpu"

        vid_freqs, txt_freqs = module(video_fhw, txt_seq_lens, device)

        # Check concatenated video frequencies
        expected_vid_len = (4 * 32 * 32) + (2 * 16 * 16)
        expected_rope_dim = sum(self.axes_dim) // 2
        max_txt_len = max(txt_seq_lens)

        self.assert_tensor_shape(vid_freqs, (expected_vid_len, expected_rope_dim))
        self.assert_tensor_shape(txt_freqs, (max_txt_len, expected_rope_dim))

    def test_scale_rope_parameter(self):
        """Test scale_rope parameter affects computation."""
        module_no_scale = QwenEmbedRope(self.theta, self.axes_dim, scale_rope=False)
        module_with_scale = QwenEmbedRope(self.theta, self.axes_dim, scale_rope=True)

        video_fhw = [(4, 32, 32)]
        txt_seq_lens = [77]
        device = "cpu"

        vid_freqs_no_scale, _ = module_no_scale(video_fhw, txt_seq_lens, device)
        vid_freqs_with_scale, _ = module_with_scale(video_fhw, txt_seq_lens, device)

        # Should produce different results
        self.assertFalse(torch.allclose(vid_freqs_no_scale, vid_freqs_with_scale))

    def test_caching_behavior(self):
        """Test rope caching behavior."""
        module = QwenEmbedRope(self.theta, self.axes_dim)
        module._compute_video_freqs.cache_clear()

        video_fhw = [(4, 32, 32)]
        txt_seq_lens = [77]
        device = "cpu"

        cache_info_before = module._compute_video_freqs.cache_info()
        module(video_fhw, txt_seq_lens, device)
        cache_info_after_first = module._compute_video_freqs.cache_info()
        module(video_fhw, txt_seq_lens, device)
        cache_info_after_second = module._compute_video_freqs.cache_info()

        self.assertEqual(cache_info_before.currsize, 0)
        self.assertGreater(cache_info_after_first.currsize, cache_info_before.currsize)
        self.assertEqual(cache_info_after_second.currsize, cache_info_after_first.currsize)

    def test_dynamic_expansion_increases_capacity(self):
        """Ensure long prompts trigger dynamic RoPE expansion."""
        module = QwenEmbedRope(self.theta, self.axes_dim)

        video_fhw = [(1, 32, 32)]
        txt_len = module._current_max_len + 200
        txt_seq_lens = [txt_len]

        _, txt_freqs = module(video_fhw, txt_seq_lens, device="cpu")
        required_len = 32 + txt_len

        self.assertGreaterEqual(module._current_max_len, required_len)
        self.assertEqual(txt_freqs.shape[0], txt_len)
        self.assertGreaterEqual(module.pos_freqs.shape[0], module._current_max_len)

    def test_long_prompt_warning(self):
        """Verify that a warning is emitted when prompts exceed training limits."""
        module = QwenEmbedRope(self.theta, self.axes_dim)
        video_fhw = [(1, 32, 32)]
        txt_len = module._current_max_len + 600
        txt_seq_lens = [txt_len]

        logger = diffusers_logging.get_logger("simpletuner.helpers.models.qwen_image.transformer")
        logger.setLevel(diffusers_logging.WARNING)

        with patch.object(logger, "warning") as mock_warning:
            module(video_fhw, txt_seq_lens, device="cpu")

        warnings = [str(call.args[0]) for call in mock_warning.call_args_list if call.args]
        concatenated = " ".join(warnings)
        self.assertIn("512 tokens", concatenated)
        self.assertIn("unpredictable behavior", concatenated)

    def test_device_handling(self):
        """Test device handling and tensor movement."""
        if torch.cuda.is_available():
            module = QwenEmbedRope(self.theta, self.axes_dim)

            video_fhw = [(4, 32, 32)]
            txt_seq_lens = [77]
            device = "cuda"

            vid_freqs, txt_freqs = module(video_fhw, txt_seq_lens, device)

            self.assertEqual(vid_freqs.device.type, "cuda")
            self.assertEqual(txt_freqs.device.type, "cuda")

    def test_input_format_variations(self):
        """Test different input format variations."""
        module = QwenEmbedRope(self.theta, self.axes_dim)

        # Test single tuple instead of list
        video_fhw = (4, 32, 32)  # Single tuple
        txt_seq_lens = [77]
        device = "cpu"

        vid_freqs, txt_freqs = module(video_fhw, txt_seq_lens, device)
        self.assertIsInstance(vid_freqs, torch.Tensor)
        self.assertIsInstance(txt_freqs, torch.Tensor)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        module = QwenEmbedRope(self.theta, self.axes_dim)

        # Test minimum sizes
        video_fhw = [(1, 1, 1)]
        txt_seq_lens = [1]
        device = "cpu"

        vid_freqs, txt_freqs = module(video_fhw, txt_seq_lens, device)
        self.assert_no_nan_or_inf(vid_freqs)
        self.assert_no_nan_or_inf(txt_freqs)

        # Test larger sizes
        video_fhw = [(8, 64, 64)]
        txt_seq_lens = [256]

        vid_freqs, txt_freqs = module(video_fhw, txt_seq_lens, device)
        self.assert_no_nan_or_inf(vid_freqs)
        self.assert_no_nan_or_inf(txt_freqs)

    def test_typo_prevention(self):
        """Test parameter name typos."""
        # Test correct instantiation parameters
        try:
            module = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
            self.assertIsNotNone(module)
        except TypeError as e:
            self.fail(f"Should accept valid parameter names: {e}")

        # Test forward method parameters
        module = QwenEmbedRope(self.theta, self.axes_dim)

        try:
            module.forward(video_fhw=[(4, 32, 32)], txt_seq_lens=[77], device="cpu")
        except TypeError as e:
            self.fail(f"Should accept valid parameter names: {e}")


class TestQwenDoubleStreamAttnProcessor2_0(TransformerBaseTest, AttentionProcessorTestMixin):
    """Test QwenDoubleStreamAttnProcessor2_0 class."""

    def setUp(self):
        super().setUp()
        self.processor = QwenDoubleStreamAttnProcessor2_0()
        self._setup_mock_attention()

    def _setup_mock_attention(self):
        """Set up mock attention module for testing."""
        self.mock_attn = Mock()
        self.mock_attn.heads = 8

        # Mock projection layers
        self.mock_attn.to_q = Mock(side_effect=lambda x, *_, **__: torch.zeros_like(x))
        self.mock_attn.to_k = Mock(side_effect=lambda x, *_, **__: torch.zeros_like(x))
        self.mock_attn.to_v = Mock(side_effect=lambda x, *_, **__: torch.zeros_like(x))

        # Mock added projections for text stream
        self.mock_attn.add_q_proj = Mock(side_effect=lambda x, *_, **__: torch.zeros_like(x))
        self.mock_attn.add_k_proj = Mock(side_effect=lambda x, *_, **__: torch.zeros_like(x))
        self.mock_attn.add_v_proj = Mock(side_effect=lambda x, *_, **__: torch.zeros_like(x))

        # Mock output projections
        self.mock_attn.to_out = [Mock(side_effect=lambda x, *_, **__: torch.zeros_like(x))]
        self.mock_attn.to_add_out = Mock(side_effect=lambda x, *_, **__: torch.zeros_like(x))

        # Mock normalization layers (optional)
        self.mock_attn.norm_q = None
        self.mock_attn.norm_k = None
        self.mock_attn.norm_added_q = None
        self.mock_attn.norm_added_k = None

    def test_instantiation(self):
        """Test basic instantiation."""
        processor = QwenDoubleStreamAttnProcessor2_0()
        self.assertIsInstance(processor, QwenDoubleStreamAttnProcessor2_0)
        self.assertTrue(callable(processor))

    def test_pytorch_version_check(self):
        """Test PyTorch version compatibility check."""
        # Mock F.scaled_dot_product_attention to not exist
        with patch("torch.nn.functional.scaled_dot_product_attention", None, create=True):
            with patch.object(torch.nn.functional, "scaled_dot_product_attention", None):
                with self.assertRaises(ImportError) as context:
                    QwenDoubleStreamAttnProcessor2_0()
                self.assertIn("PyTorch 2.0", str(context.exception))

    @patch("simpletuner.helpers.models.qwen_image.transformer.dispatch_attention_fn")
    def test_forward_pass_basic(self, mock_dispatch):
        """Test basic forward pass functionality."""
        # Setup mock dispatch function
        mock_output = torch.randn(2, 205, 8, 64)  # batch, seq_txt + seq_img, heads, head_dim
        mock_dispatch.return_value = mock_output

        # Input tensors
        hidden_states = torch.randn(2, 128, 512)  # Image stream
        encoder_hidden_states = torch.randn(2, 77, 512)  # Text stream

        # Call processor
        img_output, txt_output = self.processor(
            attn=self.mock_attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states
        )

        # Validate outputs
        self.assert_tensor_shape(img_output, (2, 128, 512))
        self.assert_tensor_shape(txt_output, (2, 77, 512))

    def test_missing_encoder_hidden_states(self):
        """Test error when encoder_hidden_states is None."""
        hidden_states = torch.randn(2, 128, 512)

        with self.assertRaises(ValueError) as context:
            self.processor(attn=self.mock_attn, hidden_states=hidden_states, encoder_hidden_states=None)

        self.assertIn("encoder_hidden_states", str(context.exception))

    @patch("simpletuner.helpers.models.qwen_image.transformer.apply_rotary_emb_qwen")
    @patch("simpletuner.helpers.models.qwen_image.transformer.dispatch_attention_fn")
    def test_rotary_embedding_application(self, mock_dispatch, mock_rotary):
        """Test rotary embedding application when provided."""
        # Setup mocks
        mock_dispatch.return_value = torch.randn(2, 205, 8, 64)
        mock_rotary.return_value = torch.randn(2, 128, 8, 64)  # Shape after rotary

        # Create image rotary embeddings
        img_freqs = torch.randn(128, 32)
        txt_freqs = torch.randn(77, 32)
        image_rotary_emb = (img_freqs, txt_freqs)

        hidden_states = torch.randn(2, 128, 512)
        encoder_hidden_states = torch.randn(2, 77, 512)

        self.processor(
            attn=self.mock_attn,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Verify rotary embedding was applied to all Q and K
        self.assertEqual(mock_rotary.call_count, 4)  # img_q, img_k, txt_q, txt_k

    def test_normalization_layers(self):
        """Test with normalization layers present."""
        # Add mock normalization layers
        self.mock_attn.norm_q = Mock(side_effect=lambda x: x)
        self.mock_attn.norm_k = Mock(side_effect=lambda x: x)
        self.mock_attn.norm_added_q = Mock(side_effect=lambda x: x)
        self.mock_attn.norm_added_k = Mock(side_effect=lambda x: x)

        with patch("simpletuner.helpers.models.qwen_image.transformer.dispatch_attention_fn") as mock_dispatch:
            mock_dispatch.return_value = torch.randn(2, 205, 8, 64)

            hidden_states = torch.randn(2, 128, 512)
            encoder_hidden_states = torch.randn(2, 77, 512)

            self.processor(attn=self.mock_attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states)

            # Verify normalization was called
            self.mock_attn.norm_q.assert_called()
            self.mock_attn.norm_k.assert_called()
            self.mock_attn.norm_added_q.assert_called()
            self.mock_attn.norm_added_k.assert_called()

    def test_attention_mask_handling(self):
        """Test attention mask parameter handling."""
        with patch("simpletuner.helpers.models.qwen_image.transformer.dispatch_attention_fn") as mock_dispatch:
            mock_dispatch.return_value = torch.randn(2, 205, 8, 64)

            hidden_states = torch.randn(2, 128, 512)
            encoder_hidden_states = torch.randn(2, 77, 512)
            attention_mask = torch.ones(2, 205)  # Combined length

            self.processor(
                attn=self.mock_attn,
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
            )

            # Verify attention mask was passed to dispatch
            call_args = mock_dispatch.call_args
            self.assertIn("attn_mask", call_args.kwargs)

    def test_joint_attention_concatenation(self):
        """Test proper concatenation of text and image for joint attention."""
        with patch("simpletuner.helpers.models.qwen_image.transformer.dispatch_attention_fn") as mock_dispatch:
            # Create predictable output to verify splitting
            joint_output = torch.cat(
                [torch.ones(2, 77, 8, 64), torch.zeros(2, 128, 8, 64)], dim=1  # Text part  # Image part
            )
            mock_dispatch.return_value = joint_output

            hidden_states = torch.randn(2, 128, 512)
            encoder_hidden_states = torch.randn(2, 77, 512)

            img_output, txt_output = self.processor(
                attn=self.mock_attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states
            )

            # Verify correct splitting by checking call to dispatch_attention_fn
            call_args = mock_dispatch.call_args[0]
            joint_query = call_args[0]

            # Should concatenate [text, image] -> [batch, txt_seq + img_seq, heads, head_dim]
            expected_seq_len = 77 + 128
            self.assert_tensor_shape(joint_query, (2, expected_seq_len, 8, 64))

    def test_output_projection_handling(self):
        """Test output projection with and without dropout."""
        with patch("simpletuner.helpers.models.qwen_image.transformer.dispatch_attention_fn") as mock_dispatch:
            mock_dispatch.return_value = torch.randn(2, 205, 8, 64)

            # Test with dropout layer
            self.mock_attn.to_out = [
                Mock(return_value=torch.randn(2, 128, 512)),  # Linear layer
                Mock(return_value=torch.randn(2, 128, 512)),  # Dropout layer
            ]

            hidden_states = torch.randn(2, 128, 512)
            encoder_hidden_states = torch.randn(2, 77, 512)

            img_output, txt_output = self.processor(
                attn=self.mock_attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states
            )

            # Both projection layers should be called
            self.assertEqual(len(self.mock_attn.to_out), 2)
            for layer in self.mock_attn.to_out:
                layer.assert_called()

    def test_typo_prevention(self):
        """Test parameter name typos."""
        hidden_states = torch.randn(2, 128, 512)
        encoder_hidden_states = torch.randn(2, 77, 512)

        with patch("simpletuner.helpers.models.qwen_image.transformer.dispatch_attention_fn") as mock_dispatch:
            mock_dispatch.return_value = torch.randn(2, 205, 8, 64)

            # Test correct parameter names
            try:
                self.processor(
                    attn=self.mock_attn,
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=None,
                    attention_mask=None,
                    image_rotary_emb=None,
                )
            except TypeError as e:
                if "unexpected keyword argument" in str(e):
                    self.fail(f"Should accept valid parameter names: {e}")

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        with patch("simpletuner.helpers.models.qwen_image.transformer.dispatch_attention_fn") as mock_dispatch:
            mock_dispatch.return_value = torch.randn(2, 205, 8, 64)

            # Test with minimal sequence lengths
            hidden_states = torch.randn(2, 1, 512)  # Single token
            encoder_hidden_states = torch.randn(2, 1, 512)

            # Adjust mock returns for smaller sequences
            self.mock_attn.to_q.side_effect = lambda *_: torch.randn(2, 1, 512)
            self.mock_attn.to_k.side_effect = lambda *_: torch.randn(2, 1, 512)
            self.mock_attn.to_v.side_effect = lambda *_: torch.randn(2, 1, 512)
            self.mock_attn.add_q_proj.side_effect = lambda *_: torch.randn(2, 1, 512)
            self.mock_attn.add_k_proj.side_effect = lambda *_: torch.randn(2, 1, 512)
            self.mock_attn.add_v_proj.side_effect = lambda *_: torch.randn(2, 1, 512)

            mock_dispatch.return_value = torch.randn(2, 2, 8, 64)  # Combined seq_len = 2

            img_output, txt_output = self.processor(
                attn=self.mock_attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states
            )

            self.assert_tensor_shape(img_output, (2, 1, 512))
            self.assert_tensor_shape(txt_output, (2, 1, 512))


class TestQwenImageTransformerBlock(TransformerBaseTest, TransformerBlockTestMixin):
    """Test QwenImageTransformerBlock class."""

    def test_instantiation(self):
        """Test basic instantiation."""
        dim = 512
        num_attention_heads = 8
        attention_head_dim = 64

        block = QwenImageTransformerBlock(
            dim=dim, num_attention_heads=num_attention_heads, attention_head_dim=attention_head_dim
        )

        self.assertIsInstance(block, nn.Module)
        self.assertEqual(block.dim, dim)
        self.assertEqual(block.num_attention_heads, num_attention_heads)
        self.assertEqual(block.attention_head_dim, attention_head_dim)

    def test_component_initialization(self):
        """Test proper initialization of all components."""
        block = QwenImageTransformerBlock(512, 8, 64)

        # Check image components
        self.assertTrue(hasattr(block, "img_mod"))
        self.assertTrue(hasattr(block, "img_norm1"))
        self.assertTrue(hasattr(block, "img_norm2"))
        self.assertTrue(hasattr(block, "img_mlp"))

        # Check text components
        self.assertTrue(hasattr(block, "txt_mod"))
        self.assertTrue(hasattr(block, "txt_norm1"))
        self.assertTrue(hasattr(block, "txt_norm2"))
        self.assertTrue(hasattr(block, "txt_mlp"))

        # Check attention
        self.assertTrue(hasattr(block, "attn"))

    @patch("simpletuner.helpers.models.qwen_image.transformer.QwenDoubleStreamAttnProcessor2_0")
    def test_forward_pass_basic(self, mock_processor_class):
        """Test basic forward pass."""
        # Setup mock processor
        mock_processor = Mock()
        mock_processor.return_value = (torch.randn(2, 128, 512), torch.randn(2, 77, 512))  # Image output  # Text output
        mock_processor_class.return_value = mock_processor

        block = QwenImageTransformerBlock(512, 8, 64)

        # Mock the attention module's __call__ method to return tuple
        hidden_states = torch.randn(2, 128, 512)
        encoder_hidden_states = torch.randn(2, 77, 512)
        encoder_hidden_states_mask = torch.ones(2, 77)
        temb = torch.randn(2, 6 * 512)  # 6 * dim for modulation parameters

        with patch.object(block, "attn", MockAttention(return_tuple=True)):
            with torch.no_grad():
                enc_out, hidden_out = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                )

        self.assert_tensor_shape(enc_out, encoder_hidden_states.shape)
        self.assert_tensor_shape(hidden_out, hidden_states.shape)
        self.assert_no_nan_or_inf(enc_out)
        self.assert_no_nan_or_inf(hidden_out)

    def test_modulation_functionality(self):
        """Test _modulate method functionality."""
        block = QwenImageTransformerBlock(512, 8, 64)

        x = torch.randn(2, 128, 512)
        mod_params = torch.randn(2, 3 * 512)  # shift, scale, gate

        modulated, gate = block._modulate(x, mod_params)

        self.assert_tensor_shape(modulated, (2, 128, 512))
        self.assert_tensor_shape(gate, (2, 1, 512))
        self.assert_no_nan_or_inf(modulated)
        self.assert_no_nan_or_inf(gate)

    def test_fp16_clipping(self):
        """Test FP16 overflow prevention clipping."""
        block = QwenImageTransformerBlock(512, 8, 64)

        # Mock attention to return extreme values without triggering construction overflow
        def make_extreme(shape, value):
            tensor = torch.ones(shape, dtype=torch.float16)
            tensor *= value
            return tensor

        extreme_values = (
            make_extreme((2, 128, 512), 70000),  # Above clipping threshold
            make_extreme((2, 77, 512), -70000),  # Below clipping threshold
        )

        hidden_states = torch.randn(2, 128, 512, dtype=torch.float16)
        encoder_hidden_states = torch.randn(2, 77, 512, dtype=torch.float16)
        encoder_hidden_states_mask = torch.ones(2, 77)
        temb = torch.randn(2, 6 * 512, dtype=torch.float16)

        with patch.object(block, "attn", MockModule(extreme_values)):
            with torch.no_grad():
                enc_out, hidden_out = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                )

        # Values should be clipped to [-65504, 65504]
        self.assertLessEqual(hidden_out.max().item(), 65504)
        self.assertGreaterEqual(hidden_out.min().item(), -65504)
        self.assertLessEqual(enc_out.max().item(), 65504)
        self.assertGreaterEqual(enc_out.min().item(), -65504)

    def test_image_rotary_emb_parameter(self):
        """Test image_rotary_emb parameter handling."""
        block = QwenImageTransformerBlock(512, 8, 64)

        # Mock attention
        hidden_states = torch.randn(2, 128, 512)
        encoder_hidden_states = torch.randn(2, 77, 512)
        encoder_hidden_states_mask = torch.ones(2, 77)
        temb = torch.randn(2, 6 * 512)

        # Create image rotary embeddings
        img_freqs = torch.randn(128, 32)
        txt_freqs = torch.randn(77, 32)
        image_rotary_emb = (img_freqs, txt_freqs)

        with patch.object(block, "attn", MockAttention(return_tuple=True)):
            with torch.no_grad():
                enc_out, hidden_out = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                )

        # Verify image_rotary_emb was passed to attention
        call_kwargs = block.attn.call_args.kwargs
        self.assertIn("image_rotary_emb", call_kwargs)

    def test_joint_attention_kwargs(self):
        """Test joint_attention_kwargs parameter handling."""
        block = QwenImageTransformerBlock(512, 8, 64)

        # Mock attention
        hidden_states = torch.randn(2, 128, 512)
        encoder_hidden_states = torch.randn(2, 77, 512)
        encoder_hidden_states_mask = torch.ones(2, 77)
        temb = torch.randn(2, 6 * 512)
        joint_attention_kwargs = {"scale": 1.0}

        with patch.object(block, "attn", MockAttention(return_tuple=True)):
            with torch.no_grad():
                block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

        # Verify kwargs were passed to attention
        call_kwargs = block.attn.call_args.kwargs
        self.assertIn("scale", call_kwargs)

    def test_different_qk_norm_options(self):
        """Test different QK normalization options."""
        qk_norm_options = ["rms_norm", "layer_norm", None]

        for qk_norm in qk_norm_options:
            with self.subTest(qk_norm=qk_norm):
                try:
                    block = QwenImageTransformerBlock(dim=512, num_attention_heads=8, attention_head_dim=64, qk_norm=qk_norm)
                    self.assertIsNotNone(block)
                except Exception as e:
                    self.fail(f"Failed to create block with qk_norm={qk_norm}: {e}")

    def test_typo_prevention(self):
        """Test parameter name typos."""
        # Test correct instantiation parameters
        try:
            block = QwenImageTransformerBlock(
                dim=512, num_attention_heads=8, attention_head_dim=64, qk_norm="rms_norm", eps=1e-6
            )
            self.assertIsNotNone(block)
        except TypeError as e:
            self.fail(f"Should accept valid parameter names: {e}")

        # Test method existence
        self.run_method_existence_tests(block, ["forward", "_modulate"])

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        block = QwenImageTransformerBlock(512, 8, 64)

        # Mock attention for edge case testing
        minimal_return = (torch.randn(2, 1, 512), torch.randn(2, 1, 512))  # Minimal sequence length

        # Test with minimal sequences
        hidden_states = torch.randn(2, 1, 512)
        encoder_hidden_states = torch.randn(2, 1, 512)
        encoder_hidden_states_mask = torch.ones(2, 1)
        temb = torch.randn(2, 6 * 512)

        with patch.object(block, "attn", MockModule(minimal_return)):
            with torch.no_grad():
                enc_out, hidden_out = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                )

        self.assert_tensor_shape(enc_out, encoder_hidden_states.shape)
        self.assert_tensor_shape(hidden_out, hidden_states.shape)


class TestQwenImageTransformer2DModel(TransformerBaseTest):
    """Test QwenImageTransformer2DModel class."""

    def setUp(self):
        super().setUp()
        self.patch_size = 2
        self.latent_channels = 4
        self.config = {
            "patch_size": self.patch_size,
            "in_channels": self.latent_channels * (self.patch_size**2),
            "out_channels": 4,
            "num_layers": 2,  # Reduced for testing
            "attention_head_dim": 64,
            "num_attention_heads": 8,
            "joint_attention_dim": 512,
            "axes_dims_rope": (16, 32, 32),
        }

    def _generate_packed_hidden_states(self, batch_size: int, height: int, width: int):
        """Create latent tensors and pack them into the Qwen patch format."""
        latents = torch.randn(batch_size, self.latent_channels, height, width)
        patch_size = self.patch_size
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("Height and width must be divisible by patch size for packing.")

        latents = latents.view(
            batch_size,
            self.latent_channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        packed = latents.reshape(
            batch_size,
            (height // patch_size) * (width // patch_size),
            self.latent_channels * (patch_size**2),
        )
        img_shapes = [(1, height // patch_size, width // patch_size)]
        return packed, img_shapes

    def test_instantiation(self):
        """Test basic instantiation."""
        model = QwenImageTransformer2DModel(**self.config)

        self.assertIsInstance(model, nn.Module)
        self.assertEqual(model.config.patch_size, self.patch_size)
        self.assertEqual(model.config.in_channels, self.config["in_channels"])
        self.assertEqual(model.config.num_layers, 2)

    def test_component_initialization(self):
        """Test proper initialization of all components."""
        model = QwenImageTransformer2DModel(**self.config)

        # Check main components
        self.assertTrue(hasattr(model, "pos_embed"))
        self.assertTrue(hasattr(model, "time_text_embed"))
        self.assertTrue(hasattr(model, "txt_norm"))
        self.assertTrue(hasattr(model, "img_in"))
        self.assertTrue(hasattr(model, "txt_in"))
        self.assertTrue(hasattr(model, "transformer_blocks"))
        self.assertTrue(hasattr(model, "norm_out"))
        self.assertTrue(hasattr(model, "proj_out"))

        # Check transformer blocks count
        self.assertEqual(len(model.transformer_blocks), self.config["num_layers"])

    @patch("simpletuner.helpers.models.qwen_image.transformer.QwenImageTransformerBlock")
    def test_forward_pass_basic(self, mock_block_class):
        """Test basic forward pass."""
        # Setup mock transformer blocks
        mock_block = Mock()
        mock_block.return_value = (
            torch.randn(2, 77, 512),  # encoder_hidden_states
            torch.randn(2, 256, 512),  # hidden_states (16x16 patches)
        )
        mock_block_class.return_value = mock_block

        model = QwenImageTransformer2DModel(**self.config)

        # Replace transformer blocks with mocks
        for i, block in enumerate(model.transformer_blocks):
            model.transformer_blocks[i] = mock_block

        # Input tensors
        batch_size = 2
        height, width = 32, 32  # Input image size

        hidden_states, img_shapes = self._generate_packed_hidden_states(batch_size, height, width)
        encoder_hidden_states = torch.randn(batch_size, 77, 512)  # Text tokens
        timestep = torch.randint(0, 1000, (batch_size,))
        txt_seq_lens = [77]

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
            )

        # Check output shape and properties
        if hasattr(output, "sample"):
            output_tensor = output.sample
        else:
            output_tensor = output

        # Output should match input spatial dimensions
        expected_shape = (batch_size, self.config["out_channels"], height, width)
        self.assert_tensor_shape(output_tensor, expected_shape)
        self.assert_no_nan_or_inf(output_tensor)

    def test_tread_router_integration(self):
        """Test TREAD router integration."""
        model = QwenImageTransformer2DModel(**self.config)

        # Initially no router
        self.assertIsNone(model._tread_router)
        self.assertIsNone(model._tread_routes)

        # Set router
        mock_router = Mock()
        routes = [{"start_layer_idx": 0, "end_layer_idx": 1, "selection_ratio": 0.5}]

        model.set_router(mock_router, routes)

        self.assertEqual(model._tread_router, mock_router)
        self.assertEqual(model._tread_routes, routes)

    def test_guidance_parameter(self):
        """Test guidance parameter handling."""
        model = QwenImageTransformer2DModel(**self.config)

        # Mock transformer blocks
        for i, block in enumerate(model.transformer_blocks):
            model.transformer_blocks[i] = Mock(return_value=(torch.randn(2, 77, 512), torch.randn(2, 256, 512)))

        hidden_states, img_shapes = self._generate_packed_hidden_states(2, 32, 32)
        encoder_hidden_states = torch.randn(2, 77, 512)
        timestep = torch.randint(0, 1000, (2,))
        guidance = torch.randn(2)  # Guidance values
        txt_seq_lens = [77]

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                guidance=guidance,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
            )

        # Should handle guidance without errors
        self.assertIsNotNone(output)

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing functionality."""
        model = QwenImageTransformer2DModel(**self.config)
        model.gradient_checkpointing = True

        # Mock transformer blocks
        for i, block in enumerate(model.transformer_blocks):
            model.transformer_blocks[i] = Mock(return_value=(torch.randn(2, 77, 512), torch.randn(2, 256, 512)))

        hidden_states, img_shapes = self._generate_packed_hidden_states(2, 32, 32)
        encoder_hidden_states = torch.randn(2, 77, 512)
        timestep = torch.randint(0, 1000, (2,))
        txt_seq_lens = [77]

        # Enable gradients to test checkpointing path
        hidden_states.requires_grad_(True)

        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
        )

        self.assertIsNotNone(output)

    def test_controlnet_integration(self):
        """Test ControlNet residual integration."""
        model = QwenImageTransformer2DModel(**self.config)

        # Mock transformer blocks
        for i, block in enumerate(model.transformer_blocks):
            model.transformer_blocks[i] = Mock(return_value=(torch.randn(2, 77, 512), torch.randn(2, 256, 512)))

        hidden_states, img_shapes = self._generate_packed_hidden_states(2, 32, 32)
        encoder_hidden_states = torch.randn(2, 77, 512)
        timestep = torch.randint(0, 1000, (2,))
        txt_seq_lens = [77]

        # Create ControlNet samples
        controlnet_block_samples = [
            torch.randn(2, 256, 512),  # One sample per block (will be repeated)
            torch.randn(2, 256, 512),
        ]

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                controlnet_block_samples=controlnet_block_samples,
            )

        self.assertIsNotNone(output)

    def test_attention_kwargs_handling(self):
        """Test attention_kwargs parameter handling including LoRA scale."""
        model = QwenImageTransformer2DModel(**self.config)

        # Mock transformer blocks
        for i, block in enumerate(model.transformer_blocks):
            model.transformer_blocks[i] = Mock(return_value=(torch.randn(2, 77, 512), torch.randn(2, 256, 512)))

        hidden_states, img_shapes = self._generate_packed_hidden_states(2, 32, 32)
        encoder_hidden_states = torch.randn(2, 77, 512)
        timestep = torch.randint(0, 1000, (2,))
        txt_seq_lens = [77]
        attention_kwargs = {"scale": 0.5}

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                attention_kwargs=attention_kwargs,
            )

        self.assertIsNotNone(output)

    def test_return_dict_parameter(self):
        """Test return_dict parameter controls output format."""
        model = QwenImageTransformer2DModel(**self.config)

        # Mock transformer blocks
        for i, block in enumerate(model.transformer_blocks):
            model.transformer_blocks[i] = Mock(return_value=(torch.randn(2, 77, 512), torch.randn(2, 256, 512)))

        hidden_states, img_shapes = self._generate_packed_hidden_states(2, 32, 32)
        encoder_hidden_states = torch.randn(2, 77, 512)
        timestep = torch.randint(0, 1000, (2,))
        txt_seq_lens = [77]

        # Test return_dict=True (default)
        with torch.no_grad():
            output_dict = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=True,
            )

        self.assertTrue(hasattr(output_dict, "sample"))

        # Test return_dict=False
        with torch.no_grad():
            output_tuple = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )

        self.assertIsInstance(output_tuple, tuple)

    def test_different_config_options(self):
        """Test model with different configuration options."""
        config_variations = [
            {"patch_size": 1, "num_layers": 1},
            {"patch_size": 4, "num_layers": 3},
            {"out_channels": None},  # Should default to in_channels
            {"guidance_embeds": True},
        ]

        for variation in config_variations:
            with self.subTest(variation=variation):
                test_config = self.config.copy()
                test_config.update(variation)

                try:
                    model = QwenImageTransformer2DModel(**test_config)
                    self.assertIsNotNone(model)
                except Exception as e:
                    self.fail(f"Failed to create model with config {variation}: {e}")

    def test_typo_prevention(self):
        """Test parameter name typos."""
        # Test correct instantiation parameters
        try:
            model = QwenImageTransformer2DModel(
                patch_size=2,
                in_channels=16,
                out_channels=4,
                num_layers=2,
                attention_head_dim=64,
                num_attention_heads=8,
                joint_attention_dim=512,
                guidance_embeds=False,
                axes_dims_rope=(16, 56, 56),
            )
            self.assertIsNotNone(model)
        except TypeError as e:
            self.fail(f"Should accept valid parameter names: {e}")

        # Test method existence
        self.run_method_existence_tests(model, ["forward", "set_router"])

    def test_performance_benchmark(self):
        """Test performance benchmark."""
        model = QwenImageTransformer2DModel(**self.config)

        # Mock transformer blocks for faster execution
        for i, block in enumerate(model.transformer_blocks):
            model.transformer_blocks[i] = Mock(
                return_value=(torch.randn(1, 77, 512), torch.randn(1, 64, 512))  # Smaller sequence for performance
            )

        packed_hidden_states, img_shapes = self._generate_packed_hidden_states(1, 16, 16)
        inputs = {
            "hidden_states": packed_hidden_states,
            "encoder_hidden_states": torch.randn(1, 77, 512),
            "timestep": torch.randint(0, 1000, (1,)),
            "img_shapes": img_shapes,
            "txt_seq_lens": [77],
        }

        # Performance test with relaxed threshold for complex model
        self.run_performance_benchmark(model, inputs, max_time_ms=2000.0)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        model = QwenImageTransformer2DModel(**self.config)

        # Mock transformer blocks
        for i, block in enumerate(model.transformer_blocks):
            model.transformer_blocks[i] = Mock(
                return_value=(torch.randn(1, 1, 512), torch.randn(1, 1, 512))  # Minimal sequences
            )

        # Test minimal input sizes
        hidden_states, img_shapes = self._generate_packed_hidden_states(1, 2, 2)  # Minimal spatial size
        encoder_hidden_states = torch.randn(1, 1, 512)  # Single token
        timestep = torch.randint(0, 1000, (1,))
        txt_seq_lens = [1]

        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
            )

        self.assertIsNotNone(output)
        if hasattr(output, "sample"):
            self.assert_no_nan_or_inf(output.sample)


# Performance and integration tests
class TestQwenImageTransformerIntegration(TransformerBaseTest):
    """Integration tests for Qwen Image transformer components."""

    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline integration."""
        # Create minimal model for integration test
        config = {
            "patch_size": 2,
            "in_channels": 16,
            "out_channels": 4,
            "num_layers": 1,
            "attention_head_dim": 32,
            "num_attention_heads": 4,
            "joint_attention_dim": 256,
            "axes_dims_rope": (8, 16, 16),
        }

        model = QwenImageTransformer2DModel(**config)

        # Create realistic inputs
        batch_size = 1
        latent_channels = 4
        patch_size = config["patch_size"]
        height = width = 16
        latents = torch.randn(batch_size, latent_channels, height, width)
        latents = latents.view(batch_size, latent_channels, height // patch_size, patch_size, width // patch_size, patch_size)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        hidden_states = latents.reshape(
            batch_size,
            (height // patch_size) * (width // patch_size),
            latent_channels * (patch_size**2),
        )
        encoder_hidden_states = torch.randn(batch_size, 77, 256)
        timestep = torch.randint(0, 1000, (batch_size,))
        img_shapes = [(1, 8, 8)]
        txt_seq_lens = [77]

        # Test forward pass
        with torch.no_grad():
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
            )

        # Validate output
        if hasattr(output, "sample"):
            output_tensor = output.sample
        else:
            output_tensor = output

        expected_shape = (batch_size, config["out_channels"], height, width)
        self.assert_tensor_shape(output_tensor, expected_shape)
        self.assert_no_nan_or_inf(output_tensor)
        self.assert_tensor_in_range(output_tensor, -5.0, 5.0)

    def test_function_integration(self):
        """Test integration of standalone functions."""
        # Test get_timestep_embedding with apply_rotary_emb_qwen
        timesteps = torch.tensor([100, 200], dtype=torch.long)
        emb = get_timestep_embedding(timesteps, 64)

        # Reshape for rotary embedding (simulate real usage)
        x = emb.view(2, 1, 1, 64)  # Add dummy spatial dims
        cos = torch.randn(1, 64)
        sin = torch.randn(1, 64)
        freqs_cis = (cos, sin)

        rotated = apply_rotary_emb_qwen(x, freqs_cis, use_real=True)

        self.assert_tensor_shape(rotated, x.shape)
        self.assert_no_nan_or_inf(rotated)

    def test_component_compatibility(self):
        """Test compatibility between different components."""
        # Test QwenTimestepProjEmbeddings with QwenEmbedRope
        timestep_embed = QwenTimestepProjEmbeddings(512)
        rope_embed = QwenEmbedRope(10000, [16, 32, 32])

        timestep = torch.randint(0, 1000, (2,))
        hidden_states = torch.randn(2, 128, 512)

        # Get timestep embeddings
        temb = timestep_embed(timestep, hidden_states)
        self.assert_tensor_shape(temb, (2, 512))

        # Get rope embeddings
        vid_freqs, txt_freqs = rope_embed([(4, 32, 32)], [77], "cpu")
        self.assertIsInstance(vid_freqs, torch.Tensor)
        self.assertIsInstance(txt_freqs, torch.Tensor)

    def test_attention_processor_in_context(self):
        """Test attention processor in realistic context."""
        processor = QwenDoubleStreamAttnProcessor2_0()

        # Create more realistic mock attention
        mock_attn = Mock()
        mock_attn.heads = 8

        # Simulate realistic tensor shapes
        batch_size, img_seq, txt_seq, dim = 2, 256, 77, 512

        mock_attn.to_q.return_value = torch.randn(batch_size, img_seq, dim)
        mock_attn.to_k.return_value = torch.randn(batch_size, img_seq, dim)
        mock_attn.to_v.return_value = torch.randn(batch_size, img_seq, dim)
        mock_attn.add_q_proj.return_value = torch.randn(batch_size, txt_seq, dim)
        mock_attn.add_k_proj.return_value = torch.randn(batch_size, txt_seq, dim)
        mock_attn.add_v_proj.return_value = torch.randn(batch_size, txt_seq, dim)
        mock_attn.to_out = [Mock(return_value=torch.randn(batch_size, img_seq, dim))]
        mock_attn.to_add_out = Mock(return_value=torch.randn(batch_size, txt_seq, dim))

        # Mock normalization layers
        mock_attn.norm_q = None
        mock_attn.norm_k = None
        mock_attn.norm_added_q = None
        mock_attn.norm_added_k = None

        with patch("simpletuner.helpers.models.qwen_image.transformer.dispatch_attention_fn") as mock_dispatch:
            mock_dispatch.return_value = torch.randn(batch_size, img_seq + txt_seq, 8, 64)

            hidden_states = torch.randn(batch_size, img_seq, dim)
            encoder_hidden_states = torch.randn(batch_size, txt_seq, dim)

            img_out, txt_out = processor(
                attn=mock_attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states
            )

            self.assert_tensor_shape(img_out, (batch_size, img_seq, dim))
            self.assert_tensor_shape(txt_out, (batch_size, txt_seq, dim))


if __name__ == "__main__":
    unittest.main()
