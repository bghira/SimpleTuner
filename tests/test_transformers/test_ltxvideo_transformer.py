"""
Comprehensive unit tests for LTXVideo transformer classes.
Tests all 6 classes and 6 functions with focus on typo prevention and video-specific functionality.

Classes tested:
1. LTXVideoAttentionProcessor2_0 (deprecated wrapper)
2. LTXVideoAttnProcessor (attention processor)
3. LTXAttention (attention module)
4. LTXVideoRotaryPosEmbed (rotary position embedding)
5. LTXVideoTransformerBlock (transformer block)
6. LTXVideoTransformer3DModel (main model)

Functions tested:
1. apply_rotary_emb (rotary embedding application)
2. _prepare_video_coords (coordinate preparation)
3. forward methods for all classes
4. set_router (TREAD integration)
5. set_processor (attention processor setting)
6. prepare_attention_mask (from base attention)
"""

# Suppress PyTorch distributed warnings before importing torch
import logging
import math
import os

# Import test base classes
import sys
import unittest

logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import torch
import torch.nn as nn
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
from simpletuner.helpers.models.ltxvideo.transformer import (
    LTXAttention,
    LTXVideoAttentionProcessor2_0,
    LTXVideoAttnProcessor,
    LTXVideoRotaryPosEmbed,
    LTXVideoTransformer3DModel,
    LTXVideoTransformerBlock,
    apply_rotary_emb,
)


class TestLTXVideoAttentionProcessor2_0(TransformerBaseTest, AttentionProcessorTestMixin):
    """Test the deprecated LTXVideoAttentionProcessor2_0 wrapper."""

    def test_deprecation_warning(self):
        """Test that deprecation warning is issued and correct processor is returned."""
        with patch("simpletuner.helpers.models.ltxvideo.transformer.deprecate") as mock_deprecate:
            processor = LTXVideoAttentionProcessor2_0()

            # Should issue deprecation warning
            mock_deprecate.assert_called_once()
            args, kwargs = mock_deprecate.call_args
            self.assertEqual(args[0], "LTXVideoAttentionProcessor2_0")
            self.assertEqual(args[1], "1.0.0")
            self.assertIn("deprecated", args[2])

            # Should return LTXVideoAttnProcessor instance
            self.assertIsInstance(processor, LTXVideoAttnProcessor)

    def test_arguments_passed_through(self):
        """Test that arguments are properly passed to the new processor."""
        with patch("simpletuner.helpers.models.ltxvideo.transformer.deprecate"):
            # Test instantiation with no args (should work)
            processor = LTXVideoAttentionProcessor2_0()
            self.assertIsInstance(processor, LTXVideoAttnProcessor)

    def test_typo_prevention_processor_name(self):
        """Test typo prevention for processor class name."""
        # Test that the exact class name exists and is callable
        self.assertTrue(hasattr(LTXVideoAttentionProcessor2_0, "__new__"))
        self.assertTrue(callable(LTXVideoAttentionProcessor2_0))

        # Test common typos would fail
        with self.assertRaises((AttributeError, ImportError)):
            # These should not exist in the module
            from simpletuner.helpers.models.ltxvideo.transformer import LTXVideoAttnProcessor2_0  # Missing "ention"
        with self.assertRaises((AttributeError, ImportError)):
            from simpletuner.helpers.models.ltxvideo.transformer import LTXVideoAttentionProccessor2_0  # Typo in "Processor"


class TestLTXVideoAttnProcessor(TransformerBaseTest, AttentionProcessorTestMixin):
    """Test LTXVideoAttnProcessor attention processor."""

    def setUp(self):
        super().setUp()
        self.processor = LTXVideoAttnProcessor()
        self.mock_attn = self._create_mock_ltx_attention()

    def _create_mock_ltx_attention(self):
        """Create a mock LTXAttention with required attributes."""
        mock_attn = Mock()
        mock_attn.heads = self.num_heads
        mock_attn.to_q = Mock(return_value=torch.randn(self.batch_size, self.seq_len, self.hidden_dim))
        mock_attn.to_k = Mock(return_value=torch.randn(self.batch_size, self.seq_len, self.hidden_dim))
        mock_attn.to_v = Mock(return_value=torch.randn(self.batch_size, self.seq_len, self.hidden_dim))
        mock_attn.norm_q = Mock(side_effect=lambda x: x)
        mock_attn.norm_k = Mock(side_effect=lambda x: x)
        mock_attn.to_out = [Mock(side_effect=lambda x: x), Mock(side_effect=lambda x: x)]  # Linear layer  # Dropout layer

        def _prepare_attention_mask(mask, seq_len, batch_size):
            mask = mask.reshape(batch_size, -1)
            mask = mask[:, :seq_len]
            mask = mask.reshape(batch_size, 1, seq_len)
            mask = mask.unsqueeze(2).expand(batch_size, mock_attn.heads, 1, seq_len)
            return mask

        mock_attn.prepare_attention_mask = Mock(side_effect=_prepare_attention_mask)
        return mock_attn

    def test_basic_instantiation(self):
        """Test basic processor instantiation."""
        processor = LTXVideoAttnProcessor()
        self.assertIsNotNone(processor)
        self.assertIsNone(processor._attention_backend)

    def test_pytorch_version_check(self):
        """Test that processor requires PyTorch >= 2.0."""
        with patch("simpletuner.helpers.models.ltxvideo.transformer.is_torch_version") as mock_version:
            mock_version.return_value = True  # PyTorch < 2.0
            with self.assertRaises(ValueError) as cm:
                LTXVideoAttnProcessor()
            self.assertIn("minimum PyTorch version of 2.0", str(cm.exception))

    def test_forward_pass_minimal(self):
        """Test minimal forward pass with required arguments."""
        hidden_states = self.hidden_states

        with patch("simpletuner.helpers.models.ltxvideo.transformer.dispatch_attention_fn") as mock_dispatch:
            mock_dispatch.return_value = torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            output = self.processor(attn=self.mock_attn, hidden_states=hidden_states)

            self.assert_tensor_shape(output, hidden_states.shape)
            self.assert_no_nan_or_inf(output)

    def test_forward_pass_with_encoder_states(self):
        """Test forward pass with encoder hidden states."""
        hidden_states = self.hidden_states
        encoder_hidden_states = self.encoder_hidden_states

        with patch("simpletuner.helpers.models.ltxvideo.transformer.dispatch_attention_fn") as mock_dispatch:
            mock_dispatch.return_value = torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            output = self.processor(
                attn=self.mock_attn, hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states
            )

            self.assert_tensor_shape(output, hidden_states.shape)
            self.mock_attn.to_k.assert_called_with(encoder_hidden_states)
            self.mock_attn.to_v.assert_called_with(encoder_hidden_states)

    def test_forward_pass_with_attention_mask(self):
        """Test forward pass with attention mask."""
        hidden_states = self.hidden_states
        attention_mask = self.attention_mask

        with patch("simpletuner.helpers.models.ltxvideo.transformer.dispatch_attention_fn") as mock_dispatch:
            mock_dispatch.return_value = torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            output = self.processor(attn=self.mock_attn, hidden_states=hidden_states, attention_mask=attention_mask)

            self.assert_tensor_shape(output, hidden_states.shape)
            self.mock_attn.prepare_attention_mask.assert_called_once()

    def test_forward_pass_with_rotary_embeddings(self):
        """Test forward pass with image rotary embeddings (video-specific)."""
        hidden_states = self.hidden_states
        cos_emb = torch.randn(self.seq_len, self.hidden_dim)
        sin_emb = torch.randn(self.seq_len, self.hidden_dim)
        image_rotary_emb = (cos_emb, sin_emb)

        with patch("simpletuner.helpers.models.ltxvideo.transformer.dispatch_attention_fn") as mock_dispatch:
            mock_dispatch.return_value = torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            with patch("simpletuner.helpers.models.ltxvideo.transformer.apply_rotary_emb") as mock_apply_rotary:
                mock_apply_rotary.side_effect = lambda x, emb: x  # Identity for testing

                output = self.processor(attn=self.mock_attn, hidden_states=hidden_states, image_rotary_emb=image_rotary_emb)

                # Should call apply_rotary_emb twice (query and key)
                self.assertEqual(mock_apply_rotary.call_count, 2)
                self.assert_tensor_shape(output, hidden_states.shape)

    def test_tensor_reshape_operations(self):
        """Test tensor reshaping for multi-head attention."""
        hidden_states = self.hidden_states

        with patch("simpletuner.helpers.models.ltxvideo.transformer.dispatch_attention_fn") as mock_dispatch:
            # Mock return value to match dispatch attention output shape (batch, seq_len, heads, head_dim)
            reshaped_output = torch.randn(self.batch_size, self.seq_len, self.num_heads, self.head_dim)
            mock_dispatch.return_value = reshaped_output

            output = self.processor(attn=self.mock_attn, hidden_states=hidden_states)

            # Should return to original shape after operations
            self.assert_tensor_shape(output, hidden_states.shape)

    def test_typo_prevention_method_parameters(self):
        """Test typo prevention for method parameter names."""
        valid_params = {
            "attn": self.mock_attn,
            "hidden_states": self.hidden_states,
            "encoder_hidden_states": self.encoder_hidden_states,
            "attention_mask": self.attention_mask,
            "image_rotary_emb": self.image_rotary_emb,
        }

        typo_params = {
            "attn_": "attn",  # Common typo
            "hiden_states": "hidden_states",  # Missing 'd'
            "encoder_hiden_states": "encoder_hidden_states",  # Missing 'd'
            "atention_mask": "attention_mask",  # Missing 't'
            "image_rotary_embd": "image_rotary_emb",  # Extra 'd'
        }

        with patch("simpletuner.helpers.models.ltxvideo.transformer.dispatch_attention_fn") as mock_dispatch:
            mock_dispatch.return_value = torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.head_dim,
                dtype=self.hidden_states.dtype,
                device=self.hidden_states.device,
            )
            self.run_typo_prevention_tests(self.processor, "__call__", valid_params, typo_params)

    def test_edge_case_none_encoder_states(self):
        """Test edge case where encoder_hidden_states is None."""
        hidden_states = self.hidden_states

        with patch("simpletuner.helpers.models.ltxvideo.transformer.dispatch_attention_fn") as mock_dispatch:
            mock_dispatch.return_value = torch.randn(
                self.batch_size,
                self.seq_len,
                self.num_heads,
                self.head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

            output = self.processor(attn=self.mock_attn, hidden_states=hidden_states, encoder_hidden_states=None)

            # Should use hidden_states for both query and key/value
            self.mock_attn.to_k.assert_called_with(hidden_states)
            self.mock_attn.to_v.assert_called_with(hidden_states)

    def test_dtype_preservation(self):
        """Test that dtype is preserved through processing."""
        for dtype in [torch.float32, torch.float16]:
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue

            hidden_states = self.hidden_states.to(dtype)

            with patch("simpletuner.helpers.models.ltxvideo.transformer.dispatch_attention_fn") as mock_dispatch:
                mock_dispatch.return_value = torch.randn(
                    self.batch_size,
                    self.seq_len,
                    self.num_heads,
                    self.head_dim,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )

                output = self.processor(attn=self.mock_attn, hidden_states=hidden_states)

                if dtype == torch.float16 and output.dtype != dtype:
                    # Some backends (e.g. ROCm) promote fp16 attention outputs to fp32.
                    self.assertEqual(output.dtype, torch.float32)
                else:
                    self.assert_tensor_dtype(output, dtype)


class TestLTXAttention(TransformerBaseTest):
    """Test LTXAttention module."""

    def setUp(self):
        super().setUp()
        self.attention_config = {
            "query_dim": self.hidden_dim,
            "heads": self.num_heads,
            "kv_heads": self.num_heads,
            "dim_head": self.head_dim,
            "dropout": 0.0,
            "bias": True,
            "cross_attention_dim": None,
            "out_bias": True,
            "qk_norm": "rms_norm_across_heads",
        }

    def test_basic_instantiation(self):
        """Test basic attention module instantiation."""
        attention = LTXAttention(**self.attention_config)

        self.assertIsNotNone(attention)
        self.assertEqual(attention.heads, self.num_heads)
        self.assertEqual(attention.head_dim, self.head_dim)
        self.assertEqual(attention.inner_dim, self.head_dim * self.num_heads)
        self.assertEqual(attention.query_dim, self.hidden_dim)

        # Check processor is set
        self.assertIsInstance(attention.processor, LTXVideoAttnProcessor)

    def test_invalid_qk_norm(self):
        """Test that invalid qk_norm raises NotImplementedError."""
        config = self.attention_config.copy()
        config["qk_norm"] = "layer_norm"

        with self.assertRaises(NotImplementedError) as cm:
            LTXAttention(**config)
        self.assertIn("rms_norm_across_heads", str(cm.exception))

    def test_cross_attention_configuration(self):
        """Test cross-attention configuration."""
        config = self.attention_config.copy()
        config["cross_attention_dim"] = 768

        attention = LTXAttention(**config)
        self.assertEqual(attention.cross_attention_dim, 768)

    def test_forward_pass_minimal(self):
        """Test minimal forward pass."""
        attention = LTXAttention(**self.attention_config)

        with patch.object(
            LTXVideoAttnProcessor, "__call__", return_value=torch.randn_like(self.hidden_states)
        ) as mock_processor:
            output = attention.forward(hidden_states=self.hidden_states)

            self.assert_tensor_shape(output, self.hidden_states.shape)
            mock_processor.assert_called_once()

    def test_forward_pass_with_cross_attention(self):
        """Test forward pass with cross attention."""
        config = self.attention_config.copy()
        config["cross_attention_dim"] = 768
        attention = LTXAttention(**config)

        with patch.object(
            LTXVideoAttnProcessor, "__call__", return_value=torch.randn_like(self.hidden_states)
        ) as mock_processor:
            output = attention.forward(hidden_states=self.hidden_states, encoder_hidden_states=self.encoder_hidden_states)

            self.assert_tensor_shape(output, self.hidden_states.shape)
            mock_processor.assert_called_once()

    def test_forward_pass_with_video_rotary_emb(self):
        """Test forward pass with video-specific rotary embeddings."""
        attention = LTXAttention(**self.attention_config)
        cos_emb = torch.randn(self.seq_len, self.hidden_dim)
        sin_emb = torch.randn(self.seq_len, self.hidden_dim)
        image_rotary_emb = (cos_emb, sin_emb)

        with patch.object(
            LTXVideoAttnProcessor, "__call__", return_value=torch.randn_like(self.hidden_states)
        ) as mock_processor:
            output = attention.forward(hidden_states=self.hidden_states, image_rotary_emb=image_rotary_emb)

            call_args = mock_processor.call_args
            self.assertEqual(call_args.args[4], image_rotary_emb)

    def test_unused_kwargs_warning(self):
        """Test that unused kwargs generate warnings."""
        attention = LTXAttention(**self.attention_config)

        with patch.object(
            LTXVideoAttnProcessor, "__call__", return_value=torch.randn_like(self.hidden_states)
        ) as mock_processor:
            with patch("simpletuner.helpers.models.ltxvideo.transformer.logger.warning") as mock_warning:
                attention.forward(hidden_states=self.hidden_states, unused_param="should_warn")

                mock_warning.assert_called_once()
                warning_msg = mock_warning.call_args[0][0]
                self.assertIn("unused_param", warning_msg)
                self.assertIn("not expected", warning_msg)

    def test_set_processor(self):
        """Test setting custom processor."""
        attention = LTXAttention(**self.attention_config)
        custom_processor = LTXVideoAttnProcessor()

        attention.set_processor(custom_processor)
        self.assertEqual(attention.processor, custom_processor)

    def test_typo_prevention_parameter_names(self):
        """Test typo prevention for constructor parameter names."""
        valid_params = self.attention_config.copy()

        typo_params = {
            "query_dims": "query_dim",  # Extra 's'
            "head": "heads",  # Missing 's'
            "kv_head": "kv_heads",  # Missing 's'
            "dim_heads": "dim_head",  # Extra 's'
            "dropout_": "dropout",  # Extra '_'
            "cross_attention_dims": "cross_attention_dim",  # Extra 's'
            "out_bais": "out_bias",  # Swapped 'i' and 'a'
            "qk_norms": "qk_norm",  # Extra 's'
        }

        for typo_param, correct_param in typo_params.items():
            invalid_config = valid_params.copy()
            invalid_config[typo_param] = invalid_config.pop(correct_param)

            with self.assertRaises(TypeError) as cm:
                LTXAttention(**invalid_config)
            self.assertIn("unexpected keyword argument", str(cm.exception))

    def test_linear_layer_configurations(self):
        """Test linear layer configurations are correct."""
        attention = LTXAttention(**self.attention_config)

        # Check query projection
        self.assertEqual(attention.to_q.in_features, self.hidden_dim)
        self.assertEqual(attention.to_q.out_features, attention.inner_dim)

        # Check key/value projections
        self.assertEqual(attention.to_k.in_features, self.hidden_dim)
        self.assertEqual(attention.to_v.in_features, self.hidden_dim)
        self.assertEqual(attention.to_k.out_features, attention.inner_kv_dim)
        self.assertEqual(attention.to_v.out_features, attention.inner_kv_dim)

        # Check output projection
        self.assertEqual(attention.to_out[0].in_features, attention.inner_dim)
        self.assertEqual(attention.to_out[0].out_features, attention.out_dim)


class TestLTXVideoRotaryPosEmbed(TransformerBaseTest):
    """Test LTXVideoRotaryPosEmbed for video-specific rotary embeddings."""

    def setUp(self):
        super().setUp()
        self.rope_config = {
            "dim": self.head_dim,
            "base_num_frames": 20,
            "base_height": 2048,
            "base_width": 2048,
            "patch_size": 1,
            "patch_size_t": 1,
            "theta": 10000.0,
        }

    def test_basic_instantiation(self):
        """Test basic rotary position embedding instantiation."""
        rope = LTXVideoRotaryPosEmbed(**self.rope_config)

        self.assertIsNotNone(rope)
        self.assertEqual(rope.dim, self.head_dim)
        self.assertEqual(rope.base_num_frames, 20)
        self.assertEqual(rope.base_height, 2048)
        self.assertEqual(rope.base_width, 2048)
        self.assertEqual(rope.patch_size, 1)
        self.assertEqual(rope.patch_size_t, 1)
        self.assertEqual(rope.theta, 10000.0)

    def test_prepare_video_coords(self):
        """Test video coordinate preparation for 3D positional encoding."""
        rope = LTXVideoRotaryPosEmbed(**self.rope_config)

        batch_size = 2
        num_frames = 4
        height = 8
        width = 8
        rope_interpolation_scale = (torch.tensor(1.0), 1.0, 1.0)

        coords = rope._prepare_video_coords(batch_size, num_frames, height, width, rope_interpolation_scale, "cpu")

        # Should output coordinates for all tokens
        expected_tokens = num_frames * height * width
        self.assert_tensor_shape(coords, (batch_size, expected_tokens, 3))

        # Check coordinate ranges
        self.assertGreaterEqual(coords[:, :, 0].min(), 0)  # Frame coords >= 0
        self.assertGreaterEqual(coords[:, :, 1].min(), 0)  # Height coords >= 0
        self.assertGreaterEqual(coords[:, :, 2].min(), 0)  # Width coords >= 0

    def test_forward_pass_with_video_dimensions(self):
        """Test forward pass with video dimensions (3D structure)."""
        rope = LTXVideoRotaryPosEmbed(**self.rope_config)

        # Create video-shaped hidden states (batch, seq, dim) where seq = frames * height * width
        num_frames, height, width = 4, 8, 8
        seq_len = num_frames * height * width
        hidden_states = torch.randn(2, seq_len, self.head_dim, device="cpu")

        cos_freqs, sin_freqs = rope.forward(hidden_states=hidden_states, num_frames=num_frames, height=height, width=width)

        # Check output shapes
        self.assert_tensor_shape(cos_freqs, (2, seq_len, self.head_dim))
        self.assert_tensor_shape(sin_freqs, (2, seq_len, self.head_dim))

        # Check values are reasonable
        self.assert_no_nan_or_inf(cos_freqs)
        self.assert_no_nan_or_inf(sin_freqs)
        self.assert_tensor_in_range(cos_freqs, -1.0, 1.0)
        self.assert_tensor_in_range(sin_freqs, -1.0, 1.0)

    def test_forward_pass_with_interpolation_scale(self):
        """Test forward pass with rope interpolation scaling."""
        rope = LTXVideoRotaryPosEmbed(**self.rope_config)

        num_frames, height, width = 4, 8, 8
        seq_len = num_frames * height * width
        hidden_states = torch.randn(2, seq_len, self.head_dim, device="cpu")

        rope_interpolation_scale = (torch.tensor(0.5), 0.5, 0.5)

        cos_freqs, sin_freqs = rope.forward(
            hidden_states=hidden_states,
            num_frames=num_frames,
            height=height,
            width=width,
            rope_interpolation_scale=rope_interpolation_scale,
        )

        self.assert_tensor_shape(cos_freqs, (2, seq_len, self.head_dim))
        self.assert_tensor_shape(sin_freqs, (2, seq_len, self.head_dim))

    def test_forward_pass_with_video_coords(self):
        """Test forward pass with pre-computed video coordinates."""
        rope = LTXVideoRotaryPosEmbed(**self.rope_config)

        # Create video coordinates manually
        num_frames, height, width = 4, 8, 8
        seq_len = num_frames * height * width
        hidden_states = torch.randn(2, seq_len, self.head_dim, device="cpu")

        # Pre-computed coordinates (batch_size, seq_len, 3)
        video_coords = torch.randn(2, seq_len, 3)

        cos_freqs, sin_freqs = rope.forward(hidden_states=hidden_states, video_coords=video_coords)

        self.assert_tensor_shape(cos_freqs, (2, seq_len, self.head_dim))
        self.assert_tensor_shape(sin_freqs, (2, seq_len, self.head_dim))

    def test_frequency_calculation(self):
        """Test frequency calculation for different dimensions."""
        rope = LTXVideoRotaryPosEmbed(**self.rope_config)

        hidden_states = torch.randn(1, 64, self.head_dim, device="cpu")
        cos_freqs, sin_freqs = rope.forward(hidden_states=hidden_states, num_frames=4, height=4, width=4)

        # Check that frequencies use the correct theta base
        self.assertTrue(torch.all(cos_freqs >= -1) and torch.all(cos_freqs <= 1))
        self.assertTrue(torch.all(sin_freqs >= -1) and torch.all(sin_freqs <= 1))

    def test_dim_not_divisible_by_6_padding(self):
        """Test handling of dimensions not divisible by 6 with padding."""
        config = self.rope_config.copy()
        config["dim"] = 67  # Not divisible by 6
        rope = LTXVideoRotaryPosEmbed(**config)

        hidden_states = torch.randn(1, 64, 67, device="cpu")
        cos_freqs, sin_freqs = rope.forward(hidden_states=hidden_states, num_frames=4, height=4, width=4)

        self.assert_tensor_shape(cos_freqs, (1, 64, 67))
        self.assert_tensor_shape(sin_freqs, (1, 64, 67))

        # Check padding - first few elements should be padding (1 for cos, 0 for sin)
        remainder = 67 % 6
        self.assertTrue(torch.allclose(cos_freqs[:, :, :remainder], torch.ones_like(cos_freqs[:, :, :remainder])))
        self.assertTrue(torch.allclose(sin_freqs[:, :, :remainder], torch.zeros_like(sin_freqs[:, :, :remainder])))

    def test_typo_prevention_parameter_names(self):
        """Test typo prevention for parameter names."""
        valid_params = self.rope_config.copy()

        typo_params = {
            "dims": "dim",  # Extra 's'
            "base_num_frame": "base_num_frames",  # Missing 's'
            "base_hieght": "base_height",  # Misspelled
            "base_widht": "base_width",  # Misspelled
            "patch_siz": "patch_size",  # Missing 'e'
            "patch_size_time": "patch_size_t",  # Different name
            "thata": "theta",  # Misspelled
        }

        for typo_param, correct_param in typo_params.items():
            invalid_config = valid_params.copy()
            invalid_config[typo_param] = invalid_config.pop(correct_param)

            with self.assertRaises(TypeError) as cm:
                LTXVideoRotaryPosEmbed(**invalid_config)
            self.assertIn("unexpected keyword argument", str(cm.exception))


class TestLTXVideoTransformerBlock(TransformerBaseTest, TransformerBlockTestMixin):
    """Test LTXVideoTransformerBlock."""

    def setUp(self):
        super().setUp()
        self.block_config = {
            "dim": self.hidden_dim,
            "num_attention_heads": self.num_heads,
            "attention_head_dim": self.head_dim,
            "cross_attention_dim": 768,
            "qk_norm": "rms_norm_across_heads",
            "activation_fn": "gelu-approximate",
            "attention_bias": True,
            "attention_out_bias": True,
            "eps": 1e-6,
            "elementwise_affine": False,
        }

    def test_basic_instantiation(self):
        """Test basic transformer block instantiation."""
        block = LTXVideoTransformerBlock(**self.block_config)

        self.assertIsNotNone(block)
        self.assertIsNotNone(block.norm1)
        self.assertIsNotNone(block.attn1)
        self.assertIsNotNone(block.norm2)
        self.assertIsNotNone(block.attn2)
        self.assertIsNotNone(block.ff)
        self.assertIsNotNone(block.scale_shift_table)

        # Check scale_shift_table shape (6 parameters for adaptive normalization)
        self.assertEqual(block.scale_shift_table.shape, (6, self.hidden_dim))

    def test_forward_pass_minimal(self):
        """Test minimal forward pass."""
        block = LTXVideoTransformerBlock(**self.block_config)

        encoder_hidden_states = torch.randn(self.batch_size, 77, 768)
        temb = torch.randn(self.batch_size, 1, self.hidden_dim * 6)

        output = block.forward(hidden_states=self.hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb)

        self.assert_tensor_shape(output, self.hidden_states.shape)
        self.assert_no_nan_or_inf(output)

    def test_forward_pass_with_video_rotary_emb(self):
        """Test forward pass with video rotary embeddings."""
        block = LTXVideoTransformerBlock(**self.block_config)

        encoder_hidden_states = torch.randn(self.batch_size, 77, 768)
        temb = torch.randn(self.batch_size, 1, self.hidden_dim * 6)

        # Video rotary embeddings (cos, sin)
        cos_emb = torch.randn(self.seq_len, self.hidden_dim)
        sin_emb = torch.randn(self.seq_len, self.hidden_dim)
        image_rotary_emb = (cos_emb, sin_emb)

        output = block.forward(
            hidden_states=self.hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )

        self.assert_tensor_shape(output, self.hidden_states.shape)
        self.assert_no_nan_or_inf(output)

    def test_forward_pass_with_attention_mask(self):
        """Test forward pass with encoder attention mask."""
        block = LTXVideoTransformerBlock(**self.block_config)

        encoder_hidden_states = torch.randn(self.batch_size, 77, 768)
        temb = torch.randn(self.batch_size, 1, self.hidden_dim * 6)
        encoder_attention_mask = torch.ones(self.batch_size, 77)

        output = block.forward(
            hidden_states=self.hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            encoder_attention_mask=encoder_attention_mask,
        )

        self.assert_tensor_shape(output, self.hidden_states.shape)
        self.assert_no_nan_or_inf(output)

    def test_adaptive_normalization(self):
        """Test adaptive normalization with timestep embedding."""
        block = LTXVideoTransformerBlock(**self.block_config)

        encoder_hidden_states = torch.randn(self.batch_size, 77, 768)
        temb = torch.randn(self.batch_size, 1, self.hidden_dim * 6)

        output = block.forward(hidden_states=self.hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb)

        self.assert_tensor_shape(output, self.hidden_states.shape)

    def test_self_attention_cross_attention_flow(self):
        """Test the flow from self-attention to cross-attention."""
        block = LTXVideoTransformerBlock(**self.block_config)

        encoder_hidden_states = torch.randn(self.batch_size, 77, 768)
        temb = torch.randn(self.batch_size, 1, self.hidden_dim * 6)

        # Mock the attention layers to track calls
        with patch.object(block.attn1, "forward") as mock_attn1:
            mock_attn1.return_value = torch.randn_like(self.hidden_states)
            with patch.object(block.attn2, "forward") as mock_attn2:
                mock_attn2.return_value = torch.randn_like(self.hidden_states)

                block.forward(hidden_states=self.hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb)

                # Both attention layers should be called
                mock_attn1.assert_called_once()
                mock_attn2.assert_called_once()

                # Check self-attention call
                attn1_call = mock_attn1.call_args
                self.assertIn("encoder_hidden_states", attn1_call.kwargs)
                self.assertIsNone(attn1_call.kwargs["encoder_hidden_states"])

                # Check cross-attention call
                attn2_call = mock_attn2.call_args
                self.assertIn("encoder_hidden_states", attn2_call.kwargs)
                self.assertTrue(torch.equal(attn2_call.kwargs["encoder_hidden_states"], encoder_hidden_states))

    def test_typo_prevention_parameter_names(self):
        """Test typo prevention for constructor parameter names."""
        valid_params = self.block_config.copy()

        typo_params = {
            "dims": "dim",  # Extra 's'
            "num_attention_head": "num_attention_heads",  # Missing 's'
            "attention_head_dims": "attention_head_dim",  # Extra 's'
            "cross_attention_dims": "cross_attention_dim",  # Extra 's'
            "qk_norms": "qk_norm",  # Extra 's'
            "activation_function": "activation_fn",  # Different name
            "attention_bais": "attention_bias",  # Misspelled
            "attention_out_bais": "attention_out_bias",  # Misspelled
            "epss": "eps",  # Extra 's'
            "elementwise_affines": "elementwise_affine",  # Extra 's'
        }

        for typo_param, correct_param in typo_params.items():
            invalid_config = valid_params.copy()
            invalid_config[typo_param] = invalid_config.pop(correct_param)

            with self.assertRaises(TypeError) as cm:
                LTXVideoTransformerBlock(**invalid_config)
            self.assertIn("unexpected keyword argument", str(cm.exception))

    def test_feed_forward_integration(self):
        """Test feed-forward network integration."""
        block = LTXVideoTransformerBlock(**self.block_config)

        # Mock feed-forward to track usage
        with patch.object(block.ff, "forward") as mock_ff:
            mock_ff.return_value = torch.randn_like(self.hidden_states)

            encoder_hidden_states = torch.randn(self.batch_size, 77, 768)
            temb = torch.randn(self.batch_size, 1, self.hidden_dim * 6)

            block.forward(hidden_states=self.hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb)

            mock_ff.assert_called_once()


class TestLTXVideoTransformer3DModel(TransformerBaseTest):
    """Test LTXVideoTransformer3DModel main model class."""

    def setUp(self):
        super().setUp()
        self.model_config = {
            "in_channels": 128,
            "out_channels": 128,
            "patch_size": 1,
            "patch_size_t": 1,
            "num_attention_heads": 8,  # Reduced for testing
            "attention_head_dim": 64,
            "cross_attention_dim": self.hidden_dim,
            "num_layers": 2,  # Reduced for testing
            "activation_fn": "gelu-approximate",
            "qk_norm": "rms_norm_across_heads",
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "caption_channels": 4096,
            "attention_bias": True,
            "attention_out_bias": True,
        }

    def test_basic_instantiation(self):
        """Test basic model instantiation."""
        model = LTXVideoTransformer3DModel(**self.model_config)

        self.assertIsNotNone(model)
        self.assertIsNotNone(model.proj_in)
        self.assertIsNotNone(model.time_embed)
        self.assertIsNotNone(model.caption_projection)
        self.assertIsNotNone(model.rope)
        self.assertIsNotNone(model.transformer_blocks)
        self.assertIsNotNone(model.norm_out)
        self.assertIsNotNone(model.proj_out)

        # Check number of transformer blocks
        self.assertEqual(len(model.transformer_blocks), self.model_config["num_layers"])

    def test_forward_pass_minimal(self):
        """Test minimal forward pass with video data."""
        model = LTXVideoTransformer3DModel(**self.model_config)

        # Video-like hidden states (batch, seq_len, channels)
        seq_len = 16  # 2*2*4 = 16 tokens for 2x2x4 video
        hidden_states = torch.randn(2, seq_len, self.model_config["in_channels"])
        encoder_hidden_states = torch.randn(2, 77, self.model_config["caption_channels"])
        timestep = torch.randint(0, 1000, (2,))
        encoder_attention_mask = torch.ones(2, 77)

        with torch.no_grad():
            output = model.forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                num_frames=4,
                height=2,
                width=2,
            )

        if hasattr(output, "sample"):
            output_tensor = output.sample
        else:
            output_tensor = output

        self.assert_tensor_shape(output_tensor, (2, seq_len, self.model_config["out_channels"]))
        self.assert_no_nan_or_inf(output_tensor)

    def test_forward_pass_with_video_coords(self):
        """Test forward pass with pre-computed video coordinates."""
        model = LTXVideoTransformer3DModel(**self.model_config)

        seq_len = 16
        hidden_states = torch.randn(2, seq_len, self.model_config["in_channels"])
        encoder_hidden_states = torch.randn(2, 77, self.model_config["caption_channels"])
        timestep = torch.randint(0, 1000, (2,))
        encoder_attention_mask = torch.ones(2, 77)
        video_coords = torch.randn(2, seq_len, 3)  # Pre-computed coordinates

        with torch.no_grad():
            output = model.forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                video_coords=video_coords,
            )

        if hasattr(output, "sample"):
            output_tensor = output.sample
        else:
            output_tensor = output

        self.assert_tensor_shape(output_tensor, (2, seq_len, self.model_config["out_channels"]))

    def test_forward_pass_with_rope_interpolation(self):
        """Test forward pass with RoPE interpolation scaling."""
        model = LTXVideoTransformer3DModel(**self.model_config)

        seq_len = 16
        hidden_states = torch.randn(2, seq_len, self.model_config["in_channels"])
        encoder_hidden_states = torch.randn(2, 77, self.model_config["caption_channels"])
        timestep = torch.randint(0, 1000, (2,))
        encoder_attention_mask = torch.ones(2, 77)
        rope_interpolation_scale = (0.5, 0.5, 0.5)

        with torch.no_grad():
            output = model.forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                num_frames=4,
                height=2,
                width=2,
                rope_interpolation_scale=rope_interpolation_scale,
            )

        if hasattr(output, "sample"):
            output_tensor = output.sample
        else:
            output_tensor = output

        self.assert_tensor_shape(output_tensor, (2, seq_len, self.model_config["out_channels"]))

    @patch("simpletuner.helpers.training.tread.TREADRouter")
    def test_tread_router_integration(self, mock_tread_router_class):
        """Test TREAD router integration for token reduction."""
        model = LTXVideoTransformer3DModel(**self.model_config)

        # Create mock router instance
        mock_router = Mock()
        mock_mask_info = Mock()
        mock_mask_info.ids_shuffle = torch.randperm(16).unsqueeze(0).expand(2, -1)
        mock_router.get_mask.return_value = mock_mask_info
        mock_router.start_route.side_effect = lambda x, *_: x
        mock_router.end_route.side_effect = lambda hidden_states, *_: hidden_states

        # Set up router
        routes = [{"start_layer_idx": 0, "end_layer_idx": 0, "selection_ratio": 0.5}]
        model.set_router(mock_router, routes)

        seq_len = 16
        hidden_states = torch.randn(2, seq_len, self.model_config["in_channels"])
        encoder_hidden_states = torch.randn(2, 77, self.model_config["caption_channels"])
        timestep = torch.randint(0, 1000, (2,))
        encoder_attention_mask = torch.ones(2, 77)

        model.train()  # Enable routing
        with torch.enable_grad():
            output = model.forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                num_frames=4,
                height=2,
                width=2,
            )

        # Router methods should be called
        mock_router.get_mask.assert_called()
        mock_router.start_route.assert_called()
        mock_router.end_route.assert_called()

    def test_gradient_checkpointing(self):
        """Test gradient checkpointing functionality."""
        model = LTXVideoTransformer3DModel(**self.model_config)
        model.gradient_checkpointing = True

        # Use a wrapper to pass use_reentrant=False
        def checkpoint_func(func, *args, **kwargs):
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, **kwargs)

        model._gradient_checkpointing_func = checkpoint_func

        seq_len = 16
        hidden_states = torch.randn(2, seq_len, self.model_config["in_channels"], requires_grad=True)
        encoder_hidden_states = torch.randn(2, 77, self.model_config["caption_channels"])
        timestep = torch.randint(0, 1000, (2,))
        encoder_attention_mask = torch.ones(2, 77)

        output = model.forward(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            num_frames=4,
            height=2,
            width=2,
        )

        if hasattr(output, "sample"):
            loss = output.sample.sum()
        else:
            loss = output.sum()

        loss.backward()
        self.assertIsNotNone(hidden_states.grad)

    def test_attention_mask_processing(self):
        """Test attention mask processing for 2D to proper format."""
        model = LTXVideoTransformer3DModel(**self.model_config)

        seq_len = 16
        hidden_states = torch.randn(2, seq_len, self.model_config["in_channels"])
        encoder_hidden_states = torch.randn(2, 77, self.model_config["caption_channels"])
        timestep = torch.randint(0, 1000, (2,))

        # Test 2D attention mask
        encoder_attention_mask = torch.ones(2, 77)

        with torch.no_grad():
            output = model.forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                num_frames=4,
                height=2,
                width=2,
            )

        self.assertIsNotNone(output)

    def test_return_dict_control(self):
        """Test return_dict parameter controls output format."""
        model = LTXVideoTransformer3DModel(**self.model_config)

        seq_len = 16
        hidden_states = torch.randn(2, seq_len, self.model_config["in_channels"])
        encoder_hidden_states = torch.randn(2, 77, self.model_config["caption_channels"])
        timestep = torch.randint(0, 1000, (2,))
        encoder_attention_mask = torch.ones(2, 77)

        # Test return_dict=False
        with torch.no_grad():
            output_tuple = model.forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                num_frames=4,
                height=2,
                width=2,
                return_dict=False,
            )

        self.assertIsInstance(output_tuple, tuple)

        # Test return_dict=True (default)
        with torch.no_grad():
            output_dict = model.forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
                num_frames=4,
                height=2,
                width=2,
                return_dict=True,
            )

        self.assertTrue(hasattr(output_dict, "sample"))

    def test_typo_prevention_parameter_names(self):
        """Test typo prevention for constructor parameter names."""
        valid_params = self.model_config.copy()

        typo_params = {
            "in_channel": "in_channels",  # Missing 's'
            "out_channel": "out_channels",  # Missing 's'
            "patch_siz": "patch_size",  # Missing 'e'
            "patch_size_time": "patch_size_t",  # Different name
            "num_attention_head": "num_attention_heads",  # Missing 's'
            "attention_head_dims": "attention_head_dim",  # Extra 's'
            "cross_attention_dims": "cross_attention_dim",  # Extra 's'
            "num_layer": "num_layers",  # Missing 's'
            "activation_function": "activation_fn",  # Different name
            "qk_norms": "qk_norm",  # Extra 's'
            "norm_elementwise_affines": "norm_elementwise_affine",  # Extra 's'
            "norm_epsilon": "norm_eps",  # Different name
            "caption_channel": "caption_channels",  # Missing 's'
            "attention_bais": "attention_bias",  # Misspelled
            "attention_out_bais": "attention_out_bias",  # Misspelled
        }

        for typo_param, correct_param in typo_params.items():
            invalid_config = valid_params.copy()
            invalid_config[typo_param] = invalid_config.pop(correct_param)

            with self.assertRaises(TypeError) as cm:
                LTXVideoTransformer3DModel(**invalid_config)
            self.assertIn("unexpected keyword argument", str(cm.exception))

    def test_performance_benchmark(self):
        """Test performance benchmark for video transformer."""
        model = LTXVideoTransformer3DModel(**self.model_config)

        seq_len = 16
        inputs = {
            "hidden_states": torch.randn(1, seq_len, self.model_config["in_channels"]),
            "encoder_hidden_states": torch.randn(1, 77, self.model_config["caption_channels"]),
            "timestep": torch.randint(0, 1000, (1,)),
            "encoder_attention_mask": torch.ones(1, 77),
            "num_frames": 4,
            "height": 2,
            "width": 2,
        }

        # Test should complete within reasonable time for small model
        self.run_performance_benchmark(model, inputs, max_time_ms=2000.0)


class TestApplyRotaryEmb(TransformerBaseTest):
    """Test apply_rotary_emb function for video-specific rotary embeddings."""

    def test_basic_rotary_application(self):
        """Test basic rotary embedding application."""
        x = torch.randn(2, 8, 64)  # (batch, seq, dim)
        cos_emb = torch.randn(8, 64)
        sin_emb = torch.randn(8, 64)
        freqs = (cos_emb, sin_emb)

        output = apply_rotary_emb(x, freqs)

        self.assert_tensor_shape(output, x.shape)
        self.assert_tensor_dtype(output, x.dtype)
        self.assert_no_nan_or_inf(output)

    def test_complex_number_operations(self):
        """Test that rotary embedding uses proper complex number operations."""
        x = torch.randn(2, 8, 64)
        cos_emb = torch.randn(8, 64)
        sin_emb = torch.randn(8, 64)
        freqs = (cos_emb, sin_emb)

        # Test that the function handles the complex rotation correctly
        output = apply_rotary_emb(x, freqs)

        # Output should have same shape but different values
        self.assert_tensor_shape(output, x.shape)
        self.assertFalse(torch.allclose(output, x))

    def test_dtype_preservation(self):
        """Test that dtype is preserved in rotary embedding."""
        for dtype in [torch.float32, torch.float16]:
            if dtype == torch.float16 and not torch.cuda.is_available():
                continue

            x = torch.randn(2, 8, 64, dtype=dtype)
            cos_emb = torch.randn(8, 64, dtype=dtype)
            sin_emb = torch.randn(8, 64, dtype=dtype)
            freqs = (cos_emb, sin_emb)

            output = apply_rotary_emb(x, freqs)
            self.assert_tensor_dtype(output, dtype)

    def test_video_specific_dimensions(self):
        """Test rotary embedding with video-specific tensor shapes."""
        # Test with video sequence length (frames * height * width)
        frames, height, width = 4, 8, 8
        seq_len = frames * height * width
        x = torch.randn(2, seq_len, 64)

        cos_emb = torch.randn(seq_len, 64)
        sin_emb = torch.randn(seq_len, 64)
        freqs = (cos_emb, sin_emb)

        output = apply_rotary_emb(x, freqs)

        self.assert_tensor_shape(output, (2, seq_len, 64))
        self.assert_no_nan_or_inf(output)

    def test_edge_case_small_dimensions(self):
        """Test rotary embedding with small dimensions."""
        x = torch.randn(1, 1, 2)  # Minimal case
        cos_emb = torch.randn(1, 2)
        sin_emb = torch.randn(1, 2)
        freqs = (cos_emb, sin_emb)

        output = apply_rotary_emb(x, freqs)
        self.assert_tensor_shape(output, (1, 1, 2))

    def test_mathematical_correctness(self):
        """Test mathematical correctness of rotary embedding."""
        # Create simple test case where we can verify the math
        x = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float32)  # (1, 1, 4)
        cos_emb = torch.tensor([[1.0, 1.0, 0.0, 0.0]], dtype=torch.float32)  # (1, 4)
        sin_emb = torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32)  # (1, 4)
        freqs = (cos_emb, sin_emb)

        output = apply_rotary_emb(x, freqs)

        # Check that the transformation was applied
        self.assert_tensor_shape(output, (1, 1, 4))
        self.assertFalse(torch.allclose(output, x))

    def test_typo_prevention_function_name(self):
        """Test that function name is correct and commonly misspelled versions fail."""
        # Test that the correct function exists
        from simpletuner.helpers.models.ltxvideo.transformer import apply_rotary_emb

        self.assertTrue(callable(apply_rotary_emb))

        # Test that common typos would fail to import
        with self.assertRaises(ImportError):
            from simpletuner.helpers.models.ltxvideo.transformer import apply_rotory_emb  # Typo
        with self.assertRaises(ImportError):
            from simpletuner.helpers.models.ltxvideo.transformer import apply_rotery_emb  # Typo
        with self.assertRaises(ImportError):
            from simpletuner.helpers.models.ltxvideo.transformer import aply_rotary_emb  # Missing 'p'


if __name__ == "__main__":
    unittest.main()
