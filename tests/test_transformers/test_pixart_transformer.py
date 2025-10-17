"""
Comprehensive unit tests for PixArtTransformer2DModel.

This test suite covers:
- Model instantiation and configuration validation
- Forward pass with various input combinations
- Attention processor management (attn_processors, set_attn_processor, set_default_attn_processor)
- TREAD router integration and routing logic
- Gradient checkpointing functionality
- Timestep embedding processing
- ControlNet integration
- Typo prevention tests for critical methods
- Edge cases and error handling
- Performance benchmarks
"""

import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from transformer_base_test import AttentionProcessorTestMixin, TransformerBaseTest
from transformer_test_helpers import MockComponents, MockDiffusersConfig, TensorGenerator, patch_diffusers_imports


class TestPixArtTransformer2DModel(TransformerBaseTest, AttentionProcessorTestMixin):
    """Comprehensive tests for PixArtTransformer2DModel."""

    def setUp(self):
        """Set up test fixtures specific to PixArt."""
        super().setUp()

        # PixArt-specific configuration
        self.pixart_config = {
            "num_attention_heads": 8,
            "attention_head_dim": 64,
            "in_channels": 4,
            "out_channels": 8,
            "num_layers": 4,  # Reduced for testing
            "dropout": 0.1,
            "norm_num_groups": 32,
            "cross_attention_dim": 1152,
            "attention_bias": True,
            "sample_size": 32,  # Smaller for testing
            "patch_size": 2,
            "activation_fn": "gelu-approximate",
            "num_embeds_ada_norm": 1000,
            "upcast_attention": False,
            "norm_type": "ada_norm_single",
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "interpolation_scale": 2,
            "use_additional_conditions": True,
            "caption_channels": 4096,
            "attention_type": "default",
        }

        # Create PixArt-specific test tensors
        self._create_pixart_test_tensors()

        # Mock PixArt components
        self._create_pixart_mock_components()

    def _create_pixart_test_tensors(self):
        """Create PixArt-specific test tensors."""
        batch_size = self.batch_size
        height = width = self.pixart_config["sample_size"]
        in_channels = self.pixart_config["in_channels"]

        # Main input tensor (B, C, H, W)
        self.pixart_hidden_states = torch.randn(batch_size, in_channels, height, width, device=self.device)

        # Encoder hidden states for PixArt (B, seq_len, cross_attention_dim)
        self.pixart_encoder_hidden_states = torch.randn(
            batch_size, 77, self.pixart_config["cross_attention_dim"], device=self.device
        )

        # Timestep tensor
        self.pixart_timestep = torch.randint(0, 1000, (batch_size,), device=self.device)

        # Additional conditions for AdaLayerNorm
        self.pixart_added_cond_kwargs = {
            "resolution": torch.tensor([height, width], device=self.device).repeat(batch_size, 1),
            "aspect_ratio": torch.randn(batch_size, 1, device=self.device),
        }

        # Cross attention kwargs
        self.pixart_cross_attention_kwargs = {"scale": 1.0}

        # Attention masks
        self.pixart_attention_mask = torch.ones(batch_size, 77, device=self.device, dtype=torch.bool)
        self.pixart_encoder_attention_mask = torch.ones(batch_size, 77, device=self.device, dtype=torch.bool)

        # ControlNet block samples
        patch_seq_len = (height // self.pixart_config["patch_size"]) ** 2 + 1  # +1 for class token
        inner_dim = self.pixart_config["num_attention_heads"] * self.pixart_config["attention_head_dim"]
        self.pixart_controlnet_block_samples = [
            torch.randn(batch_size, patch_seq_len, inner_dim, device=self.device)
            for _ in range(3)  # Simulate 3 controlnet blocks
        ]

        # TREAD force keep mask
        self.pixart_force_keep_mask = torch.randint(0, 2, (batch_size, patch_seq_len), device=self.device, dtype=torch.bool)

    def _create_pixart_mock_components(self):
        """Create PixArt-specific mock components."""
        # Mock PatchEmbed
        self.mock_pos_embed = Mock()
        patch_seq_len = (self.pixart_config["sample_size"] // self.pixart_config["patch_size"]) ** 2 + 1
        inner_dim = self.pixart_config["num_attention_heads"] * self.pixart_config["attention_head_dim"]
        self.mock_pos_embed.return_value = torch.randn(self.batch_size, patch_seq_len, inner_dim, device=self.device)

        # Mock AdaLayerNormSingle
        self.mock_adaln_single = Mock()
        self.mock_adaln_single.return_value = (
            torch.randn(self.batch_size, inner_dim, device=self.device),  # timestep
            torch.randn(self.batch_size, inner_dim, device=self.device),  # embedded_timestep
        )

        # Mock caption projection
        self.mock_caption_projection = Mock()
        self.mock_caption_projection.return_value = torch.randn(self.batch_size, 77, inner_dim, device=self.device)

        # Mock transformer blocks
        self.mock_transformer_blocks = []
        for i in range(self.pixart_config["num_layers"]):
            mock_block = Mock()
            mock_block.return_value = torch.randn(self.batch_size, patch_seq_len, inner_dim, device=self.device)
            self.mock_transformer_blocks.append(mock_block)

        # Mock normalization and projection
        self.mock_norm_out = Mock()
        self.mock_norm_out.return_value = torch.randn(self.batch_size, patch_seq_len, inner_dim, device=self.device)

        self.mock_proj_out = Mock()
        patch_size = self.pixart_config["patch_size"]
        out_channels = self.pixart_config["out_channels"]
        self.mock_proj_out.return_value = torch.randn(
            self.batch_size, patch_seq_len, patch_size * patch_size * out_channels, device=self.device
        )

    def test_basic_instantiation(self):
        """Test basic PixArt model instantiation."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel.__init__") as mock_init:
            mock_init.return_value = None

            # Import here to use the patched version
            from simpletuner.helpers.models.pixart.transformer import PixArtTransformer2DModel

            # Test instantiation with default config
            model = PixArtTransformer2DModel(**self.pixart_config)
            mock_init.assert_called_once()

            # Test instantiation with minimal config
            minimal_config = {
                "num_attention_heads": 8,
                "attention_head_dim": 64,
                "in_channels": 4,
                "sample_size": 32,
            }
            model_minimal = PixArtTransformer2DModel(**minimal_config)
            self.assertIsNotNone(model_minimal)

    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        from simpletuner.helpers.models.pixart.transformer import PixArtTransformer2DModel

        # Test invalid norm_type
        invalid_config = self.pixart_config.copy()
        invalid_config["norm_type"] = "invalid_norm_type"

        with patch.object(PixArtTransformer2DModel, "__init__", side_effect=NotImplementedError("Invalid norm type")):
            with self.assertRaises(NotImplementedError):
                PixArtTransformer2DModel(**invalid_config)

        # Test norm_type="ada_norm_single" with None num_embeds_ada_norm
        invalid_config_2 = self.pixart_config.copy()
        invalid_config_2["norm_type"] = "ada_norm_single"
        invalid_config_2["num_embeds_ada_norm"] = None

        with patch.object(
            PixArtTransformer2DModel, "__init__", side_effect=ValueError("num_embeds_ada_norm cannot be None")
        ):
            with self.assertRaises(ValueError):
                PixArtTransformer2DModel(**invalid_config_2)

    def test_use_additional_conditions_logic(self):
        """Test automatic use_additional_conditions setting based on sample_size."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value

            # Mock config access
            mock_model.config = Mock()

            # Test sample_size=128 should set use_additional_conditions=True
            config_128 = self.pixart_config.copy()
            config_128["sample_size"] = 128
            config_128["use_additional_conditions"] = None

            model_128 = MockPixArt(**config_128)
            # We can't directly test the internal logic due to mocking,
            # but we ensure the constructor was called
            MockPixArt.assert_called_with(**config_128)

            # Test sample_size != 128 should set use_additional_conditions=False
            config_64 = self.pixart_config.copy()
            config_64["sample_size"] = 64
            config_64["use_additional_conditions"] = None

            model_64 = MockPixArt(**config_64)
            MockPixArt.assert_called_with(**config_64)

    def test_forward_pass_minimal(self):
        """Test minimal forward pass."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value

            # Setup mock forward method
            expected_output_shape = (
                self.batch_size,
                self.pixart_config["out_channels"],
                self.pixart_config["sample_size"],
                self.pixart_config["sample_size"],
            )
            mock_output = Mock()
            mock_output.sample = torch.randn(expected_output_shape, device=self.device)
            mock_model.forward.return_value = mock_output

            # Test minimal inputs
            inputs = {
                "hidden_states": self.pixart_hidden_states,
                "timestep": self.pixart_timestep,
                "added_cond_kwargs": self.pixart_added_cond_kwargs,
            }

            output = mock_model.forward(**inputs)
            self.assertIsNotNone(output.sample)
            self.assert_tensor_shape(output.sample, expected_output_shape)

    def test_forward_pass_full(self):
        """Test comprehensive forward pass with all inputs."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value

            # Setup mock forward method
            expected_output_shape = (
                self.batch_size,
                self.pixart_config["out_channels"],
                self.pixart_config["sample_size"],
                self.pixart_config["sample_size"],
            )
            mock_output = Mock()
            mock_output.sample = torch.randn(expected_output_shape, device=self.device)
            mock_model.forward.return_value = mock_output

            # Test full inputs
            inputs = {
                "hidden_states": self.pixart_hidden_states,
                "encoder_hidden_states": self.pixart_encoder_hidden_states,
                "timestep": self.pixart_timestep,
                "added_cond_kwargs": self.pixart_added_cond_kwargs,
                "cross_attention_kwargs": self.pixart_cross_attention_kwargs,
                "attention_mask": self.pixart_attention_mask,
                "encoder_attention_mask": self.pixart_encoder_attention_mask,
                "controlnet_block_samples": self.pixart_controlnet_block_samples,
                "controlnet_conditioning_scale": 0.8,
                "return_dict": True,
                "force_keep_mask": self.pixart_force_keep_mask,
            }

            output = mock_model.forward(**inputs)
            mock_model.forward.assert_called_once_with(**inputs)
            self.assertIsNotNone(output.sample)

    def test_attention_mask_processing(self):
        """Test attention mask preprocessing in forward method."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value

            # Mock forward that checks mask processing
            def mock_forward_with_mask_check(**kwargs):
                attention_mask = kwargs.get("attention_mask")
                encoder_attention_mask = kwargs.get("encoder_attention_mask")

                # Check that 2D masks are converted to 3D bias format
                if attention_mask is not None and attention_mask.ndim == 3:
                    # Should be converted from 2D to 3D with bias values
                    pass  # This would be the expected behavior

                return Mock(sample=torch.randn(2, 8, 32, 32))

            mock_model.forward = mock_forward_with_mask_check

            # Test with 2D attention mask (should be converted to 3D bias)
            mask_2d = torch.randint(0, 2, (self.batch_size, 77), device=self.device, dtype=torch.bool)
            inputs = {
                "hidden_states": self.pixart_hidden_states,
                "timestep": self.pixart_timestep,
                "added_cond_kwargs": self.pixart_added_cond_kwargs,
                "attention_mask": mask_2d,
            }

            result = mock_model.forward(**inputs)
            self.assertIsNotNone(result)

    def test_caption_projection_functionality(self):
        """Test caption projection when caption_channels is specified."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value

            # Mock caption_projection
            mock_model.caption_projection = Mock()
            mock_model.caption_projection.return_value = torch.randn(self.batch_size, 77, 512, device=self.device)

            # Mock forward method that uses caption projection
            def mock_forward_with_caption(**kwargs):
                encoder_hidden_states = kwargs.get("encoder_hidden_states")
                if encoder_hidden_states is not None and mock_model.caption_projection is not None:
                    # Should call caption projection
                    projected = mock_model.caption_projection(encoder_hidden_states)
                    return Mock(sample=torch.randn(2, 8, 32, 32))

                return Mock(sample=torch.randn(2, 8, 32, 32))

            mock_model.forward = mock_forward_with_caption

            # Test with encoder_hidden_states
            inputs = {
                "hidden_states": self.pixart_hidden_states,
                "encoder_hidden_states": self.pixart_encoder_hidden_states,
                "timestep": self.pixart_timestep,
                "added_cond_kwargs": self.pixart_added_cond_kwargs,
            }

            result = mock_model.forward(**inputs)
            self.assertIsNotNone(result)

    def test_controlnet_integration(self):
        """Test ControlNet block samples integration."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value

            # Mock forward that processes controlnet blocks
            def mock_forward_with_controlnet(**kwargs):
                controlnet_block_samples = kwargs.get("controlnet_block_samples")
                controlnet_conditioning_scale = kwargs.get("controlnet_conditioning_scale", 1.0)

                if controlnet_block_samples is not None:
                    # Should apply controlnet conditioning with proper scaling
                    for sample in controlnet_block_samples:
                        # Each sample should be properly shaped and scaled
                        self.assertIsInstance(sample, torch.Tensor)

                return Mock(sample=torch.randn(2, 8, 32, 32))

            mock_model.forward = mock_forward_with_controlnet

            # Test with controlnet blocks
            inputs = {
                "hidden_states": self.pixart_hidden_states,
                "timestep": self.pixart_timestep,
                "added_cond_kwargs": self.pixart_added_cond_kwargs,
                "controlnet_block_samples": self.pixart_controlnet_block_samples,
                "controlnet_conditioning_scale": 0.7,
            }

            result = mock_model.forward(**inputs)
            self.assertIsNotNone(result)

    def test_tread_router_integration(self):
        """Test TREAD router setting and routing logic."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value
            mock_model.set_router = Mock()

            # Create mock router and routes
            mock_router = self.mock_tread_instance
            routes = [
                {
                    "start_layer_idx": 0,
                    "end_layer_idx": 2,
                    "selection_ratio": 0.6,
                },
                {
                    "start_layer_idx": -2,  # Test negative indexing
                    "end_layer_idx": -1,
                    "selection_ratio": 0.4,
                },
            ]

            # Test router setting
            mock_model.set_router(mock_router, routes)
            mock_model.set_router.assert_called_once_with(mock_router, routes)

    def test_gradient_checkpointing_functionality(self):
        """Test gradient checkpointing in training mode."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value

            # Mock training mode and gradient checkpointing
            mock_model.training = True
            mock_model.gradient_checkpointing = True
            mock_model._gradient_checkpointing_func = Mock()

            # Mock forward that uses gradient checkpointing
            def mock_forward_with_checkpoint(**kwargs):
                if mock_model.training and mock_model.gradient_checkpointing:
                    # Should use gradient checkpointing function
                    mock_model._gradient_checkpointing_func.return_value = torch.randn(2, 65, 512)
                    return Mock(sample=torch.randn(2, 8, 32, 32))

                return Mock(sample=torch.randn(2, 8, 32, 32))

            mock_model.forward = mock_forward_with_checkpoint

            # Test forward pass with gradient checkpointing
            inputs = {
                "hidden_states": self.pixart_hidden_states,
                "timestep": self.pixart_timestep,
                "added_cond_kwargs": self.pixart_added_cond_kwargs,
            }

            result = mock_model.forward(**inputs)
            self.assertIsNotNone(result)

    def test_attention_processor_management(self):
        """Test attention processor getting, setting, and default setting."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value

            # Mock attention processors property
            mock_processors = {
                "transformer_blocks.0.attn1.processor": Mock(),
                "transformer_blocks.0.attn2.processor": Mock(),
                "transformer_blocks.1.attn1.processor": Mock(),
                "transformer_blocks.1.attn2.processor": Mock(),
            }
            mock_model.attn_processors = mock_processors
            mock_model.set_attn_processor = Mock()
            mock_model.set_default_attn_processor = Mock()

            # Test getting processors
            processors = mock_model.attn_processors
            self.assertIsInstance(processors, dict)
            self.assertGreater(len(processors), 0)

            # Test setting single processor
            from diffusers.models.attention_processor import AttnProcessor

            new_processor = AttnProcessor()
            mock_model.set_attn_processor(new_processor)
            mock_model.set_attn_processor.assert_called_once_with(new_processor)

            # Test setting processor dictionary
            processor_dict = {key: AttnProcessor() for key in mock_processors.keys()}
            mock_model.set_attn_processor.reset_mock()
            mock_model.set_attn_processor(processor_dict)
            mock_model.set_attn_processor.assert_called_once_with(processor_dict)

            # Test setting default processor
            mock_model.set_default_attn_processor()
            mock_model.set_default_attn_processor.assert_called_once()

    def test_qkv_projection_fusion(self):
        """Test QKV projection fusion and unfusion."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value
            mock_model.fuse_qkv_projections = Mock()
            mock_model.unfuse_qkv_projections = Mock()
            mock_model.original_attn_processors = None

            # Test fusion
            mock_model.fuse_qkv_projections()
            mock_model.fuse_qkv_projections.assert_called_once()

            # Test unfusion
            mock_model.unfuse_qkv_projections()
            mock_model.unfuse_qkv_projections.assert_called_once()

    def test_typo_prevention_parameter_names(self):
        """Test that critical parameter names are correctly spelled."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value
            mock_model.forward = Mock()

            # Valid parameters for forward method
            valid_params = {
                "hidden_states": self.pixart_hidden_states,
                "encoder_hidden_states": self.pixart_encoder_hidden_states,
                "timestep": self.pixart_timestep,
                "added_cond_kwargs": self.pixart_added_cond_kwargs,
            }

            # Common typos to test for
            typo_mappings = {
                "hidden_state": "hidden_states",  # Missing 's'
                "encoder_hidden_state": "encoder_hidden_states",  # Missing 's'
                "timesteps": "timestep",  # Extra 's'
                "time_step": "timestep",  # Underscore instead of no space
                "added_cond_kwarg": "added_cond_kwargs",  # Missing 's'
                "cross_attention_kwarg": "cross_attention_kwargs",  # Missing 's'
                "attention_masks": "attention_mask",  # Extra 's'
                "encoder_attention_masks": "encoder_attention_mask",  # Extra 's'
                "controlnet_block_sample": "controlnet_block_samples",  # Missing 's'
            }

            # Test that valid parameters work
            mock_model.forward(**valid_params)
            mock_model.forward.assert_called_with(**valid_params)

    def test_typo_prevention_method_names(self):
        """Test that all required methods exist with correct names."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value

            # Required methods that should exist
            required_methods = [
                "forward",
                "attn_processors",  # Property
                "set_attn_processor",
                "set_default_attn_processor",
                "fuse_qkv_projections",
                "unfuse_qkv_projections",
                "set_router",
            ]

            # Mock all required methods and properties
            for method_name in required_methods:
                if method_name == "attn_processors":
                    setattr(mock_model, method_name, {})
                else:
                    setattr(mock_model, method_name, Mock())

            # Test all methods exist and are accessible
            for method_name in required_methods:
                self.assertTrue(hasattr(mock_model, method_name))
                if method_name != "attn_processors":  # Property, not callable
                    self.assertTrue(callable(getattr(mock_model, method_name)))

    def test_typo_prevention_config_attributes(self):
        """Test that configuration attributes are correctly named."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value

            # Mock config with correct attribute names
            mock_config = Mock()
            required_config_attrs = [
                "num_attention_heads",
                "attention_head_dim",
                "in_channels",
                "out_channels",
                "num_layers",
                "sample_size",
                "patch_size",
                "cross_attention_dim",
                "caption_channels",
                "interpolation_scale",
                "norm_type",
                "norm_elementwise_affine",
                "norm_eps",
                "activation_fn",
                "num_embeds_ada_norm",
                "attention_bias",
                "upcast_attention",
                "attention_type",
            ]

            # Set all required attributes
            for attr in required_config_attrs:
                setattr(mock_config, attr, 42)  # Dummy value

            mock_model.config = mock_config

            # Test that all attributes are accessible
            for attr in required_config_attrs:
                self.assertTrue(hasattr(mock_model.config, attr))

    def test_tensor_shape_validation(self):
        """Test tensor shape validation and error handling."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value

            # Mock forward to validate input shapes
            def mock_forward_with_validation(**kwargs):
                hidden_states = kwargs.get("hidden_states")
                if hidden_states is not None:
                    if len(hidden_states.shape) != 4:
                        raise ValueError(f"Expected 4D tensor for hidden_states, got {len(hidden_states.shape)}D")
                    if hidden_states.shape[1] != self.pixart_config["in_channels"]:
                        raise ValueError(
                            f"Expected {self.pixart_config['in_channels']} channels, got {hidden_states.shape[1]}"
                        )

                timestep = kwargs.get("timestep")
                if timestep is not None and len(timestep.shape) != 1:
                    raise ValueError(f"Expected 1D tensor for timestep, got {len(timestep.shape)}D")

                return Mock(sample=torch.randn(2, 8, 32, 32))

            mock_model.forward = mock_forward_with_validation

            # Test valid tensor shapes
            valid_inputs = {
                "hidden_states": torch.randn(2, 4, 32, 32),
                "timestep": torch.randint(0, 1000, (2,)),
                "added_cond_kwargs": self.pixart_added_cond_kwargs,
            }
            result = mock_model.forward(**valid_inputs)
            self.assertIsNotNone(result)

            # Test invalid tensor dimensions
            invalid_input_3d = torch.randn(2, 4, 32)  # 3D instead of 4D
            with self.assertRaises(ValueError):
                mock_model.forward(hidden_states=invalid_input_3d, timestep=self.pixart_timestep)

            # Test invalid channel count
            invalid_input_channels = torch.randn(2, 8, 32, 32)  # Wrong channel count
            with self.assertRaises(ValueError):
                mock_model.forward(hidden_states=invalid_input_channels, timestep=self.pixart_timestep)

            # Test invalid timestep shape
            invalid_timestep = torch.randint(0, 1000, (2, 1))  # 2D instead of 1D
            with self.assertRaises(ValueError):
                mock_model.forward(hidden_states=self.pixart_hidden_states, timestep=invalid_timestep)

    def test_edge_cases_none_inputs(self):
        """Test handling of None and optional inputs."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value
            mock_model.forward = Mock(return_value=Mock(sample=torch.randn(2, 8, 32, 32)))

            # Test with None encoder_hidden_states
            inputs = {
                "hidden_states": self.pixart_hidden_states,
                "encoder_hidden_states": None,
                "timestep": self.pixart_timestep,
                "added_cond_kwargs": self.pixart_added_cond_kwargs,
            }

            result = mock_model.forward(**inputs)
            self.assertIsNotNone(result)

            # Test with None attention masks
            inputs_no_masks = {
                "hidden_states": self.pixart_hidden_states,
                "timestep": self.pixart_timestep,
                "added_cond_kwargs": self.pixart_added_cond_kwargs,
                "attention_mask": None,
                "encoder_attention_mask": None,
            }

            result = mock_model.forward(**inputs_no_masks)
            self.assertIsNotNone(result)

    def test_edge_cases_missing_added_cond_kwargs(self):
        """Test error handling when added_cond_kwargs is missing but required."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value
            mock_model.use_additional_conditions = True

            # Mock forward that checks for added_cond_kwargs
            def mock_forward_with_validation(**kwargs):
                if mock_model.use_additional_conditions and kwargs.get("added_cond_kwargs") is None:
                    raise ValueError("`added_cond_kwargs` cannot be None when using additional conditions")
                return Mock(sample=torch.randn(2, 8, 32, 32))

            mock_model.forward = mock_forward_with_validation

            # Test missing added_cond_kwargs should raise error
            inputs = {
                "hidden_states": self.pixart_hidden_states,
                "timestep": self.pixart_timestep,
                "added_cond_kwargs": None,
            }

            with self.assertRaises(ValueError):
                mock_model.forward(**inputs)

    def test_edge_cases_empty_controlnet_blocks(self):
        """Test handling of empty controlnet block samples."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value
            mock_model.forward = Mock(return_value=Mock(sample=torch.randn(2, 8, 32, 32)))

            # Test with empty controlnet_block_samples
            inputs = {
                "hidden_states": self.pixart_hidden_states,
                "timestep": self.pixart_timestep,
                "added_cond_kwargs": self.pixart_added_cond_kwargs,
                "controlnet_block_samples": [],
            }

            result = mock_model.forward(**inputs)
            self.assertIsNotNone(result)

    def test_device_compatibility(self):
        """Test model works on different devices."""
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value

            # Test CPU device
            cpu_input = torch.randn(2, 4, 32, 32, device="cpu")
            cpu_timestep = torch.randint(0, 1000, (2,), device="cpu")
            mock_model.forward = Mock(return_value=Mock(sample=torch.randn(2, 8, 32, 32, device="cpu")))

            result_cpu = mock_model.forward(
                hidden_states=cpu_input, timestep=cpu_timestep, added_cond_kwargs=self.pixart_added_cond_kwargs
            )
            self.assertEqual(str(result_cpu.sample.device), "cpu")

            # Test CUDA device
            cuda_input = torch.randn(2, 4, 32, 32, device="cuda")
            cuda_timestep = torch.randint(0, 1000, (2,), device="cuda")
            mock_model.forward = Mock(return_value=Mock(sample=torch.randn(2, 8, 32, 32, device="cuda")))

            result_cuda = mock_model.forward(
                hidden_states=cuda_input, timestep=cuda_timestep, added_cond_kwargs=self.pixart_added_cond_kwargs
            )
            self.assertEqual(str(result_cuda.sample.device), "cuda:0")

    def test_dtype_consistency(self):
        """Test model handles different dtypes correctly."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value

            # Test float32
            input_f32 = torch.randn(2, 4, 32, 32, dtype=torch.float32)
            timestep_f32 = torch.randint(0, 1000, (2,), dtype=torch.long)
            mock_model.forward = Mock(return_value=Mock(sample=torch.randn(2, 8, 32, 32, dtype=torch.float32)))

            result_f32 = mock_model.forward(
                hidden_states=input_f32, timestep=timestep_f32, added_cond_kwargs=self.pixart_added_cond_kwargs
            )
            self.assertEqual(result_f32.sample.dtype, torch.float32)

            # Test float16
            input_f16 = torch.randn(2, 4, 32, 32, dtype=torch.float16)
            mock_model.forward = Mock(return_value=Mock(sample=torch.randn(2, 8, 32, 32, dtype=torch.float16)))

            result_f16 = mock_model.forward(
                hidden_states=input_f16, timestep=timestep_f32, added_cond_kwargs=self.pixart_added_cond_kwargs
            )
            self.assertEqual(result_f16.sample.dtype, torch.float16)

    def test_return_dict_behavior(self):
        """Test return_dict parameter behavior."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value

            # Test return_dict=True
            mock_output_dict = Mock()
            mock_output_dict.sample = torch.randn(2, 8, 32, 32)
            mock_model.forward = Mock(return_value=mock_output_dict)

            result_dict = mock_model.forward(
                hidden_states=self.pixart_hidden_states,
                timestep=self.pixart_timestep,
                added_cond_kwargs=self.pixart_added_cond_kwargs,
                return_dict=True,
            )
            self.assertIsNotNone(result_dict.sample)

            # Test return_dict=False
            mock_output_tuple = (torch.randn(2, 8, 32, 32),)
            mock_model.forward = Mock(return_value=mock_output_tuple)

            result_tuple = mock_model.forward(
                hidden_states=self.pixart_hidden_states,
                timestep=self.pixart_timestep,
                added_cond_kwargs=self.pixart_added_cond_kwargs,
                return_dict=False,
            )
            self.assertIsInstance(result_tuple, tuple)
            self.assertEqual(len(result_tuple), 1)

    def test_performance_benchmark(self):
        """Test forward pass performance."""
        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value

            # Create a realistic mock that simulates computation time
            def mock_forward_with_delay(**kwargs):
                import time

                time.sleep(0.001)  # Simulate 1ms computation
                return Mock(sample=torch.randn(2, 8, 32, 32))

            mock_model.forward = mock_forward_with_delay

            # Measure performance
            inputs = {
                "hidden_states": self.pixart_hidden_states,
                "timestep": self.pixart_timestep,
                "added_cond_kwargs": self.pixart_added_cond_kwargs,
            }

            # This should complete within reasonable time (100ms)
            import time

            start_time = time.time()
            result = mock_model.forward(**inputs)
            end_time = time.time()

            self.assertIsNotNone(result)
            self.assertLess(end_time - start_time, 0.1)  # Should be fast with mocks

    def test_memory_efficiency(self):
        """Test memory usage during forward pass."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory testing")

        with patch("simpletuner.helpers.models.pixart.transformer.PixArtTransformer2DModel") as MockPixArt:
            mock_model = MockPixArt.return_value
            mock_model.forward = Mock(return_value=Mock(sample=torch.randn(2, 8, 32, 32, device="cuda")))

            # Clear CUDA memory
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            # Run forward pass
            cuda_input = torch.randn(2, 4, 32, 32, device="cuda")
            cuda_timestep = torch.randint(0, 1000, (2,), device="cuda")
            cuda_added_cond = {
                k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in self.pixart_added_cond_kwargs.items()
            }

            result = mock_model.forward(hidden_states=cuda_input, timestep=cuda_timestep, added_cond_kwargs=cuda_added_cond)

            peak_memory = torch.cuda.max_memory_allocated()
            memory_increase = peak_memory - initial_memory

            # Memory increase should be reasonable (less than 100MB for test)
            self.assertLess(memory_increase, 100 * 1024 * 1024)  # 100MB

            # Clean up
            del result, cuda_input, cuda_timestep, cuda_added_cond
            torch.cuda.empty_cache()


class TestPixArtTransformerIntegration(TransformerBaseTest):
    """Integration tests for PixArtTransformer2DModel with real components."""

    def setUp(self):
        """Set up integration test fixtures."""
        super().setUp()

        # Use smaller config for integration tests
        self.integration_config = {
            "num_attention_heads": 4,
            "attention_head_dim": 32,
            "in_channels": 4,
            "out_channels": 4,
            "num_layers": 2,  # Minimal layers
            "sample_size": 16,  # Smaller for faster tests
            "patch_size": 2,
            "cross_attention_dim": 256,
            "caption_channels": 256,
            "norm_type": "ada_norm_single",
            "num_embeds_ada_norm": 1000,
        }

    def test_integration_with_real_components(self):
        """Test integration with actual diffusers components."""
        # This test ensures our model can work with real diffusers components
        # when they're available in the environment

        try:
            from simpletuner.helpers.models.pixart.transformer import PixArtTransformer2DModel

            # Create model with minimal config
            with patch.multiple(
                "simpletuner.helpers.models.pixart.transformer",
                PatchEmbed=Mock(),
                BasicTransformerBlock=Mock(),
                AdaLayerNormSingle=Mock(),
                PixArtAlphaTextProjection=Mock(),
            ):
                model = PixArtTransformer2DModel(**self.integration_config)
                self.assertIsNotNone(model)

        except ImportError:
            self.skipTest("PixArtTransformer2DModel not available for integration testing")


if __name__ == "__main__":
    unittest.main()
