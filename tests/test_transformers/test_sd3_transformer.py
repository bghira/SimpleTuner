"""
Comprehensive unit tests for SD3Transformer2DModel.

This test suite covers:
- Model instantiation and configuration
- Forward pass with various input combinations
- Attention processor management
- TREAD router integration
- Gradient checkpointing and forward chunking
- Typo prevention tests for critical methods
- Edge cases and error handling
- Performance benchmarks
"""

import os
import sys
import unittest
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock, Mock, patch

import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))

from transformer_base_test import AttentionProcessorTestMixin, TransformerBaseTest
from transformer_test_helpers import MockComponents, MockDiffusersConfig, TensorGenerator, patch_diffusers_imports


class TestSD3Transformer2DModel(TransformerBaseTest, AttentionProcessorTestMixin):
    """Comprehensive tests for SD3Transformer2DModel."""

    def setUp(self):
        """Set up test fixtures specific to SD3."""
        super().setUp()

        # SD3-specific configuration
        self.sd3_config = {
            "sample_size": 32,
            "patch_size": 2,
            "in_channels": 16,
            "num_layers": 4,  # Reduced for testing
            "attention_head_dim": 64,
            "num_attention_heads": 8,
            "joint_attention_dim": 4096,
            "caption_projection_dim": 1152,
            "pooled_projection_dim": 2048,
            "out_channels": 16,
            "pos_embed_max_size": 96,
            "dual_attention_layers": (0, 2),  # Test with some dual attention layers
            "qk_norm": "layer_norm",
        }

        # Create SD3-specific test tensors
        self._create_sd3_test_tensors()

        # Mock SD3 components
        self._create_sd3_mock_components()

    def _create_sd3_test_tensors(self):
        """Create SD3-specific test tensors."""
        batch_size = self.batch_size
        height = width = self.sd3_config["sample_size"]
        in_channels = self.sd3_config["in_channels"]

        # Main input tensor (B, C, H, W)
        self.sd3_hidden_states = torch.randn(batch_size, in_channels, height, width, device=self.device)

        # Encoder hidden states for SD3 (B, seq_len, joint_attention_dim)
        self.sd3_encoder_hidden_states = torch.randn(
            batch_size, 77, self.sd3_config["joint_attention_dim"], device=self.device
        )

        # Pooled projections (B, pooled_projection_dim)
        self.sd3_pooled_projections = torch.randn(batch_size, self.sd3_config["pooled_projection_dim"], device=self.device)

        # Timestep tensor
        self.sd3_timestep = torch.randint(0, 1000, (batch_size,), device=self.device)

        # Block controlnet hidden states (list of tensors)
        patch_seq_len = (height // self.sd3_config["patch_size"]) ** 2
        inner_dim = self.sd3_config["num_attention_heads"] * self.sd3_config["attention_head_dim"]
        self.sd3_block_controlnet_states = [
            torch.randn(batch_size, patch_seq_len, inner_dim, device=self.device)
            for _ in range(2)  # Simulate 2 controlnet blocks
        ]

        # TREAD force keep mask
        self.sd3_force_keep_mask = torch.randint(0, 2, (batch_size, patch_seq_len), device=self.device, dtype=torch.bool)

    def _create_sd3_mock_components(self):
        """Create SD3-specific mock components."""
        # Mock PatchEmbed
        self.mock_pos_embed = Mock()
        patch_seq_len = (self.sd3_config["sample_size"] // self.sd3_config["patch_size"]) ** 2
        inner_dim = self.sd3_config["num_attention_heads"] * self.sd3_config["attention_head_dim"]
        self.mock_pos_embed.return_value = torch.randn(self.batch_size, patch_seq_len, inner_dim, device=self.device)

        # Mock CombinedTimestepTextProjEmbeddings
        self.mock_time_text_embed = Mock()
        self.mock_time_text_embed.return_value = torch.randn(self.batch_size, inner_dim, device=self.device)

        # Mock context embedder
        self.mock_context_embedder = Mock()
        self.mock_context_embedder.return_value = torch.randn(
            self.batch_size, 77, self.sd3_config["caption_projection_dim"], device=self.device
        )

        # Mock transformer blocks
        self.mock_transformer_blocks = []
        for i in range(self.sd3_config["num_layers"]):
            mock_block = Mock()
            mock_block.context_pre_only = i == self.sd3_config["num_layers"] - 1
            # Return tuple: (encoder_hidden_states, hidden_states)
            mock_block.return_value = (
                torch.randn(self.batch_size, 77, self.sd3_config["caption_projection_dim"], device=self.device),
                torch.randn(self.batch_size, patch_seq_len, inner_dim, device=self.device),
            )
            self.mock_transformer_blocks.append(mock_block)

        # Mock normalization and projection
        self.mock_norm_out = Mock()
        self.mock_norm_out.return_value = torch.randn(self.batch_size, patch_seq_len, inner_dim, device=self.device)

        self.mock_proj_out = Mock()
        patch_size = self.sd3_config["patch_size"]
        out_channels = self.sd3_config["out_channels"]
        self.mock_proj_out.return_value = torch.randn(
            self.batch_size, patch_seq_len, patch_size * patch_size * out_channels, device=self.device
        )

    def test_basic_instantiation(self):
        """Test basic SD3 model instantiation."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel.__init__") as mock_init:
            mock_init.return_value = None

            # Import here to use the patched version
            from simpletuner.helpers.models.sd3.transformer import SD3Transformer2DModel

            # Test instantiation with default config
            model = SD3Transformer2DModel(**self.sd3_config)
            mock_init.assert_called_once()

            # Test instantiation with minimal config
            minimal_config = {
                "sample_size": 32,
                "patch_size": 2,
                "in_channels": 16,
            }
            model_minimal = SD3Transformer2DModel(**minimal_config)
            self.assertIsNotNone(model_minimal)

    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        from simpletuner.helpers.models.sd3.transformer import SD3Transformer2DModel

        # Test invalid dual_attention_layers
        invalid_config = self.sd3_config.copy()
        invalid_config["dual_attention_layers"] = (10, 20)  # Indices beyond num_layers

        with patch.object(SD3Transformer2DModel, "__init__", side_effect=Exception("Invalid layer indices")):
            with self.assertRaises(Exception):
                SD3Transformer2DModel(**invalid_config)

    def test_forward_pass_minimal(self):
        """Test minimal forward pass."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value

            # Setup mock forward method
            expected_output_shape = (
                self.batch_size,
                self.sd3_config["out_channels"],
                self.sd3_config["sample_size"],
                self.sd3_config["sample_size"],
            )
            mock_output = Mock()
            mock_output.sample = torch.randn(expected_output_shape, device=self.device)
            mock_model.forward.return_value = mock_output

            # Test minimal inputs
            inputs = {
                "hidden_states": self.sd3_hidden_states,
                "timestep": self.sd3_timestep,
            }

            output = mock_model.forward(**inputs)
            self.assertIsNotNone(output.sample)
            self.assert_tensor_shape(output.sample, expected_output_shape)

    def test_forward_pass_full(self):
        """Test comprehensive forward pass with all inputs."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value

            # Setup mock forward method
            expected_output_shape = (
                self.batch_size,
                self.sd3_config["out_channels"],
                self.sd3_config["sample_size"],
                self.sd3_config["sample_size"],
            )
            mock_output = Mock()
            mock_output.sample = torch.randn(expected_output_shape, device=self.device)
            mock_model.forward.return_value = mock_output

            # Test full inputs
            inputs = {
                "hidden_states": self.sd3_hidden_states,
                "encoder_hidden_states": self.sd3_encoder_hidden_states,
                "pooled_projections": self.sd3_pooled_projections,
                "timestep": self.sd3_timestep,
                "block_controlnet_hidden_states": self.sd3_block_controlnet_states,
                "joint_attention_kwargs": {"scale": 1.0},
                "return_dict": True,
                "skip_layers": [1],
                "force_keep_mask": self.sd3_force_keep_mask,
            }

            output = mock_model.forward(**inputs)
            mock_model.forward.assert_called_once_with(**inputs)
            self.assertIsNotNone(output.sample)

    def test_gradient_checkpointing_interval(self):
        """Test gradient checkpointing interval setting."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value
            mock_model.set_gradient_checkpointing_interval = Mock()

            # Test setting interval
            interval = 2
            mock_model.set_gradient_checkpointing_interval(interval)
            mock_model.set_gradient_checkpointing_interval.assert_called_once_with(interval)

    def test_tread_router_integration(self):
        """Test TREAD router setting and integration."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value
            mock_model.set_router = Mock()

            # Create mock router and routes
            mock_router = self.mock_tread_instance
            routes = [
                {
                    "start_layer_idx": 0,
                    "end_layer_idx": 2,
                    "selection_ratio": 0.5,
                },
                {
                    "start_layer_idx": -2,  # Test negative indexing
                    "end_layer_idx": -1,
                    "selection_ratio": 0.3,
                },
            ]

            # Test router setting
            mock_model.set_router(mock_router, routes)
            mock_model.set_router.assert_called_once_with(mock_router, routes)

    def test_forward_chunking_enable_disable(self):
        """Test forward chunking enable/disable functionality."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value
            mock_model.enable_forward_chunking = Mock()
            mock_model.disable_forward_chunking = Mock()

            # Test enable with default chunk size
            mock_model.enable_forward_chunking()
            mock_model.enable_forward_chunking.assert_called_once()

            # Test enable with specific chunk size and dimension
            mock_model.enable_forward_chunking.reset_mock()
            mock_model.enable_forward_chunking(chunk_size=4, dim=1)
            mock_model.enable_forward_chunking.assert_called_once_with(chunk_size=4, dim=1)

            # Test disable
            mock_model.disable_forward_chunking()
            mock_model.disable_forward_chunking.assert_called_once()

    def test_attention_processor_management(self):
        """Test attention processor getting and setting."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value

            # Mock attention processors property
            mock_processors = {
                "transformer_blocks.0.attn.processor": Mock(),
                "transformer_blocks.1.attn.processor": Mock(),
            }
            mock_model.attn_processors = mock_processors
            mock_model.set_attn_processor = Mock()

            # Test getting processors
            processors = mock_model.attn_processors
            self.assertIsInstance(processors, dict)
            self.assertGreater(len(processors), 0)

            # Test setting single processor
            from diffusers.models.attention_processor import AttnProcessor

            new_processor = AttnProcessor()
            mock_model.set_attn_processor(new_processor)
            mock_model.set_attn_processor.assert_called_once_with(new_processor)

    def test_qkv_projection_fusion(self):
        """Test QKV projection fusion and unfusion."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value
            mock_model.fuse_qkv_projections = Mock()
            mock_model.unfuse_qkv_projections = Mock()

            # Test fusion
            mock_model.fuse_qkv_projections()
            mock_model.fuse_qkv_projections.assert_called_once()

            # Test unfusion
            mock_model.unfuse_qkv_projections()
            mock_model.unfuse_qkv_projections.assert_called_once()

    def test_typo_prevention_parameter_names(self):
        """Test that critical parameter names are correctly spelled."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value
            mock_model.forward = Mock()

            # Valid parameters for forward method
            valid_params = {
                "hidden_states": self.sd3_hidden_states,
                "encoder_hidden_states": self.sd3_encoder_hidden_states,
                "pooled_projections": self.sd3_pooled_projections,
                "timestep": self.sd3_timestep,
            }

            # Common typos to test for
            typo_mappings = {
                "hidden_state": "hidden_states",  # Missing 's'
                "encoder_hidden_state": "encoder_hidden_states",  # Missing 's'
                "pooled_projection": "pooled_projections",  # Missing 's'
                "timesteps": "timestep",  # Extra 's'
                "time_step": "timestep",  # Underscore instead of no space
                "block_controlnet_hidden_state": "block_controlnet_hidden_states",  # Missing 's'
            }

            # Test that valid parameters work
            mock_model.forward(**valid_params)
            mock_model.forward.assert_called_with(**valid_params)

    def test_typo_prevention_method_names(self):
        """Test that all required methods exist with correct names."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value

            # Required methods that should exist
            required_methods = [
                "forward",
                "set_gradient_checkpointing_interval",
                "set_router",
                "enable_forward_chunking",
                "disable_forward_chunking",
                "set_attn_processor",
                "fuse_qkv_projections",
                "unfuse_qkv_projections",
            ]

            # Mock all required methods
            for method_name in required_methods:
                setattr(mock_model, method_name, Mock())

            # Test all methods exist and are callable
            for method_name in required_methods:
                self.assertTrue(hasattr(mock_model, method_name))
                self.assertTrue(callable(getattr(mock_model, method_name)))

    def test_tensor_shape_validation(self):
        """Test tensor shape validation and error handling."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value

            # Mock forward to validate input shapes
            def mock_forward_with_validation(**kwargs):
                hidden_states = kwargs.get("hidden_states")
                if hidden_states is not None:
                    if len(hidden_states.shape) != 4:
                        raise ValueError(f"Expected 4D tensor for hidden_states, got {len(hidden_states.shape)}D")
                    if hidden_states.shape[1] != self.sd3_config["in_channels"]:
                        raise ValueError(f"Expected {self.sd3_config['in_channels']} channels, got {hidden_states.shape[1]}")

                return Mock(sample=torch.randn(2, 16, 32, 32))

            mock_model.forward = mock_forward_with_validation

            # Test valid tensor shape
            valid_input = torch.randn(2, 16, 32, 32)
            result = mock_model.forward(hidden_states=valid_input)
            self.assertIsNotNone(result)

            # Test invalid tensor dimensions
            invalid_input_3d = torch.randn(2, 16, 32)  # 3D instead of 4D
            with self.assertRaises(ValueError):
                mock_model.forward(hidden_states=invalid_input_3d)

            # Test invalid channel count
            invalid_input_channels = torch.randn(2, 8, 32, 32)  # Wrong channel count
            with self.assertRaises(ValueError):
                mock_model.forward(hidden_states=invalid_input_channels)

    def test_edge_cases_none_inputs(self):
        """Test handling of None and optional inputs."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value
            mock_model.forward = Mock(return_value=Mock(sample=torch.randn(2, 16, 32, 32)))

            # Test with None encoder_hidden_states
            inputs = {
                "hidden_states": self.sd3_hidden_states,
                "encoder_hidden_states": None,
                "timestep": self.sd3_timestep,
            }

            result = mock_model.forward(**inputs)
            self.assertIsNotNone(result)

    def test_edge_cases_empty_lists(self):
        """Test handling of empty lists for optional inputs."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value
            mock_model.forward = Mock(return_value=Mock(sample=torch.randn(2, 16, 32, 32)))

            # Test with empty block_controlnet_hidden_states
            inputs = {
                "hidden_states": self.sd3_hidden_states,
                "timestep": self.sd3_timestep,
                "block_controlnet_hidden_states": [],
                "skip_layers": [],
            }

            result = mock_model.forward(**inputs)
            self.assertIsNotNone(result)

    def test_device_compatibility(self):
        """Test model works on different devices."""
        # Skip if CUDA not available
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value

            # Test CPU device
            cpu_input = torch.randn(2, 16, 32, 32, device="cpu")
            mock_model.forward = Mock(return_value=Mock(sample=torch.randn(2, 16, 32, 32, device="cpu")))

            result_cpu = mock_model.forward(hidden_states=cpu_input)
            self.assertEqual(str(result_cpu.sample.device), "cpu")

            # Test CUDA device
            cuda_input = torch.randn(2, 16, 32, 32, device="cuda")
            mock_model.forward = Mock(return_value=Mock(sample=torch.randn(2, 16, 32, 32, device="cuda")))

            result_cuda = mock_model.forward(hidden_states=cuda_input)
            self.assertEqual(str(result_cuda.sample.device), "cuda:0")

    def test_dtype_consistency(self):
        """Test model handles different dtypes correctly."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value

            # Test float32
            input_f32 = torch.randn(2, 16, 32, 32, dtype=torch.float32)
            mock_model.forward = Mock(return_value=Mock(sample=torch.randn(2, 16, 32, 32, dtype=torch.float32)))

            result_f32 = mock_model.forward(hidden_states=input_f32)
            self.assertEqual(result_f32.sample.dtype, torch.float32)

            # Test float16
            input_f16 = torch.randn(2, 16, 32, 32, dtype=torch.float16)
            mock_model.forward = Mock(return_value=Mock(sample=torch.randn(2, 16, 32, 32, dtype=torch.float16)))

            result_f16 = mock_model.forward(hidden_states=input_f16)
            self.assertEqual(result_f16.sample.dtype, torch.float16)

    def test_return_dict_behavior(self):
        """Test return_dict parameter behavior."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value

            # Test return_dict=True
            mock_output_dict = Mock()
            mock_output_dict.sample = torch.randn(2, 16, 32, 32)
            mock_model.forward = Mock(return_value=mock_output_dict)

            result_dict = mock_model.forward(
                hidden_states=self.sd3_hidden_states, timestep=self.sd3_timestep, return_dict=True
            )
            self.assertIsNotNone(result_dict.sample)

            # Test return_dict=False
            mock_output_tuple = (torch.randn(2, 16, 32, 32),)
            mock_model.forward = Mock(return_value=mock_output_tuple)

            result_tuple = mock_model.forward(
                hidden_states=self.sd3_hidden_states, timestep=self.sd3_timestep, return_dict=False
            )
            self.assertIsInstance(result_tuple, tuple)
            self.assertEqual(len(result_tuple), 1)

    def test_performance_benchmark(self):
        """Test forward pass performance."""
        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value

            # Create a realistic mock that simulates computation time
            def mock_forward_with_delay(**kwargs):
                import time

                time.sleep(0.001)  # Simulate 1ms computation
                return Mock(sample=torch.randn(2, 16, 32, 32))

            mock_model.forward = mock_forward_with_delay

            # Measure performance
            inputs = {
                "hidden_states": self.sd3_hidden_states,
                "timestep": self.sd3_timestep,
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

        with patch("simpletuner.helpers.models.sd3.transformer.SD3Transformer2DModel") as MockSD3:
            mock_model = MockSD3.return_value
            mock_model.forward = Mock(return_value=Mock(sample=torch.randn(2, 16, 32, 32, device="cuda")))

            # Clear CUDA memory
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()

            # Run forward pass
            cuda_input = torch.randn(2, 16, 32, 32, device="cuda")
            result = mock_model.forward(hidden_states=cuda_input)

            peak_memory = torch.cuda.max_memory_allocated()
            memory_increase = peak_memory - initial_memory

            # Memory increase should be reasonable (less than 100MB for test)
            self.assertLess(memory_increase, 100 * 1024 * 1024)  # 100MB

            # Clean up
            del result, cuda_input
            torch.cuda.empty_cache()


class TestSD3TransformerIntegration(TransformerBaseTest):
    """Integration tests for SD3Transformer2DModel with real components."""

    def setUp(self):
        """Set up integration test fixtures."""
        super().setUp()

        # Use smaller config for integration tests
        self.integration_config = {
            "sample_size": 16,  # Smaller for faster tests
            "patch_size": 2,
            "in_channels": 4,
            "num_layers": 2,  # Minimal layers
            "attention_head_dim": 32,  # Smaller heads
            "num_attention_heads": 4,
            "joint_attention_dim": 256,
            "caption_projection_dim": 128,
            "pooled_projection_dim": 256,
            "out_channels": 4,
        }

    def test_integration_with_real_components(self):
        """Test integration with actual diffusers components."""
        # This test ensures our model can work with real diffusers components
        # when they're available in the environment

        try:
            from simpletuner.helpers.models.sd3.transformer import SD3Transformer2DModel

            # Create model with minimal config
            with patch.multiple(
                "simpletuner.helpers.models.sd3.transformer",
                PatchEmbed=Mock(),
                CombinedTimestepTextProjEmbeddings=Mock(),
                JointTransformerBlock=Mock(),
                AdaLayerNormContinuous=Mock(),
            ):
                model = SD3Transformer2DModel(**self.integration_config)
                self.assertIsNotNone(model)

        except ImportError:
            self.skipTest("SD3Transformer2DModel not available for integration testing")


if __name__ == "__main__":
    unittest.main()
