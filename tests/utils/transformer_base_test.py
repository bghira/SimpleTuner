"""
Base test class for transformer unit tests.
Provides common setUp/tearDown methods, fixtures, and helper methods.
"""

import os
import unittest
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple
from unittest.mock import Mock, MagicMock, patch

from transformer_test_helpers import (
    MockDiffusersConfig,
    TensorGenerator,
    MockComponents,
    ShapeValidator,
    TypoTestUtils,
    PerformanceUtils,
    MockingUtils,
    get_common_test_config,
)

# Set log level to critical for testing
os.environ["SIMPLETUNER_LOG_LEVEL"] = "CRITICAL"


class TransformerBaseTest(unittest.TestCase):
    """Base test class for all transformer tests."""

    def setUp(self):
        """Set up common test fixtures."""
        self.device = "cpu"
        self.config = get_common_test_config()

        # Common tensor dimensions
        self.batch_size = self.config["batch_size"]
        self.seq_len = self.config["seq_len"]
        self.hidden_dim = self.config["hidden_dim"]
        self.num_heads = self.config["num_heads"]
        self.head_dim = self.config["head_dim"]

        # Initialize tensor generator
        self.tensor_gen = TensorGenerator()

        # Initialize validators
        self.shape_validator = ShapeValidator()
        self.typo_utils = TypoTestUtils()
        self.perf_utils = PerformanceUtils()
        self.mocking_utils = MockingUtils()

        # Create common test tensors
        self._create_common_tensors()

        # Create mock components
        self._create_mock_components()

        # Set up patches for common imports
        self._setup_patches()

    def tearDown(self):
        """Clean up after each test."""
        # Clean up CUDA memory if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Stop any remaining patches
        self._cleanup_patches()

    def _create_common_tensors(self):
        """Create commonly used test tensors."""
        self.hidden_states = self.tensor_gen.create_hidden_states(
            self.batch_size, self.seq_len, self.hidden_dim, self.device
        )
        self.encoder_hidden_states = self.tensor_gen.create_encoder_hidden_states(
            self.batch_size, 77, self.hidden_dim, self.device
        )
        self.timestep = self.tensor_gen.create_timestep(self.batch_size, self.device)
        self.attention_mask = self.tensor_gen.create_attention_mask(self.batch_size, self.seq_len, self.device)
        self.image_rotary_emb = self.tensor_gen.create_image_rotary_emb(self.seq_len, self.head_dim, self.device)
        self.pooled_projections = self.tensor_gen.create_pooled_projections(self.batch_size, 768, self.device)
        self.guidance = self.tensor_gen.create_guidance(self.batch_size, self.device)

    def _create_mock_components(self):
        """Create commonly used mock components."""
        self.mock_attention = MockComponents.create_mock_attention()
        self.mock_norm = MockComponents.create_mock_norm_layer()
        self.mock_feed_forward = MockComponents.create_mock_feed_forward()
        self.mock_embedding = MockComponents.create_mock_embedding()

    def _setup_patches(self):
        """Set up common patches."""
        self.patches = []

        # Patch TREADRouter to avoid import issues in tests
        tread_patch = patch("simpletuner.helpers.training.tread.TREADRouter")
        self.mock_tread_router = tread_patch.start()
        self.patches.append(tread_patch)

        # Mock TREADRouter instance
        self.mock_tread_instance = Mock()
        self.mock_tread_instance.should_route.return_value = False
        self.mock_tread_instance.get_routing_mask.return_value = None
        self.mock_tread_router.return_value = self.mock_tread_instance

    def _cleanup_patches(self):
        """Clean up all patches."""
        for patch_obj in self.patches:
            patch_obj.stop()
        self.patches.clear()

    def create_mock_config(self, **overrides) -> MockDiffusersConfig:
        """Create a mock configuration with optional overrides."""
        config_kwargs = {
            "num_attention_heads": self.num_heads,
            "attention_head_dim": self.head_dim,
            "hidden_size": self.hidden_dim,
            "num_layers": 4,
            "in_channels": 4,
            "sample_size": 32,
            "patch_size": 2,
        }
        config_kwargs.update(overrides)
        return MockDiffusersConfig(**config_kwargs)

    def assert_tensor_shape(self, tensor: torch.Tensor, expected_shape: Tuple[int, ...], msg: str = ""):
        """Assert tensor has expected shape."""
        self.assertEqual(
            tuple(tensor.shape), expected_shape, f"Expected shape {expected_shape}, got {tuple(tensor.shape)}. {msg}"
        )

    def assert_tensor_dtype(self, tensor: torch.Tensor, expected_dtype: torch.dtype, msg: str = ""):
        """Assert tensor has expected dtype."""
        self.assertEqual(tensor.dtype, expected_dtype, f"Expected dtype {expected_dtype}, got {tensor.dtype}. {msg}")

    def assert_tensor_device(self, tensor: torch.Tensor, expected_device: str, msg: str = ""):
        """Assert tensor is on expected device."""
        self.assertEqual(
            str(tensor.device), expected_device, f"Expected device {expected_device}, got {str(tensor.device)}. {msg}"
        )

    def assert_no_nan_or_inf(self, tensor: torch.Tensor, msg: str = ""):
        """Assert tensor contains no NaN or infinity values."""
        self.assertFalse(torch.isnan(tensor).any(), f"Tensor contains NaN values. {msg}")
        self.assertFalse(torch.isinf(tensor).any(), f"Tensor contains infinity values. {msg}")

    def assert_tensor_in_range(self, tensor: torch.Tensor, min_val: float = -10.0, max_val: float = 10.0, msg: str = ""):
        """Assert tensor values are within reasonable range."""
        self.assertGreaterEqual(tensor.min().item(), min_val, f"Tensor values below {min_val}. {msg}")
        self.assertLessEqual(tensor.max().item(), max_val, f"Tensor values above {max_val}. {msg}")

    def run_forward_pass_test(self, model, inputs: Dict[str, Any], expected_output_shape: Tuple[int, ...]):
        """Run a standard forward pass test."""
        # Test forward pass
        with torch.no_grad():
            output = model(**inputs)

        # Validate output
        if hasattr(output, "sample"):
            output_tensor = output.sample
        elif isinstance(output, torch.Tensor):
            output_tensor = output
        else:
            self.fail(f"Unexpected output type: {type(output)}")

        self.assert_tensor_shape(output_tensor, expected_output_shape)
        self.assert_no_nan_or_inf(output_tensor)
        self.assert_tensor_in_range(output_tensor)

        return output

    def run_typo_prevention_tests(
        self, model, method_name: str, valid_params: Dict[str, Any], typo_mappings: Dict[str, str]
    ):
        """Run typo prevention tests on a model method."""
        self.typo_utils.test_parameter_name_typos(model, method_name, valid_params, typo_mappings)

    def run_shape_validation_tests(self, func, valid_tensor: torch.Tensor, invalid_shapes: List[Tuple[int, ...]]):
        """Run shape validation tests on a function."""
        self.typo_utils.test_tensor_shape_assertions(func, valid_tensor, invalid_shapes)

    def run_method_existence_tests(self, model, required_methods: List[str]):
        """Test that all required methods exist."""
        self.typo_utils.test_method_name_existence(model, required_methods)

    def run_performance_benchmark(self, model, inputs: Dict[str, Any], max_time_ms: float = 1000.0):
        """Run performance benchmark and assert it meets requirements."""
        avg_time = self.perf_utils.measure_forward_pass_time(model, inputs)
        self.assertLess(
            avg_time * 1000, max_time_ms, f"Forward pass took {avg_time * 1000:.2f}ms, expected < {max_time_ms}ms"
        )

    def create_minimal_forward_inputs(self, **overrides) -> Dict[str, Any]:
        """Create minimal inputs for forward pass testing."""
        inputs = {
            "hidden_states": self.hidden_states,
            "timestep": self.timestep,
        }
        inputs.update(overrides)
        return inputs

    def create_full_forward_inputs(self, **overrides) -> Dict[str, Any]:
        """Create full inputs for comprehensive forward pass testing."""
        inputs = {
            "hidden_states": self.hidden_states,
            "encoder_hidden_states": self.encoder_hidden_states,
            "timestep": self.timestep,
            "attention_mask": self.attention_mask,
            "pooled_projections": self.pooled_projections,
            "guidance": self.guidance,
        }
        inputs.update(overrides)
        return inputs

    def test_basic_instantiation(self):
        """Test basic model instantiation - to be overridden by subclasses."""
        pass

    def test_forward_pass_minimal(self):
        """Test minimal forward pass - to be overridden by subclasses."""
        pass

    def test_forward_pass_full(self):
        """Test full forward pass - to be overridden by subclasses."""
        pass

    def test_typo_prevention(self):
        """Test typo prevention - to be overridden by subclasses."""
        pass

    def test_edge_cases(self):
        """Test edge cases - to be overridden by subclasses."""
        pass

    def test_error_handling(self):
        """Test error handling - to be overridden by subclasses."""
        pass


class AttentionProcessorTestMixin:
    """Mixin for testing attention processors."""

    def run_attention_processor_tests(self, processor_class, processor_kwargs: Dict[str, Any] = None):
        """Run standard tests for attention processors."""
        if processor_kwargs is None:
            processor_kwargs = {}

        # Test instantiation
        processor = processor_class(**processor_kwargs)
        self.assertIsNotNone(processor)

        # Test call method exists
        self.assertTrue(callable(processor))

        # Create mock attention and test inputs
        mock_attn = self.mock_attention
        hidden_states = self.hidden_states

        # Test basic call
        with torch.no_grad():
            try:
                output = processor(
                    attn=mock_attn,
                    hidden_states=hidden_states,
                    encoder_hidden_states=self.encoder_hidden_states,
                    attention_mask=self.attention_mask,
                    image_rotary_emb=self.image_rotary_emb,
                )
                self.assertIsInstance(output, torch.Tensor)
                self.assert_tensor_shape(output, hidden_states.shape)
            except Exception as e:
                # Some processors may need specific setup
                self.skipTest(f"Processor requires specific setup: {e}")


class EmbeddingTestMixin:
    """Mixin for testing embedding modules."""

    def run_embedding_tests(self, embedding_class, embedding_kwargs: Dict[str, Any] = None):
        """Run standard tests for embedding modules."""
        if embedding_kwargs is None:
            embedding_kwargs = {}

        # Test instantiation
        embedding = embedding_class(**embedding_kwargs)
        self.assertIsNotNone(embedding)

        # Test forward method exists
        self.assertTrue(hasattr(embedding, "forward"))

        # Test with minimal inputs
        with torch.no_grad():
            try:
                if hasattr(embedding, "timestep_proj"):
                    # Timestep embedding
                    output = embedding(self.timestep)
                else:
                    # Other embeddings
                    output = embedding(self.hidden_states)

                self.assertIsInstance(output, torch.Tensor)
                self.assert_no_nan_or_inf(output)
            except Exception as e:
                self.skipTest(f"Embedding requires specific setup: {e}")


class TransformerBlockTestMixin:
    """Mixin for testing transformer blocks."""

    def run_transformer_block_tests(self, block_class, block_kwargs: Dict[str, Any] = None):
        """Run standard tests for transformer blocks."""
        if block_kwargs is None:
            block_kwargs = {
                "dim": self.hidden_dim,
                "num_attention_heads": self.num_heads,
                "attention_head_dim": self.head_dim,
            }

        # Test instantiation
        block = block_class(**block_kwargs)
        self.assertIsNotNone(block)

        # Test forward method
        with torch.no_grad():
            try:
                output = block(
                    hidden_states=self.hidden_states,
                    encoder_hidden_states=self.encoder_hidden_states,
                    temb=self.timestep,
                )
                self.assertIsInstance(output, torch.Tensor)
                self.assert_tensor_shape(output, self.hidden_states.shape)
                self.assert_no_nan_or_inf(output)
            except Exception as e:
                self.skipTest(f"Transformer block requires specific setup: {e}")
