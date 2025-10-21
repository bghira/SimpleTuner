"""
Test utilities for transformer unit tests.
Provides common mock classes, tensor generation utilities, and validation helpers.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
import torch.nn as nn

# Set log level to critical for testing
os.environ["SIMPLETUNER_LOG_LEVEL"] = "CRITICAL"


class MockDiffusersConfig:
    """Mock configuration for diffusers models."""

    def __init__(self, **kwargs):
        # Default values for common transformer configs
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        self.attention_head_dim = kwargs.get("attention_head_dim", 64)
        self.in_channels = kwargs.get("in_channels", 4)
        self.num_layers = kwargs.get("num_layers", 4)
        self.sample_size = kwargs.get("sample_size", 32)
        self.patch_size = kwargs.get("patch_size", 2)
        self.hidden_size = kwargs.get("hidden_size", 512)
        self.intermediate_size = kwargs.get("intermediate_size", 2048)
        self.dropout = kwargs.get("dropout", 0.0)
        self.attention_dropout = kwargs.get("attention_dropout", 0.0)
        self.activation_fn = kwargs.get("activation_fn", "geglu")
        self.norm_type = kwargs.get("norm_type", "ada_norm_continuous")
        self.norm_elementwise_affine = kwargs.get("norm_elementwise_affine", False)
        self.norm_eps = kwargs.get("norm_eps", 1e-5)
        self.caption_channels = kwargs.get("caption_channels", 4096)
        self.pooled_projection_dim = kwargs.get("pooled_projection_dim", 768)

        # Update with any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)


class TensorGenerator:
    """Utility class for generating test tensors with consistent shapes."""

    @staticmethod
    def create_hidden_states(
        batch_size: int = 2, seq_len: int = 128, hidden_dim: int = 512, device: str = "cpu"
    ) -> torch.Tensor:
        """Create hidden states tensor."""
        return torch.randn(batch_size, seq_len, hidden_dim, device=device)

    @staticmethod
    def create_encoder_hidden_states(
        batch_size: int = 2, seq_len: int = 77, hidden_dim: int = 512, device: str = "cpu"
    ) -> torch.Tensor:
        """Create encoder hidden states tensor."""
        return torch.randn(batch_size, seq_len, hidden_dim, device=device)

    @staticmethod
    def create_timestep(batch_size: int = 2, device: str = "cpu") -> torch.Tensor:
        """Create timestep tensor."""
        return torch.randint(0, 1000, (batch_size,), device=device)

    @staticmethod
    def create_attention_mask(batch_size: int = 2, seq_len: int = 128, device: str = "cpu") -> torch.Tensor:
        """Create attention mask tensor."""
        return torch.ones(batch_size, seq_len, device=device, dtype=torch.bool)

    @staticmethod
    def create_image_rotary_emb(
        seq_len: int = 128, head_dim: int = 64, device: str = "cpu"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create image rotary embeddings (cos, sin)."""
        cos = torch.randn(seq_len, head_dim, device=device)
        sin = torch.randn(seq_len, head_dim, device=device)
        return (cos, sin)

    @staticmethod
    def create_pooled_projections(batch_size: int = 2, proj_dim: int = 768, device: str = "cpu") -> torch.Tensor:
        """Create pooled projections tensor."""
        return torch.randn(batch_size, proj_dim, device=device)

    @staticmethod
    def create_guidance(batch_size: int = 2, device: str = "cpu") -> torch.Tensor:
        """Create guidance tensor."""
        return torch.randn(batch_size, device=device)


class MockModule(nn.Module):
    """Base mock module that inherits from nn.Module."""

    def __init__(self, return_value=None, hidden_dim=768):
        super().__init__()
        self._return_value = return_value
        self.hidden_dim = hidden_dim

    def forward(self, *args, **kwargs):
        if self._return_value is not None:
            return self._return_value

        # Try to use input shape if available
        if args and hasattr(args[0], "shape"):
            input_tensor = args[0]
            if len(input_tensor.shape) == 3:  # [batch, seq, hidden]
                return torch.randn_like(input_tensor)

        return torch.randn(2, 128, self.hidden_dim)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MockNormLayer(nn.Module):
    """Mock normalization layer that returns multiple values for AdaLayerNormZero."""

    def __init__(self, return_tuple=False, num_values=1, hidden_dim=768):
        super().__init__()
        self.return_tuple = return_tuple
        self.num_values = num_values
        self.hidden_dim = hidden_dim

    def forward(self, *args, **kwargs):
        if self.return_tuple and self.num_values > 1:
            # For AdaLayerNormZero, return:
            # (norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp)
            batch_size = 2
            seq_len = 128
            if len(args) > 0 and hasattr(args[0], "shape"):
                # Use actual input dimensions if available
                batch_size, seq_len, hidden_dim = args[0].shape
                self.hidden_dim = hidden_dim

            norm_hidden_states = torch.randn(batch_size, seq_len, self.hidden_dim)
            gate_msa = torch.randn(batch_size, self.hidden_dim)
            shift_mlp = torch.randn(batch_size, self.hidden_dim)
            scale_mlp = torch.randn(batch_size, self.hidden_dim)
            gate_mlp = torch.randn(batch_size, self.hidden_dim)

            return norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp
        return torch.randn(2, 128, self.hidden_dim)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class MockAttention(nn.Module):
    """Mock attention module."""

    def __init__(self, return_tuple=False, hidden_dim=768):
        super().__init__()
        self.heads = 8
        self.return_tuple = return_tuple
        self.hidden_dim = hidden_dim
        self.last_args = ()
        self.last_kwargs = {}
        self.to_q = nn.Linear(hidden_dim, hidden_dim)
        self.to_k = nn.Linear(hidden_dim, hidden_dim)
        self.to_v = nn.Linear(hidden_dim, hidden_dim)
        self.to_out = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.0)])

    def forward(self, *args, **kwargs):
        self.last_args = args
        self.last_kwargs = kwargs.copy()
        # Get hidden_states from kwargs or args
        hidden_states = kwargs.get("hidden_states", args[0] if args else None)

        if hidden_states is not None and hasattr(hidden_states, "shape"):
            batch_size, seq_len, hidden_dim = hidden_states.shape

            if self.return_tuple:
                # For joint attention, return (attn_output, context_attn_output)
                context_len = 77  # Common context length
                return torch.randn(batch_size, seq_len, hidden_dim), torch.randn(batch_size, context_len, hidden_dim)
            return torch.randn(batch_size, seq_len, hidden_dim)

        # Fallback to default shapes
        if self.return_tuple:
            return torch.randn(2, 128, self.hidden_dim), torch.randn(2, 77, self.hidden_dim)
        return torch.randn(2, 128, self.hidden_dim)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def get_processor(self):
        return Mock()

    def set_processor(self, processor):
        pass


class MockComponents:
    """Factory for creating mock diffusers components."""

    @staticmethod
    def create_mock_attention():
        """Create mock attention module."""
        return MockAttention()

    @staticmethod
    def create_mock_norm_layer():
        """Create mock normalization layer."""
        return MockNormLayer()

    @staticmethod
    def create_mock_feed_forward():
        """Create mock feed forward module."""
        return MockModule(torch.randn(2, 128, 512))

    @staticmethod
    def create_mock_embedding():
        """Create mock embedding module."""
        return MockModule(torch.randn(2, 128, 512))

    @staticmethod
    def create_mock_ada_norm_zero(hidden_dim=768):
        """Create mock AdaLayerNormZero that returns 5 values."""
        return MockNormLayer(return_tuple=True, num_values=5, hidden_dim=hidden_dim)

    @staticmethod
    def create_mock_joint_attention(hidden_dim=768):
        """Create mock joint attention that returns tuple."""
        return MockAttention(return_tuple=True, hidden_dim=hidden_dim)


class ShapeValidator:
    """Utilities for validating tensor shapes in tests."""

    @staticmethod
    def validate_transformer_output(output: torch.Tensor, expected_batch: int, expected_seq: int, expected_dim: int):
        """Validate transformer output shape."""
        assert output.shape == (
            expected_batch,
            expected_seq,
            expected_dim,
        ), f"Expected shape ({expected_batch}, {expected_seq}, {expected_dim}), got {output.shape}"

    @staticmethod
    def validate_attention_scores(scores: torch.Tensor, expected_batch: int, expected_heads: int, expected_seq: int):
        """Validate attention scores shape."""
        assert scores.shape == (
            expected_batch,
            expected_heads,
            expected_seq,
            expected_seq,
        ), f"Expected attention scores shape ({expected_batch}, {expected_heads}, {expected_seq}, {expected_seq}), got {scores.shape}"

    @staticmethod
    def validate_embedding_output(output: torch.Tensor, expected_batch: int, expected_dim: int):
        """Validate embedding output shape."""
        assert len(output.shape) >= 2, f"Expected at least 2D tensor, got {len(output.shape)}D"
        assert output.shape[0] == expected_batch, f"Expected batch size {expected_batch}, got {output.shape[0]}"
        assert output.shape[-1] == expected_dim, f"Expected last dim {expected_dim}, got {output.shape[-1]}"


class TypoTestUtils:
    """Utilities specifically for testing typo-prone areas."""

    @staticmethod
    def test_parameter_name_typos(model, method_name: str, valid_params: Dict[str, Any], typo_params: Dict[str, str]):
        """Test that methods properly handle parameter name typos."""
        method = getattr(model, method_name)

        # Test valid parameters work
        try:
            result = method(**valid_params)
            assert result is not None, f"Method {method_name} should return a result with valid params"
        except Exception as e:
            if "unexpected keyword argument" in str(e):
                raise AssertionError(f"Method {method_name} should accept valid parameters: {e}")

        # Test typo parameters raise proper errors
        for typo_param, correct_param in typo_params.items():
            invalid_params = valid_params.copy()
            invalid_params[typo_param] = invalid_params.pop(correct_param)

            try:
                method(**invalid_params)
                raise AssertionError(f"Method {method_name} should reject typo parameter '{typo_param}'")
            except TypeError as e:
                if "unexpected keyword argument" not in str(e):
                    raise AssertionError(f"Expected 'unexpected keyword argument' error for '{typo_param}', got: {e}")

    @staticmethod
    def test_tensor_shape_assertions(func, valid_tensor: torch.Tensor, invalid_shapes: list):
        """Test that functions properly validate tensor shapes."""
        # Test valid tensor works
        try:
            result = func(valid_tensor)
            assert result is not None, "Function should return result with valid tensor shape"
        except Exception as e:
            if "shape" in str(e).lower() or "dimension" in str(e).lower():
                raise AssertionError(f"Function should accept valid tensor shape: {e}")

        # Test invalid shapes raise proper errors
        for invalid_shape in invalid_shapes:
            invalid_tensor = torch.randn(invalid_shape)
            try:
                func(invalid_tensor)
                raise AssertionError(f"Function should reject invalid shape {invalid_shape}")
            except (RuntimeError, ValueError, AssertionError) as e:
                if not any(keyword in str(e).lower() for keyword in ["shape", "dimension", "size", "mismatch"]):
                    raise AssertionError(f"Expected shape-related error for {invalid_shape}, got: {e}")

    @staticmethod
    def test_method_name_existence(model, required_methods: list):
        """Test that all required methods exist and aren't victims of typos."""
        for method_name in required_methods:
            assert hasattr(model, method_name), f"Model should have method '{method_name}'"
            assert callable(getattr(model, method_name)), f"'{method_name}' should be callable"


class PerformanceUtils:
    """Utilities for performance testing."""

    @staticmethod
    def measure_forward_pass_time(model, inputs: Dict[str, Any], num_runs: int = 10) -> float:
        """Measure average forward pass time."""
        import time

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model(**inputs)

        # Measure
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                model(**inputs)
                end = time.time()
                times.append(end - start)

        return sum(times) / len(times)

    @staticmethod
    def measure_memory_usage(model, inputs: Dict[str, Any]) -> Dict[str, int]:
        """Measure memory usage during forward pass."""
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        with torch.no_grad():
            output = model(**inputs)
            peak_memory = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

        return {
            "initial_memory": initial_memory,
            "peak_memory": peak_memory,
            "memory_increase": peak_memory - initial_memory,
        }


class MockingUtils:
    """Utilities for safely mocking PyTorch modules."""

    @staticmethod
    def safely_mock_module_list(module_list, mock_return_value=None):
        """Safely mock a nn.ModuleList by replacing its contents."""
        module_list.clear()
        if mock_return_value is None:
            mock_return_value = torch.randn(2, 128, 512)

        # Add mock modules instead of Mock objects
        mock_module = MockModule(mock_return_value)
        module_list.append(mock_module)

    @staticmethod
    def create_mock_transformer_block():
        """Create a mock transformer block that can be used in ModuleList."""

        class MockTransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = MockAttention()
                self.norm1 = MockNormLayer(return_tuple=True, num_values=5)
                self.norm2 = MockNormLayer()
                self.ff = MockModule()

            def forward(self, *args, **kwargs):
                if len(args) > 1:  # Joint transformer - returns tuple
                    return torch.randn(2, 77, 512), torch.randn(2, 128, 512)
                return torch.randn(2, 128, 512)

        return MockTransformerBlock()

    @staticmethod
    def safely_replace_modules_in_list(module_list, mock_factory=None):
        """Replace all modules in a ModuleList with mock modules."""
        if mock_factory is None:
            mock_factory = MockingUtils.create_mock_transformer_block

        # Store original length
        original_length = len(module_list)

        # Clear and replace with mock modules
        module_list.clear()
        for _ in range(original_length):
            module_list.append(mock_factory())


# Context managers for common testing scenarios
class patch_diffusers_imports:
    """Context manager to patch diffusers imports for testing."""

    def __init__(self):
        self.patches = []

    def __enter__(self):
        # Patch common diffusers imports
        self.patches.extend(
            [
                patch("diffusers.models.attention.Attention"),
                patch("diffusers.models.attention.FeedForward"),
                patch("diffusers.models.normalization.AdaLayerNormContinuous"),
                patch("diffusers.models.normalization.AdaLayerNormZero"),
                patch("diffusers.configuration_utils.ConfigMixin"),
                patch("diffusers.models.modeling_utils.ModelMixin"),
            ]
        )

        for p in self.patches:
            p.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for p in self.patches:
            p.stop()


# Common test fixtures
def get_common_test_config():
    """Get common test configuration."""
    return {"batch_size": 2, "seq_len": 128, "hidden_dim": 512, "num_heads": 8, "head_dim": 64, "device": "cpu"}
