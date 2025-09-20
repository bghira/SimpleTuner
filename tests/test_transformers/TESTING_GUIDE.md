# SimpleTuner Transformer Testing Guide

## Table of Contents
1. [Overview](#overview)
2. [Test Suite Architecture](#test-suite-architecture)
3. [Running Tests](#running-tests)
4. [Writing New Tests](#writing-new-tests)
5. [Test Patterns and Best Practices](#test-patterns-and-best-practices)
6. [Troubleshooting](#troubleshooting)
7. [Maintenance](#maintenance)

## Overview

The SimpleTuner transformer testing suite provides comprehensive coverage for all 10 transformer implementations with a focus on:

- **Functionality Validation**: Ensuring all transformer components work correctly
- **Typo Prevention**: Catching common parameter and method name errors
- **Performance Monitoring**: Benchmarking execution speed and memory usage
- **Integration Testing**: Verifying components work together seamlessly
- **Edge Case Handling**: Testing boundary conditions and error scenarios

### Supported Transformers
- **Flux Transformer**: Complex rotary embeddings and attention processors
- **HiDream Transformer**: Advanced multi-head attention mechanisms
- **Auraflow Transformer**: Flow-based attention with dynamic routing
- **Cosmos Transformer**: Scientific computing with numerical stability
- **SD3 Transformer**: Joint attention with ControlNet integration
- **PixArt Transformer**: AdaLayerNorm with caption projection
- **LTX Video Transformer**: Video generation with temporal consistency
- **WAN Transformer**: Video processing with frame interpolation
- **Qwen Image Transformer**: Multi-modal vision-language integration
- **Sana Transformer**: Efficient attention with memory optimization

## Test Suite Architecture

### Directory Structure
```
tests/
├── utils/
│   ├── transformer_base_test.py      # Base test class with common functionality
│   ├── transformer_test_helpers.py   # Utility functions and mock objects
│   └── __init__.py
├── test_transformers/
│   ├── test_flux_transformer.py      # Flux transformer tests
│   ├── test_hidream_transformer.py   # HiDream transformer tests
│   ├── test_auraflow_transformer.py  # Auraflow transformer tests
│   ├── test_cosmos_transformer.py    # Cosmos transformer tests
│   ├── test_sd3_transformer.py       # SD3 transformer tests
│   ├── test_pixart_transformer.py    # PixArt transformer tests
│   ├── test_ltxvideo_transformer.py  # LTX Video transformer tests
│   ├── test_wan_transformer.py       # WAN transformer tests
│   ├── test_qwen_image_transformer.py # Qwen Image transformer tests
│   ├── test_sana_transformer.py      # Sana transformer tests
│   ├── README.md                     # This file
│   └── __init__.py
├── run_all_transformer_tests.py      # Master test runner
├── test_transformer_integration.py   # Integration tests
└── TRANSFORMER_COVERAGE_REPORT.md   # Comprehensive coverage analysis
```

### Base Classes and Mixins

#### TransformerBaseTest
The foundation class providing:
- Common setUp/tearDown methods
- Standard tensor generators
- Shape validation utilities
- Performance measurement tools
- Mock component creation

#### Specialized Mixins
- **AttentionProcessorTestMixin**: For testing attention processors
- **EmbeddingTestMixin**: For testing embedding modules
- **TransformerBlockTestMixin**: For testing transformer blocks

### Helper Utilities

#### TensorGenerator
Creates consistent test tensors:
```python
# Standard usage
tensor_gen = TensorGenerator()
hidden_states = tensor_gen.create_hidden_states(batch_size=2, seq_len=128, hidden_dim=512)
timestep = tensor_gen.create_timestep(batch_size=2)
```

#### MockDiffusersConfig
Provides configurable mock configurations:
```python
config = MockDiffusersConfig(
    num_attention_heads=8,
    attention_head_dim=64,
    hidden_size=512
)
```

#### ShapeValidator
Validates tensor shapes and properties:
```python
validator = ShapeValidator()
validator.validate_transformer_output(output, batch_size=2, seq_len=128, hidden_dim=512)
```

## Running Tests

### Quick Start

#### Run All Transformer Tests
```bash
# Basic execution
python tests/run_all_transformer_tests.py

# Verbose output with detailed results
python tests/run_all_transformer_tests.py -v

# Generate coverage report
python tests/run_all_transformer_tests.py --coverage-report coverage.txt
```

#### Run Specific Transformer Tests
```bash
# Run individual transformer test
python -m unittest tests.test_transformers.test_flux_transformer -v

# Run multiple specific tests
python -m unittest tests.test_transformers.test_flux_transformer tests.test_transformers.test_sd3_transformer -v
```

#### Run Test Categories
```bash
# Run tests matching pattern
python tests/run_all_transformer_tests.py --filter flux

# Run with fail-fast mode
python tests/run_all_transformer_tests.py --fail-fast
```

### Integration Testing
```bash
# Run integration tests
python -m unittest tests.test_transformer_integration -v

# Run all tests including integration
python tests/run_all_transformer_tests.py -v && python -m unittest tests.test_transformer_integration -v
```

### Performance Benchmarking
```bash
# Run tests with performance monitoring
python tests/run_all_transformer_tests.py -v --benchmark

# Individual performance test
python -c "
from tests.utils.transformer_test_helpers import PerformanceUtils
# Custom performance testing code
"
```

## Writing New Tests

### Creating a New Transformer Test File

1. **Create the test file** following the naming convention:
   ```
   tests/test_transformers/test_<transformer_name>_transformer.py
   ```

2. **Use the standard template**:
   ```python
   """
   Comprehensive unit tests for <transformer_name>/transformer.py
   """

   import unittest
   import torch
   from unittest.mock import Mock, patch

   # Import base classes
   from transformer_base_test import TransformerBaseTest
   from transformer_test_helpers import TensorGenerator, MockDiffusersConfig

   # Import transformer under test
   from simpletuner.helpers.models.<transformer_name>.transformer import (
       # Import transformer classes
   )

   class Test<TransformerName>Instantiation(TransformerBaseTest):
       """Test transformer instantiation and configuration."""

       def test_basic_instantiation(self):
           """Test basic transformer instantiation."""
           # Implementation

   class Test<TransformerName>ForwardPass(TransformerBaseTest):
       """Test forward pass functionality."""

       def test_forward_pass_minimal(self):
           """Test minimal forward pass."""
           # Implementation

   class Test<TransformerName>TypoPrevention(TransformerBaseTest):
       """Test typo prevention and parameter validation."""

       def test_parameter_name_typos(self):
           """Test parameter name typo detection."""
           # Implementation

   if __name__ == '__main__':
       unittest.main(verbosity=2)
   ```

### Required Test Categories

Each transformer test file should include:

#### 1. Instantiation Tests
```python
def test_basic_instantiation(self):
    """Test that transformer can be instantiated with default config."""
    config = self.create_mock_config()
    transformer = TransformerClass(config)
    self.assertIsNotNone(transformer)

def test_instantiation_with_custom_config(self):
    """Test instantiation with custom configuration parameters."""
    config = self.create_mock_config(
        num_attention_heads=16,
        attention_head_dim=128
    )
    transformer = TransformerClass(config)
    self.assertEqual(transformer.config.num_attention_heads, 16)
```

#### 2. Forward Pass Tests
```python
def test_forward_pass_minimal(self):
    """Test forward pass with minimal required inputs."""
    transformer = self._create_transformer()
    inputs = self.create_minimal_forward_inputs()

    output = self.run_forward_pass_test(
        transformer, inputs,
        expected_output_shape=(self.batch_size, self.seq_len, self.hidden_dim)
    )

def test_forward_pass_full(self):
    """Test forward pass with all optional inputs."""
    transformer = self._create_transformer()
    inputs = self.create_full_forward_inputs()

    output = self.run_forward_pass_test(
        transformer, inputs,
        expected_output_shape=(self.batch_size, self.seq_len, self.hidden_dim)
    )
```

#### 3. Typo Prevention Tests
```python
def test_parameter_name_typos(self):
    """Test that common parameter name typos are caught."""
    transformer = self._create_transformer()

    valid_params = {"hidden_states": self.hidden_states}
    typo_params = {"hidden_state": "hidden_states"}  # Common typo

    self.run_typo_prevention_tests(
        transformer, "forward", valid_params, typo_params
    )

def test_method_existence(self):
    """Test that all required methods exist."""
    transformer = self._create_transformer()
    required_methods = ["forward", "set_attention_slice"]

    self.run_method_existence_tests(transformer, required_methods)
```

#### 4. Edge Case Tests
```python
def test_zero_batch_size_handling(self):
    """Test handling of zero batch size."""
    transformer = self._create_transformer()

    with self.assertRaises(ValueError):
        inputs = self.create_minimal_forward_inputs()
        inputs["hidden_states"] = torch.randn(0, self.seq_len, self.hidden_dim)
        transformer(**inputs)

def test_extreme_sequence_lengths(self):
    """Test handling of extreme sequence lengths."""
    transformer = self._create_transformer()

    # Test very short sequence
    short_inputs = self.create_minimal_forward_inputs()
    short_inputs["hidden_states"] = torch.randn(self.batch_size, 1, self.hidden_dim)

    output = transformer(**short_inputs)
    self.assert_tensor_shape(output, (self.batch_size, 1, self.hidden_dim))
```

#### 5. Performance Tests
```python
def test_performance_benchmark(self):
    """Test that forward pass meets performance requirements."""
    transformer = self._create_transformer()
    inputs = self.create_minimal_forward_inputs()

    self.run_performance_benchmark(
        transformer, inputs, max_time_ms=1000.0
    )

def test_memory_usage(self):
    """Test memory usage is within expected bounds."""
    if not torch.cuda.is_available():
        self.skipTest("CUDA not available for memory testing")

    transformer = self._create_transformer().cuda()
    inputs = {k: v.cuda() for k, v in self.create_minimal_forward_inputs().items()}

    memory_stats = self.perf_utils.measure_memory_usage(transformer, inputs)
    # Assert memory usage is reasonable
```

## Test Patterns and Best Practices

### Consistent Test Structure

1. **Class Organization**: Group related tests into logical classes
2. **Method Naming**: Use descriptive names that explain what is being tested
3. **Documentation**: Include docstrings explaining the test purpose
4. **Assertions**: Use descriptive assertion messages

### Mock Strategy

#### Patching External Dependencies
```python
@patch('simpletuner.helpers.training.tread.TREADRouter')
def test_tread_router_integration(self, mock_tread_router):
    """Test TREAD router integration."""
    mock_instance = Mock()
    mock_instance.should_route.return_value = False
    mock_tread_router.return_value = mock_instance

    # Test implementation
```

#### Using Mock Components
```python
def test_attention_processor_integration(self):
    """Test attention processor integration."""
    mock_processor = Mock()
    mock_processor.return_value = self.hidden_states

    transformer = self._create_transformer()
    transformer.set_attention_processor(mock_processor)

    # Test that processor is used correctly
```

### Error Handling Best Practices

#### Descriptive Error Messages
```python
def test_input_validation(self):
    """Test input validation with descriptive errors."""
    transformer = self._create_transformer()

    # Test wrong tensor dtype
    with self.assertRaises(TypeError) as context:
        inputs = self.create_minimal_forward_inputs()
        inputs["hidden_states"] = inputs["hidden_states"].int()  # Wrong dtype
        transformer(**inputs)

    self.assertIn("dtype", str(context.exception).lower())
```

#### Graceful Skipping
```python
def test_gpu_specific_feature(self):
    """Test GPU-specific functionality."""
    if not torch.cuda.is_available():
        self.skipTest("CUDA not available")

    # GPU-specific test implementation
```

### Performance Testing Guidelines

#### Timing Tests
```python
def test_forward_pass_timing(self):
    """Test forward pass timing performance."""
    transformer = self._create_transformer()
    inputs = self.create_minimal_forward_inputs()

    # Warmup
    for _ in range(3):
        with torch.no_grad():
            transformer(**inputs)

    # Actual timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(10):
            transformer(**inputs)
    avg_time = (time.time() - start_time) / 10

    self.assertLess(avg_time, 0.1, "Forward pass too slow")
```

#### Memory Tests
```python
def test_memory_efficiency(self):
    """Test memory efficiency."""
    if not torch.cuda.is_available():
        self.skipTest("CUDA required for memory testing")

    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()

    transformer = self._create_transformer().cuda()
    inputs = {k: v.cuda() for k, v in self.create_minimal_forward_inputs().items()}

    with torch.no_grad():
        output = transformer(**inputs)

    peak_memory = torch.cuda.max_memory_allocated()
    memory_increase = peak_memory - initial_memory

    # Assert reasonable memory usage
    self.assertLess(memory_increase, 1e9, "Memory usage too high")  # < 1GB
```

## Troubleshooting

### Common Issues and Solutions

#### Import Errors
```
ImportError: No module named 'simpletuner.helpers.models.flux.transformer'
```

**Solution**: Ensure PYTHONPATH includes the SimpleTuner root directory:
```bash
export PYTHONPATH="/path/to/SimpleTuner:$PYTHONPATH"
python -m unittest tests.test_transformers.test_flux_transformer
```

#### Missing Dependencies
```
ImportError: No module named 'diffusers'
```

**Solution**: Install required dependencies or skip tests that require them:
```python
try:
    from diffusers import ConfigMixin
except ImportError:
    ConfigMixin = None

class TestTransformer(TransformerBaseTest):
    def setUp(self):
        if ConfigMixin is None:
            self.skipTest("diffusers not available")
        super().setUp()
```

#### CUDA Memory Errors
```
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size or skip GPU tests:
```python
def test_large_batch_processing(self):
    if not torch.cuda.is_available():
        self.skipTest("CUDA not available")

    # Use smaller batch size for testing
    batch_size = 1  # Instead of larger batch
    inputs = self.create_minimal_forward_inputs(batch_size=batch_size)
```

#### Test Timeout Issues
```
Test execution taking too long
```

**Solution**: Use smaller test cases or mock heavy operations:
```python
def test_transformer_with_large_inputs(self):
    # Use smaller dimensions for testing
    config = self.create_mock_config(
        hidden_size=256,  # Instead of 1024
        num_layers=2      # Instead of 12
    )
```

### Debugging Test Failures

#### Verbose Output
```bash
python -m unittest tests.test_transformers.test_flux_transformer.TestFluxForwardPass.test_forward_pass_minimal -v
```

#### Debug Mode
```python
def test_debug_forward_pass(self):
    """Debug version of forward pass test."""
    transformer = self._create_transformer()
    inputs = self.create_minimal_forward_inputs()

    print(f"Input shapes: {[f'{k}: {v.shape}' for k, v in inputs.items()]}")

    output = transformer(**inputs)

    print(f"Output shape: {output.shape}")
    print(f"Output dtype: {output.dtype}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
```

#### Isolated Testing
```python
# Test individual components
def test_attention_only(self):
    """Test attention mechanism in isolation."""
    # Create minimal attention test
```

## Maintenance

### Regular Maintenance Tasks

#### Monthly Reviews
1. **Coverage Analysis**: Review test coverage reports
2. **Performance Baselines**: Update performance benchmarks
3. **Documentation Updates**: Keep docs in sync with code changes
4. **Dependency Updates**: Check for new dependency versions

#### Quarterly Tasks
1. **Test Suite Optimization**: Profile and optimize slow tests
2. **Mock Strategy Review**: Ensure mocks are still accurate
3. **Integration Testing**: Verify cross-transformer compatibility
4. **Hardware Testing**: Test on different GPU configurations

#### Adding New Transformer Support

1. **Create Test File**: Follow the naming convention
2. **Implement Base Tests**: Use the template and required categories
3. **Add Integration Tests**: Update integration test file
4. **Update Documentation**: Add transformer to this guide
5. **Update Master Runner**: Ensure new tests are discovered

### Performance Monitoring

#### Baseline Establishment
```bash
# Run performance benchmarks and save results
python tests/run_all_transformer_tests.py --benchmark > baseline_performance.txt
```

#### Regression Detection
```bash
# Compare current performance to baseline
python tests/run_all_transformer_tests.py --benchmark > current_performance.txt
diff baseline_performance.txt current_performance.txt
```

### Test Suite Health Metrics

Monitor these metrics regularly:
- **Test Count**: Total number of tests
- **Success Rate**: Percentage of passing tests
- **Execution Time**: Total and per-test execution time
- **Memory Usage**: Peak memory consumption
- **Coverage Percentage**: Code coverage metrics

### Contributing Guidelines

When contributing new tests:

1. **Follow Patterns**: Use established test patterns
2. **Add Documentation**: Include clear docstrings
3. **Test Your Tests**: Verify tests pass and fail appropriately
4. **Update Integration**: Add to integration tests if needed
5. **Performance Aware**: Consider test execution time

---

**Last Updated**: December 2024
**Version**: 1.0
**Maintainer**: SimpleTuner Test Team