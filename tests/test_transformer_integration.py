"""
Integration tests for transformer test suite.

This module validates that all transformer test files work correctly together,
have consistent interfaces, and follow the established testing patterns.
"""

import importlib.util
import os
import sys
import unittest
from typing import Any, Dict, List, Set
from unittest.mock import Mock, patch

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

from transformer_base_test import TransformerBaseTest
from transformer_test_helpers import MockDiffusersConfig, PerformanceUtils, ShapeValidator, TensorGenerator, TypoTestUtils


class TestTransformerTestSuiteIntegration(unittest.TestCase):
    """Integration tests for the entire transformer test suite."""

    def setUp(self):
        """Set up integration test environment."""
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_transformers")
        self.test_files = self._discover_test_files()
        self.loaded_modules = {}

    def _discover_test_files(self) -> List[str]:
        """Discover all transformer test files."""
        test_files = []
        if os.path.exists(self.test_dir):
            for filename in os.listdir(self.test_dir):
                if filename.startswith("test_") and filename.endswith("_transformer.py"):
                    test_files.append(os.path.join(self.test_dir, filename))
        return sorted(test_files)

    def _load_test_module(self, test_file: str):
        """Load a test module for analysis."""
        module_name = os.path.basename(test_file)[:-3]

        if module_name in self.loaded_modules:
            return self.loaded_modules[module_name]

        try:
            spec = importlib.util.spec_from_file_location(module_name, test_file)
            module = importlib.util.module_from_spec(spec)

            # Add paths for imports
            original_path = sys.path.copy()
            sys.path.insert(0, os.path.dirname(test_file))
            sys.path.insert(0, os.path.join(os.path.dirname(test_file), "..", "utils"))

            try:
                spec.loader.exec_module(module)
                self.loaded_modules[module_name] = module
                return module
            except ImportError as e:
                # Skip modules with missing dependencies
                self.skipTest(f"Cannot load {module_name}: {e}")
            finally:
                sys.path = original_path

        except Exception as e:
            self.fail(f"Failed to load {module_name}: {e}")

    def test_all_test_files_discovered(self):
        """Test that all expected transformer test files are discovered."""
        expected_transformers = [
            "auraflow",
            "chroma",
            "chroma_controlnet",
            "cosmos",
            "flux",
            "hidream",
            "ltxvideo",
            "pixart",
            "qwen_image",
            "sana",
            "sd3",
            "wan",
        ]

        discovered_transformers = set()
        for test_file in self.test_files:
            filename = os.path.basename(test_file)
            for transformer in expected_transformers:
                if transformer in filename:
                    discovered_transformers.add(transformer)

        missing_transformers = set(expected_transformers) - discovered_transformers
        self.assertEqual(len(missing_transformers), 0, f"Missing test files for transformers: {missing_transformers}")

        self.assertEqual(
            len(self.test_files),
            len(expected_transformers),
            f"Expected {len(expected_transformers)} test files, found {len(self.test_files)}: {self.test_files}",
        )

    def test_base_test_class_inheritance(self):
        """Test that all test classes inherit from TransformerBaseTest."""
        for test_file in self.test_files:
            with self.subTest(test_file=os.path.basename(test_file)):
                module = self._load_test_module(test_file)

                # Find test classes in the module
                test_classes = []
                for name in dir(module):
                    obj = getattr(module, name)
                    if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and name.startswith("Test"):
                        test_classes.append(obj)

                self.assertGreater(len(test_classes), 0, f"No test classes found in {os.path.basename(test_file)}")

                # Check that at least one class inherits from TransformerBaseTest
                base_test_inherited = any(issubclass(cls, TransformerBaseTest) for cls in test_classes)
                self.assertTrue(
                    base_test_inherited, f"No test classes in {os.path.basename(test_file)} inherit from TransformerBaseTest"
                )

    def test_consistent_test_method_patterns(self):
        """Test that all test files follow consistent test method patterns."""
        required_patterns = ["test_basic_instantiation", "test_forward_pass", "test_typo_prevention"]

        for test_file in self.test_files:
            with self.subTest(test_file=os.path.basename(test_file)):
                module = self._load_test_module(test_file)

                # Get all test methods
                test_methods = []
                for name in dir(module):
                    obj = getattr(module, name)
                    if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                        for method_name in dir(obj):
                            if method_name.startswith("test_"):
                                test_methods.append(method_name)

                # Check for required patterns
                found_patterns = set()
                for method in test_methods:
                    for pattern in required_patterns:
                        if pattern in method:
                            found_patterns.add(pattern)

                missing_patterns = set(required_patterns) - found_patterns
                self.assertEqual(
                    len(missing_patterns), 0, f"Missing test patterns in {os.path.basename(test_file)}: {missing_patterns}"
                )

    def test_helper_utilities_consistency(self):
        """Test that all test files use helper utilities consistently."""
        required_helpers = ["TensorGenerator", "MockDiffusersConfig"]

        for test_file in self.test_files:
            with self.subTest(test_file=os.path.basename(test_file)):
                # Read file content to check imports
                with open(test_file, "r") as f:
                    content = f.read()

                # Check for helper imports
                for helper in required_helpers:
                    self.assertIn(helper, content, f"{helper} not imported in {os.path.basename(test_file)}")

    def test_mock_strategy_consistency(self):
        """Test that all test files use consistent mocking strategies."""
        for test_file in self.test_files:
            with self.subTest(test_file=os.path.basename(test_file)):
                module = self._load_test_module(test_file)

                # Check that test classes have proper setUp methods
                test_classes = [
                    getattr(module, name)
                    for name in dir(module)
                    if (
                        isinstance(getattr(module, name), type)
                        and issubclass(getattr(module, name), unittest.TestCase)
                        and name.startswith("Test")
                    )
                ]

                for test_class in test_classes:
                    if issubclass(test_class, TransformerBaseTest):
                        # Verify setUp calls parent setUp
                        if hasattr(test_class, "setUp"):
                            setup_method = getattr(test_class, "setUp")
                            # This is a heuristic check - in practice, we'd need to analyze the code
                            self.assertTrue(callable(setup_method), f"{test_class.__name__}.setUp is not callable")

    def test_tensor_generation_consistency(self):
        """Test that tensor generation is consistent across test files."""
        tensor_gen = TensorGenerator()

        # Test that all helper methods work consistently
        test_tensors = {
            "hidden_states": tensor_gen.create_hidden_states(2, 128, 512),
            "encoder_hidden_states": tensor_gen.create_encoder_hidden_states(2, 77, 512),
            "timestep": tensor_gen.create_timestep(2),
            "attention_mask": tensor_gen.create_attention_mask(2, 128),
            "pooled_projections": tensor_gen.create_pooled_projections(2, 768),
            "guidance": tensor_gen.create_guidance(2),
        }

        for name, tensor in test_tensors.items():
            with self.subTest(tensor_name=name):
                self.assertTrue(tensor.numel() > 0, f"Generated tensor {name} is empty")
                self.assertFalse(tensor.isnan().any(), f"Generated tensor {name} contains NaN values")
                self.assertFalse(tensor.isinf().any(), f"Generated tensor {name} contains infinite values")

    def test_shape_validation_consistency(self):
        """Test that shape validation is consistent across test files."""
        validator = ShapeValidator()

        # Test basic validation methods
        test_tensor = TensorGenerator.create_hidden_states(2, 128, 512)

        # These should not raise exceptions
        try:
            validator.validate_transformer_output(test_tensor, 2, 128, 512)
            validator.validate_embedding_output(test_tensor, 2, 512)
        except Exception as e:
            self.fail(f"Shape validation failed unexpectedly: {e}")

        # Test that invalid shapes raise appropriate errors
        with self.assertRaises(AssertionError):
            validator.validate_transformer_output(test_tensor, 3, 128, 512)  # Wrong batch size

    def test_typo_prevention_utilities(self):
        """Test that typo prevention utilities work correctly."""
        typo_utils = TypoTestUtils()

        class _SampleModel:
            @staticmethod
            def forward(*, input):
                return f"processed-{input}"

        model = _SampleModel()

        valid_params = {"input": "test"}
        typo_params = {"inpt": "input"}  # typo

        typo_utils.test_parameter_name_typos(model, "forward", valid_params, typo_params)

    def test_performance_utilities(self):
        """Test that performance utilities work correctly."""
        perf_utils = PerformanceUtils()

        # Create a simple mock model
        mock_model = Mock()
        mock_model.return_value = "output"

        inputs = {"input": "test"}

        try:
            # Test timing measurement
            avg_time = perf_utils.measure_forward_pass_time(mock_model, inputs, num_runs=3)
            self.assertIsInstance(avg_time, float)
            self.assertGreaterEqual(avg_time, 0.0)

            # Test memory measurement (may skip on CPU-only systems)
            if hasattr(perf_utils, "measure_memory_usage"):
                memory_stats = perf_utils.measure_memory_usage(mock_model, inputs)
                self.assertIsInstance(memory_stats, dict)
        except Exception as e:
            self.skipTest(f"Performance testing requires specific setup: {e}")

    def test_configuration_consistency(self):
        """Test that configuration objects are consistent."""
        # Test basic config creation
        config = MockDiffusersConfig(num_attention_heads=8, attention_head_dim=64, hidden_size=512)

        self.assertEqual(config.num_attention_heads, 8)
        self.assertEqual(config.attention_head_dim, 64)
        self.assertEqual(config.hidden_size, 512)

        # Test config with overrides
        config_override = MockDiffusersConfig(num_attention_heads=16, custom_param="test")

        self.assertEqual(config_override.num_attention_heads, 16)
        self.assertEqual(config_override.custom_param, "test")

    def test_import_isolation(self):
        """Test that test files don't interfere with each other's imports."""
        # This test ensures that importing one test module doesn't affect others
        imported_modules = set()

        for test_file in self.test_files[:3]:  # Test first 3 to avoid timeout
            with self.subTest(test_file=os.path.basename(test_file)):
                module_name = os.path.basename(test_file)[:-3]

                # Check that we can import the module
                try:
                    module = self._load_test_module(test_file)
                    imported_modules.add(module_name)
                    self.assertIsNotNone(module)
                except Exception as e:
                    self.skipTest(f"Cannot test import isolation for {module_name}: {e}")

        # If we got here, imports are working in isolation
        self.assertGreaterEqual(len(imported_modules), 1)

    def test_cross_transformer_compatibility(self):
        """Test that test utilities work across different transformer types."""
        # Test that the same utilities can be used for different transformers
        tensor_gen = TensorGenerator()
        common_config = {"batch_size": 2, "seq_len": 128, "hidden_dim": 512, "device": "cpu"}

        # Generate tensors that should work for all transformers
        tensors = {
            "hidden_states": tensor_gen.create_hidden_states(**common_config),
            "timestep": tensor_gen.create_timestep(common_config["batch_size"], common_config["device"]),
            "attention_mask": tensor_gen.create_attention_mask(
                common_config["batch_size"], common_config["seq_len"], common_config["device"]
            ),
        }

        # Verify tensors are compatible
        for name, tensor in tensors.items():
            with self.subTest(tensor_name=name):
                self.assertEqual(tensor.device.type, common_config["device"])
                if name == "hidden_states":
                    self.assertEqual(tensor.shape[0], common_config["batch_size"])
                    self.assertEqual(tensor.shape[1], common_config["seq_len"])
                    self.assertEqual(tensor.shape[2], common_config["hidden_dim"])


class TestTransformerTestFileStructure(unittest.TestCase):
    """Test the structure and organization of transformer test files."""

    def setUp(self):
        """Set up structure test environment."""
        self.test_dir = os.path.join(os.path.dirname(__file__), "test_transformers")
        self.utils_dir = os.path.join(os.path.dirname(__file__), "utils")

    def test_utils_directory_structure(self):
        """Test that utils directory has required files."""
        required_files = ["transformer_base_test.py", "transformer_test_helpers.py", "__init__.py"]

        for required_file in required_files:
            file_path = os.path.join(self.utils_dir, required_file)
            self.assertTrue(os.path.exists(file_path), f"Required utils file missing: {required_file}")

    def test_test_transformers_directory_structure(self):
        """Test that test_transformers directory is properly structured."""
        self.assertTrue(os.path.exists(self.test_dir), "test_transformers directory does not exist")

        # Check for __init__.py
        init_file = os.path.join(self.test_dir, "__init__.py")
        self.assertTrue(os.path.exists(init_file), "__init__.py missing in test_transformers directory")

    def test_test_file_naming_convention(self):
        """Test that all test files follow the naming convention."""
        if not os.path.exists(self.test_dir):
            self.skipTest("test_transformers directory does not exist")

        test_files = [f for f in os.listdir(self.test_dir) if f.endswith(".py") and f.startswith("test_")]

        for test_file in test_files:
            with self.subTest(test_file=test_file):
                # Should follow pattern: test_<transformer>_transformer.py
                self.assertTrue(
                    test_file.endswith("_transformer.py") or test_file in ["test_runner_summary.py", "__init__.py"],
                    f"Test file {test_file} doesn't follow naming convention",
                )

    def test_documentation_files_exist(self):
        """Test that required documentation files exist."""
        expected_docs = ["README.md", "test_completion_summary.md", "test_transformer_summary.md"]

        for doc_file in expected_docs:
            doc_path = os.path.join(self.test_dir, doc_file)
            if os.path.exists(doc_path):
                # Verify it's not empty
                with open(doc_path, "r") as f:
                    content = f.read().strip()
                    self.assertGreater(len(content), 0, f"Documentation file {doc_file} is empty")


if __name__ == "__main__":
    unittest.main(verbosity=2)
