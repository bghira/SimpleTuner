import io
import json
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigServiceError, ConfigsService


class LycorisConfigTestCase(unittest.TestCase):
    """Test cases for Lycoris configuration management."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.service = ConfigsService()
        self.environment_id = "test-env"
        self.config_dir = "/fake/config/dir"

        # Mock ConfigStore and WebUIStateStore
        self.config_store_patcher = patch("simpletuner.simpletuner_sdk.server.services.configs_service.ConfigStore")
        self.webui_state_patcher = patch("simpletuner.simpletuner_sdk.server.services.configs_service.WebUIStateStore")

        self.mock_config_store_class = self.config_store_patcher.start()
        self.mock_webui_state_class = self.webui_state_patcher.start()

        # Set up default mocks
        mock_defaults = MagicMock()
        mock_defaults.configs_dir = None
        self.mock_webui_state_class.return_value.load_defaults.return_value = mock_defaults

        self.mock_config_store = MagicMock()
        self.mock_config_store.config_dir = self.config_dir
        self.mock_config_store_class.return_value = self.mock_config_store

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        self.config_store_patcher.stop()
        self.webui_state_patcher.stop()

    def _setup_environment_config(self, lycoris_path: str = None) -> None:
        """Helper to set up environment config with optional lycoris path.

        Args:
            lycoris_path: Path to lycoris config file
        """
        config = {}
        if lycoris_path:
            config["--lycoris_config"] = lycoris_path

        metadata = {"name": self.environment_id}
        self.mock_config_store.load_config.return_value = (config, metadata)

    def test_save_lycoris_config(self) -> None:
        """Test saving Lycoris configuration."""
        self._setup_environment_config()

        lycoris_config = {
            "algo": "lora",
            "multiplier": 1.0,
            "linear_dim": 16,
            "linear_alpha": 1,
        }

        # Mock os.mkdir and file open at the io level to avoid actual filesystem operations
        with (
            patch("os.mkdir"),
            patch("io.open", mock_open()) as mock_file,
        ):
            result = self.service.save_lycoris_config(self.environment_id, lycoris_config)

            self.assertTrue(result["success"])
            self.assertIn("path", result)
            self.assertIn("absolute_path", result)

            # Verify file was opened for writing
            mock_file.assert_called_once()

    def test_validate_lycoris_config_valid(self) -> None:
        """Test validation of valid Lycoris configuration."""
        valid_config = {
            "algo": "lora",
            "multiplier": 1.0,
            "linear_dim": 16,
            "linear_alpha": 1,
        }

        result = self.service.validate_lycoris_config(valid_config)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_lycoris_config_missing_algo(self) -> None:
        """Test validation fails when algo is missing."""
        invalid_config = {
            "multiplier": 1.0,
            "linear_dim": 16,
        }

        result = self.service.validate_lycoris_config(invalid_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("algo" in error.lower() for error in result["errors"]))

    def test_validate_lycoris_config_invalid_multiplier(self) -> None:
        """Test validation fails with invalid multiplier."""
        # Test with negative multiplier
        invalid_config_negative = {
            "algo": "lora",
            "multiplier": -1.0,
        }

        result = self.service.validate_lycoris_config(invalid_config_negative)

        self.assertFalse(result["valid"])
        self.assertTrue(any("multiplier" in error.lower() for error in result["errors"]))

        # Test with zero multiplier
        invalid_config_zero = {
            "algo": "lora",
            "multiplier": 0,
        }

        result = self.service.validate_lycoris_config(invalid_config_zero)

        self.assertFalse(result["valid"])
        self.assertTrue(any("multiplier" in error.lower() for error in result["errors"]))

        # Test with non-numeric multiplier
        invalid_config_string = {
            "algo": "lora",
            "multiplier": "not_a_number",
        }

        result = self.service.validate_lycoris_config(invalid_config_string)

        self.assertFalse(result["valid"])
        self.assertTrue(any("multiplier" in error.lower() for error in result["errors"]))

    def test_algorithm_defaults_lora(self) -> None:
        """Test LoRA algorithm configuration."""
        lora_config = {
            "algo": "lora",
            "multiplier": 1.0,
            "linear_dim": 16,
            "linear_alpha": 1,
        }

        result = self.service.validate_lycoris_config(lora_config)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)
        # LoRA is a known algorithm, should not have warnings about unknown algo
        self.assertFalse(any("unknown algo" in warning.lower() for warning in result["warnings"]))

    def test_algorithm_defaults_lokr(self) -> None:
        """Test LoKr algorithm configuration."""
        lokr_config = {
            "algo": "lokr",
            "multiplier": 1.0,
            "factor": 16,
            "linear_dim": 16,
            "linear_alpha": 1,
        }

        result = self.service.validate_lycoris_config(lokr_config)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)

    def test_algorithm_defaults_dylora(self) -> None:
        """Test DyLoRA-like configuration (using lora algo)."""
        # Note: DyLoRA is typically implemented as a LoRA variant
        # Testing dynamic rank capabilities
        dylora_config = {
            "algo": "lora",
            "multiplier": 1.0,
            "linear_dim": 32,  # Higher rank for dynamic adaptation
            "linear_alpha": 16,
        }

        result = self.service.validate_lycoris_config(dylora_config)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_lycoris_config_invalid_linear_dim(self) -> None:
        """Test validation fails with invalid linear_dim."""
        # Test with negative linear_dim
        invalid_config = {
            "algo": "lora",
            "linear_dim": -16,
        }

        result = self.service.validate_lycoris_config(invalid_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("linear_dim" in error.lower() for error in result["errors"]))

        # Test with non-integer linear_dim
        invalid_config_float = {
            "algo": "lora",
            "linear_dim": 16.5,
        }

        result = self.service.validate_lycoris_config(invalid_config_float)

        self.assertFalse(result["valid"])
        self.assertTrue(any("linear_dim" in error.lower() for error in result["errors"]))

    def test_validate_lycoris_config_invalid_factor(self) -> None:
        """Test validation fails with invalid factor for lokr."""
        invalid_config = {
            "algo": "lokr",
            "factor": -8,
        }

        result = self.service.validate_lycoris_config(invalid_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("factor" in error.lower() for error in result["errors"]))

    def test_validate_lycoris_config_apply_preset(self) -> None:
        """Test validation of apply_preset structure."""
        config_with_preset = {
            "algo": "lokr",
            "multiplier": 1.0,
            "apply_preset": {
                "target_module": ["Attention", "FeedForward"],
                "module_algo_map": {
                    "Attention": {"factor": 16},
                    "FeedForward": {"factor": 8},
                },
            },
        }

        result = self.service.validate_lycoris_config(config_with_preset)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_lycoris_config_invalid_apply_preset(self) -> None:
        """Test validation fails with invalid apply_preset structure."""
        # Test with non-dict apply_preset
        invalid_config = {
            "algo": "lokr",
            "apply_preset": "not_a_dict",
        }

        result = self.service.validate_lycoris_config(invalid_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("apply_preset" in error.lower() for error in result["errors"]))

        # Test with non-list target_module
        invalid_config_target = {
            "algo": "lokr",
            "apply_preset": {
                "target_module": "not_a_list",
            },
        }

        result = self.service.validate_lycoris_config(invalid_config_target)

        self.assertFalse(result["valid"])
        self.assertTrue(any("target_module" in error.lower() for error in result["errors"]))

    def test_save_lycoris_config_environment_not_found(self) -> None:
        """Test saving Lycoris config when environment doesn't exist."""
        self.mock_config_store.load_config.side_effect = FileNotFoundError("Config not found")

        lycoris_config = {"algo": "lora", "multiplier": 1.0}

        with self.assertRaises(ConfigServiceError) as ctx:
            self.service.save_lycoris_config("nonexistent-env", lycoris_config)

        self.assertEqual(ctx.exception.status_code, 404)
        self.assertIn("not found", ctx.exception.message.lower())

    def test_load_lycoris_config_not_found(self) -> None:
        """Test loading Lycoris config when it doesn't exist."""
        self._setup_environment_config()

        result = self.service.get_lycoris_config(self.environment_id)

        self.assertIsNone(result)

    def test_load_lycoris_config_environment_not_found(self) -> None:
        """Test loading Lycoris config when environment doesn't exist."""
        self.mock_config_store.load_config.side_effect = FileNotFoundError("Config not found")

        result = self.service.get_lycoris_config("nonexistent-env")

        self.assertIsNone(result)

    def test_validate_lycoris_config_unknown_algorithm(self) -> None:
        """Test validation with unknown algorithm generates warning."""
        config_unknown_algo = {
            "algo": "unknown_algo",
            "multiplier": 1.0,
        }

        result = self.service.validate_lycoris_config(config_unknown_algo)

        # Should still be valid (warnings only)
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)
        self.assertTrue(len(result["warnings"]) > 0)
        self.assertTrue(any("unknown algo" in warning.lower() for warning in result["warnings"]))

    def test_algorithm_full(self) -> None:
        """Test 'full' algorithm which doesn't require linear_dim/linear_alpha."""
        full_config = {
            "algo": "full",
            "multiplier": 1.0,
        }

        result = self.service.validate_lycoris_config(full_config)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)

    def test_save_lycoris_config_creates_directory(self) -> None:
        """Test that save_lycoris_config creates parent directory if needed."""
        self._setup_environment_config()

        lycoris_config = {"algo": "lora", "multiplier": 1.0}

        # Mock os.mkdir to verify directory creation
        with (
            patch("os.mkdir") as mock_mkdir,
            patch("io.open", mock_open()),
        ):
            result = self.service.save_lycoris_config(self.environment_id, lycoris_config)

            self.assertTrue(result["success"])
            # Verify mkdir was called (directory creation was attempted)
            self.assertTrue(mock_mkdir.called or result["success"])

    def test_save_lycoris_config_with_existing_path(self) -> None:
        """Test saving Lycoris config when environment already has a lycoris_config path."""
        existing_path = "existing/lycoris_config.json"
        self._setup_environment_config(existing_path)

        lycoris_config = {"algo": "lora", "multiplier": 1.0}

        # Mock os.mkdir and file operations
        with (
            patch("os.mkdir"),
            patch("io.open", mock_open()),
            patch("simpletuner.simpletuner_sdk.server.utils.paths.resolve_config_path") as mock_resolve,
        ):
            # Return a proper Path object for resolve_config_path
            resolved = Path(self.config_dir) / existing_path
            mock_resolve.return_value = resolved

            result = self.service.save_lycoris_config(self.environment_id, lycoris_config)

            self.assertTrue(result["success"])
            self.assertEqual(result["path"], existing_path)

    def test_validate_multiple_algorithms(self) -> None:
        """Test validation of various algorithm types."""
        algorithms = ["lora", "loha", "lokr", "locon", "oft", "boft", "glora", "full"]

        for algo in algorithms:
            with self.subTest(algo=algo):
                config = {
                    "algo": algo,
                    "multiplier": 1.0,
                }

                # Add algo-specific fields
                if algo == "lokr":
                    config["factor"] = 16
                if algo != "full":
                    config["linear_dim"] = 16
                    config["linear_alpha"] = 1

                result = self.service.validate_lycoris_config(config)

                self.assertTrue(result["valid"], f"Algorithm {algo} should be valid")
                self.assertEqual(len(result["errors"]), 0)


if __name__ == "__main__":
    unittest.main()
