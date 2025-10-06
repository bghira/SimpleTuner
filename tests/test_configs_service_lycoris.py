import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
    from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigServiceError, ConfigsService
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    ConfigStore = None  # type: ignore[assignment]
    ConfigsService = None  # type: ignore[assignment]
    ConfigServiceError = None  # type: ignore[assignment]
    _SKIP_REASON = f"Dependencies unavailable: {exc}"
else:
    _SKIP_REASON = ""


@unittest.skipIf(ConfigStore is None or ConfigsService is None or ConfigServiceError is None, _SKIP_REASON)
class ConfigsServiceLycorisTests(unittest.TestCase):
    """Test suite for Lycoris configuration management in ConfigsService."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self._instances_backup = ConfigStore._instances.copy()
        ConfigStore._instances = {}
        self.addCleanup(self._restore_config_store_instances)

        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.config_root = Path(self._tempdir.name).resolve()

        # Create isolated config store
        self.model_store = ConfigStore(config_dir=self.config_root, config_type="model")

    def _restore_config_store_instances(self) -> None:
        """Restore ConfigStore singleton instances."""
        ConfigStore._instances = self._instances_backup

    def _create_test_environment(self, env_name: str, lycoris_path: str = None) -> Path:
        """Create a test environment with optional lycoris config path."""
        env_dir = self.config_root / env_name
        env_dir.mkdir(parents=True, exist_ok=True)

        config = {
            "--model_family": "testfamily",
            "--model_type": "lora",
            "--lora_type": "lycoris",
        }
        if lycoris_path:
            config["lycoris_config"] = lycoris_path

        config_file = env_dir / "config.json"
        with config_file.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")

        return env_dir

    def test_get_lycoris_config_returns_none_when_environment_not_found(self) -> None:
        """Test that get_lycoris_config returns None when environment doesn't exist."""
        service = ConfigsService()

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            result = service.get_lycoris_config("nonexistent-env")

        self.assertIsNone(result)

    def test_get_lycoris_config_returns_none_when_no_lycoris_path(self) -> None:
        """Test that get_lycoris_config returns None when no lycoris_config path is set."""
        self._create_test_environment("test-env")
        service = ConfigsService()

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            result = service.get_lycoris_config("test-env")

        self.assertIsNone(result)

    def test_get_lycoris_config_returns_none_when_file_not_exists(self) -> None:
        """Test that get_lycoris_config returns None when lycoris config file doesn't exist."""
        self._create_test_environment("test-env", "test-env/lycoris_config.json")
        service = ConfigsService()

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            result = service.get_lycoris_config("test-env")

        self.assertIsNone(result)

    def test_get_lycoris_config_loads_existing_config(self) -> None:
        """Test that get_lycoris_config successfully loads an existing config."""
        env_dir = self._create_test_environment("test-env", "test-env/lycoris_config.json")

        # Create the lycoris config file
        lycoris_config = {
            "algo": "lokr",
            "multiplier": 1.0,
            "linear_dim": 10000,
            "linear_alpha": 1,
            "factor": 12,
        }
        lycoris_file = env_dir / "lycoris_config.json"
        with lycoris_file.open("w", encoding="utf-8") as handle:
            json.dump(lycoris_config, handle, indent=2)

        service = ConfigsService()
        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            result = service.get_lycoris_config("test-env")

        self.assertIsNotNone(result)
        self.assertEqual(result["algo"], "lokr")
        self.assertEqual(result["multiplier"], 1.0)
        self.assertEqual(result["linear_dim"], 10000)

    def test_save_lycoris_config_raises_error_for_nonexistent_environment(self) -> None:
        """Test that save_lycoris_config raises error when environment doesn't exist."""
        service = ConfigsService()
        lycoris_config = {"algo": "lokr", "multiplier": 1.0}

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            with self.assertRaises(ConfigServiceError) as context:
                service.save_lycoris_config("nonexistent-env", lycoris_config)

            self.assertEqual(context.exception.status_code, 404)
            self.assertIn("not found", context.exception.message)

    def test_save_lycoris_config_creates_default_path(self) -> None:
        """Test that save_lycoris_config creates default path when none exists."""
        self._create_test_environment("test-env")
        service = ConfigsService()

        lycoris_config = {
            "algo": "lokr",
            "multiplier": 1.0,
            "linear_dim": 10000,
        }

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            result = service.save_lycoris_config("test-env", lycoris_config)

        self.assertTrue(result["success"])
        self.assertIn("test-env/lycoris_config.json", result["path"])

        # Verify file was created
        lycoris_file = self.config_root / "test-env" / "lycoris_config.json"
        self.assertTrue(lycoris_file.exists())

        with lycoris_file.open("r", encoding="utf-8") as handle:
            saved_config = json.load(handle)

        self.assertEqual(saved_config["algo"], "lokr")
        self.assertEqual(saved_config["multiplier"], 1.0)

    def test_save_lycoris_config_uses_existing_path(self) -> None:
        """Test that save_lycoris_config uses existing path when set."""
        env_dir = self._create_test_environment("test-env", "test-env/my_lycoris.json")
        service = ConfigsService()

        lycoris_config = {
            "algo": "loha",
            "multiplier": 0.5,
        }

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            result = service.save_lycoris_config("test-env", lycoris_config)

        self.assertTrue(result["success"])

        # Verify file was created at the specified path
        lycoris_file = env_dir / "my_lycoris.json"
        self.assertTrue(lycoris_file.exists())

        with lycoris_file.open("r", encoding="utf-8") as handle:
            saved_config = json.load(handle)

        self.assertEqual(saved_config["algo"], "loha")
        self.assertEqual(saved_config["multiplier"], 0.5)

    def test_save_lycoris_config_rejects_invalid_environment_name(self) -> None:
        """Environment identifiers ending with .json should be stripped and result in not found if env doesn't exist."""
        self._create_test_environment("test-env")
        service = ConfigsService()

        lycoris_config = {"algo": "lokr", "multiplier": 1.0}

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            with self.assertRaises(ConfigServiceError) as ctx:
                service.save_lycoris_config("invalid.json", lycoris_config)

        # The .json suffix is stripped, so it looks for "invalid" environment which doesn't exist
        self.assertEqual(ctx.exception.status_code, 404)

    def test_save_lycoris_config_rejects_paths_outside_configs_dir(self) -> None:
        """Saving LyCORIS configs outside the workspace should raise an error."""
        self._create_test_environment("test-env", "../outside.json")
        service = ConfigsService()

        lycoris_config = {"algo": "lokr", "multiplier": 1.0}

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            with self.assertRaises(ConfigServiceError) as ctx:
                service.save_lycoris_config("test-env", lycoris_config)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("configs directory", ctx.exception.message)

    def test_validate_lycoris_config_accepts_valid_config(self) -> None:
        """Test that validate_lycoris_config accepts a valid configuration."""
        service = ConfigsService()

        lycoris_config = {
            "algo": "lokr",
            "multiplier": 1.0,
            "linear_dim": 10000,
            "linear_alpha": 1,
            "factor": 12,
        }

        result = service.validate_lycoris_config(lycoris_config)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)

    def test_validate_lycoris_config_rejects_missing_algo(self) -> None:
        """Test that validate_lycoris_config rejects config without algo field."""
        service = ConfigsService()

        lycoris_config = {
            "multiplier": 1.0,
        }

        result = service.validate_lycoris_config(lycoris_config)

        self.assertFalse(result["valid"])
        self.assertIn("Missing required field: 'algo'", result["errors"])

    def test_validate_lycoris_config_warns_on_unknown_algo(self) -> None:
        """Test that validate_lycoris_config warns on unknown algorithm."""
        service = ConfigsService()

        lycoris_config = {
            "algo": "unknown_algo",
            "multiplier": 1.0,
        }

        result = service.validate_lycoris_config(lycoris_config)

        self.assertTrue(result["valid"])  # Warning, not error
        self.assertTrue(any("Unknown algo" in w for w in result["warnings"]))

    def test_validate_lycoris_config_rejects_invalid_multiplier(self) -> None:
        """Test that validate_lycoris_config rejects invalid multiplier values."""
        service = ConfigsService()

        # Test negative multiplier
        lycoris_config = {
            "algo": "lokr",
            "multiplier": -1.0,
        }

        result = service.validate_lycoris_config(lycoris_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("multiplier" in e and "greater than 0" in e for e in result["errors"]))

        # Test zero multiplier
        lycoris_config["multiplier"] = 0
        result = service.validate_lycoris_config(lycoris_config)

        self.assertFalse(result["valid"])

        # Test non-numeric multiplier
        lycoris_config["multiplier"] = "not_a_number"
        result = service.validate_lycoris_config(lycoris_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("multiplier" in e and "number" in e for e in result["errors"]))

    def test_validate_lycoris_config_checks_linear_dim(self) -> None:
        """Test that validate_lycoris_config validates linear_dim."""
        service = ConfigsService()

        # Test non-integer linear_dim
        lycoris_config = {
            "algo": "lokr",
            "linear_dim": "not_an_int",
        }

        result = service.validate_lycoris_config(lycoris_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("linear_dim" in e and "integer" in e for e in result["errors"]))

        # Test negative linear_dim
        lycoris_config["linear_dim"] = -100
        result = service.validate_lycoris_config(lycoris_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("linear_dim" in e and "positive" in e for e in result["errors"]))

    def test_validate_lycoris_config_checks_factor_for_lokr(self) -> None:
        """Test that validate_lycoris_config validates factor for lokr algorithm."""
        service = ConfigsService()

        # Test non-integer factor
        lycoris_config = {
            "algo": "lokr",
            "factor": 12.5,
        }

        result = service.validate_lycoris_config(lycoris_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("factor" in e and "integer" in e for e in result["errors"]))

        # Test negative factor
        lycoris_config["factor"] = -12
        result = service.validate_lycoris_config(lycoris_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("factor" in e and "positive" in e for e in result["errors"]))

    def test_validate_lycoris_config_validates_apply_preset(self) -> None:
        """Test that validate_lycoris_config validates apply_preset structure."""
        service = ConfigsService()

        # Test invalid apply_preset type
        lycoris_config = {
            "algo": "lokr",
            "apply_preset": "not_a_dict",
        }

        result = service.validate_lycoris_config(lycoris_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("apply_preset" in e and "dictionary" in e for e in result["errors"]))

        # Test invalid target_module type
        lycoris_config["apply_preset"] = {
            "target_module": "not_a_list",
        }

        result = service.validate_lycoris_config(lycoris_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("target_module" in e and "list" in e for e in result["errors"]))

        # Test invalid module_algo_map type
        lycoris_config["apply_preset"] = {
            "module_algo_map": "not_a_dict",
        }

        result = service.validate_lycoris_config(lycoris_config)

        self.assertFalse(result["valid"])
        self.assertTrue(any("module_algo_map" in e and "dictionary" in e for e in result["errors"]))

    def test_validate_lycoris_config_accepts_complex_valid_config(self) -> None:
        """Test that validate_lycoris_config accepts complex valid configuration."""
        service = ConfigsService()

        lycoris_config = {
            "bypass_mode": True,
            "algo": "lokr",
            "multiplier": 1.0,
            "linear_dim": 10000,
            "linear_alpha": 1,
            "factor": 12,
            "apply_preset": {
                "target_module": ["Attention"],
                "module_algo_map": {
                    "Attention": {
                        "factor": 12,
                    }
                },
            },
        }

        result = service.validate_lycoris_config(lycoris_config)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["errors"]), 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
