import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.simpletuner_sdk.server.dependencies import common
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore


class ActiveConfigCacheTests(unittest.TestCase):
    """Tests for loading the active configuration via dependency helper."""

    def setUp(self) -> None:
        self._instances_backup = ConfigStore._instances.copy()
        ConfigStore._instances = {}
        common._load_active_config_cached.clear_cache()

    def tearDown(self) -> None:
        ConfigStore._instances = self._instances_backup
        common._load_active_config_cached.clear_cache()

    def test_load_active_config_from_flat_file(self) -> None:
        """Ensure flat `<name>.json` configs are supported when resolving the active entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_name = "flat-config"
            config_payload = {"model_family": "wan", "learning_rate": 0.123}
            (config_dir / f"{config_name}.json").write_text(json.dumps(config_payload))

            defaults = SimpleNamespace(configs_dir=str(config_dir), active_config=None)

            with (
                patch.object(common, "WebUIStateStore") as mock_store_cls,
                patch.object(ConfigStore, "get_active_config", return_value=config_name),
            ):
                mock_store = mock_store_cls.return_value
                mock_store.load_defaults.return_value = defaults

                resolved = common._load_active_config_cached()

        self.assertIsInstance(resolved, dict)
        self.assertEqual(resolved.get("learning_rate"), 0.123)
        self.assertEqual(resolved.get("model_family"), "wan")

    def test_load_active_config_from_directory_based_layout(self) -> None:
        """Ensure standard `<name>/config.json` layouts still resolve."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)
            config_name = "folder-config"
            payload = {"model_family": "wan", "learning_rate": 9.99}
            env_dir = config_dir / config_name
            env_dir.mkdir(parents=True, exist_ok=True)
            (env_dir / "config.json").write_text(json.dumps(payload))

            defaults = SimpleNamespace(configs_dir=str(config_dir), active_config=None)

            with (
                patch.object(common, "WebUIStateStore") as mock_store_cls,
                patch.object(ConfigStore, "get_active_config", return_value=config_name),
            ):
                mock_store = mock_store_cls.return_value
                mock_store.load_defaults.return_value = defaults

                resolved = common._load_active_config_cached()

        self.assertIsInstance(resolved, dict)
        self.assertEqual(resolved.get("learning_rate"), 9.99)
        self.assertEqual(resolved.get("model_family"), "wan")


if __name__ == "__main__":
    unittest.main()
