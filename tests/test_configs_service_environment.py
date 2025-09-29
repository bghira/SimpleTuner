import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigsService


class _DummyModel:
    DEFAULT_MODEL_NAME = "dummy/model"
    DEFAULT_MODEL_FLAVOUR = "default"
    HUGGINGFACE_PATHS = {"default": "dummy/model"}


class ConfigsServiceEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._instances_backup = ConfigStore._instances.copy()
        ConfigStore._instances = {}
        self.addCleanup(self._restore_config_store_instances)

        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.config_root = Path(self._tempdir.name).resolve()

        # Create isolated config stores for model and dataloader configs
        self.model_store = ConfigStore(config_dir=self.config_root, config_type="model")

        # Register a dummy model family so _resolve_pretrained_path works
        ModelRegistry.register("testfamily", _DummyModel)
        self.addCleanup(lambda: ModelRegistry._registry.pop("testfamily", None))  # type: ignore[attr-defined]

    def _restore_config_store_instances(self) -> None:
        ConfigStore._instances = self._instances_backup

    def test_create_environment_saves_flat_config(self) -> None:
        service = ConfigsService()

        request = SimpleNamespace(
            name="test-env",
            model_family="testfamily",
            model_flavour="default",
            model_type="lora",
            lora_type="standard",
            description="Test environment",
            example=None,
            dataloader_path=None,
            create_dataloader=True,
        )

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            result = service.create_environment(request)

        config_file = self.config_root / "test-env" / "config.json"
        self.assertTrue(config_file.exists(), "Training config not written to environment directory")

        with config_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.assertIsInstance(payload, dict)
        self.assertNotIn("_metadata", payload)
        self.assertNotIn("config", payload)
        self.assertIn("--model_family", payload)
        self.assertEqual(payload.get("--model_family"), "testfamily")

        dataloader_file = self.config_root / "test-env" / "multidatabackend.json"
        self.assertTrue(dataloader_file.exists(), "Default dataloader plan not created alongside environment")

        expected_rel_path = "test-env/multidatabackend.json"
        self.assertEqual(result["dataloader"]["path"], expected_rel_path)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
