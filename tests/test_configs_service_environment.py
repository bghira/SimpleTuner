import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.helpers.models.registry import ModelRegistry

try:
    from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
    from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigServiceError, ConfigsService
    from simpletuner.simpletuner_sdk.server.services.example_configs_service import ExampleConfigInfo
    from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    ConfigStore = None  # type: ignore[assignment]
    ConfigsService = None  # type: ignore[assignment]
    ConfigServiceError = None  # type: ignore[assignment]
    ExampleConfigInfo = None  # type: ignore[assignment]
    _SKIP_REASON = f"Dependencies unavailable: {exc}"
else:
    _SKIP_REASON = ""


class _DummyModel:
    DEFAULT_MODEL_NAME = "dummy/model"
    DEFAULT_MODEL_FLAVOUR = "default"
    HUGGINGFACE_PATHS = {"default": "dummy/model"}


@unittest.skipIf(
    ConfigStore is None or ConfigsService is None or ExampleConfigInfo is None or ConfigServiceError is None, _SKIP_REASON
)
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

    def test_example_environment_rewrites_lycoris_path(self) -> None:
        service = ConfigsService()

        example_dir = Path(self._tempdir.name) / "example-demo"
        example_dir.mkdir(parents=True, exist_ok=True)

        example_config = {
            "lycoris_config": "config/examples/pixart.lycoris-lokr/lycoris_config.json",
            "data_backend_config": "config/examples/pixart.lycoris-lokr/multidatabackend.json",
        }

        (example_dir / "config.json").write_text(json.dumps(example_config, indent=2), encoding="utf-8")
        (example_dir / "lycoris_config.json").write_text("{}", encoding="utf-8")
        (example_dir / "multidatabackend.json").write_text("{}", encoding="utf-8")

        example_info = ExampleConfigInfo(
            name="demo",
            config_path=example_dir / "config.json",
            defaults={},
            description=None,
            dataloader_path=None,
            dataloader_payload=None,
        )

        request = SimpleNamespace(
            name="tidal-timber",
            model_family="testfamily",
            model_flavour="default",
            model_type="lora",
            lora_type="lycoris",
            description=None,
            example="demo",
            dataloader_path=None,
            create_dataloader=True,
        )

        with (
            patch.object(ConfigsService, "_get_store", return_value=self.model_store),
            patch(
                "simpletuner.simpletuner_sdk.server.services.configs_service.EXAMPLE_CONFIGS_SERVICE.get_example",
                return_value=example_info,
            ),
        ):
            result = service.create_environment(request)

        env_config_path = self.config_root / "tidal-timber" / "config.json"
        self.assertTrue(env_config_path.exists())

        with env_config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        expected_relative = "tidal-timber/lycoris_config.json"
        self.assertEqual(payload.get("lycoris_config"), expected_relative)
        self.assertNotIn("--lycoris_config", payload)
        self.assertEqual(result["config"].get("lycoris_config"), expected_relative)

        lycoris_abs = self.config_root / expected_relative
        self.assertTrue(lycoris_abs.exists())

        stray_root = self.config_root / "lycoris_config.json.json"
        self.assertFalse(stray_root.exists())
        self.assertFalse(stray_root.with_suffix(".metadata.json").exists())

    def test_create_environment_rejects_json_suffix_name(self) -> None:
        service = ConfigsService()

        request = SimpleNamespace(
            name="invalid-name.json",
            model_family="testfamily",
            model_flavour="default",
            model_type="lora",
            lora_type="lycoris",
            description=None,
            example=None,
            dataloader_path=None,
            create_dataloader=True,
        )

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            with self.assertRaises(ConfigServiceError) as ctx:
                service.create_environment(request)

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("must not end with", ctx.exception.message)

    def test_config_store_rejects_json_suffix_directly(self) -> None:
        store = ConfigStore(config_dir=self.config_root, config_type="model")
        metadata = store.save_config(
            "bad.json",
            config={"--model_family": "testfamily"},
            metadata=None,
            overwrite=False,
        )
        self.assertEqual(metadata.name, "bad")

    def test_webui_defaults_clears_missing_active_config(self) -> None:
        webui_dir = self.config_root / "webui_state"
        webui_dir.mkdir(parents=True, exist_ok=True)
        defaults_path = webui_dir / "defaults.json"
        defaults_payload = {
            "active_config": "ghost-config",
            "configs_dir": str(self.config_root),
            "output_dir": str(self.config_root / "output"),
            "event_polling_interval": 5,
            "event_stream_enabled": True,
            "auto_preserve_defaults": True,
            "theme": "dark",
        }
        defaults_path.write_text(json.dumps(defaults_payload), encoding="utf-8")

        store = WebUIStateStore(base_dir=webui_dir)
        loaded = store.load_defaults()
        self.assertIsNone(loaded.active_config)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
