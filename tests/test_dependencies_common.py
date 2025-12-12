import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.simpletuner_sdk.server.dependencies import common
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
from simpletuner.simpletuner_sdk.server.services.tab_service import TabService
from tests.unittest_support import run_async


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


class AssetVersionTests(unittest.TestCase):
    """Ensure asset versioning is exposed through dependencies and tab rendering."""

    def test_get_webui_defaults_provides_asset_version(self) -> None:
        """Defaults dependency should always emit an asset_version token."""
        resolved_defaults = {
            "configs_dir": "/tmp/configs",
            "output_dir": "/tmp/output",
            "theme": "dark",
            "event_polling_interval": 5,
            "event_stream_enabled": True,
            # Intentionally omit asset_version to exercise fallback
        }

        with (
            patch.object(common, "WebUIStateStore") as mock_store_cls,
            patch.object(common, "get_asset_version", return_value="fixed-token"),
        ):
            mock_store = mock_store_cls.return_value
            mock_store.get_defaults_bundle.return_value = {
                "raw": {},
                "resolved": resolved_defaults,
                "fallbacks": {},
            }

            defaults = run_async(common.get_webui_defaults())

        self.assertEqual(defaults["asset_version"], "fixed-token")
        self.assertEqual(defaults["configs_dir"], resolved_defaults["configs_dir"])

    def test_tab_render_context_includes_asset_version(self) -> None:
        """TabService should propagate asset_version into template context."""
        dummy_templates = SimpleNamespace(TemplateResponse=lambda request, name, context: {"name": name, "context": context})

        with patch(
            "simpletuner.simpletuner_sdk.server.services.tab_service.get_asset_version",
            return_value="context-token",
        ):
            service = TabService(dummy_templates)  # type: ignore[arg-type]

        result = run_async(
            service.render_tab(
                request=SimpleNamespace(),
                tab_name="datasets",
                fields=[],
                config_values={},
                sections=None,
                raw_config={},
                webui_defaults={"asset_version": "from-defaults"},
            )
        )

        context = result["context"]
        self.assertEqual(context["asset_version"], "from-defaults")

        # When asset_version is missing, fallback should be used
        with patch(
            "simpletuner.simpletuner_sdk.server.services.tab_service.get_asset_version",
            return_value="context-token",
        ):
            result_fallback = run_async(
                service.render_tab(
                    request=SimpleNamespace(),
                    tab_name="datasets",
                    fields=[],
                    config_values={},
                    sections=None,
                    raw_config={},
                    webui_defaults={},
                )
            )
        self.assertEqual(result_fallback["context"]["asset_version"], "context-token")


if __name__ == "__main__":
    unittest.main()
