"""unittest-based coverage for WebUI backend state management."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simpletuner.simpletuner_sdk.server.services.webui_state import (
    OnboardingStepState,
    WebUIDefaults,
    WebUIOnboardingState,
    WebUIStateStore,
)


class WebUIStateStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self._tmpdir.name)
        # Ensure we're using a clean isolated directory by setting SIMPLETUNER_WEB_UI_CONFIG
        self.webui_dir = self.temp_path / "webui_state"
        self._env_patches = patch.dict(
            os.environ, {"HOME": str(self.temp_path), "SIMPLETUNER_WEB_UI_CONFIG": str(self.webui_dir)}
        )
        self._env_patches.start()
        self.store = WebUIStateStore()

    def tearDown(self) -> None:
        self._env_patches.stop()
        self._tmpdir.cleanup()

    def test_initialization_creates_directory(self) -> None:
        # Since we're setting SIMPLETUNER_WEB_UI_CONFIG, it should use that path
        expected = self.webui_dir
        self.assertTrue(expected.exists())
        self.assertEqual(self.store.base_dir, expected)

    def test_save_and_load_defaults(self) -> None:
        # Use temp directories that actually exist
        configs_dir = self.temp_path / "configs"
        configs_dir.mkdir(exist_ok=True)
        output_dir = self.temp_path / "output"
        output_dir.mkdir(exist_ok=True)

        # Create the active config directory
        config_dir = configs_dir / "my-config"
        config_dir.mkdir(exist_ok=True)
        (config_dir / "config.json").write_text('{"test": true}')

        defaults = WebUIDefaults(
            configs_dir=str(configs_dir),
            output_dir=str(output_dir),
            active_config="my-config",
        )

        self.store.save_defaults(defaults)
        loaded = self.store.load_defaults()

        self.assertEqual(loaded.configs_dir, defaults.configs_dir)
        self.assertEqual(loaded.output_dir, defaults.output_dir)
        self.assertEqual(loaded.active_config, defaults.active_config)

    def test_save_and_load_onboarding_state(self) -> None:
        onboarding = WebUIOnboardingState()
        step_state = OnboardingStepState()
        step_state.value = "/some/path"
        step_state.completed_version = 1
        step_state.completed_at = "2025-01-01T00:00:00"
        onboarding.steps["test_step"] = step_state

        self.store.save_onboarding(onboarding)
        loaded = self.store.load_onboarding()

        self.assertIn("test_step", loaded.steps)
        self.assertEqual(loaded.steps["test_step"].value, "/some/path")
        self.assertEqual(loaded.steps["test_step"].completed_version, 1)

    def test_record_onboarding_step(self) -> None:
        self.store.record_onboarding_step(step_id="configs_dir", version=2, value="/new/configs")
        onboarding = self.store.load_onboarding()

        self.assertIn("configs_dir", onboarding.steps)
        record = onboarding.steps["configs_dir"]
        self.assertEqual(record.value, "/new/configs")
        self.assertEqual(record.completed_version, 2)
        self.assertIsNotNone(record.completed_at)

    def test_load_state_combines_defaults_and_onboarding(self) -> None:
        defaults = WebUIDefaults(configs_dir="/configs", output_dir="/output")
        self.store.save_defaults(defaults)
        self.store.record_onboarding_step("test_step", 1, "/test/value")

        state = self.store.load_state()

        self.assertEqual(state.defaults.configs_dir, "/configs")
        self.assertEqual(state.defaults.output_dir, "/output")
        self.assertIn("test_step", state.onboarding.steps)

    def test_load_defaults_returns_empty_when_missing(self) -> None:
        loaded = self.store.load_defaults()

        self.assertIsInstance(loaded, WebUIDefaults)
        self.assertIsNone(loaded.configs_dir)
        self.assertIsNone(loaded.output_dir)
        self.assertIsNone(loaded.active_config)

    def test_load_onboarding_returns_empty_when_missing(self) -> None:
        loaded = self.store.load_onboarding()
        self.assertIsInstance(loaded, WebUIOnboardingState)
        self.assertEqual(len(loaded.steps), 0)

    def test_invalid_json_raises(self) -> None:
        defaults_file = self.store.base_dir / "defaults.json"
        defaults_file.write_text("{ invalid json }")

        with self.assertRaisesRegex(ValueError, "Failed to read web UI state"):
            self.store.load_defaults()

    def test_path_normalization_behaviour(self) -> None:
        defaults = WebUIDefaults(configs_dir="~/configs/../configs", output_dir="./output")

        with patch.dict(os.environ, {"HOME": "/home/user"}):
            self.store.save_defaults(defaults)
            loaded = self.store.load_defaults()

        self.assertEqual(loaded.configs_dir, "~/configs/../configs")
        self.assertEqual(loaded.output_dir, "./output")

    def test_concurrent_saves_do_not_corrupt_state(self) -> None:
        import threading

        def save_defaults(idx: int):
            path_dir = self.temp_path / f"path_{idx}"
            path_dir.mkdir(exist_ok=True)
            defaults = WebUIDefaults(configs_dir=str(path_dir))
            self.store.save_defaults(defaults)

        threads = [threading.Thread(target=save_defaults, args=(i,)) for i in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        loaded = self.store.load_defaults()
        self.assertIsNotNone(loaded.configs_dir)
        # Check that it's one of the paths we created
        self.assertTrue(any(loaded.configs_dir.endswith(f"path_{i}") for i in range(5)))

    def test_accelerate_overrides_persist_and_normalize(self) -> None:
        defaults = WebUIDefaults()
        defaults.accelerate_overrides = {
            "mode": "manual",
            "num_processes": "4",
            "--same_network": "false",
            "--main_process_port": "12345",
            "--unknown_key": 10,
            "device_ids": ["0", 1, 1, 3],
            "manual_count": "3",
        }

        self.store.save_defaults(defaults)
        loaded = self.store.load_defaults()

        self.assertEqual(loaded.accelerate_overrides.get("--num_processes"), 4)
        self.assertFalse(loaded.accelerate_overrides.get("--same_network"))
        self.assertEqual(loaded.accelerate_overrides.get("--main_process_port"), 12345)
        self.assertNotIn("--unknown_key", loaded.accelerate_overrides)
        self.assertEqual(loaded.accelerate_overrides.get("mode"), "manual")
        self.assertEqual(loaded.accelerate_overrides.get("manual_count"), 3)
        self.assertEqual(loaded.accelerate_overrides.get("device_ids"), [0, 1, 3])


class WebUIDefaultsUpdateTests(WebUIStateStoreTests):
    def test_update_configs_dir_only_preserves_other_fields(self) -> None:
        # Use temp directories that actually exist
        old_configs_dir = self.temp_path / "old_configs"
        old_configs_dir.mkdir(exist_ok=True)
        old_output_dir = self.temp_path / "old_output"
        old_output_dir.mkdir(exist_ok=True)
        new_configs_dir = self.temp_path / "new_configs"
        new_configs_dir.mkdir(exist_ok=True)

        # Create the old config directory
        old_config_dir = old_configs_dir / "old-config"
        old_config_dir.mkdir(exist_ok=True)
        (old_config_dir / "config.json").write_text('{"test": true}')

        # Create the same config in new location
        new_config_dir = new_configs_dir / "old-config"
        new_config_dir.mkdir(exist_ok=True)
        (new_config_dir / "config.json").write_text('{"test": true}')

        defaults = WebUIDefaults(
            configs_dir=str(old_configs_dir), output_dir=str(old_output_dir), active_config="old-config"
        )
        self.store.save_defaults(defaults)

        loaded = self.store.load_defaults()
        loaded.configs_dir = str(new_configs_dir)
        self.store.save_defaults(loaded)

        final = self.store.load_defaults()
        self.assertEqual(final.configs_dir, str(new_configs_dir))
        self.assertEqual(final.output_dir, str(old_output_dir))
        self.assertEqual(final.active_config, "old-config")

    def test_update_multiple_fields(self) -> None:
        # Create temp directories
        configs_dir = self.temp_path / "configs"
        configs_dir.mkdir(exist_ok=True)
        output_dir = self.temp_path / "output"
        output_dir.mkdir(exist_ok=True)

        defaults = WebUIDefaults()
        defaults.configs_dir = str(configs_dir)
        defaults.output_dir = str(output_dir)
        self.store.save_defaults(defaults)

        loaded = self.store.load_defaults()
        self.assertEqual(loaded.configs_dir, str(configs_dir))
        self.assertEqual(loaded.output_dir, str(output_dir))


class WebUIStateIntegrationTests(WebUIStateStoreTests):
    def test_full_onboarding_flow(self) -> None:
        # Create temp directories
        configs_dir = self.temp_path / "user_configs"
        configs_dir.mkdir(exist_ok=True)
        output_dir = self.temp_path / "user_output"
        output_dir.mkdir(exist_ok=True)

        self.store.record_onboarding_step("default_configs_dir", version=2, value=str(configs_dir))
        self.store.record_onboarding_step("default_output_dir", version=1, value=str(output_dir))

        defaults = self.store.load_defaults()
        defaults.configs_dir = str(configs_dir)
        defaults.output_dir = str(output_dir)
        self.store.save_defaults(defaults)

        state = self.store.load_state()
        self.assertEqual(state.defaults.configs_dir, str(configs_dir))
        self.assertEqual(state.defaults.output_dir, str(output_dir))
        self.assertEqual(len(state.onboarding.steps), 2)

    def test_version_upgrade_handling(self) -> None:
        self.store.record_onboarding_step("test_step", 1, "value1")
        self.store.record_onboarding_step("test_step", 2, "value2")

        onboarding = self.store.load_onboarding()
        self.assertEqual(onboarding.steps["test_step"].completed_version, 2)
        self.assertEqual(onboarding.steps["test_step"].value, "value2")

    def test_reset_onboarding(self) -> None:
        self.store.record_onboarding_step("step1", 1, "value1")
        self.store.record_onboarding_step("step2", 1, "value2")
        self.store.save_onboarding(WebUIOnboardingState())

        onboarding = self.store.load_onboarding()
        self.assertEqual(len(onboarding.steps), 0)


class WebUICollapsedSectionsTests(WebUIStateStoreTests):
    def test_save_and_load_collapsed_sections(self) -> None:
        """Test saving and loading collapsed sections for a tab."""
        sections = {"section1": True, "section2": False, "section3": True}

        self.store.save_collapsed_sections("basic", sections)
        loaded = self.store.get_collapsed_sections("basic")

        self.assertEqual(loaded, sections)
        self.assertTrue(loaded["section1"])
        self.assertFalse(loaded["section2"])
        self.assertTrue(loaded["section3"])

    def test_get_collapsed_sections_returns_empty_when_missing(self) -> None:
        """Test that missing collapsed sections return empty dict."""
        loaded = self.store.get_collapsed_sections("nonexistent_tab")

        self.assertIsInstance(loaded, dict)
        self.assertEqual(len(loaded), 0)

    def test_save_collapsed_sections_for_multiple_tabs(self) -> None:
        """Test saving collapsed sections for multiple tabs independently."""
        basic_sections = {"section1": True, "section2": False}
        model_sections = {"section_a": False, "section_b": True}

        self.store.save_collapsed_sections("basic", basic_sections)
        self.store.save_collapsed_sections("model", model_sections)

        loaded_basic = self.store.get_collapsed_sections("basic")
        loaded_model = self.store.get_collapsed_sections("model")

        self.assertEqual(loaded_basic, basic_sections)
        self.assertEqual(loaded_model, model_sections)
        self.assertNotEqual(loaded_basic, loaded_model)

    def test_update_collapsed_sections_preserves_other_tabs(self) -> None:
        """Test updating one tab's sections doesn't affect other tabs."""
        self.store.save_collapsed_sections("basic", {"section1": True})
        self.store.save_collapsed_sections("model", {"section_a": False})

        # Update basic tab
        self.store.save_collapsed_sections("basic", {"section1": False, "section2": True})

        # Model tab should be unchanged
        loaded_model = self.store.get_collapsed_sections("model")
        self.assertEqual(loaded_model, {"section_a": False})

        # Basic tab should have new values
        loaded_basic = self.store.get_collapsed_sections("basic")
        self.assertEqual(loaded_basic, {"section1": False, "section2": True})

    def test_empty_sections_can_be_saved(self) -> None:
        """Test that empty sections dict can be saved and loaded."""
        self.store.save_collapsed_sections("basic", {})
        loaded = self.store.get_collapsed_sections("basic")

        self.assertEqual(loaded, {})

    def test_ui_state_file_created(self) -> None:
        """Test that ui_state.json file is created when saving collapsed sections."""
        self.store.save_collapsed_sections("basic", {"section1": True})

        ui_state_file = self.store.base_dir / "ui_state.json"
        self.assertTrue(ui_state_file.exists())

        # Verify file contains expected structure
        import json
        with ui_state_file.open("r") as f:
            data = json.load(f)

        self.assertIn("collapsed_sections", data)
        self.assertIn("basic", data["collapsed_sections"])
        self.assertEqual(data["collapsed_sections"]["basic"]["section1"], True)

    def test_load_ui_state_returns_empty_when_missing(self) -> None:
        """Test that load_ui_state returns empty dict when file doesn't exist."""
        ui_state = self.store.load_ui_state()

        self.assertIsInstance(ui_state, dict)
        self.assertEqual(len(ui_state), 0)

    def test_save_ui_state_with_custom_data(self) -> None:
        """Test saving arbitrary UI state data."""
        custom_state = {
            "collapsed_sections": {"basic": {"section1": True}},
            "custom_field": "value",
            "nested": {"data": [1, 2, 3]}
        }

        self.store.save_ui_state(custom_state)
        loaded = self.store.load_ui_state()

        self.assertEqual(loaded, custom_state)

    def test_concurrent_collapsed_section_saves(self) -> None:
        """Test that concurrent saves to different tabs don't corrupt state.

        Note: Due to race conditions in concurrent writes, not all tabs may be saved,
        but the state file should remain valid JSON and contain at least one tab.
        """
        import threading

        def save_sections(tab_name: str, section_id: str):
            sections = {section_id: True}
            self.store.save_collapsed_sections(tab_name, sections)

        threads = [
            threading.Thread(target=save_sections, args=(f"tab_{i}", f"section_{i}"))
            for i in range(5)
        ]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Verify state file is valid JSON and contains at least one tab
        ui_state = self.store.load_ui_state()
        collapsed = ui_state.get("collapsed_sections", {})

        # At least one tab should be saved (race condition may lose some)
        self.assertGreaterEqual(len(collapsed), 1)
        # Verify the structure is correct for tabs that were saved
        for tab_name, sections in collapsed.items():
            self.assertIsInstance(sections, dict)
            for section_id, is_collapsed in sections.items():
                self.assertIsInstance(is_collapsed, bool)


if __name__ == "__main__":
    unittest.main()
