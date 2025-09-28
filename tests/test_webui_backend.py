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
        self._home_patch = patch.dict(os.environ, {"HOME": str(self.temp_path)})
        self._home_patch.start()
        self.store = WebUIStateStore()

    def tearDown(self) -> None:
        self._home_patch.stop()
        self._tmpdir.cleanup()

    def test_initialization_creates_directory(self) -> None:
        expected = self.temp_path / ".simpletuner" / "webui"
        self.assertTrue(expected.exists())
        self.assertEqual(self.store.base_dir, expected)

    def test_save_and_load_defaults(self) -> None:
        defaults = WebUIDefaults(
            configs_dir="/path/to/configs",
            output_dir="/path/to/output",
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
            defaults = WebUIDefaults(configs_dir=f"/path/{idx}")
            self.store.save_defaults(defaults)

        threads = [threading.Thread(target=save_defaults, args=(i,)) for i in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        loaded = self.store.load_defaults()
        self.assertIsNotNone(loaded.configs_dir)
        self.assertTrue(loaded.configs_dir.startswith("/path/"))


class WebUIDefaultsUpdateTests(WebUIStateStoreTests):
    def test_update_configs_dir_only_preserves_other_fields(self) -> None:
        defaults = WebUIDefaults(configs_dir="/old/configs", output_dir="/old/output", active_config="old-config")
        self.store.save_defaults(defaults)

        loaded = self.store.load_defaults()
        loaded.configs_dir = "/new/configs"
        self.store.save_defaults(loaded)

        final = self.store.load_defaults()
        self.assertEqual(final.configs_dir, "/new/configs")
        self.assertEqual(final.output_dir, "/old/output")
        self.assertEqual(final.active_config, "old-config")

    def test_update_multiple_fields(self) -> None:
        defaults = WebUIDefaults()
        defaults.configs_dir = "/configs"
        defaults.output_dir = "/output"
        self.store.save_defaults(defaults)

        loaded = self.store.load_defaults()
        self.assertEqual(loaded.configs_dir, "/configs")
        self.assertEqual(loaded.output_dir, "/output")


class WebUIStateIntegrationTests(WebUIStateStoreTests):
    def test_full_onboarding_flow(self) -> None:
        self.store.record_onboarding_step("default_configs_dir", version=2, value="/home/user/configs")
        self.store.record_onboarding_step("default_output_dir", version=1, value="/home/user/output")

        defaults = self.store.load_defaults()
        defaults.configs_dir = "/home/user/configs"
        defaults.output_dir = "/home/user/output"
        self.store.save_defaults(defaults)

        state = self.store.load_state()
        self.assertEqual(state.defaults.configs_dir, "/home/user/configs")
        self.assertEqual(state.defaults.output_dir, "/home/user/output")
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


if __name__ == "__main__":
    unittest.main()
