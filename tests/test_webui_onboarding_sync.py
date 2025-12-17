import tempfile
import unittest
from pathlib import Path

from simpletuner.simpletuner_sdk.server.routes import webui_state as webui_routes
from simpletuner.simpletuner_sdk.server.services.webui_state import (
    OnboardingStepState,
    WebUIDefaults,
    WebUIState,
    WebUIStateStore,
)


class WebUIOnboardingSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = WebUIStateStore(base_dir=Path(self.tmpdir.name))

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_sync_applies_onboarding_values(self) -> None:
        defaults = WebUIDefaults(
            output_dir="/old/output",
            configs_dir="/old/configs",
            datasets_dir="/old/datasets",
            sync_onboarding_defaults=True,
            onboarding_sync_opt_out=["output_dir"],
        )
        self.store.save_defaults(defaults)

        onboarding = self.store.load_onboarding()
        onboarding.steps["default_output_dir"] = OnboardingStepState(completed_version=1, value="/new/output")
        onboarding.steps["default_configs_dir"] = OnboardingStepState(completed_version=2, value="/new/configs")
        onboarding.steps["default_datasets_dir"] = OnboardingStepState(completed_version=2, value="/new/datasets")
        self.store.save_onboarding(onboarding)

        synced = webui_routes._sync_defaults_from_onboarding(self.store, defaults)
        self.assertEqual(synced.configs_dir, "/new/configs")
        self.assertEqual(synced.datasets_dir, "/new/datasets")
        # output_dir is opted out
        self.assertEqual(synced.output_dir, "/old/output")

    def test_sync_disabled_keeps_values(self) -> None:
        defaults = WebUIDefaults(
            output_dir="/old/output",
            configs_dir="/old/configs",
            sync_onboarding_defaults=False,
        )
        onboarding = self.store.load_onboarding()
        onboarding.steps["default_configs_dir"] = OnboardingStepState(completed_version=2, value="/new/configs")
        self.store.save_onboarding(onboarding)
        self.store.save_defaults(defaults)

        synced = webui_routes._sync_defaults_from_onboarding(self.store, defaults)
        self.assertEqual(synced.configs_dir, "/old/configs")


if __name__ == "__main__":
    unittest.main()
