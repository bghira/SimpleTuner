"""Common unittest helpers for WebUI Selenium flows."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from tests.selenium_support import SeleniumTestCase


class WebUITestCase(SeleniumTestCase):
    """Base selenium test case that provisions state/config directories."""

    def setUp(self) -> None:
        super().setUp()
        self.state_dir = self.home_path / ".simpletuner" / "webui"
        self.config_dir = self.home_path / "configs"
        if self.state_dir.exists():
            shutil.rmtree(self.state_dir)
        if self.config_dir.exists():
            shutil.rmtree(self.config_dir)
        self.state_dir.mkdir(parents=True)
        self.config_dir.mkdir()
        (self.state_dir / "onboarding.json").write_text(json.dumps({"steps": {}}), encoding="utf-8")
        self._seed_default_environment()

    def seed_defaults(
        self,
        *,
        configs_dir: Path | None = None,
        output_dir: str = "/tmp/output",
        active_config: str | None = "default",
    ) -> None:
        defaults = {
            "configs_dir": str(configs_dir or self.config_dir),
            "output_dir": output_dir,
        }
        if active_config:
            defaults["active_config"] = active_config
        (self.state_dir / "defaults.json").write_text(json.dumps(defaults), encoding="utf-8")

    def write_config(self, name: str, payload: dict) -> Path:
        config_path = self.config_dir / f"{name}.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        return config_path

    def with_sample_environment(self) -> None:
        self.write_config(
            "test-config",
            {
                "--job_id": "test-model",
                "--output_dir": "/test/output",
                "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
            },
        )
        self.seed_defaults(active_config="test-config")

    def _seed_default_environment(self) -> None:
        default_env_dir = self.config_dir / "default"
        default_env_dir.mkdir(parents=True, exist_ok=True)
        default_config = {
            "--model_family": "flux",
            "--model_type": "lora",
            "--model_flavour": "flux-dev",
            "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
            "--output_dir": "/tmp/output",
            "--data_backend_config": str(default_env_dir / "multidatabackend.json"),
            "--job_id": "autogen-default",
            "--report_to": "none",
        }
        (default_env_dir / "config.json").write_text(json.dumps(default_config), encoding="utf-8")
        (default_env_dir / "multidatabackend.json").write_text("[]", encoding="utf-8")

    @staticmethod
    def dismiss_onboarding(driver) -> None:
        """Hide the onboarding overlay if it is currently visible."""

        driver.execute_script(
            """
            if (window.Alpine && Alpine.store && Alpine.store('trainer')) {
                const store = Alpine.store('trainer');
                store.overlayVisible = false;
                store.activeOnboardingStep = null;
                store.onboardingSteps = [];
            }
            const overlay = document.querySelector('.onboarding-overlay');
            if (overlay) {
                overlay.style.display = 'none';
            }
            """
        )
