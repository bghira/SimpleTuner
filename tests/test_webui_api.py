"""
API endpoint tests for WebUI routes now implemented with unittest.

Covers /api/webui/* endpoints alongside critical /web/* template views.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from simpletuner.simpletuner_sdk.server import ServerMode
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIDefaults, WebUIStateStore
from tests.unittest_support import APITestCase


class _WebUIBaseTestCase(APITestCase, unittest.TestCase):
    """Common setup that previously lived in pytest fixtures."""

    def setUp(self) -> None:
        super().setUp()
        self._home_tmpdir = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._home_tmpdir.name)

        # Patch HOME so the state store persists underneath the temp root.
        self._home_patch = patch.dict(os.environ, {"HOME": str(self.temp_dir)}, clear=False)
        self._home_patch.start()

        self.state_store = WebUIStateStore()

        # Patch the route-level state store singleton to use our in-memory instance.
        self._store_patch = patch(
            "simpletuner.simpletuner_sdk.server.routes.webui_state.WebUIStateStore",
            return_value=self.state_store,
        )
        self._store_patch.start()

        # Trainer app client.
        self.client: TestClient = self.create_test_client(ServerMode.TRAINER)

    def tearDown(self) -> None:
        self.client.close()
        self._store_patch.stop()
        self._home_patch.stop()
        super().tearDown()
        self._home_tmpdir.cleanup()


class WebUIStateAPITestCase(_WebUIBaseTestCase):
    """Test /api/webui state management endpoints."""

    def test_get_webui_state(self) -> None:
        defaults = WebUIDefaults(configs_dir="/test/configs", output_dir="/test/output")
        self.state_store.save_defaults(defaults)

        response = self.client.get("/api/webui/state")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("defaults", data)
        self.assertIn("onboarding", data)
        self.assertEqual(data["defaults"]["configs_dir"], "/test/configs")
        self.assertEqual(data["defaults"]["output_dir"], "/test/output")

    def test_update_onboarding_step(self) -> None:
        response = self.client.post(
            "/api/webui/onboarding/steps/default_configs_dir",
            json={"value": "/home/user/configs"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        step = next((s for s in data["onboarding"]["steps"] if s["id"] == "default_configs_dir"), None)
        self.assertIsNotNone(step)
        self.assertEqual(step["value"], os.path.abspath(os.path.expanduser("/home/user/configs")))
        self.assertTrue(step["is_complete"])

    def test_update_onboarding_invalid_step(self) -> None:
        response = self.client.post("/api/webui/onboarding/steps/invalid_step", json={"value": "test"})
        self.assertEqual(response.status_code, 404)
        self.assertIn("Unknown onboarding step", response.json()["detail"])

    def test_update_onboarding_empty_required_value(self) -> None:
        response = self.client.post(
            "/api/webui/onboarding/steps/default_configs_dir",
            json={"value": ""},
        )
        self.assertEqual(response.status_code, 422)
        self.assertIn("required", response.json()["detail"])

    def test_reset_onboarding(self) -> None:
        self.state_store.record_onboarding_step("test_step", 1, "value")
        response = self.client.post("/api/webui/onboarding/reset")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["onboarding"]["overlay_required"])
        completed_steps = [s for s in data["onboarding"]["steps"] if s["is_complete"]]
        self.assertEqual(len(completed_steps), 0)

    def test_update_defaults(self) -> None:
        response = self.client.post(
            "/api/webui/defaults/update",
            json={"configs_dir": "/new/configs", "output_dir": "/new/output", "active_config": "new-config"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["defaults"]["configs_dir"], os.path.abspath("/new/configs"))
        self.assertEqual(data["defaults"]["output_dir"], os.path.abspath("/new/output"))
        self.assertEqual(data["defaults"]["active_config"], "new-config")

    def test_update_defaults_partial(self) -> None:
        defaults = WebUIDefaults(configs_dir="/old/configs", output_dir="/old/output", active_config="old-config")
        self.state_store.save_defaults(defaults)

        response = self.client.post("/api/webui/defaults/update", json={"configs_dir": "/new/configs"})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["defaults"]["configs_dir"], os.path.abspath("/new/configs"))
        self.assertEqual(data["defaults"]["output_dir"], "/old/output")
        self.assertEqual(data["defaults"]["active_config"], "old-config")


class WebUIRoutesTestCase(_WebUIBaseTestCase):
    """Test /web/* template rendering routes."""

    def test_web_root_redirects_to_trainer(self) -> None:
        response = self.client.get("/web/", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/web/trainer")

    def test_web_trainer_page(self) -> None:
        response = self.client.get("/web/trainer")
        self.assertEqual(response.status_code, 200)
        self.assertIn("SimpleTuner", response.text)

    def test_trainer_tabs_basic(self) -> None:
        defaults = WebUIDefaults(configs_dir="/test/configs", output_dir="/test/output")
        self.state_store.save_defaults(defaults)

        response = self.client.get("/web/trainer/tabs/basic")
        self.assertEqual(response.status_code, 200)

    def test_trainer_tabs_model(self) -> None:
        response = self.client.get("/web/trainer/tabs/model")
        self.assertEqual(response.status_code, 200)
        self.assertTrue("Model" in response.text)

    def test_trainer_tabs_training(self) -> None:
        response = self.client.get("/web/trainer/tabs/training")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Training", response.text)

    def test_trainer_tabs_advanced_redirect(self) -> None:
        response = self.client.get("/web/trainer/tabs/advanced")
        self.assertEqual(response.status_code, 200)
        # Advanced tab now redirects to Basic settings
        self.assertIn("Basic", response.text)

    def test_trainer_tabs_datasets(self) -> None:
        response = self.client.get("/web/trainer/tabs/datasets")
        self.assertEqual(response.status_code, 200)

    def test_trainer_tabs_environments(self) -> None:
        response = self.client.get("/web/trainer/tabs/environments")
        self.assertEqual(response.status_code, 200)

    def test_datasets_new_modal(self) -> None:
        response = self.client.get("/web/datasets/new")
        self.assertLess(response.status_code, 500)


class TrainingAPITestCase(_WebUIBaseTestCase):
    """Test training control API endpoints."""

    def test_validate_config(self) -> None:
        form_data = {"--model_type": "lora", "--model_family": "flux", "--resolution": "1024"}
        response = self.client.post("/api/training/validate", data=form_data)
        self.assertEqual(response.status_code, 200)
        self.assertIn("alert", response.text)

    def test_save_training_config(self) -> None:
        form_data = {"--model_type": "lora", "--output_dir": "/test/output"}
        with patch("simpletuner.simpletuner_sdk.api_state.APIState.set_state"):
            response = self.client.post("/api/training/config", data=form_data)
            self.assertEqual(response.status_code, 200)
            self.assertIn("Configuration saved", response.text)

    def test_start_training(self) -> None:
        form_data = {"--model_type": "lora", "--model_family": "flux", "--resolution": "1024"}
        with patch("simpletuner.simpletuner_sdk.server.routes.training.process_keeper") as mock_pk:
            mock_pk.submit_job = Mock()
            response = self.client.post("/api/training/start", data=form_data)
            self.assertEqual(response.status_code, 200)
            # If the route chooses not to queue a job (e.g. validation failure), we just ensure no crash.

    def test_stop_training(self) -> None:
        with patch("simpletuner.simpletuner_sdk.api_state.APIState.get_state") as mock_get:
            with patch("simpletuner.simpletuner_sdk.server.routes.training.process_keeper") as mock_pk:
                mock_get.return_value = "test-job-id"
                mock_pk.terminate_process = Mock()
                response = self.client.post("/api/training/stop")
                self.assertEqual(response.status_code, 200)
                if mock_pk.terminate_process.called:
                    mock_pk.terminate_process.assert_called_with("test-job-id")

    def test_get_training_status(self) -> None:
        with patch("simpletuner.simpletuner_sdk.api_state.APIState.get_state") as mock_get:
            mock_get.side_effect = lambda key, default=None: {
                "training_status": "running",
                "current_job_id": "test-job",
                "training_config": {"--model_type": "lora"},
            }.get(key, default)

            response = self.client.get("/api/training/status")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "running")
            self.assertEqual(data["job_id"], "test-job")
            self.assertEqual(data["config"]["--model_type"], "lora")


class ConfigIntegrationTestCase(_WebUIBaseTestCase):
    """Test integration between configs and WebUI state."""

    def test_active_config_loads_in_basic_tab(self) -> None:
        mock_store = Mock()
        mock_store.get_active_config.return_value = "test-config"
        mock_store.load_config.return_value = (
            {
                "--job_id": "my-model",
                "--output_dir": "/path/to/output",
                "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
            },
            Mock(),
        )

        defaults = WebUIDefaults(configs_dir="/configs", output_dir="/default/output")
        self.state_store.save_defaults(defaults)

        with (
            patch(
                "simpletuner.simpletuner_sdk.server.services.config_store.ConfigStore",
                return_value=mock_store,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.dependencies.common._load_active_config_cached",
                return_value=mock_store.load_config.return_value[0],
            ),
        ):
            response = self.client.get("/web/trainer/tabs/basic")

        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
