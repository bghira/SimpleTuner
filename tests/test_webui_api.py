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

    def setUp(self) -> None:
        super().setUp()
        # Set up valid default directories for all tests
        self.default_configs_dir = str(self.temp_dir / "default_configs")
        self.default_output_dir = str(self.temp_dir / "default_output")

        # Create the directories
        os.makedirs(self.default_configs_dir, exist_ok=True)
        os.makedirs(self.default_output_dir, exist_ok=True)

        # Save default state
        defaults = WebUIDefaults(configs_dir=self.default_configs_dir, output_dir=self.default_output_dir)
        self.state_store.save_defaults(defaults)

    def test_get_webui_state(self) -> None:
        # Use temp directories
        configs_dir = str(self.temp_dir / "configs")
        output_dir = str(self.temp_dir / "output")

        # Create the directories
        os.makedirs(configs_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        defaults = WebUIDefaults(configs_dir=configs_dir, output_dir=output_dir)
        self.state_store.save_defaults(defaults)

        response = self.client.get("/api/webui/state")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("defaults", data)
        self.assertIn("onboarding", data)
        self.assertEqual(data["defaults"]["configs_dir"], configs_dir)
        self.assertEqual(data["defaults"]["output_dir"], output_dir)
        accelerate_step = next((s for s in data["onboarding"]["steps"] if s["id"] == "accelerate_defaults"), None)
        self.assertIsNotNone(accelerate_step)
        self.assertTrue(accelerate_step["required"])

    def test_update_onboarding_step(self) -> None:
        # Use a path within our temp directory that can actually be created
        test_configs_dir = str(self.temp_dir / "user_configs")

        response = self.client.post(
            "/api/webui/onboarding/steps/default_configs_dir",
            json={"value": test_configs_dir},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        step = next((s for s in data["onboarding"]["steps"] if s["id"] == "default_configs_dir"), None)
        self.assertIsNotNone(step)
        self.assertEqual(step["value"], os.path.abspath(os.path.expanduser(test_configs_dir)))
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
        # Use temp directories
        configs_dir = str(self.temp_dir / "configs")
        output_dir = str(self.temp_dir / "output")

        # Create the directories
        os.makedirs(configs_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Create the config file that active_config points to
        config_dir = Path(configs_dir) / "new-config"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "config.json"
        config_file.write_text('{"test": "config"}')

        response = self.client.post(
            "/api/webui/defaults/update",
            json={
                "configs_dir": configs_dir,
                "output_dir": output_dir,
                "active_config": "new-config",
                "accelerate_overrides": {"mode": "manual", "device_ids": [0, 1], "manual_count": 4},
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["defaults"]["configs_dir"], os.path.abspath(configs_dir))
        self.assertEqual(data["defaults"]["output_dir"], os.path.abspath(output_dir))
        self.assertEqual(data["defaults"]["active_config"], "new-config")
        overrides = data["defaults"].get("accelerate_overrides", {})
        self.assertEqual(overrides.get("mode"), "manual")
        self.assertEqual(overrides.get("device_ids"), [0, 1])
        self.assertEqual(overrides.get("manual_count"), 4)

    def test_update_defaults_partial(self) -> None:
        # Use temp directories
        old_configs_dir = str(self.temp_dir / "old_configs")
        old_output_dir = str(self.temp_dir / "old_output")
        new_configs_dir = str(self.temp_dir / "new_configs")

        # Create the directories
        os.makedirs(old_configs_dir, exist_ok=True)
        os.makedirs(old_output_dir, exist_ok=True)
        os.makedirs(new_configs_dir, exist_ok=True)

        # Create the config file that active_config points to in old directory
        old_config_dir = Path(old_configs_dir) / "old-config"
        old_config_dir.mkdir(parents=True, exist_ok=True)
        old_config_file = old_config_dir / "config.json"
        old_config_file.write_text('{"test": "old-config"}')

        # Also create the same config file in the new directory so validation passes
        new_config_dir = Path(new_configs_dir) / "old-config"
        new_config_dir.mkdir(parents=True, exist_ok=True)
        new_config_file = new_config_dir / "config.json"
        new_config_file.write_text('{"test": "old-config"}')

        defaults = WebUIDefaults(configs_dir=old_configs_dir, output_dir=old_output_dir, active_config="old-config")
        self.state_store.save_defaults(defaults)

        response = self.client.post("/api/webui/defaults/update", json={"configs_dir": new_configs_dir})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["defaults"]["configs_dir"], os.path.abspath(new_configs_dir))
        self.assertEqual(data["defaults"]["output_dir"], os.path.abspath(old_output_dir))
        self.assertEqual(data["defaults"]["active_config"], "old-config")

    def test_gpu_inventory_endpoint_returns_payload(self) -> None:
        response = self.client.get("/api/hardware/gpus")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("count", data)
        self.assertIn("optimal_processes", data)


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

    def test_trainer_tabs_advanced_removed(self) -> None:
        """Advanced tab was removed, should return 404."""
        response = self.client.get("/web/trainer/tabs/advanced")
        self.assertEqual(response.status_code, 404)

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

    def setUp(self) -> None:
        super().setUp()
        # Set up valid default directories for all tests
        self.default_configs_dir = str(self.temp_dir / "configs")
        self.default_output_dir = str(self.temp_dir / "output")

        # Create the directories
        os.makedirs(self.default_configs_dir, exist_ok=True)
        os.makedirs(self.default_output_dir, exist_ok=True)

        # Save default state
        defaults = WebUIDefaults(configs_dir=self.default_configs_dir, output_dir=self.default_output_dir)
        self.state_store.save_defaults(defaults)

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
                "--pretrained_model_name_or_path": "jimmycarter/LibreFlux-SimpleTuner",
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


class WebUICollapsedSectionsAPITestCase(_WebUIBaseTestCase):
    """Test /api/webui/ui-state/collapsed-sections/* endpoints."""

    def test_get_collapsed_sections_empty_by_default(self) -> None:
        """Test that GET returns empty dict when no state exists."""
        response = self.client.get("/api/webui/ui-state/collapsed-sections/basic")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data, {})

    def test_save_and_get_collapsed_sections(self) -> None:
        """Test POST saves sections and GET retrieves them."""
        sections = {"section1": True, "section2": False, "section3": True}

        # Save sections
        response = self.client.post("/api/webui/ui-state/collapsed-sections/basic", json={"sections": sections})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("basic", data["message"])

        # Retrieve sections
        response = self.client.get("/api/webui/ui-state/collapsed-sections/basic")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data, sections)

    def test_save_collapsed_sections_for_multiple_tabs(self) -> None:
        """Test saving sections for multiple tabs independently."""
        basic_sections = {"section1": True, "section2": False}
        model_sections = {"section_a": False, "section_b": True}

        # Save for basic tab
        response = self.client.post("/api/webui/ui-state/collapsed-sections/basic", json={"sections": basic_sections})
        self.assertEqual(response.status_code, 200)

        # Save for model tab
        response = self.client.post("/api/webui/ui-state/collapsed-sections/model", json={"sections": model_sections})
        self.assertEqual(response.status_code, 200)

        # Verify basic tab
        response = self.client.get("/api/webui/ui-state/collapsed-sections/basic")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), basic_sections)

        # Verify model tab
        response = self.client.get("/api/webui/ui-state/collapsed-sections/model")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), model_sections)

    def test_update_collapsed_sections_replaces_previous(self) -> None:
        """Test that updating sections replaces previous values."""
        # Save initial state
        initial_sections = {"section1": True, "section2": False}
        response = self.client.post("/api/webui/ui-state/collapsed-sections/basic", json={"sections": initial_sections})
        self.assertEqual(response.status_code, 200)

        # Update with new state
        updated_sections = {"section1": False, "section3": True}
        response = self.client.post("/api/webui/ui-state/collapsed-sections/basic", json={"sections": updated_sections})
        self.assertEqual(response.status_code, 200)

        # Verify updated state
        response = self.client.get("/api/webui/ui-state/collapsed-sections/basic")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data, updated_sections)
        self.assertFalse(data["section1"])
        self.assertNotIn("section2", data)
        self.assertTrue(data["section3"])

    def test_save_empty_sections(self) -> None:
        """Test that empty sections dict can be saved."""
        response = self.client.post("/api/webui/ui-state/collapsed-sections/basic", json={"sections": {}})

        self.assertEqual(response.status_code, 200)

        # Verify it was saved
        response = self.client.get("/api/webui/ui-state/collapsed-sections/basic")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {})

    def test_save_collapsed_sections_invalid_payload(self) -> None:
        """Test that invalid payload returns 422."""
        # Missing 'sections' key
        response = self.client.post("/api/webui/ui-state/collapsed-sections/basic", json={"invalid": "data"})

        self.assertEqual(response.status_code, 422)

    def test_tab_name_with_special_characters(self) -> None:
        """Test tab names with hyphens and underscores work correctly."""
        sections = {"section1": True}

        # Test with hyphen
        response = self.client.post("/api/webui/ui-state/collapsed-sections/advanced-settings", json={"sections": sections})
        self.assertEqual(response.status_code, 200)

        response = self.client.get("/api/webui/ui-state/collapsed-sections/advanced-settings")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), sections)

        # Test with underscore
        response = self.client.post("/api/webui/ui-state/collapsed-sections/model_config", json={"sections": sections})
        self.assertEqual(response.status_code, 200)

        response = self.client.get("/api/webui/ui-state/collapsed-sections/model_config")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), sections)

    def test_persistence_across_requests(self) -> None:
        """Test that state persists across multiple GET requests."""
        sections = {"section1": True, "section2": False}

        # Save sections
        response = self.client.post("/api/webui/ui-state/collapsed-sections/basic", json={"sections": sections})
        self.assertEqual(response.status_code, 200)

        # Make multiple GET requests
        for _ in range(3):
            response = self.client.get("/api/webui/ui-state/collapsed-sections/basic")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), sections)


if __name__ == "__main__":
    unittest.main()
