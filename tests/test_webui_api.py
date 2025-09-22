"""
API endpoint tests for WebUI routes.

Tests all /api/webui/* endpoints and /web/* template routes.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from simpletuner.simpletuner_sdk.server import ServerMode, create_app
from simpletuner.simpletuner_sdk.server.services.webui_state import (
    WebUIDefaults,
    WebUIOnboardingState,
    WebUIStateStore,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_state_store(temp_dir):
    """Create a mocked WebUIStateStore."""
    with patch.dict(os.environ, {"HOME": str(temp_dir)}):
        store = WebUIStateStore()
        yield store


@pytest.fixture
def client(mock_state_store):
    """Create test client with mocked state store."""
    with patch("simpletuner.simpletuner_sdk.server.routes.webui_state.WebUIStateStore") as mock_cls:
        mock_cls.return_value = mock_state_store
        app = create_app(mode=ServerMode.TRAINER)
        yield TestClient(app)


class TestWebUIStateAPI:
    """Test /api/webui state management endpoints."""

    def test_get_webui_state(self, client, mock_state_store):
        """Test GET /api/webui/state endpoint."""
        # Set up test data
        defaults = WebUIDefaults(configs_dir="/test/configs", output_dir="/test/output")
        mock_state_store.save_defaults(defaults)

        response = client.get("/api/webui/state")
        assert response.status_code == 200

        data = response.json()
        assert "defaults" in data
        assert "onboarding" in data
        assert data["defaults"]["configs_dir"] == "/test/configs"
        assert data["defaults"]["output_dir"] == "/test/output"

    def test_update_onboarding_step(self, client):
        """Test POST /api/webui/onboarding/steps/{step_id} endpoint."""
        response = client.post(
            "/api/webui/onboarding/steps/default_configs_dir",
            json={"value": "/home/user/configs"}
        )

        assert response.status_code == 200
        data = response.json()

        # Find the step in the response
        step = next(
            (s for s in data["onboarding"]["steps"] if s["id"] == "default_configs_dir"),
            None
        )
        assert step is not None
        assert step["value"] == os.path.abspath(os.path.expanduser("/home/user/configs"))
        assert step["is_complete"] is True

    def test_update_onboarding_invalid_step(self, client):
        """Test updating non-existent onboarding step."""
        response = client.post(
            "/api/webui/onboarding/steps/invalid_step",
            json={"value": "test"}
        )

        assert response.status_code == 404
        assert "Unknown onboarding step" in response.json()["detail"]

    def test_update_onboarding_empty_required_value(self, client):
        """Test updating required step with empty value."""
        response = client.post(
            "/api/webui/onboarding/steps/default_configs_dir",
            json={"value": ""}
        )

        assert response.status_code == 422
        assert "required" in response.json()["detail"]

    def test_reset_onboarding(self, client, mock_state_store):
        """Test POST /api/webui/onboarding/reset endpoint."""
        # Set up some initial state
        mock_state_store.record_onboarding_step("test_step", 1, "value")

        response = client.post("/api/webui/onboarding/reset")
        assert response.status_code == 200

        data = response.json()
        assert data["onboarding"]["overlay_required"] is True
        assert len([s for s in data["onboarding"]["steps"] if s["is_complete"]]) == 0

    def test_update_defaults(self, client):
        """Test POST /api/webui/defaults/update endpoint."""
        response = client.post(
            "/api/webui/defaults/update",
            json={
                "configs_dir": "/new/configs",
                "output_dir": "/new/output",
                "active_config": "new-config"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["defaults"]["configs_dir"] == os.path.abspath("/new/configs")
        assert data["defaults"]["output_dir"] == os.path.abspath("/new/output")
        assert data["defaults"]["active_config"] == "new-config"

    def test_update_defaults_partial(self, client, mock_state_store):
        """Test partial update of defaults."""
        # Set initial state
        defaults = WebUIDefaults(
            configs_dir="/old/configs",
            output_dir="/old/output",
            active_config="old-config"
        )
        mock_state_store.save_defaults(defaults)

        # Update only configs_dir
        response = client.post(
            "/api/webui/defaults/update",
            json={"configs_dir": "/new/configs"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["defaults"]["configs_dir"] == os.path.abspath("/new/configs")
        assert data["defaults"]["output_dir"] == "/old/output"  # Unchanged
        assert data["defaults"]["active_config"] == "old-config"  # Unchanged


class TestWebUIRoutes:
    """Test /web/* template rendering routes."""

    def test_web_root_redirects_to_trainer(self, client):
        """Test that /web/ redirects to /web/trainer."""
        response = client.get("/web/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/web/trainer"

    def test_web_trainer_page(self, client):
        """Test /web/trainer renders template."""
        response = client.get("/web/trainer")
        assert response.status_code == 200
        assert "SimpleTuner Training Studio" in response.text
        assert "trainer_htmx.html" in response.text or "SimpleTuner" in response.text

    def test_trainer_tabs_basic(self, client, mock_state_store):
        """Test /web/trainer/tabs/basic renders with correct context."""
        # Set up test defaults
        defaults = WebUIDefaults(
            configs_dir="/test/configs",
            output_dir="/test/output"
        )
        mock_state_store.save_defaults(defaults)

        response = client.get("/web/trainer/tabs/basic")
        assert response.status_code == 200

        # Check that the save button is present
        assert "Save Changes" in response.text
        assert "fa-save" in response.text

    def test_trainer_tabs_model(self, client):
        """Test /web/trainer/tabs/model renders."""
        response = client.get("/web/trainer/tabs/model")
        assert response.status_code == 200
        assert "Model Configuration" in response.text or "model-config" in response.text

    def test_trainer_tabs_training(self, client):
        """Test /web/trainer/tabs/training renders."""
        response = client.get("/web/trainer/tabs/training")
        assert response.status_code == 200
        assert "Training Parameters" in response.text or "training-config" in response.text

    def test_trainer_tabs_advanced(self, client):
        """Test /web/trainer/tabs/advanced renders."""
        response = client.get("/web/trainer/tabs/advanced")
        assert response.status_code == 200
        assert "Advanced Options" in response.text or "advanced-config" in response.text

    def test_trainer_tabs_datasets(self, client):
        """Test /web/trainer/tabs/datasets renders."""
        response = client.get("/web/trainer/tabs/datasets")
        assert response.status_code == 200
        # Should render datasets tab
        assert response.status_code == 200

    def test_trainer_tabs_environments(self, client):
        """Test /web/trainer/tabs/environments renders."""
        response = client.get("/web/trainer/tabs/environments")
        assert response.status_code == 200
        # Should render environments tab
        assert response.status_code == 200

    def test_datasets_new_modal(self, client):
        """Test /web/datasets/new modal content."""
        response = client.get("/web/datasets/new")
        assert response.status_code == 200
        # Should render dataset card partial
        assert response.status_code == 200


class TestTrainingAPI:
    """Test training control API endpoints."""

    def test_validate_config(self, client):
        """Test POST /api/training/validate endpoint."""
        # Send minimal valid config
        form_data = {
            "--model_type": "lora",
            "--model_family": "flux",
            "--resolution": "1024",
        }

        response = client.post(
            "/api/training/validate",
            data=form_data
        )

        assert response.status_code == 200
        # Response should be HTML with validation results
        assert "alert" in response.text

    def test_save_training_config(self, client):
        """Test POST /api/training/config endpoint."""
        form_data = {
            "--model_type": "lora",
            "--output_dir": "/test/output",
        }

        with patch("simpletuner.simpletuner_sdk.api_state.APIState.set_state"):
            response = client.post(
                "/api/training/config",
                data=form_data
            )

            assert response.status_code == 200
            assert "Configuration saved" in response.text

    @pytest.mark.asyncio
    async def test_start_training(self, client):
        """Test POST /api/training/start endpoint."""
        form_data = {
            "--model_type": "lora",
            "--model_family": "flux",
            "--resolution": "1024",
        }

        with patch("simpletuner.simpletuner_sdk.server.routes.training.process_keeper") as mock_pk:
            mock_pk.submit_job = Mock()

            response = client.post(
                "/api/training/start",
                data=form_data
            )

            assert response.status_code == 200
            assert "Training Starting" in response.text
            mock_pk.submit_job.assert_called_once()

    def test_stop_training(self, client):
        """Test POST /api/training/stop endpoint."""
        with patch("simpletuner.simpletuner_sdk.api_state.APIState.get_state") as mock_get:
            with patch("simpletuner.simpletuner_sdk.server.routes.training.process_keeper") as mock_pk:
                mock_get.return_value = "test-job-id"
                mock_pk.terminate_process = Mock()

                response = client.post("/api/training/stop")

                assert response.status_code == 200
                assert "stop requested" in response.json()["message"]
                mock_pk.terminate_process.assert_called_with("test-job-id")

    def test_get_training_status(self, client):
        """Test GET /api/training/status endpoint."""
        with patch("simpletuner.simpletuner_sdk.api_state.APIState.get_state") as mock_get:
            mock_get.side_effect = lambda key, default=None: {
                "training_status": "running",
                "current_job_id": "test-job",
                "training_config": {"--model_type": "lora"}
            }.get(key, default)

            response = client.get("/api/training/status")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "running"
            assert data["job_id"] == "test-job"
            assert data["config"]["--model_type"] == "lora"


class TestConfigIntegration:
    """Test integration between configs and WebUI state."""

    def test_active_config_loads_in_basic_tab(self, client, mock_state_store):
        """Test that active config values load in basic tab."""
        # Set up config store mock
        with patch("simpletuner.simpletuner_sdk.server.routes.web.ConfigStore") as mock_config_cls:
            mock_store = Mock()
            mock_config_cls.return_value = mock_store

            # Mock active config
            mock_store.get_active_config.return_value = "test-config"
            mock_store.load_config.return_value = (
                {
                    "--job_id": "my-model",
                    "--output_dir": "/path/to/output",
                    "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev"
                },
                Mock()  # metadata
            )

            # Set WebUI defaults
            defaults = WebUIDefaults(configs_dir="/configs", output_dir="/default/output")
            mock_state_store.save_defaults(defaults)

            response = client.get("/web/trainer/tabs/basic")

            assert response.status_code == 200
            # Values should be populated from config
            assert "my-model" in response.text
            assert "/path/to/output" in response.text
            assert "black-forest-labs/FLUX.1-dev" in response.text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])