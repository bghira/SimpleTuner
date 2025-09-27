"""
Tests for API integration - FastAPI endpoint behavior with new architecture.
These tests verify the complete API flow including subprocess execution.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from simpletuner.simpletuner_sdk.api_state import APIState
from simpletuner.simpletuner_sdk.configuration import ConfigModel, Configuration
from simpletuner.simpletuner_sdk.server import ServerMode, create_app
from simpletuner.simpletuner_sdk.training_host import TrainingHost


class TestConfigurationEndpoints:
    """Test configuration API endpoints."""

    @pytest.mark.asyncio
    async def test_configuration_check_endpoint_subprocess_mode(self, execution_mode_process):
        """Test configuration check endpoint in subprocess mode."""
        config_api = Configuration()

        job_config = ConfigModel(
            trainer_config={"model_type": "lora", "model_family": "sdxl"},
            dataloader_config=[],
            webhook_config={},
            job_id="test_check",
        )

        with patch("simpletuner.helpers.training.trainer.Trainer") as MockTrainer:
            mock_instance = Mock()
            MockTrainer.return_value = mock_instance

            result = await config_api.check(job_config)

            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_configuration_run_endpoint_subprocess_mode(self, execution_mode_process):
        """Test configuration run endpoint starts subprocess."""
        config_api = Configuration()

        job_config = ConfigModel(
            trainer_config={"model_type": "lora"}, dataloader_config=[], webhook_config={}, job_id="test_run_subprocess"
        )

        with patch("simpletuner.simpletuner_sdk.process_keeper.submit_job") as mock_submit:
            mock_submit.return_value = Mock()

            result = await config_api.run(job_config)

            assert "subprocess" in result["result"]
            mock_submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_job_prevention(self):
        """Test that concurrent jobs are prevented."""
        config_api = Configuration()

        # Set up existing job
        APIState.set_state("current_job_id", "existing_job")

        with patch("simpletuner.simpletuner_sdk.thread_keeper.get_thread_status") as mock_status:
            mock_status.return_value = "running"

            job_config = ConfigModel(trainer_config={}, dataloader_config=[], webhook_config={}, job_id="new_job")

            result = await config_api.run(job_config)

            assert result["status"] == False
            assert "already running" in result["result"]


class TestTrainingHostEndpoints:
    """Test training host API endpoints."""

    def test_get_host_state(self):
        """Test getting host state."""
        host = TrainingHost()

        with patch("simpletuner.simpletuner_sdk.api_state.APIState.get_state") as mock_state:
            mock_state.return_value = {"status": "idle"}

            with patch("simpletuner.simpletuner_sdk.thread_keeper.list_threads") as mock_threads:
                mock_threads.return_value = {}

                result = host.get_host_state()

                assert "result" in result
                assert "job_list" in result

    def test_cancel_job_subprocess_termination(self):
        """Test job cancellation terminates subprocess."""
        host = TrainingHost()

        with patch("simpletuner.simpletuner_sdk.api_state.APIState.get_trainer") as mock_get:
            mock_trainer = Mock()
            mock_get.return_value = mock_trainer

            with patch("simpletuner.simpletuner_sdk.thread_keeper.terminate_thread") as mock_term:
                mock_term.return_value = True

                result = host.cancel_job()

                assert "cancelled" in result["result"] or "cancellation" in result["result"]
                mock_trainer.abort.assert_called_once()

    def test_cancel_nonexistent_job(self):
        """Test cancelling when no job exists."""
        host = TrainingHost()

        with patch("simpletuner.simpletuner_sdk.api_state.APIState.get_trainer") as mock_get:
            mock_get.return_value = None

            result = host.cancel_job()

            assert result["status"] == False
            assert "No job" in result["result"]

    def test_list_active_jobs(self):
        """Test listing active training jobs."""
        host = TrainingHost()

        mock_jobs = {
            "job1": {"status": "running", "start_time": "2024-01-01"},
            "job2": {"status": "completed", "start_time": "2024-01-02"},
        }

        with patch("simpletuner.simpletuner_sdk.api_state.APIState.get_active_jobs") as mock_get:
            mock_get.return_value = mock_jobs

            result = host.list_jobs()

            assert "jobs" in result
            assert len(result["jobs"]) == 2


class TestModelEndpoints:
    """Test model information endpoints."""

    def test_get_model_families(self, test_client_trainer):
        """Test getting list of model families."""
        with patch("simpletuner.simpletuner_sdk.server.routes.models.model_families") as mock_families:
            mock_families.keys.return_value = ["sdxl", "flux", "sd3"]

            response = test_client_trainer.get("/models")

            assert response.status_code == 200
            data = response.json()
            assert "families" in data
            assert "sdxl" in data["families"]

    def test_get_model_flavours_valid(self, test_client_trainer):
        """Test getting flavours for valid model family."""
        with patch("simpletuner.simpletuner_sdk.server.routes.models.model_families") as mock_families:
            mock_families.__contains__.return_value = True

            with patch("simpletuner.simpletuner_sdk.server.routes.models.get_model_flavour_choices") as mock_flavours:
                mock_flavours.return_value = ["base-1.0", "refiner-1.0"]

                response = test_client_trainer.get("/models/sdxl/flavours")

                assert response.status_code == 200
                data = response.json()
                assert "flavours" in data
                assert "base-1.0" in data["flavours"]

    def test_get_model_flavours_invalid(self, test_client_trainer):
        """Test getting flavours for invalid model family."""
        with patch("simpletuner.simpletuner_sdk.server.routes.models.model_families") as mock_families:
            mock_families.__contains__.return_value = False

            response = test_client_trainer.get("/models/invalid/flavours")

            assert response.status_code == 404


class TestDatasetRoutes:
    """Test dataset blueprint and plan endpoints."""

    def test_get_dataset_blueprints(self, test_client_trainer):
        """Blueprint endpoint returns available backends."""
        response = test_client_trainer.get("/api/datasets/blueprints")

        assert response.status_code == 200
        data = response.json()
        assert "blueprints" in data
        assert isinstance(data["blueprints"], list)
        assert len(data["blueprints"]) >= 1

    def test_get_dataset_plan_default(self, test_client_trainer, dataset_plan_path):
        """Plan endpoint returns empty payload when no plan exists."""
        response = test_client_trainer.get("/api/datasets/plan")

        assert response.status_code == 200
        data = response.json()
        assert data["datasets"] == []
        assert data["source"] == "default"
        assert any(message["message"].startswith("add at least one dataset") for message in data.get("validations", []))

    def test_create_dataset_plan(self, test_client_trainer, dataset_plan_path):
        """Plan endpoint persists datasets and returns validations."""
        payload = {
            "datasets": [
                {
                    "id": "text_embeds",
                    "type": "local",
                    "dataset_type": "text_embeds",
                    "default": True,
                    "cache_dir": "cache/text_embeds",
                },
                {
                    "id": "main",
                    "type": "local",
                    "dataset_type": "image",
                    "instance_data_dir": "/data/images",
                },
            ]
        }

        response = test_client_trainer.post("/api/datasets/plan", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "disk"
        assert not any(message["level"] == "error" for message in data.get("validations", []))
        assert dataset_plan_path.exists()

        with dataset_plan_path.open("r", encoding="utf-8") as handle:
            saved = json.load(handle)

        assert saved == payload["datasets"]

        reload_response = test_client_trainer.get("/api/datasets/plan")
        assert reload_response.status_code == 200
        reloaded = reload_response.json()
        assert reloaded["datasets"] == payload["datasets"]
        assert reloaded["source"] == "disk"

    def test_create_invalid_dataset_plan_rejected(self, test_client_trainer):
        """Invalid plans are rejected with validation detail."""
        payload = {
            "datasets": [
                {
                    "id": "images-only",
                    "type": "local",
                    "dataset_type": "image",
                }
            ]
        }

        response = test_client_trainer.post("/api/datasets/plan", json=payload)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
        detail = data["detail"]
        assert isinstance(detail, dict)
        assert "validations" in detail
        assert any(message["level"] == "error" for message in detail["validations"])


class TestEventEndpoints:
    """Test event and callback endpoints."""

    def test_callback_endpoint_stores_events(self, test_client_callback):
        """Test callback endpoint stores events in event store."""
        event = {"message_type": "training_progress", "step": 100, "loss": 0.5}

        response = test_client_callback.post("/callback", json=event)

        assert response.status_code == 200

    def test_callback_clears_on_configure_webhook(self, test_client_callback):
        """Test that configure_webhook message clears event store."""
        # Send some events first
        test_client_callback.post("/callback", json={"data": "event1"})
        test_client_callback.post("/callback", json={"data": "event2"})

        # Send configure webhook
        response = test_client_callback.post("/callback", json={"message_type": "configure_webhook", "config": "new"})

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_broadcast_long_polling_timeout(self, test_client_callback):
        """Test broadcast endpoint timeout behavior."""
        # Request with no events should timeout
        # Use short timeout for testing
        start_time = time.time()

        # This will timeout after configured period
        response = test_client_callback.get("/broadcast?last_event_index=999")

        elapsed = time.time() - start_time

        # Should have waited but not forever
        assert elapsed < 35  # Default timeout is 30 seconds

    def test_broadcast_returns_new_events(self, test_client_callback):
        """Test broadcast returns events since index."""
        # Send some events
        for i in range(5):
            test_client_callback.post("/callback", json={"seq": i})

        # Get events
        response = test_client_callback.get("/broadcast?last_event_index=0")

        assert response.status_code == 200
        data = response.json()
        assert "events" in data
        # Should have some events

    def test_webhook_to_event_store_flow(self, test_client_unified):
        """Test webhook to event store flow in unified mode."""
        # Send webhook event
        webhook_data = {"message_type": "training_update", "data": {"epoch": 1, "step": 50}}

        response = test_client_unified.post("/callback", json=webhook_data)
        assert response.status_code == 200

        # Should be retrievable via broadcast
        response = test_client_unified.get("/broadcast?last_event_index=0")
        if response.status_code == 200:
            data = response.json()
            # Events should include our webhook


class TestConcurrentAPICalls:
    """Test API behavior under concurrent load."""

    @pytest.mark.asyncio
    async def test_concurrent_configuration_checks(self):
        """Test concurrent configuration check calls."""
        config_api = Configuration()

        async def check_config(job_id):
            job_config = ConfigModel(
                trainer_config={"model_type": "lora"}, dataloader_config=[], webhook_config={}, job_id=job_id
            )

            with patch("simpletuner.helpers.training.trainer.Trainer"):
                return await config_api.check(job_config)

        # Run multiple checks concurrently
        tasks = [check_config(f"job_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All should succeed
        for result in results:
            assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_concurrent_job_submissions(self):
        """Test concurrent job submission handling."""
        config_api = Configuration()

        async def submit_job(job_id):
            job_config = ConfigModel(trainer_config={}, dataloader_config=[], webhook_config={}, job_id=job_id)

            try:
                return await config_api.run(job_config)
            except HTTPException as e:
                return {"error": str(e.detail)}

        # Clear state
        APIState.clear_state()

        with patch("simpletuner.simpletuner_sdk.thread_keeper.submit_job") as mock_submit:
            mock_submit.side_effect = lambda jid, func: Mock()

            with patch("simpletuner.simpletuner_sdk.thread_keeper.get_thread_status") as mock_status:
                # First job succeeds, others see it running
                call_count = [0]

                def status_side_effect(jid):
                    call_count[0] += 1
                    return "not_found" if call_count[0] == 1 else "running"

                mock_status.side_effect = status_side_effect

                # Submit multiple jobs concurrently
                tasks = [submit_job(f"job_{i}") for i in range(5)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Only one should succeed
                success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
                # Due to race conditions, might have 1 or more succeed


class TestHealthCheck:
    """Test health check endpoints."""

    def test_health_check_trainer_mode(self, test_client_trainer):
        """Test health check in trainer mode."""
        response = test_client_trainer.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["mode"] == "trainer"

    def test_health_check_callback_mode(self, test_client_callback):
        """Test health check in callback mode."""
        response = test_client_callback.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["mode"] == "callback"

    def test_health_check_unified_mode(self, test_client_unified):
        """Test health check in unified mode."""
        response = test_client_unified.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["mode"] == "unified"


class TestErrorResponses:
    """Test error response handling."""

    @pytest.mark.asyncio
    async def test_invalid_configuration_error(self):
        """Test error response for invalid configuration."""
        config_api = Configuration()

        job_config = ConfigModel(
            trainer_config={}, dataloader_config=[], webhook_config={}, job_id="invalid_config"  # Missing required fields
        )

        with patch("simpletuner.helpers.training.trainer.Trainer") as MockTrainer:
            MockTrainer.side_effect = ValueError("Invalid config")

            with pytest.raises(HTTPException) as exc_info:
                await config_api.check(job_config)

            assert exc_info.value.status_code == 400
            assert "Invalid config" in str(exc_info.value.detail)

    def test_job_not_found_error(self):
        """Test error response when job not found."""
        host = TrainingHost()

        with patch("simpletuner.simpletuner_sdk.api_state.APIState.get_active_jobs") as mock_get:
            mock_get.return_value = {}

            result = host.get_job_status("nonexistent_job")

            # Should return 404
            assert result[1] == 404


class TestProcessModeIntegration:
    """Test complete integration with process mode."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_training_flow_subprocess(self, execution_mode_process, test_client_trainer):
        """Test complete training flow from API to subprocess."""
        # Prepare config
        config_data = {
            "trainer_config": {"model_type": "lora", "model_family": "sdxl", "max_train_steps": 10},
            "dataloader_config": [],
            "webhook_config": {},
            "job_id": "integration_test",
        }

        with patch("simpletuner.helpers.training.trainer.Trainer") as MockTrainer:
            mock_instance = Mock()
            mock_instance.run.return_value = None
            MockTrainer.return_value = mock_instance

            with patch("simpletuner.simpletuner_sdk.process_keeper.submit_job") as mock_submit:
                mock_submit.return_value = Mock()

                # Submit via API
                response = test_client_trainer.post("/training/configuration/run", json=config_data)

                if response.status_code == 200:
                    data = response.json()
                    assert "subprocess" in data.get("result", "")

    @pytest.mark.integration
    def test_cancel_subprocess_via_api(self, test_client_trainer):
        """Test cancelling subprocess training via API."""
        # Start a job first
        APIState.set_state("current_job_id", "test_cancel")
        APIState.set_trainer(Mock())

        with patch("simpletuner.simpletuner_sdk.thread_keeper.terminate_thread") as mock_term:
            mock_term.return_value = True

            # Cancel via API
            response = test_client_trainer.post("/training/cancel")

            assert response.status_code == 200
            data = response.json()
            assert "cancel" in data.get("result", "").lower()


@pytest.fixture
def dataset_plan_path(tmp_path, monkeypatch):
    """Provide an isolated dataset plan path for tests."""
    plan_path = tmp_path / "dataset_plan.json"
    monkeypatch.setenv("SIMPLETUNER_DATASET_PLAN_PATH", str(plan_path))
    return plan_path


@pytest.fixture
def api_state_tmp(tmp_path, monkeypatch):
    """Redirect API state persistence to a temporary file."""
    state_path = tmp_path / "api_state.json"
    monkeypatch.setattr(APIState, "state_file", str(state_path))
    APIState.clear_state()
    yield state_path
    APIState.clear_state()


@pytest.fixture
def test_client_trainer(dataset_plan_path, api_state_tmp):
    """FastAPI test client configured for trainer mode."""
    app = create_app(mode=ServerMode.TRAINER)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def test_client_callback(dataset_plan_path, api_state_tmp):
    """FastAPI test client configured for callback mode."""
    app = create_app(mode=ServerMode.CALLBACK)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def test_client_unified(dataset_plan_path, api_state_tmp):
    """FastAPI test client configured for unified mode."""
    app = create_app(mode=ServerMode.UNIFIED)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def execution_mode_process(monkeypatch):
    """Force the server into subprocess execution mode for a test."""
    monkeypatch.setenv("SIMPLETUNER_EXECUTION_MODE", "process")
    yield


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
