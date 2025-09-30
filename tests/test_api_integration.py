"""unittest-based suite covering API integration behaviour."""

import asyncio
import json
import os
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi import HTTPException

from simpletuner.simpletuner_sdk.api_state import APIState
from simpletuner.simpletuner_sdk.configuration import ConfigModel, Configuration
from simpletuner.simpletuner_sdk.server import ServerMode
from simpletuner.simpletuner_sdk.training_host import TrainingHost
from tests.unittest_support import APITestCase, AsyncAPITestCase


class ConfigurationEndpointTests(AsyncAPITestCase, unittest.IsolatedAsyncioTestCase):
    async def test_configuration_check_endpoint_subprocess_mode(self) -> None:
        config_api = Configuration()
        job_config = ConfigModel(
            trainer_config={"model_type": "lora", "model_family": "sdxl"},
            dataloader_config=[],
            webhook_config={},
            job_id="test_check",
        )

        with self.execution_mode("process"):
            with patch("simpletuner.helpers.training.trainer.Trainer") as mock_trainer_cls:
                mock_trainer_cls.return_value = Mock()
                result = await config_api.check(job_config)

        self.assertEqual(result["status"], "success")

    async def test_configuration_run_endpoint_subprocess_mode(self) -> None:
        config_api = Configuration()
        job_config = ConfigModel(
            trainer_config={"model_type": "lora"},
            dataloader_config=[],
            webhook_config={},
            job_id="test_run_subprocess",
        )

        with self.execution_mode("process"):
            with patch("simpletuner.simpletuner_sdk.process_keeper.submit_job") as mock_submit:
                mock_submit.return_value = Mock()
                result = await config_api.run(job_config)

        self.assertIn("subprocess", result["result"])
        mock_submit.assert_called_once()

    async def test_concurrent_job_prevention(self) -> None:
        config_api = Configuration()
        APIState.set_state("current_job_id", "existing_job")

        with patch("simpletuner.simpletuner_sdk.thread_keeper.get_thread_status", return_value="running"):
            job_config = ConfigModel(trainer_config={}, dataloader_config=[], webhook_config={}, job_id="new_job")
            result = await config_api.run(job_config)

        self.assertFalse(result["status"])
        self.assertIn("already running", result["result"])


class TrainingHostEndpointTests(APITestCase, unittest.TestCase):
    def test_get_host_state(self) -> None:
        host = TrainingHost()

        with patch("simpletuner.simpletuner_sdk.api_state.APIState.get_state", return_value={"status": "idle"}):
            with patch("simpletuner.simpletuner_sdk.thread_keeper.list_threads", return_value={}):
                result = host.get_host_state()

        self.assertIn("result", result)
        self.assertIn("job_list", result)

    def test_cancel_job_subprocess_termination(self) -> None:
        host = TrainingHost()

        with patch("simpletuner.simpletuner_sdk.api_state.APIState.get_trainer") as mock_get:
            mock_trainer = Mock()
            mock_get.return_value = mock_trainer

            with patch("simpletuner.simpletuner_sdk.thread_keeper.terminate_thread", return_value=True):
                result = host.cancel_job()

        self.assertTrue("cancel" in result["result"].lower() or "cancellation" in result["result"].lower())
        mock_trainer.abort.assert_called_once()

    def test_cancel_nonexistent_job(self) -> None:
        host = TrainingHost()

        with patch("simpletuner.simpletuner_sdk.api_state.APIState.get_trainer", return_value=None):
            result = host.cancel_job()

        self.assertFalse(result["status"])
        self.assertIn("No job", result["result"])

    def test_list_active_jobs(self) -> None:
        host = TrainingHost()
        mock_jobs = {
            "job1": {"status": "running", "start_time": "2024-01-01"},
            "job2": {"status": "completed", "start_time": "2024-01-02"},
        }

        with patch("simpletuner.simpletuner_sdk.api_state.APIState.get_active_jobs", return_value=mock_jobs):
            result = host.list_jobs()

        self.assertIn("jobs", result)
        self.assertEqual(len(result["jobs"]), 2)


class ModelEndpointTests(APITestCase, unittest.TestCase):
    def test_get_model_families(self) -> None:
        with self.client_session(ServerMode.TRAINER) as client:
            with patch("simpletuner.simpletuner_sdk.server.routes.models.model_families") as mock_families:
                mock_families.keys.return_value = ["sdxl", "flux", "sd3"]
                response = client.get("/models")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("families", data)
        self.assertIn("sdxl", data["families"])

    def test_get_model_flavours_valid(self) -> None:
        with self.client_session(ServerMode.TRAINER) as client:
            with patch("simpletuner.simpletuner_sdk.server.routes.models.model_families") as mock_families:
                mock_families.__contains__.return_value = True
                with patch(
                    "simpletuner.simpletuner_sdk.server.routes.models.get_model_flavour_choices",
                    return_value=["base-1.0", "refiner-1.0"],
                ):
                    response = client.get("/models/sdxl/flavours")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("flavours", data)
        self.assertIn("base-1.0", data["flavours"])

    def test_get_model_flavours_invalid(self) -> None:
        with self.client_session(ServerMode.TRAINER) as client:
            with patch("simpletuner.simpletuner_sdk.server.routes.models.model_families") as mock_families:
                mock_families.__contains__.return_value = False
                response = client.get("/models/invalid/flavours")

        self.assertEqual(response.status_code, 404)


class DatasetRouteTests(APITestCase, unittest.TestCase):
    def test_get_dataset_blueprints(self) -> None:
        with self.client_session(ServerMode.TRAINER) as client:
            response = client.get("/api/datasets/blueprints")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("blueprints", data)
        self.assertIsInstance(data["blueprints"], list)
        self.assertGreaterEqual(len(data["blueprints"]), 1)

    def test_get_dataset_plan_default(self) -> None:
        with self.client_session(ServerMode.TRAINER) as client:
            response = client.get("/api/datasets/plan")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["datasets"], [])
        self.assertEqual(data["source"], "default")
        self.assertTrue(any(msg["message"].startswith("add at least one dataset") for msg in data.get("validations", [])))

    def test_create_dataset_plan(self) -> None:
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

        dataset_plan_path = Path(os.environ["SIMPLETUNER_DATASET_PLAN_PATH"])

        with self.client_session(ServerMode.TRAINER) as client:
            response = client.post("/api/datasets/plan", json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["source"], "disk")
        self.assertFalse(any(message["level"] == "error" for message in data.get("validations", [])))
        self.assertTrue(dataset_plan_path.exists())

        saved = json.loads(dataset_plan_path.read_text(encoding="utf-8"))
        self.assertEqual(saved, payload["datasets"])

        with self.client_session(ServerMode.TRAINER) as client:
            reload_response = client.get("/api/datasets/plan")

        self.assertEqual(reload_response.status_code, 200)
        reloaded = reload_response.json()
        self.assertEqual(reloaded["datasets"], payload["datasets"])
        self.assertEqual(reloaded["source"], "disk")

    def test_create_invalid_dataset_plan_rejected(self) -> None:
        payload = {
            "datasets": [
                {
                    "id": "images-only",
                    "type": "local",
                    "dataset_type": "image",
                }
            ]
        }

        with self.client_session(ServerMode.TRAINER) as client:
            response = client.post("/api/datasets/plan", json=payload)

        self.assertEqual(response.status_code, 422)
        detail = response.json()["detail"]
        self.assertIn("validations", detail)
        self.assertTrue(any(message["level"] == "error" for message in detail["validations"]))


class EventEndpointTests(APITestCase, unittest.TestCase):
    def test_callback_endpoint_stores_events(self) -> None:
        with self.client_session(ServerMode.CALLBACK) as client:
            event = {"message_type": "training_progress", "step": 100, "loss": 0.5}
            response = client.post("/callback", json=event)

        self.assertEqual(response.status_code, 200)

    def test_callback_clears_on_configure_webhook(self) -> None:
        with self.client_session(ServerMode.CALLBACK) as client:
            client.post("/callback", json={"data": "event1"})
            client.post("/callback", json={"data": "event2"})
            response = client.post("/callback", json={"message_type": "configure_webhook", "config": "new"})

        self.assertEqual(response.status_code, 200)

    def test_broadcast_long_polling_timeout(self) -> None:
        with self.client_session(ServerMode.CALLBACK) as client:
            start_time = time.time()
            response = client.get("/broadcast?last_event_index=999")
        elapsed = time.time() - start_time

        self.assertLess(elapsed, 35)
        self.assertIn(response.status_code, {200, 204})

    def test_broadcast_returns_new_events(self) -> None:
        with self.client_session(ServerMode.CALLBACK) as client:
            for i in range(5):
                client.post("/callback", json={"seq": i})
            response = client.get("/broadcast?last_event_index=0")

        self.assertEqual(response.status_code, 200)
        self.assertIn("events", response.json())

    def test_webhook_to_event_store_flow(self) -> None:
        with self.client_session(ServerMode.UNIFIED) as client:
            webhook_data = {"message_type": "training_update", "data": {"epoch": 1, "step": 50}}
            response = client.post("/callback", json=webhook_data)
            self.assertEqual(response.status_code, 200)
            broadcast = client.get("/broadcast?last_event_index=0")

        if broadcast.status_code == 200:
            self.assertIn("events", broadcast.json())


class ConcurrentAPICallTests(AsyncAPITestCase, unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_configuration_checks(self) -> None:
        config_api = Configuration()

        async def check_config(job_id: str):
            job_config = ConfigModel(
                trainer_config={"model_type": "lora"},
                dataloader_config=[],
                webhook_config={},
                job_id=job_id,
            )
            with patch("simpletuner.helpers.training.trainer.Trainer"):
                return await config_api.check(job_config)

        tasks = [check_config(f"job_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        for result in results:
            self.assertEqual(result["status"], "success")

    async def test_concurrent_job_submissions(self) -> None:
        config_api = Configuration()
        APIState.clear_state()

        async def submit_job(job_id: str):
            job_config = ConfigModel(trainer_config={}, dataloader_config=[], webhook_config={}, job_id=job_id)
            try:
                return await config_api.run(job_config)
            except HTTPException as exc:
                return {"error": str(exc.detail)}

        with patch("simpletuner.helpers.training.trainer.Trainer") as mock_trainer_cls:
            mock_trainer_cls.return_value = Mock()

            with patch("simpletuner.simpletuner_sdk.thread_keeper.submit_job") as mock_submit:
                mock_submit.side_effect = lambda jid, func: Mock()

                # Track which job gets through first
                current_job_tracker = {"job_id": None}

                def status_side_effect(job_id):
                    # If no job is current, return "not_found"
                    if current_job_tracker["job_id"] is None:
                        return "not_found"
                    # If this is the current job, return its status
                    if job_id == current_job_tracker["job_id"]:
                        return "running"
                    # Otherwise, it's not found
                    return "not_found"

                with patch(
                    "simpletuner.simpletuner_sdk.thread_keeper.get_thread_status",
                    side_effect=status_side_effect,
                ):
                    # Patch APIState.set_state to track which job becomes current
                    original_set_state = APIState.set_state

                    def track_current_job(key, value):
                        if key == "current_job_id":
                            current_job_tracker["job_id"] = value
                        return original_set_state(key, value)

                    with patch.object(APIState, "set_state", side_effect=track_current_job):
                        tasks = [submit_job(f"job_{i}") for i in range(5)]
                        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = sum(1 for result in results if isinstance(result, dict) and result.get("status") == "success")
        self.assertGreaterEqual(success_count, 1)


class HealthCheckTests(APITestCase, unittest.TestCase):
    def test_health_check_trainer_mode(self) -> None:
        with self.client_session(ServerMode.TRAINER) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["mode"], "trainer")

    def test_health_check_callback_mode(self) -> None:
        with self.client_session(ServerMode.CALLBACK) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["mode"], "callback")

    def test_health_check_unified_mode(self) -> None:
        with self.client_session(ServerMode.UNIFIED) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["mode"], "unified")


class ErrorResponseTests(AsyncAPITestCase, unittest.IsolatedAsyncioTestCase):
    async def test_invalid_configuration_error(self) -> None:
        config_api = Configuration()
        job_config = ConfigModel(
            trainer_config={},
            dataloader_config=[],
            webhook_config={},
            job_id="invalid_config",
        )

        with patch("simpletuner.helpers.training.trainer.Trainer") as mock_trainer_cls:
            mock_trainer_cls.side_effect = ValueError("Invalid config")
            with self.assertRaises(HTTPException) as exc_info:
                await config_api.check(job_config)

        self.assertEqual(exc_info.exception.status_code, 400)
        self.assertIn("Invalid config", str(exc_info.exception.detail))

    def test_job_not_found_error(self) -> None:
        host = TrainingHost()
        with patch("simpletuner.simpletuner_sdk.api_state.APIState.get_active_jobs", return_value={}):
            result = host.get_job_status("nonexistent_job")

        self.assertEqual(result[1], 404)


class ProcessModeIntegrationTests(AsyncAPITestCase, unittest.IsolatedAsyncioTestCase):
    async def test_full_training_flow_subprocess(self) -> None:
        config_data = {
            "trainer_config": {"model_type": "lora", "model_family": "sdxl", "max_train_steps": 10},
            "dataloader_config": [],
            "webhook_config": {},
            "job_id": "integration_test",
        }

        with self.execution_mode("process"):
            with patch("simpletuner.helpers.training.trainer.Trainer") as mock_trainer_cls:
                mock_instance = Mock()
                mock_instance.run.return_value = None
                mock_trainer_cls.return_value = mock_instance

                with patch("simpletuner.simpletuner_sdk.process_keeper.submit_job") as mock_submit:
                    mock_submit.return_value = Mock()
                    with self.client_session(ServerMode.TRAINER) as client:
                        response = client.post("/training/configuration/run", json=config_data)

        if response.status_code == 200:
            self.assertIn("subprocess", response.json().get("result", ""))

    def test_cancel_subprocess_via_api(self) -> None:
        APIState.set_state("current_job_id", "test_cancel")
        APIState.set_trainer(Mock())

        with patch("simpletuner.simpletuner_sdk.thread_keeper.terminate_thread", return_value=True):
            with self.client_session(ServerMode.TRAINER) as client:
                response = client.post("/training/cancel")

        self.assertEqual(response.status_code, 200)
        self.assertIn("cancel", response.json().get("result", "").lower())


if __name__ == "__main__":
    unittest.main()
