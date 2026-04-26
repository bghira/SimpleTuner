from __future__ import annotations

import importlib.machinery
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from simpletuner.simpletuner_sdk.server import ServerMode
from simpletuner.simpletuner_sdk.server.services.captionflow_job_service import CaptionFlowJobResult
from tests.unittest_support import APITestCase


class DatasetCaptioningCapabilitiesTestCase(APITestCase, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.client: TestClient = self.create_test_client(ServerMode.TRAINER)

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def test_capabilities_reports_missing_captionflow(self) -> None:
        with patch(
            "simpletuner.simpletuner_sdk.server.routes.dataset_viewer.importlib.util.find_spec",
            return_value=None,
        ):
            response = self.client.get("/api/datasets/captioning/capabilities")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["installed"])
        self.assertFalse(data["ready"])
        self.assertIn("simpletuner[", data["install_command"])
        self.assertIn("captioning", data["install_command"])
        self.assertEqual(data["required_version"], "0.5.0")

    def test_capabilities_reports_installed_captionflow(self) -> None:
        fake_spec = importlib.machinery.ModuleSpec("caption_flow", loader=None)
        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.dataset_viewer.importlib.util.find_spec",
                return_value=fake_spec,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.routes.dataset_viewer.importlib.metadata.version",
                return_value="0.5.0",
            ),
        ):
            response = self.client.get("/api/datasets/captioning/capabilities")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["installed"])
        self.assertTrue(data["ready"])
        self.assertEqual(data["version"], "0.5.0")

    def test_capabilities_reports_old_captionflow_not_ready(self) -> None:
        fake_spec = importlib.machinery.ModuleSpec("caption_flow", loader=None)
        with (
            patch(
                "simpletuner.simpletuner_sdk.server.routes.dataset_viewer.importlib.util.find_spec",
                return_value=fake_spec,
            ),
            patch(
                "simpletuner.simpletuner_sdk.server.routes.dataset_viewer.importlib.metadata.version",
                return_value="0.4.2",
            ),
        ):
            response = self.client.get("/api/datasets/captioning/capabilities")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["installed"])
        self.assertFalse(data["ready"])
        self.assertEqual(data["version"], "0.4.2")

    def test_start_captioning_job_requires_captionflow(self) -> None:
        with patch(
            "simpletuner.simpletuner_sdk.server.routes.dataset_viewer.importlib.util.find_spec",
            return_value=None,
        ):
            response = self.client.post(
                "/api/datasets/captioning/jobs",
                json={"dataset_id": "images"},
            )

        self.assertEqual(response.status_code, 409)
        self.assertIn("CaptionFlow", response.json()["detail"])

    def test_start_captioning_job_submits_to_service(self) -> None:
        fake_spec = importlib.machinery.ModuleSpec("caption_flow", loader=None)
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "images"
            dataset_dir.mkdir()
            with (
                patch(
                    "simpletuner.simpletuner_sdk.server.routes.dataset_viewer.importlib.util.find_spec",
                    return_value=fake_spec,
                ),
                patch(
                    "simpletuner.simpletuner_sdk.server.routes.dataset_viewer.importlib.metadata.version",
                    return_value="0.5.0",
                ),
                patch(
                    "simpletuner.simpletuner_sdk.server.routes.dataset_viewer._require_dataset_config",
                    return_value={"id": "images", "dataset_type": "image", "instance_data_dir": str(dataset_dir)},
                ),
                patch(
                    "simpletuner.simpletuner_sdk.server.routes.dataset_viewer._get_global_config",
                    return_value={"--output_dir": str(Path(tmpdir) / "output")},
                ),
                patch(
                    "simpletuner.simpletuner_sdk.server.services.captionflow_job_service.start_captionflow_job",
                    return_value=CaptionFlowJobResult(
                        job_id="abc12345",
                        status="running",
                        allocated_gpus=[0],
                    ),
                ) as start_job,
            ):
                response = self.client.post(
                    "/api/datasets/captioning/jobs",
                    json={
                        "dataset_id": "images",
                        "worker_count": 1,
                        "batch_size": 2,
                        "any_gpu": True,
                    },
                )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["job_id"], "abc12345")
        self.assertEqual(data["status"], "running")
        self.assertEqual(data["allocated_gpus"], [0])
        start_job.assert_called_once()
