"""A windowed dataset is indexed and encoded while training, so the WebUI's scan and cache jobs refuse it."""

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from simpletuner.simpletuner_sdk.server import ServerMode
from simpletuner.simpletuner_sdk.server.services.cache_job_service import CacheJobService
from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import DatasetScanService
from tests.unittest_support import APITestCase

WINDOWED = {"id": "windowed", "type": "webshart", "dataset_type": "image", "webshart": {"shuffle_window": 2}}
LOCAL = {"id": "local", "type": "local", "dataset_type": "image", "instance_data_dir": "images"}
ROUTES = "simpletuner.simpletuner_sdk.server.routes.dataset_viewer"


def _no_job_can_start(test):
    """Without the refusal, a request would be turned away for another reason instead of starting a job."""
    for patcher in (
        patch("simpletuner.simpletuner_sdk.server.services.cache_job_service.threading"),
        patch("simpletuner.simpletuner_sdk.server.services.dataset_scan_service.threading"),
        patch.object(CacheJobService, "_check_training_running", return_value=True),
    ):
        patcher.start()
        test.addCleanup(patcher.stop)


class TestServicesRefuseAWindowedDataset(unittest.TestCase):
    def setUp(self):
        CacheJobService._active_job = None
        CacheJobService._thread = None
        _no_job_can_start(self)

    def test_cache_job_is_refused_before_it_starts(self):
        service = CacheJobService(broadcast_fn=MagicMock())
        for cache_type in ("vae", "text_embeds"):
            with self.subTest(cache_type=cache_type), self.assertRaisesRegex(ValueError, "encoded on demand"):
                service.start_cache_job("windowed", cache_type, WINDOWED, {})
        self.assertIsNone(CacheJobService._active_job)

    def test_scan_is_refused_before_it_starts(self):
        service = DatasetScanService(broadcast_fn=MagicMock())
        with self.assertRaisesRegex(ValueError, "nothing to scan or cache"):
            service.scan_dataset("windowed", WINDOWED, {})
        self.assertIsNone(service.get_active_status())


class TestRoutesWithAWindowedDataset(APITestCase, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.client: TestClient = self.create_test_client(ServerMode.TRAINER)
        _no_job_can_start(self)

    def tearDown(self) -> None:
        self.client.close()
        super().tearDown()

    def test_scan_and_cache_requests_answer_with_the_reason(self):
        requests = (
            ("/api/datasets/scan", {"dataset_id": "windowed"}),
            ("/api/datasets/cache/start", {"dataset_id": "windowed", "cache_type": "vae"}),
        )
        with (
            patch(f"{ROUTES}._get_dataset_config_by_id", return_value=WINDOWED),
            patch(f"{ROUTES}._get_global_config", return_value={}),
        ):
            for path, body in requests:
                with self.subTest(path=path):
                    response = self.client.post(path, json=body)
                    self.assertEqual(response.status_code, 400)
                    self.assertIn("webshart.shuffle_window", response.json()["detail"])

    def test_scanning_every_dataset_leaves_the_windowed_one_out(self):
        scan_service = MagicMock()
        scan_service.scan_all.return_value = "queue"
        with (
            patch(f"{ROUTES}._load_plan", return_value=[WINDOWED, LOCAL]),
            patch(f"{ROUTES}._get_global_config", return_value={}),
            patch(f"{ROUTES}.get_scan_service", return_value=scan_service),
        ):
            response = self.client.post("/api/datasets/scan/all", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["dataset_count"], 1)
        self.assertEqual([ds["id"] for ds in scan_service.scan_all.call_args.args[0]], ["local"])


if __name__ == "__main__":
    unittest.main()
