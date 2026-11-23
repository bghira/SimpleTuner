import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from simpletuner.simpletuner_sdk.server import ServerMode
from simpletuner.simpletuner_sdk.server.services.cloud.base import JobType, UnifiedJob
from simpletuner.simpletuner_sdk.server.services.training_metrics_service import (
    TRAINING_METRICS_SERVICE,
    TrainingMetricsServiceError,
)
from simpletuner.simpletuner_sdk.server.services.worker_run_metrics import WorkerRunMetrics
from tests.unittest_support import APITestCase


class TrainingMetricsRoutesTests(APITestCase, unittest.TestCase):
    def test_training_run_routes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            report = root / "training_report.html"
            report.write_text("<html>report</html>", encoding="utf-8")
            media = root / "validation.webp"
            media.write_bytes(b"RIFF")
            run_payload = {
                "run": {"environment": "anima", "last_step": 2},
                "records": [{"step": 2, "metrics": {"loss": 0.5}}],
                "media": [],
                "available_metrics": ["loss"],
            }

            with (
                patch.object(TRAINING_METRICS_SERVICE, "list_runs", return_value={"runs": [], "count": 0}),
                patch.object(TRAINING_METRICS_SERVICE, "get_run", return_value=run_payload) as get_run,
                patch.object(TRAINING_METRICS_SERVICE, "report_path", return_value=report),
                patch.object(TRAINING_METRICS_SERVICE, "media_path", return_value=media),
                self.client_session(ServerMode.UNIFIED) as client,
            ):
                self.assertEqual(client.get("/api/metrics/training/runs").json(), {"runs": [], "count": 0})

                response = client.get(
                    "/api/metrics/training/runs/anima",
                    params={"start_step": 1, "end_step": 2, "max_points": 25, "metric": ["loss"]},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["run"]["environment"], "anima")
                get_run.assert_called_once_with(
                    "anima",
                    start_step=1,
                    end_step=2,
                    max_points=25,
                    metric_names=["loss"],
                )

                report_response = client.get("/api/metrics/training/runs/anima/report")
                self.assertEqual(report_response.status_code, 200)
                self.assertEqual(report_response.headers["content-type"], "text/html; charset=utf-8")
                self.assertNotIn("attachment", report_response.headers.get("content-disposition", ""))

                media_response = client.get("/api/metrics/training/runs/anima/media/validation_images/validation.webp")
                self.assertEqual(media_response.status_code, 200)
                self.assertEqual(media_response.content, b"RIFF")

    def test_jpeg_xl_media_has_explicit_content_type(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            validation_dir = output / "validation_images"
            validation_dir.mkdir()
            image = validation_dir / "validation.jxl"
            image.write_bytes(b"\xff\x0a")
            (output / "validation_media.jsonl").write_text(
                json.dumps({"path": "validation_images/validation.jxl"}) + "\n", encoding="utf-8"
            )

            with (
                patch.object(TRAINING_METRICS_SERVICE, "_environment", return_value=({}, output)),
                self.client_session(ServerMode.UNIFIED) as client,
            ):
                response = client.get("/api/metrics/training/runs/anima/media/validation_images/validation.jxl")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "image/jxl")
            self.assertEqual(response.content, b"\xff\x0a")

    def test_training_run_rejects_reversed_step_range(self):
        with self.client_session(ServerMode.UNIFIED) as client:
            response = client.get(
                "/api/metrics/training/runs/anima",
                params={"start_step": 10, "end_step": 1},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("start_step", response.json()["detail"])


class WorkerTrainingMetricsTests(unittest.IsolatedAsyncioTestCase):
    async def test_visible_job_series_is_sampled_and_scoped_to_dispatch(self):
        job = UnifiedJob(
            job_id="worker-run",
            job_type=JobType.LOCAL,
            provider="worker",
            status="running",
            config_name="shared",
            created_at=datetime.now(timezone.utc).isoformat(),
            started_at="2026-01-01T00:00:00+00:00",
            user_id=1,
            metadata={"worker_id": "worker"},
        )
        history = WorkerRunMetrics()
        store = AsyncMock()
        store.get.return_value = job
        store.list.return_value = [job]
        owner = Mock(id=1)
        owner.has_permission.return_value = False
        other_user = Mock(id=2)
        other_user.has_permission.return_value = False
        report = {"current": 1, "metrics": {"loss": 0.3, "learning_rate": 0.0001, "ignored": 1}}
        with (
            patch("simpletuner.simpletuner_sdk.server.services.worker_run_metrics.monotonic") as clock,
            patch("simpletuner.simpletuner_sdk.server.services.training_metrics_service.WORKER_RUN_METRICS", history),
            patch(
                "simpletuner.simpletuner_sdk.server.services.training_metrics_service.get_job_repository", return_value=store
            ),
        ):
            clock.return_value = 0
            history.record(job, "Worker", report)
            history.record(job, "Worker", {**report, "current": 3})
            clock.return_value = 6
            history.record(job, "Worker", {**report, "current": 2})
            result = await TRAINING_METRICS_SERVICE.get_worker_run(job.job_id, owner)
            self.assertEqual([row["step"] for row in result["records"]], [1, 2])
            self.assertEqual(result["available_metrics"], ["learning_rate", "loss"])
            self.assertEqual([run["job_id"] for run in await TRAINING_METRICS_SERVICE.list_worker_runs(owner)], [job.job_id])
            store.list.assert_awaited_once_with(job_ids=[job.job_id], user_id=owner.id)
            with self.assertRaises(TrainingMetricsServiceError) as error:
                await TRAINING_METRICS_SERVICE.get_worker_run(job.job_id, other_user)
            self.assertEqual(error.exception.status_code, 404)
            job.started_at = "2026-01-01T00:10:00+00:00"
            self.assertIsNone(history.snapshot(job))
            history.record(job, "Worker", report)
            self.assertEqual(len(history.snapshot(job)[1]), 1)
            clock.return_value = 3600
            self.assertEqual(history.job_ids(), [])


if __name__ == "__main__":
    unittest.main()
