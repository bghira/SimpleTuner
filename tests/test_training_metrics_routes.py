import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simpletuner.simpletuner_sdk.server import ServerMode
from simpletuner.simpletuner_sdk.server.services.training_metrics_service import TRAINING_METRICS_SERVICE
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

    def test_training_run_rejects_reversed_step_range(self):
        with self.client_session(ServerMode.UNIFIED) as client:
            response = client.get(
                "/api/metrics/training/runs/anima",
                params={"start_step": 10, "end_step": 1},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("start_step", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
