import json
import tempfile
import unittest
from pathlib import Path

from simpletuner.helpers.training.local_metrics import (
    MANIFEST_FILENAME,
    MEDIA_FILENAME,
    METRICS_FILENAME,
    REPORT_FILENAME,
    TIMESTEP_DISTRIBUTION_FILENAME,
)
from simpletuner.simpletuner_sdk.server.services.training_metrics_service import (
    TrainingMetricsService,
    TrainingMetricsServiceError,
)


class _ConfigStore:
    def __init__(self, configs):
        self.configs = configs

    def list_configs(self):
        return [{"name": name} for name in self.configs]

    def load_config(self, name):
        if name not in self.configs:
            raise FileNotFoundError(name)
        return self.configs[name], object()


class TrainingMetricsServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.output_dir = self.root / "output"
        self.output_dir.mkdir()
        self.store = _ConfigStore(
            {
                "anima": {
                    "--output_dir": str(self.output_dir),
                    "--model_family": "anima",
                },
                "untracked": {"--output_dir": str(self.root / "untracked")},
            }
        )
        self.service = TrainingMetricsService()
        self.service._get_config_store = lambda: self.store

        self._write_json(
            MANIFEST_FILENAME,
            {
                "run_name": "anima-local",
                "project_name": "local-metrics",
                "status": "completed",
                "last_step": 9,
                "record_count": 10,
                "metric_names": ["loss", "lr"],
                "updated_at": "2026-08-21T12:00:00+00:00",
            },
        )
        self._write_jsonl(
            METRICS_FILENAME,
            [{"step": step, "metrics": {"loss": 10.0 - step, "lr": step / 1000, "epoch": step / 10}} for step in range(10)],
        )
        self._write_jsonl(TIMESTEP_DISTRIBUTION_FILENAME, [{"step": 9, "timesteps": [10.0, 20.0]}])
        validation_dir = self.output_dir / "validation_images"
        validation_dir.mkdir()
        (validation_dir / "step_9_prompt_0.webp").write_bytes(b"RIFF")
        self._write_jsonl(
            MEDIA_FILENAME,
            [
                {
                    "step": 9,
                    "type": "image",
                    "label": "prompt",
                    "index": 0,
                    "path": "validation_images/step_9_prompt_0.webp",
                }
            ],
        )
        (self.output_dir / REPORT_FILENAME).write_text("<html>report</html>", encoding="utf-8")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _write_json(self, filename, payload):
        (self.output_dir / filename).write_text(json.dumps(payload), encoding="utf-8")

    def _write_jsonl(self, filename, records):
        content = "".join(f"{json.dumps(record)}\n" for record in records)
        (self.output_dir / filename).write_text(content, encoding="utf-8")

    def test_list_runs_only_returns_environments_with_metrics(self):
        result = self.service.list_runs()

        self.assertEqual(result["count"], 1)
        self.assertEqual(result["runs"][0]["environment"], "anima")
        self.assertEqual(result["runs"][0]["model_family"], "anima")
        self.assertTrue(result["runs"][0]["has_report"])

    def test_get_run_filters_metrics_steps_and_downsamples(self):
        result = self.service.get_run(
            "anima",
            start_step=2,
            end_step=8,
            max_points=3,
            metric_names=["loss"],
        )

        self.assertEqual([record["step"] for record in result["records"]], [2, 5, 8])
        self.assertEqual([set(record["metrics"]) for record in result["records"]], [{"loss"}] * 3)
        self.assertEqual(result["available_metrics"], ["loss", "lr"])
        self.assertEqual(result["timesteps"], [{"step": 9, "timesteps": [10.0, 20.0]}])
        self.assertEqual(result["raw_files"]["timesteps"], TIMESTEP_DISTRIBUTION_FILENAME)
        self.assertEqual(len(result["media"]), 1)
        self.assertIn("/api/metrics/training/runs/anima/media/validation_images/", result["media"][0]["url"])

    def test_media_path_requires_manifest_entry_and_validation_directory(self):
        indexed = self.service.media_path("anima", "validation_images/step_9_prompt_0.webp")
        self.assertEqual(indexed.name, "step_9_prompt_0.webp")

        unindexed = self.output_dir / "validation_images" / "unindexed.webp"
        unindexed.write_bytes(b"RIFF")
        with self.assertRaises(TrainingMetricsServiceError) as context:
            self.service.media_path("anima", "validation_images/unindexed.webp")
        self.assertEqual(context.exception.status_code, 404)

        outside = self.root / "outside.webp"
        outside.write_bytes(b"RIFF")
        with self.assertRaises(TrainingMetricsServiceError):
            self.service.media_path("anima", "../outside.webp")

    def test_report_path_and_missing_run(self):
        self.assertEqual(self.service.report_path("anima").name, REPORT_FILENAME)
        with self.assertRaises(TrainingMetricsServiceError) as context:
            self.service.get_run("missing")
        self.assertEqual(context.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
