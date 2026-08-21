from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from simpletuner.helpers.training.local_metrics import (
    MANIFEST_FILENAME,
    MEDIA_FILENAME,
    METRICS_FILENAME,
    REPORT_FILENAME,
    LocalMetricsTracker,
    downsample_records,
    is_local_metrics_enabled,
    read_media_records,
    read_metric_records,
)
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.validation_images import save_validation_image


class LocalMetricsTrackerTests(unittest.TestCase):
    def test_tracker_writes_scalars_manifest_and_self_contained_report(self):
        with tempfile.TemporaryDirectory() as directory:
            tracker = LocalMetricsTracker("anima-smoke", directory, "local-tests")
            tracker.store_init_configuration(
                {
                    "model_family": "anima",
                    "learning_rate": 1e-4,
                    "output_dir": "/private/output",
                }
            )
            tracker.log(
                {
                    "train_loss": torch.tensor(1.25),
                    "learning_rate": 1e-4,
                    "table": {"not": "scalar"},
                    "invalid": float("nan"),
                },
                step=4,
            )
            tracker.finish()

            output_dir = Path(directory)
            records = read_metric_records(output_dir)
            self.assertEqual(records[0]["step"], 4)
            self.assertEqual(records[0]["metrics"], {"learning_rate": 1e-4, "train_loss": 1.25})

            manifest = json.loads((output_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["record_count"], 1)
            self.assertNotIn("output_dir", manifest["config"])

            report = (output_dir / REPORT_FILENAME).read_text(encoding="utf-8")
            self.assertIn("anima-smoke", report)
            self.assertIn('"train_loss":1.25', report)
            self.assertNotIn("fetch(", report)
            self.assertTrue((output_dir / METRICS_FILENAME).is_file())

    def test_resume_appends_to_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            first = LocalMetricsTracker("resume", directory, "tests")
            first.store_init_configuration({"model_family": "anima"})
            first.log({"train_loss": 2.0}, step=1)

            resumed = LocalMetricsTracker("resume", directory, "tests")
            resumed.store_init_configuration({"model_family": "anima"})
            resumed.log({"train_loss": 1.0}, step=2)
            resumed.finish()

            records = read_metric_records(directory)
            self.assertEqual([record["step"] for record in records], [1, 2])
            manifest = json.loads((Path(directory) / MANIFEST_FILENAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["record_count"], 2)

    def test_downsampling_preserves_first_and_last_records(self):
        records = [{"step": index, "metrics": {"loss": float(index)}} for index in range(100)]

        sampled = downsample_records(records, max_points=10)

        self.assertEqual(len(sampled), 10)
        self.assertEqual(sampled[0]["step"], 0)
        self.assertEqual(sampled[-1]["step"], 99)

    def test_local_metrics_enablement_supports_explicit_and_all_modes(self):
        self.assertTrue(is_local_metrics_enabled("simpletuner"))
        self.assertTrue(is_local_metrics_enabled("all"))
        self.assertTrue(is_local_metrics_enabled(["wandb", "simpletuner"]))
        self.assertFalse(is_local_metrics_enabled("wandb"))


class ValidationMediaManifestTests(unittest.TestCase):
    def setUp(self):
        StateTracker.set_global_step(12)

    def test_webp_image_is_saved_and_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            validation_dir = output_dir / "validation_images"
            validation_dir.mkdir()
            config = SimpleNamespace(
                output_dir=str(output_dir),
                report_to="simpletuner",
                validation_image_format="webp",
                validation_image_quality=82,
            )

            saved_path = save_validation_image(
                Image.new("RGB", (16, 12), color="red"),
                validation_dir,
                "step_12_sample_0_16x12",
                config,
                label="sample",
                index=0,
                resolution="16x12",
            )

            self.assertTrue(saved_path.endswith(".webp"))
            self.assertTrue(Path(saved_path).is_file())
            media = read_media_records(output_dir)
            self.assertEqual(media[0]["path"], "validation_images/step_12_sample_0_16x12.webp")
            self.assertEqual(media[0]["step"], 12)
            self.assertEqual(media[0]["type"], "image")
            self.assertTrue((output_dir / MEDIA_FILENAME).is_file())

    def test_png_remains_default_and_external_trackers_do_not_create_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            validation_dir = output_dir / "validation_images"
            validation_dir.mkdir()
            config = SimpleNamespace(output_dir=str(output_dir), report_to="wandb")

            saved_path = save_validation_image(
                Image.new("RGBA", (8, 8), color=(0, 0, 0, 0)),
                validation_dir,
                "step_12_sample_0_8x8",
                config,
                label="sample",
                index=0,
                resolution="8x8",
            )

            self.assertTrue(saved_path.endswith(".png"))
            self.assertFalse((output_dir / MEDIA_FILENAME).exists())

    def test_invalid_validation_image_quality_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = SimpleNamespace(
                output_dir=directory,
                report_to="simpletuner",
                validation_image_format="webp",
                validation_image_quality=0,
            )

            with self.assertRaisesRegex(ValueError, "within"):
                save_validation_image(
                    Image.new("RGB", (8, 8)),
                    directory,
                    "invalid-quality",
                    config,
                    label="sample",
                    index=0,
                    resolution="8x8",
                )


if __name__ == "__main__":
    unittest.main()
