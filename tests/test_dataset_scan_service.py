import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import DatasetScanService


class TestClearConditioningCache(unittest.TestCase):
    """Verify clear_conditioning_cache only removes auto-generated dirs."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.service = DatasetScanService()

        # Set up a cache root with conditioning_data dirs
        self.cache_root = Path(self.tmpdir) / "cache"
        self.cond_root = self.cache_root / "conditioning_data"

        # Auto-generated conditioning dir (should be cleared)
        self.auto_dir = self.cond_root / "myds_conditioning_canny"
        self.auto_dir.mkdir(parents=True)
        (self.auto_dir / "img001.png").write_bytes(b"fake")

        # Manually declared conditioning dir (should NOT be cleared)
        self.manual_dir = self.cond_root / "my_manual_cond"
        self.manual_dir.mkdir(parents=True)
        (self.manual_dir / "img001.png").write_bytes(b"fake")

        # VAE dir so _derive works
        self.vae_dir = self.cache_root / "vae" / "sdxl" / "myds"
        self.vae_dir.mkdir(parents=True)

    def _dataset_config(self):
        return {
            "id": "myds",
            "type": "local",
            "instance_data_dir": str(self.tmpdir),
            "cache_dir_vae": str(self.vae_dir),
            "conditioning": [
                {"type": "canny", "conditioning_type": "controlnet"},
            ],
            "conditioning_data": ["my_manual_cond"],
        }

    @patch.object(DatasetScanService, "_resolve_scan_output_dir", side_effect=lambda x: x)
    def test_clears_auto_generated_only(self, _mock):
        cleared = self.service.clear_conditioning_cache(self._dataset_config(), {})
        self.assertEqual(len(cleared), 1)
        self.assertIn("myds_conditioning_canny", cleared[0])
        self.assertFalse(self.auto_dir.exists())
        self.assertTrue(self.manual_dir.exists())

    @patch.object(DatasetScanService, "_resolve_scan_output_dir", side_effect=lambda x: x)
    def test_no_generators_returns_empty(self, _mock):
        config = self._dataset_config()
        del config["conditioning"]
        cleared = self.service.clear_conditioning_cache(config, {})
        self.assertEqual(cleared, [])

    @patch.object(DatasetScanService, "_resolve_scan_output_dir", side_effect=lambda x: x)
    def test_nonexistent_dir_skipped(self, _mock):
        # Remove the auto-generated dir so it doesn't exist
        import shutil

        shutil.rmtree(self.auto_dir)
        cleared = self.service.clear_conditioning_cache(self._dataset_config(), {})
        self.assertEqual(cleared, [])


class TestDatasetScanServiceOutputDir(unittest.TestCase):
    def test_relative_scan_output_dir_is_resolved_under_webui_output_root(self):
        with patch.object(
            DatasetScanService,
            "_get_resolved_webui_output_dir",
            return_value="/srv/simpletuner/output",
        ):
            resolved = DatasetScanService._resolve_scan_output_dir("output/scan")

        self.assertEqual(resolved, "/srv/simpletuner/output/scan")

    def test_absolute_scan_output_dir_is_used_as_is(self):
        with patch.object(
            DatasetScanService,
            "_get_resolved_webui_output_dir",
            return_value="/srv/simpletuner/output",
        ):
            resolved = DatasetScanService._resolve_scan_output_dir("/var/tmp/simpletuner-scan")

        self.assertEqual(resolved, "/var/tmp/simpletuner-scan")


class TestScanUsesTrainingSettings(unittest.TestCase):
    """A scan records geometry under the recipe training would use, and keeps its store where training would."""

    def setUp(self):
        from PIL import Image

        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.images = Path(temp.name) / "images"
        self.images.mkdir()
        for index in range(2):
            Image.new("RGB", (256 + 64 * index, 192)).save(self.images / f"{index}.png")
            (self.images / f"{index}.txt").write_text("caption")
        self.output_dir = str(Path(temp.name) / "output")
        patcher = patch.object(DatasetScanService, "_resolve_scan_output_dir", return_value=self.output_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        from simpletuner.helpers.training.state_tracker import StateTracker

        StateTracker.all_image_files.pop("images", None)
        self.addCleanup(StateTracker.all_image_files.pop, "images", None)

    def scan(self, global_config, compute=None):
        from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import ScanJob

        dataset = {
            "id": "images",
            "type": "local",
            "instance_data_dir": str(self.images),
            "resolution": 128,
            "resolution_type": "pixel_area",
            "crop": True,
            "crop_aspect": "random",
            "crop_aspect_buckets": [1.0, 1.5],
            "caption_strategy": "textfile",
        }
        DatasetScanService()._execute_scan(ScanJob(job_id="job", dataset_id="images"), dataset, global_config, False)

    def recipe(self):
        import json

        return json.loads((self.images / "aspect_ratio_bucket_indices_images.json").read_text())["transform_recipe"]

    def test_a_dataset_that_recrops_each_epoch_is_recorded_for_the_first_epoch(self):
        self.scan({})
        self.assertEqual(self.recipe(), "discovery-epoch-1")

    def test_disabling_the_rebuild_records_one_recipe_for_every_epoch(self):
        self.scan({"aspect_bucket_disable_rebuild": True})
        self.assertEqual(self.recipe(), "discovery")

    def test_the_store_directory_follows_the_training_cache_dir(self):
        from simpletuner.helpers.metadata.backends.discovery import DiscoveryMetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        seen = []

        def compute(backend, **kwargs):
            seen.append(StateTracker.get_args().cache_dir)

        with patch.object(
            DiscoveryMetadataBackend, "compute_aspect_ratio_bucket_indices", autospec=True, side_effect=compute
        ):
            self.scan({})
            self.scan({"cache_dir": "/srv/cache"})
        self.assertEqual(seen, [os.path.join(self.output_dir, "cache"), "/srv/cache"])


if __name__ == "__main__":
    unittest.main()
