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


if __name__ == "__main__":
    unittest.main()
