import unittest
from unittest.mock import patch

from simpletuner.simpletuner_sdk.server.services.dataset_scan_service import DatasetScanService


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
