"""
Tests for dataset routes: /api/datasets/browse and /api/datasets/detect

Covers file browser functionality and existing dataset detection.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from simpletuner.simpletuner_sdk.server import ServerMode
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIDefaults, WebUIStateStore
from tests.unittest_support import APITestCase


class DatasetBrowseTestCase(APITestCase, unittest.TestCase):
    """Test /api/datasets/browse endpoint."""

    def setUp(self) -> None:
        super().setUp()
        self._home_tmpdir = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._home_tmpdir.name)

        # Patch HOME so the state store persists underneath the temp root
        self._home_patch = patch.dict(os.environ, {"HOME": str(self.temp_dir)}, clear=False)
        self._home_patch.start()

        self.state_store = WebUIStateStore()

        # Patch the route-level state store singleton
        self._store_patch = patch(
            "simpletuner.simpletuner_sdk.server.routes.datasets.WebUIStateStore",
            return_value=self.state_store,
        )
        self._store_patch.start()

        # Set up default datasets directory
        self.datasets_dir = self.temp_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        defaults = WebUIDefaults(
            datasets_dir=str(self.datasets_dir),
            allow_dataset_paths_outside_dir=False,
        )
        self.state_store.save_defaults(defaults)

        # Trainer app client
        self.client: TestClient = self.create_test_client(ServerMode.TRAINER)

    def tearDown(self) -> None:
        self.client.close()
        self._store_patch.stop()
        self._home_patch.stop()
        super().tearDown()
        self._home_tmpdir.cleanup()

    def test_browse_default_path(self) -> None:
        """Test browsing with no path parameter uses datasets_dir."""
        # Create a subdirectory
        subdir = self.datasets_dir / "test-dataset"
        subdir.mkdir()

        response = self.client.get("/api/datasets/browse")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["currentPath"], str(self.datasets_dir.resolve()))
        self.assertEqual(len(data["directories"]), 1)
        self.assertEqual(data["directories"][0]["name"], "test-dataset")
        self.assertEqual(data["directories"][0]["hasDataset"], False)

    def test_browse_specific_path(self) -> None:
        """Test browsing a specific path."""
        subdir = self.datasets_dir / "my-images"
        subdir.mkdir()

        response = self.client.get(f"/api/datasets/browse?path={str(subdir)}")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["currentPath"], str(subdir.resolve()))
        self.assertEqual(len(data["directories"]), 0)

    def test_browse_detects_existing_dataset(self) -> None:
        """Test that browse detects existing SimpleTuner datasets."""
        # Create a directory with aspect bucket metadata
        dataset_dir = self.datasets_dir / "existing-dataset"
        dataset_dir.mkdir()

        # Create aspect bucket metadata file
        bucket_file = dataset_dir / "aspect_ratio_bucket_indices_test-id.json"
        bucket_data = {
            "config": {"resolution": 512},
            "aspect_ratio_bucket_indices": {
                "1.0": ["image1.jpg", "image2.jpg"],
            },
        }
        bucket_file.write_text(json.dumps(bucket_data))

        # Create some image files
        (dataset_dir / "image1.jpg").touch()
        (dataset_dir / "image2.jpg").touch()

        response = self.client.get("/api/datasets/browse")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["directories"]), 1)
        self.assertEqual(data["directories"][0]["hasDataset"], True)
        self.assertEqual(data["directories"][0]["datasetId"], "test-id")
        self.assertEqual(data["directories"][0]["fileCount"], 2)

    def test_browse_outside_datasets_dir_denied(self) -> None:
        """Test that browsing outside datasets_dir is denied by default."""
        outside_dir = self.temp_dir / "outside"
        outside_dir.mkdir()

        response = self.client.get(f"/api/datasets/browse?path={str(outside_dir)}")

        self.assertEqual(response.status_code, 403)
        self.assertIn("outside configured datasets directory", response.json()["detail"])

    def test_browse_outside_datasets_dir_allowed(self) -> None:
        """Test that browsing outside datasets_dir works when allowed."""
        # Enable outside access
        defaults = WebUIDefaults(
            datasets_dir=str(self.datasets_dir),
            allow_dataset_paths_outside_dir=True,
        )
        self.state_store.save_defaults(defaults)

        outside_dir = self.temp_dir / "outside"
        outside_dir.mkdir()

        response = self.client.get(f"/api/datasets/browse?path={str(outside_dir)}")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["currentPath"], str(outside_dir.resolve()))

    def test_browse_nonexistent_path(self) -> None:
        """Test browsing a nonexistent path returns 404."""
        nonexistent = self.datasets_dir / "does-not-exist"

        response = self.client.get(f"/api/datasets/browse?path={str(nonexistent)}")

        self.assertEqual(response.status_code, 404)
        self.assertIn("does not exist", response.json()["detail"])

    def test_browse_file_path_returns_400(self) -> None:
        """Test browsing a file path returns 400."""
        file_path = self.datasets_dir / "test.txt"
        file_path.touch()

        response = self.client.get(f"/api/datasets/browse?path={str(file_path)}")

        self.assertEqual(response.status_code, 400)
        self.assertIn("not a directory", response.json()["detail"])

    def test_browse_can_go_up(self) -> None:
        """Test canGoUp flag when parent is within datasets_dir."""
        subdir = self.datasets_dir / "subdir"
        subdir.mkdir()

        response = self.client.get(f"/api/datasets/browse?path={str(subdir)}")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["canGoUp"])
        self.assertEqual(data["parentPath"], str(self.datasets_dir.resolve()))

    def test_browse_cannot_go_up_beyond_datasets_dir(self) -> None:
        """Test canGoUp is false when parent is outside datasets_dir."""
        response = self.client.get(f"/api/datasets/browse?path={str(self.datasets_dir)}")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        # Can technically go up to parent, but should be prevented by security
        # The implementation sets canGoUp to False if parent is outside datasets_dir
        self.assertFalse(data["canGoUp"])


class DatasetDetectTestCase(APITestCase, unittest.TestCase):
    """Test /api/datasets/detect endpoint."""

    def setUp(self) -> None:
        super().setUp()
        self._home_tmpdir = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._home_tmpdir.name)

        # Patch HOME
        self._home_patch = patch.dict(os.environ, {"HOME": str(self.temp_dir)}, clear=False)
        self._home_patch.start()

        self.state_store = WebUIStateStore()

        # Patch the route-level state store singleton
        self._store_patch = patch(
            "simpletuner.simpletuner_sdk.server.routes.datasets.WebUIStateStore",
            return_value=self.state_store,
        )
        self._store_patch.start()

        # Set up default datasets directory
        self.datasets_dir = self.temp_dir / "datasets"
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        defaults = WebUIDefaults(
            datasets_dir=str(self.datasets_dir),
            allow_dataset_paths_outside_dir=False,
        )
        self.state_store.save_defaults(defaults)

        # Trainer app client
        self.client: TestClient = self.create_test_client(ServerMode.TRAINER)

    def tearDown(self) -> None:
        self.client.close()
        self._store_patch.stop()
        self._home_patch.stop()
        super().tearDown()
        self._home_tmpdir.cleanup()

    def test_detect_existing_dataset(self) -> None:
        """Test detecting an existing SimpleTuner dataset."""
        dataset_dir = self.datasets_dir / "my-dataset"
        dataset_dir.mkdir()

        # Create aspect bucket metadata
        bucket_file = dataset_dir / "aspect_ratio_bucket_indices_my-id.json"
        bucket_data = {
            "config": {"resolution": 1024, "crop_style": "center"},
            "aspect_ratio_bucket_indices": {
                "1.0": ["img1.jpg", "img2.jpg"],
                "1.5": ["img3.jpg"],
            },
        }
        bucket_file.write_text(json.dumps(bucket_data))

        response = self.client.get(f"/api/datasets/detect?path={str(dataset_dir)}")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["hasDataset"])
        self.assertEqual(data["datasetId"], "my-id")
        self.assertEqual(data["path"], str(dataset_dir.resolve()))
        self.assertEqual(data["config"]["resolution"], 1024)
        self.assertEqual(data["config"]["crop_style"], "center")
        self.assertEqual(data["totalFiles"], 3)
        self.assertIn("1.0", data["aspectRatios"])
        self.assertIn("1.5", data["aspectRatios"])

    def test_detect_no_dataset(self) -> None:
        """Test detecting a directory without a dataset."""
        empty_dir = self.datasets_dir / "empty"
        empty_dir.mkdir()

        response = self.client.get(f"/api/datasets/detect?path={str(empty_dir)}")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["hasDataset"])
        self.assertEqual(data["path"], str(empty_dir.resolve()))

    def test_detect_outside_datasets_dir_denied(self) -> None:
        """Test that detecting outside datasets_dir is denied by default."""
        outside_dir = self.temp_dir / "outside"
        outside_dir.mkdir()

        response = self.client.get(f"/api/datasets/detect?path={str(outside_dir)}")

        self.assertEqual(response.status_code, 403)
        self.assertIn("outside configured datasets directory", response.json()["detail"])

    def test_detect_invalid_json(self) -> None:
        """Test detecting a dataset with invalid JSON metadata."""
        dataset_dir = self.datasets_dir / "broken-dataset"
        dataset_dir.mkdir()

        # Create invalid JSON file
        bucket_file = dataset_dir / "aspect_ratio_bucket_indices_broken.json"
        bucket_file.write_text("{ invalid json }")

        response = self.client.get(f"/api/datasets/detect?path={str(dataset_dir)}")

        self.assertEqual(response.status_code, 500)
        self.assertIn("Error parsing dataset metadata", response.json()["detail"])

    def test_detect_nonexistent_path(self) -> None:
        """Test detecting a nonexistent path returns 404."""
        nonexistent = self.datasets_dir / "does-not-exist"

        response = self.client.get(f"/api/datasets/detect?path={str(nonexistent)}")

        self.assertEqual(response.status_code, 404)
        self.assertIn("does not exist", response.json()["detail"])

    def test_detect_returns_filtering_statistics(self) -> None:
        """Test that detect endpoint returns filtering_statistics when present (issue #2474)."""
        dataset_dir = self.datasets_dir / "dataset-with-stats"
        dataset_dir.mkdir()

        # Create aspect bucket metadata with filtering_statistics
        bucket_file = dataset_dir / "aspect_ratio_bucket_indices_stats-test.json"
        bucket_data = {
            "config": {"resolution": 1024},
            "aspect_ratio_bucket_indices": {
                "1.0": ["img1.jpg", "img2.jpg"],
            },
            "filtering_statistics": {
                "total_processed": 10,
                "skipped": {
                    "already_exists": 0,
                    "metadata_missing": 0,
                    "not_found": 0,
                    "too_small": 4,
                    "too_long": 0,
                    "other": 0,
                },
            },
        }
        bucket_file.write_text(json.dumps(bucket_data))

        response = self.client.get(f"/api/datasets/detect?path={str(dataset_dir)}")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["hasDataset"])
        self.assertIn("filteringStatistics", data)
        self.assertEqual(data["filteringStatistics"]["total_processed"], 10)
        self.assertEqual(data["filteringStatistics"]["skipped"]["too_small"], 4)

    def test_detect_no_filtering_statistics_when_absent(self) -> None:
        """Test that detect endpoint omits filteringStatistics when not in cache."""
        dataset_dir = self.datasets_dir / "dataset-no-stats"
        dataset_dir.mkdir()

        # Create aspect bucket metadata without filtering_statistics
        bucket_file = dataset_dir / "aspect_ratio_bucket_indices_no-stats.json"
        bucket_data = {
            "config": {"resolution": 512},
            "aspect_ratio_bucket_indices": {
                "1.0": ["img1.jpg"],
            },
        }
        bucket_file.write_text(json.dumps(bucket_data))

        response = self.client.get(f"/api/datasets/detect?path={str(dataset_dir)}")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["hasDataset"])
        self.assertNotIn("filteringStatistics", data)


if __name__ == "__main__":
    unittest.main()
