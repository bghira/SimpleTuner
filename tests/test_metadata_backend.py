import unittest, json
from PIL import Image
from unittest.mock import Mock, patch, MagicMock
from helpers.metadata.backends.discovery import DiscoveryMetadataBackend
from helpers.training.state_tracker import StateTracker
from tests.helpers.data import MockDataBackend


class TestMetadataBackend(unittest.TestCase):
    def setUp(self):
        self.data_backend = MockDataBackend()
        self.data_backend.id = "foo"
        self.test_image = Image.new("RGB", (512, 256), color="red")
        self.accelerator = Mock()
        self.data_backend.exists = Mock(return_value=True)
        self.data_backend.write = Mock(return_value=True)
        self.data_backend.list_files = Mock(
            return_value=[("subdir", "", "image_path.png")]
        )
        self.data_backend.read = Mock(return_value=self.test_image.tobytes())
        # Mock image data to simulate reading from the backend
        self.image_path_str = "test_image.jpg"

        self.instance_data_dir = "/some/fake/path"
        self.cache_file = "/some/fake/cache"
        self.metadata_file = "/some/fake/metadata.json"
        StateTracker.set_args(MagicMock())
        # Overload cache file with json:
        with patch(
            "helpers.training.state_tracker.StateTracker._save_to_disk",
            return_value=True,
        ), patch("pathlib.Path.exists", return_value=True):
            with self.assertLogs("DiscoveryMetadataBackend", level="WARNING"):
                self.metadata_backend = DiscoveryMetadataBackend(
                    id="foo",
                    instance_data_dir=self.instance_data_dir,
                    cache_file=self.cache_file,
                    metadata_file=self.metadata_file,
                    batch_size=1,
                    data_backend=self.data_backend,
                    resolution=1,
                    resolution_type="area",
                    accelerator=self.accelerator,
                    repeats=0,
                )

    def test_len(self):
        self.metadata_backend.aspect_ratio_bucket_indices = {
            "1.0": ["image1", "image2"],
            "1.5": ["image3"],
        }
        self.assertEqual(len(self.metadata_backend), 3)

    def test_discover_new_files(self):
        # Assuming that StateTracker.get_image_files returns known files
        # and list_files should return both known and potentially new files
        with patch(
            "helpers.training.state_tracker.StateTracker.get_image_files",
            return_value=["image1.jpg", "image2.png", "image3.jpg", "image4.png"],
        ), patch(
            "helpers.training.state_tracker.StateTracker.set_image_files",
            return_value=None,
        ), patch.object(
            self.data_backend,
            "list_files",
            return_value=["image1.jpg", "image2.png", "image3.jpg", "image4.png"],
        ):

            self.metadata_backend.aspect_ratio_bucket_indices = {
                "1.0": ["image1.jpg", "image2.png"]
            }
            new_files = self.metadata_backend._discover_new_files(for_metadata=False)
            # Assuming the method's logic excludes files known (["image1.jpg", "image2.png"])
            # The expectation is that only ["image3.jpg", "image4.png"] are returned as new
            self.assertEqual(sorted(new_files), sorted(["image3.jpg", "image4.png"]))

    def test_load_cache_valid(self):
        valid_cache_data = {
            "aspect_ratio_bucket_indices": {"1.0": ["image1", "image2"]},
        }
        with patch.object(
            self.data_backend, "read", return_value=json.dumps(valid_cache_data)
        ):
            self.metadata_backend.reload_cache()
        self.assertEqual(
            self.metadata_backend.aspect_ratio_bucket_indices,
            {"1.0": ["image1", "image2"]},
        )

    def test_load_cache_invalid(self):
        invalid_cache_data = "this is not valid json"
        with patch.object(self.data_backend, "read", return_value=invalid_cache_data):
            with self.assertLogs("DiscoveryMetadataBackend", level="WARNING"):
                self.metadata_backend.reload_cache()

    def test_save_cache(self):
        self.metadata_backend.aspect_ratio_bucket_indices = {
            "1.0": ["image1", "image2"]
        }
        with patch.object(self.data_backend, "write") as mock_write:
            self.metadata_backend.save_cache()
        mock_write.assert_called_once()

    def test_minimum_aspect_size(self):
        # when metadata_backend.minimum_aspect_ratio is not None and > 0.0 it will remove buckets from the list.
        # this test ensures that the bucket is removed when the value is set correctly.
        self.metadata_backend.aspect_ratio_bucket_indices = {
            "1.0": ["image1", "image2"],
            "1.5": ["image3"],
        }
        self.metadata_backend.minimum_aspect_ratio = 1.25
        self.metadata_backend._enforce_min_aspect_ratio()
        self.assertEqual(
            self.metadata_backend.aspect_ratio_bucket_indices, {"1.5": ["image3"]}
        )

    def test_maximum_aspect_size(self):
        # when metadata_backend.maximum_aspect_ratio is not None and > 0.0 it will remove buckets from the list.
        # this test ensures that the bucket is removed when the value is set correctly.
        self.metadata_backend.aspect_ratio_bucket_indices = {
            "1.0": ["image1", "image2"],
            "1.5": ["image3"],
        }
        self.metadata_backend.maximum_aspect_ratio = 1.25
        self.metadata_backend._enforce_max_aspect_ratio()
        self.assertEqual(
            self.metadata_backend.aspect_ratio_bucket_indices, {"1.0": ["image1", "image2"]}
        )

    def test_unbound_aspect_list(self):
        # when metadata_backend.maximum_aspect_ratio is None and metadata_backend.minimum_aspect_ratio is None
        # the aspect_ratio_bucket_indices should not be modified.
        self.metadata_backend.aspect_ratio_bucket_indices = {
            "1.0": ["image1", "image2"],
            "1.5": ["image3"],
        }
        self.metadata_backend._enforce_min_aspect_ratio()
        self.metadata_backend._enforce_max_aspect_ratio()
        self.assertEqual(
            self.metadata_backend.aspect_ratio_bucket_indices,
            {"1.0": ["image1", "image2"], "1.5": ["image3"]},
        )


if __name__ == "__main__":
    unittest.main()
