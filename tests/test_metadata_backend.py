import json
import tempfile
import unittest
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

try:
    from tests.helpers.data import MockDataBackend
except ModuleNotFoundError:
    from helpers.data import MockDataBackend
from PIL import Image

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.data_backend.filters import build_dataset_filter
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.metadata.backends.discovery import DiscoveryMetadataBackend
from simpletuner.helpers.metadata.backends.huggingface import HuggingfaceMetadataBackend
from simpletuner.helpers.metadata.backends.parquet import ParquetMetadataBackend
from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
from simpletuner.helpers.training.state_tracker import StateTracker


class TestMetadataBackend(unittest.TestCase):
    def setUp(self):
        self.data_backend = MockDataBackend()
        self.data_backend.id = "foo"
        self.test_image = Image.new("RGB", (512, 256), color="red")
        self.accelerator = Mock()
        self.data_backend.exists = Mock(return_value=True)
        self.data_backend.write = Mock(return_value=True)
        self.data_backend.list_files = Mock(return_value=[("subdir", "", "image_path.png")])
        self.data_backend.read = Mock(return_value=self.test_image.tobytes())
        # Mock image data to simulate reading from the backend
        self.image_path_str = "test_image.jpg"

        self.instance_data_dir = "/some/fake/path"
        self.cache_file = "/some/fake/cache"
        self.metadata_file = "/some/fake/metadata.json"
        StateTracker.set_args(MagicMock())
        # Overload cache file with json:
        with (
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker._save_to_disk",
                return_value=True,
            ),
            patch("pathlib.Path.exists", return_value=True),
        ):
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

    def test_scan_for_metadata_records_new_files(self):
        """A metadata-only scan stores what its workers produce."""
        new_files = [f"image_{index}.png" for index in range(20)]

        def process(file, buckets, metadata_updates=None, **kwargs):
            metadata_updates[str(file)] = {"original_size": (512, 256)}
            return buckets

        StateTracker.set_args(MagicMock(enable_multiprocessing=False))
        self.metadata_backend.image_metadata = {}
        with (
            patch.object(self.metadata_backend, "load_image_metadata"),
            patch.object(self.metadata_backend, "_discover_new_files", return_value=new_files),
            patch.object(self.metadata_backend, "_process_for_bucket", side_effect=process),
            patch.object(self.metadata_backend, "save_image_metadata") as save_image_metadata,
            patch.object(self.metadata_backend, "save_cache"),
            patch.object(self.metadata_backend, "record_transforms") as record_transforms,
        ):
            self.metadata_backend.scan_for_metadata()

        self.assertEqual(sorted(self.metadata_backend.image_metadata), sorted(new_files))
        recorded = [path for call in record_transforms.call_args_list for path in call.args[0]]
        self.assertEqual(sorted(recorded), sorted(new_files))
        save_image_metadata.assert_called_once()

    def test_removed_images_leave_every_bucket_list(self):
        self.metadata_backend.aspect_ratio_bucket_indices = {"1.0": ["a.png", "b.png"], "2.0": ["c.png", "a.png"]}
        self.metadata_backend.remove_images(["a.png", "c.png", "unlisted.png"])
        self.assertEqual(self.metadata_backend.aspect_ratio_bucket_indices, {"1.0": ["b.png"], "2.0": []})

    def test_probe_image_dimensions_applies_exif_orientation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "oriented.jpg"
            exif = Image.Exif()
            exif[0x0112] = 6
            Image.new("RGB", (64, 32)).save(image_path, exif=exif)

            metadata = self.metadata_backend._probe_image_dimensions(str(image_path))

        self.assertEqual(metadata, {"original_size": (32, 64)})

    def test_local_image_metadata_scan_skips_backend_read(self):
        self.data_backend.type = "local"
        self.metadata_backend.dataset_type = DatasetType.IMAGE
        self.metadata_backend.dataset_config = {"dataset_type": "image", "crop": False}
        self.data_backend.read.reset_mock()
        prepared = SimpleNamespace(
            crop_coordinates=(0, 0),
            target_size=(512, 256),
            intermediary_size=(512, 256),
            aspect_ratio=2.0,
        )

        with (
            patch.object(
                self.metadata_backend,
                "_probe_image_dimensions",
                return_value={"original_size": (512, 256)},
            ),
            patch("simpletuner.helpers.metadata.backends.discovery.TrainingSample") as training_sample,
        ):
            training_sample.return_value.prepare.return_value = prepared
            metadata_updates = {}
            buckets = self.metadata_backend._process_for_bucket(
                "image.png",
                {},
                metadata_updates=metadata_updates,
            )

        self.data_backend.read.assert_not_called()
        self.assertEqual(buckets, {"2.0": ["image.png"]})
        self.assertEqual(metadata_updates["image.png"]["original_size"], (512, 256))

    def test_face_crop_forces_full_image_decode(self):
        self.data_backend.type = "local"
        self.metadata_backend.dataset_type = DatasetType.IMAGE
        self.metadata_backend.dataset_config = {
            "dataset_type": "image",
            "crop": True,
            "crop_style": "face",
        }
        self.data_backend.read = Mock(return_value=b"image payload")
        prepared = SimpleNamespace(
            crop_coordinates=(0, 0),
            target_size=(512, 256),
            intermediary_size=(512, 256),
            aspect_ratio=2.0,
        )

        with (
            patch.object(self.metadata_backend, "_probe_image_dimensions") as probe_dimensions,
            patch("simpletuner.helpers.metadata.backends.discovery.load_image", return_value=self.test_image),
            patch("simpletuner.helpers.metadata.backends.discovery.TrainingSample") as training_sample,
        ):
            training_sample.return_value.prepare.return_value = prepared
            self.metadata_backend._process_for_bucket("image.png", {})

        probe_dimensions.assert_not_called()
        self.data_backend.read.assert_called_once_with("image.png")

    def test_discover_new_files(self):
        # Assuming that StateTracker.get_image_files returns known files
        # and list_files should return both known and potentially new files
        with (
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker.get_image_files",
                return_value=["image1.jpg", "image2.png", "image3.jpg", "image4.png"],
            ),
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker.set_image_files",
                return_value=None,
            ),
            patch.object(
                self.data_backend,
                "list_files",
                return_value=["image1.jpg", "image2.png", "image3.jpg", "image4.png"],
            ),
        ):

            self.metadata_backend.aspect_ratio_bucket_indices = {"1.0": ["image1.jpg", "image2.png"]}
            new_files = self.metadata_backend._discover_new_files(for_metadata=False)
            # Assuming the method's logic excludes files known (["image1.jpg", "image2.png"])
            # The expectation is that only ["image3.jpg", "image4.png"] are returned as new
            self.assertEqual(sorted(new_files), sorted(["image3.jpg", "image4.png"]))

    def test_discover_new_files_applies_path_filter(self):
        self.metadata_backend.dataset_filter = build_dataset_filter(
            {"filter_func": {"path": {"include": ["keep"], "exclude": ["blocked"]}}}
        )
        with (
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker.get_image_files",
                return_value=["keep/image1.jpg", "drop/image2.png", "keep/blocked.png", "keep/image3.jpg"],
            ),
            patch.object(
                self.data_backend,
                "list_files",
                return_value=["keep/image1.jpg", "drop/image2.png", "keep/blocked.png", "keep/image3.jpg"],
            ),
        ):
            self.metadata_backend.aspect_ratio_bucket_indices = {}

            new_files = self.metadata_backend._discover_new_files(for_metadata=False)

        self.assertEqual(sorted(new_files), sorted(["keep/image1.jpg", "keep/image3.jpg"]))

    def test_load_cache_valid(self):
        valid_cache_data = {
            "aspect_ratio_bucket_indices": {"1.0": ["image1", "image2"]},
        }
        with patch.object(self.data_backend, "read", return_value=json.dumps(valid_cache_data)):
            self.metadata_backend.reload_cache()
        self.assertEqual(
            self.metadata_backend.aspect_ratio_bucket_indices,
            {"1.0": ["image1", "image2"]},
        )

    def test_cache_loaders_preserve_string_bucket_keys(self):
        indices = {
            "1.0": ["square.jpg", "square.jpg"],
            "1.5": ["wide.jpg"],
            "1920x1080@125": ["video.mp4"],
            "3s": ["audio.wav"],
            "captions": ["caption.txt"],
        }
        for backend_class in (
            DiscoveryMetadataBackend,
            HuggingfaceMetadataBackend,
            ParquetMetadataBackend,
            WebshartMetadataBackend,
        ):
            with self.subTest(backend=backend_class.__name__):
                backend = backend_class.__new__(backend_class)
                backend.id = "cache-keys"
                backend.cache_file = "buckets.json"
                backend.data_backend = self.data_backend
                backend.load_image_metadata = Mock()
                backend._sync_image_files_with_buckets = Mock()
                with patch.object(
                    self.data_backend, "read", return_value=json.dumps({"aspect_ratio_bucket_indices": indices})
                ):
                    backend.reload_cache(set_config=False)
                self.assertEqual(backend.aspect_ratio_bucket_indices, indices)

    def test_discovery_after_cache_reload_keeps_one_bucket(self):
        cache = {"aspect_ratio_bucket_indices": {"1.0": ["cached.jpg", "cached.jpg"]}}
        with patch.object(self.data_backend, "read", return_value=json.dumps(cache)):
            self.metadata_backend.reload_cache()

        self.data_backend.type = "local"
        self.metadata_backend.dataset_type = DatasetType.IMAGE
        self.metadata_backend.dataset_config = {"dataset_type": "image", "crop": False}
        prepared = SimpleNamespace(
            crop_coordinates=(0, 0),
            target_size=(512, 512),
            intermediary_size=(512, 512),
            aspect_ratio=1.0,
        )
        with (
            patch.object(self.metadata_backend, "_probe_image_dimensions", return_value={"original_size": (1024, 1024)}),
            patch("simpletuner.helpers.metadata.backends.discovery.TrainingSample") as training_sample,
        ):
            training_sample.return_value.prepare.return_value = prepared
            self.metadata_backend._process_for_bucket("new.jpg", self.metadata_backend.aspect_ratio_bucket_indices)

        self.assertEqual(
            self.metadata_backend.aspect_ratio_bucket_indices,
            {"1.0": ["cached.jpg", "cached.jpg", "new.jpg"]},
        )

    def test_bucket_assignment_merges_key_types_without_changing_occurrences(self):
        indices = {
            1.0: ["first.jpg", "repeat.jpg"],
            "1.0": ["repeat.jpg", "last.jpg"],
            "1920x1080@125": ["video.mp4"],
            "3s": ["audio.wav"],
            "captions": ["caption.txt"],
        }
        self.metadata_backend.aspect_ratio_bucket_indices = indices

        expected = {
            "1.0": ["first.jpg", "repeat.jpg", "repeat.jpg", "last.jpg"],
            "1920x1080@125": ["video.mp4"],
            "3s": ["audio.wav"],
            "captions": ["caption.txt"],
        }
        self.assertEqual(self.metadata_backend.aspect_ratio_bucket_indices, expected)
        self.metadata_backend.aspect_ratio_bucket_indices = self.metadata_backend.aspect_ratio_bucket_indices
        self.assertEqual(self.metadata_backend.aspect_ratio_bucket_indices, expected)
        self.assertEqual(indices[1.0], ["first.jpg", "repeat.jpg"])

    def test_load_cache_invalid(self):
        invalid_cache_data = "this is not valid json"
        # Need to ensure exists returns True so it tries to read the file
        self.data_backend.exists = Mock(return_value=True)

        with patch.object(self.data_backend, "read", return_value=invalid_cache_data):
            # Patch the logger.warning method directly to verify it's called
            with patch("simpletuner.helpers.metadata.backends.discovery.logger.warning") as mock_warning:
                self.metadata_backend.reload_cache()
                # Verify warning was called with the expected message
                mock_warning.assert_called_once()
                warning_msg = mock_warning.call_args[0][0]
                self.assertIn("Error loading aspect bucket cache", warning_msg)
            # Should have empty indices due to invalid JSON
            self.assertEqual(self.metadata_backend.aspect_ratio_bucket_indices, {})

    def test_save_cache(self):
        self.metadata_backend.aspect_ratio_bucket_indices = {"1.0": ["image1", "image2"]}
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
        self.assertEqual(self.metadata_backend.aspect_ratio_bucket_indices, {"1.5": ["image3"]})

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
            self.metadata_backend.aspect_ratio_bucket_indices,
            {"1.0": ["image1", "image2"]},
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


class TestMaxNumSamplesLimit(unittest.TestCase):
    """Test the max_num_samples limit feature for deterministic dataset limiting (issue #2469)."""

    def test_max_num_samples_limit_applied(self):
        """max_num_samples should limit the file list to the specified count."""
        from simpletuner.helpers.metadata.backends.base import MetadataBackend

        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_limit"
        mock_backend.max_num_samples = 3

        file_list = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"]
        result = MetadataBackend._apply_max_num_samples_limit(mock_backend, file_list)

        self.assertEqual(len(result), 3)
        # All results should be from the original list
        for item in result:
            self.assertIn(item, file_list)

    def test_max_num_samples_no_limit(self):
        """When max_num_samples is None, the full list should be returned."""
        from simpletuner.helpers.metadata.backends.base import MetadataBackend

        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_no_limit"
        mock_backend.max_num_samples = None

        file_list = ["img1.jpg", "img2.jpg", "img3.jpg"]
        result = MetadataBackend._apply_max_num_samples_limit(mock_backend, file_list)

        self.assertEqual(result, file_list)

    def test_max_num_samples_larger_than_list(self):
        """When max_num_samples > len(file_list), return all files."""
        from simpletuner.helpers.metadata.backends.base import MetadataBackend

        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_larger_limit"
        mock_backend.max_num_samples = 10

        file_list = ["img1.jpg", "img2.jpg", "img3.jpg"]
        result = MetadataBackend._apply_max_num_samples_limit(mock_backend, file_list)

        self.assertEqual(result, file_list)

    def test_max_num_samples_deterministic(self):
        """Same dataset ID should produce same selection across multiple calls."""
        from simpletuner.helpers.metadata.backends.base import MetadataBackend

        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "deterministic_test"
        mock_backend.max_num_samples = 3

        file_list = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg", "g.jpg"]

        # Call multiple times, should get same result
        result1 = MetadataBackend._apply_max_num_samples_limit(mock_backend, file_list)
        result2 = MetadataBackend._apply_max_num_samples_limit(mock_backend, file_list)
        result3 = MetadataBackend._apply_max_num_samples_limit(mock_backend, file_list)

        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)

    def test_max_num_samples_is_independent_of_discovery_order(self):
        """Equivalent listings should produce the same bounded subset regardless of order."""
        from simpletuner.helpers.metadata.backends.base import MetadataBackend

        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "ordered_sample_limit"
        mock_backend.max_num_samples = 3
        file_list = [f"image_{index}.jpg" for index in range(10)]

        forward = MetadataBackend._apply_max_num_samples_limit(mock_backend, file_list)
        reversed_order = MetadataBackend._apply_max_num_samples_limit(mock_backend, list(reversed(file_list)))

        self.assertEqual(forward, reversed_order)

    def test_max_num_samples_different_ids_different_selection(self):
        """Different dataset IDs should produce different selections."""
        from simpletuner.helpers.metadata.backends.base import MetadataBackend

        file_list = ["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg", "f.jpg", "g.jpg"]

        mock_backend1 = MagicMock(spec=MetadataBackend)
        mock_backend1.id = "dataset_alpha"
        mock_backend1.max_num_samples = 3

        mock_backend2 = MagicMock(spec=MetadataBackend)
        mock_backend2.id = "dataset_beta"
        mock_backend2.max_num_samples = 3

        result1 = MetadataBackend._apply_max_num_samples_limit(mock_backend1, file_list)
        result2 = MetadataBackend._apply_max_num_samples_limit(mock_backend2, file_list)

        # With different seeds, selections should differ (statistically almost certain)
        # Use sets to compare regardless of order
        self.assertNotEqual(set(result1), set(result2))


class TestPruneSmallBucketsEvalDataset(unittest.TestCase):
    """Test that eval datasets use batch_size=1 for bucket pruning (issue #2475)."""

    def test_eval_dataset_single_image_not_pruned(self):
        """Eval dataset with 1 image and batch_size=4 should NOT be pruned."""
        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_eval"
        mock_backend.batch_size = 4
        mock_backend.repeats = 0
        mock_backend.bucket_report = None
        mock_backend.dataset_type = DatasetType.EVAL
        mock_backend.aspect_ratio_bucket_indices = {"1.0": ["eval_img1.jpg"]}

        with patch.object(StateTracker, "get_args") as mock_get_args:
            mock_args = MagicMock()
            mock_args.disable_bucket_pruning = False
            mock_get_args.return_value = mock_args

            MetadataBackend._prune_small_buckets(mock_backend, "1.0")

        self.assertIn("1.0", mock_backend.aspect_ratio_bucket_indices)
        self.assertEqual(mock_backend.aspect_ratio_bucket_indices["1.0"], ["eval_img1.jpg"])

    def test_training_dataset_single_image_pruned(self):
        """Training dataset with 1 image and batch_size=4 should be pruned."""
        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_training"
        mock_backend.batch_size = 4
        mock_backend.repeats = 0
        mock_backend.bucket_report = None
        mock_backend.dataset_type = DatasetType.IMAGE
        mock_backend.aspect_ratio_bucket_indices = {"1.0": ["train_img1.jpg"]}

        with patch.object(StateTracker, "get_args") as mock_get_args:
            mock_args = MagicMock()
            mock_args.disable_bucket_pruning = False
            mock_get_args.return_value = mock_args

            MetadataBackend._prune_small_buckets(mock_backend, "1.0")

        self.assertNotIn("1.0", mock_backend.aspect_ratio_bucket_indices)

    def test_eval_dataset_no_dataset_type_attribute(self):
        """Backend without dataset_type attr defaults to IMAGE behavior."""
        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_no_type"
        mock_backend.batch_size = 4
        mock_backend.repeats = 0
        mock_backend.bucket_report = None
        del mock_backend.dataset_type
        mock_backend.aspect_ratio_bucket_indices = {"1.0": ["img1.jpg"]}

        with patch.object(StateTracker, "get_args") as mock_get_args:
            mock_args = MagicMock()
            mock_args.disable_bucket_pruning = False
            mock_get_args.return_value = mock_args

            MetadataBackend._prune_small_buckets(mock_backend, "1.0")

        self.assertNotIn("1.0", mock_backend.aspect_ratio_bucket_indices)

    def test_eval_dataset_with_repeats(self):
        """Eval dataset respects repeats but still uses batch_size=1."""
        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_eval_repeats"
        mock_backend.batch_size = 4
        mock_backend.repeats = 2
        mock_backend.bucket_report = None
        mock_backend.dataset_type = DatasetType.EVAL
        mock_backend.aspect_ratio_bucket_indices = {"1.0": ["eval_img1.jpg"]}

        with patch.object(StateTracker, "get_args") as mock_get_args:
            mock_args = MagicMock()
            mock_args.disable_bucket_pruning = False
            mock_get_args.return_value = mock_args

            MetadataBackend._prune_small_buckets(mock_backend, "1.0")

        self.assertIn("1.0", mock_backend.aspect_ratio_bucket_indices)


class TestEmptyBucketKeys(unittest.TestCase):
    def test_refresh_leaves_an_empty_bucket_key_behind(self):
        # update_buckets_with_existing_files assigns [] rather than dropping the key, which is
        # how an empty bucket reaches the dataset cursor.
        backend = MagicMock()
        backend.aspect_ratio_bucket_indices = {"1.0": ["a.jpg"], "1.5": ["gone.jpg"]}
        backend.bucket_report = None
        MetadataBackend.update_buckets_with_existing_files(backend, {"a.jpg"})

        self.assertEqual(backend.aspect_ratio_bucket_indices, {"1.0": ["a.jpg"], "1.5": []})


class TestFilteringStatistics(unittest.TestCase):
    """Test filtering_statistics storage and retrieval in metadata backends (issue #2474)."""

    def setUp(self):
        self.data_backend = MockDataBackend()
        self.data_backend.id = "filtering_test"
        self.accelerator = Mock()
        self.data_backend.exists = Mock(return_value=False)
        self.data_backend.write = Mock(return_value=True)
        self.data_backend.list_files = Mock(return_value=[])
        StateTracker.set_args(MagicMock())

        with (
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker._save_to_disk",
                return_value=True,
            ),
            patch("pathlib.Path.exists", return_value=False),
        ):
            self.metadata_backend = DiscoveryMetadataBackend(
                id="filtering_test",
                instance_data_dir="/some/fake/path",
                cache_file="/some/fake/cache",
                metadata_file="/some/fake/metadata.json",
                batch_size=1,
                data_backend=self.data_backend,
                resolution=1,
                resolution_type="area",
                accelerator=self.accelerator,
                repeats=0,
            )

    def test_filtering_statistics_initialized_to_none(self):
        """filtering_statistics should be initialized to None."""
        self.assertIsNone(self.metadata_backend.filtering_statistics)

    def test_filtered_file_queue_defers_delete_until_drained(self):
        filtered_files_queue = Queue()
        self.data_backend.delete = Mock(return_value=True)

        self.metadata_backend._queue_or_delete_filtered_file(
            filepath="too-small.png",
            reason="too_small",
            delete_from_backend=True,
            filtered_files_queue=filtered_files_queue,
        )

        self.data_backend.delete.assert_not_called()
        self.assertEqual(self.metadata_backend._drain_filtered_files_queue(filtered_files_queue), 1)
        self.data_backend.delete.assert_called_once_with("too-small.png")

    def test_filtering_statistics_saved_in_cache(self):
        """filtering_statistics should be included when save_cache is called."""
        self.metadata_backend.aspect_ratio_bucket_indices = {"1.0": ["image1", "image2"]}
        self.metadata_backend.filtering_statistics = {
            "total_processed": 10,
            "skipped": {
                "already_exists": 0,
                "metadata_missing": 0,
                "not_found": 0,
                "too_small": 3,
                "too_long": 0,
                "other": 0,
            },
        }

        with patch.object(self.data_backend, "write") as mock_write:
            self.metadata_backend.save_cache()

        mock_write.assert_called_once()
        call_args = mock_write.call_args
        written_data = json.loads(call_args[0][1])

        self.assertIn("filtering_statistics", written_data)
        self.assertEqual(written_data["filtering_statistics"]["total_processed"], 10)
        self.assertEqual(written_data["filtering_statistics"]["skipped"]["too_small"], 3)

    def test_filtering_statistics_not_saved_when_none(self):
        """filtering_statistics should not be included in cache when None."""
        self.metadata_backend.aspect_ratio_bucket_indices = {"1.0": ["image1"]}
        self.metadata_backend.filtering_statistics = None

        with patch.object(self.data_backend, "write") as mock_write:
            self.metadata_backend.save_cache()

        mock_write.assert_called_once()
        call_args = mock_write.call_args
        written_data = json.loads(call_args[0][1])

        self.assertNotIn("filtering_statistics", written_data)

    def test_filtering_statistics_loaded_from_cache(self):
        """filtering_statistics should be loaded from cache when present."""
        cache_data = {
            "aspect_ratio_bucket_indices": {"1.0": ["image1"]},
            "filtering_statistics": {
                "total_processed": 5,
                "skipped": {
                    "already_exists": 2,
                    "metadata_missing": 0,
                    "not_found": 0,
                    "too_small": 1,
                    "too_long": 0,
                    "other": 0,
                },
            },
        }
        self.data_backend.exists = Mock(return_value=True)

        with patch.object(self.data_backend, "read", return_value=json.dumps(cache_data)):
            self.metadata_backend.reload_cache()

        self.assertIsNotNone(self.metadata_backend.filtering_statistics)
        self.assertEqual(self.metadata_backend.filtering_statistics["total_processed"], 5)
        self.assertEqual(self.metadata_backend.filtering_statistics["skipped"]["too_small"], 1)

    def test_filtering_statistics_none_when_not_in_cache(self):
        """filtering_statistics should remain None if not present in cache."""
        cache_data = {
            "aspect_ratio_bucket_indices": {"1.0": ["image1"]},
        }
        self.data_backend.exists = Mock(return_value=True)

        with patch.object(self.data_backend, "read", return_value=json.dumps(cache_data)):
            self.metadata_backend.reload_cache()

        self.assertIsNone(self.metadata_backend.filtering_statistics)


class TestStateTrackerDiskCache(unittest.TestCase):
    def test_set_image_files_creates_missing_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "output" / "scan"
            StateTracker.set_args(SimpleNamespace(output_dir=str(output_dir)))
            StateTracker.all_image_files = {}

            result = StateTracker.set_image_files(
                [("subdir", [], ["image_a.png", "image_b.png"])],
                data_backend_id="subject-1024",
            )

            self.assertEqual(result, {"image_a.png": False, "image_b.png": False})
            self.assertTrue((output_dir / "all_image_files_subject-1024.json").exists())


if __name__ == "__main__":
    unittest.main()


class TestParquetCaptionCells(unittest.TestCase):
    """Pandas reads a null caption cell as NaN; it is no caption, not the text "nan"."""

    def captions(self, frame, **config):
        stub = SimpleNamespace(
            parquet_database=frame,
            parquet_config={"filename_column": "filename", **config},
            dataset_type=DatasetType.IMAGE,
        )
        return ParquetMetadataBackend._extract_captions_to_fast_list(stub)

    def test_a_null_caption_takes_the_fallback_column_in_parquet_and_json_lines(self):
        import io

        import pandas as pd

        rows = [{"filename": "a", "caption": "cat", "tags": "ta"}, {"filename": "b", "caption": None, "tags": "tb"}]
        parquet = io.BytesIO()
        pd.DataFrame(rows).to_parquet(parquet)
        parquet.seek(0)
        json_lines = io.StringIO("\n".join(json.dumps(row) for row in rows))
        for name, frame in (
            ("parquet", pd.read_parquet(parquet, engine="pyarrow")),
            ("json_lines", pd.read_json(json_lines, lines=True)),
        ):
            with self.subTest(source=name):
                self.assertEqual(
                    self.captions(frame, caption_column="caption", fallback_caption_column="tags"), {"a": "cat", "b": "tb"}
                )
                self.assertEqual(self.captions(frame, caption_column=["caption", "tags"]), {"a": ["cat", "ta"], "b": ["tb"]})
                self.assertEqual(self.captions(frame, caption_column="caption"), {"a": "cat"})

    def test_null_entries_of_a_list_caption_are_dropped(self):
        import io

        import pandas as pd

        parquet = io.BytesIO()
        pd.DataFrame([{"filename": "a", "caption": ["cat", None]}]).to_parquet(parquet)
        parquet.seek(0)
        frame = pd.read_parquet(parquet, engine="pyarrow")
        self.assertEqual(self.captions(frame, caption_column="caption"), {"a": ["cat"]})
