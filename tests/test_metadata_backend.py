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

    def test_len(self):
        self.metadata_backend.aspect_ratio_bucket_indices = {
            "1.0": ["image1", "image2"],
            "1.5": ["image3"],
        }
        self.assertEqual(len(self.metadata_backend), 3)

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
        # JSON string keys are coerced back to floats when loading
        self.assertEqual(
            self.metadata_backend.aspect_ratio_bucket_indices,
            {1.0: ["image1", "image2"]},
        )

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


class TestSplitBucketsEvalDataset(unittest.TestCase):
    """Test that eval datasets use effective_batch_size=1 for bucket splitting (issue #2507)."""

    def test_eval_dataset_single_image_no_validation_error(self):
        """Eval dataset with 1 image and batch_size=4 should NOT raise ValueError."""
        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_eval_split"
        mock_backend.batch_size = 4
        mock_backend.repeats = 0
        mock_backend.bucket_report = None
        mock_backend.dataset_type = DatasetType.EVAL
        mock_backend.aspect_ratio_bucket_indices = {"1.0": ["eval_img1.jpg"]}
        mock_backend.read_only = False

        mock_accelerator = MagicMock()
        mock_accelerator.num_processes = 1
        mock_accelerator.process_index = 0
        mock_backend.accelerator = mock_accelerator

        with (
            patch.object(StateTracker, "get_args") as mock_get_args,
            patch.object(StateTracker, "get_data_backend_config", return_value={}),
            patch(
                "simpletuner.helpers.metadata.backends.base.get_cp_aware_dp_info",
                return_value=(1, 0, 1),
            ),
        ):
            mock_args = MagicMock()
            mock_args.allow_dataset_oversubscription = False
            mock_get_args.return_value = mock_args

            # Should NOT raise ValueError
            MetadataBackend.split_buckets_between_processes(mock_backend, gradient_accumulation_steps=4, apply_padding=False)

        # Bucket should still exist after splitting
        self.assertIn("1.0", mock_backend.aspect_ratio_bucket_indices)

    def test_training_dataset_single_image_raises_validation_error(self):
        """Training dataset with 1 image and batch_size=4 should raise ValueError."""
        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_training_split"
        mock_backend.batch_size = 4
        mock_backend.repeats = 0
        mock_backend.bucket_report = None
        mock_backend.dataset_type = DatasetType.IMAGE
        mock_backend.aspect_ratio_bucket_indices = {"1.0": ["train_img1.jpg"]}

        mock_accelerator = MagicMock()
        mock_accelerator.num_processes = 1
        mock_backend.accelerator = mock_accelerator

        with (
            patch.object(StateTracker, "get_args") as mock_get_args,
            patch.object(StateTracker, "get_data_backend_config", return_value={}),
            patch(
                "simpletuner.helpers.metadata.backends.base.get_cp_aware_dp_info",
                return_value=(1, 0, 1),
            ),
        ):
            mock_args = MagicMock()
            mock_args.allow_dataset_oversubscription = False
            mock_get_args.return_value = mock_args

            # Should raise ValueError for training dataset
            with self.assertRaises(ValueError) as context:
                MetadataBackend.split_buckets_between_processes(
                    mock_backend, gradient_accumulation_steps=1, apply_padding=False
                )

            self.assertIn("zero usable batches", str(context.exception))

    def test_eval_dataset_ignores_gradient_accumulation(self):
        """Eval dataset should use effective_batch_size=1 regardless of grad accum setting."""
        mock_backend = MagicMock(spec=MetadataBackend)
        mock_backend.id = "test_eval_grad_accum"
        mock_backend.batch_size = 2
        mock_backend.repeats = 0
        mock_backend.bucket_report = None
        mock_backend.dataset_type = DatasetType.EVAL
        mock_backend.aspect_ratio_bucket_indices = {"1.0": ["eval_img1.jpg"]}
        mock_backend.read_only = False

        mock_accelerator = MagicMock()
        mock_accelerator.num_processes = 1
        mock_accelerator.process_index = 0
        mock_backend.accelerator = mock_accelerator

        with (
            patch.object(StateTracker, "get_args") as mock_get_args,
            patch.object(StateTracker, "get_data_backend_config", return_value={}),
            patch(
                "simpletuner.helpers.metadata.backends.base.get_cp_aware_dp_info",
                return_value=(1, 0, 1),
            ),
        ):
            mock_args = MagicMock()
            mock_args.allow_dataset_oversubscription = False
            mock_get_args.return_value = mock_args

            # With grad_accum=8, training would need 16 samples (2*1*8)
            # But eval should still work with 1 sample
            MetadataBackend.split_buckets_between_processes(mock_backend, gradient_accumulation_steps=8, apply_padding=False)

        self.assertIn("1.0", mock_backend.aspect_ratio_bucket_indices)


class TestDistributedBucketPadding(unittest.TestCase):
    """Regression coverage for PR #2897 distributed bucket policies."""

    @staticmethod
    def _backend(images, *, num_processes=8):
        backend = MagicMock(spec=MetadataBackend)
        backend.id = "distributed-padding"
        backend.batch_size = 1
        backend.repeats = 0
        backend.bucket_report = None
        backend.dataset_type = DatasetType.IMAGE
        backend.aspect_ratio_bucket_indices = {"1.0": list(images)}
        backend.read_only = False
        backend.accelerator = MagicMock(
            num_processes=num_processes,
            process_index=0,
            is_main_process=True,
        )
        return backend

    def _split_rank(self, images, rank, *, cp_size, apply_padding, repeats=0):
        backend = self._backend(images)
        backend.repeats = repeats
        with (
            patch.dict("os.environ", {"SIMPLETUNER_SHUFFLE_BUCKETS": "0"}),
            patch.object(
                StateTracker,
                "get_args",
                return_value=SimpleNamespace(allow_dataset_oversubscription=apply_padding),
            ),
            patch.object(
                StateTracker,
                "get_data_backend_config",
                return_value={"repeats": repeats} if repeats else {},
            ),
            patch(
                "simpletuner.helpers.metadata.backends.base.get_cp_aware_dp_info",
                return_value=(8, rank, cp_size),
            ),
        ):
            MetadataBackend.split_buckets_between_processes(
                backend,
                gradient_accumulation_steps=1,
                apply_padding=apply_padding,
            )
        return backend.aspect_ratio_bucket_indices["1.0"]

    def test_cp_padding_fills_empty_dp_shards_from_final_global_item(self):
        images = ["img0", "img1", "img2", "img3"]
        shards = [self._split_rank(images, rank, cp_size=2, apply_padding=True, repeats=1) for rank in range(8)]

        self.assertEqual([len(shard) for shard in shards], [1] * 8)
        self.assertEqual([item for shard in shards for item in shard], images + ["img3"] * 4)

    def test_cp_no_padding_uses_balanced_qr_partition(self):
        images = [f"img{index}" for index in range(9)]
        shards = [self._split_rank(images, rank, cp_size=2, apply_padding=False) for rank in range(8)]

        self.assertEqual([len(shard) for shard in shards], [2, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual([item for shard in shards for item in shard], images)

    def test_standard_split_preserves_balanced_qr_order(self):
        images = [f"img{index}" for index in range(9)]
        shards = [self._split_rank(images, rank, cp_size=1, apply_padding=False) for rank in range(8)]

        self.assertEqual([len(shard) for shard in shards], [2, 1, 1, 1, 1, 1, 1, 1])
        self.assertEqual([item for shard in shards for item in shard], images)

    def test_standard_split_padding_exact_division_preserves_cardinality(self):
        images = [f"img{index}" for index in range(8)]
        shards = [self._split_rank(images, rank, cp_size=1, apply_padding=True) for rank in range(8)]

        self.assertEqual([len(shard) for shard in shards], [1] * 8)
        self.assertEqual([item for shard in shards for item in shard], images)

    def test_shuffled_split_is_independent_of_input_order(self):
        images = [f"img{index:02d}" for index in range(16)]
        shards = []
        for rank in range(8):
            rank_images = images if rank % 2 == 0 else list(reversed(images))
            backend = self._backend(rank_images)
            backend.accelerator.is_main_process = rank == 0
            with (
                patch.dict("os.environ", {"SIMPLETUNER_SHUFFLE_BUCKETS": "1"}),
                patch.object(
                    StateTracker,
                    "get_args",
                    return_value=SimpleNamespace(allow_dataset_oversubscription=False, seed=42),
                ),
                patch.object(StateTracker, "get_data_backend_config", return_value={}),
                patch(
                    "simpletuner.helpers.metadata.backends.base.get_cp_aware_dp_info",
                    return_value=(8, rank, 1),
                ),
            ):
                MetadataBackend.split_buckets_between_processes(backend)
            shards.append(backend.aspect_ratio_bucket_indices["1.0"])

        flattened = [item for shard in shards for item in shard]
        self.assertEqual(len(flattened), len(images))
        self.assertEqual(set(flattened), set(images))

    def test_padding_non_divisible_preserves_qr_boundaries(self):
        images = [f"img{index}" for index in range(10)]
        expected = [["img0", "img1"], ["img2", "img3"]] + [[f"img{index}", "img9"] for index in range(4, 10)]

        for cp_size in (1, 2):
            with self.subTest(cp_size=cp_size):
                shards = [self._split_rank(images, rank, cp_size=cp_size, apply_padding=True) for rank in range(8)]
                self.assertEqual(shards, expected)


class TestAutomaticOversubscriptionLogicalSequence(unittest.TestCase):
    """Automatic repeats are materialised into a rank-local cyclic schedule."""

    @staticmethod
    def _backend(images, dp_size, rank, cp_size=1, batch_size=1, repeats=0):
        backend = object.__new__(MetadataBackend)
        backend.id = "auto-oversubscription"
        backend.batch_size = batch_size
        backend.repeats = repeats
        backend.bucket_report = None
        backend.dataset_type = DatasetType.IMAGE
        backend.read_only = False
        backend._aspect_ratio_bucket_indices = {"1.0": list(images)}
        backend.accelerator = SimpleNamespace(
            num_processes=dp_size * cp_size,
            process_index=rank,
            is_main_process=True,
        )
        return backend

    def _split(self, backend, dp_size, rank, cp_size, gradient_accumulation_steps, config=None):
        with (
            patch.dict("os.environ", {"SIMPLETUNER_SHUFFLE_BUCKETS": "0"}),
            patch.object(
                StateTracker,
                "get_args",
                return_value=SimpleNamespace(allow_dataset_oversubscription=True, seed=0),
            ),
            patch.object(StateTracker, "get_data_backend_config", return_value=config or {}),
            patch(
                "simpletuner.helpers.metadata.backends.base.get_cp_aware_dp_info",
                return_value=(dp_size, rank, cp_size),
            ),
        ):
            MetadataBackend.split_buckets_between_processes(
                backend,
                gradient_accumulation_steps=gradient_accumulation_steps,
                apply_padding=True,
            )

    def test_standard_auto_repeat_provides_one_complete_ga_window_per_rank(self):
        shards = []
        for rank in range(8):
            backend = self._backend(["img0"], dp_size=8, rank=rank)
            self._split(backend, 8, rank, 1, gradient_accumulation_steps=4, config={"repeats": 0})
            shards.append(backend.aspect_ratio_bucket_indices["1.0"])
            self.assertEqual(backend.repeats, 0)

        self.assertEqual([len(shard) for shard in shards], [4] * 8)
        self.assertEqual(sum(map(len, shards)), 32)
        self.assertEqual({item for shard in shards for item in shard}, {"img0"})

    def test_context_parallel_auto_repeat_uses_effective_dp_and_cyclic_tail(self):
        images = [f"img{index}" for index in range(5)]
        shards = []
        for rank in range(4):
            backend = self._backend(images, dp_size=4, rank=rank, cp_size=2)
            self._split(backend, 4, rank, 2, gradient_accumulation_steps=2, config={"repeats": 0})
            shards.append(backend.aspect_ratio_bucket_indices["1.0"])

        self.assertEqual([len(shard) for shard in shards], [4] * 4)
        self.assertEqual(
            [item for shard in shards for item in shard],
            images + images + (images * 2)[:6],
        )

    def test_docs_style_auto_repeat_cardinality_is_padded_to_full_window(self):
        images = [f"img{index}" for index in range(25)]
        shards = []
        for rank in range(8):
            backend = self._backend(images, dp_size=8, rank=rank)
            self._split(backend, 8, rank, 1, gradient_accumulation_steps=4, config={"repeats": 0})
            shards.append(backend.aspect_ratio_bucket_indices["1.0"])

        self.assertEqual([len(shard) for shard in shards], [8] * 8)
        self.assertEqual(
            [item for shard in shards for item in shard],
            images + images + images[:14],
        )

    def test_auto_repeat_count_is_scoped_to_undersized_buckets(self):
        backend = self._backend([], dp_size=2, rank=0, batch_size=4)
        backend._aspect_ratio_bucket_indices = {
            "small": ["s0", "s1", "s2"],
            "large": ["l0", "l1", "l2", "l3", "l4", "l5", "l6"],
        }
        self._split(backend, 2, 0, 1, gradient_accumulation_steps=1, config={"repeats": 0})

        self.assertEqual(backend.repeats, 0)
        self.assertEqual(
            {bucket: len(values) for bucket, values in backend.aspect_ratio_bucket_indices.items()},
            {"small": 8, "large": 8},
        )
        self.assertEqual(
            backend.aspect_ratio_bucket_indices["small"],
            ["s0", "s1", "s2", "s0", "s1", "s2", "s0", "s1"],
        )
        self.assertEqual(
            backend.aspect_ratio_bucket_indices["large"],
            ["l0", "l1", "l2", "l3", "l4", "l5", "l6", "l0"],
        )

    def test_manual_repeats_are_not_materialised(self):
        backend = self._backend(["img0"], dp_size=8, rank=0, repeats=31)
        self._split(backend, 8, 0, 1, gradient_accumulation_steps=4, config={"repeats": 31})

        self.assertEqual(backend.aspect_ratio_bucket_indices["1.0"], ["img0"])
        self.assertEqual(backend.repeats, 31)

    def test_empty_buckets_are_ignored_without_division_by_zero(self):
        backend = self._backend([], dp_size=8, rank=0, batch_size=1)
        backend._aspect_ratio_bucket_indices = {"empty": [], "full": ["img0"]}
        self._split(backend, 8, 0, 1, gradient_accumulation_steps=1, config={"repeats": 0})
        self.assertEqual(backend.aspect_ratio_bucket_indices["empty"], [])
        self.assertEqual(len(backend.aspect_ratio_bucket_indices["full"]), 1)


class TestEmptyBucketRepeatValidation(unittest.TestCase):
    """An empty bucket key survives update_buckets_with_existing_files and reaches the split."""

    @staticmethod
    def _backend(buckets, *, repeats=0):
        backend = MagicMock(spec=MetadataBackend)
        backend.id = "empty-bucket"
        backend.batch_size = 1
        backend.repeats = repeats
        backend.bucket_report = None
        backend.dataset_type = DatasetType.IMAGE
        backend.aspect_ratio_bucket_indices = {key: list(value) for key, value in buckets.items()}
        backend.read_only = False
        backend.accelerator = MagicMock(num_processes=1, process_index=0, is_main_process=True)
        return backend

    @staticmethod
    def _split(backend, *, num_processes, allow_oversubscription, dp_rank=0):
        with (
            patch.dict("os.environ", {"SIMPLETUNER_SHUFFLE_BUCKETS": "0"}),
            patch.object(
                StateTracker,
                "get_args",
                return_value=SimpleNamespace(allow_dataset_oversubscription=allow_oversubscription),
            ),
            patch.object(StateTracker, "get_data_backend_config", return_value={}),
            patch(
                "simpletuner.helpers.metadata.backends.base.get_cp_aware_dp_info",
                return_value=(num_processes, dp_rank, 1),
            ),
        ):
            MetadataBackend.split_buckets_between_processes(
                backend,
                gradient_accumulation_steps=1,
                apply_padding=allow_oversubscription,
            )

    def test_empty_bucket_does_not_divide_by_zero(self):
        # The crash is not distribution-specific: effective_batch_size is at least 1, so an
        # empty bucket always joins buckets_that_will_fail regardless of world size.
        for num_processes in (1, 2, 8):
            for allow_oversubscription in (True, False):
                with self.subTest(num_processes=num_processes, allow_oversubscription=allow_oversubscription):
                    backend = self._backend({"1.0": ["a.jpg", "b.jpg"], "1.5": []})
                    try:
                        self._split(backend, num_processes=num_processes, allow_oversubscription=allow_oversubscription)
                    except ZeroDivisionError as error:
                        self.fail(f"empty bucket divided by zero: {error}")
                    except ValueError:
                        # The pre-existing "zero usable batches" guard is allowed to fire; it is
                        # asserted on its own below.
                        pass

    def test_oversubscription_ignores_the_empty_bucket_when_adjusting_repeats(self):
        # The auto repeat factor ceil(8 / 2) - 1 = 3 is driven by the two-image bucket alone.
        # Since #2918, repeats is left unmutated and the factor is materialised into the
        # rank-local shard instead: logical 2 * (3 + 1) = 8 samples, batch-aligned to the
        # effective batch size of 8, so every rank holds exactly 8 // 8 = 1 cyclic sample.
        shards = []
        for dp_rank in range(8):
            backend = self._backend({"1.0": ["a.jpg", "b.jpg"], "1.5": []})
            self._split(backend, num_processes=8, allow_oversubscription=True, dp_rank=dp_rank)
            self.assertEqual(backend.repeats, 0, "auto oversubscription must not mutate repeats")
            self.assertEqual(backend.aspect_ratio_bucket_indices["1.5"], [])
            shards.append(backend.aspect_ratio_bucket_indices["1.0"])

        self.assertEqual([len(shard) for shard in shards], [1] * 8)
        self.assertEqual([shard[0] for shard in shards], ["a.jpg", "b.jpg"] * 4)

    def test_empty_bucket_still_reports_zero_usable_batches_without_oversubscription(self):
        backend = self._backend({"1.0": ["a.jpg", "b.jpg"], "1.5": []})
        with self.assertRaises(ValueError) as raised:
            self._split(backend, num_processes=8, allow_oversubscription=False)
        self.assertIn("zero usable batches", str(raised.exception))

    def test_bucket_without_empty_keys_is_unaffected(self):
        backend = self._backend({"1.0": ["a.jpg", "b.jpg"]})
        self._split(backend, num_processes=8, allow_oversubscription=True)

        # Same materialised schedule as the empty-bucket case: repeats stays 0 and rank 0
        # holds the first of the eight batch-aligned cyclic samples.
        self.assertEqual(backend.repeats, 0, "auto oversubscription must not mutate repeats")
        self.assertEqual(backend.aspect_ratio_bucket_indices["1.0"], ["a.jpg"])

    def test_refresh_leaves_an_empty_bucket_key_behind(self):
        # update_buckets_with_existing_files assigns [] rather than dropping the key, which is
        # how the empty bucket reaches the split in the first place.
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
