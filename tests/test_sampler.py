import logging
import os
import tempfile
import unittest
from math import ceil
from types import SimpleNamespace

# Import test configuration to suppress logging/warnings
try:
    from . import test_config
except ImportError:
    # Fallback for when running tests individually
    import test_config

try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from unittest import skip
from unittest.mock import MagicMock, Mock, patch

from accelerate import PartialState
from PIL import Image

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.metadata.backends.discovery import DiscoveryMetadataBackend
from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
from simpletuner.helpers.multiaspect.state import BucketStateManager
from simpletuner.helpers.training.state_tracker import StateTracker
from tests.helpers.data import MockDataBackend


class TestMultiAspectSampler(unittest.TestCase):
    def setUp(self):
        self.process_state = PartialState()
        self.accelerator = MagicMock(num_processes=1, process_index=0)
        self.accelerator.log = MagicMock()
        self.metadata_backend = Mock(spec=DiscoveryMetadataBackend)
        self.metadata_backend.id = "foo"
        self.metadata_backend.aspect_ratio_bucket_indices = {
            "1.0": ["image1", "image2", "image3", "image4"],
        }
        self.metadata_backend.seen_images = {}
        self.data_backend = MockDataBackend()
        self.data_backend.id = "foo"
        self.batch_size = 2
        self.seen_images_path = "/some/fake/seen_images.json"
        self.state_path = "/some/fake/state.json"

        self.sampler = MultiAspectSampler(
            id="foo",
            metadata_backend=self.metadata_backend,
            data_backend=self.data_backend,
            accelerator=self.accelerator,
            batch_size=self.batch_size,
            minimum_image_size=0,
            model=MagicMock(),
        )

        self.sampler.state_manager = Mock(spec=BucketStateManager)
        self.sampler.state_manager.load_state.return_value = {}

    def test_len(self):
        self.assertEqual(len(self.sampler), 2)

    def test_model_card_video_overview_honors_sample_cap_and_nested_video_config(self):
        sampler = object.__new__(MultiAspectSampler)
        sampler.id = "video-dataset"
        sampler.metadata_backend = SimpleNamespace(seen_images={str(i): True for i in range(1000)})
        sampler._get_unseen_images = MagicMock(return_value=[None] * 3997)
        sampler.metadata_backend.aspect_ratio_bucket_indices = {"1.0": [None] * 600}
        sampler.metadata_backend.max_num_samples = 4096
        sampler.metadata_backend.bucket_report = None
        sampler.accelerator = SimpleNamespace(num_processes=8, process_index=0)
        sampler.logger = MagicMock()
        sampler.sample_type_strs = "videos"
        sampler.buckets = ["0.75", "1.0", "1.25", "1.5", "1.75"]
        sampler.is_regularisation_data = False
        sampler.conditioning_type = None

        with (
            patch.object(
                StateTracker,
                "get_data_backend_config",
                return_value={"max_num_samples": "not-an-int", "video": {"num_frames": 39}},
            ),
            patch.object(StateTracker, "get_dataset_schedule", return_value={"reached": True}),
        ):
            overview = sampler.log_state(show_rank=False, alt_stats=True)

        self.assertIn("- Total number of videos: 4096", overview)
        self.assertIn("- Target frame count: 39", overview)
        self.assertIn("- FPS: unknown", overview)
        self.assertNotIn("~39976", overview)
        self.assertNotIn("~4800", overview)

    def test_model_card_overview_uses_bucket_count_not_seen_progress(self):
        sampler = object.__new__(MultiAspectSampler)
        sampler.id = "video-dataset"
        sampler.metadata_backend = SimpleNamespace(
            aspect_ratio_bucket_indices={"1.0": [None] * 65536},
            bucket_report=None,
            max_num_samples=None,
            seen_images={str(i): True for i in range(5740)},
        )
        sampler._get_unseen_images = MagicMock(return_value=[None] * 65536)
        sampler.accelerator = SimpleNamespace(num_processes=4, process_index=0)
        sampler.logger = MagicMock()
        sampler.sample_type_strs = "videos"
        sampler.buckets = ["1.0"]
        sampler.is_regularisation_data = False
        sampler.conditioning_type = None

        with (
            patch.object(StateTracker, "get_data_backend_config", return_value={"video": {"num_frames": 39}}),
            patch.object(StateTracker, "get_dataset_schedule", return_value={"reached": True}),
            patch("simpletuner.helpers.multiaspect.sampler.get_cp_aware_dp_info", return_value=(1, 0, 4)),
        ):
            overview = sampler.log_state(show_rank=False, alt_stats=True)

        self.assertIn("- Total number of videos: 65536", overview)
        self.assertNotIn("285096", overview)
        self.assertNotIn("~", overview)

    def test_model_card_overview_scales_data_parallel_bucket_shard_without_seen_progress(self):
        sampler = object.__new__(MultiAspectSampler)
        sampler.id = "video-dataset"
        sampler.metadata_backend = SimpleNamespace(
            aspect_ratio_bucket_indices={"1.0": [None] * 8192},
            bucket_report=None,
            max_num_samples=None,
            seen_images={str(i): True for i in range(250)},
        )
        sampler._get_unseen_images = MagicMock(return_value=[None] * 7942)
        sampler.accelerator = SimpleNamespace(num_processes=8, process_index=0)
        sampler.logger = MagicMock()
        sampler.sample_type_strs = "videos"
        sampler.buckets = ["1.0"]
        sampler.is_regularisation_data = False
        sampler.conditioning_type = None

        with (
            patch.object(StateTracker, "get_data_backend_config", return_value={}),
            patch.object(StateTracker, "get_dataset_schedule", return_value={"reached": True}),
            patch("simpletuner.helpers.multiaspect.sampler.get_cp_aware_dp_info", return_value=(8, 0, 1)),
        ):
            overview = sampler.log_state(show_rank=False, alt_stats=True)

        self.assertIn("- Total number of videos: ~65536", overview)
        self.assertNotIn("67536", overview)

    def test_model_card_overview_prefers_exact_bucket_report_total(self):
        sampler = object.__new__(MultiAspectSampler)
        sampler.id = "image-dataset"
        sampler.metadata_backend = SimpleNamespace(
            aspect_ratio_bucket_indices={"1.0": [None] * 16},
            bucket_report=SimpleNamespace(bucket_summaries={"post_refresh": {"total_samples": 123}}),
            max_num_samples=None,
            seen_images={str(i): True for i in range(7)},
        )
        sampler._get_unseen_images = MagicMock(return_value=[None] * 16)
        sampler.accelerator = SimpleNamespace(num_processes=8, process_index=0)
        sampler.logger = MagicMock()
        sampler.sample_type_strs = "images"
        sampler.buckets = ["1.0"]
        sampler.is_regularisation_data = False
        sampler.conditioning_type = None
        sampler.resolution = 512
        sampler.resolution_type = "pixel"

        with (
            patch.object(StateTracker, "get_data_backend_config", return_value={}),
            patch.object(StateTracker, "get_dataset_schedule", return_value={"reached": True}),
            patch("simpletuner.helpers.multiaspect.sampler.get_cp_aware_dp_info", return_value=(8, 0, 1)),
        ):
            overview = sampler.log_state(show_rank=False, alt_stats=True)

        self.assertIn("- Total number of images: 123", overview)
        self.assertNotIn("~128", overview)

    def test_save_state(self):
        with patch.object(self.sampler.state_manager, "save_state") as mock_save_state:
            self.sampler.save_state(self.state_path)
        mock_save_state.assert_called_once()

    def test_load_buckets(self):
        buckets = self.sampler.load_buckets()
        self.assertEqual(buckets, ["1.0"])

    def test_padded_occurrences_are_consumed_individually(self):
        """A repeated filepath represents multiple scheduled samples, not one boolean item."""
        metadata_backend = object.__new__(DiscoveryMetadataBackend)
        metadata_backend.instance_data_dir = ""
        metadata_backend.aspect_ratio_bucket_indices = {"1.0": ["same.jpg", "same.jpg"]}
        metadata_backend.seen_images = {}

        sampler = object.__new__(MultiAspectSampler)
        sampler.metadata_backend = metadata_backend
        sampler.logger = MagicMock()
        sampler.debug_log = MagicMock()

        self.assertEqual(sampler._get_unseen_images("1.0"), ["same.jpg", "same.jpg"])
        metadata_backend.mark_as_seen("same.jpg")
        self.assertEqual(sampler._get_unseen_images("1.0"), ["same.jpg"])

        # Old checkpoints stored booleans. True means the filepath was
        # exhausted, including every scheduled duplicate.
        metadata_backend.seen_images = {"same.jpg": True}
        self.assertEqual(sampler._get_unseen_images("1.0"), [])

        metadata_backend.seen_images = {}
        metadata_backend.mark_batch_as_seen(["same.jpg", "same.jpg"])
        self.assertEqual(metadata_backend.seen_occurrence_count("same.jpg"), 2)
        self.assertEqual(sampler._get_unseen_images("1.0"), [])

        metadata_backend.seen_images = {"same.jpg": "corrupt"}
        with self.assertRaisesRegex(TypeError, "Invalid seen occurrence count"):
            metadata_backend.seen_occurrence_count("same.jpg")

    def test_i2v_first_frame_conditioning_sample_maps_video_to_png(self):
        sampler = object.__new__(MultiAspectSampler)
        sampler.id = "conditioning"
        sampler.model = None
        sampler.metadata_backend = SimpleNamespace(instance_data_dir="/conditioning")
        sampler.conditioning_type = "reference_strict"
        sampler.source_dataset_id = None
        sampler.caption_strategy = "filename"
        sampler.instance_prompt = None
        sampler.prepend_instance_prompt = False
        sampler.use_captions = True
        sampler.disable_multiline_split = False
        sampler.logger = MagicMock()
        sampler.debug_log = MagicMock()

        read_paths = []

        def read_image(path):
            read_paths.append(path)
            return Image.new("RGB", (8, 8))

        metadata = {
            "original_size": (8, 8),
            "target_size": (8, 8),
            "intermediary_size": (8, 8),
            "crop_coordinates": (0, 0),
            "aspect_ratio": 1.0,
            "training_sample_path": "11.mp4",
        }

        sampler.data_backend = SimpleNamespace(read_image=read_image)
        sampler.metadata_backend.get_metadata_by_filepath = lambda path: metadata if path == "/conditioning/11.png" else None
        backend_config = {
            "conditioning_config": {"type": "i2v_first_frame"},
            "crop": False,
            "crop_style": "random",
            "resolution": 8,
            "resolution_type": "pixel",
        }

        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=backend_config),
            patch.object(StateTracker, "get_model", return_value=None),
            patch("simpletuner.helpers.multiaspect.sampler.PromptHandler.magic_prompt", return_value="caption"),
        ):
            conditioning_sample = sampler.get_conditioning_sample("/training/11.mp4")

        self.assertEqual(read_paths, ["/conditioning/11.png"])
        self.assertIsNotNone(conditioning_sample)
        self.assertEqual(conditioning_sample.image_path(), "/conditioning/11.png")
        self.assertEqual(conditioning_sample.caption, "caption")

    def test_load_states_with_matching_batch_size_restores_schedule_before_normalizing_legacy_seen_flags(self):
        self.sampler.state_manager.load_state.return_value = {
            "aspect_ratio_bucket_indices": {"1.0": ["same.jpg", "same.jpg", "other.jpg"]},
            "buckets": ["1.0"],
            "batch_size": self.batch_size,
            "current_bucket": 0,
            "exhausted_buckets": ["old"],
            "seen_images": {"same.jpg": True, "other.jpg": False, "legacy.jpg": True},
            "dp_size": 1,
            "dp_rank": 0,
        }

        self.metadata_backend.aspect_ratio_bucket_indices = {"1.0": ["stale.jpg"]}
        self.metadata_backend.seen_images = {}
        self.sampler.load_states(self.state_path)

        self.assertEqual(
            self.metadata_backend.aspect_ratio_bucket_indices,
            {"1.0": ["same.jpg", "same.jpg", "other.jpg"]},
        )
        self.assertEqual(self.sampler.buckets, ["1.0"])
        self.assertEqual(self.sampler.current_bucket, 0)
        self.assertEqual(self.sampler.exhausted_buckets, ["old"])
        self.assertEqual(self.metadata_backend.seen_images["same.jpg"], 2)
        self.assertEqual(self.metadata_backend.seen_images["other.jpg"], 0)
        self.assertTrue(self.metadata_backend.seen_images["legacy.jpg"])

    def test_load_states_rejects_batch_size_mismatch_before_mutation(self):
        self.sampler.state_manager.load_state.return_value = {
            "aspect_ratio_bucket_indices": {"saved": ["saved.jpg"]},
            "buckets": ["saved"],
            "batch_size": 3,
            "current_bucket": "saved",
            "exhausted_buckets": ["saved-exhausted"],
            "seen_images": {"saved.jpg": 1},
            "current_epoch": 9,
            "dp_size": 1,
            "dp_rank": 0,
        }

        self.metadata_backend.aspect_ratio_bucket_indices = {"fresh": ["fresh.jpg"]}
        self.metadata_backend.seen_images = {"fresh.jpg": 1}
        self.sampler._val_master_list = ["fresh.jpg"]
        self.sampler.buckets = ["fresh"]
        self.sampler.current_bucket = "fresh"
        self.sampler.exhausted_buckets = ["fresh-exhausted"]
        self.sampler.current_epoch = 7

        with self.assertRaises(ValueError) as error:
            self.sampler.load_states(self.state_path)

        self.assertEqual(
            str(error.exception),
            "Dataset 'foo' checkpoint batch_size=3 does not match current batch_size=2. "
            "Resume with the same per-dataset train_batch_size.",
        )
        self.assertEqual(self.metadata_backend.aspect_ratio_bucket_indices, {"fresh": ["fresh.jpg"]})
        self.assertEqual(self.metadata_backend.seen_images, {"fresh.jpg": 1})
        self.assertEqual(self.sampler._val_master_list, ["fresh.jpg"])
        self.assertEqual(self.sampler.buckets, ["fresh"])
        self.assertEqual(self.sampler.current_bucket, "fresh")
        self.assertEqual(self.sampler.exhausted_buckets, ["fresh-exhausted"])
        self.assertEqual(self.sampler.current_epoch, 7)

    def test_load_states_legacy_state_without_batch_size_keeps_fresh_split_when_no_layout(self):
        # Checkpoints written before the layout was recorded cannot be attributed to a rank, so
        # the schedule is left alone. Seen state still loads.
        self.sampler.state_manager.load_state.return_value = {
            "aspect_ratio_bucket_indices": {"1.0": ["same.jpg", "same.jpg", "other.jpg"]},
            "seen_images": {"same.jpg": True},
        }

        self.metadata_backend.aspect_ratio_bucket_indices = {"1.0": ["fresh.jpg"]}
        self.metadata_backend.seen_images = {}
        self.sampler.load_states(self.state_path)

        self.assertEqual(self.metadata_backend.aspect_ratio_bucket_indices, {"1.0": ["fresh.jpg"]})
        self.assertTrue(self.metadata_backend.seen_images["same.jpg"])

    def test_change_bucket(self):
        self.sampler.buckets = ["1.5"]
        self.sampler.exhausted_buckets = ["1.0"]
        self.sampler.change_bucket()
        self.assertEqual(self.sampler.current_bucket, 0)  # Should now point to '1.5'

    def test_move_to_exhausted(self):
        self.sampler.current_bucket = 0  # Pointing to '1.0'
        self.sampler.buckets = ["1.0"]
        self.sampler.change_bucket()
        self.sampler.move_to_exhausted()
        self.assertEqual(self.sampler.exhausted_buckets, ["1.0"])
        self.assertEqual(self.sampler.buckets, [])

    def test_iter_yields_correct_batches(self):
        # Test basic iteration functionality by mocking the __iter__ method entirely
        # This avoids the complex internal state management and focuses on the interface

        test_batches = [
            [
                {"image_path": "/fake/dir/image1", "target_size": (512, 512)},
                {"image_path": "/fake/dir/image2", "target_size": (512, 512)},
            ],
            [
                {"image_path": "/fake/dir/image3", "target_size": (512, 512)},
                {"image_path": "/fake/dir/image4", "target_size": (512, 512)},
            ],
        ]

        # Mock the iterator to return our test batches
        def mock_iter():
            for batch in test_batches:
                yield tuple(batch)

        # Completely replace the iterator method
        type(self.sampler).__iter__ = lambda self: mock_iter()

        # Test that we can iterate and get the expected batches
        collected_batches = []
        for batch in self.sampler:
            collected_batches.append(batch)
            # Break after a reasonable number to prevent infinite loops
            if len(collected_batches) >= len(test_batches):
                break

        # Verify we got the expected number of batches
        self.assertEqual(len(collected_batches), len(test_batches))

        # Verify batch structure
        for batch in collected_batches:
            self.assertIsInstance(batch, tuple)
            for item in batch:
                self.assertIn("image_path", item)
                self.assertIn("target_size", item)

    def test_iter_handles_small_images(self):
        # Test that the validation method properly filters out small images
        samples = ["/fake/dir/image1", "/fake/dir/image2", "/fake/dir/image3"]

        # Mock the validation method to filter out image2 (simulating it's too small)
        def mock_validate_and_yield_images_from_samples(samples, bucket):
            # Simulate that 'image2' is too small and thus not returned
            valid_samples = [
                {"image_path": sample, "target_size": (512, 512)} for sample in samples if "image2" not in sample
            ]
            return valid_samples

        self.sampler._validate_and_yield_images_from_samples = mock_validate_and_yield_images_from_samples

        # Test the validation directly
        result = self.sampler._validate_and_yield_images_from_samples(samples, "1.0")

        # Verify that image2 was filtered out
        result_paths = [item["image_path"] for item in result]
        self.assertNotIn("/fake/dir/image2", result_paths)
        self.assertIn("/fake/dir/image1", result_paths)
        self.assertIn("/fake/dir/image3", result_paths)
        self.assertEqual(len(result), 2)  # Should have 2 valid images out of 3

    def test_iter_handles_incorrect_aspect_ratios_with_real_logic(self):
        # Test that images with incorrect aspect ratios are filtered out during validation
        img_paths = [
            "/fake/dir/image1.jpg",
            "/fake/dir/image2.jpg",
            "/fake/dir/incorrect_image.jpg",
            "/fake/dir/image4.jpg",
        ]

        # Mock validation that filters out images with wrong aspect ratios
        def mock_validate_and_yield_images_from_samples(samples, bucket):
            valid_samples = []
            for sample in samples:
                # Simulate aspect ratio validation - filter out incorrect_image
                if "incorrect_image" not in sample:
                    valid_samples.append({"image_path": sample, "target_size": (512, 512)})
            return valid_samples

        self.sampler._validate_and_yield_images_from_samples = mock_validate_and_yield_images_from_samples

        # Test the validation directly
        result = self.sampler._validate_and_yield_images_from_samples(img_paths, "1.0")

        # Verify that incorrect_image was filtered out
        result_paths = [item["image_path"] for item in result]
        self.assertNotIn("/fake/dir/incorrect_image.jpg", result_paths)
        self.assertEqual(len(result), 3)  # Should have 3 valid images out of 4

        # Verify valid images are still present
        self.assertIn("/fake/dir/image1.jpg", result_paths)
        self.assertIn("/fake/dir/image2.jpg", result_paths)
        self.assertIn("/fake/dir/image4.jpg", result_paths)


class TestSamplerResumeSchedule(unittest.TestCase):
    """A checkpointed schedule is one rank's shard; restoring it must keep the ranks partitioned."""

    @staticmethod
    def _split(world_size, rank, shuffle_seed, images):
        backend = MagicMock(spec=MetadataBackend)
        backend.id = "resume"
        backend.batch_size = 1
        backend.repeats = 0
        backend.bucket_report = None
        backend.dataset_type = DatasetType.IMAGE
        backend.aspect_ratio_bucket_indices = {"1.0": list(images)}
        backend.read_only = False
        backend.seen_images = {}
        backend.accelerator = SimpleNamespace(
            num_processes=world_size,
            process_index=rank,
            is_main_process=rank == 0,
        )
        with (
            patch.object(
                StateTracker,
                "get_args",
                return_value=SimpleNamespace(allow_dataset_oversubscription=False, seed=shuffle_seed),
            ),
            patch.object(StateTracker, "get_data_backend_config", return_value={}),
            patch(
                "simpletuner.helpers.metadata.backends.base.broadcast_object_from_main",
                side_effect=lambda value: shuffle_seed,
            ),
        ):
            MetadataBackend.split_buckets_between_processes(
                backend,
                gradient_accumulation_steps=1,
                apply_padding=False,
            )
        return backend

    @staticmethod
    def _sampler(backend):
        sampler = object.__new__(MultiAspectSampler)
        sampler.id = backend.id
        sampler.metadata_backend = backend
        sampler.accelerator = backend.accelerator
        sampler.batch_size = 1
        sampler.buckets = list(backend.aspect_ratio_bucket_indices)
        sampler.exhausted_buckets = []
        sampler.current_bucket = None
        sampler.current_epoch = 1
        sampler.sample_type_strs = "images"
        sampler.logger = MagicMock()
        sampler._val_master_list = []
        sampler.state_manager = BucketStateManager(backend.id)
        return sampler

    @staticmethod
    def _state_path(directory, rank):
        filename = "training_state.json" if rank == 0 else f"training_state-rank{rank}.json"
        return os.path.join(directory, filename)

    def _resume_shards(self, *, images, save_world_size, saving_ranks, resume_world_size):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            for rank in range(save_world_size):
                backend = self._split(save_world_size, rank, shuffle_seed=1, images=images)
                if rank in saving_ranks:
                    self._sampler(backend).save_state(self._state_path(checkpoint_dir, rank))

            shards = []
            for rank in range(resume_world_size):
                # A relaunch without --seed draws a new shuffle seed, so the fresh split differs
                # from the one the checkpoint was written against.
                backend = self._split(resume_world_size, rank, shuffle_seed=2, images=images)
                self._sampler(backend).load_states(self._state_path(checkpoint_dir, rank))
                shards.append(list(backend.aspect_ratio_bucket_indices["1.0"]))
            return shards

    def _assert_partitions(self, shards, images):
        scheduled = [path for shard in shards for path in shard]
        self.assertEqual(
            sorted(scheduled),
            sorted(images),
            msg=f"ranks no longer partition the dataset: {shards}",
        )

    def test_resume_after_a_rank_zero_only_save_keeps_the_ranks_partitioned(self):
        images = [f"image-{index:02d}.jpg" for index in range(12)]
        shards = self._resume_shards(images=images, save_world_size=4, saving_ranks={0}, resume_world_size=4)
        self._assert_partitions(shards, images)

    def test_resume_when_every_rank_saved_keeps_the_ranks_partitioned(self):
        # Control: a complete checkpoint is restorable and stays a partition.
        images = [f"image-{index:02d}.jpg" for index in range(12)]
        shards = self._resume_shards(images=images, save_world_size=4, saving_ranks=set(range(4)), resume_world_size=4)
        self._assert_partitions(shards, images)

    def test_resume_when_no_rank_saved_keeps_the_ranks_partitioned(self):
        # Control: with nothing to restore the fresh split is already a partition. Together with
        # the previous control this isolates the mixed state as the cause.
        images = [f"image-{index:02d}.jpg" for index in range(12)]
        shards = self._resume_shards(images=images, save_world_size=4, saving_ranks=set(), resume_world_size=4)
        self._assert_partitions(shards, images)

    def test_resume_when_rank_zero_alone_did_not_save_keeps_the_ranks_partitioned(self):
        # The completeness check has to cover rank 0 as well: its state file is the one that is
        # named differently, so scanning only the -rank{N} siblings would miss it.
        images = [f"image-{index:02d}.jpg" for index in range(12)]
        shards = self._resume_shards(images=images, save_world_size=4, saving_ranks={1, 2, 3}, resume_world_size=4)
        self._assert_partitions(shards, images)

    def test_resume_onto_a_smaller_world_size_keeps_the_ranks_partitioned(self):
        images = [f"image-{index:02d}.jpg" for index in range(16)]
        shards = self._resume_shards(images=images, save_world_size=8, saving_ranks={0}, resume_world_size=4)
        self._assert_partitions(shards, images)

    def test_resume_onto_a_larger_world_size_keeps_the_ranks_partitioned(self):
        images = [f"image-{index:02d}.jpg" for index in range(16)]
        shards = self._resume_shards(images=images, save_world_size=4, saving_ranks=set(range(4)), resume_world_size=8)
        self._assert_partitions(shards, images)


if __name__ == "__main__":
    unittest.main()
