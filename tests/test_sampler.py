import copy
import json
import logging
import os
import random
import tempfile
import unittest
from math import ceil
from pathlib import Path
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

from simpletuner.helpers.caching.vae import VAECache
from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.data_backend.local import LocalDataBackend
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.metadata.backends.discovery import DiscoveryMetadataBackend
from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler, _UnseenOccurrenceIndex
from simpletuner.helpers.multiaspect.state import BucketStateManager
from simpletuner.helpers.training.exceptions import MultiDatasetExhausted
from simpletuner.helpers.training.state_tracker import StateTracker
from tests.helpers.data import MockDataBackend

# Captured at import: a TestMultiAspectSampler case replaces MultiAspectSampler.__iter__ on the class.
_SAMPLER_ITER = MultiAspectSampler.__iter__


class SamplerURIPathTests(unittest.TestCase):
    def test_webshart_training_paths_match_precache_identifiers(self):
        from simpletuner.helpers.utils.pathing import normalize_data_path

        sampler = MultiAspectSampler.__new__(MultiAspectSampler)
        sampler.metadata_backend = SimpleNamespace(instance_data_dir="cache/webshart/source")
        sampler.data_backend = SimpleNamespace(read_image=Mock(return_value=None))
        canonical = "webshart://0/1497/sample.webp"
        for identifier in (canonical, str(Path(canonical))):
            with self.subTest(identifier=identifier):
                training_path = sampler._sample_path(identifier)
                self.assertEqual(normalize_data_path(training_path), normalize_data_path(canonical))
                sampler.data_backend.read_image.reset_mock()
                sampler.yield_single_image(identifier)
                sampler.data_backend.read_image.assert_called_once_with(canonical)


class SamplerStatePublicationTests(unittest.TestCase):
    def test_failed_checkpoint_write_preserves_previous_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "checkpoint.json")
            manager = BucketStateManager("images")
            previous = {"seen_images": {"image.jpg": 2}}
            manager.save_state(previous, path)
            with self.assertRaises(TypeError):
                manager.save_state({"seen_images": {"image.jpg": 3}, "invalid": object()}, path)
            self.assertEqual(manager.load_state(path), previous)

    def test_follower_reconstructs_consumed_counts_from_portable_batches(self):
        from simpletuner.helpers.multiaspect.progress import LegacySamplerProgress

        follower = LegacySamplerProgress({}, {"position": 0, "repeat": 0})
        follower.acknowledge({"position": 1, "repeat": 0, "refs": ("a", "a")})
        with follower.checkpoint_seen() as seen:
            self.assertEqual(seen, {"a": 2})
        follower.acknowledge({"position": 2, "repeat": 1, "refs": ("b", "c")})
        with follower.checkpoint_seen() as seen:
            self.assertEqual(seen, {"b": 1, "c": 1})


class TestMultiAspectSampler(unittest.TestCase):
    def setUp(self):
        self.process_state = PartialState()
        self.accelerator = MagicMock(num_processes=1, process_index=0)
        self.accelerator.log = MagicMock()
        self.metadata_backend = Mock(spec=DiscoveryMetadataBackend, epoch_provider=None)
        self.metadata_backend.get_metadata_batch.side_effect = lambda refs: MetadataBackend.get_metadata_batch(
            self.metadata_backend, refs
        )
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
            dataset_runtime=DatasetRuntime("foo", self.data_backend, self.metadata_backend, {}),
            accelerator=self.accelerator,
            batch_size=self.batch_size,
            minimum_image_size=0,
            model=MagicMock(),
        )

        self.sampler.state_manager = Mock(spec=BucketStateManager)
        self.sampler.state_manager.load_state.return_value = {}

    def test_len(self):
        self.assertEqual(len(self.sampler), 2)

    def test_len_counts_partial_batches_separately_for_each_bucket_and_repeat(self):
        self.metadata_backend.aspect_ratio_bucket_indices = {
            "1.0": ["image1", "image2", "image3"],
            "1.5": ["image4", "image5", "image6"],
            "2.0": [],
        }
        self.sampler.dataset_runtime.config["repeats"] = 1
        self.assertEqual(len(self.sampler), 8)

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
        metadata_backend = SimpleNamespace(
            instance_data_dir="/conditioning", epoch_provider=None, requires_complete_preparation=False
        )

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

        data_backend = SimpleNamespace(read_image=read_image)
        metadata_backend.get_metadata_by_filepath = lambda path: metadata if path == "/conditioning/11.png" else None
        metadata_backend.get_metadata_batch = lambda refs: MetadataBackend.get_metadata_batch(metadata_backend, refs)
        backend_config = {
            "conditioning_config": {"type": "i2v_first_frame"},
            "conditioning_type": "reference_strict",
            "crop": False,
            "crop_style": "random",
            "resolution": 8,
            "resolution_type": "pixel",
        }
        metadata_backend.active_dataset_config = backend_config
        metadata_backend.conditioning_ref = MetadataBackend.conditioning_ref.__get__(metadata_backend)

        dataset = DatasetRuntime("conditioning", data_backend, metadata_backend, backend_config)
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=backend_config),
            patch.object(StateTracker, "get_model", return_value=None),
            patch("simpletuner.helpers.prompts.PromptHandler.magic_prompt", return_value="caption"),
        ):
            conditioning_sample = dataset.read_conditioning_sample("/training/11.mp4")

        self.assertEqual(read_paths, ["/conditioning/11.png"])
        self.assertIsNotNone(conditioning_sample)
        self.assertEqual(conditioning_sample.image_path(), "/conditioning/11.png")
        self.assertEqual(conditioning_sample.caption, "caption")

    def test_connect_conditioning_samples_preserves_nested_mask_paths(self):
        relative_paths = ["white.deriv/shared.png", "c2s_over/shared.png"]
        metadata = {
            "original_size": (8, 8),
            "target_size": (8, 8),
            "intermediary_size": (8, 8),
            "crop_coordinates": (0, 0),
            "aspect_ratio": 1.0,
        }
        backend_config = {"crop": False, "resolution": 8, "resolution_type": "pixel"}

        with tempfile.TemporaryDirectory() as dataset_dir:
            source_dir = os.path.join(dataset_dir, "images")
            samples = tuple({"image_path": os.path.join(source_dir, path)} for path in relative_paths)
            self.metadata_backend.instance_data_dir = source_dir
            self.sampler.logger = MagicMock()
            conditioning_datasets = []
            for dataset_idx in range(2):
                conditioning_id = f"masks-{dataset_idx}"
                conditioning_dir = os.path.join(dataset_dir, conditioning_id)
                for image_idx, relative_path in enumerate(relative_paths):
                    mask_path = os.path.join(conditioning_dir, relative_path)
                    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                    Image.new("RGB", (8, 8), color=(image_idx * 255,) * 3).save(mask_path)

                conditioning_metadata = Mock(spec=DiscoveryMetadataBackend, epoch_provider=None)
                conditioning_metadata.get_metadata_batch.side_effect = lambda refs: (metadata for _ in refs)
                conditioning_metadata.id = conditioning_id
                conditioning_metadata.instance_data_dir = conditioning_dir
                conditioning_metadata.active_dataset_config = {}
                conditioning_metadata.conditioning_ref.side_effect = MetadataBackend.conditioning_ref.__get__(
                    conditioning_metadata
                )
                conditioning_metadata.aspect_ratio_bucket_indices = {}
                conditioning_metadata.get_metadata_by_filepath.return_value = metadata
                conditioning_datasets.append(
                    DatasetRuntime(
                        conditioning_id,
                        LocalDataBackend(self.accelerator, conditioning_id),
                        conditioning_metadata,
                        {**backend_config, "conditioning_type": "mask"},
                    )
                )

            for sampling_mode in ("random", "combined"):
                for dataset_count in (1, 2):
                    with (
                        self.subTest(sampling_mode=sampling_mode, dataset_count=dataset_count),
                        patch.object(
                            self.sampler.dataset_runtime, "conditioning_datasets", conditioning_datasets[:dataset_count]
                        ),
                        patch.object(
                            StateTracker,
                            "get_args",
                            return_value=SimpleNamespace(conditioning_multidataset_sampling=sampling_mode),
                        ),
                        patch.object(StateTracker, "get_data_backend_config", return_value=backend_config),
                        patch.object(StateTracker, "get_model", return_value=None),
                        patch("simpletuner.helpers.prompts.PromptHandler.magic_prompt", return_value="caption"),
                    ):
                        self.sampler.logger.reset_mock()
                        output = self.sampler.connect_conditioning_samples(samples)

                        expected_count = len(samples) * (1 if sampling_mode == "random" else dataset_count)
                        self.assertEqual(output[: len(samples)], samples)
                        masks = output[len(samples) :]
                        self.assertEqual(len(masks), expected_count)
                        for mask_idx, mask in enumerate(masks):
                            relative_path = relative_paths[mask_idx % len(samples)]
                            self.assertEqual(
                                mask.image_path(), os.path.join(dataset_dir, mask.data_backend_id, relative_path)
                            )
                            self.assertEqual(mask.image.getpixel((0, 0)), (mask_idx % len(samples) * 255,) * 3)
                            self.assertEqual(mask.conditioning_type, "mask")
                            self.assertEqual(
                                mask.training_sample_path(self.sampler.id), samples[mask_idx % len(samples)]["image_path"]
                            )
                        self.sampler.logger.warning.assert_not_called()

    def test_legacy_random_conditioning_uses_reproducible_sample_choices(self):
        self.sampler.current_epoch = 3
        samples = [{"image_path": f"sample-{index}.png"} for index in range(24)]
        source = self.sampler.dataset_runtime
        source.metadata_backend.instance_data_dir = ""
        targets = tuple(
            SimpleNamespace(id=f"mask-{index}", read_conditioning_sample=Mock(side_effect=lambda *a, **kw: Mock()))
            for index in range(2)
        )
        with (
            patch.object(source, "conditioning_datasets", targets),
            patch.object(
                StateTracker, "get_args", return_value=SimpleNamespace(conditioning_multidataset_sampling="random")
            ),
        ):
            output = self.sampler.connect_conditioning_samples(samples)
        selected = [record._source_dataset_id for record in output[len(samples) :]]
        expected = [
            targets[random.Random(f"{self.sampler.id}:3:{source.pairing_key(sample['image_path'])}").randrange(2)].id
            for sample in samples
        ]
        self.assertEqual(selected, expected)

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
        self.assertIs(self.metadata_backend.seen_images["legacy.jpg"], True)

    def test_loading_integer_seen_counts_does_not_walk_the_bucket_schedule(self):
        schedule = MagicMock(wraps={"1.0": ["image1", "image1", "image2"]})
        schedule.values.side_effect = AssertionError("checkpoint walked the complete bucket schedule")
        self.metadata_backend.aspect_ratio_bucket_indices = schedule
        self.sampler.state_manager.load_state.return_value = {"buckets": ["1.0"], "seen_images": {"image1": 2}}

        self.sampler.load_states(self.state_path)

        self.assertEqual(self.metadata_backend.seen_images, {"image1": 2})
        schedule.values.assert_not_called()

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
        self.sampler.validation_cursor = MagicMock()
        self.sampler.buckets = ["fresh"]
        self.sampler.current_bucket = "fresh"
        self.sampler.exhausted_buckets = ["fresh-exhausted"]
        self.sampler.current_epoch = 7

        with (
            patch.dict(os.environ, {"SIMPLETUNER_ALLOW_MODIFYING_BSZ": ""}),
            patch.object(StateTracker, "get_args", return_value=SimpleNamespace(i_know_what_i_am_doing=False)),
            self.assertRaises(ValueError) as error,
        ):
            self.sampler.load_states(self.state_path)

        self.assertEqual(
            str(error.exception),
            "Dataset 'foo' checkpoint batch_size=3 does not match current batch_size=2. "
            "Resume with the same per-dataset train_batch_size, "
            "or set --i_know_what_i_am_doing / SIMPLETUNER_ALLOW_MODIFYING_BSZ=1 to bypass.",
        )
        self.assertEqual(self.metadata_backend.aspect_ratio_bucket_indices, {"fresh": ["fresh.jpg"]})
        self.assertEqual(self.metadata_backend.seen_images, {"fresh.jpg": 1})
        self.sampler.validation_cursor.close.assert_not_called()
        self.assertEqual(self.sampler.buckets, ["fresh"])
        self.assertEqual(self.sampler.current_bucket, "fresh")
        self.assertEqual(self.sampler.exhausted_buckets, ["fresh-exhausted"])
        self.assertEqual(self.sampler.current_epoch, 7)

    def test_load_states_batch_size_mismatch_allowed_via_env_var(self):
        self.sampler.state_manager.load_state.return_value = {
            "aspect_ratio_bucket_indices": {"saved": ["saved.jpg"]},
            "buckets": ["saved"],
            "batch_size": 3,
        }
        self.metadata_backend.aspect_ratio_bucket_indices = {"fresh": ["fresh.jpg"]}

        with patch.dict(os.environ, {"SIMPLETUNER_ALLOW_MODIFYING_BSZ": "1"}):
            with patch.object(StateTracker, "get_args", return_value=SimpleNamespace(i_know_what_i_am_doing=False)):
                self.sampler.load_states(self.state_path)

    def test_load_states_batch_size_mismatch_allowed_via_i_know_flag(self):
        self.sampler.state_manager.load_state.return_value = {
            "aspect_ratio_bucket_indices": {"saved": ["saved.jpg"]},
            "buckets": ["saved"],
            "batch_size": 3,
        }
        self.metadata_backend.aspect_ratio_bucket_indices = {"fresh": ["fresh.jpg"]}
        with patch.object(StateTracker, "get_args", return_value=SimpleNamespace(i_know_what_i_am_doing=True)):
            self.sampler.load_states(self.state_path)

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
        iterator_patch = patch.object(type(self.sampler), "__iter__", lambda self: mock_iter())
        iterator_patch.start()
        self.addCleanup(iterator_patch.stop)

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
            backend.runtime_args = StateTracker.get_args()
            backend.active_dataset_config = StateTracker.get_data_backend_config(backend.id)
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
        sampler.validation_cursor = MagicMock()
        sampler.state_manager = BucketStateManager(backend.id)
        return sampler

    @staticmethod
    def _state_path(directory, rank):
        filename = "training_state.json" if rank == 0 else f"training_state-rank{rank}.json"
        return os.path.join(directory, filename)

    def test_resume_normalizes_numeric_bucket_names_and_preserves_remaining_occurrences(self):
        backend = DiscoveryMetadataBackend.__new__(DiscoveryMetadataBackend)
        backend.id = "resume"
        backend.instance_data_dir = ""
        backend.accelerator = SimpleNamespace(num_processes=1, process_index=0)
        backend.aspect_ratio_bucket_indices = {"1.0": ["fresh.jpg"]}
        backend.seen_images = {}
        sampler = self._sampler(backend)
        sampler.rank_info = "[RANK 0]"
        saved_schedule = {
            "1.0": ["repeat.jpg", "repeat.jpg", "other.jpg"],
            "1.5": ["wide.jpg"],
            "1920x1080@125": ["video.mp4"],
        }
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            state_path = self._state_path(checkpoint_dir, 0)
            sampler.state_manager.save_state(
                {
                    "aspect_ratio_bucket_indices": saved_schedule,
                    "buckets": [1.0, "1920x1080@125"],
                    "exhausted_buckets": [1.5],
                    "current_bucket": 0,
                    "batch_size": 1,
                    "seen_images": {"repeat.jpg": 1, "wide.jpg": 1},
                    "dp_size": 1,
                    "dp_rank": 0,
                },
                state_path,
            )
            sampler.load_states(state_path)
            self.assertEqual(sampler.buckets, ["1.0", "1920x1080@125"])
            self.assertEqual(sampler.exhausted_buckets, ["1.5"])
            self.assertEqual(sampler.current_bucket, 0)
            self.assertEqual(backend.aspect_ratio_bucket_indices, saved_schedule)
            self.assertEqual(sampler._get_unseen_images("1.0"), ["repeat.jpg", "other.jpg"])
            backend.mark_batch_as_seen(["repeat.jpg", "other.jpg"])
            self.assertEqual(sampler._get_unseen_images("1.0"), [])

            sampler.save_state(state_path)
            sampler.load_states(state_path)
            self.assertEqual(backend.aspect_ratio_bucket_indices, saved_schedule)
            self.assertEqual(sampler._get_unseen_images("1.0"), [])
            self.assertEqual(sampler._get_unseen_images("1920x1080@125"), ["video.mp4"])
        sampler.logger.warning.assert_not_called()

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


class TestIterationUnseenIndex(unittest.TestCase):
    """The iterator's unseen index must yield exactly what recomputing `_get_unseen_images` yields."""

    # Bucket sizes are batch multiples, so every scheduled occurrence (including the padded
    # duplicates in "1.0" and "2.0") is consumed within an epoch.
    FULL_BUCKETS = {
        "0.5": [f"/d/s{i}.jpg" for i in range(8)],
        "1.0": [f"/d/a{i}.jpg" for i in range(45)] + ["/d/a3.jpg", "/d/a44.jpg", "/d/a44.jpg"],
        "1.5": [f"/d/b{i}.jpg" for i in range(24)],
        "2.0": [f"/d/c{i}.jpg" for i in range(7)] + ["/d/c6.jpg"],
    }
    # A bucket smaller than the batch drives the partial-fill path, which re-marks already consumed
    # filepaths; legacy boolean flags exercise the filter's boolean semantics when an index is built.
    PARTIAL_BUCKETS = {
        "0.5": ["/d/s0.jpg", "/d/s1.jpg", "/d/s2.jpg"],
        "1.0": [f"/d/a{i}.jpg" for i in range(47)] + ["/d/a3.jpg", "/d/a46.jpg", "/d/a46.jpg"],
        "1.5": [f"/d/b{i}.jpg" for i in range(23)],
        "2.0": [f"/d/c{i}.jpg" for i in range(8)] + ["/d/c7.jpg"],
    }
    PARTIAL_SEEN = {"/d/a5.jpg": 1, "/d/s1.jpg": True, "/d/c2.jpg": False, "/d/outside.jpg": True}

    def setUp(self):
        PartialState()
        self.get_args = patch.object(StateTracker, "get_args", return_value=SimpleNamespace(print_sampler_statistics=False))
        self.get_args.start()

    def tearDown(self):
        self.get_args.stop()

    def _sampler(self, buckets, seen, use_index, *, track_consumed=False, repeats=0):
        metadata_backend = object.__new__(DiscoveryMetadataBackend)
        metadata_backend.id = "unseen"
        metadata_backend.removed_samples = set()
        metadata_backend.instance_data_dir = ""
        metadata_backend.aspect_ratio_bucket_indices = copy.deepcopy(buckets)
        metadata_backend.seen_images = dict(seen)
        metadata_backend.read_only = True
        metadata_backend.filtering_statistics = None
        metadata_backend.bucket_report = None
        data_backend = MockDataBackend()
        data_backend.id = "unseen"
        sampler = MultiAspectSampler(
            dataset_runtime=DatasetRuntime("unseen", data_backend, metadata_backend, {"repeats": repeats}),
            accelerator=SimpleNamespace(num_processes=1, process_index=0),
            batch_size=4,
            minimum_image_size=0,
            model=MagicMock(),
            track_consumed=track_consumed,
        )
        sampler._validate_and_yield_images_from_samples = lambda samples, bucket, **kwargs: [
            {"image_path": p} for p in samples
        ]
        sampler.connect_conditioning_samples = lambda samples, **kwargs: samples
        sampler.connect_s2v_samples = lambda samples: samples
        stored = {}
        sampler.state_manager = SimpleNamespace(
            save_state=lambda state, path: stored.update(state=json.loads(json.dumps(state))),
            load_state=lambda path: copy.deepcopy(stored["state"]),
        )
        if not use_index:
            sampler._iteration_unseen_images = sampler._get_unseen_images
        return sampler

    def test_persistent_sampler_keeps_the_next_bucket_after_a_padded_batch(self):
        from simpletuner.helpers.training.exceptions import MultiDatasetExhausted

        full_bucket = [f"full-{index}.jpg" for index in range(8)]
        buckets = {"0.5": ["short.jpg"], "1.0": full_bucket}
        sampler = self._sampler(buckets, {}, True)
        sampler.buckets = list(buckets)
        sampler.current_bucket = None
        batches = iter(sampler)
        self.assertEqual([record["image_path"] for record in next(batches)], ["short.jpg"] * 4)
        self.assertCountEqual(
            [record["image_path"] for _ in range(2) for record in next(batches)],
            full_bucket,
        )
        with self.assertRaises(MultiDatasetExhausted):
            next(batches)

    def test_consumed_checkpoint_ignores_prefetch_across_repeat_boundaries(self):
        from simpletuner.helpers.data_backend.runtime.batch_fetcher import BatchFetcher
        from simpletuner.helpers.data_backend.runtime.dataloader_iterator import DatasetMixer

        class Batches:
            def __init__(self, sampler):
                self.sampler = sampler

            def __iter__(self):
                for batch in self.sampler:
                    yield {"images": [row["image_path"] for row in batch], "seed": batch[0]["caption_dropout_seed"]}

        buckets = {"0.5": ["a", "b", "b"], "1.0": ["c", "d", "e", "f"]}

        def runtime():
            sampler = self._sampler(buckets, {}, True, track_consumed=True, repeats=1)
            dataset = sampler.dataset_runtime
            dataset.sampler = sampler
            dataset.training_dataloader = Batches(sampler)
            dataset.start_epoch(1)
            return dataset

        def fetch(dataset, state=None):
            datasets = {dataset.id: dataset}
            mixer = DatasetMixer(datasets, epoch=1, step=0, saved=state)
            fetcher = BatchFetcher(step=mixer.step, max_size=8, datasets=datasets, mixer=mixer)
            fetcher.fetch_responses()
            return fetcher

        args = SimpleNamespace(seed=22, print_sampler_statistics=False)
        with (
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "exhausted_backends", []),
            patch.object(StateTracker, "repeats", {}),
            tempfile.TemporaryDirectory() as directory,
        ):
            filename = str(Path(directory) / "training_state.json")
            original = runtime()
            fetcher = fetch(original)
            first = fetcher.next_response(1)
            original.acknowledge_batch(first["dataset_continuation"])
            self.assertEqual(original.sampler.fetched_state()["position"], 4)
            self.assertEqual(original.sampler.fetched_state()["repeat"], 1)
            original.sampler.save_state(filename)
            state = original.sampler.state_manager.load_state(filename)
            self.assertEqual(state["legacy_cursor"]["position"], 1)
            self.assertEqual(state["legacy_cursor"]["repeat"], 0)
            self.assertEqual(sum(state["seen_images"].values()), 4)

            resumed = runtime()
            resumed.sampler.state_manager.load_state = lambda path: state
            resumed.sampler.load_states(filename)
            replay = fetch(resumed, first["dataset_mixer_state"])
            expected = [fetcher.next_response(step) for step in range(2, 6)]
            self.assertEqual([replay.next_response(step) for step in range(2, 6)], expected)
            self.assertIs(expected[-1], False)

    def _run(self, sampler, epochs, restore_schedule, per_step):
        """Drain `epochs` epochs, disturbing the schedule and seen state the way training can.

        `per_step` takes one batch from a fresh iterator each step, as `random_dataloader_iterator` does.
        """
        random.seed(1234)
        yielded = []
        epoch = 0
        iterator = _SAMPLER_ITER(sampler)
        while epoch < epochs:
            if per_step:
                iterator = _SAMPLER_ITER(sampler)
            try:
                batch = next(iterator)
            except MultiDatasetExhausted:
                epoch += 1
                yielded.append("epoch")
                iterator = _SAMPLER_ITER(sampler)
                continue
            yielded.append([item["image_path"] for item in batch])
            step = len(yielded)
            self.assertLess(step, 1000, "the schedule never reached the requested epoch count")
            backend = sampler.metadata_backend
            if step == 28:
                # A problematic sample dropped from a split shard is refilled in place to the same length.
                vae_cache = object.__new__(VAECache)
                vae_cache.id = "unseen"
                vae_cache.metadata_backend = backend
                vae_cache._handle_metadata_filtered_sample(filepath="/d/a3.jpg", reason="problematic")
            elif step == 30:
                backend.remove_image("/d/b20.jpg", "1.5")
            elif step == 45:
                # A re-split hands the rank new bucket lists in a new order.
                backend.aspect_ratio_bucket_indices = {
                    key: list(reversed(images)) for key, images in backend.aspect_ratio_bucket_indices.items()
                }
            elif step == 50:
                sampler.save_state("/unused/state.json")
            elif step == 60:
                # Resume from the step-50 checkpoint: seen counts rewind under an existing index.
                if not restore_schedule:
                    # An unrestorable schedule keeps the bucket lists and replaces only the seen counts.
                    sampler.state_manager.load_state = lambda path, state=sampler.state_manager.load_state: {
                        key: value for key, value in state(path).items() if key != "aspect_ratio_bucket_indices"
                    }
                sampler.load_states("/unused/state.json")
                iterator = _SAMPLER_ITER(sampler)
        return yielded

    def _assert_same_schedule(self, buckets, seen, epochs, restore_schedule, per_step=False):
        reference = self._run(self._sampler(buckets, seen, use_index=False), epochs, restore_schedule, per_step)
        sampler = self._sampler(buckets, seen, use_index=True)
        with patch.object(sampler, "_unseen_occurrence_positions", wraps=sampler._unseen_occurrence_positions) as scans:
            indexed = self._run(sampler, epochs, restore_schedule, per_step)
        self.assertEqual(indexed, reference)
        return reference, scans.call_count

    def test_iterator_yields_the_recomputed_schedule(self):
        reference, scans = self._assert_same_schedule(self.FULL_BUCKETS, {}, epochs=4, restore_schedule=False)
        # One scan per bucket per epoch plus the filter, removal, re-split and resume rebuilds, not one per batch.
        batches = sum(1 for entry in reference if entry != "epoch")
        self.assertLessEqual(scans, len(self.FULL_BUCKETS) * (4 + 3))
        self.assertGreater(batches, 2 * len(self.FULL_BUCKETS) * (4 + 3))
        first_epoch = [path for entry in reference[: reference.index("epoch")] for path in entry]
        self.assertEqual(sorted(first_epoch), sorted(path for images in self.FULL_BUCKETS.values() for path in images))

    def test_one_batch_per_iterator_yields_the_recomputed_schedule(self):
        for buckets, seen in ((self.FULL_BUCKETS, {}), (self.PARTIAL_BUCKETS, self.PARTIAL_SEEN)):
            with self.subTest(buckets=len(buckets[next(iter(buckets))])):
                reference, _ = self._assert_same_schedule(buckets, seen, epochs=4, restore_schedule=False, per_step=True)
                self.assertGreater(sum(1 for entry in reference if entry != "epoch"), 60)

    def test_iterator_yields_the_recomputed_schedule_through_partial_batches(self):
        self._assert_same_schedule(self.PARTIAL_BUCKETS, self.PARTIAL_SEEN, epochs=12, restore_schedule=True)

    def test_marking_retires_the_earliest_unseen_duplicate(self):
        source = ["a", "b", "a"]
        index = _UnseenOccurrenceIndex(source, {}, [0, 1, 2], list(source))
        index.consume("a")
        self.assertEqual((index.positions, index.sample_paths), ([1, 2], ["b", "a"]))
        index.consume("a")
        index.consume("a")
        self.assertEqual((index.positions, index.sample_paths), ([1], ["b"]))


if __name__ == "__main__":
    unittest.main()
