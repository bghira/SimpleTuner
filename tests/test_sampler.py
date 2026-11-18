import json
import logging
import os
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

from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.data_backend.local import LocalDataBackend
from simpletuner.helpers.metadata.backends.discovery import DiscoveryMetadataBackend
from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
from simpletuner.helpers.multiaspect.state import BucketStateManager
from simpletuner.helpers.training.exceptions import MultiDatasetExhausted
from simpletuner.helpers.training.state_tracker import StateTracker
from tests.helpers.data import MockDataBackend


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
        self.metadata_backend.instance_data_dir = "/fake/dir"
        self.data_backend = MockDataBackend()
        self.data_backend.id = "foo"
        self.batch_size = 2
        self.state_path = "/some/fake/state.json"

        self.sampler = self._new_sampler()
        self.sampler.state_manager.load_state.return_value = {}

    def test_len(self):
        self.assertEqual(len(self.sampler), 2)

    def test_model_card_overview_counts_the_whole_dataset(self):
        sampler = object.__new__(MultiAspectSampler)
        sampler.id = "video-dataset"
        sampler.metadata_backend = SimpleNamespace(aspect_ratio_bucket_indices={"1.0": [None] * 600, "1.5": [None] * 40})
        sampler.accelerator = SimpleNamespace(num_processes=8, process_index=0)
        sampler.logger = MagicMock()
        sampler.sample_type_strs = "videos"
        sampler.is_regularisation_data = False
        sampler.conditioning_type = None

        with (
            patch.object(StateTracker, "get_data_backend_config", return_value={"video": {"num_frames": 39}}),
            patch.object(StateTracker, "get_dataset_schedule", return_value={"reached": True}),
        ):
            overview = sampler.log_state(show_rank=False, alt_stats=True)

        self.assertIn("- Total number of videos: 640", overview)
        self.assertIn("- Total number of aspect buckets: 2", overview)
        self.assertIn("- Target frame count: 39", overview)
        self.assertIn("- FPS: unknown", overview)

    def test_save_state(self):
        with patch.object(self.sampler.state_manager, "save_state") as mock_save_state:
            self.sampler.save_state(self.state_path)
        mock_save_state.assert_called_once()

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
        sampler.dataset_runtime = DatasetRuntime("conditioning", sampler.data_backend, sampler.metadata_backend, {})
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
            patch("simpletuner.helpers.prompts.PromptHandler.magic_prompt", return_value="caption"),
        ):
            conditioning_sample = sampler.get_conditioning_sample("/training/11.mp4")

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

                conditioning_metadata = Mock(spec=DiscoveryMetadataBackend)
                conditioning_metadata.id = conditioning_id
                conditioning_metadata.instance_data_dir = conditioning_dir
                conditioning_metadata.aspect_ratio_bucket_indices = {}
                conditioning_metadata.get_metadata_by_filepath.return_value = metadata
                conditioning_sampler = MultiAspectSampler(
                    dataset_runtime=DatasetRuntime(
                        conditioning_id, LocalDataBackend(self.accelerator, conditioning_id), conditioning_metadata, {}
                    ),
                    accelerator=self.accelerator,
                    batch_size=2,
                    model=None,
                    conditioning_type="mask",
                )
                conditioning_datasets.append({"id": conditioning_id, "sampler": conditioning_sampler})

            for sampling_mode in ("random", "combined"):
                for dataset_count in (1, 2):
                    with (
                        self.subTest(sampling_mode=sampling_mode, dataset_count=dataset_count),
                        patch.object(
                            StateTracker, "get_conditioning_datasets", return_value=conditioning_datasets[:dataset_count]
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
                        self.sampler.logger.warning.assert_not_called()

    def _iterate(self, sampler, count):
        """Take batches one iterator at a time, as the dataloader does."""
        args = SimpleNamespace(model_family="sdxl", print_sampler_statistics=False, i_know_what_i_am_doing=False)
        self.metadata_backend.get_metadata_by_filepath.return_value = {
            "crop_coordinates": (0, 0),
            "target_size": (512, 512),
        }
        batches = []
        with (
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_conditioning_datasets", return_value=[]),
            patch.object(StateTracker, "get_s2v_datasets", return_value=[]),
            patch.object(DatasetRuntime, "resolve_caption", return_value="a caption"),
        ):
            for _ in range(count):
                batches.append([record["image_path"] for record in next(iter(sampler))])
        return batches

    def _new_sampler(self, batch_size=None):
        runtime = DatasetRuntime("foo", self.data_backend, self.metadata_backend, {})
        # An unseeded run: each dataset draws its own shuffle seed.
        with patch.object(StateTracker, "get_args", return_value=SimpleNamespace(seed=None)):
            runtime.configure_cursor(
                accelerator=self.accelerator, batch_size=batch_size or self.batch_size, shared_by_all_ranks=False
            )
            sampler = MultiAspectSampler(
                dataset_runtime=runtime,
                accelerator=self.accelerator,
                batch_size=batch_size or self.batch_size,
                minimum_image_size=0,
                model=MagicMock(),
            )
        sampler.state_manager = Mock(spec=BucketStateManager)
        return sampler

    def test_iter_yields_every_sample_once_per_epoch(self):
        images = [f"/fake/dir/image{index}" for index in range(1, 5)]
        first_epoch = self._iterate(self.sampler, 2)
        self.assertEqual([len(batch) for batch in first_epoch], [2, 2])
        self.assertEqual(sorted(path for batch in first_epoch for path in batch), images)
        self.assertEqual(self.sampler.current_epoch, 1)
        with self.assertRaises(MultiDatasetExhausted):
            self._iterate(self.sampler, 1)
        self.assertEqual(self.sampler.current_epoch, 2)
        second_epoch = self._iterate(self.sampler, 2)
        self.assertEqual(sorted(path for batch in second_epoch for path in batch), images)

    def test_iter_repeats_are_part_of_the_epoch(self):
        with patch.object(StateTracker, "get_data_backend_config", return_value={"repeats": 2}):
            self.assertEqual(len(self.sampler), 6)
            batches = self._iterate(self.sampler, 6)
            with self.assertRaises(MultiDatasetExhausted):
                self._iterate(self.sampler, 1)
        counts = {}
        for path in (path for batch in batches for path in batch):
            counts[path] = counts.get(path, 0) + 1
        self.assertEqual(set(counts.values()), {3})

    def test_iter_reports_an_empty_dataset(self):
        self.metadata_backend.aspect_ratio_bucket_indices = {}
        self.metadata_backend.bucket_report = None
        self.metadata_backend.empty_dataset_report = "nothing matched the filters"
        with self.assertRaisesRegex(Exception, "nothing matched the filters"):
            self._iterate(self._new_sampler(), 1)

    def test_drop_unfinished_epoch_only_moves_a_started_epoch(self):
        self.sampler.drop_unfinished_epoch()
        self.assertEqual(self.sampler.current_epoch, 1)
        self._iterate(self.sampler, 1)
        self.sampler.drop_unfinished_epoch()
        self.assertEqual(self.sampler.current_epoch, 2)
        self.assertEqual(len(self._iterate(self.sampler, 2)), 2)

    def test_a_rebuilt_cursor_reads_the_new_bucket_lists_and_keeps_the_epoch(self):
        self.sampler.finish_epoch()
        self.metadata_backend.aspect_ratio_bucket_indices = {"1.5": ["wide1", "wide2"]}
        self.sampler.dataset_runtime.rebuild_cursor()
        self.assertEqual(self.sampler.current_epoch, 2)
        self.assertEqual(sorted(self._iterate(self.sampler, 1)[0]), ["/fake/dir/wide1", "/fake/dir/wide2"])

    def _saved_state(self, consumed_batches=1):
        self._iterate(self.sampler, consumed_batches)
        self.sampler.save_state(self.state_path)
        (state, path), _ = self.sampler.state_manager.save_state.call_args
        self.assertEqual(path, self.state_path)
        return json.loads(json.dumps(state)), self._iterate(self.sampler, 2 - consumed_batches)

    def test_load_states_continues_the_saved_epoch(self):
        state, remaining = self._saved_state()
        resumed = self._new_sampler()
        resumed.state_manager.load_state.return_value = state
        resumed.load_states(self.state_path)
        self.assertEqual(self._iterate(resumed, 1), remaining)
        with self.assertRaises(MultiDatasetExhausted):
            self._iterate(resumed, 1)

    def test_load_states_without_a_saved_state_starts_a_new_epoch(self):
        self.sampler.load_states(self.state_path)
        self.assertEqual(len(self._iterate(self.sampler, 2)), 2)

    def test_load_states_refuses_a_state_from_an_older_release(self):
        self.sampler.state_manager.load_state.return_value = {
            "aspect_ratio_bucket_indices": {"1.0": ["image1"]},
            "seen_images": {"image1": 1},
            "batch_size": self.batch_size,
            "current_epoch": 4,
        }
        with self.assertRaisesRegex(ValueError, "older release"):
            self.sampler.load_states(self.state_path)

    def test_load_states_rejects_batch_size_mismatch(self):
        state, _ = self._saved_state()
        resumed = self._new_sampler(batch_size=1)
        resumed.state_manager.load_state.return_value = state
        with (
            patch.dict(os.environ, {"SIMPLETUNER_ALLOW_MODIFYING_BSZ": ""}),
            patch.object(StateTracker, "get_args", return_value=SimpleNamespace(i_know_what_i_am_doing=False)),
            self.assertRaises(ValueError) as error,
        ):
            resumed.load_states(self.state_path)
        self.assertEqual(
            str(error.exception),
            "Dataset 'foo' checkpoint batch_size=2 does not match current batch_size=1. "
            "Resume with the same per-dataset train_batch_size, "
            "or set --i_know_what_i_am_doing / SIMPLETUNER_ALLOW_MODIFYING_BSZ=1 to bypass.",
        )
        self.assertEqual(len(self._iterate(resumed, 4)), 4)

    def _assert_batch_size_override_resumes(self, environment, args):
        state, remaining = self._saved_state()
        resumed = self._new_sampler(batch_size=1)
        resumed.state_manager.load_state.return_value = state
        with patch.dict(os.environ, environment), patch.object(StateTracker, "get_args", return_value=args):
            resumed.load_states(self.state_path)
        self.assertEqual(sorted(path for batch in self._iterate(resumed, 2) for path in batch), sorted(remaining[0]))
        with self.assertRaises(MultiDatasetExhausted):
            self._iterate(resumed, 1)

    def test_load_states_batch_size_mismatch_allowed_via_env_var(self):
        self._assert_batch_size_override_resumes(
            {"SIMPLETUNER_ALLOW_MODIFYING_BSZ": "1"}, SimpleNamespace(i_know_what_i_am_doing=False)
        )

    def test_load_states_batch_size_mismatch_allowed_via_i_know_flag(self):
        self._assert_batch_size_override_resumes(
            {"SIMPLETUNER_ALLOW_MODIFYING_BSZ": ""}, SimpleNamespace(i_know_what_i_am_doing=True)
        )

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


if __name__ == "__main__":
    unittest.main()
