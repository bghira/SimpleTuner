import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
from simpletuner.helpers.training.state_tracker import StateTracker


class TestAudioSampler(unittest.TestCase):
    def setUp(self):
        # Mock StateTracker
        self.patcher_tracker = patch("simpletuner.helpers.multiaspect.sampler.StateTracker")
        self.mock_tracker = self.patcher_tracker.start()
        self.mock_tracker.get_args.return_value.print_sampler_statistics = False
        self.mock_tracker.get_args.return_value.model_family = "ace_step"
        self.mock_tracker.get_data_backend_config.return_value = {"repeats": 0}
        self.mock_tracker.get_conditioning_datasets.return_value = []

        # Mock Metadata Backend
        self.mock_metadata = MagicMock()
        self.mock_metadata.id = "test_backend"
        self.mock_metadata.aspect_ratio_bucket_indices = {"10s": ["a.wav", "b.wav"], "20s": ["c.wav"]}
        self.mock_metadata.instance_data_dir = ""

        def get_metadata(path):
            return {}

        self.mock_metadata.get_metadata_by_filepath.side_effect = get_metadata

        # Mock Data Backend
        self.mock_data = MagicMock()
        self.mock_data.id = "test_backend"

        # Mock Model
        self.mock_model = MagicMock()

        # Mock Accelerator
        self.mock_accelerator = MagicMock()
        self.mock_accelerator.num_processes = 1
        self.mock_accelerator.process_index = 0

        # Mock PromptHandler to avoid magic prompt logic
        self.patcher_prompt = patch("simpletuner.helpers.prompts.PromptHandler.magic_prompt")
        self.mock_prompt = self.patcher_prompt.start()
        self.mock_prompt.return_value = "test caption"

        # Mock accelerate logger
        self.patcher_logger = patch("simpletuner.helpers.multiaspect.sampler.get_logger")
        self.mock_logger = self.patcher_logger.start()
        self.mock_logger.return_value = MagicMock()

    def tearDown(self):
        self.patcher_tracker.stop()
        self.patcher_prompt.stop()
        self.patcher_logger.stop()

    def _runtime(self):
        runtime = DatasetRuntime("test_backend", self.mock_data, self.mock_metadata, {"dataset_type": "audio"})
        # The one-clip bucket only fills a batch because short buckets may be padded.
        args = SimpleNamespace(seed=1, allow_dataset_oversubscription=True)
        with patch.object(StateTracker, "get_args", return_value=args):
            runtime.configure_cursor(accelerator=self.mock_accelerator, batch_size=2, shared_by_all_ranks=False)
        return runtime

    def test_sampler_iterates_audio_buckets(self):
        sampler = MultiAspectSampler(
            dataset_runtime=self._runtime(),
            model=self.mock_model,
            accelerator=self.mock_accelerator,
            batch_size=2,
        )

        # Bucket "10s" has a.wav and b.wav; the cursor pads "20s", which only has c.wav, to fill its batch.

        iterator = iter(sampler)

        # First batch
        batch1 = next(iterator)
        self.assertEqual(len(batch1), 2)
        # Should be from same bucket
        paths1 = [x["image_path"] for x in batch1]
        # The cursor's seeded schedule decides which bucket comes first.
        if "a.wav" in paths1:
            self.assertIn("b.wav", paths1)
        else:
            self.assertIn("c.wav", paths1)

        # Second batch
        batch2 = next(iterator)
        self.assertEqual(len(batch2), 2)

        paths2 = [x["image_path"] for x in batch2]
        # If batch1 was 10s, batch2 is 20s.
        if "a.wav" in paths1:
            self.assertIn("c.wav", paths2)
            # Should contain c.wav twice or similar if filling up
            self.assertTrue(all(p == "c.wav" for p in paths2))

    def test_sampler_bucket_exhaustion(self):
        # Test that sampler correctly resets when all exhausted
        sampler = MultiAspectSampler(
            dataset_runtime=self._runtime(),
            model=self.mock_model,
            accelerator=self.mock_accelerator,
            batch_size=2,
        )

        iterator = iter(sampler)
        # One batch from each bucket makes the epoch; the third draw ends it.

        from simpletuner.helpers.training.exceptions import MultiDatasetExhausted

        self.assertEqual([len(next(iterator)) for _ in range(2)], [2, 2])
        with self.assertRaises(MultiDatasetExhausted):
            next(iterator)

    def test_missing_reloaded_metadata_does_not_require_image_crop_coordinates(self):
        self.mock_tracker.get_args.return_value.model_family = "minimaxmusic"
        self.mock_metadata.get_metadata_by_filepath.side_effect = lambda _: None
        sampler = MultiAspectSampler(
            dataset_runtime=self._runtime(),
            model=self.mock_model,
            accelerator=self.mock_accelerator,
            batch_size=1,
        )

        samples = sampler._validate_and_yield_images_from_samples(["1.wav"], "10s")

        self.assertEqual(sampler.sample_type_str, "audio")
        self.assertEqual(sampler.sample_type_strs, "audio")
        self.assertEqual(samples[0]["image_path"], "1.wav")
        self.assertEqual(samples[0]["dataset_type"], "audio")
        self.assertEqual(samples[0]["instance_prompt_text"], "test caption")


if __name__ == "__main__":
    unittest.main()
