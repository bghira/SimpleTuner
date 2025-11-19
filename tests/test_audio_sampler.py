import unittest
from unittest.mock import MagicMock, patch

from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler


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
        self.mock_metadata.seen_images = {}
        self.mock_metadata.is_seen = lambda x: x in self.mock_metadata.seen_images
        self.mock_metadata.mark_batch_as_seen = lambda batch: [
            self.mock_metadata.seen_images.update({x: True}) for x in batch
        ]
        self.mock_metadata.reset_seen_images = lambda: self.mock_metadata.seen_images.clear()
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

        # Mock PromptHandler to avoid magic prompt logic
        self.patcher_prompt = patch("simpletuner.helpers.multiaspect.sampler.PromptHandler")
        self.mock_prompt = self.patcher_prompt.start()
        self.mock_prompt.magic_prompt.return_value = "test caption"

        # Mock accelerate logger
        self.patcher_logger = patch("simpletuner.helpers.multiaspect.sampler.get_logger")
        self.mock_logger = self.patcher_logger.start()
        self.mock_logger.return_value = MagicMock()

    def tearDown(self):
        self.patcher_tracker.stop()
        self.patcher_prompt.stop()
        self.patcher_logger.stop()

    def test_sampler_iterates_audio_buckets(self):
        sampler = MultiAspectSampler(
            id="test_backend",
            metadata_backend=self.mock_metadata,
            data_backend=self.mock_data,
            model=self.mock_model,
            accelerator=self.mock_accelerator,
            batch_size=2,
            dataset_type="audio",
        )

        # Bucket "10s" has 2 items. Bucket "20s" has 1 item.
        # Batch size 2.
        # Expect:
        # 1. Batch of 2 from "10s" (a.wav, b.wav)
        # 2. Batch of 2 from "20s" (c.wav + duplicate/exhausted fill?)
        #    Wait, _handle_bucket_with_insufficient_images moves insufficient buckets to exhausted.
        #    If "20s" has 1 item < batch_size 2, it might be moved to exhausted.
        #    Then _yield_n_from_exhausted_bucket will fill it.

        iterator = iter(sampler)

        # First batch
        batch1 = next(iterator)
        self.assertEqual(len(batch1), 2)
        # Should be from same bucket
        paths1 = [x["image_path"] for x in batch1]
        # We don't know which bucket comes first (random or sequential init), but usually sequential [0].
        # Let's check if they are consistent.
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
            id="test_backend",
            metadata_backend=self.mock_metadata,
            data_backend=self.mock_data,
            model=self.mock_model,
            accelerator=self.mock_accelerator,
            batch_size=2,
            dataset_type="audio",
        )

        iterator = iter(sampler)
        # Total items = 3. Batch size = 2.
        # Epoch 1:
        # Batch 1: 2 items (e.g. 10s bucket)
        # Batch 2: 2 items (e.g. 20s bucket, 1 item recycled)
        # End of epoch.

        # We expect MultiDatasetExhausted exception eventually if we keep iterating,
        # OR it resets buckets and continues (depending on _reset_buckets raise_exhaustion_signal=True default)

        from simpletuner.helpers.training.exceptions import MultiDatasetExhausted

        with self.assertRaises(MultiDatasetExhausted):
            next(iterator)
            next(iterator)
            next(iterator)  # Should raise here or before


if __name__ == "__main__":
    unittest.main()
