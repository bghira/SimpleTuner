import unittest
from unittest.mock import patch

import numpy as np

from simpletuner.helpers.image_manipulation.batched_training_samples import BatchedTrainingSamples


class BatchedTrainingSamplesVideoResizeTests(unittest.TestCase):
    def test_target_sizes_are_normalized_to_tuples_for_videos(self):
        batches = BatchedTrainingSamples()
        videos = [np.zeros((2, 4, 6, 3), dtype=np.uint8)]
        target_sizes = [[8, 10]]

        captured = {}

        def _fake_batch_resize_videos(videos_arg, target_sizes_arg):
            captured["videos"] = videos_arg
            captured["target_sizes"] = target_sizes_arg
            return videos_arg

        with patch("simpletuner.helpers.image_manipulation.batched_training_samples.ts.batch_resize_videos") as mock_resize:
            mock_resize.side_effect = _fake_batch_resize_videos
            result = batches.batch_resize_videos(videos, target_sizes)

        self.assertIn("target_sizes", captured)
        self.assertIsInstance(captured["target_sizes"], tuple)
        self.assertEqual(captured["target_sizes"], ((8, 10),))
        self.assertEqual(result, videos)


if __name__ == "__main__":
    unittest.main()
