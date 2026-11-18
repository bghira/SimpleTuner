import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from simpletuner.helpers.image_manipulation.batched_training_samples import BatchedTrainingSamples


class BatchedTrainingSamplesVideoResizeTests(unittest.TestCase):
    def test_grouped_image_preparation_keeps_grayscale_and_cmyk_payloads(self):
        images = [
            Image.new("L", (6, 8), 37),
            Image.new("LA", (6, 8), (10, 20)),
            Image.new("CMYK", (6, 8), (10, 20, 30, 40)),
        ]
        geometry = {"target_size": (6, 8)}
        grouped = {"0.75": [(str(index), image, geometry) for index, image in enumerate(images)]}

        results = BatchedTrainingSamples().process_aspect_grouped_images(grouped)

        self.assertEqual([ref for ref, _, _ in results], ["0", "1", "2"])
        for image, (_, pixels, _) in zip(images, results, strict=True):
            np.testing.assert_array_equal(pixels, np.asarray(image.convert("RGB")))

    def test_grouped_processing_preserves_image_pixels_and_resizes_video(self):
        batches = BatchedTrainingSamples()
        pixels = np.random.default_rng(42).integers(0, 256, (8, 6, 3), dtype=np.uint8)
        video = np.stack([pixels[::2], pixels[::2]])
        geometry = {"target_size": [6, 8]}
        grouped = {"0.75": [("image", pixels, geometry), ("video", video, geometry)]}

        results = batches.process_aspect_grouped_images(grouped)

        self.assertEqual([path for path, _, _ in results], ["image", "video"])
        np.testing.assert_array_equal(results[0][1], pixels)
        expected_video = batches.batch_resize_videos([video], [(6, 8)])[0]
        np.testing.assert_array_equal(results[1][1], expected_video)
        self.assertEqual(results[1][1].shape, (2, 8, 6, 3))
        self.assertEqual(geometry["target_size"], [6, 8])
        self.assertEqual(results[1][2]["target_size"], (6, 8))

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
