import unittest
from hashlib import sha256
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

from simpletuner.helpers.caching.vae import VAECache
from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
from simpletuner.helpers.training.state_tracker import StateTracker


class TestVaeCache(unittest.TestCase):
    def test_filename_mapping(self):
        # Test cases
        test_cases = [
            # 0 Filepath ends with .pt (no change expected in the path)
            {"image_path": "/data/image1.pt", "cache_path": "/data/image1.pt"},
            # 1 Normal filepath
            {"image_path": "/data/image1.png", "cache_path": "cache/image1.pt"},
            # 2, 3 Nested subdirectories
            {
                "image_path": "/data/subdir1/subdir2/image2.jpg",
                "cache_path": "cache/subdir1/subdir2/image2.pt",
            },
            {
                "image_path": "data/subdir1/subdir2/image2.jpg",
                "cache_path": "cache/subdir1/subdir2/image2.pt",
                "instance_dir": "data",
            },
            # 4 No instance_data_dir, direct cache dir placement
            {
                "image_path": "/anotherdir/image3.png",
                "cache_path": "cache/image3.pt",
                "instance_dir": None,
            },
            # 5 Instance data directory is None
            {
                "image_path": "/data/image4.png",
                "cache_path": "cache/image4.pt",
                "instance_dir": None,
            },
            # 6 Filepath in root directory
            {"image_path": "/image5.png", "cache_path": "cache/image5.pt"},
            # 7 Hash filenames enabled
            {
                "image_path": "/data/image6.png",
                "cache_path": "cache/" + sha256("image6".encode()).hexdigest() + ".pt",
                "should_hash": True,
            },
            # 8 Invalid cache_dir
            {"image_path": "/data/image7.png", "cache_path": "cache/image7.pt"},
        ]

        # Running test cases
        for i, test_case in enumerate(test_cases, 1):
            filepath = test_case["image_path"]
            # expected = os.path.abspath(test_case['cache_path'])
            expected = test_case["cache_path"]
            cache_dir = test_case.get("cache_dir", "cache")
            instance_dir = test_case.get("instance_dir", "/data")
            should_hash = test_case.get("should_hash", False)
            vae_cache = VAECache(
                id="test-cache",
                vae=None,
                accelerator=None,
                metadata_backend=None,
                image_data_backend=None,
                hash_filenames=should_hash,
                instance_data_dir=instance_dir,
                cache_dir=cache_dir,
                model=MagicMock(),
            )
            generated = vae_cache.generate_vae_cache_filename(filepath)[0]
            self.assertEqual(generated, expected, f"Test {i} failed: {generated} != {expected}")


if __name__ == "__main__":
    unittest.main()
