import unittest, os, logging
from math import ceil

try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image
from unittest import skip
from unittest.mock import Mock, MagicMock, patch
from helpers.multiaspect.sampler import MultiAspectSampler
from helpers.metadata.backends.discovery import DiscoveryMetadataBackend
from helpers.multiaspect.state import BucketStateManager
from tests.helpers.data import MockDataBackend
from accelerate import PartialState
from PIL import Image


class TestMultiAspectSampler(unittest.TestCase):
    def setUp(self):
        self.process_state = PartialState()
        self.accelerator = MagicMock()
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

    def test_save_state(self):
        with patch.object(self.sampler.state_manager, "save_state") as mock_save_state:
            self.sampler.save_state(self.state_path)
        mock_save_state.assert_called_once()

    def test_load_buckets(self):
        buckets = self.sampler.load_buckets()
        self.assertEqual(buckets, ["1.0"])

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

    @skip("Infinite Loop Boulevard")
    def test_iter_yields_correct_batches(self):
        # Add about 100 images to the metadata_backend
        all_images = ["image" + str(i) for i in range(100)]

        self.metadata_backend.aspect_ratio_bucket_indices = {"1.0": all_images}
        self.metadata_backend.buckets = ["1.0"]
        self.sampler._get_image_files = MagicMock(return_value=all_images)
        self.sampler._get_unseen_images = MagicMock(return_value=all_images)
        self.data_backend.exists = MagicMock(return_value=True)

        # Loop over __iter__ about 100 times:
        batches = []
        batch_size = 4
        for _ in range(ceil(len(all_images) / batch_size)):
            # extract batch_item from generator:
            with patch(
                "PIL.Image.open", return_value=MagicMock(spec=Image.Image)
            ) as mock_image:
                logging.warning("mock_image: %s", mock_image)
                batch_item = next(self.sampler.__iter__())
            self.assertIn(batch_item, all_images)
            batches.append(batch_item)
        self.assertEqual(len(batches), len(all_images))

    @skip("Infinite Loop Boulevard")
    def test_iter_handles_small_images(self):
        # Mocking the _validate_and_yield_images_from_samples method to simulate small images
        def mock_validate_and_yield_images_from_samples(samples, bucket):
            # Simulate that 'image2' is too small and thus not returned
            return [img for img in samples if img != "image2"]

        self.metadata_backend.aspect_ratio_bucket_indices = {
            "1.0": ["image1", "image2", "image3", "image4"]
        }
        self.sampler._validate_and_yield_images_from_samples = (
            mock_validate_and_yield_images_from_samples
        )

        batches = list(self.sampler)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches, [["image1", "image3"], ["image4"]])

    @skip("Currently broken test.")
    def test_iter_handles_incorrect_aspect_ratios_with_real_logic(self):
        # Create mock image files with different sizes using PIL
        img_paths = [
            "/tmp/image1.jpg",
            "/tmp/image2.jpg",
            "/tmp/incorrect_image.jpg",
            "/tmp/image4.jpg",
        ]

        img1 = Image.new("RGB", (100, 100), color="red")
        img1.save(img_paths[0])

        img2 = Image.new("RGB", (100, 100), color="green")
        img2.save(img_paths[1])

        img3 = Image.new(
            "RGB", (50, 100), color="blue"
        )  # This image has a different size
        img3.save(img_paths[2])

        img4 = Image.new("RGB", (100, 100), color="yellow")
        img4.save(img_paths[3])

        self.metadata_backend.aspect_ratio_bucket_indices = {"1.0": img_paths}

        # Collect batches by iterating over the generator
        batches = [next(self.sampler.__iter__()) for _ in range(len(img_paths))]
        # Ensure that all batches have consistent image sizes
        # We retrieve the size using PIL for validation
        first_img_size = Image.open(batches[0]).size
        self.assertNotIn(img_paths[2], batches)
        self.assertTrue(
            all(Image.open(img_path).size == first_img_size for img_path in batches)
        )

        # Clean up the mock images
        for img_path in img_paths:
            os.remove(img_path)


if __name__ == "__main__":
    unittest.main()
