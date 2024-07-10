import unittest
from PIL import Image
import numpy as np
from helpers.image_manipulation.training_sample import TrainingSample
from helpers.training.state_tracker import StateTracker
from unittest.mock import MagicMock


class TestTrainingSample(unittest.TestCase):

    def setUp(self):
        # Create a simple image for testing
        self.image = Image.new("RGB", (1024, 768), "white")
        self.data_backend_id = "test_backend"
        self.image_metadata = {"original_size": (1024, 768)}

        # Assume StateTracker and other helpers are correctly set up to return meaningful values
        StateTracker.get_args = MagicMock()
        StateTracker.get_args.return_value = MagicMock(aspect_bucket_alignment=8)
        StateTracker.get_data_backend_config = MagicMock(
            return_value={
                "crop": True,
                "crop_style": "center",
                "crop_aspect": "square",
                "resolution": 512,
                "resolution_type": "pixel",
                "target_downsample_size": 768,
                "maximum_image_size": 1024,
                "aspect_bucket_alignment": 8,
            }
        )

    def test_image_initialization(self):
        """Test that the image is correctly initialized and converted."""
        sample = TrainingSample(self.image, self.data_backend_id, self.image_metadata)
        self.assertEqual(sample.original_size, (1024, 768))

    def test_image_downsample(self):
        """Test that downsampling is correctly applied before cropping."""
        sample = TrainingSample(self.image, self.data_backend_id, self.image_metadata)
        sample.prepare()
        self.assertLessEqual(
            sample.image.size[0], 512
        )  # Assuming downsample before crop applies

    def test_no_crop(self):
        """Test handling when cropping is disabled."""
        StateTracker.get_data_backend_config = lambda x: {
            "crop": False,
            "crop_style": "random",
            "resolution": 512,
            "resolution_type": "pixel",
        }
        sample = TrainingSample(self.image, self.data_backend_id, self.image_metadata)
        original_size = sample.image.size
        sample.prepare()
        self.assertNotEqual(sample.image.size, original_size)  # Ensure resizing occurs

    def test_crop_coordinates(self):
        """Test that cropping returns correct coordinates."""
        sample = TrainingSample(self.image, self.data_backend_id, self.image_metadata)
        sample.prepare()
        self.assertIsNotNone(sample.crop_coordinates)  # Crop coordinates should be set

    def test_aspect_ratio_square_up(self):
        """Test that the aspect ratio is preserved after processing."""
        sample = TrainingSample(self.image, self.data_backend_id, self.image_metadata)
        original_aspect = sample.original_size[0] / sample.original_size[1]
        sample.prepare()
        processed_aspect = sample.image.size[0] / sample.image.size[1]
        self.assertEqual(processed_aspect, 1.0)

    def test_return_tensor(self):
        """Test tensor conversion if requested."""
        sample = TrainingSample(self.image, self.data_backend_id, self.image_metadata)
        prepared_sample = sample.prepare(return_tensor=True)
        # Check if returned object is a tensor (mock or check type if actual tensor transformation is applied)
        self.assertTrue(
            isinstance(prepared_sample.aspect_ratio, float)
        )  # Placeholder check


# Helper mock classes and functions
class MockCropper:
    def __init__(self, image, image_metadata):
        self.image = image
        self.image_metadata = image_metadata

    def crop(self, width, height):
        return self.image.crop((0, 0, width, height)), (0, 0, width, height)


def mock_resize_helper(aspect_ratio, resolution):
    # Simulates resizing logic
    width, height = resolution, int(resolution / aspect_ratio)
    return width, height, aspect_ratio


if __name__ == "__main__":
    unittest.main()
