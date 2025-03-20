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

    # -----------------------
    # New Tests for Video Data
    # -----------------------

    def test_video_initialization_4d(self):
        """
        Test that a 4D NumPy array (frames, height, width, channels)
        is recognized and processed similarly to images.
        """
        # Create a dummy "video" with shape [frames, H, W, C] = [10, 720, 1280, 3]
        video_data = np.zeros((10, 720, 1280, 3), dtype=np.uint8)
        video_metadata = {"original_size": (1280, 720)}

        sample = TrainingSample(video_data, self.data_backend_id, video_metadata)
        self.assertEqual(sample.original_size, (1280, 720))
        # Confirm it doesn't crash
        sample.prepare()
        # After crop to square=512, we might see shape [frames, 512, 512, 3] or smaller
        self.assertTrue(isinstance(sample.image, np.ndarray))
        self.assertTrue(sample.image.shape[-1] == 3)  # last dim still color channels

    def test_video_initialization_5d_fails(self):
        """
        Test that a 5D NumPy array (batch, frames, channels, height, width)
        fails since it is invalid.
        """
        # Create a dummy "video" with shape [B, F, C, H, W] = [2, 10, 3, 720, 1280]
        video_data = np.zeros((2, 10, 3, 720, 1280), dtype=np.uint8)
        video_metadata = {"original_size": (1280, 720)}

        with self.assertRaises(ValueError):
            TrainingSample(video_data, self.data_backend_id, video_metadata)

    def test_video_square_crop(self):
        """
        Test that a 'square' aspect ratio truly yields a square shape for 4D video data.
        """
        # Create dummy video: [frames=5, H=600, W=800, C=3]
        video_data = np.zeros((5, 600, 800, 3), dtype=np.uint8)
        video_metadata = {"original_size": (800, 600)}

        sample = TrainingSample(video_data, self.data_backend_id, video_metadata)
        sample.prepare()
        # The shape should reflect a final square dimension <= 512
        final_shape = sample.image.shape
        # E.g. [5, newH, newW, 3]
        self.assertEqual(final_shape[-1], 3)
        self.assertEqual(
            final_shape[1], final_shape[2], "Video should be square in H/W"
        )

    def test_video_random_crop(self):
        """
        Test that random cropping works for 4D video data.
        """
        # Overwrite config to use random cropping
        StateTracker.get_data_backend_config = MagicMock(
            return_value={
                "crop": True,
                "crop_style": "random",
                "crop_aspect": "square",
                "resolution": 256,  # smaller for quick test
                "resolution_type": "pixel",
            }
        )
        # shape [frames=3, H=300, W=400, C=3]
        video_data = np.ones((3, 300, 400, 3), dtype=np.uint8)
        video_metadata = {"original_size": (400, 300)}

        sample = TrainingSample(video_data, self.data_backend_id, video_metadata)
        sample.prepare()

        # The final shape should be [3, 256, 256, 3] or smaller
        self.assertEqual(sample.image.shape[0], 3)
        self.assertEqual(sample.image.shape[-1], 3)
        self.assertTrue(sample.image.shape[1] == sample.image.shape[2])

    def test_video_no_crop(self):
        """
        Ensure that when crop=False, a video is simply resized or left alone,
        but does not do a random or center crop.
        """
        # Overwrite config to disable cropping
        StateTracker.get_data_backend_config = MagicMock(
            return_value={
                "crop": False,
                "crop_style": "center",
                "resolution": 128,
                "resolution_type": "pixel",
            }
        )
        video_data = np.zeros((4, 240, 320, 3), dtype=np.uint8)
        video_metadata = {"original_size": (320, 240)}

        sample = TrainingSample(video_data, self.data_backend_id, video_metadata)
        sample.prepare()
        # Without crop, the pipeline might just do a direct resize to e.g. 128 px on the shorter edge
        final_shape = sample.image.shape
        self.assertEqual(final_shape[0], 4)  # frames unchanged
        self.assertTrue(final_shape[1] <= 128 or final_shape[2] <= 128)
        # or whatever your code does if it sees crop=False


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
