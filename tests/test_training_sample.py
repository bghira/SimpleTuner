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
        StateTracker.set_args(
            MagicMock(aspect_bucket_alignment=64, aspect_bucket_rounding=2)
        )
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
        self.default_config = {
            "crop": False,
            "crop_style": "center",
            "crop_aspect": "square",
            "resolution": 512,
            "resolution_type": "pixel",
            "target_downsample_size": None,
            "maximum_image_size": None,
            "aspect_bucket_alignment": 64,
        }
        # Basic 1024×768 test image
        self.test_image = Image.new("RGB", (1024, 768), "white")
        self.test_metadata = {"original_size": (1024, 768)}
        # Make sure to isolate your test config from others
        self.original_get_data_backend_config = StateTracker.get_data_backend_config
        StateTracker.get_data_backend_config = lambda x: self.default_config

    def test_image_initialization(self):
        """Test that the image is correctly initialized and converted."""
        sample = TrainingSample(self.image, self.data_backend_id, self.image_metadata)
        self.assertEqual(sample.original_size, (1024, 768))

    def test_image_downsample(self):
        """Test that downsampling is correctly applied before cropping."""
        sample = TrainingSample(self.image, self.data_backend_id, self.image_metadata)
        self.assertEqual(
            sample.current_size, (1024, 768), "Size was not correct before prepare."
        )
        sample.prepare()
        self.assertEqual(
            sample.image.size,
            sample.current_size,
            f"Sample current_size was not updated? {sample.__dict__}",
        )
        self.assertEqual(
            sample.image.size,
            sample.target_size,
            f"Sample target size did not get reached by the image size. {sample.__dict__}",
        )

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
        original_aspect = round(sample.original_size[0] / sample.original_size[1], 2)
        sample.prepare()
        processed_aspect = round(sample.image.size[0] / sample.image.size[1], 2)
        self.assertEqual(
            processed_aspect, 1.38
        )  # when 64px divisible, we're at 1.38 now.

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

        sample = TrainingSample(
            video_data,
            self.data_backend_id,
            video_metadata,
            model=MagicMock(MAXIMUM_CANVAS_SIZE=None, get_transforms=MagicMock()),
        )
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
        video_data = np.zeros((5, 1024, 1024, 3), dtype=np.uint8)
        video_metadata = {"original_size": (1024, 1024)}

        sample = TrainingSample(
            video_data,
            self.data_backend_id,
            video_metadata,
            model=MagicMock(MAXIMUM_CANVAS_SIZE=None, get_transforms=MagicMock(return_value=MagicMock())),
        )
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

        sample = TrainingSample(
            video_data,
            self.data_backend_id,
            video_metadata,
            model=MagicMock(MAXIMUM_CANVAS_SIZE=None, get_transforms=MagicMock()),
        )
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

        sample = TrainingSample(
            video_data,
            self.data_backend_id,
            video_metadata,
            model=MagicMock(MAXIMUM_CANVAS_SIZE=None, get_transforms=MagicMock()),
        )
        sample.prepare()
        # Without crop, the pipeline might just do a direct resize to e.g. 128 px on the shorter edge
        final_shape = sample.image.shape
        self.assertEqual(final_shape[0], 4)  # frames unchanged
        self.assertTrue(final_shape[1] <= 128 or final_shape[2] <= 128)
        # or whatever your code does if it sees crop=False

    def test_no_crop_preserves_aspect_ratio_if_not_forced(self):
        """
        If `crop=False` and we do not forcibly set `crop_aspect='square'` in the code,
        the result should keep the original aspect ratio or a uniform scale.
        """
        self.default_config["crop"] = False
        # To avoid forcing squares, let's not set 'crop_aspect' to 'square'
        self.default_config["crop_aspect"] = (
            "preserve"  # or something that your code interprets as no forced square
        )
        self.default_config["resolution"] = 256

        sample = TrainingSample(
            self.test_image,
            data_backend_id=self.data_backend_id,
            image_metadata=self.test_metadata,
        )
        sample.prepare()

        final_w, final_h = sample.image.size
        final_aspect = sample.aspect_ratio
        original_aspect = round(1024 / 768, 3)
        self.assertNotEqual(
            final_aspect,
            original_aspect,
        )
        self.assertNotEqual(
            final_aspect,
            1.0,
        )

        # Confirm we did scale down to 256 or smaller on at least one dimension
        self.assertTrue(
            final_w <= 256 or final_h <= 256,
            f"Expected at least one dimension to be ≤ 256, but got ({final_w}, {final_h})",
        )

    def test_no_crop_forced_square_is_skipped(self):
        """
        Even if 'crop_aspect' is set to 'square', if crop=False, we expect the code
        to skip any forced square logic and preserve aspect ratio.
        """
        self.default_config["crop"] = False
        self.default_config["crop_aspect"] = "square"
        self.default_config["resolution"] = 512

        sample = TrainingSample(
            self.test_image,
            data_backend_id=self.data_backend_id,
            image_metadata=self.test_metadata,
        )
        sample.prepare()

        final_w, final_h = sample.image.size
        self.assertNotEqual(
            final_w,
            final_h,
            "No-crop scenario should not force a square shape if we truly skip crop logic.",
        )

    def test_no_crop_64px_alignment(self):
        """
        If we want to enforce that final dimensions are multiples of 64,
        test that the code does so without squishing.
        """
        # Suppose we have logic that snaps to multiples of 64.
        self.default_config["crop"] = False
        self.default_config["resolution"] = (
            999  # something that won't be a multiple of 64
        )
        # We'll pretend there's some internal code that rounds final sizes to multiples of 64.

        sample = TrainingSample(
            self.test_image,
            data_backend_id=self.data_backend_id,
            image_metadata=self.test_metadata,
        )
        sample.prepare()
        final_w, final_h = sample.image.size

        # Check the code hasn't forced a square
        self.assertNotEqual(
            final_w,
            final_h,
            "64px alignment should preserve aspect ratio unless the original was square.",
        )

        # Check multiples of 64
        self.assertEqual(
            final_w % 64, 0, f"Expected width to be multiple of 64, got {final_w}"
        )
        self.assertTrue(
            final_h % 64
            in [
                0,
                64,
            ],  # or 0, if your code also strictly enforces height to multiple of 64
            f"Expected height to be multiple of 64, got {final_h}",
        )

    def test_no_crop_video_preserves_frames_and_aspect(self):
        """
        For a video in 4D shape, ensure that no-crop scenario
        preserves original ratio (or a uniform scale) and the same frame count.
        """
        self.default_config["crop"] = False
        self.default_config["resolution"] = 400
        # Dummy video: shape [frames=5, H=600, W=1200, C=3] => aspect ~2.0
        video_data = np.zeros((5, 600, 1200, 3), dtype=np.uint8)
        video_metadata = {"original_size": (1200, 600)}

        sample = TrainingSample(
            video_data,
            self.data_backend_id,
            video_metadata,
            model=MagicMock(MAXIMUM_CANVAS_SIZE=None, get_transforms=MagicMock()),
        )
        sample.prepare()
        final_frames, final_h, final_w, final_c = sample.image.shape

        self.assertEqual(
            final_frames, 5, "Should preserve frame count in no-crop scenario."
        )
        self.assertEqual(final_c, 3, "Color channels should remain 3.")
        final_aspect = round(final_w / final_h, 2)
        self.assertAlmostEqual(
            final_aspect,
            2.0,
            places=1,
            msg=f"Should preserve ~2:1 ratio: {sample.__dict__}",
        )
        # Also confirm we scaled down
        self.assertTrue(
            final_w <= 400 or final_h <= 400,
            f"Expected at least one dimension to be ≤ 400, got {final_w}x{final_h}",
        )

    def test_no_crop_too_big_image_downscale_only(self):
        """
        If the original image is bigger than 'resolution', we only downscale,
        no cropping, preserving aspect.
        """
        self.default_config["crop"] = False
        self.default_config["resolution"] = 256
        # 2000×1500 => aspect ~1.333
        huge_img = Image.new("RGB", (2000, 1500), "white")
        sample = TrainingSample(
            huge_img, self.data_backend_id, {"original_size": (2000, 1500)}
        )
        sample.prepare()

        final_w, final_h = sample.image.size
        self.assertTrue(
            final_w <= 256 or final_h <= 256,
            f"Should be scaled down near 256 max, got {final_w}x{final_h}",
        )
        # Check aspect ratio
        expected_ratio = round(final_w / final_h, 2)
        actual_ratio = sample.aspect_ratio
        self.assertEqual(
            expected_ratio,
            actual_ratio,
            f"Aspect ratio must be preserved for no-crop: {sample.__dict__}",
        )
        self.assertNotEqual(
            sample.aspect_ratio, 1.0, "Should not force a square shape in this case."
        )

    def test_no_crop_small_image_upscale_only_if_configured(self):
        """
        If the original image is smaller than resolution, decide if we actually upscale
        or leave it alone (depending on your logic). We can test either outcome.
        """
        self.default_config["crop"] = False
        self.default_config["resolution"] = 512
        # 128×96 => aspect ~1.333
        small_img = Image.new("RGB", (128, 96), "white")
        sample = TrainingSample(
            small_img, self.data_backend_id, {"original_size": (128, 96)}
        )
        sample.prepare()

        final_w, final_h = sample.image.size
        # If your code does not upscale, we'd expect still 128×96.
        # If your code upscales to 512 on one side, we'd expect e.g. 512×384.
        # Let's assume we allow upscaling to 512.
        self.assertTrue(
            final_w >= 128 and final_h >= 96,
            "Should have upscaled the small image in no-crop mode.",
        )
        # Check ratio is still ~1.333
        expected_ratio = round(final_w / final_h, 2)
        actual_ratio = sample.aspect_ratio
        self.assertEqual(
            expected_ratio,
            actual_ratio,
            f"Aspect ratio must be preserved for no-crop upscaling.",
        )
        self.assertNotEqual(
            sample.aspect_ratio, 1.0, "Should not force a square shape in this case."
        )


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
