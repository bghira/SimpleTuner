import unittest
from unittest.mock import patch
from unittest.mock import Mock, MagicMock
from PIL import Image
from io import BytesIO
from helpers.multiaspect.image import MultiaspectImage
from helpers.training.state_tracker import StateTracker
from tests.helpers.data import MockDataBackend


class TestMultiaspectImage(unittest.TestCase):
    def setUp(self):
        self.resolution = 128
        self.test_image = Image.new("RGB", (512, 256), color="red")

    def test_crop_corner(self):
        cropped_image, _ = MultiaspectImage._crop_corner(
            self.test_image, self.resolution, self.resolution
        )
        self.assertEqual(cropped_image.size, (self.resolution, self.resolution))

    def test_crop_center(self):
        cropped_image, _ = MultiaspectImage._crop_center(
            self.test_image, self.resolution, self.resolution
        )
        self.assertEqual(cropped_image.size, (self.resolution, self.resolution))

    def test_crop_random(self):
        cropped_image, _ = MultiaspectImage._crop_random(
            self.test_image, self.resolution, self.resolution
        )
        self.assertEqual(cropped_image.size, (self.resolution, self.resolution))

    def test_prepare_image_valid(self):
        with patch("helpers.training.state_tracker.StateTracker.get_args") as mock_args:
            mock_args.return_value = Mock(
                resolution_type="pixel", resolution=self.resolution, crop_style="random"
            )
            prepared_img, _ = MultiaspectImage.prepare_image(
                self.test_image, self.resolution
            )
        self.assertIsInstance(prepared_img, Image.Image)

    def test_prepare_image_invalid(self):
        with self.assertRaises(Exception):
            MultiaspectImage.prepare_image(None, self.resolution)

    def test_resize_for_condition_image_valid(self):
        resized_img = MultiaspectImage._resize_image(
            self.test_image, self.resolution, self.resolution
        )
        self.assertIsInstance(resized_img, Image.Image)

    def test_resize_for_condition_image_invalid(self):
        with self.assertRaises(Exception):
            MultiaspectImage._resize_image(None, self.resolution)


if __name__ == "__main__":
    unittest.main()
