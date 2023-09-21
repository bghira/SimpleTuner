import unittest
from unittest.mock import patch
from unittest.mock import Mock, MagicMock
from PIL import Image
from io import BytesIO
from helpers.multiaspect.image import MultiaspectImage
from tests.helpers.data import MockDataBackend


class TestMultiaspectImage(unittest.TestCase):
    def setUp(self):
        self.data_backend = MockDataBackend()
        self.image_path_str = "dummy_image_path"
        self.aspect_ratio_bucket_indices = {}
        self.resolution = 128

    @patch("logging.Logger")
    def test_process_for_bucket_valid_image(self, mock_logger):
        # Test with a valid image
        with patch("PIL.Image.open") as mock_image_open:
            # Create a blank canvas:
            mock_image = Image.new(
                size=(16, 8),
                mode="RGB",
            )
            mock_image_open.return_value.__enter__.return_value = mock_image

            result = MultiaspectImage.process_for_bucket(
                self.data_backend, self.image_path_str, self.aspect_ratio_bucket_indices
            )
            self.assertEqual(result, {"2.0": ["dummy_image_path"]})

    def test_process_for_bucket_invalid_image(self):
        # Test with an invalid image (e.g., backend read fails)
        with self.assertLogs("MultiaspectImage", level="ERROR") as cm:
            MultiaspectImage.process_for_bucket(
                None, self.image_path_str, self.aspect_ratio_bucket_indices
            )

    def test_prepare_image_valid(self):
        # Test with a valid image
        img = Image.new("RGB", (60, 30), color="red")
        prepared_img = MultiaspectImage.prepare_image(img, self.resolution)
        self.assertIsInstance(prepared_img, Image.Image)

    def test_prepare_image_invalid(self):
        # Test with an invalid image
        with self.assertRaises(Exception):
            MultiaspectImage.prepare_image(None, self.resolution)

    def test_resize_for_condition_image_valid(self):
        # Test with a valid image
        img = Image.new("RGB", (60, 30), color="red")
        resized_img = MultiaspectImage.resize_by_pixel_edge(img, self.resolution)
        self.assertIsInstance(resized_img, Image.Image)

    def test_resize_for_condition_image_invalid(self):
        # Test with an invalid image
        with self.assertRaises(Exception):
            MultiaspectImage.resize_by_pixel_edge(None, self.resolution)


if __name__ == "__main__":
    unittest.main()
