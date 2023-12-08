import unittest
from PIL import Image
from helpers.multiaspect.image import (
    MultiaspectImage,
)  # Adjust import according to your project structure


class TestMultiaspectImage(unittest.TestCase):
    def setUp(self):
        # Creating a sample image for testing
        self.sample_image = Image.new("RGB", (500, 300), "white")

    def test_crop_corner(self):
        target_width, target_height = 300, 200
        cropped_image, (left, top) = MultiaspectImage._crop_corner(
            self.sample_image, target_width, target_height
        )

        # Check if cropped coordinates are within original image bounds
        self.assertTrue(0 <= left < self.sample_image.width)
        self.assertTrue(0 <= top < self.sample_image.height)
        self.assertTrue(left + target_width <= self.sample_image.width)
        self.assertTrue(top + target_height <= self.sample_image.height)
        self.assertEqual(cropped_image.size, (target_width, target_height))

    def test_crop_center(self):
        target_width, target_height = 300, 200
        cropped_image, (left, top) = MultiaspectImage._crop_center(
            self.sample_image, target_width, target_height
        )

        # Similar checks as above
        self.assertTrue(0 <= left < self.sample_image.width)
        self.assertTrue(0 <= top < self.sample_image.height)
        self.assertTrue(left + target_width <= self.sample_image.width)
        self.assertTrue(top + target_height <= self.sample_image.height)
        self.assertEqual(cropped_image.size, (target_width, target_height))

    def test_crop_random(self):
        target_width, target_height = 300, 200
        cropped_image, (left, top) = MultiaspectImage._crop_random(
            self.sample_image, target_width, target_height
        )

        # Similar checks as above
        self.assertTrue(0 <= left < self.sample_image.width)
        self.assertTrue(0 <= top < self.sample_image.height)
        self.assertTrue(left + target_width <= self.sample_image.width)
        self.assertTrue(top + target_height <= self.sample_image.height)
        self.assertEqual(cropped_image.size, (target_width, target_height))

    # Add additional tests for other methods as necessary


if __name__ == "__main__":
    unittest.main()
