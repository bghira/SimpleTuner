import unittest
from PIL import Image
from simpletuner.helpers.multiaspect.image import (
    MultiaspectImage,
)  # Adjust import according to your project structure


class TestCropping(unittest.TestCase):
    def setUp(self):
        # Creating a sample image for testing
        self.sample_image = Image.new("RGB", (500, 300), "white")

    def test_crop_corner(self):
        target_width, target_height = 300, 200
        from simpletuner.helpers.image_manipulation.cropping import CornerCropping

        cropper = CornerCropping(self.sample_image)
        cropped_image, (top, left) = cropper.set_intermediary_size(
            target_width + 10, target_height + 10
        ).crop(target_width, target_height)

        # Check if cropped coordinates are within original image bounds
        self.assertTrue(0 <= left < self.sample_image.width)
        self.assertTrue(0 <= top < self.sample_image.height)
        self.assertTrue(left + target_width <= self.sample_image.width)
        self.assertTrue(top + target_height <= self.sample_image.height)
        self.assertEqual(cropped_image.size, (target_width, target_height))

    def test_crop_center(self):
        from simpletuner.helpers.image_manipulation.cropping import CenterCropping

        cropper = CenterCropping(self.sample_image)
        target_width, target_height = 300, 200
        cropper.set_intermediary_size(target_width + 10, target_height + 10)
        cropped_image, (left, top) = cropper.crop(target_width, target_height)

        # Similar checks as above
        self.assertTrue(0 <= left < self.sample_image.width)
        self.assertTrue(0 <= top < self.sample_image.height)
        self.assertTrue(left + target_width <= self.sample_image.width)
        self.assertTrue(top + target_height <= self.sample_image.height)
        self.assertEqual(cropped_image.size, (target_width, target_height))

    def test_crop_random(self):
        from simpletuner.helpers.image_manipulation.cropping import RandomCropping

        target_width, target_height = 300, 200
        cropped_image, (top, left) = (
            RandomCropping(self.sample_image)
            .set_intermediary_size(target_width + 10, target_height + 10)
            .crop(target_width, target_height)
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
