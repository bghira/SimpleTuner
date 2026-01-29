import inspect
import unittest

import numpy as np
from PIL import Image

from simpletuner.helpers.multiaspect.image import MultiaspectImage  # Adjust import according to your project structure


class TestCropping(unittest.TestCase):
    def setUp(self):
        # Creating a sample image for testing
        self.sample_image = Image.new("RGB", (500, 300), "white")
        # Create numpy array version
        self.sample_array = np.ones((300, 500, 3), dtype=np.uint8) * 255
        # Create video (4D array) version - 10 frames
        self.sample_video = np.ones((10, 300, 500, 3), dtype=np.uint8) * 255

    def test_crop_corner(self):
        target_width, target_height = 300, 200
        from simpletuner.helpers.image_manipulation.cropping import CornerCropping

        cropper = CornerCropping(self.sample_image)
        cropped_image, (top, left) = cropper.set_intermediary_size(target_width + 10, target_height + 10).crop(
            target_width, target_height
        )

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


class TestFaceCropping(unittest.TestCase):
    def setUp(self):
        # Creating sample images for testing (no faces)
        self.sample_image = Image.new("RGB", (500, 300), "white")
        self.sample_array = np.ones((300, 500, 3), dtype=np.uint8) * 255
        self.sample_video = np.ones((10, 300, 500, 3), dtype=np.uint8) * 255
        self.target_width = 300
        self.target_height = 200

    def test_face_cropping_signature_matches_parent(self):
        """FaceCropping.crop() should have the same signature as RandomCropping.crop()"""
        from simpletuner.helpers.image_manipulation.cropping import FaceCropping, RandomCropping

        fc_params = list(inspect.signature(FaceCropping.crop).parameters.keys())
        rc_params = list(inspect.signature(RandomCropping.crop).parameters.keys())
        self.assertEqual(fc_params, rc_params)

    def test_face_cropping_pil_no_faces_fallback(self):
        """FaceCropping with PIL image and no faces should fall back to random cropping"""
        from simpletuner.helpers.image_manipulation.cropping import FaceCropping

        cropper = FaceCropping(self.sample_image)
        cropper.set_intermediary_size(500, 300)
        cropped_image, (top, left) = cropper.crop(self.target_width, self.target_height)

        # Should return a valid cropped image
        self.assertIsNotNone(cropped_image)
        self.assertIsInstance(cropped_image, Image.Image)
        self.assertEqual(cropped_image.size, (self.target_width, self.target_height))

        # Coordinates should be within bounds
        self.assertTrue(0 <= left < self.sample_image.width)
        self.assertTrue(0 <= top < self.sample_image.height)

    def test_face_cropping_numpy_no_faces_fallback(self):
        """FaceCropping with numpy array and no faces should fall back to random cropping"""
        from simpletuner.helpers.image_manipulation.cropping import FaceCropping

        cropper = FaceCropping(self.sample_array)
        cropper.set_intermediary_size(500, 300)
        cropped_array, (top, left) = cropper.crop(self.target_width, self.target_height)

        # Should return a valid cropped array
        self.assertIsNotNone(cropped_array)
        self.assertIsInstance(cropped_array, np.ndarray)
        self.assertEqual(cropped_array.shape, (self.target_height, self.target_width, 3))

        # Coordinates should be within bounds
        self.assertTrue(0 <= left < self.sample_array.shape[1])
        self.assertTrue(0 <= top < self.sample_array.shape[0])

    def test_face_cropping_video_no_faces_fallback(self):
        """FaceCropping with video (4D array) and no faces should fall back to random cropping"""
        from simpletuner.helpers.image_manipulation.cropping import FaceCropping

        cropper = FaceCropping(self.sample_video)
        cropper.set_intermediary_size(500, 300)
        cropped_video, (top, left) = cropper.crop(self.target_width, self.target_height)

        # Should return a valid cropped video
        self.assertIsNotNone(cropped_video)
        self.assertIsInstance(cropped_video, np.ndarray)
        # Video shape: (num_frames, height, width, channels)
        self.assertEqual(cropped_video.shape, (10, self.target_height, self.target_width, 3))

        # Coordinates should be within bounds
        self.assertTrue(0 <= left < self.sample_video.shape[2])
        self.assertTrue(0 <= top < self.sample_video.shape[1])

    def test_face_cropping_uses_correct_api(self):
        """FaceCropping should use detect_multi_scale (snake_case), not detectMultiScale"""
        import os

        cropping_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "simpletuner",
            "helpers",
            "image_manipulation",
            "cropping.py",
        )
        with open(cropping_path) as f:
            content = f.read()

        self.assertNotIn("detectMultiScale", content, "Should not use camelCase detectMultiScale")
        self.assertIn("detect_multi_scale", content, "Should use snake_case detect_multi_scale")


if __name__ == "__main__":
    unittest.main()
