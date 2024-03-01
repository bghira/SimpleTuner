import unittest, random
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
        self.data_backend = MagicMock()
        self.metadata_backend = MagicMock()
        self.metadata_backend.resolution = 1.0  # Example resolution
        self.metadata_backend.resolution_type = "dimension"  # Example resolution type
        self.metadata_backend.meets_resolution_requirements.return_value = True

        # Mock image data to simulate reading from the backend
        self.image_path_str = "test_image.jpg"

        # Convert the test image to bytes
        image_bytes = BytesIO()
        self.test_image.save(image_bytes, format="JPEG")
        image_bytes.seek(0)  # Important: move back to the start of the BytesIO object

        self.mock_image_data = image_bytes.getvalue()
        self.data_backend.read.return_value = self.mock_image_data

    def test_correct_aspect_ratio_and_size_adjustment(self):
        # Mock MultiaspectImage.prepare_image to simulate resizing and cropping
        with patch(
            "helpers.multiaspect.image.MultiaspectImage.prepare_image",
            return_value=(self.test_image, (0, 0), 1.0),
        ) as mock_prepare_image:
            aspect_ratio_bucket_indices = {}
            metadata_updates = {}

            # Call the method under test
            result = MultiaspectImage.process_for_bucket(
                self.data_backend,
                self.metadata_backend,
                self.image_path_str,
                aspect_ratio_bucket_indices,
                aspect_ratio_rounding=3,
                metadata_updates=metadata_updates,
            )

            # Verify the image was processed as expected
            mock_prepare_image.assert_called_once()
            self.assertIn("aspect_ratio", metadata_updates[self.image_path_str])
            self.assertIn(
                self.image_path_str,
                result[
                    str(round(self.test_image.size[0] / self.test_image.size[1], 3))
                ],
            )

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
            prepared_img, crop_coordinates, aspect_ratio = (
                MultiaspectImage.prepare_image(self.test_image, self.resolution)
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

    def test_calculate_new_size_by_pixel_area(self):
        # Define test cases for 1.0 and 0.5 megapixels
        test_megapixels = [1.0, 0.5]
        # Number of random tests to perform
        num_random_tests = 100

        for mp in test_megapixels:
            for _ in range(num_random_tests):
                # Generate a random original width and height
                original_width = random.randint(100, 5000)
                original_height = random.randint(100, 5000)
                original_aspect_ratio = original_width / original_height

                # Calculate new size
                new_width, new_height, new_aspect_ratio = (
                    MultiaspectImage.calculate_new_size_by_pixel_area(
                        original_width, original_height, mp
                    )
                )

                # Calculate the resulting megapixels
                resulting_mp = (new_width * new_height) / 1e6

                # Check that the resulting image size is not below the specified megapixels
                self.assertTrue(
                    resulting_mp >= mp,
                    f"Resulting size {new_width}x{new_height} = {resulting_mp} MP is below the specified {mp} MP",
                )


if __name__ == "__main__":
    unittest.main()
