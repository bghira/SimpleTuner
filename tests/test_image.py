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

    def test_aspect_ratio_calculation(self):
        """
        Test that the aspect ratio calculation returns expected results.
        """
        StateTracker.set_args(MagicMock(aspect_bucket_rounding=2))
        self.assertEqual(
            MultiaspectImage.calculate_image_aspect_ratio((1920, 1080)), 1.78
        )
        self.assertEqual(
            MultiaspectImage.calculate_image_aspect_ratio((1080, 1920)), 0.56
        )

    def test_resize_for_condition_image_valid(self):
        resized_img = MultiaspectImage._resize_image(
            self.test_image, self.resolution, self.resolution
        )
        self.assertIsInstance(resized_img, Image.Image)

    def test_resize_for_condition_image_invalid(self):
        with self.assertRaises(Exception):
            MultiaspectImage._resize_image(None, self.resolution)

    def test_calculate_new_size_by_pixel_edge(self):
        # Define test cases for 1.0 and 0.5 megapixels
        test_edge_lengths = [1024, 512, 256, 64]
        # Number of random tests to perform
        num_random_tests = 100000
        with patch("helpers.training.state_tracker.StateTracker.get_args") as mock_args:
            for edge_length in test_edge_lengths:
                mock_args.return_value = Mock(
                    resolution_type="pixel",
                    resolution=self.resolution,
                    crop_style="random",
                    aspect_bucket_rounding=2,
                    aspect_bucket_alignment=64 if edge_length > 256 else 8,
                )
                for _ in range(num_random_tests):
                    # Generate a random original width and height
                    original_width = random.randint(edge_length, edge_length * 50)
                    original_height = random.randint(edge_length, edge_length * 50)
                    original_aspect_ratio = original_width / original_height

                    # Calculate new size
                    reformed_size, intermediary_size, new_aspect_ratio = (
                        MultiaspectImage.calculate_new_size_by_pixel_edge(
                            original_aspect_ratio,
                            edge_length,
                            (original_width, original_height),
                        )
                    )

                    # Calculate the resulting megapixels
                    new_width, new_height = reformed_size
                    # Check that the resulting image size is not below the specified minimum edge length.
                    self.assertTrue(
                        min(new_width, new_height) >= edge_length,
                        f"Final target size {new_width}x{new_height} = {min(new_width, new_height)} px is below the specified {edge_length} px from original size {original_width}x{original_height}",
                    )
                    # Check that the resulting image size is not below the specified minimum edge length.
                    new_width, new_height = intermediary_size
                    self.assertTrue(
                        min(new_width, new_height) >= edge_length,
                        f"Intermediary size {new_width}x{new_height} = {min(new_width, new_height)} px is below the specified {edge_length} px from original size {original_width}x{original_height}",
                    )
                    # Check that the intermediary size is larger than the target size.
                    self.assertTrue(
                        intermediary_size >= reformed_size,
                        f"Intermediary size is less than reformed size: {intermediary_size} < {reformed_size}",
                    )
                    # Check that the target size is equal or smaller than the original size
                    self.assertTrue(
                        reformed_size <= (original_width, original_height),
                        f"Reformed size is bigger than original size: {reformed_size} > {(original_width, original_height)}",
                    )

    def test_calculate_batch_size_by_pixel_edge(self):
        test_edge_lengths = [1024, 512, 256, 64]
        num_images_per_batch = 100
        fixed_aspect_ratio = 1.5  # Example fixed aspect ratio for all test cases

        with patch("helpers.training.state_tracker.StateTracker.get_args") as mock_args:
            for edge_length in test_edge_lengths:
                mock_args.return_value = Mock(
                    resolution_type="pixel",
                    resolution=edge_length,
                    crop_style="random",
                    aspect_bucket_rounding=2,
                    aspect_bucket_alignment=64 if edge_length > 256 else 8,
                )

                # Generate a batch of original sizes correctly adhering to the fixed aspect ratio
                original_sizes = []
                for _ in range(num_images_per_batch):
                    # Generate a random height and calculate the corresponding width
                    height = random.randint(edge_length, edge_length * 50)
                    width = int(height * fixed_aspect_ratio)
                    original_sizes.append((width, height))
                # print(f"Original sizes: {original_sizes}")
                # print(f"Aspect ratios: {[w/h for w, h in original_sizes]}")

                # Ensure aspect ratios are correctly calculated
                for width, height in original_sizes:
                    calculated_aspect_ratio = width / height
                    self.assertAlmostEqual(
                        calculated_aspect_ratio,
                        fixed_aspect_ratio,
                        msg=f"Generated size {width}x{height} has aspect ratio {calculated_aspect_ratio}, expected {fixed_aspect_ratio}",
                        places=1,  # Allow some small tolerance for floating-point precision
                    )

                first_aspect_ratio = None
                first_reformed_size = None
                for original_width, original_height in original_sizes:
                    reformed_size, intermediary_size, new_aspect_ratio = (
                        MultiaspectImage.calculate_new_size_by_pixel_edge(
                            fixed_aspect_ratio,
                            edge_length,
                            (original_width, original_height),
                        )
                    )
                    if first_reformed_size is None:
                        first_reformed_size = reformed_size
                    if first_aspect_ratio is None:
                        first_aspect_ratio = new_aspect_ratio
                    self.assertEqual(new_aspect_ratio, fixed_aspect_ratio)
                    self.assertEqual(first_reformed_size, reformed_size)
                    self.assertEqual(first_aspect_ratio, new_aspect_ratio)

    def test_calculate_new_size_by_pixel_area(self):
        # Define test cases for 1.0 and 0.5 megapixels
        test_megapixels = [1.0, 0.5]
        # Number of random tests to perform
        num_random_tests = 100
        with patch("helpers.training.state_tracker.StateTracker.get_args") as mock_args:
            mock_args.return_value = Mock(
                resolution_type="pixel",
                resolution=self.resolution,
                crop_style="random",
                aspect_bucket_rounding=2,
                aspect_bucket_alignment=64,
            )

            for mp in test_megapixels:
                for _ in range(num_random_tests):
                    # Generate a random original width and height
                    original_width = random.randint(100, 5000)
                    original_height = random.randint(100, 5000)
                    original_aspect_ratio = original_width / original_height

                    # Calculate new size
                    reformed_size, intermediary_size, new_aspect_ratio = (
                        MultiaspectImage.calculate_new_size_by_pixel_area(
                            original_aspect_ratio, mp, (original_width, original_height)
                        )
                    )

                    # Calculate the resulting megapixels
                    new_width, new_height = reformed_size
                    resulting_mp = (new_width * new_height) / 1e6

                    # Check that the resulting image size is not below the specified megapixels
                    self.assertTrue(
                        resulting_mp >= mp,
                        f"Resulting size {new_width}x{new_height} = {resulting_mp} MP is below the specified {mp} MP",
                    )

    def test_calculate_new_size_by_pixel_area_uniformity(self):
        # Example input resolutions and expected output
        test_cases = [
            (
                3911,
                5476,
                1.0,
            ),  # Original resolution and target megapixels, ar=0.714
            (
                4539,
                6527,
                1.0,
            ),  # Original resolution and target megapixels, ar=0.695
            (
                832,
                1216,
                1.0,
            ),
        ]
        expected_size = (
            832,
            1216,
        )  # Expected final size for all test cases based on a fixed aspect ratio

        with patch("helpers.training.state_tracker.StateTracker.get_args") as mock_args:
            mock_args.return_value = Mock(
                resolution_type="pixel",
                resolution=self.resolution,
                crop_style="random",
                aspect_bucket_rounding=2,
                aspect_bucket_alignment=64,
            )
            for W, H, megapixels in test_cases:
                final_size, intermediary_size, new_aspect_ratio = (
                    MultiaspectImage.calculate_new_size_by_pixel_area(
                        MultiaspectImage.calculate_image_aspect_ratio(W / H),
                        megapixels,
                        (W, H),
                    )
                )
                W_final, H_final = final_size
                self.assertEqual(
                    (W_final, H_final),
                    expected_size,
                    f"Failed for original size {W}x{H}",
                )
                self.assertNotEqual(
                    new_aspect_ratio,
                    (W / H),
                    f"Failed for original size {W}x{H}",
                )

    def test_calculate_new_size_by_pixel_area_squares(self):
        # Example input resolutions and expected output
        test_cases = [
            (
                4000,
                4000,
                1.0,
            ),  # Original resolution and target megapixels, ar=0.714
            (
                2000,
                2000,
                1.0,
            ),  # Original resolution and target megapixels, ar=0.695
        ]
        expected_size = (
            1024,
            1024,
        )  # Expected final size for all test cases based on a fixed aspect ratio
        with patch("helpers.training.state_tracker.StateTracker.get_args") as mock_args:
            mock_args.return_value = Mock(
                resolution_type="pixel",
                resolution=self.resolution,
                crop_style="random",
                aspect_bucket_rounding=2,
                aspect_bucket_alignment=64,
            )

            for W, H, megapixels in test_cases:
                final_size, intermediary_size, _ = (
                    MultiaspectImage.calculate_new_size_by_pixel_area(
                        MultiaspectImage.calculate_image_aspect_ratio((W, H)),
                        megapixels,
                        (W, H),
                    )
                )
                W_final, H_final = final_size
                self.assertEqual(
                    (W_final, H_final),
                    expected_size,
                    f"Failed for original size {W}x{H}",
                )


if __name__ == "__main__":
    unittest.main()
