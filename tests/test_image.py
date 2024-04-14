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
        self.assertEqual(
            MultiaspectImage.calculate_image_aspect_ratio((1920, 1080)), 1.78
        )
        self.assertEqual(
            MultiaspectImage.calculate_image_aspect_ratio((1080, 1920)), 0.56
        )

    def test_image_resize(self):
        """
        Test that images are resized to the expected dimensions.
        """
        # Create a mock image
        original_image = Image.new("RGB", (1920, 1080))

        # Define target resolutions and expected output sizes
        tests = [
            (1024, "pixel", (1824, 1024)),
            (1.0, "area", (1344, 768)),  # Assuming target is 1 megapixel
        ]

        for resolution, resolution_type, expected_size in tests:
            resized_image, _, _ = MultiaspectImage.prepare_image(
                resolution=resolution,
                image=original_image,
                resolution_type=resolution_type,
                id="test",
            )

            # Verify the size of the resized image
            self.assertEqual(resized_image.size, expected_size)

    def test_image_size_consistency(self):
        """
        Test that `prepare_image` returns consistent size for images with similar aspect ratios.
        """
        # Generate random input aspect ratios and resolutions:
        input_aspect_ratios = [random.uniform(0.5, 2.0) for _ in range(10)]
        # Sizes should follow the list of resolutions, with between 2-4 images in each aspect
        input_sizes = []
        for aspect_ratio in input_aspect_ratios:
            count = 0
            for resolution in range(5, 50, 5):
                count += 1
                width = resolution * 100
                height = int(width / aspect_ratio)
                input_sizes.append((width, height))

        # Sort into bucket dictionary using MultiaspectImage.calculate_image_aspect_ratio
        input_sizes_dict = {}
        for size in input_sizes:
            aspect_ratio = size[0] / size[1]
            if aspect_ratio not in input_sizes_dict:
                input_sizes_dict[aspect_ratio] = []
            input_sizes_dict[aspect_ratio].append(size)

        resolutions = range(
            5, 20, 5
        )  # Using a simplified resolution from the logs for the test
        for aspect_ratio in set(input_sizes_dict.keys()):
            for resolution in resolutions:
                resolution = resolution / 10  # Convert to megapixels
                output_sizes = []
                new_aspect_ratios = []
                for size in input_sizes_dict[aspect_ratio]:
                    should_use_real_image = random.choice([True, False])
                    image = (
                        Image.new("RGB", size) if should_use_real_image else None
                    )  # Creating a dummy PIL image with the given size
                    image_metadata = (
                        None if should_use_real_image else {"original_size": size}
                    )
                    function_result, _, new_aspect_ratio = (
                        MultiaspectImage.prepare_image(
                            image=image,
                            image_metadata=image_metadata,
                            resolution=resolution,
                            resolution_type="area",
                        )
                    )
                    if hasattr(function_result, "size"):
                        output_size = function_result.size
                    else:
                        output_size = function_result
                    output_sizes.append(output_size)
                    new_aspect_ratios.append(new_aspect_ratio)

                # Check if all output sizes are the same, indicating consistent resizing/cropping
                self.assertTrue(
                    all(size == output_sizes[0] for size in output_sizes),
                    f"Output sizes are not consistent for {resolution} MP",
                )
                self.assertTrue(
                    all(size == new_aspect_ratios[0] for size in new_aspect_ratios),
                    f"Output sizes are not consistent for {resolution} MP",
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
                MultiaspectImage.prepare_image(
                    image=self.test_image, resolution=self.resolution
                )
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
                        original_aspect_ratio, mp
                    )
                )

                # Calculate the resulting megapixels
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
        ]
        expected_size = (
            896,
            1216,
        )  # Expected final size for all test cases based on a fixed aspect ratio

        for W, H, megapixels in test_cases:
            W_final, H_final, new_aspect_ratio = (
                MultiaspectImage.calculate_new_size_by_pixel_area((W / H), megapixels)
            )
            self.assertEqual(
                (W_final, H_final), expected_size, f"Failed for original size {W}x{H}"
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

        for W, H, megapixels in test_cases:
            W_final, H_final, _ = MultiaspectImage.calculate_new_size_by_pixel_area(
                MultiaspectImage.calculate_image_aspect_ratio((W, H)), megapixels
            )
            self.assertEqual(
                (W_final, H_final), expected_size, f"Failed for original size {W}x{H}"
            )


if __name__ == "__main__":
    unittest.main()
