import unittest, random, logging, os

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", logging.INFO))
from unittest.mock import patch
from unittest.mock import Mock, MagicMock

try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image
from io import BytesIO
from simpletuner.helpers.multiaspect.image import MultiaspectImage
from simpletuner.helpers.training.state_tracker import StateTracker
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

    def test_calculate_new_size_by_pixel_edge(self):
        # Define test cases for 1.0 and 0.5 megapixels
        test_edge_lengths = [1024, 512, 256, 64]
        # Number of random tests to perform
        num_random_tests = 1000
        with patch(
            "simpletuner.helpers.training.state_tracker.StateTracker.get_args"
        ) as mock_args:
            for edge_length in test_edge_lengths:
                mock_args.return_value = Mock(
                    resolution_type="pixel",
                    resolution=self.resolution,
                    crop_style="random",
                    aspect_bucket_rounding=2,
                    aspect_bucket_alignment=64 if edge_length > 64 else 8,
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
                        f"Final target size {new_width}x{new_height} = {min(new_width, new_height)} px is below the specified {edge_length} px from original size {original_width}x{original_height}, alignment {64 if edge_length > 256 else 8}",
                    )
                    # Check that the resulting image size is not below the specified minimum edge length.
                    new_width, new_height = intermediary_size
                    self.assertTrue(
                        min(new_width, new_height) >= edge_length,
                        f"Intermediary size {new_width}x{new_height} = {min(new_width, new_height)} px is below the specified {edge_length} px from original size {original_width}x{original_height}, alignment {64 if edge_length > 256 else 8}",
                    )
                    # Check that the intermediary size is larger than the target size.
                    self.assertTrue(
                        intermediary_size >= reformed_size,
                        f"Intermediary size is less than reformed size: {intermediary_size} < {reformed_size} (original size: {original_width}x{original_height})",
                    )

    def test_calculate_batch_size_by_pixel_edge(self):
        test_edge_lengths = [1024, 768, 512, 256, 64]
        num_images_per_batch = 100
        aspect_ratios = [
            1.5,
            1.0,
            0.67,
            0.76,
            1.33,
            2.0,
            8.0,
            1.78,
        ]  # Example fixed aspect ratio for all test cases

        with patch(
            "simpletuner.helpers.training.state_tracker.StateTracker.get_args"
        ) as mock_args:
            for edge_length in test_edge_lengths:
                for fixed_aspect_ratio in aspect_ratios:
                    aspect_bucket_alignment = 64 if edge_length > 64 else 8
                    edge_length += 512
                    logger.debug(
                        f"Using aspect bucket alignment: {aspect_bucket_alignment} for {edge_length}px edge length, test aspect ratio: {fixed_aspect_ratio}"
                    )
                    mock_args.return_value = Mock(
                        resolution_type="pixel",
                        resolution=edge_length,
                        crop_style="random",
                        aspect_bucket_rounding=2,
                        aspect_bucket_alignment=aspect_bucket_alignment,
                    )

                    # Generate a batch of original sizes correctly adhering to the fixed aspect ratio
                    original_sizes = []
                    for _ in range(num_images_per_batch):
                        # Generate a random height and calculate the corresponding width
                        height = max(
                            edge_length, random.randint(edge_length, edge_length * 50)
                        )
                        if fixed_aspect_ratio >= 1:
                            height = random.randint(edge_length, edge_length * 10)
                            width = int(height * fixed_aspect_ratio)
                        else:
                            width = random.randint(edge_length, edge_length * 10)
                            height = int(width / fixed_aspect_ratio)
                        # Check if size is large enough once pixel-adjusted
                        self.assertTrue(
                            min(width, height) >= edge_length,
                            f"Original size {width}x{height} is below the specified {edge_length} px",
                        )
                        original_sizes.append((width, height))

                    # Ensure aspect ratios are correctly calculated
                    for width, height in original_sizes:
                        calculated_aspect_ratio = (
                            MultiaspectImage.calculate_image_aspect_ratio(
                                width / height
                            )
                        )
                        self.assertEqual(
                            calculated_aspect_ratio,
                            fixed_aspect_ratio,
                            msg=f"Generated size {width}x{height} has aspect ratio {calculated_aspect_ratio}, expected {fixed_aspect_ratio}",
                        )

                    first_aspect_ratio = None
                    first_transformed_aspect_ratio = None
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
                            first_aspect_ratio = (
                                MultiaspectImage.calculate_image_aspect_ratio(
                                    intermediary_size
                                )
                            )
                        if first_transformed_aspect_ratio is None:
                            first_transformed_aspect_ratio = new_aspect_ratio
                        if (
                            new_aspect_ratio != first_transformed_aspect_ratio
                            or MultiaspectImage.calculate_image_aspect_ratio(
                                intermediary_size
                            )
                            != fixed_aspect_ratio
                        ):
                            logger.debug("####")
                            logger.debug(
                                f"-> First aspect ratio: {first_aspect_ratio}, first transformed aspect ratio: {first_transformed_aspect_ratio}, first reformed size: {first_reformed_size}"
                            )
                            logger.debug(
                                f"-> Original size: {original_width}x{original_height} ({MultiaspectImage.calculate_image_aspect_ratio((original_width, original_height))})"
                            )
                            logger.debug(
                                f"-> Reformed size: {reformed_size} ({MultiaspectImage.calculate_image_aspect_ratio(reformed_size)})"
                            )
                            logger.debug(
                                f"-> {'*' if MultiaspectImage.calculate_image_aspect_ratio(intermediary_size) != fixed_aspect_ratio else ''}Intermediary size: {intermediary_size} ({MultiaspectImage.calculate_image_aspect_ratio(intermediary_size)} vs {fixed_aspect_ratio})"
                            )
                            logger.debug(
                                f"-> {'*' if new_aspect_ratio != first_transformed_aspect_ratio else ''}New aspect ratio: {new_aspect_ratio} vs {first_transformed_aspect_ratio}"
                            )
                            logger.debug("####")
                        self.assertEqual(
                            MultiaspectImage.calculate_image_aspect_ratio(
                                intermediary_size
                            ),
                            fixed_aspect_ratio,
                        )
                        self.assertEqual(first_reformed_size, reformed_size)
                        self.assertEqual(
                            MultiaspectImage.calculate_image_aspect_ratio(
                                reformed_size
                            ),
                            first_transformed_aspect_ratio,
                        )

    def test_calculate_new_size_by_pixel_area(self):
        # Define test cases for 1.0 and 0.5 megapixels
        test_megapixels = [1.0, 0.5]
        # Number of random tests to perform
        num_random_tests = 100
        with (
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker.get_args"
            ) as mock_args,
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker._load_from_disk"
            ) as load_from_disk_mock,
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker._save_to_disk"
            ) as save_to_disk_mock,
        ):
            load_from_disk_mock.return_value = {}
            save_to_disk_mock.return_value = True
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
                    original_width = random.randint(512, 2500)
                    original_height = random.randint(512, 2500)
                    original_aspect_ratio = original_width / original_height

                    # Calculate new size
                    target_size, intermediary_size, new_aspect_ratio = (
                        MultiaspectImage.calculate_new_size_by_pixel_area(
                            original_aspect_ratio, mp, (original_width, original_height)
                        )
                    )

                    # Calculate the resulting megapixels
                    target_width, target_height = target_size
                    intermediary_width, intermediary_height = intermediary_size
                    self.assertGreaterEqual(
                        intermediary_width,
                        target_width,
                        (
                            f"Final width {target_width} is greater than the intermediary {intermediary_size}"
                            f", original size {original_width}x{original_height} and target megapixels {mp}"
                            f" for aspect ratio {original_aspect_ratio}"
                        ),
                    )
                    self.assertGreaterEqual(
                        intermediary_height,
                        target_height,
                        f"Final height {target_height} is greater than the intermediary {intermediary_size}",
                    )
                    self.assertAlmostEqual(
                        MultiaspectImage.calculate_image_aspect_ratio(
                            (original_width, original_height)
                        ),
                        MultiaspectImage.calculate_image_aspect_ratio(
                            intermediary_size
                        ),
                        delta=0.02,
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

        with (
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker.get_args"
            ) as mock_args,
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker._load_from_disk"
            ) as load_from_disk_mock,
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker._save_to_disk"
            ) as save_to_disk_mock,
        ):
            load_from_disk_mock.return_value = {}
            save_to_disk_mock.return_value = True
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
        with patch(
            "simpletuner.helpers.training.state_tracker.StateTracker.get_args"
        ) as mock_args:
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

    def test_limit_canvas_size(self):
        """
        Test the limit_canvas_size method to ensure it properly reduces canvas dimensions
        when they exceed the maximum allowed size.
        """
        with patch(
            "simpletuner.helpers.training.state_tracker.StateTracker.get_args"
        ) as mock_args:
            aspect_bucket_alignment = 64
            mock_args.return_value = Mock(
                aspect_bucket_alignment=aspect_bucket_alignment
            )

            # Test case 1: Canvas size already within limit - no adjustment needed
            result = MultiaspectImage.limit_canvas_size(1024, 1024, 1024 * 1024)
            self.assertEqual(result["width"], 1024)
            self.assertEqual(result["height"], 1024)
            self.assertEqual(result["canvas_size"], 1024 * 1024)

            # Test case 2: Canvas size exceeds limit - adjust larger dimension only
            # 2048 * 1024 = 2,097,152 > 2,000,000
            # Should reduce width by aspect_bucket_alignment: (2048-aspect_bucket_alignment) * 1024 = 2,031,616
            # Then reduce height by aspect_bucket_alignment: (1984) * (1024-aspect_bucket_alignment) = 1984 * 960 < 2,000,000
            result = MultiaspectImage.limit_canvas_size(2048, 1024, 2000000)
            self.assertEqual(result["width"], 2048 - aspect_bucket_alignment)  # 1984
            self.assertEqual(result["height"], 1024 - aspect_bucket_alignment)  # 960
            self.assertEqual(result["canvas_size"], 1984 * 960)
            self.assertLess(result["canvas_size"], 2000000)

            # Test case 3: Canvas size still exceeds after first adjustment - adjust both dimensions
            # 2048 * 2048 = 4,194,304 > 2,000,000
            # First reduce larger dimension: (2048-aspect_bucket_alignment) * 2048 = 4,063,232 > 2,000,000
            # Then reduce other dimension: (2048-aspect_bucket_alignment) * (2048-aspect_bucket_alignment) = 3,936,256 > 2,000,000
            # But method only does one adjustment per dimension
            result = MultiaspectImage.limit_canvas_size(2048, 2048, 2000000)
            self.assertEqual(result["width"], 2048 - aspect_bucket_alignment)  # 1984
            self.assertEqual(result["height"], 2048 - aspect_bucket_alignment)  # 1984
            self.assertEqual(result["canvas_size"], 1984 * 1984)

            # Test case 4: Portrait orientation (height > width)
            # 1024 * 2048 = 2,097,152 > 2,000,000
            # Should reduce both width and height by aspect_bucket_alignment if necessary: (1024-aspect_bucket_alignment) * (2048-aspect_bucket_alignment) = 960 * 1984 < 2,000,000
            result = MultiaspectImage.limit_canvas_size(1024, 2048, 2000000)
            self.assertEqual(result["width"], 1024 - aspect_bucket_alignment)  # 960
            self.assertEqual(result["height"], 2048 - aspect_bucket_alignment)  # 1984
            self.assertEqual(result["canvas_size"], 960 * 1984)
            self.assertLess(result["canvas_size"], 2000000)

            # Test case 5: Different alignment value
            aspect_bucket_alignment = 32
            mock_args.return_value = Mock(
                aspect_bucket_alignment=aspect_bucket_alignment
            )
            # 1536 * 1536 = 2,359,296 > 2,000,000
            # First reduce: (1536-aspect_bucket_alignment) * 1536 = 2,310,144 > 2,000,000
            # Second reduce: (1536-aspect_bucket_alignment) * (1536-aspect_bucket_alignment) = 2,262,016 > 2,000,000
            result = MultiaspectImage.limit_canvas_size(1536, 1536, 2000000)
            self.assertEqual(result["width"], 1536 - aspect_bucket_alignment)  # 1504
            self.assertEqual(result["height"], 1536 - aspect_bucket_alignment)  # 1504
            self.assertEqual(result["canvas_size"], 1504 * 1504)

            # Test case 6: Small alignment, large canvas that needs both adjustments
            mock_args.return_value = Mock(aspect_bucket_alignment=8)
            # 1600 * 1280 = 2,048,000 > 2,000,000
            # First reduce width: (1600-8) * 1280 = 2,037,760 > 2,000,000
            # Then reduce height: (1600-8) * (1280-8) = 2,027,584 > 2,000,000
            result = MultiaspectImage.limit_canvas_size(1600, 1280, 2000000)
            self.assertEqual(result["width"], 1600 - 8)  # 1592
            self.assertEqual(result["height"], 1280 - 8)  # 1272
            self.assertEqual(result["canvas_size"], 1592 * 1272)

            # Test case 7: Edge case - exactly at limit after one adjustment
            mock_args.return_value = Mock(aspect_bucket_alignment=100)
            # Find dimensions where one adjustment brings us exactly to or just under limit
            # 1100 * 1000 = 1,100,000 > 1,000,000
            # (1100-100) * 1000 = 1,000,000 = 1,000,000 (exactly at limit)
            result = MultiaspectImage.limit_canvas_size(1100, 1000, 1000000)
            self.assertEqual(result["width"], 1000)
            self.assertEqual(result["height"], 1000)
            self.assertEqual(result["canvas_size"], 1000000)

            # Test case 8: Very small max_size requiring both adjustments
            mock_args.return_value = Mock(aspect_bucket_alignment=64)
            # 512 * 512 = 262,144 > 200,000
            # (512-64) * 512 = 229,376 > 200,000
            # (512-64) * (512-64) = 200,704 > 200,000 (still over, but method stops here)
            result = MultiaspectImage.limit_canvas_size(512, 512, 200000)
            self.assertEqual(result["width"], 448)
            self.assertEqual(result["height"], 448)
            self.assertEqual(result["canvas_size"], 448 * 448)


if __name__ == "__main__":
    unittest.main()
