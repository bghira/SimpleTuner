import unittest
import pandas as pd
from unittest.mock import patch, Mock, MagicMock
try:
    import pillow_jxl
except ModuleNotFoundError:
    pass
from PIL import Image
from pathlib import Path
from helpers.multiaspect.dataset import MultiAspectDataset
from helpers.metadata.backends.discovery import DiscoveryMetadataBackend
from helpers.data_backend.base import BaseDataBackend
from helpers.data_backend.factory import check_column_values


class TestMultiAspectDataset(unittest.TestCase):
    def setUp(self):
        self.instance_data_dir = "/some/fake/path"
        self.accelerator = Mock()
        self.metadata_backend = Mock(spec=DiscoveryMetadataBackend)
        self.metadata_backend.__len__ = Mock(return_value=10)
        self.image_metadata = {
            "image_path": "fake_image_path",
            "original_size": (16, 8),
            "crop_coordinates": (0, 0),
            "target_size": (16, 8),
            "aspect_ratio": 1.0,
            "luminance": 0.5,
        }
        self.metadata_backend.get_metadata_by_filepath = Mock(
            return_value=self.image_metadata
        )
        self.data_backend = Mock(spec=BaseDataBackend)
        self.image_path = "fake_image_path"
        # Mock the Path.exists method to return True
        with patch("pathlib.Path.exists", return_value=True):
            self.dataset = MultiAspectDataset(
                id="foo",
                datasets=[range(10)],
            )

    def test_init_invalid_instance_data_dir(self):
        MultiAspectDataset(
            id="foo",
            datasets=[range(10)],
        )

    def test_len(self):
        self.metadata_backend.__len__.return_value = 10
        self.assertEqual(len(self.dataset), 10)

    def test_getitem_valid_image(self):
        mock_image_data = b"fake_image_data"
        self.data_backend.read.return_value = mock_image_data

        with patch("PIL.Image.open") as mock_image_open:
            # Create a blank canvas:
            mock_image = Image.new(mode="RGB", size=(16, 8))
            mock_image_open.return_value = mock_image
            target = tuple(
                [
                    {
                        "image_path": self.image_path,
                        "image_data": mock_image,
                        "instance_prompt_text": "fake_prompt_text",
                        "original_size": (16, 8),
                        "target_size": (16, 8),
                        "aspect_ratio": 1.0,
                        "luminance": 0.5,
                    }
                ]
            )
            examples = self.dataset.__getitem__(target)
        # Grab the size of the first image:
        examples = examples["training_samples"]
        first_size = examples[0]["original_size"]
        # Are all sizes the same?
        for example in examples:
            self.assertIsNotNone(example)
            self.assertEqual(example["original_size"], first_size)
            self.assertEqual(example["image_path"], self.image_path)

    def test_getitem_invalid_image(self):
        self.data_backend.read.side_effect = Exception("Some error")

        with self.assertRaises(Exception):
            with self.assertLogs("MultiAspectDataset", level="ERROR") as cm:
                self.dataset.__getitem__(self.image_metadata)


class TestDataBackendFactory(unittest.TestCase):
    def test_all_null(self):
        column_data = pd.Series([None, None, None])
        with self.assertRaises(ValueError) as context:
            check_column_values(column_data, "test_column", "test_file.parquet")
        self.assertIn("contains only null values", str(context.exception))

    def test_arrays_with_nulls(self):
        column_data = pd.Series([[1, 2], None, [3, 4]])
        with self.assertRaises(ValueError) as context:
            check_column_values(column_data, "test_column", "test_file.parquet")
        self.assertIn("contains null arrays", str(context.exception))

    def test_empty_arrays(self):
        column_data = pd.Series([[1, 2], [], [3, 4]])
        with self.assertRaises(ValueError) as context:
            check_column_values(column_data, "test_column", "test_file.parquet")
        self.assertIn("contains empty arrays", str(context.exception))

    def test_null_elements_in_arrays(self):
        column_data = pd.Series([[1, None], [2, 3], [3, 4]])
        with self.assertRaises(ValueError) as context:
            check_column_values(column_data, "test_column", "test_file.parquet")
        self.assertIn("contains null values within arrays", str(context.exception))

    def test_empty_strings_in_arrays(self):
        column_data = pd.Series([["", ""], ["", ""], ["", ""]])
        with self.assertRaises(ValueError) as context:
            check_column_values(column_data, "test_column", "test_file.parquet")
        self.assertIn(
            "contains only empty strings within arrays", str(context.exception)
        )

    def test_scalar_strings_with_nulls(self):
        column_data = pd.Series(["a", None, "b"])
        with self.assertRaises(ValueError) as context:
            check_column_values(column_data, "test_column", "test_file.parquet")
        self.assertIn("contains null values", str(context.exception))

    def test_scalar_strings_with_empty(self):
        column_data = pd.Series(["a", "", "b"])
        with self.assertRaises(ValueError) as context:
            check_column_values(column_data, "test_column", "test_file.parquet")
        self.assertIn("contains empty strings", str(context.exception))

    def test_with_fallback_caption(self):
        column_data = pd.Series([None, "", [None], [""]])
        try:
            check_column_values(
                column_data,
                "test_column",
                "test_file.parquet",
                fallback_caption_column=True,
            )
        except ValueError:
            self.fail(
                "check_column_values() raised ValueError unexpectedly with fallback_caption_column=True"
            )

    def test_invalid_data_type(self):
        column_data = pd.Series([1, 2, 3])
        with self.assertRaises(TypeError) as context:
            check_column_values(column_data, "test_column", "test_file.parquet")
        self.assertIn("Unsupported data type in column", str(context.exception))


if __name__ == "__main__":
    unittest.main()
