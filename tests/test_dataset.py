import unittest
from unittest.mock import patch, Mock, MagicMock
from PIL import Image
from pathlib import Path
from helpers.multiaspect.dataset import MultiAspectDataset
from helpers.multiaspect.bucket import BucketManager
from helpers.data_backend.base import BaseDataBackend


class TestMultiAspectDataset(unittest.TestCase):
    def setUp(self):
        self.instance_data_root = "/some/fake/path"
        self.accelerator = Mock()
        self.bucket_manager = Mock(spec=BucketManager)
        self.bucket_manager.__len__ = Mock(return_value=10)
        self.data_backend = Mock(spec=BaseDataBackend)
        self.image_path = "fake_image_path"
        # Mock the Path.exists method to return True
        with patch("pathlib.Path.exists", return_value=True):
            self.dataset = MultiAspectDataset(
                instance_data_root=self.instance_data_root,
                accelerator=self.accelerator,
                bucket_manager=self.bucket_manager,
                data_backend=self.data_backend,
            )

    def test_init_invalid_instance_data_root(self):
        with self.assertRaises(ValueError):
            MultiAspectDataset(
                instance_data_root="/invalid/path",
                accelerator=self.accelerator,
                bucket_manager=self.bucket_manager,
                data_backend=self.data_backend,
            )

    def test_len(self):
        self.bucket_manager.__len__.return_value = 10
        self.assertEqual(len(self.dataset), 10)

    def test_getitem_valid_image(self):
        mock_image_data = b"fake_image_data"
        self.data_backend.read.return_value = mock_image_data

        with patch("PIL.Image.open") as mock_image_open, patch(
            "helpers.training.state_tracker.StateTracker.status_training",
            return_value=True,
        ):
            # Create a blank canvas:
            mock_image = Image.new(mode="RGB", size=(16, 8))
            mock_image_open.return_value = mock_image
            target = tuple([{"image_path": self.image_path, "image_data": mock_image}])
            examples = self.dataset.__getitem__(target)
        # Grab the size of the first image:
        example = examples[0]
        first_size = example["original_size"]
        # Are all sizes the same?
        for example in examples:
            self.assertIsNotNone(example)
            self.assertEqual(example["original_size"], first_size)
            self.assertEqual(example["image_path"], self.image_path)

    def test_getitem_invalid_image(self):
        self.data_backend.read.side_effect = Exception("Some error")

        with self.assertRaises(Exception):
            with patch(
                "helpers.training.state_tracker.StateTracker.status_training",
                return_value=True,
            ):
                with self.assertLogs("MultiAspectDataset", level="ERROR") as cm:
                    self.dataset.__getitem__(self.image_path)

    def test_getitem_not_in_training_state(self):
        with patch(
            "helpers.training.state_tracker.StateTracker.status_training",
            return_value=False,
        ):
            example = self.dataset.__getitem__(self.image_path)
        self.assertIsNone(example)


if __name__ == "__main__":
    unittest.main()
