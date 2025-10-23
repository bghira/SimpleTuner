import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simpletuner.simpletuner_sdk.server.services.dataset_connection_service import (
    DatasetConnectionError,
    DatasetConnectionService,
)


class DatasetConnectionServiceTestCase(unittest.TestCase):
    """Exercises lightweight connection checks without relying on pytest helpers."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        self.service = DatasetConnectionService()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_csv(self, relative_path: str, content: str) -> Path:
        csv_path = self.tmp_path / relative_path
        csv_path.write_text(content, encoding="utf-8")
        return csv_path

    def test_csv_connection_success(self) -> None:
        csv_path = self._write_csv("data.csv", "url,caption\nfoo,bar\n")

        result = self.service.test_connection(
            {
                "type": "csv",
                "csv_file": str(csv_path),
                "csv_caption_column": "caption",
                "csv_url_column": "url",
            }
        )

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["backend"], "csv")
        self.assertEqual(result["details"]["warnings"], [])
        self.assertEqual(result["details"]["rows_sampled"], 1)

    def test_csv_connection_missing_caption_warns(self) -> None:
        csv_path = self._write_csv("data.csv", "url,text\nfoo,bar\n")

        result = self.service.test_connection(
            {
                "type": "csv",
                "csv_file": str(csv_path),
                "csv_caption_column": "caption",
                "csv_url_column": "url",
            }
        )

        self.assertEqual(result["status"], "ok")
        warnings = result["details"]["warnings"]
        self.assertTrue(any("caption" in warning.lower() for warning in warnings))

    @patch("simpletuner.simpletuner_sdk.server.services.dataset_connection_service.test_huggingface_dataset")
    def test_huggingface_connection_uses_helper(self, mock_test) -> None:
        mock_test.return_value = {
            "available_splits": ["train"],
            "features": ["image"],
        }

        result = self.service.test_connection(
            {
                "type": "huggingface",
                "dataset_name": "example/dataset",
                "split": "train",
            }
        )

        self.assertEqual(result["backend"], "huggingface")
        mock_test.assert_called_once()

    @patch(
        "simpletuner.simpletuner_sdk.server.services.dataset_connection_service.test_s3_connection",
        side_effect=ValueError("boom"),
    )
    def test_aws_connection_error_propagates(self, mock_test) -> None:
        with self.assertRaises(DatasetConnectionError) as excinfo:
            self.service.test_connection(
                {
                    "type": "aws",
                    "aws_bucket_name": "bucket-name",
                }
            )

        self.assertIn("boom", str(excinfo.exception))
        mock_test.assert_called_once()


if __name__ == "__main__":
    unittest.main()
