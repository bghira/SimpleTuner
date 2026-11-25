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

    @patch("webshart._webshart._S3Store")
    def test_webshart_checks_capture_manifest_with_matching_host_profile(self, storage):
        config_file = self.tmp_path / "aws-config"
        credentials_file = self.tmp_path / "aws-credentials"
        config_file.write_text("[profile datasets]\nendpoint_url = https://storage.example\nregion = us-east-2\n")
        credentials_file.write_text(
            "[datasets]\naws_access_key_id = fixture-access\naws_secret_access_key = fixture-secret\n"
            "aws_session_token = fixture-token\n"
        )
        with patch.dict(
            "os.environ",
            {"AWS_CONFIG_FILE": str(config_file), "AWS_SHARED_CREDENTIALS_FILE": str(credentials_file)},
            clear=True,
        ):
            result = self.service.test_connection(
                {
                    "type": "webshart",
                    "aws_endpoint_url": "https://storage.example",
                    "webshart": {
                        "source": "s3://images/training",
                        "format": "parquet",
                        "manifest": "capture/manifest.json",
                        "subfolder": "shards",
                    },
                }
            )

        self.assertEqual(result["details"], {"transport": "s3", "manifest_checked": True})
        storage.assert_called_once_with(
            "images",
            endpoint_url="https://storage.example",
            region_name="us-east-2",
            aws_access_key_id="fixture-access",
            aws_secret_access_key="fixture-secret",
            aws_session_token="fixture-token",
        )
        storage.return_value.head.assert_called_once_with("training/capture/manifest.json")

    @patch("simpletuner.helpers.data_backend.aws.boto3.client")
    def test_webshart_without_manifest_checks_only_its_s3_prefix(self, create_client):
        client = create_client.return_value
        client.list_objects_v2.return_value = {"Contents": [{"Key": "training/shards/part.parquet"}]}
        result = self.service.test_connection(
            {
                "type": "webshart",
                "source": "s3://images/training",
                "webshart": {"format": "parquet", "subfolder": "shards"},
            }
        )
        self.assertTrue(result["details"]["has_objects"])
        client.head_bucket.assert_not_called()
        client.list_objects_v2.assert_called_once_with(Bucket="images", Prefix="training/shards/", MaxKeys=1)

    @patch("webshart._webshart._S3Store")
    def test_webshart_access_denied_does_not_return_transport_errors(self, storage):
        storage.return_value.head.side_effect = PermissionError("private diagnostic")
        with self.assertRaisesRegex(DatasetConnectionError, "denied access") as result:
            self.service.test_connection(
                {
                    "type": "webshart",
                    "source": "s3://images",
                    "webshart": {"format": "parquet", "manifest": "capture.json"},
                }
            )
        self.assertNotIn("private diagnostic", str(result.exception))
        self.assertEqual(result.exception.backend, "webshart")

    @patch("huggingface_hub.HfApi")
    def test_webshart_checks_hub_repo_with_its_hf_token(self, hub):
        result = self.service.test_connection(
            {
                "type": "webshart",
                "source": "https://huggingface.co/datasets/example/images",
                "webshart": {"hf_token": "fixture-token"},
            }
        )
        hub.assert_called_once_with(token="fixture-token")
        hub.return_value.dataset_info.assert_called_once_with("example/images", expand=["id"], timeout=10)
        self.assertEqual(result["details"], {"transport": "huggingface", "manifest_checked": False})

    @patch("simpletuner.simpletuner_sdk.server.services.dataset_connection_service.test_s3_connection")
    def test_aws_connection_passes_the_session_token(self, probe):
        probe.return_value = {"sample_keys": ["sample.jpg"]}
        result = self.service.test_connection(
            {"type": "aws", "aws_bucket_name": "images", "aws_session_token": "fixture-token"}
        )
        self.assertEqual(result["backend"], "aws")
        self.assertEqual(probe.call_args.kwargs["aws_session_token"], "fixture-token")


if __name__ == "__main__":
    unittest.main()
