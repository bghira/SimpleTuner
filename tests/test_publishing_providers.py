import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from simpletuner.helpers.publishing.providers.azure_blob import AzureBlobPublishingProvider
from simpletuner.helpers.publishing.providers.dropbox import DropboxPublishingProvider
from simpletuner.helpers.publishing.providers.s3 import S3PublishingProvider


class TestS3Provider(unittest.TestCase):
    def setUp(self):
        self.config = {
            "provider": "s3",
            "bucket": "test-bucket",
            "access_key": "acc",
            "secret_key": "sec",
            "region": "us-east-1",
            "name": "TestS3",
        }

    def test_init(self):
        mock_boto3 = MagicMock()
        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            provider = S3PublishingProvider(self.config)

            mock_boto3.session.Session.assert_called_with(
                aws_access_key_id="acc", aws_secret_access_key="sec", region_name="us-east-1"
            )
            mock_boto3.session.Session.return_value.client.assert_called_with("s3", endpoint_url=None, use_ssl=True)
            self.assertEqual(provider.bucket, "test-bucket")

    def test_publish_file(self):
        mock_boto3 = MagicMock()
        mock_session = mock_boto3.session.Session.return_value
        mock_client = mock_session.client.return_value

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            provider = S3PublishingProvider(self.config)

            with patch("pathlib.Path.is_file", return_value=True), patch("pathlib.Path.name", "test_file.txt"):

                # We need to mock _iter_files because it touches the filesystem
                with patch.object(provider, "_iter_files", return_value=[Path("test_file.txt")]):
                    result = provider.publish("test_file.txt")

            mock_client.upload_file.assert_called()
            # Check if the destination key is correct
            args, _ = mock_client.upload_file.call_args
            self.assertEqual(args[1], "test-bucket")
            self.assertTrue(args[2].endswith("test_file.txt"))
            self.assertIn("s3://test-bucket", result.uri)

    def test_endpoint_url(self):
        config = self.config.copy()
        config["endpoint_url"] = "https://custom.endpoint"

        mock_boto3 = MagicMock()

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            provider = S3PublishingProvider(config)
            mock_boto3.session.Session.return_value.client.assert_called_with(
                "s3", endpoint_url="https://custom.endpoint", use_ssl=True
            )

            # Also check URI generation with endpoint
            with patch.object(provider, "_iter_files", return_value=[Path("f.txt")]):
                result = provider.publish("f.txt")
                self.assertIn("https://custom.endpoint", result.uri)

    def test_public_base_url(self):
        config = self.config.copy()
        config["public_base_url"] = "https://cdn.example.com"

        mock_boto3 = MagicMock()

        with patch.dict(sys.modules, {"boto3": mock_boto3}):
            provider = S3PublishingProvider(config)

            with patch.object(provider, "_iter_files", return_value=[Path("f.txt")]):
                result = provider.publish("f.txt")
                # Should use public_base_url instead of s3:// or endpoint_url
                self.assertEqual(result.uri, "https://cdn.example.com/f.txt")


class TestAzureProvider(unittest.TestCase):
    def setUp(self):
        self.config = {
            "provider": "azure_blob",
            "connection_string": "DefaultEndpointsProtocol=https;AccountName=test;AccountKey=key;EndpointSuffix=core.windows.net",
            "container": "test-container",
        }

    def test_init_connection_string(self):
        mock_azure = MagicMock()
        mock_service_client = mock_azure.storage.blob.BlobServiceClient
        mock_client_instance = mock_service_client.from_connection_string.return_value

        modules = {
            "azure": mock_azure,
            "azure.core": MagicMock(),
            "azure.core.exceptions": MagicMock(),
            "azure.storage": MagicMock(),
            "azure.storage.blob": mock_azure.storage.blob,
        }

        with patch.dict(sys.modules, modules):
            provider = AzureBlobPublishingProvider(self.config)

            mock_service_client.from_connection_string.assert_called_with(self.config["connection_string"])
            mock_client_instance.get_container_client.assert_called_with("test-container")

    def test_publish(self):
        mock_azure = MagicMock()
        mock_service_client = mock_azure.storage.blob.BlobServiceClient
        mock_client_instance = mock_service_client.from_connection_string.return_value
        mock_container_client = mock_client_instance.get_container_client.return_value
        mock_container_client.url = "https://test.blob.core.windows.net/test-container"

        modules = {
            "azure": mock_azure,
            "azure.core": MagicMock(),
            "azure.core.exceptions": MagicMock(),
            "azure.storage": MagicMock(),
            "azure.storage.blob": mock_azure.storage.blob,
        }

        with patch.dict(sys.modules, modules):
            provider = AzureBlobPublishingProvider(self.config)

            with patch("builtins.open", unittest.mock.mock_open(read_data=b"data")) as mock_file:
                with patch.object(provider, "_iter_files", return_value=[Path("test_file.txt")]):
                    result = provider.publish("test_file.txt")

            mock_container_client.upload_blob.assert_called()
            self.assertIn("https://test.blob.core.windows.net/test-container", result.uri)


class TestDropboxProvider(unittest.TestCase):
    def setUp(self):
        self.config = {"provider": "dropbox", "token": "test-token", "base_path": "/runs"}

    def test_init(self):
        mock_dropbox = MagicMock()
        with patch.dict(sys.modules, {"dropbox": mock_dropbox, "dropbox.files": MagicMock()}):
            provider = DropboxPublishingProvider(self.config)
            mock_dropbox.Dropbox.assert_called_with("test-token")

    def test_publish_simple(self):
        mock_dropbox = MagicMock()
        mock_dbx_instance = mock_dropbox.Dropbox.return_value
        mock_dbx_instance.sharing_create_shared_link_with_settings.return_value.url = "https://dbx.com/link"

        with patch.dict(sys.modules, {"dropbox": mock_dropbox, "dropbox.files": MagicMock()}):
            provider = DropboxPublishingProvider(self.config)

            # Mock file size to be small (simple upload)
            with (
                patch("pathlib.Path.stat") as mock_stat,
                patch("pathlib.Path.is_dir", return_value=False),
                patch("pathlib.Path.is_file", return_value=True),
            ):

                mock_stat.return_value.st_size = 100

                with patch("builtins.open", unittest.mock.mock_open(read_data=b"data")):
                    with patch.object(provider, "_iter_files", return_value=[Path("test_file.txt")]):
                        result = provider.publish("test_file.txt")

            mock_dbx_instance.files_upload.assert_called()
            self.assertEqual(result.uri, "https://dbx.com/link")


if __name__ == "__main__":
    unittest.main()
