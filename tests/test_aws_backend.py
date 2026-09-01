import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from simpletuner.helpers.data_backend.aws import S3DataBackend, normalise_s3_prefixes, test_s3_connection


class FakePaginator:
    def __init__(self, objects_by_prefix):
        self.objects_by_prefix = objects_by_prefix
        self.calls = []

    def paginate(self, **kwargs):
        self.calls.append(kwargs)
        prefix = kwargs.get("Prefix", "")
        return [{"Contents": [{"Key": key} for key in self.objects_by_prefix.get(prefix, [])]}]


class TestS3PrefixHandling(unittest.TestCase):
    def test_normalise_s3_prefixes_accepts_string_or_list(self):
        self.assertEqual(normalise_s3_prefixes("train/"), ("train/",))
        self.assertEqual(normalise_s3_prefixes(["train/", "reg/"]), ("train/", "reg/"))
        self.assertEqual(normalise_s3_prefixes([]), ("",))

    def test_list_files_paginates_each_prefix_and_deduplicates_keys(self):
        paginator = FakePaginator(
            {
                "train/": ["train/a.jpg", "train/readme.txt", "train/shared.png"],
                "train/sub/": ["train/sub/b.jpg", "train/shared.png"],
            }
        )
        client = Mock()
        client.get_paginator.return_value = paginator

        backend = S3DataBackend.__new__(S3DataBackend)
        backend.bucket_name = "bucket"
        backend.client = client
        backend.data_prefixes = ("",)

        result = backend.list_files(["jpg", "png"], instance_data_dir=["train/", "train/sub/"])

        self.assertEqual([call["Prefix"] for call in paginator.calls], ["train/", "train/sub/"])
        flattened = [key for _, _, keys in result for key in keys]
        self.assertEqual(flattened, ["train/a.jpg", "train/shared.png", "train/sub/b.jpg"])

    def test_s3_connection_checks_each_prefix(self):
        client = Mock()
        client.meta = SimpleNamespace(region_name="us-east-1", endpoint_url="https://s3.local")
        client.list_objects_v2.side_effect = [
            {"Contents": [{"Key": "train/a.jpg"}], "IsTruncated": False},
            {"Contents": [{"Key": "reg/b.jpg"}], "IsTruncated": True},
        ]

        with patch("simpletuner.helpers.data_backend.aws.boto3.client", return_value=client):
            details = test_s3_connection(
                bucket_name="bucket",
                prefix=["train/", "reg/"],
                endpoint_url="https://s3.local",
                max_keys=5,
            )

        client.head_bucket.assert_called_once_with(Bucket="bucket")
        self.assertEqual(
            [call.kwargs["Prefix"] for call in client.list_objects_v2.call_args_list],
            ["train/", "reg/"],
        )
        self.assertEqual(details["prefix"], ["train/", "reg/"])
        self.assertEqual(details["sample_keys"], ["train/a.jpg", "reg/b.jpg"])
        self.assertTrue(details["truncated"])


if __name__ == "__main__":
    unittest.main()
