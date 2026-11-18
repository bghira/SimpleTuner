import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from botocore.exceptions import ClientError

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

    def test_bulk_exists_probes_only_requested_keys_in_order(self):
        backend = S3DataBackend.__new__(S3DataBackend)
        backend.max_pool_connections = 2
        backend.client = Mock()
        backend.exists = Mock(side_effect=lambda key: key.endswith("/1001"))
        self.assertEqual(backend.bulk_exists(["a/1001", "a/1001", "b/2", "a/2"], prefix="a/"), [True, True, False, False])
        self.assertEqual(sorted(call.args[0] for call in backend.exists.call_args_list), ["a/1001", "a/2"])
        backend.client.list_objects_v2.assert_not_called()

    def test_read_cached_tensors_uses_get_for_hits_and_missing_objects(self):
        backend = S3DataBackend.__new__(S3DataBackend)
        backend.bucket_name = "cache"
        backend.read_retry_limit = 1
        backend.max_pool_connections = 2
        backend.client = Mock()
        backend.client.exceptions.NoSuchKey = type("NoSuchKey", (ClientError,), {})

        def get_object(*, Bucket, Key):
            if Key == "missing":
                raise ClientError({"Error": {"Code": "404"}}, "GetObject")
            body = BytesIO()
            torch.save({"value": torch.tensor(int(Key))}, body)
            body.seek(0)
            return {"Body": body}

        backend.client.get_object.side_effect = get_object
        results = list(backend.read_cached_tensors(["2", "missing", "1"]))
        self.assertEqual([None if entry is None else entry["value"].item() for entry in results], [2, None, 1])
        self.assertEqual(backend.client.get_object.call_count, 3)
        backend.client.head_object.assert_not_called()
        backend.client.get_paginator.assert_not_called()

        backend.client.get_object.side_effect = ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
        with self.assertRaisesRegex(ClientError, "AccessDenied"):
            list(backend.read_cached_tensors(["2"]))

    def test_iter_file_keys_reads_pages_lazily(self):
        backend = S3DataBackend.__new__(S3DataBackend)
        backend.bucket_name = "cache"
        backend.client = Mock()
        pages_read = []

        def pages(**kwargs):
            for name in ("a.pt", "ignore.txt", "b.pt"):
                pages_read.append(name)
                yield {"Contents": [{"Key": f"cache/{name}"}]}

        backend.client.get_paginator.return_value.paginate.side_effect = pages
        keys = backend.iter_file_keys(["pt"], instance_data_dir="cache/")
        self.assertEqual(next(keys), "cache/a.pt")
        self.assertEqual(pages_read, ["a.pt"])
        self.assertEqual(list(keys), ["cache/b.pt"])

    def test_open_read_stream_leaves_metadata_consumption_to_the_caller(self):
        backend = S3DataBackend.__new__(S3DataBackend)
        backend.bucket_name = "datasets"
        backend.client = Mock()
        body = BytesIO(b"published membership")
        backend.client.get_object.return_value = {"Body": body}

        with backend.open_read_stream("buckets.json") as stream:
            self.assertIs(stream, body)
            self.assertEqual(stream.tell(), 0)
            self.assertEqual(stream.read(9), b"published")
        self.assertTrue(body.closed)
        backend.client.get_object.assert_called_once_with(Bucket="datasets", Key="buckets.json")

    def test_normalise_s3_prefixes_accepts_string_or_list(self):
        self.assertEqual(normalise_s3_prefixes("train/"), ("train/",))
        self.assertEqual(normalise_s3_prefixes(["train/", "reg/"]), ("train/", "reg/"))
        self.assertEqual(normalise_s3_prefixes([]), ("",))

    def test_list_files_paginates_each_prefix_and_deduplicates_keys(self):
        paginator = FakePaginator(
            {
                "train/": ["train/a.jpg", "train/readme.txt", "train/sub/b.jpg", "train/sub/ignored.raw"],
                "train/sub/": ["train/sub/b.jpg"],
            }
        )
        client = Mock()
        client.get_paginator.return_value = paginator

        backend = S3DataBackend.__new__(S3DataBackend)
        backend.bucket_name = "bucket"
        backend.client = client
        backend.data_prefixes = ("",)

        result = backend.list_files(["jpg", "png"], instance_data_dir=["train/sub/", "train/"])

        self.assertEqual([call["Prefix"] for call in paginator.calls], ["train/sub/", "train/"])
        flattened = [key for _, _, keys in result for key in keys]
        self.assertEqual(flattened, ["train/sub/b.jpg", "train/a.jpg"])

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
