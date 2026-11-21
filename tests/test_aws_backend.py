import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from botocore.exceptions import ClientError

from simpletuner.helpers.caching.artifact_listing import ArtifactListing
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

    def test_delete_batch_streams_keys_and_surfaces_per_object_errors(self):
        backend = S3DataBackend.__new__(S3DataBackend)
        backend.bucket_name = "cache"
        backend.client = Mock()
        backend.client.delete_objects.side_effect = [{}, {"Errors": [{"Code": "AccessDenied", "Key": "old/1000.pt"}]}]

        def keys():
            for position in range(1001):
                if position == 1000:
                    self.assertEqual(backend.client.delete_objects.call_count, 1)
                yield f"old/{position}.pt"

        with self.assertRaisesRegex(ClientError, "AccessDenied"):
            backend.delete_batch(keys())
        first, last = [entry.kwargs for entry in backend.client.delete_objects.call_args_list]
        self.assertEqual(first["Bucket"], "cache")
        self.assertEqual(first["Delete"], {"Objects": [{"Key": f"old/{i}.pt"} for i in range(1000)], "Quiet": True})
        self.assertEqual(last, {"Bucket": "cache", "Delete": {"Objects": [{"Key": "old/1000.pt"}], "Quiet": True}})
        backend.client.delete_object.assert_not_called()

    def test_bulk_exists_probes_only_requested_keys_in_order(self):
        backend = S3DataBackend.__new__(S3DataBackend)
        backend.max_pool_connections = 2
        backend.client = Mock()
        backend.exists = Mock(side_effect=lambda key: key.endswith("/1001"))
        self.assertEqual(backend.bulk_exists(["a/1001", "a/1001", "b/2", "a/2"], prefix="a/"), [True, True, False, False])
        self.assertEqual(sorted(call.args[0] for call in backend.exists.call_args_list), ["a/1001", "a/2"])
        backend.client.list_objects_v2.assert_not_called()

    def test_shared_artifact_index_streams_one_bounded_listing(self):
        backend = S3DataBackend.__new__(S3DataBackend)
        backend.bucket_name = "cache"
        backend.max_pool_connections = 2
        backend.client = Mock()
        backend.exists = Mock(return_value=False)
        paginator = FakePaginator({"prefix/": ["prefix/a.pt", "prefix/readme", "prefix/b.pt"]})
        backend.client.get_paginator.return_value = paginator

        with (
            TemporaryDirectory() as output_dir,
            patch("simpletuner.helpers.caching.artifact_listing.broadcast_object_from_main", return_value="shared"),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            main = ArtifactListing(backend, "prefix/", output_dir, SimpleNamespace(is_local_main_process=True))
            worker = ArtifactListing(backend, "prefix/", output_dir, SimpleNamespace(is_local_main_process=False))

            def check(index):
                with index.open() as probe:
                    return probe(["prefix/b.pt", "prefix/b.pt", "prefix/a.pt", "prefix/absent.pt"])

            self.assertEqual(list(pool.map(check, (main, worker))), [[True, True, True, False]] * 2)
            self.assertEqual(paginator.calls, [{"Bucket": "cache", "Prefix": "prefix/", "MaxKeys": 1000}])
            self.assertEqual([call.args for call in backend.exists.call_args_list], [("prefix/absent.pt",)] * 2)
            worker.close()
            self.assertTrue(main.path.is_file())
            main.close()
            self.assertFalse(main.path.exists())

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
        backend.read_retry_limit = 2
        backend.read_retry_interval = 0
        backend.client = Mock()
        backend.client.exceptions.NoSuchKey = type("NoSuchKey", (ClientError,), {})
        body = BytesIO(b"published membership")
        backend.client.get_object.side_effect = [
            ClientError({"Error": {"Code": "503"}}, "GetObject"),
            {"Body": body},
        ]

        with backend.open_read_stream("buckets.json") as stream:
            self.assertIs(stream, body)
            self.assertEqual(stream.tell(), 0)
            self.assertEqual(stream.read(9), b"published")
        self.assertTrue(body.closed)
        self.assertEqual(backend.client.get_object.call_count, 2)
        self.assertEqual(backend.client.get_object.call_args.kwargs, {"Bucket": "datasets", "Key": "buckets.json"})

        backend.client.get_object.reset_mock()
        backend.client.get_object.side_effect = ClientError({"Error": {"Code": "NotFound"}}, "GetObject")
        with self.assertRaisesRegex(FileNotFoundError, "buckets.json"):
            backend.open_read_stream("buckets.json")
        backend.client.get_object.assert_called_once()

    def test_versioned_metadata_uses_conditional_s3_writes(self):
        backend = S3DataBackend.__new__(S3DataBackend)
        backend.bucket_name = "datasets"
        backend.read_retry_limit = 1
        backend.client = Mock()
        backend.client.get_object.return_value = {"Body": BytesIO(b"old index"), "ETag": '"old"'}
        backend.client.put_object.return_value = {"ETag": '"new"'}

        index, version = backend.read_with_version("buckets.json")
        self.assertEqual((index, version), (b"old index", '"old"'))
        self.assertEqual(backend.write_if_version("buckets.json", b"next index", expected_version=version), '"new"')
        backend.client.put_object.assert_called_once_with(
            Bucket="datasets", Key="buckets.json", Body=b"next index", IfMatch='"old"'
        )

        backend.client.put_object.reset_mock()
        backend.write_if_version("buckets.json", b"first index", expected_version=None)
        backend.client.put_object.assert_called_once_with(
            Bucket="datasets", Key="buckets.json", Body=b"first index", IfNoneMatch="*"
        )

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
