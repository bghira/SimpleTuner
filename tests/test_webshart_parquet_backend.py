import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from simpletuner.helpers.data_backend.config.image import ImageBackendConfig
from simpletuner.helpers.data_backend.filters import DatasetFilter
from simpletuner.helpers.data_backend.webshart import WebshartDataBackend


class TestWebshartParquetBackend(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "source"
        source.mkdir()
        self.source = source
        data = BytesIO()
        Image.new("RGB", (16, 8), "red").save(data, format="PNG")
        self.image = data.getvalue()
        self.rows = [
            dict(
                image={"bytes": self.image, "path": None},
                captions="first",
                width=16,
                height=8,
                url_status="ok",
                stored_bytes=len(self.image),
            ),
            dict(image=None, captions="dead", width=None, height=None, url_status="http_error", stored_bytes=None),
            dict(
                image={"bytes": self.image, "path": None},
                captions="third",
                width=16,
                height=8,
                url_status="ok",
                stored_bytes=len(self.image),
            ),
        ]
        pq.write_table(pa.Table.from_pylist(self.rows), source / "part.parquet", row_group_size=2)
        self.backend = WebshartDataBackend(
            None, "parquet", str(source), format="parquet", cache_dir=str(self.root / "cache")
        )

    def test_filtered_rows_keep_physical_identity_captions_and_pixels(self):
        entries = self.backend.list_samples_in_shard(0)
        self.assertEqual([e["sample_idx"] for e in entries], [0, 2])
        paths = self.backend.list_files()[0][2]
        self.assertEqual([self.backend.parse_sample_id(path).sample_idx for path in paths], [0, 2])
        self.assertEqual(self.backend.get_caption(paths[1]), "third")
        self.assertEqual(self.backend.read(paths[1]), self.image)
        keys, images = self.backend.read_image_batch([paths[1], paths[0], paths[1]])
        self.assertEqual(keys, [paths[1], paths[0], paths[1]])
        self.assertEqual([image.size for image in images], [(16, 8)] * 3)
        self.assertEqual(images[0].getpixel((0, 0)), (255, 0, 0))
        filtered = self.backend.list_shard_sample_aspect_buckets(
            [0], dataset_filter=DatasetFilter({"path_include": "000000000002"})
        )
        self.assertEqual([e["sample_idx"] for e in filtered[0]["buckets"]["2.00"]], [2])

    def test_roundtrip_preserves_format_and_omits_credentials(self):
        value = self.backend.get_instance_representation()
        self.assertEqual(value["format"], "parquet")
        self.assertNotIn("aws_access_key_id", value)
        self.assertNotIn("aws_secret_access_key", value)
        self.assertNotIn("aws_session_token", value)
        restored = self.backend.from_instance_representation(value)
        self.assertEqual(restored.get_caption(restored.list_files()[0][2][1]), "third")

    def test_aws_config_reaches_native_discovery_without_python_transfers(self):
        from simpletuner.helpers.data_backend.builders.webshart import WebshartBackendBuilder

        payload = dict(
            id="s3-images",
            type="webshart",
            source="s3://images",
            metadata_backend="webshart",
            caption_strategy="webshart",
            aws_region_name="us-east-1",
            aws_endpoint_url="https://storage.example",
            aws_access_key_id="example-access",
            aws_secret_access_key="example-secret",
            aws_session_token="example-session",
            aws_max_pool_connections=16,
            webshart=dict(format="parquet", manifest="combined/dataset.json", columns={"captions": "text"}),
        )
        config = ImageBackendConfig.from_dict(payload, {})
        roundtrip = config.to_dict()["config"]
        self.assertEqual(roundtrip["webshart"]["manifest"], "combined/dataset.json")
        self.assertEqual(roundtrip["aws_session_token"], "example-session")
        with patch("simpletuner.helpers.data_backend.builders.webshart.WebshartDataBackend") as cls:
            WebshartBackendBuilder(None, {})._create_backend(config)
            for key in (
                "aws_access_key_id",
                "aws_secret_access_key",
                "aws_session_token",
                "aws_endpoint_url",
                "aws_max_pool_connections",
            ):
                self.assertEqual(cls.call_args.kwargs[key], payload[key])
            self.assertEqual(cls.call_args.kwargs["format"], "parquet")

    def test_missing_and_corrupt_images_are_counted_as_decode_failures(self):
        self.rows[2]["image"]["bytes"] = b"broken image"
        pq.write_table(pa.Table.from_pylist(self.rows), self.source / "part.parquet", row_group_size=2)
        backend = WebshartDataBackend(
            None, "broken", str(self.source), format="parquet", cache_dir=str(self.root / "broken-cache")
        )
        keys, images = backend.read_image_batch(backend.list_files()[0][2])
        self.assertEqual(len(keys), 1)
        self.assertEqual(len(images), 1)

    def test_metadata_bucketing_uses_dimensions_without_image_reads(self):
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        config = dict(caption_strategy="webshart", resolution=64, resolution_type="pixel", crop=False)
        args = SimpleNamespace(
            disable_bucket_pruning=False, allow_dataset_oversubscription=False, ignore_missing_files=False, seed=42
        )
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
            patch.object(StateTracker, "set_image_files"),
        ):
            metadata = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "buckets"),
                metadata_file=str(self.root / "metadata"),
                data_backend=self.backend,
                accelerator=None,
                batch_size=1,
                resolution=64,
                resolution_type="pixel",
            )
            with patch.object(
                self.backend.loader, "load_sample", side_effect=AssertionError("metadata must not load image bytes")
            ):
                metadata.compute_aspect_ratio_bucket_indices()
            paths = [path for bucket in metadata.aspect_ratio_bucket_indices.values() for path in bucket]
            self.assertEqual(len(paths), 2)
            self.assertEqual(metadata.filtering_statistics["skipped"]["error"], 0)
            self.assertEqual({self.backend.parse_sample_id(path).sample_idx for path in paths}, {0, 2})
            reloaded = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "buckets"),
                metadata_file=str(self.root / "metadata"),
                data_backend=self.backend,
                accelerator=None,
                batch_size=1,
                resolution=64,
                resolution_type="pixel",
            )
            self.assertEqual(reloaded.aspect_ratio_bucket_indices, metadata.aspect_ratio_bucket_indices)

    def test_catalog_changes_reject_old_sample_refs_and_worker_state(self):
        old_path = self.backend.list_files()[0][2][0]
        representation = self.backend.get_instance_representation()
        pq.write_table(pa.Table.from_pylist(self.rows), self.source / "aaa.parquet", row_group_size=2)
        changed = WebshartDataBackend(
            None, "parquet", str(self.source), format="parquet", cache_dir=str(self.root / "cache")
        )
        self.assertNotEqual(changed.dataset.fingerprint, self.backend.dataset.fingerprint)
        with self.assertRaisesRegex(ValueError, "identity"):
            changed.read(old_path)
        with self.assertRaisesRegex(ValueError, "catalog changed"):
            self.backend.from_instance_representation(representation)

    def test_jxl_image_bytes_decode(self):
        try:
            import pillow_jxl
        except ImportError:
            self.skipTest("pillow-jxl-plugin is optional")
        data = BytesIO()
        Image.new("RGB", (16, 8), "blue").save(data, format="JXL", lossless=True)
        self.rows[0]["image"]["bytes"] = data.getvalue()
        self.rows[0]["stored_format"] = "jxl"
        pq.write_table(pa.Table.from_pylist(self.rows), self.source / "part.parquet", row_group_size=2)
        backend = WebshartDataBackend(
            None, "jxl", str(self.source), format="parquet", cache_dir=str(self.root / "jxl-cache")
        )
        image = backend.read_image(backend.list_files()[0][2][0])
        self.assertEqual(image.size, (16, 8))
        self.assertEqual(image.getpixel((0, 0)), (0, 0, 255))


if __name__ == "__main__":
    unittest.main()
