import json
import sqlite3
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Event, get_ident
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

    def test_prepared_geometry_reads_compact_and_existing_text_records(self):
        from simpletuner.helpers.metadata.backends.webshart_geometry import GeometryLookup, PreparedGeometry

        record = {
            "original_size": [1024, 1536],
            "crop_coordinates": [256, 512],
            "target_size": [512, 512],
            "webshart": {"shard_idx": 1, "sample_idx": 3, "filename": "data/0003.jpg"},
        }
        with closing(sqlite3.connect(":memory:")) as connection:
            geometry = PreparedGeometry(connection)
            geometry.add("512x512", "new", record)
            connection.execute("INSERT INTO geometry VALUES (?,?,?,?)", ("original", json.dumps(record), "512x512", 1))
            geometry.finish()
            lookup = GeometryLookup(connection)
            self.assertEqual(connection.execute("SELECT typeof(record) FROM geometry WHERE ref='new'").fetchone(), ("blob",))
            self.assertEqual(lookup.get_metadata_by_filepath("new"), record)
            self.assertEqual(
                lookup.get_metadata_batch(["original", "new", "new", "missing"]), [record, record, record, None]
            )

    def test_runtime_can_request_exact_source_count_without_null_image_rows(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime

        runtime = DatasetRuntime(self.backend.id, self.backend, None, {})
        self.assertEqual(runtime.count_source_rows(), 3)
        if not hasattr(self.backend.dataset, "count_rows"):
            with self.assertRaisesRegex(NotImplementedError, "installed Webshart"):
                runtime.count_source_rows(exclude_null_images=True)
            return
        self.assertEqual(runtime.count_source_rows(exclude_null_images=True), 2)
        self.assertEqual([row["sample_idx"] for row in self.backend.list_samples_in_shard(0)], [0, 2])

    def test_prepared_view_uses_factory_bound_settings_without_tracker_lookups(self):
        from simpletuner.helpers.data_backend.builders.base import create_metadata_provider
        from simpletuner.helpers.training.state_tracker import StateTracker

        config = {
            "metadata_backend": "webshart",
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 8,
            "resolution_type": "pixel",
        }
        args = {"train_batch_size": 1, "disable_bucket_pruning": False, "aspect_bucket_alignment": 8, "seed": 123}
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        with (
            patch.object(StateTracker, "get_data_backend_config", side_effect=AssertionError("dataset settings lookup")),
            patch.object(StateTracker, "get_args", side_effect=AssertionError("run settings lookup")),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            metadata = create_metadata_provider(
                self.backend.id,
                config,
                self.backend,
                accelerator,
                args,
                instance_data_dir=self.backend.cache_dir,
                cache_root=self.backend.cache_dir,
            )
            self.addCleanup(metadata.close)
            metadata.compute_aspect_ratio_bucket_indices()
            self.assertEqual(metadata.describe().accepted_samples, 2)

    def test_capture_catalogue_uses_the_configured_metadata_cache(self):
        if not getattr(type(self.backend.dataset), "supports_catalogue_cache", False):
            self.skipTest("installed Webshart has no shared capture catalogue")
        self._publish_capture_manifest()
        metadata_cache = self.root / "shared-metadata"
        datasets = [
            WebshartDataBackend(
                None,
                name,
                str(self.source),
                format="parquet",
                manifest="capture.json",
                cache_dir=str(self.root / name),
                metadata_cache_dir=str(metadata_cache),
            )
            for name in ("global-schedule", "shard-local-schedule")
        ]
        self.assertEqual(datasets[0].dataset.fingerprint, datasets[1].dataset.fingerprint)
        self.assertEqual(len(list(metadata_cache.glob("capture-catalogue-*.sqlite3"))), 1)

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

    def test_projected_caption_batch_skips_shard_hydration_and_keeps_physical_order(self):
        if not self.backend.supports_caption_projection:
            self.skipTest("installed Webshart lacks projected row reads")
        paths = self.backend.list_files()[0][2]
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime

        cfg = {"caption_strategy": "webshart", "prepend_instance_prompt": True, "instance_prompt": "prefix"}
        runtime = DatasetRuntime(self.backend.id, self.backend, None, cfg)
        with (
            patch.object(self.backend.dataset, "_metadata", side_effect=AssertionError("whole shard")),
            patch.object(self.backend.dataset, "read_metadata_rows", wraps=self.backend.dataset.read_metadata_rows) as read,
        ):
            refs = [paths[1], paths[0], paths[1]]
            result = list(runtime.iter_caption_records(refs, batch_size=3))
            self.assertEqual(result, list(zip(["prefix third", "prefix first", "prefix third"], refs)))
            self.assertEqual(read.call_args.args, (0, [2, 0, 2]))
            self.assertEqual(read.call_count, 1)
            self.assertNotIn("image", read.call_args.kwargs["fields"])

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
            from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime

            runtime = DatasetRuntime(
                self.backend.id,
                self.backend,
                metadata,
                {
                    **config,
                    "dataset_type": "image",
                    "use_captions": True,
                    "prepend_instance_prompt": False,
                    "instance_prompt": None,
                    "disable_multiline_split": False,
                },
            )
            requests = [paths[1], paths[0], paths[1]]
            with (
                patch.object(StateTracker, "get_data_backend", side_effect=AssertionError("registry lookup")),
                patch.object(self.backend.loader, "load_sample", side_effect=AssertionError("unexpected payload read")),
            ):
                records = runtime.read_samples(requests)
                self.assertEqual([record["image_path"] for record in records], requests)
                self.assertEqual([runtime.resolve_caption(ref) for ref in requests], ["third", "first", "third"])
                records[0]["webshart"]["sample_idx"] = 99
                self.assertEqual(records[2]["webshart"]["sample_idx"], 2)
                self.assertEqual(metadata.get_metadata_by_filepath(paths[1])["webshart"]["sample_idx"], 2)
            with (
                patch.object(self.backend.loader, "load_samples", wraps=self.backend.loader.load_samples) as batch_read,
                patch.object(metadata, "get_metadata_by_filepath", side_effect=AssertionError("geometry read")),
            ):
                payloads = runtime.read_samples(requests, fields={"image"})
                self.assertEqual([record["image"].size for record in payloads], [(16, 8)] * 3)
                self.assertEqual(batch_read.call_count, 1)
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

    def test_native_bucket_shapes_need_the_same_preparation_recipe(self):
        from ast import literal_eval

        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        native = self.backend.loader.list_shard_sample_aspect_buckets(
            [0], key="geometry-tuple", metadata_fields=("width", "height")
        )[0]["buckets"]
        native_shapes = {entry["sample_idx"]: literal_eval(shape) for shape, entries in native.items() for entry in entries}
        self.assertEqual(native_shapes, {0: (16, 8), 2: (16, 8)})

        scaled = self.backend.loader.list_shard_sample_aspect_buckets(
            [0],
            key="geometry-tuple",
            target_pixel_area=64 * 64,
            target_resolution_multiple=8,
            metadata_fields=("width", "height"),
        )[0]["buckets"]
        self.assertEqual([literal_eval(shape) for shape in scaled], [(88, 48)])

        args = SimpleNamespace(model_family="sdxl", aspect_bucket_alignment=8, aspect_bucket_rounding=2)
        accel = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        for resolution_type, resolution, crop, minimum_size, expected in (
            ("pixel", 8, False, None, native_shapes),
            ("pixel", 64, False, None, {0: (128, 64), 2: (128, 64)}),
            ("pixel", 64, True, None, {0: (64, 64), 2: (64, 64)}),
            ("pixel", 8, False, 9, {}),
            ("area", 64 * 64 / 1e6, False, None, {0: (88, 48), 2: (88, 48)}),
        ):
            config = {
                "caption_strategy": "webshart",
                "webshart": {"sampling_policy": "global"},
                "resolution": resolution,
                "resolution_type": resolution_type,
                "crop": crop,
                "crop_aspect": "square",
            }
            with (
                self.subTest(resolution=resolution, resolution_type=resolution_type, crop=crop, minimum_size=minimum_size),
                patch.object(StateTracker, "get_data_backend_config", return_value=config),
                patch.object(StateTracker, "get_args", return_value=args),
                patch.object(StateTracker, "get_model", return_value=None),
                patch.object(StateTracker, "aspect_resolution_map", {}),
                patch.object(StateTracker, "_save_to_disk"),
            ):
                metadata = WebshartMetadataBackend(
                    id=self.backend.id,
                    instance_data_dir=self.backend.cache_dir,
                    cache_file=str(self.root / "buckets"),
                    metadata_file=str(self.root / "metadata"),
                    data_backend=self.backend,
                    accelerator=accel,
                    batch_size=1,
                    resolution=resolution,
                    resolution_type=resolution_type,
                    minimum_image_size=minimum_size,
                )
                stats = {"total_processed": 0, "skipped": {"too_small": 0}}
                scanned = metadata._scan_rows((), stats, True, None)
                shapes = {self.backend.parse_sample_id(ref).sample_idx: record["target_size"] for _, ref, record in scanned}
                self.assertEqual(shapes, expected)

    def test_native_preparation_shape_is_independent_of_geometry_memo_and_scan_order(self):
        from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        config = {
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 1.0,
            "resolution_type": "area",
            "crop": False,
        }
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        args = SimpleNamespace(model_family="sdxl", aspect_bucket_alignment=8, aspect_bucket_rounding=2, seed=21)
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            metadata = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "buckets"),
                metadata_file=str(self.root / "metadata"),
                data_backend=self.backend,
                accelerator=accelerator,
                batch_size=1,
                resolution=1.0,
                resolution_type="area",
            )
            aligned_8 = metadata._geometry_recipe(config)
            args.aspect_bucket_alignment = 64
            aligned_64 = metadata._geometry_recipe(config)
            config["crop"] = True
            config["crop_style"] = "random"
            config["crop_aspect"] = "random"
            config["crop_aspect_buckets"] = [0.75, 1.5]
            random_crop = metadata._geometry_recipe(config)

            with (
                patch.object(StateTracker, "get_data_backend_config", side_effect=AssertionError("used live config")),
                patch.object(
                    StateTracker, "get_resolution_by_aspect", side_effect=AssertionError("used global aspect memo")
                ),
                patch.object(
                    StateTracker, "set_resolution_by_aspect", side_effect=AssertionError("wrote global aspect memo")
                ),
            ):
                for recipe, target in (
                    (aligned_64, (768, 1408)),
                    (aligned_8, (744, 1352)),
                    (aligned_64, (768, 1408)),
                ):
                    bucket, record = metadata._prepare_metadata("sample", {"original_size": (550, 1000)}, recipe)
                    self.assertEqual(bucket, f"{target[0]}x{target[1]}")
                    self.assertEqual(record["target_size"], target)
                    self.assertTrue(record["prepared_geometry"])

                def scan(refs):
                    return {
                        ref: metadata._prepare_metadata(ref, {"original_size": (1100, 2000)}, random_crop) for ref in refs
                    }

                first = scan(("a", "b", "c"))
                self.assertEqual(first, scan(("c", "b", "a")))

            _, record = first["a"]
            image = Image.new("RGB", (1100, 2000))
            sample = TrainingSample(image, self.backend.id, image_metadata=record)
            with patch.object(sample, "calculate_target_size", side_effect=AssertionError("recomputed prepared geometry")):
                self.assertEqual(sample.prepare().image.size, record["target_size"])

    def test_instance_prompt_eligibility_does_not_require_source_captions(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        self.rows[-1]["captions"] = None
        pq.write_table(pa.Table.from_pylist(self.rows), self.source / "part.parquet", row_group_size=2)
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        source = WebshartDataBackend(
            accelerator, "instance-only", str(self.source), format="parquet", cache_dir=str(self.root / "instance-view")
        )
        args = SimpleNamespace(seed=12, disable_bucket_pruning=True)
        config = {
            "caption_strategy": "webshart",
            "resolution": 8,
            "resolution_type": "pixel",
            "webshart": {"sampling_policy": "global"},
            "instance_prompt": "shared prompt",
        }
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            metadata = WebshartMetadataBackend(
                id=source.id,
                instance_data_dir=source.cache_dir,
                cache_file=str(self.root / "instance-buckets"),
                metadata_file=str(self.root / "instance-geometry"),
                data_backend=source,
                accelerator=accelerator,
                batch_size=1,
                resolution=8,
                resolution_type="pixel",
            )
            self.addCleanup(metadata.close)
            metadata.compute_aspect_ratio_bucket_indices()
            with_captions = metadata.describe()
            self.assertEqual(with_captions.accepted_samples, 1)

            config["only_instance_prompt"] = True
            with patch.object(source, "caption_fields", side_effect=AssertionError("projected source captions")):
                metadata.compute_aspect_ratio_bucket_indices()
            instance_only = metadata.describe()
            self.assertEqual(instance_only.accepted_samples, 2)
            self.assertNotEqual(instance_only.view_revision, with_captions.view_revision)
            dataset = DatasetRuntime(source.id, source, metadata, {**config, **DatasetRuntime.caption_config(config, args)})
            refs, count = dataset.read_page(limit=2)
            self.assertEqual(count, 2)
            with patch.object(source, "read_caption_batch", side_effect=AssertionError("read source captions")):
                self.assertEqual(dataset.resolve_captions(refs), ["shared prompt", "shared prompt"])

    def test_native_publication_scans_each_canonical_group_separately(self):
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        pq.write_table(pa.Table.from_pylist(self.rows), self.source / "second.parquet", row_group_size=2)
        source = WebshartDataBackend(None, "groups", str(self.source), format="parquet", cache_dir=str(self.root / "groups"))
        source.prepared_partition_shards = 1
        config = {
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 8,
            "resolution_type": "pixel",
        }
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        args = SimpleNamespace(model_family="sdxl", disable_bucket_pruning=False)
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            metadata = WebshartMetadataBackend(
                id=source.id,
                instance_data_dir=source.cache_dir,
                cache_file=str(self.root / "buckets"),
                metadata_file=str(self.root / "geometry"),
                data_backend=source,
                accelerator=accelerator,
                batch_size=1,
                resolution=8,
                resolution_type="pixel",
            )
            self.addCleanup(metadata.close)
            with patch.object(metadata, "_scan_rows", wraps=metadata._scan_rows) as scan:
                progress = []
                metadata.compute_aspect_ratio_bucket_indices(progress_callback=lambda *state: progress.append(state))
            self.assertEqual([call.kwargs["shard_indices"] for call in scan.call_args_list], [(0,), (1,)])
            self.assertEqual(progress, [(1, 2), (2, 2)])
            prepared, total = metadata.read_page(limit=10)
            self.assertEqual(total, 4)

            recipe = metadata._geometry_recipe(config)
            separate = {}
            for shard in (1, 0):
                records = metadata._scan_rows((), {"total_processed": 0}, True, None, recipe=recipe, shard_indices=(shard,))
                for bucket, ref, record in records:
                    self.assertEqual(bucket, "16x8")
                    self.assertEqual(tuple(metadata.get_metadata_by_filepath(ref)["target_size"]), record["target_size"])
                    separate[ref] = record
            self.assertEqual(set(separate), set(prepared))

    def _publish_capture_manifest(self):
        with (self.source / "files.jsonl").open("wb") as stream:
            manifest_digest = sha256()
            for shard in sorted(self.source.glob("*.parquet")):
                entry = {
                    "key": shard.name,
                    "num_rows": len(self.rows),
                    "size_bytes": shard.stat().st_size,
                    "sha256": sha256(shard.read_bytes()).hexdigest(),
                }
                line = json.dumps(entry).encode() + b"\n"
                manifest_digest.update(line)
                stream.write(line)
        (self.source / "capture.json").write_text(
            json.dumps(
                {
                    "format": "parquet-image-capture",
                    "version": 1,
                    "files_manifest": "files.jsonl",
                    "manifest_sha256": manifest_digest.hexdigest(),
                }
            )
        )

    def test_conditioning_and_embedding_requests_share_native_target_membership(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime, TextRequestBuilder
        from simpletuner.helpers.data_backend.factory import FactoryRegistry
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.metadata.backends.webshart_exclusions import Exclusions, _publish
        from simpletuner.helpers.models.common import TextEmbedCacheKey
        from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
        from simpletuner.helpers.training.state_tracker import StateTracker

        config = {
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 8,
            "resolution_type": "pixel",
        }
        args = SimpleNamespace(
            model_family="sdxl",
            disable_bucket_pruning=False,
            skip_file_discovery="",
            max_train_steps=1,
            allow_dataset_oversubscription=False,
            gradient_accumulation_steps=1,
            eval_dataset_id=None,
        )
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True, is_local_main_process=True)
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            metadata = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "conditioning-buckets"),
                metadata_file=str(self.root / "conditioning-geometry"),
                data_backend=self.backend,
                accelerator=accelerator,
                batch_size=1,
                resolution=8,
                resolution_type="pixel",
            )
            self.addCleanup(metadata.close)
            factory = FactoryRegistry.__new__(FactoryRegistry)
            factory.args = args
            factory.accelerator = accelerator
            config["conditioning_type"] = "reference_strict"
            with (
                patch.object(StateTracker, "get_data_backend", side_effect=AssertionError("legacy source clone")),
                patch.object(factory, "_handle_config_versioning"),
            ):
                factory._handle_bucket_operations(
                    {**config, "id": self.backend.id},
                    {"id": self.backend.id, "config": config, "metadata_backend": metadata},
                    "reference_strict",
                )
            (accepted, excluded), count = metadata.read_page(limit=2)
            self.assertEqual(count, 2)
            runtime = DatasetRuntime(self.backend.id, self.backend, metadata, config)
            self.assertEqual(runtime.conditioning_ref(excluded), excluded)

            with closing(sqlite3.connect("")) as db:
                changes = Exclusions(db)
                changes.add(excluded, next(iter(metadata.published_bucket_counts())), "decode")
                publication = self.backend._cache_path(metadata.cache_file)
                _publish(publication, metadata.describe().view_revision, db)
            metadata.reload_cache()
            self.assertEqual(runtime.conditioning_ref(accepted), accepted)
            self.assertIsNone(runtime.conditioning_ref(excluded))
            self.assertTrue(self.backend.exists(excluded))

            sampler = MultiAspectSampler.__new__(MultiAspectSampler)
            sampler.dataset_runtime = runtime
            sampler.metadata_backend = metadata
            with patch.object(self.backend, "read_image", side_effect=AssertionError("excluded payload read")):
                self.assertIsNone(sampler.get_conditioning_sample(excluded))

            model = SimpleNamespace(text_embed_cache_key=lambda: TextEmbedCacheKey.CAPTION)
            builder = TextRequestBuilder(model, "source", None, conditioning_datasets=[runtime])
            records = list(builder.records([("kept", accepted), ("other", excluded)], {}))
            self.assertEqual(records[0]["metadata"]["image_paths"], [accepted])
            self.assertNotIn("image_paths", records[1]["metadata"])

    def test_generated_conditioning_reuses_native_prepared_geometry_without_cloning(self):
        from simpletuner.helpers.data_backend.builders.base import create_metadata_provider
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.data_backend.local import LocalDataBackend
        from simpletuner.helpers.data_generation import DataGenerator
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.metadata.backends.webshart_exclusions import Exclusions, _publish
        from simpletuner.helpers.training.state_tracker import StateTracker

        native = {
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 8,
            "resolution_type": "pixel",
        }
        root = self.root / "generated"
        derived = {
            **native,
            "dataset_type": "conditioning",
            "metadata_backend": "discovery",
            "instance_data_dir": str(root),
            "conditioning_config": {"type": "sdr"},
        }
        args = SimpleNamespace(model_family="sdxl", disable_bucket_pruning=False, train_batch_size=1)
        accelerator = SimpleNamespace(
            num_processes=1, process_index=0, is_main_process=True, device="cpu", wait_for_everyone=lambda: None
        )
        with (
            patch.object(
                StateTracker, "get_data_backend_config", side_effect=lambda id: native if id == self.backend.id else derived
            ),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            metadata = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "derived-source-buckets"),
                metadata_file=str(self.root / "derived-source-geometry"),
                data_backend=self.backend,
                accelerator=accelerator,
                batch_size=1,
                resolution=8,
                resolution_type="pixel",
            )
            self.addCleanup(metadata.close)
            metadata.compute_aspect_ratio_bucket_indices()
            source = DatasetRuntime(self.backend.id, self.backend, metadata, native)
            store = LocalDataBackend(accelerator, "derived")
            provider = create_metadata_provider(
                store.id,
                derived,
                store,
                accelerator,
                args,
                instance_data_dir=str(root),
                cache_root=str(root),
                aligned_source=source,
            )
            target = DatasetRuntime(store.id, store, provider, derived)
            refs, count = source.read_page(limit=2)
            self.assertEqual(count, 2)
            with (
                patch.object(self.backend, "list_files", side_effect=AssertionError("source list materialized")),
                patch.object(store, "list_files", side_effect=AssertionError("target list materialized")),
            ):
                DataGenerator(source, target, accelerator, num_workers=0).generate_dataset()
                with patch.object(self.backend, "read_image_batch", side_effect=AssertionError("warm source payload read")):
                    DataGenerator(source, target, accelerator, num_workers=0).generate_dataset()

            targets, target_count = target.read_page(limit=2)
            self.assertEqual(target_count, 2)
            self.assertEqual(targets, [target.conditioning_ref(ref) for ref in refs])
            self.assertTrue(all(Path(path).is_file() for path in targets))
            records = target.read_samples(targets)
            self.assertEqual([record["training_sample_path"] for record in records], refs)
            self.assertEqual(
                [record["target_size"] for record in records],
                [record["target_size"] for record in source.read_samples(refs)],
            )
            self.assertEqual(provider.aspect_ratio_bucket_indices, {})
            self.assertEqual(metadata.aspect_ratio_bucket_indices, {})
            self.assertEqual(provider.image_metadata, {})

            with closing(sqlite3.connect("")) as db:
                changes = Exclusions(db)
                changes.add(refs[0], next(iter(metadata.published_bucket_counts())), "decode")
                _publish(self.backend._cache_path(metadata.cache_file), source.describe().view_revision, db)
            metadata.reload_cache()
            with self.assertRaisesRegex(ValueError, "Source membership changed"):
                target.read_page(limit=2)

    def test_capture_publication_reuses_only_unchanged_groups(self):
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.metadata.backends.webshart_exclusions import Exclusions, _publish
        from simpletuner.helpers.metadata.backends.webshart_geometry import snapshot_path
        from simpletuner.helpers.training.state_tracker import StateTracker

        second = self.source / "second.parquet"
        pq.write_table(pa.Table.from_pylist(self.rows), second, row_group_size=2)

        config = {
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 8,
            "resolution_type": "pixel",
        }
        args = SimpleNamespace(model_family="sdxl", disable_bucket_pruning=False)
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)

        def build(*, cap=None, refresh=False, unopened=()):
            source = WebshartDataBackend(
                None,
                "capture",
                str(self.source),
                manifest="capture.json",
                format="parquet",
                cache_dir=str(self.root / "cache"),
            )
            source.prepared_partition_shards = 1
            metadata = WebshartMetadataBackend(
                id=source.id,
                instance_data_dir=source.cache_dir,
                cache_file=str(self.root / "buckets"),
                metadata_file=str(self.root / "geometry"),
                data_backend=source,
                accelerator=accelerator,
                batch_size=1,
                resolution=config["resolution"],
                resolution_type="pixel",
                max_num_samples=cap,
            )
            self.addCleanup(metadata.close)
            original_open = source.dataset._file

            def open_shard(shard, **kwargs):
                self.assertNotIn(shard, unopened)
                return original_open(shard, **kwargs)

            with (
                patch.object(metadata, "_scan_rows", wraps=metadata._scan_rows) as scan,
                patch.object(source.dataset, "_file", side_effect=open_shard),
            ):
                metadata.compute_aspect_ratio_bucket_indices(ignore_existing_cache=refresh)
            return metadata, [call.kwargs["shard_indices"] for call in scan.call_args_list]

        def first_partition(metadata):
            path = metadata.data_backend._cache_path(metadata.metadata_file)
            with closing(sqlite3.connect(snapshot_path(path, metadata._published["geometry_sha256"]))) as db:
                (digest,) = db.execute("SELECT digest FROM partitions WHERE id=0").fetchone()
            return snapshot_path(path, digest)

        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            self._publish_capture_manifest()
            before, scanned = build()
            self.assertCountEqual(scanned, [(0,), (1,)])
            self.assertEqual(before.read_page(limit=10)[1], 4)

            updated_rows = [dict(row) for row in self.rows]
            updated_rows[-1]["captions"] = "   "
            pq.write_table(pa.Table.from_pylist(updated_rows), second, row_group_size=2)
            self._publish_capture_manifest()
            after, scanned = build(unopened=(0,))
            self.assertEqual(scanned, [(1,)])
            refs, total = after.read_page(limit=10)
            self.assertEqual(total, 3)
            self.assertEqual(after.filtering_statistics["total_processed"], 3)
            self.assertEqual(after.filtering_statistics["skipped"]["caption_missing"], 1)
            skipped = after.data_backend.sample_id(1, 2, "second.parquet/000000000002.jpg")
            self.assertIsNone(after.data_backend.get_caption(skipped))
            self.assertNotIn(skipped, refs)
            self.assertEqual(refs[:2], before.read_page(limit=10)[0][:2])
            self.assertEqual(after.get_metadata_by_filepath(refs[0]), before.get_metadata_by_filepath(refs[0]))
            self.assertNotEqual(after._published["source_revision"], before._published["source_revision"])
            self.assertEqual(after._published["physical_rows"], 6)
            self.assertTrue(first_partition(before).samefile(first_partition(after)))

            clean, scanned = build(refresh=True)
            self.assertCountEqual(scanned, [(0,), (1,)])
            self.assertEqual(clean._published["geometry_sha256"], after._published["geometry_sha256"])
            self.assertEqual(clean.filtering_statistics, after.filtering_statistics)
            self.assertTrue(first_partition(before).samefile(first_partition(clean)))

            config["resolution"] = 16
            resized, scanned = build()
            self.assertCountEqual(scanned, [(0,), (1,)])
            self.assertNotEqual(resized._published["bucket_counts"], before._published["bucket_counts"])

            with closing(sqlite3.connect("")) as db:
                changes = Exclusions(db)
                bucket = next(iter(resized.published_bucket_counts()))
                changes.add(refs[-1], bucket, "decode")
                publication = resized.data_backend._cache_path(resized.cache_file)
                _publish(publication, resized.describe().view_revision, db)

            config["resolution"] = 8
            filtered, scanned = build(unopened=(0,))
            self.assertEqual(scanned, [(1,)])
            self.assertEqual(filtered.read_page(limit=10), (refs[:2], 2))
            self.assertEqual(filtered._published["exclusion_counts"], {"decode": 1})

            capped, scanned = build(cap=1)
            self.assertEqual(scanned, [(0,)])
            self.assertEqual(capped.read_page(limit=10)[1], 1)

    def test_concurrent_views_share_cold_capture_group(self):
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        self._publish_capture_manifest()
        config = {
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 8,
            "resolution_type": "pixel",
        }
        args = SimpleNamespace(model_family="sdxl", disable_bucket_pruning=False)
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        ready = Barrier(2, timeout=10)

        def make_metadata(view):
            source = WebshartDataBackend(
                None,
                f"concurrent-{view}",
                str(self.source),
                manifest="capture.json",
                format="parquet",
                cache_dir=str(self.root / "shared-cache"),
            )
            metadata = WebshartMetadataBackend(
                id=source.id,
                instance_data_dir=source.cache_dir,
                cache_file=str(self.root / f"{view}-buckets"),
                metadata_file=str(self.root / f"{view}-geometry"),
                data_backend=source,
                accelerator=accelerator,
                batch_size=1,
                resolution=8,
                resolution_type="pixel",
            )
            self.addCleanup(metadata.close)
            return metadata

        def align_lookup(lookup):
            def resolve(*args):
                path = lookup(*args)
                self.assertIsNotNone(path)
                ready.wait()
                return path

            return resolve

        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            first, second = (make_metadata(view) for view in ("first", "second"))
            with (
                patch.object(first, "_partition_cache_path", side_effect=align_lookup(first._partition_cache_path)),
                patch.object(second, "_partition_cache_path", side_effect=align_lookup(second._partition_cache_path)),
                patch.object(first, "_scan_rows", wraps=first._scan_rows) as first_scan,
                patch.object(second, "_scan_rows", wraps=second._scan_rows) as second_scan,
                ThreadPoolExecutor(max_workers=2) as pool,
            ):
                tasks = [pool.submit(view.compute_aspect_ratio_bucket_indices) for view in (first, second)]
                for task in tasks:
                    task.result(timeout=15)

            self.assertEqual(first_scan.call_count + second_scan.call_count, 1)
            self.assertEqual(first.read_page(limit=8), second.read_page(limit=8))
            self.assertEqual(first.describe().view_revision, second.describe().view_revision)

    def test_parallel_cold_preparation_resumes_after_a_worker_failure(self):
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        for index in range(3):
            shard = self.source / f"additional-{index}.parquet"
            pq.write_table(pa.Table.from_pylist(self.rows), shard, row_group_size=2)
        self._publish_capture_manifest()
        source = WebshartDataBackend(
            None,
            "parallel",
            str(self.source),
            manifest="capture.json",
            format="parquet",
            parallel_downloads=2,
            cache_dir=str(self.root / "parallel"),
        )
        if not getattr(source.dataset, "supports_parallel_metadata_reads", False):
            self.skipTest("Requires a Webshart reader with synchronized capture catalogue access")
        source.prepared_partition_shards = 1
        config = {
            "caption_strategy": "webshart",
            "webshart": {"sampling_policy": "global"},
            "resolution": 8,
            "resolution_type": "pixel",
        }
        args = SimpleNamespace(model_family="sdxl", disable_bucket_pruning=False)
        accelerator = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
        ):
            metadata = WebshartMetadataBackend(
                id=source.id,
                instance_data_dir=source.cache_dir,
                cache_file=str(self.root / "parallel" / "buckets"),
                metadata_file=str(self.root / "parallel" / "geometry"),
                data_backend=source,
                accelerator=accelerator,
                batch_size=1,
                resolution=8,
                resolution_type="pixel",
            )
            self.addCleanup(metadata.close)
            first_started = Event()
            second_completed = Event()
            original_scan = metadata._scan_rows

            def fail_first_group(*args, shard_indices, **kwargs):
                if shard_indices == (0,):
                    first_started.set()
                    self.assertTrue(second_completed.wait(15), "second group did not run while the first was blocked")
                    raise OSError("first group could not read source")
                self.assertEqual(shard_indices, (1,), "scheduled past the worker window")
                self.assertTrue(first_started.wait(15))
                yield from original_scan(*args, shard_indices=shard_indices, **kwargs)
                second_completed.set()

            with (
                patch.object(metadata, "_scan_rows", side_effect=fail_first_group),
                self.assertRaisesRegex(OSError, "first group could not read source"),
            ):
                metadata.compute_aspect_ratio_bucket_indices()
            self.assertFalse(source.exists(metadata.cache_file))

            owner = get_ident()
            progress = []

            def report(current, total):
                self.assertEqual(get_ident(), owner)
                progress.append((current, total))

            with patch.object(metadata, "_scan_rows", wraps=original_scan) as scan:
                metadata.compute_aspect_ratio_bucket_indices(progress_callback=report)
            self.assertCountEqual([call.kwargs["shard_indices"] for call in scan.call_args_list], [(0,), (2,), (3,)])
            self.assertEqual(progress, [(1, 4), (2, 4), (3, 4), (4, 4)])
            refs, total = metadata.read_page(limit=10)
            self.assertEqual(total, 8)
            self.assertEqual([source.parse_sample_id(ref).shard_idx for ref in refs], [0, 0, 1, 1, 2, 2, 3, 3])
            result = (metadata.describe(), metadata.filtering_statistics)

            source.parallel_downloads = 1
            metadata.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
            self.assertEqual((metadata.describe(), metadata.filtering_statistics), result)

            source.parallel_downloads = 2
            publication = source._cache_path(metadata.cache_file)
            inode = publication.stat().st_ino

            def cancel(current, total):
                self.assertEqual(get_ident(), owner)
                raise RuntimeError("scan cancelled")

            with self.assertRaisesRegex(RuntimeError, "scan cancelled"):
                metadata.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True, progress_callback=cancel)
            self.assertEqual(publication.stat().st_ino, inode)
            self.assertEqual(metadata.read_page(limit=10), (refs, total))

    def test_caption_native_and_local_schedules_resume_one_prefetched_mixer(self):
        from accelerate import PartialState
        from torch.utils.data import DataLoader

        from simpletuner.helpers.data_backend.caption_dataset import CaptionDataset
        from simpletuner.helpers.data_backend.caption_sampler import CaptionSampler
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.data_backend.local import LocalDataBackend
        from simpletuner.helpers.data_backend.runtime.batch_fetcher import BatchFetcher
        from simpletuner.helpers.data_backend.runtime.dataloader_iterator import DatasetMixer
        from simpletuner.helpers.metadata.backends.discovery import DiscoveryMetadataBackend
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
        from simpletuner.helpers.training.caption_collate import collate_caption_batch
        from simpletuner.helpers.training.state_tracker import StateTracker
        from tests.helpers.data_backend.test_caption_pipeline import DummyCaptionMetadataBackend

        accelerator = PartialState()
        config = {
            "webshart": {"sampling_policy": "shard_local"},
            "caption_strategy": "webshart",
            "resolution": 8,
            "resolution_type": "pixel",
            "slider_strength": -1,
        }
        args = SimpleNamespace(
            model_family="sdxl",
            disable_bucket_pruning=False,
            gradient_accumulation_steps=1,
            max_train_steps=0,
            seed=9,
            print_sampler_statistics=False,
        )

        class SampleBatches:
            def __init__(self, sampler):
                self.sampler = sampler

            def __iter__(self):
                for (record,) in self.sampler:
                    yield {
                        key: record[key]
                        for key in ("data_backend_id", "image_path", "dataset_continuation", "caption_dropout_seed")
                        if key in record
                    }

        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
            patch.object(StateTracker, "epoch", 1),
            patch.object(StateTracker, "repeats", {}),
            patch.object(StateTracker, "exhausted_backends", []),
        ):
            metadata = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "mixed-buckets"),
                metadata_file=str(self.root / "mixed-geometry"),
                data_backend=self.backend,
                accelerator=accelerator,
                batch_size=1,
                resolution=8,
                resolution_type="pixel",
            )
            self.addCleanup(metadata.close)
            metadata.compute_aspect_ratio_bucket_indices()
            captions = DummyCaptionMetadataBackend(count=1)

            def make_runtimes():
                images = DatasetRuntime(self.backend.id, self.backend, metadata, config)
                images.sampler = MultiAspectSampler(images, model=None, accelerator=accelerator, batch_size=1)
                self.addCleanup(images.sampler.close)
                images.training_dataloader = SampleBatches(images.sampler)
                text = DatasetRuntime("captions", None, captions, {"dataset_type": "caption", "slider_strength": 1})
                text.sampler = CaptionSampler("captions", captions, accelerator, 1, shuffle=False, track_consumed=True)
                text.training_dataloader = DataLoader(
                    CaptionDataset("captions", captions),
                    batch_size=1,
                    sampler=text.sampler,
                    collate_fn=collate_caption_batch,
                )

                local_storage = LocalDataBackend(accelerator, "local")
                local_metadata = DiscoveryMetadataBackend(
                    id="local",
                    instance_data_dir=str(self.root),
                    cache_file=str(self.root / "local-buckets"),
                    metadata_file=str(self.root / "local-geometry"),
                    data_backend=local_storage,
                    accelerator=accelerator,
                    batch_size=1,
                    resolution=8,
                    resolution_type="pixel",
                )
                ref = str(self.root / "local.jpg")
                local_metadata.aspect_ratio_bucket_indices = {"1.0": [ref]}
                local_metadata.image_metadata = {ref: {"original_size": (8, 8), "crop_coordinates": (0, 0)}}
                local = DatasetRuntime("local", local_storage, local_metadata, {"repeats": 1})
                local.sampler = MultiAspectSampler(
                    local, model=None, accelerator=accelerator, batch_size=1, track_consumed=True
                )
                self.addCleanup(local.sampler.close)
                local.training_dataloader = SampleBatches(local.sampler)
                return {dataset.id: dataset for dataset in (images, text, local)}

            def fetch(runtimes, saved=None):
                backends = dict(runtimes)
                for dataset in backends.values():
                    dataset.start_epoch(1)
                    self.assertTrue(dataset.supports_consumed_checkpoint)
                mixer = DatasetMixer(backends, epoch=1, step=0, saved=saved)
                fetcher = BatchFetcher(step=mixer.step, max_size=8, datasets=backends, mixer=mixer)
                fetcher.fetch_responses()
                return fetcher

            original = make_runtimes()
            fetcher = fetch(original)
            consumed = [fetcher.next_response(step) for step in (1, 2, 3)]
            self.assertEqual([batch["data_backend_id"] for batch in consumed], ["captions", self.backend.id, "local"])
            for batch in consumed:
                original[batch["data_backend_id"]].acknowledge_batch(batch["dataset_continuation"])
            checkpoint = str(self.root / "mixed-train-state.json")
            for dataset in original.values():
                dataset.sampler.save_state(checkpoint)
            expected = [fetcher.next_response(step) for step in (4, 5, 6)]
            self.assertCountEqual([batch["data_backend_id"] for batch in expected[:-1]], [self.backend.id, "local"])
            self.assertIs(expected[-1], False)

            resumed = make_runtimes()
            for dataset in resumed.values():
                dataset.sampler.load_states(checkpoint)
            replay = fetch(resumed, saved=consumed[-1]["dataset_mixer_state"])
            self.assertEqual([replay.next_response(step) for step in (4, 5, 6)], expected)

    def test_native_scan_partition_sampler_and_consumed_checkpoint(self):
        from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler
        from simpletuner.helpers.training.state_tracker import StateTracker

        config = dict(
            id=self.backend.id,
            type="webshart",
            webshart={"sampling_policy": "shard_local"},
            caption_strategy="webshart",
            resolution=64,
            resolution_type="pixel",
            crop=False,
        )
        args = SimpleNamespace(
            seed=32,
            model_family="sdxl",
            disable_bucket_pruning=False,
            gradient_accumulation_steps=1,
            max_train_steps=0,
            allow_dataset_oversubscription=False,
        )
        accel = SimpleNamespace(num_processes=1, process_index=0, is_main_process=True)
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
            patch.object(StateTracker, "get_s2v_datasets", return_value=[]),
            patch.object(self.backend.source_files, "publish", side_effect=AssertionError("corpus inventory")),
            patch.object(self.backend, "read", side_effect=AssertionError("whole geometry or membership read")),
        ):
            metadata = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "buckets"),
                metadata_file=str(self.root / "metadata"),
                data_backend=self.backend,
                accelerator=accel,
                batch_size=1,
                resolution=64,
                resolution_type="pixel",
            )
            metadata.compute_aspect_ratio_bucket_indices()
            with (
                patch.object(metadata, "max_num_samples", 1),
                patch.object(metadata, "dataset_filter", DatasetFilter({"path_include": "000000000002"})),
                patch.object(metadata, "_all_shard_indices", return_value=range(2)),
                patch.object(
                    self.backend.loader, "iter_samples_in_shard", wraps=self.backend.loader.iter_samples_in_shard
                ) as project,
            ):
                scanned = list(metadata._scan_rows(set(), {"total_processed": 0}, True, None))
            self.assertEqual([self.backend.parse_sample_id(ref).sample_idx for _, ref, _ in scanned], [2])
            self.assertEqual([call.args[0] for call in project.call_args_list], [0])
            self.assertEqual(metadata.describe().accepted_samples, 2)
            with patch.object(metadata, "_scan_rows", side_effect=AssertionError("repeated scan")):
                metadata.compute_aspect_ratio_bucket_indices()
            metadata.split_buckets_between_processes()
            self.assertEqual(metadata.image_metadata, {})
            self.assertEqual(metadata.aspect_ratio_bucket_indices, {})
            runtime = DatasetRuntime(self.backend.id, self.backend, metadata, config)
            partitions = [
                [
                    ref
                    for _, batch in runtime.iter_preparation_batches(worker=worker, workers=3, batch_size=1)
                    for ref in batch
                ]
                for worker in range(3)
            ]
            self.assertEqual([len(p) for p in partitions], [1, 1, 0])
            self.assertEqual(len(set(partitions[0] + partitions[1])), 2)
            ref = partitions[0][0]
            other = partitions[1][0]
            from simpletuner.helpers.metadata.backends import webshart_geometry

            statements = []
            open_snapshot = webshart_geometry.read_snapshot

            def trace_snapshot(*args):
                db = open_snapshot(*args)
                db.set_trace_callback(statements.append)
                return db

            with patch.object(webshart_geometry, "read_snapshot", side_effect=trace_snapshot):
                rows = runtime.read_samples([other, ref, other], fields=("metadata",))
            self.assertEqual([row["image_path"] for row in rows], [other, ref, other])
            self.assertIsNot(rows[0]["webshart"], rows[2]["webshart"])
            self.assertEqual(sum("FROM geometry" in sql for sql in statements), 1)
            weights = self.root / "model" / "vae" / "weights.safetensors"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"first")
            encoder = SimpleNamespace(config=SimpleNamespace(vae_path=str(weights.parent.parent), revision="v1"))
            key = metadata.image_artifact_key(ref, "vae", encoder)
            with patch("simpletuner.helpers.metadata.backends.webshart.file_digest", side_effect=AssertionError("rehash")):
                self.assertEqual(metadata.image_artifact_key(ref, "vae", encoder), key)
            encoder.config.revision = "v2"
            self.assertNotEqual(metadata.image_artifact_key(ref, "vae", encoder), key)
            encoder.config.revision = "v1"
            weights.write_bytes(b"other")
            changed_weights = metadata.image_artifact_key(ref, "vae", encoder)
            self.assertNotEqual(changed_weights, key)
            with patch.dict(metadata.dataset_config, crop_style="corner"):
                self.assertNotEqual(metadata.image_artifact_key(ref, "vae", encoder), changed_weights)
            processor_file = self.root / "processor" / "preprocessor_config.json"
            processor_file.parent.mkdir()
            processor_file.write_text('{"size": 64}')
            encoder.config.image_encoder_pretrained_model_name_or_path = str(weights.parent)
            encoder.config.image_processor_pretrained_model_name_or_path = str(processor_file.parent)
            encoder.config.image_encoder_subfolder = ["vision"]
            original = metadata.image_artifact_key(ref, "conditioning", encoder)
            processor_file.write_text('{"size": 128}')
            metadata._artifact_models.clear()
            self.assertNotEqual(metadata.image_artifact_key(ref, "conditioning", encoder), original)
            sampler = MultiAspectSampler(runtime, model=None, accelerator=accel, batch_size=1)
            runtime.sampler = sampler
            self.addCleanup(metadata.close)
            self.addCleanup(sampler.close)
            self.assertEqual(len(sampler), 2)
            from simpletuner.helpers.metadata.backends.webshart_geometry import prepared_geometry_path
            from simpletuner.helpers.metadata.backends.webshart_membership import IndexedMembership

            publication = self.backend._cache_path(metadata.cache_file)
            original_index = prepared_geometry_path(publication, metadata._published)
            original_stat = original_index.stat()
            self.assertNotIn("aspect_ratio_bucket_indices", json.loads(publication.read_text()))
            with patch.object(IndexedMembership, "iter_refs", side_effect=AssertionError("startup scanned membership")):
                sampler._open_training_cursor()
            first = next(iter(sampler))[0]
            second = next(iter(sampler))[0]
            self.assertNotEqual(first["image_path"], second["image_path"])
            runtime.acknowledge_batch(first["dataset_continuation"])
            name = str(self.root / "checkpoint.json")
            validation_first = sampler.validation_cursor.next_refs(1)
            sampler.save_state(name)
            validation_second = sampler.validation_cursor.next_refs(1)
            self.assertNotEqual(validation_first, validation_second)
            sampler.load_states(name)
            self.assertEqual(sampler.validation_cursor.next_refs(1), validation_second)
            again = next(iter(sampler))[0]
            self.assertEqual(again, second)
            import random

            training_state = sampler.training_cursor.state_dict()
            validation_state = sampler.validation_cursor.state_dict()
            with (
                patch.object(runtime, "resolve_caption", return_value=["alpha", "beta", "gamma"]),
                patch.object(sampler, "debug_log"),
            ):
                random_state = random.getstate()
                first_preview = sampler.retrieve_validation_set(1)
                self.assertEqual(random.getstate(), random_state)
                sampler.validation_cursor.load_state_dict(validation_state)
                same_preview = sampler.retrieve_validation_set(1)
            self.assertEqual(first_preview[0][:3], same_preview[0][:3])
            self.assertEqual(first_preview[0][3].tobytes(), same_preview[0][3].tobytes())
            self.assertEqual(sampler.training_cursor.state_dict(), training_state)
            sampler.reset_evaluation()
            self.assertEqual(next(iter(sampler))[0], first)
            sampler.reset_evaluation()
            metadata.reload_cache()
            with metadata.open_preparation_view(worker=1, workers=2) as view:
                refs = list(view)
                self.assertEqual(len(refs), 1)
                self.assertIn("target_size", view.metadata.get_metadata_by_filepath(refs[0]))

            bucket = next(iter(metadata.published_bucket_counts()))
            all_refs, total = runtime.read_page(bucket=bucket)
            self.assertEqual(total, 2)
            spec = metadata.epoch_spec(1, 1)
            self.assertEqual(runtime.describe(spec).scheduled_occurrences, 2)
            old_publication = metadata._published
            with (
                patch.object(IndexedMembership, "iter_refs", side_effect=AssertionError("filtering scanned membership")),
                metadata.collect_exclusions() as ledger,
            ):
                ledger.add(all_refs[1], bucket, "nsfw")
            with patch.object(metadata, "_published", old_publication):
                with self.assertRaisesRegex(ValueError, "membership changed"):
                    len(sampler)
                with self.assertRaisesRegex(ValueError, "membership changed"), metadata.open_preparation_view():
                    self.fail("opened preparation with stale source pins")
            self.assertEqual(len(sampler), 1)
            self.assertEqual(prepared_geometry_path(publication, metadata._published), original_index)
            self.assertEqual(original_index.stat().st_mtime_ns, original_stat.st_mtime_ns)
            self.assertNotEqual(metadata.describe().view_revision, old_publication["view_revision"])
            with patch.object(
                IndexedMembership, "iter_refs", side_effect=AssertionError("startup scanned filtered membership")
            ):
                with closing(metadata.open_epoch(spec)) as cursor:
                    self.assertEqual(cursor.batches, 1)
            with self.assertRaisesRegex(ValueError, "accepted Webshart view"):
                sampler.validation_cursor.load_state_dict(validation_state)
            sampler.validation_cursor.close()
            self.assertEqual(runtime.read_page(), ([all_refs[0]], 1))
            metadata.compute_aspect_ratio_bucket_indices(ignore_existing_cache=True)
            self.assertEqual(runtime.read_page(), ([all_refs[0]], 1))
            with self.assertRaisesRegex(ValueError, "unchanged"):
                sampler.load_states(name)

            replacement = self.root / "replacement.parquet"
            original = self.source / "part.parquet"
            rows = [{**self.rows[0], "captions": "other"}, *self.rows[1:]]
            pq.write_table(pa.Table.from_pylist(rows), replacement, row_group_size=2)
            self.assertEqual(replacement.stat().st_size, original.stat().st_size)
            replacement.replace(original)
            metadata.reload_cache()
            for policy in ("global", "shard_local"):
                with closing(metadata.open_epoch(replace(spec, policy=policy))) as cursor:
                    _, refs = cursor.next_batch()
                    with self.assertRaisesRegex(ValueError, "object changed since the accepted dataset view"):
                        self.backend.read_caption_batch(refs)

    def test_text_preparation_after_real_rank_split_still_covers_all_accepted_rows(self):
        from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
        from simpletuner.helpers.models.common import TextEmbedCacheKey
        from simpletuner.helpers.training.state_tracker import StateTracker
        from simpletuner.simpletuner_sdk.server.services.cache_job_service import CacheJobService

        config = dict(id=self.backend.id, caption_strategy="webshart", resolution=64, resolution_type="pixel", crop=False)
        args = SimpleNamespace(seed=123, disable_bucket_pruning=False, allow_dataset_oversubscription=False)
        accelerator = SimpleNamespace(num_processes=2, process_index=1, is_main_process=False)
        model = SimpleNamespace(text_embed_cache_key=lambda: TextEmbedCacheKey.CAPTION)
        with (
            patch.object(StateTracker, "get_data_backend_config", return_value=config),
            patch.object(StateTracker, "get_args", return_value=args),
            patch.object(StateTracker, "get_model", return_value=None),
            patch.object(self.backend.loader, "load_sample", side_effect=AssertionError("unexpected payload read")),
        ):
            metadata = WebshartMetadataBackend(
                id=self.backend.id,
                instance_data_dir=self.backend.cache_dir,
                cache_file=str(self.root / "buckets"),
                metadata_file=str(self.root / "metadata"),
                data_backend=self.backend,
                accelerator=accelerator,
                batch_size=1,
                resolution=64,
                resolution_type="pixel",
            )
            if self.backend.supports_caption_projection:
                with patch.object(self.backend.dataset, "_metadata", side_effect=AssertionError("entire shard loaded")):
                    metadata.compute_aspect_ratio_bucket_indices()
                self.assertEqual(self.backend._shard_metadata_cache, {})
            else:
                metadata.compute_aspect_ratio_bucket_indices()
            metadata.split_buckets_between_processes(apply_padding=True)
            self.assertEqual(sum(map(len, metadata.aspect_ratio_bucket_indices.values())), 1)
            with patch.object(metadata, "read_cache", side_effect=AssertionError("read rank-local membership")):
                records = list(
                    CacheJobService._iter_prompt_records(
                        model, CacheJobService._text_dataset(config, self.backend, metadata)
                    )
                )
            self.assertEqual([r["prompt"] for r in records], ["first", "third"])

            from simpletuner.helpers.data_backend.dataset_runtime import DatasetRuntime
            from simpletuner.helpers.data_backend.factory import FactoryRegistry

            args.output_dir = str(self.root)
            args.caption_dropout_probability = 0
            accelerator.device = "cpu"
            model.uses_text_embeddings_cache = lambda: True
            factory = FactoryRegistry.__new__(FactoryRegistry)
            factory.args, factory.accelerator, factory.model = args, accelerator, model
            factory.text_encoders = None
            factory._deferred_text_embed_backends = []
            runtime = DatasetRuntime(self.backend.id, self.backend, metadata, config)
            with (
                patch.object(metadata, "read_cache", side_effect=AssertionError("rank-local inventory")),
                patch(
                    "simpletuner.helpers.metadata.backends.webshart.open_geometry", side_effect=AssertionError("geometry")
                ),
            ):
                cursor = runtime.open_validation()
                try:
                    selected = cursor.next_refs(1)
                    self.assertEqual(self.backend.parse_sample_id(selected[0]).sample_idx, 0)
                    self.assertEqual([self.backend.parse_sample_id(ref).sample_idx for ref in cursor.next_refs(5)], [2, 0])
                finally:
                    cursor.close()
            prepared = []
            cache = SimpleNamespace(
                text_cache_ondemand=False,
                read_batch_size=8,
                text_encoder_batch_size=2,
                precompute_embeddings=lambda stream, **kwargs: prepared.extend(stream),
            )
            runtime.text_embed_cache = cache
            bundle = dict(
                id=runtime.id,
                config=dict(config),
                instance_data_dir=None,
                data_backend=self.backend,
                metadata_backend=metadata,
                dataset_runtime=runtime,
                text_embed_cache=cache,
            )
            with (
                patch("simpletuner.helpers.prompts.PromptHandler.get_all_captions", side_effect=AssertionError("inventory")),
                patch.object(StateTracker, "set_data_backend_config"),
            ):
                for deferred in (False, True):
                    prepared.clear()
                    bundle["config"]["conditioning_data"] = ["reference"] if deferred else []
                    model.requires_text_embed_image_context = lambda: deferred
                    for rank in range(2):
                        accelerator.process_index = rank
                        factory._process_text_embeddings(config, bundle, None)
                        if deferred:
                            self.assertEqual(len(factory._deferred_text_embed_backends), 1)
                            factory._process_deferred_text_embeddings()
                        self.assertEqual(prepared[-1], records[rank])
                    self.assertEqual(prepared, records)

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
