import json
import tarfile
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.data_backend.filters import DatasetFilter
from simpletuner.helpers.data_backend.webshart import WebshartDataBackend
from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend


class TestWebshartDataBackend(unittest.TestCase):
    @patch("simpletuner.helpers.data_backend.webshart.random.uniform", return_value=0.0)
    @patch("simpletuner.helpers.data_backend.webshart.time.sleep")
    def test_read_sample_bytes_retries_rate_limit(self, sleep, _uniform):
        backend = WebshartDataBackend.__new__(WebshartDataBackend)
        backend.loader = Mock(
            load_sample=Mock(side_effect=[RuntimeError("Rate limit exceeded"), SimpleNamespace(data=b"sample")])
        )

        result = backend._read_sample_bytes("webshart://2/7/sample.mp4")

        self.assertEqual(result, b"sample")
        self.assertEqual(backend.loader.load_sample.call_count, 2)
        sleep.assert_called_once_with(1.0)

    @patch("simpletuner.helpers.data_backend.webshart.requests.get")
    def test_read_sample_head_tail_uses_tar_member_ranges(self, requests_get):
        head_response = Mock(status_code=206, content=b"head")
        tail_response = Mock(status_code=206, content=b"tailtail")
        requests_get.side_effect = [head_response, tail_response]
        backend = WebshartDataBackend.__new__(WebshartDataBackend)
        backend.hf_token = None
        backend.dataset = Mock()
        backend.dataset.get_shard_info.return_value = {"tar_path": "https://example.test/shard.tar"}
        file_metadata = {"offset": 1000, "length": 200000}

        head, tail, length = backend.read_sample_head_tail(
            "webshart://2/7/sample.mp4",
            file_metadata=file_metadata,
            head_bytes=4,
            tail_bytes=8,
        )

        self.assertEqual((head, tail, length), (b"head", b"tailtail", 200000))
        self.assertEqual(requests_get.call_args_list[0].kwargs["headers"]["Range"], "bytes=1000-1003")
        self.assertEqual(requests_get.call_args_list[1].kwargs["headers"]["Range"], "bytes=200992-200999")
        head_response.close.assert_called_once()
        tail_response.close.assert_called_once()

    def test_zero_shard_cache_size_disables_whole_shard_cache(self):
        dataset = Mock()
        loader = Mock()
        loader.list_shard_sample_aspect_buckets = Mock()
        webshart = Mock()
        webshart.discover_dataset.return_value = dataset
        webshart.TarDataLoader.return_value = loader

        with TemporaryDirectory() as cache_dir, patch.dict("sys.modules", {"webshart": webshart}):
            backend = WebshartDataBackend(
                accelerator=None,
                id="range-reads-only",
                source="source",
                cache_dir=cache_dir,
                shard_cache_gb=0,
            )

        self.assertIsNone(backend.shard_cache_dir)
        dataset.enable_shard_cache.assert_not_called()

    def test_compressed_torch_cache_round_trip_allows_internal_metadata(self):
        with TemporaryDirectory() as cache_dir:
            backend = WebshartDataBackend.__new__(WebshartDataBackend)
            backend.cache_dir = cache_dir
            backend.id = "cache-round-trip"
            backend.compress_cache = True
            value = {"latents": torch.ones(1), "metadata": "sample.mp4"}

            backend.torch_save(value, "sample.pt")
            loaded = backend.torch_load("sample.pt")

        self.assertTrue(torch.equal(loaded["latents"], value["latents"]))
        self.assertEqual(loaded["metadata"], value["metadata"])

    def test_torch_cache_load_resets_uncompressed_fallback_stream(self):
        with TemporaryDirectory() as cache_dir:
            backend = WebshartDataBackend.__new__(WebshartDataBackend)
            backend.cache_dir = cache_dir
            backend.id = "cache-fallback"
            backend.compress_cache = False
            backend.torch_save(torch.ones(1), "sample.pt")

            backend.compress_cache = True
            loaded = backend.torch_load("sample.pt")

        self.assertTrue(torch.equal(loaded, torch.ones(1)))

    def test_sample_id_round_trips_nested_filenames(self):
        sample_id = WebshartDataBackend.sample_id(2, 17, "nested/path/sample.webp")

        self.assertEqual(sample_id, "webshart://2/17/nested/path/sample.webp")
        ref = WebshartDataBackend.parse_sample_id(sample_id)
        self.assertEqual(ref.shard_idx, 2)
        self.assertEqual(ref.sample_idx, 17)
        self.assertEqual(ref.filename, "nested/path/sample.webp")

    def test_path_normalized_sample_id_round_trips(self):
        path_normalized_id = Path("webshart://2/17/nested/path/sample.mp4")

        ref = WebshartDataBackend.parse_sample_id(path_normalized_id)

        self.assertEqual(
            WebshartDataBackend.normalize_sample_id(path_normalized_id),
            "webshart://2/17/nested/path/sample.mp4",
        )
        self.assertTrue(WebshartDataBackend.is_sample_id(path_normalized_id))
        self.assertEqual(ref.shard_idx, 2)
        self.assertEqual(ref.sample_idx, 17)
        self.assertEqual(ref.filename, "nested/path/sample.mp4")

    def test_parse_sample_id_rejects_non_webshart_identifier(self):
        with self.assertRaises(ValueError):
            WebshartDataBackend.parse_sample_id(Path("/tmp/sample.webp"))

    def test_sample_id_classification_matches_suffix_semantics_without_path_objects(self):
        filenames = (
            "nested/sample.jpg",
            "sample.PT",
            "sample.safetensors",
            ".pt",
            "...pt",
            "sample.pt/",
            "sample.pt/.",
            "sample.pt/..",
            "sample.jpg/",
            "sample.",
            "nested/.sample.safetensors",
            "nested/../sample.webp",
        )
        for filename in filenames:
            for prefix in ("webshart://", "webshart:/"):
                identifier = f"{prefix}2/17/{filename}"
                expected = Path(filename).suffix.lower() not in {".pt", ".safetensors"}
                with (
                    self.subTest(identifier=identifier),
                    patch(
                        "simpletuner.helpers.data_backend.webshart.Path", side_effect=AssertionError("filesystem parsing")
                    ),
                ):
                    self.assertEqual(WebshartDataBackend.is_sample_id(identifier), expected)
        self.assertFalse(WebshartDataBackend.is_sample_id("local/sample.jpg"))

    def test_caption_sidecar_names_preserve_nested_and_hidden_filename_semantics(self):
        for filename in ("nested/sample.JPG", ".hidden", "...jpg", "sample", "sample."):
            with self.subTest(filename=filename):
                self.assertEqual(
                    WebshartDataBackend.caption_sidecar_filename(filename), str(Path(filename).with_suffix(".txt"))
                )

    def test_nested_txt_sidecar_uses_native_pairing_for_lookup_and_filtering(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "shard.tar"
            with tarfile.open(archive_path, "w") as archive:
                for name, payload in {"nested/sample.jpg": b"image", "nested/sample.txt": b" caption text "}.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, BytesIO(payload))
            with tarfile.open(archive_path) as archive:
                files = {info.name: {"offset": info.offset_data, "length": info.size} for info in archive.getmembers()}
            files["nested/sample.jpg"].update(width=16, height=16)
            (root / "shard.json").write_text(
                json.dumps({"path": "shard.tar", "filesize": archive_path.stat().st_size, "files": files})
            )
            backend = WebshartDataBackend(None, "nested", directory, cache_dir=str(root / "cache"), shard_cache_gb=0)
            sample_id = backend.sample_id(0, 0, "nested/sample.jpg")
            with patch.object(backend, "_read_sample_bytes", side_effect=AssertionError("image payload read")):
                self.assertEqual(backend.get_caption(sample_id), "caption text")
            metadata = WebshartMetadataBackend.__new__(WebshartMetadataBackend)
            metadata.data_backend = backend
            self.assertTrue(
                metadata._entry_has_caption(backend.get_shard_metadata(0), {"filename": "nested/sample.jpg"}, sample_id)
            )

    @patch("simpletuner.helpers.data_backend.webshart.load_video")
    def test_read_image_decodes_video_samples(self, load_video):
        backend = WebshartDataBackend.__new__(WebshartDataBackend)
        backend.dataset_type = DatasetType.VIDEO
        backend.read = Mock(return_value=Mock())
        load_video.return_value = "decoded-video"

        result = backend.read_image("webshart://0/3/sample.mp4")

        self.assertEqual(result, "decoded-video")
        load_video.assert_called_once_with(backend.read.return_value)

    def test_get_caption_uses_indexed_sample_metadata(self):
        backend = WebshartDataBackend.__new__(WebshartDataBackend)
        backend.caption_key = None
        backend.get_shard_metadata = Mock(return_value={"sample.mp4": {"captions": "A moving subject."}})

        caption = backend.get_caption("webshart://2/7/sample.mp4")

        self.assertEqual(caption, "A moving subject.")

    def test_get_caption_preserves_indexed_caption_variants(self):
        backend = WebshartDataBackend.__new__(WebshartDataBackend)
        backend.caption_key = None
        backend.get_shard_metadata = Mock(return_value={"sample.mp4": {"captions": ["first caption", "second caption"]}})

        caption = backend.get_caption("webshart://2/7/sample.mp4")

        self.assertEqual(caption, ["first caption", "second caption"])

    def test_filtered_aspect_bucket_call_uses_webshart_api(self):
        backend = WebshartDataBackend.__new__(WebshartDataBackend)
        backend.loader = Mock()
        backend.loader.list_shard_sample_aspect_buckets_filtered.return_value = [{"buckets": {}}]
        dataset_filter = DatasetFilter({"path": {"include": ["keep"], "exclude": ["drop"], "mode": "glob"}})

        result = backend.list_shard_sample_aspect_buckets([0], dataset_filter=dataset_filter)

        self.assertEqual(result, [{"buckets": {}}])
        backend.loader.list_shard_sample_aspect_buckets_filtered.assert_called_once_with(
            [0],
            key="aspect",
            target_pixel_area=None,
            target_resolution_multiple=64,
            round_to=2,
            path_include=["keep"],
            path_exclude=["drop"],
            path_filter_mode="glob",
        )

    def test_filtered_aspect_bucket_call_requires_webshart_api(self):
        backend = WebshartDataBackend.__new__(WebshartDataBackend)
        backend.loader = Mock(spec=[])
        dataset_filter = DatasetFilter({"path": {"include": ["keep"]}})

        with self.assertRaisesRegex(ImportError, "list_shard_sample_aspect_buckets_filtered"):
            backend.list_shard_sample_aspect_buckets([0], dataset_filter=dataset_filter)

    @patch("simpletuner.helpers.metadata.backends.webshart.TrainingSample")
    def test_prepared_image_bucket_matches_cached_string_key(self, training_sample):
        backend = WebshartMetadataBackend.__new__(WebshartMetadataBackend)
        backend.id = "test-backend"
        backend.dataset_type = DatasetType.IMAGE
        backend.meets_resolution_requirements = Mock(return_value=True)
        backend.aspect_ratio_bucket_indices = {"1.0": ["cached.webp"]}
        training_sample.return_value.prepare.return_value = SimpleNamespace(
            aspect_ratio=1.0,
            intermediary_size=(512, 512),
            crop_coordinates=(0, 0),
            target_size=(512, 512),
        )

        key, metadata = backend._prepare_metadata("new.webp", {"original_size": (1024, 1024)})
        backend.aspect_ratio_bucket_indices.setdefault(key, []).append("new.webp")

        self.assertEqual(backend.aspect_ratio_bucket_indices, {"1.0": ["cached.webp", "new.webp"]})
        self.assertEqual(metadata["aspect_ratio"], 1.0)

    def test_video_metadata_uses_indexed_frame_fields_and_probe_geometry(self):
        backend = WebshartMetadataBackend.__new__(WebshartMetadataBackend)
        backend.data_backend = Mock(caption_key=None)
        backend.dataset_type = DatasetType.VIDEO
        backend._probe_video_metadata = Mock(return_value={"original_size": (1280, 720)})
        shard_metadata = {
            "sample.mp4": {
                "captions": "A moving subject.",
                "json_metadata": {"fps": 24.0, "frame": 64, "seconds": 2.67},
            }
        }

        metadata = backend._metadata_for_entry(
            shard_metadata,
            "sample.mp4",
            {"shard_idx": 0, "sample_idx": 3, "filename": "sample.mp4"},
            "webshart://0/3/sample.mp4",
        )

        self.assertEqual(metadata["original_size"], (1280, 720))
        self.assertEqual(metadata["num_frames"], 64)
        self.assertEqual(metadata["fps"], 24.0)
        self.assertEqual(metadata["video_duration"], 2.67)
        self.assertNotIn("captions", metadata)

    @patch("simpletuner.helpers.metadata.backends.webshart.shutil.which", return_value="/usr/bin/ffprobe")
    def test_video_metadata_probe_prefers_sparse_range_payload(self, _which):
        backend = WebshartMetadataBackend.__new__(WebshartMetadataBackend)
        backend.data_backend = Mock()
        backend.data_backend.parse_sample_id.return_value = Mock(filename="sample.mp4")
        backend.data_backend.read_sample_head_tail.return_value = (b"head", b"tail", 1024 * 1024)
        backend._ffprobe_video_path = Mock(return_value={"original_size": (1920, 1080), "num_frames": 64})

        metadata = backend._probe_video_metadata("webshart://0/3/sample.mp4", file_metadata={"offset": 10})

        self.assertEqual(metadata["original_size"], (1920, 1080))
        backend.data_backend.read_sample_head_tail.assert_called_once_with(
            "webshart://0/3/sample.mp4", file_metadata={"offset": 10}
        )
        backend.data_backend.read.assert_not_called()


class TestWebshartMetadataCaptionFiltering(unittest.TestCase):
    def _build_backend(self, entries):
        backend = WebshartMetadataBackend.__new__(WebshartMetadataBackend)
        backend.id = "test-backend"
        backend.dataset_type = DatasetType.IMAGE
        backend.native_policy = None
        backend.removed_samples = set()
        backend._runtime_args = SimpleNamespace(only_instance_prompt=False)
        backend.bind_dataset_config({"caption_strategy": "webshart"})
        backend.max_num_samples = None
        backend.metadata_update_interval = 3600
        backend.aspect_ratio_bucket_indices = {}
        backend.bucket_report = None
        backend.data_backend = Mock(parallel_downloads=1, caption_key=None)
        backend.data_backend.caption_sidecar_filename = WebshartDataBackend.caption_sidecar_filename
        backend.data_backend.get_shard_metadata = Mock(return_value={entry["filename"]: entry for entry in entries})
        backend.reload_cache = Mock()
        backend.load_image_metadata = Mock()
        backend.save_cache = Mock()
        backend._sync_image_files_with_buckets = Mock()
        backend.enqueue_prepared_metadata = Mock()
        backend.flush_prepared_metadata = Mock()
        backend._all_shard_indices = Mock(return_value=[0])
        backend._entries_for_shard = Mock(return_value=entries)
        backend._sample_id_from_entry = Mock(side_effect=lambda shard_idx, entry: f"webshart://0/0/{entry['path']}")
        backend._metadata_for_entry = Mock(return_value={"original_size": (512, 512)})
        backend._prepare_metadata = Mock(
            side_effect=lambda sample_path, sample_metadata, recipe=None: ("1.0", sample_metadata)
        )
        return backend

    def _entries(self):
        return [
            {"path": "captioned.webp", "filename": "captioned.webp", "captions": ["a caption"]},
            {"path": "uncaptioned.webp", "filename": "uncaptioned.webp", "captions": None},
        ]

    def test_webshart_caption_strategy_skips_caption_less_samples(self):
        backend = self._build_backend(self._entries())
        backend.compute_aspect_ratio_bucket_indices()

        self.assertEqual(
            backend.aspect_ratio_bucket_indices,
            {"1.0": ["webshart://0/0/captioned.webp"]},
        )
        self.assertEqual(backend.filtering_statistics["skipped"]["caption_missing"], 1)
        self.assertEqual(backend.filtering_statistics["total_processed"], 1)
        backend.enqueue_prepared_metadata.assert_called_once_with(
            {"webshart://0/0/captioned.webp": {"original_size": (512, 512)}}
        )
        backend.flush_prepared_metadata.assert_called_once_with()

    def test_webshart_caption_strategy_accepts_txt_sidecar_samples(self):
        backend = self._build_backend(self._entries())
        backend.data_backend.get_shard_metadata = Mock(
            return_value={
                "captioned.webp": {"captions": ["a caption"]},
                "uncaptioned.txt": {"path": "uncaptioned.txt", "offset": 0, "length": 10},
            }
        )
        backend.compute_aspect_ratio_bucket_indices()

        self.assertEqual(
            backend.aspect_ratio_bucket_indices["1.0"],
            ["webshart://0/0/captioned.webp", "webshart://0/0/uncaptioned.webp"],
        )
        self.assertEqual(backend.filtering_statistics["skipped"]["caption_missing"], 0)
        self.assertEqual(backend.filtering_statistics["total_processed"], 2)
        backend.data_backend.get_caption.assert_not_called()

    def test_selected_caption_read_failure_remains_a_per_sample_scan_error(self):
        backend = self._build_backend(self._entries())
        backend.data_backend.caption_key = "short"
        backend.data_backend.get_caption.side_effect = [RuntimeError("source read failed"), "caption"]
        backend.compute_aspect_ratio_bucket_indices()
        self.assertEqual(backend.filtering_statistics["skipped"]["error"], 1)
        self.assertEqual(backend.filtering_statistics["total_processed"], 1)
        self.assertEqual(backend.aspect_ratio_bucket_indices, {"1.0": ["webshart://0/0/uncaptioned.webp"]})

    def test_other_caption_strategies_keep_caption_less_samples(self):
        backend = self._build_backend(self._entries())
        backend.bind_dataset_config({"caption_strategy": "filename"})
        backend.compute_aspect_ratio_bucket_indices()

        self.assertEqual(
            backend.aspect_ratio_bucket_indices["1.0"],
            ["webshart://0/0/captioned.webp", "webshart://0/0/uncaptioned.webp"],
        )
        self.assertEqual(backend.filtering_statistics["skipped"]["caption_missing"], 0)


if __name__ == "__main__":
    unittest.main()


class TestWebshartCaptionOptimization(unittest.TestCase):
    def _build_backend(self, layout: str, has_api: bool = True):
        backend = WebshartDataBackend.__new__(WebshartDataBackend)
        backend.id = "test-backend"
        backend.accelerator = None
        backend.optimize_captions = True
        backend.dataset = Mock()
        backend.loader = Mock()
        if has_api:
            backend.dataset.probe_caption_layout = Mock(return_value={"layout": layout})
            backend.loader.coalesce_caption_metadata = Mock(return_value={"coalesced_samples": 128, "shards": 4})
        else:
            del backend.dataset.probe_caption_layout
            del backend.loader.coalesce_caption_metadata
        return backend

    def test_sidecar_layout_triggers_coalescing(self):
        for layout in ("txt_sidecar", "json_sidecar", "mixed"):
            with self.subTest(layout=layout):
                backend = self._build_backend(layout)
                backend._optimize_caption_metadata()
                backend.loader.coalesce_caption_metadata.assert_called_once_with()

    def test_embedded_layout_skips_coalescing(self):
        for layout in ("embedded", "none"):
            with self.subTest(layout=layout):
                backend = self._build_backend(layout)
                backend._optimize_caption_metadata()
                backend.loader.coalesce_caption_metadata.assert_not_called()

    def test_missing_webshart_api_raises(self):
        backend = self._build_backend("txt_sidecar", has_api=False)
        with self.assertRaises(ImportError):
            backend._optimize_caption_metadata()


class TestWebshartConfig(unittest.TestCase):
    def _config_from(self, backend_dict):
        from simpletuner.helpers.data_backend.config.image import ImageBackendConfig

        base = {
            "id": "ws",
            "type": "webshart",
            "source": "org/dataset",
            "metadata_backend": "webshart",
            "caption_strategy": "webshart",
        }
        base.update(backend_dict)
        return ImageBackendConfig.from_dict(base, {})

    def test_accepts_both_spellings_and_block_form(self):
        cases = [
            ({"webshart_optimize_captions": True}, True),
            ({"webshart_optimise_captions": True}, True),
            ({"webshart": {"optimize_captions": True}}, True),
            ({"webshart": {"optimise_captions": True}}, True),
            ({"webshart_optimize_captions": False}, False),
            ({}, None),
        ]
        for backend_dict, expected in cases:
            with self.subTest(backend_dict=backend_dict):
                config = self._config_from(backend_dict)
                self.assertEqual(config.webshart_optimize_captions, expected)

    def test_caption_key_round_trip(self):
        for key in ("long_caption", ["long_caption", "short_caption"]):
            with self.subTest(key=key):
                config = self._config_from({"webshart": {"caption_key": key}})
                self.assertEqual(config.webshart_caption_key, key)
                self.assertEqual(config.to_dict()["config"]["webshart"]["caption_key"], key)

    def test_invalid_caption_keys_are_rejected(self):
        for key in ("", "  ", [], ["caption", ""], ["caption", None], 1, True, {"caption": "text"}):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "webshart.caption_key"):
                self._config_from({"webshart": {"caption_key": key}})

    def test_builder_passes_caption_keys(self):
        from simpletuner.helpers.data_backend.builders.webshart import WebshartBackendBuilder

        config = self._config_from({"webshart": {"caption_key": ["short", "long"], "cache_dir": "cache/ws"}})
        with patch("simpletuner.helpers.data_backend.builders.webshart.WebshartDataBackend") as backend:
            WebshartBackendBuilder(accelerator=None)._create_backend(config)
        self.assertEqual(backend.call_args.kwargs["caption_key"], ["short", "long"])


class TestWebshartCaptionKeyIntegration(unittest.TestCase):
    def setUp(self):
        from simpletuner.helpers.training.state_tracker import StateTracker

        self.enterContext(patch.object(StateTracker, "get_args", return_value=SimpleNamespace(only_instance_prompt=False)))
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.json_metadata = {
            "long": [" detailed caption ", "alternative caption"],
            "short": " brief caption ",
            "empty": " ",
            "captions": {"named": "named caption", "short": "shadowed caption"},
        }
        self.files = {}
        archive_path = self.source / "shard.tar"
        with tarfile.open(archive_path, "w") as archive:
            for name, payload in {
                "sample.jpg": b"image payload",
                "sample.json": json.dumps(self.json_metadata).encode(),
                "missing.jpg": b"image payload",
                "missing.txt": b"unselected text caption",
            }.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, BytesIO(payload))
        with tarfile.open(archive_path) as archive:
            for info in archive.getmembers():
                self.files[info.name] = {"offset": info.offset_data, "length": info.size}
                if info.name.endswith(".jpg"):
                    self.files[info.name].update(width=512, height=512, aspect=1.0, captions="default caption")
        self._write_index(embedded=True)

    def _write_index(self, embedded):
        sample = self.files["sample.jpg"]
        if embedded:
            sample["json_metadata"] = self.json_metadata
        else:
            sample.pop("json_metadata", None)
        (self.source / "shard.json").write_text(
            json.dumps(
                {
                    "path": "shard.tar",
                    "filesize": (self.source / "shard.tar").stat().st_size,
                    "files": self.files,
                }
            ),
            encoding="utf-8",
        )

    def _backend(self, key=None, optimize=False):
        return WebshartDataBackend(
            accelerator=None,
            id="captions",
            source=str(self.source),
            cache_dir=str(self.root / "cache"),
            shard_cache_gb=0,
            caption_key=key,
            optimize_captions=optimize,
        )

    def _sample_id(self, backend, filename="sample.jpg"):
        index = backend.dataset.list_samples_in_shard(0).index(filename)
        return backend.sample_id(0, index, filename)

    def _metadata_backend(self, backend, **kwargs):
        return WebshartMetadataBackend(
            id=backend.id,
            instance_data_dir=backend.cache_dir,
            cache_file=str(self.root / "buckets"),
            metadata_file=str(self.root / "metadata"),
            data_backend=backend,
            accelerator=None,
            batch_size=1,
            resolution=512,
            resolution_type="pixel",
            **kwargs,
        )

    def test_membership_never_serializes_source_credentials(self):
        source = self._backend("short")
        metadata = self._metadata_backend(source)
        config = {
            "webshart": {"source": "s3://test", "hf_token": "test-value"},
            "aws_session_token": "test-value",
            "aws_access_key_id": "test-value",
            "aws_secret_access_key": "test-value",
        }
        metadata.bind_dataset_config(config)
        metadata.save_cache()
        publication = json.loads(source.read(metadata.cache_file))
        self.assertEqual(publication["config"], {"webshart": {"source": "s3://test"}})

    def test_preparation_reads_published_membership_across_rank_split_and_replacement(self):
        source = self._backend("short")
        metadata = self._metadata_backend(source)
        # Older scans predate the scan_in_progress marker.
        source.write(metadata.cache_file, {"aspect_ratio_bucket_indices": {"1.0": ["c", "a"], "2.5": ["b"]}})
        with patch.object(source, "read", side_effect=AssertionError("eager bucket read")):
            reader = self._metadata_backend(source, preparation_only=True)
            self.assertEqual(reader.aspect_ratio_bucket_indices, {})
            reader.aspect_ratio_bucket_indices = {"1.0": ["c", "c"]}
            reader.set_readonly()
            with reader.open_preparation_view() as original:
                source.write(metadata.cache_file, {"aspect_ratio_bucket_indices": {"1.5": ["new"]}})
                self.assertEqual(list(original), ["c", "a", "b"])
            with reader.open_preparation_view() as replacement:
                self.assertEqual(list(replacement), ["new"])

    def test_preparation_distinguishes_partial_scan_from_empty_completed_scan(self):
        metadata = self._metadata_backend(self._backend("short"))
        metadata.aspect_ratio_bucket_indices = {"1.0": ["partial"]}
        metadata.save_cache(scan_in_progress=True)
        with self.assertRaisesRegex(ValueError, "incomplete"), metadata.open_preparation_view():
            self.fail("incomplete membership was opened")
        metadata.aspect_ratio_bucket_indices = {}
        metadata.save_cache()
        with metadata.open_preparation_view() as refs:
            self.assertEqual(list(refs), [])

    def test_preparation_pins_geometry_for_point_lookup_and_reuses_index(self):
        source = self._backend("short")
        writer = self._metadata_backend(source)
        writer.aspect_ratio_bucket_indices = {"1.0": ["a", "b"]}
        writer.image_metadata = {
            "b": {"target_size": [2, 3]},
            "a": {"target_size": [4, 5], "bbox_entities": [{"label": "x"}]},
        }
        writer.save_image_metadata()
        writer.save_cache()
        with patch.object(source, "read", side_effect=AssertionError("whole metadata read")):
            reader = self._metadata_backend(source, preparation_only=True)
            with reader.open_preparation_view() as old:
                self.assertEqual(list(old), ["a", "b"])
                self.assertEqual(old.metadata.get_metadata_by_filepath("a")["bbox_entities"], [{"label": "x"}])
                with patch(
                    "simpletuner.helpers.metadata.backends.webshart_geometry._build_index", side_effect=AssertionError
                ):
                    with reader.open_preparation_view() as warm:
                        self.assertEqual(warm.metadata.get_metadata_by_filepath("b")["target_size"], [2, 3])
                writer.image_metadata = {"z": {"target_size": [9, 10]}}
                writer.save_image_metadata()
                with self.assertRaisesRegex(ValueError, "revisions differ"), reader.open_preparation_view():
                    self.fail("opened mismatched geometry and membership")
                writer.aspect_ratio_bucket_indices = {"1.0": ["z"]}
                writer.save_cache()
                with reader.open_preparation_view() as new:
                    self.assertEqual(list(new), ["z"])
                    self.assertEqual(new.metadata.get_metadata_by_filepath("z")["target_size"], [9, 10])
                self.assertEqual(old.metadata.get_metadata_by_filepath("b")["target_size"], [2, 3])
            self.assertEqual(reader.image_metadata, {})

    def test_selects_fields_and_named_captions_in_requested_order(self):
        for key, expected in [
            (None, "default caption"),
            ("short", "brief caption"),
            (["named", "long", "short"], ["named caption", "detailed caption", "alternative caption", "brief caption"]),
            (["absent", "empty", "short"], ["brief caption"]),
            ("absent", None),
            ("empty", None),
        ]:
            with self.subTest(key=key):
                backend = self._backend(key)
                self.assertEqual(backend.get_caption(self._sample_id(backend)), expected)

    def test_selects_indexed_fields_and_literal_key_names(self):
        self.files["sample.jpg"].update({"custom.key": "literal key caption", "short": "indexed caption"})
        self._write_index(embedded=True)
        for key, expected in [
            (["custom.key", "short"], ["literal key caption", "brief caption"]),
            ("custom.key", "literal key caption"),
        ]:
            with self.subTest(key=key):
                backend = self._backend(key)
                self.assertEqual(backend.get_caption(self._sample_id(backend)), expected)

    def test_selects_json_literal_keys_and_indexed_captions(self):
        self.json_metadata.pop("captions")
        self.json_metadata.update({"custom.key": "literal key caption", "custom": {"key": "nested caption"}})
        self.files["sample.jpg"]["captions"] = "indexed caption"
        self._write_index(embedded=True)
        backend = self._backend(["custom.key", "captions", "short"])
        self.assertEqual(
            backend.get_caption(self._sample_id(backend)), ["literal key caption", "indexed caption", "brief caption"]
        )

    def test_selects_literal_keys_in_json_caption_dictionary(self):
        self.json_metadata["captions"] = {"custom.key": "literal key caption", "short": "shadowed caption"}
        self._write_index(embedded=True)
        backend = self._backend(["custom.key", "short"])
        self.assertEqual(backend.get_caption(self._sample_id(backend)), ["literal key caption", "brief caption"])

    def test_reads_json_sidecar_without_loading_image_bytes(self):
        import webshart

        self._write_index(embedded=False)
        with patch.object(webshart, "TarDataLoader", wraps=webshart.TarDataLoader) as loader_factory:
            backend = self._backend("short")
        self.assertTrue(loader_factory.call_args_list[0].kwargs["load_file_data"])
        self.assertFalse(loader_factory.call_args_list[1].kwargs["load_file_data"])
        sample_id = self._sample_id(backend)
        backend.dataset = Mock(open_shard=Mock(side_effect=AssertionError("Uncached shard reader requested")))
        with patch.object(backend, "_read_sample_bytes", side_effect=AssertionError("Image bytes requested")):
            self.assertEqual(backend.get_caption(sample_id), "brief caption")
        self.assertEqual(backend.get_shard_metadata(0)["sample.jpg"]["json_metadata"], self.json_metadata)

    def test_optimization_preserves_custom_selection(self):
        backend = self._backend("short", optimize=True)
        self.assertEqual(backend.get_caption(self._sample_id(backend)), "brief caption")

    def test_worker_representation_preserves_selection(self):
        backend = self._backend(["named", "short"])
        restored = WebshartDataBackend.from_instance_representation(backend.get_instance_representation())
        self.assertEqual(restored.get_caption(self._sample_id(restored)), ["named caption", "brief caption"])

    def test_txt_sidecar_uses_logical_sample_without_image_reads(self):
        self.files["missing.jpg"].pop("captions")
        self._write_index(embedded=True)
        backend = self._backend()
        sample_id = self._sample_id(backend, "missing.jpg")
        with patch.object(backend, "_read_sample_bytes", side_effect=AssertionError("image payload read")):
            self.assertEqual(backend.get_caption(sample_id), "unselected text caption")

    def test_custom_caption_json_is_available_before_video_geometry_preparation(self):
        self._write_index(embedded=False)
        data_backend = self._backend("short")
        sample_id = self._sample_id(data_backend)
        sample_ref = data_backend.parse_sample_id(sample_id)
        metadata = WebshartMetadataBackend.__new__(WebshartMetadataBackend)
        metadata.data_backend = data_backend
        metadata.dataset_type = DatasetType.VIDEO
        metadata._probe_video_metadata = Mock(side_effect=AssertionError("unnecessary video probing"))
        payload = {**self.json_metadata, "fps": 24.0, "frame": 64, "seconds": 2.67}
        with patch.object(
            data_backend, "caption_loader", SimpleNamespace(load_sample_json=Mock(return_value=json.dumps(payload).encode()))
        ):
            record = metadata._metadata_for_entry(
                data_backend.get_shard_metadata(0),
                sample_ref.filename,
                {"shard_idx": 0, "sample_idx": sample_ref.sample_idx, "filename": sample_ref.filename},
                sample_id,
            )
        self.assertEqual(record["num_frames"], 64)
        self.assertEqual(record["fps"], 24.0)
        self.assertNotIn("captions", record)

    def test_native_caption_objects_survive_index_selection_and_cache(self):
        caption = {"high_level_description": "a café", "elements": [{"bbox": [1, 2, 30, 40]}]}
        for index, value in enumerate([caption, [caption], [caption, "alternate"]]):
            with self.subTest(value=value):
                self.files["sample.jpg"]["captions"] = "indexed fallback caption"
                self.json_metadata["captions"] = value
                self.json_metadata[f"version_{index}"] = value
                self._write_index(embedded=True)
                backend = WebshartDataBackend(
                    accelerator=None,
                    id=f"native-{index}",
                    source=str(self.source),
                    cache_dir=str(self.root / f"native-cache-{index}"),
                    shard_cache_gb=0,
                    caption_key="captions",
                )
                sample_id = self._sample_id(backend)
                expected = caption if value == [caption] else value
                self.assertEqual(backend.get_caption(sample_id), expected)
                backend.caption_key = f"version_{index}"
                selected = backend.get_caption(sample_id)
                self.assertEqual(selected, expected)
                metadata = self._metadata_backend(backend)
                self.assertEqual(metadata.caption_cache_entry(sample_id), selected)
                metadata.reload_cache()
                self.assertEqual(metadata.caption_cache_entry(sample_id), selected)
                self.assertNotIn("caption_cache", vars(metadata))

    def test_worker_reload_receives_captions_and_metadata_after_main_process_refresh(self):
        from simpletuner.helpers.training.state_tracker import StateTracker

        with (
            patch.object(StateTracker, "get_data_backend_config", return_value={"caption_strategy": "webshart"}),
            patch.object(StateTracker, "set_data_backend_config"),
        ):
            main = self._metadata_backend(self._backend("short"))
            worker = self._metadata_backend(self._backend("short"))
            self.assertNotIn("caption_cache", vars(worker))
            with (
                patch.object(main, "_prepare_metadata", side_effect=lambda path, data, recipe=None: (1.0, data)),
                patch.object(main, "_enforce_min_bucket_size"),
            ):
                main.compute_aspect_ratio_bucket_indices()

            sample_id = self._sample_id(main.data_backend)
            for set_config in (False, True):
                with self.subTest(set_config=set_config):
                    worker.image_metadata = {"stale": {"captions": "old caption"}}
                    worker.image_metadata_loaded = True
                    worker.reload_cache(set_config=set_config)
                    self.assertEqual(worker.aspect_ratio_bucket_indices, {"1.0": [sample_id]})
                    self.assertEqual(worker.caption_cache_entry(sample_id), "brief caption")
                    self.assertNotIn("caption_cache", vars(worker))
                    self.assertTrue(worker.image_metadata_loaded)
                    self.assertNotIn("stale", worker.image_metadata)
                    self.assertNotIn("captions", worker.image_metadata[sample_id])

    def test_bucketing_and_cache_reload_use_selected_captions(self):
        from simpletuner.helpers.training.state_tracker import StateTracker

        with (
            patch.object(StateTracker, "get_data_backend_config", return_value={"caption_strategy": "webshart"}),
            patch.object(StateTracker, "set_data_backend_config"),
        ):
            backend = self._backend("short")
            metadata = self._metadata_backend(backend)
            with (
                patch.object(metadata, "_prepare_metadata", side_effect=lambda path, data, recipe=None: (1.0, data)),
                patch.object(metadata, "_enforce_min_bucket_size"),
            ):
                metadata.compute_aspect_ratio_bucket_indices()
            sample_id = self._sample_id(backend)
            self.assertEqual(metadata.aspect_ratio_bucket_indices, {1.0: [sample_id]})
            self.assertEqual(metadata.filtering_statistics["skipped"]["caption_missing"], 1)
            self.assertEqual(metadata.caption_cache_entry(sample_id), "brief caption")
            reloaded = self._metadata_backend(self._backend("short"))
            self.assertEqual(reloaded.caption_cache_entry(sample_id), "brief caption")
            changed = self._metadata_backend(self._backend("named"))
            self.assertNotEqual(changed.cache_file, metadata.cache_file)
            self.assertNotEqual(changed.metadata_file, metadata.metadata_file)
            self.assertEqual(changed.aspect_ratio_bucket_indices, {})
            self.assertEqual(changed.caption_cache_entry(sample_id), "named caption")
            default = self._metadata_backend(self._backend())
            self.assertNotEqual(default.cache_file, metadata.cache_file)
            self.assertEqual(default.caption_cache_entry(sample_id), "default caption")


class TestWebshartShardMetadataMemoization(unittest.TestCase):
    def _backend(self, limit=8):
        backend = WebshartDataBackend.__new__(WebshartDataBackend)
        backend._shard_metadata_cache = {}
        backend._shard_metadata_cache_limit = limit
        backend.loader = Mock(get_metadata=Mock(side_effect=lambda idx: {"file.jpg": {"captions": f"shard {idx}"}}))
        return backend

    def test_repeat_lookups_hit_cache(self):
        backend = self._backend()

        first = backend.get_shard_metadata(3)
        second = backend.get_shard_metadata(3)

        self.assertIs(first, second)
        backend.loader.get_metadata.assert_called_once_with(3)

    def test_cache_evicts_oldest_shard_at_limit(self):
        backend = self._backend(limit=2)

        backend.get_shard_metadata(1)
        backend.get_shard_metadata(2)
        backend.get_shard_metadata(3)

        self.assertEqual(sorted(backend._shard_metadata_cache), [2, 3])
        backend.get_shard_metadata(1)
        self.assertEqual(backend.loader.get_metadata.call_count, 4)


class TestWebshartCaptionAccess(unittest.TestCase):
    def test_legacy_caption_json_is_ignored_and_geometry_captions_are_not_retained(self):
        backend = WebshartMetadataBackend.__new__(WebshartMetadataBackend)
        backend.cache_file = "buckets.json"
        backend.metadata_file = "geometry.json"
        backend.aspect_ratio_bucket_indices = {}
        files = {
            "buckets.json_captions.json": '{"sample.jpg": "stale caption"}',
            "geometry.json": '{"sample.jpg": {"captions": "old caption", "original_size": [16, 16]}}',
        }
        backend.data_backend = Mock()
        backend.data_backend.exists.side_effect = lambda path: path in files
        backend.data_backend.read.side_effect = files.__getitem__
        backend.data_backend.normalize_sample_id.side_effect = str
        backend.data_backend.get_caption.return_value = "source caption"
        backend.reload_cache()
        self.assertEqual(backend.image_metadata, {"sample.jpg": {"original_size": [16, 16]}})
        backend.data_backend.read.assert_called_once_with("geometry.json")
        self.assertEqual(backend.caption_cache_entry("sample.jpg"), "source caption")
        self.assertIn("buckets.json_captions.json", files)
        backend.data_backend.delete.assert_not_called()
        self.assertNotIn("caption_cache", vars(backend))

    def test_caption_lookup_does_not_accumulate_records(self):
        backend = WebshartMetadataBackend.__new__(WebshartMetadataBackend)
        backend.data_backend = SimpleNamespace(normalize_sample_id=str, get_caption=lambda ref: [ref, "alternate"])
        for index in range(20000):
            ref = f"webshart://{index}/0/sample.jpg"
            self.assertEqual(backend.caption_cache_entry(ref), [ref, "alternate"])
        self.assertEqual(set(vars(backend)), {"data_backend"})
