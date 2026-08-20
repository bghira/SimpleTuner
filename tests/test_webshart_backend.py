import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from simpletuner.helpers.data_backend.dataset_types import DatasetType
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
        backend.get_shard_metadata = Mock(return_value={"sample.mp4": {"captions": "A moving subject."}})

        caption = backend.get_caption("webshart://2/7/sample.mp4")

        self.assertEqual(caption, "A moving subject.")

    def test_get_caption_preserves_indexed_caption_variants(self):
        backend = WebshartDataBackend.__new__(WebshartDataBackend)
        backend.get_shard_metadata = Mock(return_value={"sample.mp4": {"captions": ["first caption", "second caption"]}})

        caption = backend.get_caption("webshart://2/7/sample.mp4")

        self.assertEqual(caption, ["first caption", "second caption"])

    def test_video_metadata_uses_indexed_frame_fields_and_probe_geometry(self):
        backend = WebshartMetadataBackend.__new__(WebshartMetadataBackend)
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
        self.assertEqual(metadata["captions"], "A moving subject.")

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
        backend.max_num_samples = None
        backend.metadata_update_interval = 3600
        backend.aspect_ratio_bucket_indices = {}
        backend.caption_cache = {}
        backend.bucket_report = None
        backend.data_backend = Mock(parallel_downloads=1)
        backend.data_backend.get_shard_metadata = Mock(return_value={})
        backend.reload_cache = Mock()
        backend.load_image_metadata = Mock()
        backend.save_cache = Mock()
        backend.save_image_metadata = Mock()
        backend._save_caption_cache = Mock()
        backend._sync_image_files_with_buckets = Mock()
        backend.set_metadata_by_filepath = Mock()
        backend._all_shard_indices = Mock(return_value=[0])
        backend._entries_for_shard = Mock(return_value=entries)
        backend._sample_id_from_entry = Mock(side_effect=lambda shard_idx, entry: f"webshart://0/0/{entry['path']}")
        backend._prepare_bucket_entry = Mock(
            side_effect=lambda shard_metadata, entry, sample_path: (
                {"captions": entry.get("captions")},
                ("1.0", {"captions": entry.get("captions")}),
                None,
            )
        )
        return backend

    def _entries(self):
        return [
            {"path": "captioned.webp", "filename": "captioned.webp", "captions": ["a caption"]},
            {"path": "uncaptioned.webp", "filename": "uncaptioned.webp", "captions": None},
        ]

    def test_webshart_caption_strategy_skips_caption_less_samples(self):
        backend = self._build_backend(self._entries())
        with patch(
            "simpletuner.helpers.metadata.backends.webshart.StateTracker.get_data_backend_config",
            return_value={"caption_strategy": "webshart"},
        ):
            backend.compute_aspect_ratio_bucket_indices()

        self.assertEqual(
            backend.aspect_ratio_bucket_indices,
            {"1.0": ["webshart://0/0/captioned.webp"]},
        )
        self.assertEqual(backend.filtering_statistics["skipped"]["caption_missing"], 1)
        self.assertEqual(backend.filtering_statistics["total_processed"], 1)

    def test_webshart_caption_strategy_accepts_txt_sidecar_samples(self):
        backend = self._build_backend(self._entries())
        backend.data_backend.get_shard_metadata = Mock(
            return_value={"uncaptioned.txt": {"path": "uncaptioned.txt", "offset": 0, "length": 10}}
        )
        with patch(
            "simpletuner.helpers.metadata.backends.webshart.StateTracker.get_data_backend_config",
            return_value={"caption_strategy": "webshart"},
        ):
            backend.compute_aspect_ratio_bucket_indices()

        self.assertEqual(
            backend.aspect_ratio_bucket_indices["1.0"],
            ["webshart://0/0/captioned.webp", "webshart://0/0/uncaptioned.webp"],
        )
        self.assertEqual(backend.filtering_statistics["skipped"]["caption_missing"], 0)
        self.assertEqual(backend.filtering_statistics["total_processed"], 2)

    def test_other_caption_strategies_keep_caption_less_samples(self):
        backend = self._build_backend(self._entries())
        with patch(
            "simpletuner.helpers.metadata.backends.webshart.StateTracker.get_data_backend_config",
            return_value={"caption_strategy": "filename"},
        ):
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


class TestWebshartOptimizeCaptionsConfig(unittest.TestCase):
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


class TestWebshartCaptionCacheLoad(unittest.TestCase):
    def _metadata_backend(self, cache_payload):
        backend = WebshartMetadataBackend.__new__(WebshartMetadataBackend)
        backend.cache_file = "aspect_ratio_bucket_indices_test.json"
        backend.data_backend = Mock(exists=Mock(return_value=True), read=Mock(return_value=cache_payload))
        backend.image_metadata = {"webshart://0/1/sample.jpg": {"captions": "a red bicycle"}}
        return backend

    def test_empty_cache_file_regenerates_from_image_metadata(self):
        backend = self._metadata_backend("{}")

        backend._load_caption_cache()

        self.assertEqual(backend.caption_cache, {"webshart://0/1/sample.jpg": "a red bicycle"})

    def test_populated_cache_file_is_used_directly(self):
        backend = self._metadata_backend('{"webshart://0/1/sample.jpg": "cached caption"}')

        backend._load_caption_cache()

        self.assertEqual(backend.caption_cache, {"webshart://0/1/sample.jpg": "cached caption"})
