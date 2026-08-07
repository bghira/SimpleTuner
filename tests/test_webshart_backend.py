import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import torch

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.data_backend.webshart import WebshartDataBackend
from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend


class TestWebshartDataBackend(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
