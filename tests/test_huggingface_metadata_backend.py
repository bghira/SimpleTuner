import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from simpletuner.helpers.metadata.backends.huggingface import HuggingfaceMetadataBackend


class _DummyDataBackend:
    def __init__(self, item):
        self.item = item
        self.id = "test-backend"

    def _get_index_from_path(self, path):
        return 0

    def get_dataset_item(self, index):
        return self.item


class _MinimalHuggingfaceMetadataBackend(HuggingfaceMetadataBackend):
    """Collect just enough state to exercise _process_for_bucket without side effects."""

    def __init__(self, data_backend):
        self.data_backend = data_backend
        self.video_column = "video"
        self.minimum_num_frames = None
        self.maximum_num_frames = None
        self.minimum_image_size = None
        self.dataset_type = "video"
        self.bucket_report = None
        self.aspect_ratio_bucket_indices = {}
        self.quality_filter = None
        self.quality_column = "quality_assessment"
        self.bbox_column = None
        self.id = data_backend.id

    def _get_video_metadata_from_item(self, item):
        return item["metadata"]


class HuggingfaceMetadataBackendTests(unittest.TestCase):
    def test_metadata_reload_keeps_existing_entries_visible_until_read_completes(self):
        read_started = threading.Event()
        allow_read = threading.Event()

        class BlockingDataBackend:
            def exists(self, _path):
                return True

            def get_abs_path(self, path):
                return path

            def read(self, _path):
                read_started.set()
                if not allow_read.wait(timeout=5):
                    raise TimeoutError("metadata read was not released")
                return json.dumps({"1.wav": {"prompt": "new", "lyrics": "lyrics"}})

        backend = HuggingfaceMetadataBackend.__new__(HuggingfaceMetadataBackend)
        backend.id = "audio-test"
        backend.metadata_file = Path("metadata.json")
        backend.data_backend = BlockingDataBackend()
        backend.metadata_semaphor = threading.Semaphore()
        backend.image_metadata = {"1.wav": {"prompt": "old", "lyrics": "lyrics"}}
        backend.image_metadata_loaded = True

        reload_thread = threading.Thread(target=backend.load_image_metadata)
        reload_thread.start()
        self.assertTrue(read_started.wait(timeout=5))
        self.assertEqual(backend.get_metadata_by_filepath("1.wav")["prompt"], "old")

        allow_read.set()
        reload_thread.join(timeout=5)
        self.assertFalse(reload_thread.is_alive())
        self.assertEqual(backend.get_metadata_by_filepath("1.wav")["prompt"], "new")

    def test_audio_caption_falls_back_to_text_sibling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "song.flac"
            audio_path.touch()
            audio_path.with_suffix(".txt").write_text("riff-heavy caption", encoding="utf-8")
            media = SimpleNamespace(_hf_encoded={"path": str(audio_path)})

            backend = HuggingfaceMetadataBackend.__new__(HuggingfaceMetadataBackend)
            backend.dataset_type = "audio"
            backend.caption_column = "caption"
            backend.fallback_caption_column = None
            backend.audio_caption_fields = ["prompt", "tags"]
            backend.description_column = "description"

            self.assertEqual(backend._extract_caption_from_item({"audio": media}), "riff-heavy caption")

    @patch("simpletuner.helpers.metadata.backends.huggingface.TrainingSample")
    def test_video_without_maximum_num_frames_is_not_flagged_as_too_many(self, mock_training_sample):
        prepared = SimpleNamespace(
            aspect_ratio=1.0,
            intermediary_size=(1, 1),
            crop_coordinates=(0, 0),
            target_size=(1, 1),
        )
        mock_training_sample.return_value.prepare.return_value = prepared

        item = {"video": object(), "metadata": {"original_size": (640, 480), "num_frames": 5}}
        backend = _MinimalHuggingfaceMetadataBackend(_DummyDataBackend(item))

        statistics = {"skipped": {"too_many_frames": 0}}
        aspect_ratio_buckets = backend._process_for_bucket("0.mp4", {}, statistics=statistics)

        self.assertIn("1.0", aspect_ratio_buckets)
        self.assertEqual(aspect_ratio_buckets["1.0"], ["0.mp4"])
        self.assertEqual(statistics["skipped"]["too_many_frames"], 0)


if __name__ == "__main__":
    unittest.main()
