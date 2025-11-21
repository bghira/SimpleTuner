import unittest
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
        self.id = data_backend.id

    def _get_video_metadata_from_item(self, item):
        return item["metadata"]


class HuggingfaceMetadataBackendTests(unittest.TestCase):
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
