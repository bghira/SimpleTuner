"""Tests for video bucket strategy (resolution_frames) functionality."""

import json
import os
import tempfile
import unittest
from io import BytesIO
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.metadata.backends.base import MetadataBackend
from simpletuner.helpers.metadata.backends.discovery import DiscoveryMetadataBackend
from simpletuner.helpers.training.state_tracker import StateTracker


class MockDataBackend(BaseDataBackend):
    """Minimal mock data backend for testing."""

    def __init__(self, dataset_id: str = "test-video"):
        self.id = dataset_id
        self.dataset_type = DatasetType.VIDEO
        self._files = {}

    def read(self, identifier, as_byteIO: bool = False):
        data = self._files.get(identifier)
        if as_byteIO and data:
            return BytesIO(data)
        return data

    def write(self, identifier, data):
        payload = data.encode("utf-8") if isinstance(data, str) else data
        self._files[identifier] = payload

    def delete(self, identifier):
        self._files.pop(identifier, None)

    def exists(self, identifier):
        return identifier in self._files

    def list_files(self, file_extensions=None, instance_data_dir=None):
        return []

    def get_abs_path(self, sample_path: str = None):
        return sample_path or "/mock/path"

    def open_file(self, identifier, mode):
        raise NotImplementedError

    def read_image(self, filepath, delete_problematic_images=False):
        raise NotImplementedError

    def read_image_batch(self, filepaths, delete_problematic_images=False):
        raise NotImplementedError

    def create_directory(self, directory_path):
        pass

    def torch_load(self, filename):
        raise NotImplementedError

    def torch_save(self, data, filename):
        raise NotImplementedError

    def write_batch(self, identifiers, files):
        for identifier, data in zip(identifiers, files):
            self.write(identifier, data)

    def get_instance_representation(self):
        return {"backend_type": "mock", "id": self.id}

    @staticmethod
    def from_instance_representation(representation):
        return MockDataBackend(representation["id"])


class TestVideoBucketStrategyConfig(unittest.TestCase):
    """Test configuration extraction and resolution for video bucket strategy."""

    def setUp(self):
        StateTracker.clear_data_backends()
        self.data_backend = MockDataBackend("test-video")
        self.accelerator = Mock()

    def tearDown(self):
        StateTracker.clear_data_backends()

    def _create_metadata_backend(self, dataset_config: dict) -> DiscoveryMetadataBackend:
        """Helper to create a metadata backend with given config."""
        StateTracker.set_data_backend_config("test-video", dataset_config)
        StateTracker.set_args(MagicMock())

        with (
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker._save_to_disk",
                return_value=True,
            ),
            patch("pathlib.Path.exists", return_value=True),
        ):
            return DiscoveryMetadataBackend(
                id="test-video",
                instance_data_dir="/fake/path",
                cache_file="/fake/cache",
                metadata_file="/fake/metadata.json",
                batch_size=1,
                data_backend=self.data_backend,
                resolution=720,
                resolution_type="pixel_area",
                accelerator=self.accelerator,
                repeats=0,
            )

    def test_extract_video_config_with_video_section(self):
        """Test that video config is correctly extracted from dataset config."""
        config = {
            "dataset_type": "video",
            "video": {
                "bucket_strategy": "resolution_frames",
                "frame_interval": 25,
                "num_frames": 125,
            },
        }
        backend = self._create_metadata_backend(config)
        self.assertEqual(backend.video_config, config["video"])

    def test_extract_video_config_without_video_section(self):
        """Test that empty dict is returned when no video section exists."""
        config = {"dataset_type": "video"}
        backend = self._create_metadata_backend(config)
        self.assertEqual(backend.video_config, {})

    def test_resolve_bucket_strategy_default_for_video(self):
        """Test default bucket strategy for video is aspect_ratio."""
        config = {"dataset_type": "video"}
        backend = self._create_metadata_backend(config)
        self.assertEqual(backend.bucket_strategy, "aspect_ratio")

    def test_resolve_bucket_strategy_resolution_frames(self):
        """Test bucket strategy can be set to resolution_frames."""
        config = {
            "dataset_type": "video",
            "video": {"bucket_strategy": "resolution_frames"},
        }
        backend = self._create_metadata_backend(config)
        self.assertEqual(backend.bucket_strategy, "resolution_frames")

    def test_resolve_bucket_strategy_from_top_level(self):
        """Test bucket strategy can be set at top level."""
        config = {
            "dataset_type": "video",
            "bucket_strategy": "resolution_frames",
        }
        backend = self._create_metadata_backend(config)
        self.assertEqual(backend.bucket_strategy, "resolution_frames")

    def test_resolve_bucket_strategy_video_section_takes_precedence(self):
        """Test video section bucket_strategy takes precedence over top level."""
        config = {
            "dataset_type": "video",
            "bucket_strategy": "aspect_ratio",
            "video": {"bucket_strategy": "resolution_frames"},
        }
        backend = self._create_metadata_backend(config)
        self.assertEqual(backend.bucket_strategy, "resolution_frames")

    def test_resolve_bucket_strategy_case_insensitive(self):
        """Test bucket strategy is case-insensitive."""
        config = {
            "dataset_type": "video",
            "video": {"bucket_strategy": "Resolution_Frames"},
        }
        backend = self._create_metadata_backend(config)
        self.assertEqual(backend.bucket_strategy, "resolution_frames")

    def test_resolve_video_frame_interval_set(self):
        """Test frame interval is correctly resolved."""
        config = {
            "dataset_type": "video",
            "video": {
                "bucket_strategy": "resolution_frames",
                "frame_interval": 25,
            },
        }
        backend = self._create_metadata_backend(config)
        self.assertEqual(backend.video_frame_interval, 25)

    def test_resolve_video_frame_interval_none_without_resolution_frames(self):
        """Test frame interval is None when not using resolution_frames strategy."""
        config = {
            "dataset_type": "video",
            "video": {"frame_interval": 25},  # no bucket_strategy
        }
        backend = self._create_metadata_backend(config)
        self.assertIsNone(backend.video_frame_interval)

    def test_resolve_video_frame_interval_none_for_image(self):
        """Test frame interval is None for image datasets."""
        config = {
            "dataset_type": "image",
            "video": {"bucket_strategy": "resolution_frames", "frame_interval": 25},
        }
        backend = self._create_metadata_backend(config)
        self.assertIsNone(backend.video_frame_interval)

    def test_resolve_video_frame_interval_invalid_value(self):
        """Test invalid frame interval is handled gracefully."""
        config = {
            "dataset_type": "video",
            "video": {
                "bucket_strategy": "resolution_frames",
                "frame_interval": "invalid",
            },
        }
        backend = self._create_metadata_backend(config)
        self.assertIsNone(backend.video_frame_interval)

    def test_resolve_video_frame_interval_zero_value(self):
        """Test zero frame interval is rejected."""
        config = {
            "dataset_type": "video",
            "video": {
                "bucket_strategy": "resolution_frames",
                "frame_interval": 0,
            },
        }
        backend = self._create_metadata_backend(config)
        self.assertIsNone(backend.video_frame_interval)

    def test_resolve_video_frame_interval_negative_value(self):
        """Test negative frame interval is rejected."""
        config = {
            "dataset_type": "video",
            "video": {
                "bucket_strategy": "resolution_frames",
                "frame_interval": -10,
            },
        }
        backend = self._create_metadata_backend(config)
        self.assertIsNone(backend.video_frame_interval)


class TestComputeVideoBucket(unittest.TestCase):
    """Test _compute_video_bucket() method."""

    def setUp(self):
        StateTracker.clear_data_backends()
        self.data_backend = MockDataBackend("test-video")
        self.accelerator = Mock()

    def tearDown(self):
        StateTracker.clear_data_backends()

    def _create_backend_with_interval(self, frame_interval: int = None) -> DiscoveryMetadataBackend:
        """Helper to create backend with specific frame interval."""
        video_config = {"bucket_strategy": "resolution_frames"}
        if frame_interval is not None:
            video_config["frame_interval"] = frame_interval

        StateTracker.set_data_backend_config(
            "test-video",
            {"dataset_type": "video", "video": video_config},
        )
        StateTracker.set_args(MagicMock())

        with (
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker._save_to_disk",
                return_value=True,
            ),
            patch("pathlib.Path.exists", return_value=True),
        ):
            return DiscoveryMetadataBackend(
                id="test-video",
                instance_data_dir="/fake/path",
                cache_file="/fake/cache",
                metadata_file="/fake/metadata.json",
                batch_size=1,
                data_backend=self.data_backend,
                resolution=720,
                resolution_type="pixel_area",
                accelerator=self.accelerator,
                repeats=0,
            )

    def test_compute_video_bucket_basic(self):
        """Test basic bucket key generation."""
        backend = self._create_backend_with_interval(None)
        bucket_key, rounded_frames = backend._compute_video_bucket(1920, 1080, 125)
        self.assertEqual(bucket_key, "1920x1080@125")
        self.assertEqual(rounded_frames, 125)

    def test_compute_video_bucket_with_interval_exact_multiple(self):
        """Test bucket key when frames are exact multiple of interval."""
        backend = self._create_backend_with_interval(25)
        bucket_key, rounded_frames = backend._compute_video_bucket(1920, 1080, 125)
        self.assertEqual(bucket_key, "1920x1080@125")
        self.assertEqual(rounded_frames, 125)

    def test_compute_video_bucket_with_interval_rounds_down(self):
        """Test bucket key rounds frames down to nearest interval."""
        backend = self._create_backend_with_interval(25)
        bucket_key, rounded_frames = backend._compute_video_bucket(1920, 1080, 137)
        self.assertEqual(bucket_key, "1920x1080@125")
        self.assertEqual(rounded_frames, 125)

    def test_compute_video_bucket_with_interval_rounds_down_edge(self):
        """Test bucket key rounds down at edge case (one less than next interval)."""
        backend = self._create_backend_with_interval(25)
        bucket_key, rounded_frames = backend._compute_video_bucket(1920, 1080, 149)
        self.assertEqual(bucket_key, "1920x1080@125")
        self.assertEqual(rounded_frames, 125)

    def test_compute_video_bucket_small_frame_count(self):
        """Test bucket key with frames less than interval."""
        backend = self._create_backend_with_interval(25)
        # 20 frames < 25 interval, should keep original since floor(20/25)*25 = 0
        bucket_key, rounded_frames = backend._compute_video_bucket(640, 480, 20)
        self.assertEqual(bucket_key, "640x480@20")
        self.assertEqual(rounded_frames, 20)

    def test_compute_video_bucket_different_resolutions(self):
        """Test bucket keys for different resolutions."""
        backend = self._create_backend_with_interval(25)

        test_cases = [
            ((1920, 1080, 100), "1920x1080@100"),
            ((1280, 720, 75), "1280x720@75"),
            ((640, 480, 50), "640x480@50"),
            ((3840, 2160, 125), "3840x2160@125"),
        ]

        for (w, h, f), expected_key in test_cases:
            bucket_key, _ = backend._compute_video_bucket(w, h, f)
            self.assertEqual(bucket_key, expected_key, f"Failed for {w}x{h}@{f}")

    def test_compute_video_bucket_no_interval(self):
        """Test bucket key when no frame interval is set."""
        backend = self._create_backend_with_interval(None)
        bucket_key, rounded_frames = backend._compute_video_bucket(1920, 1080, 137)
        self.assertEqual(bucket_key, "1920x1080@137")
        self.assertEqual(rounded_frames, 137)

    def test_compute_video_bucket_integer_types(self):
        """Test that bucket key uses integers in output."""
        backend = self._create_backend_with_interval(25)
        # Pass floats, should output integers
        bucket_key, rounded_frames = backend._compute_video_bucket(1920.0, 1080.0, 125.0)
        self.assertEqual(bucket_key, "1920x1080@125")
        self.assertIsInstance(rounded_frames, int)


class TestVideoBucketStrategyIntegration(unittest.TestCase):
    """Integration tests for video bucket strategy in _process_for_bucket."""

    def setUp(self):
        StateTracker.clear_data_backends()
        self.data_backend = MockDataBackend("test-video")
        self.accelerator = Mock()

    def tearDown(self):
        StateTracker.clear_data_backends()

    def _create_backend(self, bucket_strategy: str, frame_interval: int = None) -> DiscoveryMetadataBackend:
        """Helper to create backend with specific bucket strategy."""
        video_config = {"bucket_strategy": bucket_strategy}
        if frame_interval is not None:
            video_config["frame_interval"] = frame_interval

        StateTracker.set_data_backend_config(
            "test-video",
            {"dataset_type": "video", "video": video_config},
        )
        StateTracker.set_args(MagicMock())

        with (
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker._save_to_disk",
                return_value=True,
            ),
            patch("pathlib.Path.exists", return_value=True),
        ):
            return DiscoveryMetadataBackend(
                id="test-video",
                instance_data_dir="/fake/path",
                cache_file="/fake/cache",
                metadata_file="/fake/metadata.json",
                batch_size=1,
                data_backend=self.data_backend,
                resolution=720,
                resolution_type="pixel_area",
                accelerator=self.accelerator,
                repeats=0,
            )

    @patch("simpletuner.helpers.metadata.backends.discovery.load_video")
    @patch("simpletuner.helpers.metadata.backends.discovery.TrainingSample")
    def test_process_for_bucket_resolution_frames_strategy(self, mock_training_sample, mock_load_video):
        """Test that resolution_frames strategy creates WxH@F bucket keys."""
        backend = self._create_backend("resolution_frames", frame_interval=25)

        # Mock video data: shape is (frames, height, width, channels)
        mock_video = np.zeros((130, 720, 1280, 3), dtype=np.uint8)
        mock_load_video.return_value = mock_video

        # Mock TrainingSample
        mock_sample = MagicMock()
        mock_sample.prepare.return_value = mock_sample
        mock_sample.target_size = (1280, 720)
        mock_sample.aspect_ratio = 1.78
        mock_sample.crop_coordinates = (0, 0)
        mock_sample.intermediary_size = (1280, 720)
        mock_training_sample.return_value = mock_sample

        # Mock StateTracker.get_model()
        with patch.object(StateTracker, "get_model", return_value=MagicMock()):
            # Set up mock video file
            self.data_backend._files["test_video.mp4"] = b"fake video data"

            bucket_indices = backend._process_for_bucket(
                "test_video.mp4",
                {},
                metadata_updates={},
                statistics={},
            )

            # Should create bucket key in WxH@F format
            # 130 frames with interval 25 -> rounds to 125
            self.assertIn("1280x720@125", bucket_indices)
            self.assertEqual(bucket_indices["1280x720@125"], ["test_video.mp4"])

    @patch("simpletuner.helpers.metadata.backends.discovery.load_video")
    @patch("simpletuner.helpers.metadata.backends.discovery.TrainingSample")
    def test_process_for_bucket_aspect_ratio_strategy(self, mock_training_sample, mock_load_video):
        """Test that aspect_ratio strategy creates float bucket keys."""
        backend = self._create_backend("aspect_ratio")

        # Mock video data
        mock_video = np.zeros((130, 720, 1280, 3), dtype=np.uint8)
        mock_load_video.return_value = mock_video

        # Mock TrainingSample
        mock_sample = MagicMock()
        mock_sample.prepare.return_value = mock_sample
        mock_sample.target_size = (1280, 720)
        mock_sample.aspect_ratio = 1.78
        mock_sample.crop_coordinates = (0, 0)
        mock_sample.intermediary_size = (1280, 720)
        mock_training_sample.return_value = mock_sample

        with patch.object(StateTracker, "get_model", return_value=MagicMock()):
            self.data_backend._files["test_video.mp4"] = b"fake video data"

            bucket_indices = backend._process_for_bucket(
                "test_video.mp4",
                {},
                metadata_updates={},
                statistics={},
            )

            # Should create bucket key as aspect ratio string
            self.assertIn("1.78", bucket_indices)
            self.assertEqual(bucket_indices["1.78"], ["test_video.mp4"])

    @patch("simpletuner.helpers.metadata.backends.discovery.load_video")
    @patch("simpletuner.helpers.metadata.backends.discovery.TrainingSample")
    def test_process_for_bucket_stores_bucket_frames_in_metadata(self, mock_training_sample, mock_load_video):
        """Test that bucket_frames is stored in metadata when using resolution_frames."""
        backend = self._create_backend("resolution_frames", frame_interval=25)

        mock_video = np.zeros((137, 720, 1280, 3), dtype=np.uint8)
        mock_load_video.return_value = mock_video

        mock_sample = MagicMock()
        mock_sample.prepare.return_value = mock_sample
        mock_sample.target_size = (1280, 720)
        mock_sample.aspect_ratio = 1.78
        mock_sample.crop_coordinates = (0, 0)
        mock_sample.intermediary_size = (1280, 720)
        mock_training_sample.return_value = mock_sample

        with patch.object(StateTracker, "get_model", return_value=MagicMock()):
            self.data_backend._files["test_video.mp4"] = b"fake video data"
            metadata_updates = {}

            backend._process_for_bucket(
                "test_video.mp4",
                {},
                metadata_updates=metadata_updates,
                statistics={},
            )

            # Check metadata includes bucket_frames
            self.assertIn("test_video.mp4", metadata_updates)
            self.assertEqual(metadata_updates["test_video.mp4"]["bucket_frames"], 125)
            self.assertEqual(metadata_updates["test_video.mp4"]["num_frames"], 137)


class TestAudioBucketStrategyUnchanged(unittest.TestCase):
    """Verify audio bucket strategy still works as expected."""

    def setUp(self):
        StateTracker.clear_data_backends()
        self.data_backend = MockDataBackend("test-audio")
        self.data_backend.dataset_type = DatasetType.AUDIO
        self.accelerator = Mock()

    def tearDown(self):
        StateTracker.clear_data_backends()

    def test_audio_still_uses_duration_strategy(self):
        """Test that audio datasets still default to duration strategy."""
        StateTracker.set_data_backend_config(
            "test-audio",
            {
                "dataset_type": "audio",
                "audio": {"duration_interval": 15.0},
            },
        )
        StateTracker.set_args(MagicMock())

        with (
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker._save_to_disk",
                return_value=True,
            ),
            patch("pathlib.Path.exists", return_value=True),
        ):
            backend = DiscoveryMetadataBackend(
                id="test-audio",
                instance_data_dir="/fake/path",
                cache_file="/fake/cache",
                metadata_file="/fake/metadata.json",
                batch_size=1,
                data_backend=self.data_backend,
                resolution=1,
                resolution_type="area",
                accelerator=self.accelerator,
                repeats=0,
            )

        self.assertEqual(backend.bucket_strategy, "duration")
        self.assertIsNone(backend.video_frame_interval)


class TestImageBucketStrategyUnchanged(unittest.TestCase):
    """Verify image bucket strategy still works as expected."""

    def setUp(self):
        StateTracker.clear_data_backends()
        self.data_backend = MockDataBackend("test-image")
        self.data_backend.dataset_type = DatasetType.IMAGE
        self.accelerator = Mock()

    def tearDown(self):
        StateTracker.clear_data_backends()

    def test_image_still_uses_aspect_ratio_strategy(self):
        """Test that image datasets still default to aspect_ratio strategy."""
        StateTracker.set_data_backend_config(
            "test-image",
            {"dataset_type": "image"},
        )
        StateTracker.set_args(MagicMock())

        with (
            patch(
                "simpletuner.helpers.training.state_tracker.StateTracker._save_to_disk",
                return_value=True,
            ),
            patch("pathlib.Path.exists", return_value=True),
        ):
            backend = DiscoveryMetadataBackend(
                id="test-image",
                instance_data_dir="/fake/path",
                cache_file="/fake/cache",
                metadata_file="/fake/metadata.json",
                batch_size=1,
                data_backend=self.data_backend,
                resolution=1024,
                resolution_type="pixel",
                accelerator=self.accelerator,
                repeats=0,
            )

        self.assertEqual(backend.bucket_strategy, "aspect_ratio")
        self.assertIsNone(backend.video_frame_interval)


if __name__ == "__main__":
    unittest.main()
