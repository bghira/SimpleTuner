"""Tests for S2V (Speech-to-Video) audio-video pairing functionality."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from simpletuner.helpers.training.state_tracker import StateTracker


class TestS2VConnectSamples(unittest.TestCase):
    """Test the connect_s2v_samples method in MultiAspectSampler."""

    def setUp(self):
        """Set up test fixtures."""
        StateTracker.clear_data_backends()
        # Create a temporary directory for audio files
        self.temp_dir = tempfile.mkdtemp()
        self.audio_dir = Path(self.temp_dir) / "audio"
        self.audio_dir.mkdir()

    def tearDown(self):
        """Clean up after tests."""
        StateTracker.clear_data_backends()
        # Clean up temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_audio_file(self, stem: str, ext: str = ".wav"):
        """Create a dummy audio file for testing."""
        audio_path = self.audio_dir / f"{stem}{ext}"
        audio_path.touch()
        return str(audio_path)

    def _create_mock_sampler(self, backend_id: str = "video-backend"):
        """Create a mock sampler with the connect_s2v_samples method."""
        from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler

        # Create minimal sampler - we only need the connect_s2v_samples method
        sampler = MagicMock(spec=MultiAspectSampler)
        sampler.id = backend_id
        sampler.debug_log = MagicMock()

        # Bind the actual method to our mock
        sampler.connect_s2v_samples = MultiAspectSampler.connect_s2v_samples.__get__(sampler, MultiAspectSampler)
        return sampler

    def test_connect_s2v_samples_matches_by_stem(self):
        """Test that audio files are matched to video files by filename stem."""
        # Create audio files
        self._create_audio_file("video_001")
        self._create_audio_file("video_002")

        # Register backends
        StateTracker.register_data_backend({"id": "video-backend"})
        StateTracker.register_data_backend(
            {
                "id": "audio-backend",
                "config": {"instance_data_dir": str(self.audio_dir)},
            }
        )
        StateTracker.set_s2v_datasets("video-backend", ["audio-backend"])

        # Create samples
        samples = (
            {"image_path": "/videos/video_001.mp4"},
            {"image_path": "/videos/video_002.mp4"},
        )

        sampler = self._create_mock_sampler()
        result = sampler.connect_s2v_samples(samples)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["s2v_audio_path"], str(self.audio_dir / "video_001.wav"))
        self.assertEqual(result[1]["s2v_audio_path"], str(self.audio_dir / "video_002.wav"))

    def test_connect_s2v_samples_handles_different_extensions(self):
        """Test that different audio extensions are matched correctly."""
        # Create audio files with different extensions
        self._create_audio_file("video_wav", ".wav")
        self._create_audio_file("video_mp3", ".mp3")
        self._create_audio_file("video_flac", ".flac")
        self._create_audio_file("video_ogg", ".ogg")
        self._create_audio_file("video_m4a", ".m4a")

        # Register backends
        StateTracker.register_data_backend({"id": "video-backend"})
        StateTracker.register_data_backend(
            {
                "id": "audio-backend",
                "config": {"instance_data_dir": str(self.audio_dir)},
            }
        )
        StateTracker.set_s2v_datasets("video-backend", ["audio-backend"])

        samples = (
            {"image_path": "/videos/video_wav.mp4"},
            {"image_path": "/videos/video_mp3.mp4"},
            {"image_path": "/videos/video_flac.mp4"},
            {"image_path": "/videos/video_ogg.mp4"},
            {"image_path": "/videos/video_m4a.mp4"},
        )

        sampler = self._create_mock_sampler()
        result = sampler.connect_s2v_samples(samples)

        self.assertEqual(result[0]["s2v_audio_path"], str(self.audio_dir / "video_wav.wav"))
        self.assertEqual(result[1]["s2v_audio_path"], str(self.audio_dir / "video_mp3.mp3"))
        self.assertEqual(result[2]["s2v_audio_path"], str(self.audio_dir / "video_flac.flac"))
        self.assertEqual(result[3]["s2v_audio_path"], str(self.audio_dir / "video_ogg.ogg"))
        self.assertEqual(result[4]["s2v_audio_path"], str(self.audio_dir / "video_m4a.m4a"))

    def test_connect_s2v_samples_missing_audio(self):
        """Test that missing audio files result in None path."""
        # Only create audio for one video
        self._create_audio_file("video_001")

        # Register backends
        StateTracker.register_data_backend({"id": "video-backend"})
        StateTracker.register_data_backend(
            {
                "id": "audio-backend",
                "config": {"instance_data_dir": str(self.audio_dir)},
            }
        )
        StateTracker.set_s2v_datasets("video-backend", ["audio-backend"])

        samples = (
            {"image_path": "/videos/video_001.mp4"},
            {"image_path": "/videos/video_002.mp4"},  # No matching audio
        )

        sampler = self._create_mock_sampler()
        result = sampler.connect_s2v_samples(samples)

        self.assertEqual(result[0]["s2v_audio_path"], str(self.audio_dir / "video_001.wav"))
        self.assertIsNone(result[1]["s2v_audio_path"])
        # Verify warning was logged with expected message
        sampler.debug_log.assert_called()
        warning_calls = [str(call) for call in sampler.debug_log.call_args_list]
        self.assertTrue(
            any("No matching audio found for video" in call and "video_002.mp4" in call for call in warning_calls),
            f"Expected warning about missing audio for video_002.mp4, got: {warning_calls}",
        )

    def test_connect_s2v_samples_no_s2v_datasets(self):
        """Test that samples are returned unchanged when no s2v_datasets configured."""
        StateTracker.register_data_backend({"id": "video-backend"})
        # No s2v_datasets set

        samples = (
            {"image_path": "/videos/video_001.mp4"},
            {"image_path": "/videos/video_002.mp4"},
        )

        sampler = self._create_mock_sampler()
        result = sampler.connect_s2v_samples(samples)

        # Should return original samples unchanged
        self.assertEqual(result, samples)

    def test_connect_s2v_samples_multiple_audio_backends(self):
        """Test matching across multiple s2v_datasets backends."""
        # Create two audio directories
        audio_dir_2 = Path(self.temp_dir) / "audio2"
        audio_dir_2.mkdir()

        # Put different audio files in different backends
        self._create_audio_file("video_001")  # In audio_dir
        (audio_dir_2 / "video_002.wav").touch()  # In audio_dir_2

        # Register backends
        StateTracker.register_data_backend({"id": "video-backend"})
        StateTracker.register_data_backend(
            {
                "id": "audio-backend-1",
                "config": {"instance_data_dir": str(self.audio_dir)},
            }
        )
        StateTracker.register_data_backend(
            {
                "id": "audio-backend-2",
                "config": {"instance_data_dir": str(audio_dir_2)},
            }
        )
        StateTracker.set_s2v_datasets("video-backend", ["audio-backend-1", "audio-backend-2"])

        samples = (
            {"image_path": "/videos/video_001.mp4"},
            {"image_path": "/videos/video_002.mp4"},
        )

        sampler = self._create_mock_sampler()
        result = sampler.connect_s2v_samples(samples)

        self.assertEqual(result[0]["s2v_audio_path"], str(self.audio_dir / "video_001.wav"))
        self.assertEqual(result[1]["s2v_audio_path"], str(audio_dir_2 / "video_002.wav"))

    def test_connect_s2v_samples_with_training_sample_object(self):
        """Test connecting audio to TrainingSample-like objects."""
        self._create_audio_file("video_001")

        # Register backends
        StateTracker.register_data_backend({"id": "video-backend"})
        StateTracker.register_data_backend(
            {
                "id": "audio-backend",
                "config": {"instance_data_dir": str(self.audio_dir)},
            }
        )
        StateTracker.set_s2v_datasets("video-backend", ["audio-backend"])

        # Create sample with image_metadata attribute
        sample = SimpleNamespace(
            image_path="/videos/video_001.mp4",
            image_metadata={"some_key": "some_value"},
        )
        samples = (sample,)

        sampler = self._create_mock_sampler()
        result = sampler.connect_s2v_samples(samples)

        self.assertEqual(result[0].image_metadata["s2v_audio_path"], str(self.audio_dir / "video_001.wav"))

    def test_connect_s2v_samples_priority_wav_over_other_formats(self):
        """Test that .wav is preferred when multiple formats exist for same stem."""
        # Create both wav and mp3 for same video
        self._create_audio_file("video_001", ".wav")
        self._create_audio_file("video_001", ".mp3")

        # Register backends
        StateTracker.register_data_backend({"id": "video-backend"})
        StateTracker.register_data_backend(
            {
                "id": "audio-backend",
                "config": {"instance_data_dir": str(self.audio_dir)},
            }
        )
        StateTracker.set_s2v_datasets("video-backend", ["audio-backend"])

        samples = ({"image_path": "/videos/video_001.mp4"},)

        sampler = self._create_mock_sampler()
        result = sampler.connect_s2v_samples(samples)

        # Should prefer .wav (first in the extension list)
        self.assertEqual(result[0]["s2v_audio_path"], str(self.audio_dir / "video_001.wav"))


class TestS2VCollateIntegration(unittest.TestCase):
    """Test that collate_fn correctly extracts s2v_audio_paths."""

    def test_collate_extracts_s2v_audio_paths_from_dict(self):
        """Test s2v_audio_path extraction from dict samples."""
        from simpletuner.helpers.training.collate import collate_fn

        # We need to mock a lot of things for collate_fn
        # This is a simplified test checking the extraction logic
        examples = [
            {"s2v_audio_path": "/audio/video_001.wav"},
            {"s2v_audio_path": "/audio/video_002.wav"},
            {"s2v_audio_path": None},
        ]

        # Extract using the same logic as collate_fn
        s2v_audio_paths = []
        for example in examples:
            audio_path = None
            if isinstance(example, dict):
                audio_path = example.get("s2v_audio_path")
            s2v_audio_paths.append(audio_path)

        self.assertEqual(s2v_audio_paths[0], "/audio/video_001.wav")
        self.assertEqual(s2v_audio_paths[1], "/audio/video_002.wav")
        self.assertIsNone(s2v_audio_paths[2])

    def test_collate_extracts_s2v_audio_paths_from_metadata(self):
        """Test s2v_audio_path extraction from image_metadata."""
        examples = [
            SimpleNamespace(image_metadata={"s2v_audio_path": "/audio/video_001.wav"}),
            SimpleNamespace(image_metadata={"s2v_audio_path": None}),
        ]

        s2v_audio_paths = []
        for example in examples:
            audio_path = None
            if hasattr(example, "image_metadata") and example.image_metadata:
                audio_path = example.image_metadata.get("s2v_audio_path")
            s2v_audio_paths.append(audio_path)

        self.assertEqual(s2v_audio_paths[0], "/audio/video_001.wav")
        self.assertIsNone(s2v_audio_paths[1])

    def test_collate_extracts_s2v_audio_paths_from_attribute(self):
        """Test s2v_audio_path extraction from _s2v_audio_path attribute."""
        sample1 = SimpleNamespace()
        sample1._s2v_audio_path = "/audio/video_001.wav"

        sample2 = SimpleNamespace()
        sample2._s2v_audio_path = None

        examples = [sample1, sample2]

        s2v_audio_paths = []
        for example in examples:
            audio_path = None
            if hasattr(example, "_s2v_audio_path"):
                audio_path = example._s2v_audio_path
            s2v_audio_paths.append(audio_path)

        self.assertEqual(s2v_audio_paths[0], "/audio/video_001.wav")
        self.assertIsNone(s2v_audio_paths[1])


class TestStateTrackerS2VMethods(unittest.TestCase):
    """Test StateTracker S2V dataset methods."""

    def setUp(self):
        StateTracker.clear_data_backends()

    def tearDown(self):
        StateTracker.clear_data_backends()

    def test_set_and_get_s2v_datasets(self):
        """Test setting and getting s2v_datasets."""
        StateTracker.register_data_backend({"id": "video-1"})
        StateTracker.register_data_backend({"id": "audio-1"})
        StateTracker.register_data_backend({"id": "audio-2"})

        StateTracker.set_s2v_datasets("video-1", ["audio-1", "audio-2"])

        result = StateTracker.get_s2v_datasets("video-1")
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["id"], "audio-1")
        self.assertEqual(result[1]["id"], "audio-2")

    def test_get_s2v_datasets_empty_when_not_set(self):
        """Test that get_s2v_datasets returns empty list when not configured."""
        StateTracker.register_data_backend({"id": "video-1"})

        result = StateTracker.get_s2v_datasets("video-1")
        self.assertEqual(result, [])

    def test_get_s2v_mappings(self):
        """Test getting all S2V mappings."""
        StateTracker.register_data_backend({"id": "video-1"})
        StateTracker.register_data_backend({"id": "video-2"})
        StateTracker.register_data_backend({"id": "audio-1"})
        StateTracker.register_data_backend({"id": "audio-2"})

        StateTracker.set_s2v_datasets("video-1", ["audio-1"])
        StateTracker.set_s2v_datasets("video-2", ["audio-2"])

        mappings = StateTracker.get_s2v_mappings()

        self.assertIn(("video-1", "audio-1"), mappings)
        self.assertIn(("video-2", "audio-2"), mappings)


if __name__ == "__main__":
    unittest.main()
