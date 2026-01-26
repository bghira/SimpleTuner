"""Tests for S2V (Sound-to-Video) auto-split audio feature."""

import unittest
from unittest.mock import MagicMock, patch

from simpletuner.helpers.data_backend.dataset_types import DatasetType


class TestS2VAudioInjection(unittest.TestCase):
    """Test automatic S2V audio dataset injection."""

    def setUp(self):
        self.args = MagicMock()
        self.accelerator = MagicMock()
        self.args.cache_dir = "/tmp/cache"
        self.args.output_dir = "/tmp/output"
        self.args.model_family = "wan"
        self.args.resolution = 512
        self.args.resolution_type = "pixel"
        self.args.train_batch_size = 1

    def _make_factory(self):
        """Create a factory with mocked model that requires S2V."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        model = MagicMock()
        model.requires_s2v_datasets.return_value = True

        factory = FactoryRegistry(self.args, self.accelerator, None, None, model)
        return factory

    def test_inject_s2v_audio_creates_audio_backend(self):
        """Test that audio backend is created for video with auto_split."""
        factory = self._make_factory()

        data_backend_config = [
            {
                "id": "test-videos",
                "type": "local",
                "dataset_type": "video",
                "instance_data_dir": "/data/videos",
                "cache_dir_vae": "/cache/vae/videos",
                "audio": {
                    "auto_split": True,
                    "sample_rate": 16000,
                    "channels": 1,
                },
            }
        ]

        result = factory._inject_s2v_audio_configs(data_backend_config)

        # Should have original video backend + auto-generated audio backend
        self.assertEqual(len(result), 2)

        # Find the auto-generated audio backend
        audio_backends = [b for b in result if b.get("dataset_type") == "audio"]
        self.assertEqual(len(audio_backends), 1)

        audio_backend = audio_backends[0]
        self.assertEqual(audio_backend["id"], "test-videos_audio")
        self.assertEqual(audio_backend["type"], "local")
        self.assertEqual(audio_backend["instance_data_dir"], "/data/videos")
        self.assertEqual(audio_backend.get("source_dataset_id"), "test-videos")
        self.assertTrue(audio_backend["audio"]["source_from_video"])
        self.assertEqual(audio_backend["audio"]["sample_rate"], 16000)
        self.assertEqual(audio_backend["audio"]["channels"], 1)

        # Check source backend was modified
        video_backend = result[0]
        self.assertTrue(video_backend.get("_s2v_audio_autoinjected"))
        self.assertEqual(video_backend["s2v_datasets"], ["test-videos_audio"])

    def test_inject_s2v_audio_defaults_to_auto_split(self):
        """Test that auto_split defaults to true when unset."""
        factory = self._make_factory()

        data_backend_config = [
            {
                "id": "test-videos",
                "type": "local",
                "dataset_type": "video",
                "instance_data_dir": "/data/videos",
            }
        ]

        result = factory._inject_s2v_audio_configs(data_backend_config)

        self.assertEqual(len(result), 2)
        video_backend = result[0]
        self.assertTrue(video_backend.get("_s2v_audio_autoinjected"))
        self.assertTrue(video_backend.get("audio", {}).get("auto_split"))

    def test_inject_s2v_audio_skips_without_flag(self):
        """Test that injection is skipped when auto_split is false."""
        factory = self._make_factory()

        data_backend_config = [
            {
                "id": "test-videos",
                "type": "local",
                "dataset_type": "video",
                "instance_data_dir": "/data/videos",
                "audio": {
                    "auto_split": False,
                },
            }
        ]

        result = factory._inject_s2v_audio_configs(data_backend_config)

        # Should only have the original backend
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].get("s2v_datasets"))

    def test_inject_s2v_audio_skips_existing_s2v(self):
        """Test that injection is skipped when s2v_datasets already set."""
        factory = self._make_factory()

        data_backend_config = [
            {
                "id": "test-videos",
                "type": "local",
                "dataset_type": "video",
                "instance_data_dir": "/data/videos",
                "s2v_datasets": ["existing-audio"],
                "audio": {
                    "auto_split": True,
                },
            }
        ]

        result = factory._inject_s2v_audio_configs(data_backend_config)

        # Should only have the original backend
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["s2v_datasets"], ["existing-audio"])

    def test_inject_s2v_audio_skips_non_s2v_models(self):
        """Test that injection is skipped when model doesn't require S2V."""
        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        model = MagicMock()
        model.requires_s2v_datasets.return_value = False

        factory = FactoryRegistry(self.args, self.accelerator, None, None, model)

        data_backend_config = [
            {
                "id": "test-videos",
                "type": "local",
                "dataset_type": "video",
                "instance_data_dir": "/data/videos",
                "audio": {
                    "auto_split": True,
                },
            }
        ]

        result = factory._inject_s2v_audio_configs(data_backend_config)

        # Should only have the original backend
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0].get("s2v_datasets"))

    def test_inject_s2v_audio_inherits_backend_settings(self):
        """Test that S3/HF backend settings are inherited."""
        factory = self._make_factory()

        data_backend_config = [
            {
                "id": "test-videos",
                "type": "aws",
                "dataset_type": "video",
                "instance_data_dir": "videos/",
                "aws_bucket_name": "my-bucket",
                "aws_data_prefix": "datasets/",
                "audio": {
                    "auto_split": True,
                },
            }
        ]

        result = factory._inject_s2v_audio_configs(data_backend_config)

        audio_backend = [b for b in result if b.get("dataset_type") == "audio"][0]
        self.assertEqual(audio_backend["type"], "aws")
        self.assertEqual(audio_backend["aws_bucket_name"], "my-bucket")
        self.assertEqual(audio_backend["aws_data_prefix"], "datasets/")

    def test_inject_s2v_audio_allow_zero_audio(self):
        """Test that allow_zero_audio is passed through."""
        factory = self._make_factory()

        data_backend_config = [
            {
                "id": "test-videos",
                "type": "local",
                "dataset_type": "video",
                "instance_data_dir": "/data/videos",
                "audio": {
                    "auto_split": True,
                    "allow_zero_audio": True,
                },
            }
        ]

        result = factory._inject_s2v_audio_configs(data_backend_config)

        audio_backend = [b for b in result if b.get("dataset_type") == "audio"][0]
        self.assertTrue(audio_backend["audio"]["allow_zero_audio"])


class TestLoadAudioFromVideo(unittest.TestCase):
    """Test audio extraction from video files."""

    def test_generate_zero_audio(self):
        """Test zero audio generation."""
        from simpletuner.helpers.audio.load import generate_zero_audio

        waveform, sample_rate = generate_zero_audio(duration_seconds=5.0, sample_rate=16000, channels=1)

        self.assertEqual(sample_rate, 16000)
        self.assertEqual(waveform.shape, (1, 80000))  # 5 seconds * 16000 samples
        self.assertEqual(waveform.sum().item(), 0.0)

    def test_generate_zero_audio_stereo(self):
        """Test stereo zero audio generation."""
        from simpletuner.helpers.audio.load import generate_zero_audio

        waveform, sample_rate = generate_zero_audio(duration_seconds=2.0, sample_rate=44100, channels=2)

        self.assertEqual(sample_rate, 44100)
        self.assertEqual(waveform.shape, (2, 88200))  # 2 seconds * 44100 samples


class TestS2VSampleConnection(unittest.TestCase):
    """Test S2V sample connection with source_from_video."""

    def test_connect_s2v_samples_source_from_video(self):
        """Test that video path is used as audio path when source_from_video=True."""
        from simpletuner.helpers.training.state_tracker import StateTracker

        # Mock StateTracker.get_s2v_datasets to return a dataset with source_from_video
        s2v_datasets = [
            {
                "config": {
                    "instance_data_dir": "/data/audio",
                    "audio": {
                        "source_from_video": True,
                    },
                }
            }
        ]

        samples = [{"image_path": "/data/videos/test.mp4"}]

        with patch.object(StateTracker, "get_s2v_datasets", return_value=s2v_datasets):
            # Create a minimal sampler mock
            from unittest.mock import MagicMock

            sampler = MagicMock()
            sampler.id = "test-videos"
            sampler.debug_log = MagicMock()

            # Import and bind the method
            from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler

            result = MultiAspectSampler.connect_s2v_samples(sampler, tuple(samples))

            # Video path should be used directly as audio path
            self.assertEqual(result[0]["s2v_audio_path"], "/data/videos/test.mp4")

    def test_connect_s2v_samples_source_from_video_skips_missing_metadata(self):
        """Skip source_from_video samples when the audio metadata is missing."""
        from simpletuner.helpers.training.state_tracker import StateTracker

        metadata_backend = MagicMock()
        metadata_backend.get_metadata_by_filepath.side_effect = lambda path: (
            {"duration_seconds": 1.0} if path.endswith("has_audio.mp4") else None
        )

        s2v_datasets = [
            {
                "id": "audio-backend",
                "config": {"audio": {"source_from_video": True}},
                "metadata_backend": metadata_backend,
            }
        ]

        samples = [
            {"image_path": "/data/videos/has_audio.mp4"},
            {"image_path": "/data/videos/no_audio.mp4"},
        ]

        with patch.object(StateTracker, "get_s2v_datasets", return_value=s2v_datasets):
            sampler = MagicMock()
            sampler.id = "test-videos"
            sampler.debug_log = MagicMock()

            from simpletuner.helpers.multiaspect.sampler import MultiAspectSampler

            result = MultiAspectSampler.connect_s2v_samples(sampler, tuple(samples))

        self.assertEqual(result[0]["s2v_audio_path"], "/data/videos/has_audio.mp4")
        self.assertEqual(result[0]["s2v_audio_backend_id"], "audio-backend")
        self.assertIsNone(result[1]["s2v_audio_path"])
        self.assertIsNone(result[1]["s2v_audio_backend_id"])


class TestConditioningDatasetAudioIsolation(unittest.TestCase):
    """Test that conditioning datasets don't inherit audio settings from video datasets."""

    def test_virtual_conditioning_excludes_audio(self):
        """Test that virtual I2V conditioning datasets don't inherit audio settings."""
        from copy import deepcopy

        from simpletuner.helpers.data_backend.factory import FactoryRegistry

        args = MagicMock()
        args.cache_dir = "/tmp/cache"
        args.output_dir = "/tmp/output"
        args.model_family = "wan"
        args.resolution = 512
        args.resolution_type = "pixel"
        args.train_batch_size = 1
        args.controlnet = False

        model = MagicMock()
        model.requires_s2v_datasets.return_value = True

        factory = FactoryRegistry(args, MagicMock(), None, None, model)

        # Simulate a video backend with audio config (as would be set by _inject_s2v_audio_configs)
        backend = {
            "id": "test-videos",
            "type": "local",
            "dataset_type": "video",
            "instance_data_dir": "/data/videos",
            "cache_dir_vae": "/cache/vae/videos",
            "video": {"is_i2v": True},
            "audio": {
                "auto_split": True,
                "sample_rate": 16000,
                "channels": 1,
            },
            "s2v_datasets": ["test-videos_audio"],
            "_s2v_audio_autoinjected": True,
        }

        data_backend_config = [backend]
        result = factory.process_conditioning_datasets(deepcopy(data_backend_config))

        # Find the generated conditioning dataset
        conditioning_backends = [b for b in result if b.get("dataset_type") == "conditioning"]

        # Should have created a virtual conditioning dataset
        self.assertGreaterEqual(len(conditioning_backends), 1)

        # Conditioning dataset should NOT have audio settings
        for cond_backend in conditioning_backends:
            self.assertIsNone(
                cond_backend.get("audio"),
                f"Conditioning dataset {cond_backend['id']} should not have audio config",
            )
            self.assertIsNone(
                cond_backend.get("s2v_datasets"),
                f"Conditioning dataset {cond_backend['id']} should not have s2v_datasets",
            )
            self.assertIsNone(
                cond_backend.get("_s2v_audio_autoinjected"),
                f"Conditioning dataset {cond_backend['id']} should not have _s2v_audio_autoinjected",
            )

    def test_duplicator_conditioning_excludes_audio(self):
        """Test that DatasetDuplicator strips audio settings from conditioning configs."""
        import os
        import tempfile

        from simpletuner.helpers.metadata.utils.duplicator import DatasetDuplicator

        with tempfile.TemporaryDirectory() as tmpdir:
            global_config = MagicMock()
            global_config.cache_dir = tmpdir
            global_config.controlnet = False
            global_config.conditioning_multidataset_sampling = "separate"

            source_backend_config = {
                "id": "test-videos",
                "type": "local",
                "dataset_type": "video",
                "instance_data_dir": os.path.join(tmpdir, "data/videos"),
                "cache_dir_vae": os.path.join(tmpdir, "cache/vae/videos"),
                "conditioning": [{"type": "reference_strict"}],
                "audio": {
                    "auto_split": True,
                    "sample_rate": 16000,
                    "channels": 1,
                },
                "s2v_datasets": ["test-videos_audio"],
                "_s2v_audio_autoinjected": True,
            }

            modified_source, generated_datasets = DatasetDuplicator.generate_conditioning_datasets(
                global_config=global_config,
                source_backend_config=source_backend_config,
            )

            # Should have generated conditioning datasets
            self.assertGreater(len(generated_datasets), 0)

            # Conditioning datasets should NOT have audio settings
            for cond_dataset in generated_datasets:
                self.assertIsNone(
                    cond_dataset.get("audio"),
                    f"Conditioning dataset {cond_dataset['id']} should not have audio config",
                )
                self.assertIsNone(
                    cond_dataset.get("s2v_datasets"),
                    f"Conditioning dataset {cond_dataset['id']} should not have s2v_datasets",
                )
                self.assertIsNone(
                    cond_dataset.get("_s2v_audio_autoinjected"),
                    f"Conditioning dataset {cond_dataset['id']} should not have _s2v_audio_autoinjected",
                )


if __name__ == "__main__":
    unittest.main()
