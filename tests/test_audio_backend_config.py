import unittest
from unittest.mock import MagicMock, patch

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.data_backend.factory import FactoryRegistry, init_backend_config


class TestAudioBackendConfig(unittest.TestCase):
    """Test audio backend configuration logic."""

    def setUp(self):
        self.args = MagicMock()
        self.accelerator = MagicMock()
        self.args.audio_min_duration_seconds = 1.0
        self.args.audio_max_duration_seconds = 30.0
        self.args.audio_channels = 1
        self.args.audio_bucket_strategy = "duration"
        self.args.audio_duration_interval = 3.0
        self.args.audio_truncation_mode = "beginning"
        self.args.train_batch_size = 1
        self.args.resolution = 512
        self.args.resolution_type = "pixel"

    def test_init_backend_config_audio_defaults(self):
        """Test that global args are used when dataset config is empty."""
        backend_config = {"id": "test_audio", "type": "local", "dataset_type": "audio", "instance_data_dir": "/tmp/audio"}

        result = init_backend_config(backend_config, self.args, self.accelerator)

        self.assertIn("audio", result["config"])
        audio_conf = result["config"]["audio"]
        self.assertEqual(audio_conf.get("min_duration_seconds"), 1.0)
        self.assertEqual(audio_conf.get("max_duration_seconds"), 30.0)
        self.assertEqual(audio_conf.get("truncation_mode"), "beginning")

    def test_init_backend_config_audio_overrides(self):
        """Test that dataset config overrides global args."""
        backend_config = {
            "id": "test_audio",
            "type": "local",
            "dataset_type": "audio",
            "instance_data_dir": "/tmp/audio",
            "audio_min_duration_seconds": 5.0,
            "audio_truncation_mode": "random",
        }

        result = init_backend_config(backend_config, self.args, self.accelerator)

        audio_conf = result["config"]["audio"]
        self.assertEqual(audio_conf.get("min_duration_seconds"), 5.0)
        self.assertEqual(audio_conf.get("truncation_mode"), "random")
        # Should still have global defaults for unspecified
        self.assertEqual(audio_conf.get("max_duration_seconds"), 30.0)

    @patch("simpletuner.helpers.audio.load_audio")
    def test_attach_audio_backend(self, mock_load_audio):
        """Test that FactoryRegistry correctly attaches the runtime backend."""
        factory = FactoryRegistry(self.args, self.accelerator, None, None, MagicMock())

        init_backend = {
            "id": "test_audio",
            "dataset_type": DatasetType.AUDIO,
            "config": {
                "audio": {
                    "min_duration_seconds": 2.0,
                    "max_duration_seconds": 10.0,
                    "channels": 2,
                    "truncation_mode": "end",
                    "cache_dir": "/tmp/cache",
                }
            },
            "instance_data_dir": "/tmp/data",
        }

        factory._attach_audio_backend(init_backend, DatasetType.AUDIO)

        self.assertIn("audio_data_backend", init_backend)
        backend = init_backend["audio_data_backend"]
        self.assertEqual(backend["min_duration_seconds"], 2.0)
        self.assertEqual(backend["max_duration_seconds"], 10.0)
        self.assertEqual(backend["channels"], 2)
        self.assertEqual(backend["truncation_mode"], "end")


if __name__ == "__main__":
    unittest.main()
