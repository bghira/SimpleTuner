import importlib.machinery
import sys
import types
import unittest
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

if "trainingsample" not in sys.modules:
    trainingsample_stub = types.ModuleType("trainingsample")
    trainingsample_stub.batch_resize_images = lambda *args, **kwargs: []
    trainingsample_stub.batch_center_crop_images = lambda *args, **kwargs: []
    trainingsample_stub.batch_random_crop_images = lambda *args, **kwargs: []
    trainingsample_stub.batch_calculate_luminance = lambda *args, **kwargs: []
    trainingsample_stub.batch_resize_videos = lambda *args, **kwargs: []
    trainingsample_stub.__spec__ = importlib.machinery.ModuleSpec("trainingsample", loader=None)
    sys.modules["trainingsample"] = trainingsample_stub

if "imageio" not in sys.modules:

    class _DummyWriter:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def append_data(self, *args, **kwargs):
            return None

    imageio_stub = types.ModuleType("imageio")
    imageio_stub.get_writer = lambda *args, **kwargs: _DummyWriter()
    imageio_stub.__spec__ = importlib.machinery.ModuleSpec("imageio", loader=None)
    sys.modules["imageio"] = imageio_stub

from simpletuner.helpers.caching.vae import VAECache
from simpletuner.helpers.image_manipulation.training_sample import TrainingSample
from simpletuner.helpers.models.common import AudioModelFoundation
from simpletuner.helpers.training import audio_file_extensions
from simpletuner.helpers.training.state_tracker import StateTracker


class TestVaeCache(unittest.TestCase):
    def test_filename_mapping(self):
        # Test cases - hash_filenames is always True in production
        test_cases = [
            # 0 Filepath ends with .pt (no change expected in the path)
            {"image_path": "/data/image1.pt", "cache_path": "/data/image1.pt"},
            # 1 Normal filepath (hashed)
            {"image_path": "/data/image1.png", "cache_path": "cache/" + sha256("image1".encode()).hexdigest() + ".pt"},
            # 2, 3 Nested subdirectories (hashed)
            {
                "image_path": "/data/subdir1/subdir2/image2.jpg",
                "cache_path": "cache/subdir1/subdir2/" + sha256("image2".encode()).hexdigest() + ".pt",
            },
            {
                "image_path": "data/subdir1/subdir2/image2.jpg",
                "cache_path": "cache/subdir1/subdir2/" + sha256("image2".encode()).hexdigest() + ".pt",
                "instance_dir": "data",
            },
            # 4 No instance_data_dir, direct cache dir placement (hashed)
            {
                "image_path": "/anotherdir/image3.png",
                "cache_path": "cache/" + sha256("image3".encode()).hexdigest() + ".pt",
                "instance_dir": None,
            },
            # 5 Instance data directory is None (hashed)
            {
                "image_path": "/data/image4.png",
                "cache_path": "cache/" + sha256("image4".encode()).hexdigest() + ".pt",
                "instance_dir": None,
            },
            # 6 Filepath in root directory (hashed)
            {"image_path": "/image5.png", "cache_path": "cache/" + sha256("image5".encode()).hexdigest() + ".pt"},
            # 7 Another hashed filename test
            {
                "image_path": "/data/image6.png",
                "cache_path": "cache/" + sha256("image6".encode()).hexdigest() + ".pt",
            },
            # 8 Another hashed filename test
            {"image_path": "/data/image7.png", "cache_path": "cache/" + sha256("image7".encode()).hexdigest() + ".pt"},
        ]

        # Running test cases - hash_filenames is always True
        for i, test_case in enumerate(test_cases, 1):
            filepath = test_case["image_path"]
            expected = test_case["cache_path"]
            cache_dir = test_case.get("cache_dir", "cache")
            instance_dir = test_case.get("instance_dir", "/data")
            vae_cache = VAECache(
                id="test-cache",
                vae=None,
                accelerator=None,
                metadata_backend=None,
                image_data_backend=None,
                hash_filenames=True,  # always enabled
                instance_data_dir=instance_dir,
                cache_dir=cache_dir,
                model=MagicMock(),
            )
            generated = vae_cache.generate_vae_cache_filename(filepath)[0]
            self.assertEqual(generated, expected, f"Test {i} failed: {generated} != {expected}")


class DummyAccelerator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.is_local_main_process = True
        self.is_main_process = True

    def wait_for_everyone(self):
        return None


class DummyMetadataBackend:
    def __init__(self, metadata: dict):
        self._metadata = metadata
        self.image_metadata_loaded = True

    def load_image_metadata(self):
        self.image_metadata_loaded = True

    def get_metadata_by_filepath(self, filepath, data_backend_id=None):
        return self._metadata.get(filepath, {})

    def get_metadata_attribute_by_filepath(self, filepath, attribute):
        if attribute == "aspect_bucket":
            return self._metadata.get(filepath, {}).get("aspect_bucket", "audio")
        return None


class DummyModel:
    def __init__(self):
        self.transform_calls = []

    def get_transforms(self, dataset_type: str = "image"):
        def _transform(sample):
            self.transform_calls.append(sample)
            return sample["waveform"] * 2

        return _transform


class MiniAudioModel(AudioModelFoundation):
    TEXT_ENCODER_CONFIGURATION = {}

    def __init__(self):
        self.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.config = SimpleNamespace(weight_dtype=torch.float32)

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        return {}

    def convert_text_embed_for_pipeline(self, text_embedding) -> dict:
        return {}

    def convert_negative_text_embed_for_pipeline(self, text_embedding) -> dict:
        return {}

    def model_predict(self, *args, **kwargs):
        raise NotImplementedError


class TestVaeCacheAudio(unittest.TestCase):
    @patch("simpletuner.helpers.caching.vae.StateTracker.set_vae_cache_files")
    @patch("simpletuner.helpers.caching.vae.StateTracker.get_vae_cache_files", return_value=[])
    @patch("simpletuner.helpers.caching.vae.StateTracker.set_image_files", return_value={})
    @patch("simpletuner.helpers.caching.vae.StateTracker.get_image_files", return_value=None)
    def test_discover_all_files_audio_extensions(
        self,
        mock_get_image_files,
        mock_set_image_files,
        mock_get_vae_cache_files,
        mock_set_vae_cache_files,
    ):
        image_backend = MagicMock()
        image_backend.id = "audio-cache"
        image_backend.list_files.return_value = []
        cache_backend = MagicMock()
        cache_backend.type = "local"
        cache_backend.list_files.return_value = []
        cache_backend.create_directory = MagicMock()
        accelerator = DummyAccelerator()
        vae = MagicMock()
        vae.dtype = torch.float32
        model = MagicMock()
        model.get_transforms.return_value = MagicMock(return_value=torch.zeros(1))
        metadata_backend = DummyMetadataBackend(metadata={})

        vae_cache = VAECache(
            id="audio-cache",
            model=model,
            vae=vae,
            accelerator=accelerator,
            metadata_backend=metadata_backend,
            instance_data_dir="/tmp/audio",
            image_data_backend=image_backend,
            cache_data_backend=cache_backend,
            dataset_type="audio",
        )

        vae_cache.discover_all_files()

        self.assertTrue(image_backend.list_files.called)
        kwargs = image_backend.list_files.call_args.kwargs
        self.assertTrue(set(kwargs["file_extensions"]).issuperset(audio_file_extensions))
        cache_backend.create_directory.assert_called_once()
        mock_get_image_files.assert_called_once_with(data_backend_id="audio-cache")
        mock_set_image_files.assert_called_once()
        mock_get_vae_cache_files.assert_called_once_with(data_backend_id="audio-cache")
        mock_set_vae_cache_files.assert_called_once()

    def test_audio_model_foundation_encode_with_vae(self):
        class DummyVAE:
            def encode(self, audio):
                return audio * 2, torch.tensor([audio.shape[-1]])

        model = MiniAudioModel()
        samples = torch.ones(2, 3, 4)
        output = model.encode_with_vae(DummyVAE(), samples)
        self.assertIn("latents", output)
        self.assertIn("latent_lengths", output)
        self.assertTrue(torch.equal(output["latents"], samples * 2))
        self.assertEqual(output["latent_lengths"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
