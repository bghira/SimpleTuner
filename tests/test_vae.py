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

from simpletuner.helpers.data_backend.dataset_types import DatasetType

if "trainingsample" not in sys.modules:
    trainingsample_stub = types.ModuleType("trainingsample")
    trainingsample_stub.batch_resize_images = lambda *args, **kwargs: []
    trainingsample_stub.batch_center_crop_images = lambda *args, **kwargs: []
    trainingsample_stub.batch_random_crop_images = lambda *args, **kwargs: []
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
from simpletuner.helpers.training import audio_file_extensions, video_file_extensions
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

    @patch("simpletuner.helpers.caching.vae.StateTracker.get_model_family", return_value="anima")
    def test_anima_process_video_latents_uses_wan_style_config_stats(self, _mock_model_family):
        vae_cache = VAECache.__new__(VAECache)
        vae_cache.vae = SimpleNamespace(
            config=SimpleNamespace(
                latents_mean=[1.0, 2.0],
                latents_std=[2.0, 4.0],
                z_dim=2,
            )
        )

        posterior_parameters = torch.tensor(
            [
                [
                    [[[3.0]]],
                    [[[10.0]]],
                    [[[0.0]]],
                    [[[0.0]]],
                ]
            ],
            dtype=torch.float32,
        )

        latents = vae_cache.process_video_latents(posterior_parameters)

        expected = torch.tensor([[[[[1.0]]], [[[2.0]]]]], dtype=torch.float32)
        torch.testing.assert_close(latents, expected)

    def test_video_conditioning_discovers_video_files(self):
        vae_cache = VAECache.__new__(VAECache)
        vae_cache.id = "reference"
        vae_cache.dataset_type_enum = DatasetType.CONDITIONING
        vae_cache.instance_data_dir = "/data/reference"
        vae_cache.cache_dir = "/cache/reference"
        vae_cache.image_data_backend = MagicMock()
        vae_cache.cache_data_backend = MagicMock()
        vae_cache.num_video_frames = 17
        vae_cache.debug_log = MagicMock()
        vae_cache.image_data_backend.list_files.return_value = ["/data/reference/sample.mp4"]
        vae_cache.cache_data_backend.list_files.return_value = []

        with (
            patch("simpletuner.helpers.caching.vae.StateTracker.get_image_files", return_value=None),
            patch(
                "simpletuner.helpers.caching.vae.StateTracker.set_image_files", return_value=["/data/reference/sample.mp4"]
            ),
            patch("simpletuner.helpers.caching.vae.StateTracker.get_vae_cache_files", return_value=None),
            patch("simpletuner.helpers.caching.vae.StateTracker.set_vae_cache_files", return_value=[]),
        ):
            files = vae_cache.discover_all_files()

        self.assertEqual(files, ["/data/reference/sample.mp4"])
        vae_cache.image_data_backend.list_files.assert_called_once_with(
            instance_data_dir="/data/reference",
            file_extensions=video_file_extensions,
        )

    def test_local_unprocessed_file_set_tracks_reassigned_file_list(self):
        vae_cache = VAECache.__new__(VAECache)
        vae_cache.local_unprocessed_files = ["/data/first.png"]

        self.assertIn("/data/first.png", vae_cache._local_unprocessed_file_set())

        vae_cache.local_unprocessed_files = ["/data/second.png"]

        self.assertNotIn("/data/first.png", vae_cache._local_unprocessed_file_set())
        self.assertIn("/data/second.png", vae_cache._local_unprocessed_file_set())


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


class TestMetadataFilterOnSplitShard(unittest.TestCase):
    """A filtered sample must leave this rank's shard the same length as its DP peers'."""

    IMAGES = [f"image-{index}.jpg" for index in range(5)]

    @classmethod
    def _padded_shard_backend(cls, dp_rank, *, world_size=2, images=None):
        from simpletuner.helpers.metadata.backends.base import MetadataBackend
        from simpletuner.helpers.training.state_tracker import StateTracker

        backend = MagicMock(spec=MetadataBackend)
        backend.id = "filtered-shard"
        backend.batch_size = 1
        backend.repeats = 0
        backend.bucket_report = None
        backend.dataset_type = DatasetType.IMAGE
        backend.aspect_ratio_bucket_indices = {"1.0": list(images or cls.IMAGES)}
        backend.read_only = False
        backend.filtering_statistics = None
        backend.accelerator = SimpleNamespace(
            num_processes=world_size,
            process_index=dp_rank,
            is_main_process=dp_rank == 0,
        )
        with (
            patch.dict("os.environ", {"SIMPLETUNER_SHUFFLE_BUCKETS": "0"}),
            patch.object(
                StateTracker,
                "get_args",
                return_value=SimpleNamespace(allow_dataset_oversubscription=True),
            ),
            patch.object(StateTracker, "get_data_backend_config", return_value={}),
        ):
            MetadataBackend.split_buckets_between_processes(
                backend,
                gradient_accumulation_steps=1,
                apply_padding=True,
            )
        return backend

    @staticmethod
    def _cache(metadata_backend):
        from simpletuner.helpers.caching.vae import VAECache

        cache = object.__new__(VAECache)
        cache.id = "filtered-shard"
        cache.metadata_backend = metadata_backend
        return cache

    def test_padded_shard_keeps_its_length_when_a_duplicate_is_filtered(self):
        backend = self._padded_shard_backend(1)
        # Five images over two DP shards: this rank holds three slots, the last of which is a
        # padding copy of the final global item.
        self.assertEqual(backend.aspect_ratio_bucket_indices["1.0"], ["image-3.jpg", "image-4.jpg", "image-4.jpg"])
        self.assertTrue(backend.read_only)

        self._cache(backend)._handle_metadata_filtered_sample(filepath="image-4.jpg", bucket="1.0", reason="problematic")

        shard = backend.aspect_ratio_bucket_indices["1.0"]
        self.assertNotIn("image-4.jpg", shard)
        self.assertEqual(len(shard), 3)
        self.assertEqual(shard, ["image-3.jpg"] * 3)

    def test_padded_shard_keeps_its_length_when_a_unique_sample_is_filtered(self):
        backend = self._padded_shard_backend(1)
        self._cache(backend)._handle_metadata_filtered_sample(filepath="image-3.jpg", bucket="1.0", reason="nsfw")

        shard = backend.aspect_ratio_bucket_indices["1.0"]
        self.assertNotIn("image-3.jpg", shard)
        self.assertEqual(shard, ["image-4.jpg"] * 3)

    def test_a_shard_made_entirely_of_one_filtered_sample_cannot_be_refilled(self):
        # Known limitation: with nothing left in the bucket there is no sample to repeat. The
        # shortened cache is picked up by the next split.
        backend = self._padded_shard_backend(4, world_size=5, images=[f"image-{index}.jpg" for index in range(9)])
        self.assertEqual(backend.aspect_ratio_bucket_indices["1.0"], ["image-8.jpg", "image-8.jpg"])

        self._cache(backend)._handle_metadata_filtered_sample(filepath="image-8.jpg", bucket="1.0", reason="problematic")

        self.assertEqual(backend.aspect_ratio_bucket_indices["1.0"], [])

    def test_unsplit_backend_is_not_repadded(self):
        # Control: before the split the backend holds the dataset, not a shard, so a filtered
        # sample simply disappears.
        backend = self._padded_shard_backend(1)
        backend.aspect_ratio_bucket_indices = {"1.0": list(self.IMAGES)}
        backend.read_only = False

        self._cache(backend)._handle_metadata_filtered_sample(filepath="image-4.jpg", bucket="1.0", reason="problematic")

        self.assertEqual(backend.aspect_ratio_bucket_indices["1.0"], self.IMAGES[:4])

    def test_filter_action_is_still_queued_for_the_unsplit_cache(self):
        backend = self._padded_shard_backend(1)
        cache = self._cache(backend)
        cache._handle_metadata_filtered_sample(filepath="image-4.jpg", bucket="1.0", reason="problematic")

        self.assertEqual(
            [(action["filepath"], action["reason"]) for action in cache._deferred_metadata_filter_actions],
            [("image-4.jpg", "problematic")],
        )


if __name__ == "__main__":
    unittest.main()
