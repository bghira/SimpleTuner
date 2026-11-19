import importlib.machinery
import sys
import types
import unittest
from hashlib import sha256
from queue import Queue
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

from simpletuner.helpers.caching.vae import VAECache, prepare_sample
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
    def test_audio_batch_metadata_uses_encoded_waveform_length(self):
        vae_cache = VAECache.__new__(VAECache)
        vae_cache.id = "audio-cache"
        vae_cache.dataset_type_enum = DatasetType.AUDIO
        record = {"num_samples": 350}

        entries = vae_cache._gather_sample_metadata(["track.wav"], {"track.wav": record}, samples=torch.zeros(1, 2, 300))

        self.assertEqual(entries[0]["metadata"]["num_samples"], 300)
        self.assertEqual(record["num_samples"], 350)

    def test_audio_preparation_truncates_to_duration_bucket(self):
        vae_cache = VAECache.__new__(VAECache)
        vae_cache.id = "audio-cache"
        vae_cache._log_audio_tensor_stats = MagicMock()

        with patch.object(
            StateTracker,
            "get_data_backend_config",
            return_value={"audio": {"source_from_video": False}},
        ):
            prepared = vae_cache._prepare_audio_sample(
                "track.wav",
                {"waveform": torch.arange(700, dtype=torch.float32).reshape(2, 350), "sample_rate": 100},
                {"bucket_duration_seconds": 3.0, "truncation_mode": "beginning"},
            )

        self.assertEqual(prepared["waveform"].shape, (2, 300))
        self.assertEqual(prepared["metadata"]["duration_seconds"], 3.0)

    def test_audio_preparation_preserves_max_duration_limit(self):
        vae_cache = VAECache.__new__(VAECache)
        vae_cache.id = "audio-cache"
        vae_cache._log_audio_tensor_stats = MagicMock()

        with patch.object(
            StateTracker,
            "get_data_backend_config",
            return_value={"audio": {"max_duration_seconds": 3.0}},
        ):
            prepared = vae_cache._prepare_audio_sample(
                "track.wav",
                {"waveform": torch.zeros(2, 350), "sample_rate": 100},
                {},
            )

        self.assertEqual(prepared["waveform"].shape, (2, 300))

    def test_video_audio_preparation_does_not_truncate_to_duration_bucket(self):
        vae_cache = VAECache.__new__(VAECache)
        vae_cache.id = "audio-cache"
        vae_cache._log_audio_tensor_stats = MagicMock()

        with patch.object(
            StateTracker,
            "get_data_backend_config",
            return_value={"audio": {"source_from_video": True}},
        ):
            prepared = vae_cache._prepare_audio_sample(
                "track.mp4",
                {"waveform": torch.zeros(2, 350), "sample_rate": 100},
                {"bucket_duration_seconds": 3.0, "truncation_mode": "beginning"},
            )

        self.assertEqual(prepared["waveform"].shape, (2, 350))

    def test_audio_vae_queue_partitions_variable_length_waveforms(self):
        vae_cache = VAECache.__new__(VAECache)
        vae_cache.dataset_type_enum = DatasetType.AUDIO
        vae_cache.vae_batch_size = 4
        vae_cache.vae_input_queue = Queue()
        vae_cache._cache_vae_dtype = MagicMock(return_value=torch.float32)
        vae_cache.generate_vae_cache_filename = MagicMock(side_effect=lambda filepath: (f"{filepath}.pt", f"{filepath}.pt"))
        vae_cache.encode_images = MagicMock(
            side_effect=lambda images, filepaths, load_from_cache, records: [torch.zeros(1) for _ in images]
        )
        vae_cache._handle_metadata_filtered_sample = MagicMock()
        vae_cache.debug_log = MagicMock()
        vae_cache.vae_input_queue.put((torch.zeros(2, 100), "short.wav", "3s", False))
        vae_cache.vae_input_queue.put((torch.zeros(2, 120), "long.wav", "3s", False))
        vae_cache.vae_input_queue.put((torch.zeros(2, 100), "short-2.wav", "3s", True))

        output = vae_cache._encode_images_in_batch(disable_queue=True)

        self.assertEqual([entry[1] for entry in output], ["short.wav", "long.wav", "short-2.wav"])
        self.assertEqual(vae_cache.encode_images.call_count, 2)
        encoded_shapes = [
            [tuple(sample.shape) for sample in call.args[0]] for call in vae_cache.encode_images.call_args_list
        ]
        self.assertEqual(encoded_shapes, [[(2, 100), (2, 100)], [(2, 120)]])

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


class TestVaeCacheRecords(unittest.TestCase):
    """Geometry reaches the VAE cache in the batch's records; only a caller without records has it looked up."""

    RECORD = {
        "original_size": (64, 48),
        "target_size": (32, 24),
        "intermediary_size": (32, 24),
        "crop_coordinates": (0, 0),
        "aspect_ratio": 1.33,
    }

    def _ondemand_cache(self, cached=()):
        cache = VAECache.__new__(VAECache)
        cache.id = "images"
        cache.vae_cache_ondemand = True
        cache.dataset_type_enum = DatasetType.IMAGE
        cache.metadata_backend = MagicMock()
        cache.metadata_backend.get_metadata_by_filepath.side_effect = lambda path: {"looked_up": path}
        cache.cache_data_backend = MagicMock()
        cache.cache_data_backend.exists.side_effect = lambda filename: filename in {f"{path}.pt" for path in cached}
        cache.generate_vae_cache_filename = lambda filepath: (f"{filepath}.pt", f"{filepath}.pt")
        cache._read_from_storage_concurrently = lambda paths, hide_errors=False: [(path, "pixels") for path in paths]
        cache._read_from_storage = MagicMock(side_effect=lambda filename, hide_errors=False: f"cached:{filename}")
        cache._process_images_in_batch = MagicMock(
            side_effect=lambda paths, data, disable_queue=False, records=None: [
                ("pixels", path, None, index == len(paths) - 1) for index, path in enumerate(paths)
            ]
        )
        cache._encode_images_in_batch = MagicMock(return_value=["encoded"])
        cache._write_latents_in_batch = MagicMock(side_effect=lambda outputs: ["latent-a", "latent-b"])
        return cache

    def test_an_ondemand_batch_is_prepared_from_the_records_it_brings(self):
        cache = self._ondemand_cache()
        records = {"a.png": dict(self.RECORD), "b.png": dict(self.RECORD)}
        latents = cache.encode_images([None, None], ["a.png", "b.png"], records=records)
        self.assertEqual(latents, ["latent-a", "latent-b"])
        self.assertIs(cache._process_images_in_batch.call_args.kwargs["records"], records)
        self.assertIs(cache._encode_images_in_batch.call_args.kwargs["records"], records)
        cache.metadata_backend.get_metadata_by_filepath.assert_not_called()

    def test_an_ondemand_caller_with_only_paths_gets_the_metadata_backends_geometry(self):
        cache = self._ondemand_cache()
        cache.encode_images([None, None], ["a.png", "b.png"])
        expected = {"a.png": {"looked_up": "a.png"}, "b.png": {"looked_up": "b.png"}}
        self.assertEqual(cache._process_images_in_batch.call_args.kwargs["records"], expected)
        self.assertEqual(cache._encode_images_in_batch.call_args.kwargs["records"], expected)

    def test_cached_latents_need_no_geometry(self):
        cache = self._ondemand_cache(cached=("a.png", "b.png"))
        self.assertEqual(cache.encode_images([None, None], ["a.png", "b.png"]), ["cached:a.png.pt", "cached:b.png.pt"])
        cache.metadata_backend.get_metadata_by_filepath.assert_not_called()
        cache._process_images_in_batch.assert_not_called()

    def test_images_are_prepared_with_their_records(self):
        cache = VAECache.__new__(VAECache)
        cache.id = "images"
        cache.dataset_type = "image"
        cache.dataset_type_enum = DatasetType.IMAGE
        cache.model = MagicMock()
        cache.accelerator = SimpleNamespace(device="cpu")
        cache.vae = SimpleNamespace(dtype=torch.float32)
        cache.metadata_backend = MagicMock()
        cache.debug_log = MagicMock()
        cache.transform_sample = lambda image: torch.zeros(3, 2, 2)
        cache.batch_processor = MagicMock()
        cache.batch_processor.process_aspect_grouped_images.side_effect = lambda groups, records: [
            (path, np.zeros((4, 4, 3), dtype=np.uint8), records[path]) for path, _, _ in groups[None]
        ]
        records = {"a.png": dict(self.RECORD), "b.png": dict(self.RECORD, crop_coordinates=(4, 0))}
        with patch("simpletuner.helpers.caching.vae.prepare_sample", return_value=("image", (0, 0), 1.33)) as prepare:
            outputs = cache._process_images_in_batch(
                ["a.png", "b.png"], ["pixels-a", "pixels-b"], disable_queue=True, records=records
            )
        self.assertIs(cache.batch_processor.process_aspect_grouped_images.call_args.args[1], records)
        prepared_with = {call.kwargs["filepath"]: call.kwargs["metadata"] for call in prepare.call_args_list}
        self.assertEqual(prepared_with, records)
        self.assertEqual(sorted(output[1] for output in outputs), ["a.png", "b.png"])
        cache.metadata_backend.get_metadata_by_filepath.assert_not_called()
        cache.metadata_backend.get_metadata_attribute_by_filepath.assert_not_called()

    def test_queued_images_are_prepared_with_the_metadata_backends_geometry(self):
        cache = VAECache.__new__(VAECache)
        cache.id = "audio"
        cache.dataset_type_enum = DatasetType.AUDIO
        cache.metadata_backend = MagicMock()
        cache.metadata_backend.get_metadata_by_filepath.side_effect = lambda path: {"looked_up": path}
        cache.process_queue = Queue()
        cache.process_queue.put(("track.wav", "waveform", "3s"))
        cache._process_audio_samples_in_batch = MagicMock(return_value=[])
        cache._process_images_in_batch()
        initial_data, records = cache._process_audio_samples_in_batch.call_args.args
        self.assertEqual(initial_data, [("track.wav", "waveform", "3s")])
        self.assertEqual(records, {"track.wav": {"looked_up": "track.wav"}})

    def _prepare(self, metadata, backends, mappings):
        with (
            patch.object(StateTracker, "get_data_backend", side_effect=lambda backend_id: backends[backend_id]),
            patch.object(StateTracker, "data_backends", backends),
            patch.object(StateTracker, "get_conditioning_mappings", return_value=mappings),
            patch("simpletuner.helpers.caching.vae.TrainingSample") as training_sample,
        ):
            prepare_sample(image="pixels", data_backend_id="control", filepath="a.png", model="model", metadata=metadata)
        return training_sample

    def test_prepare_sample_uses_the_metadata_it_is_given(self):
        backends = {"control": {"id": "control", "dataset_type": "eval"}}
        training_sample = self._prepare(self.RECORD, backends, mappings=[])
        training_sample.assert_called_once_with(
            image="pixels", data_backend_id="control", image_metadata=self.RECORD, image_path="a.png", model="model"
        )
        training_sample.return_value.prepare.assert_called_once_with()

    def test_a_conditioning_sample_without_geometry_is_prepared_like_its_training_sample(self):
        train_runtime = MagicMock()
        train_runtime.describe.return_value = dict(self.RECORD)
        backends = {"control": {"id": "control"}, "train": {"id": "train", "dataset_runtime": train_runtime}}
        training_sample = self._prepare(None, backends, mappings=[("train", "control")])
        train_runtime.describe.assert_called_once_with(training_sample.return_value.training_sample_path.return_value)
        paired = training_sample.call_args_list[1]
        self.assertEqual(paired.kwargs["data_backend_id"], "train")
        self.assertEqual(paired.kwargs["image_metadata"], self.RECORD)
        training_sample.return_value.prepare_like.assert_called_once()
        training_sample.return_value.prepare.assert_not_called()

    def test_a_conditioning_sample_with_geometry_is_prepared_on_its_own(self):
        backends = {"control": {"id": "control"}, "train": {"id": "train", "dataset_runtime": MagicMock()}}
        training_sample = self._prepare(self.RECORD, backends, mappings=[("train", "control")])
        training_sample.return_value.prepare.assert_called_once_with()
        training_sample.return_value.prepare_like.assert_not_called()
        backends["train"]["dataset_runtime"].describe.assert_not_called()


class TestMetadataFilteredSample(unittest.TestCase):
    """A sample the VAE cache rejects leaves the bucket lists, and the main process is told to persist that."""

    @staticmethod
    def _cache(images):
        from simpletuner.helpers.caching.vae import VAECache

        cache = object.__new__(VAECache)
        cache.id = "filtered"
        cache.metadata_backend = SimpleNamespace(
            aspect_ratio_bucket_indices={"1.0": list(images)}, filtering_statistics=None, bucket_report=None
        )
        return cache

    def test_filtered_sample_leaves_the_bucket_lists(self):
        cache = self._cache(["image-0.jpg", "image-1.jpg", "image-2.jpg"])
        cache._handle_metadata_filtered_sample(filepath="image-1.jpg", bucket="1.0", reason="problematic")

        self.assertEqual(cache.metadata_backend.aspect_ratio_bucket_indices["1.0"], ["image-0.jpg", "image-2.jpg"])
        self.assertEqual(cache.metadata_backend.filtering_statistics["skipped"], {"problematic": 1})

    def test_filter_action_is_queued_for_the_main_process(self):
        cache = self._cache(["image-0.jpg", "image-1.jpg"])
        cache._handle_metadata_filtered_sample(filepath="image-1.jpg", bucket="1.0", reason="problematic")

        self.assertEqual(
            [(action["filepath"], action["reason"]) for action in cache._deferred_metadata_filter_actions],
            [("image-1.jpg", "problematic")],
        )


if __name__ == "__main__":
    unittest.main()


class TestVaeCacheClearsHuggingFaceLatents(unittest.TestCase):
    """A Hugging Face dataset keeps its latents beside it; clearing the VAE cache must remove them."""

    def test_clearing_the_cache_deletes_the_latents_the_dataset_backend_wrote(self):
        import tempfile
        from pathlib import Path

        from simpletuner.helpers.data_backend.huggingface import HuggingfaceDatasetsBackend

        with tempfile.TemporaryDirectory() as temp:
            backend = HuggingfaceDatasetsBackend.__new__(HuggingfaceDatasetsBackend)
            backend.id, backend.cache_dir = "hub", temp
            latents = Path(temp) / "vae"
            latents.mkdir()
            for name in ("a.pt", "b.pt"):
                backend.torch_save(torch.zeros(1), str(latents / name))
            cache = VAECache.__new__(VAECache)
            cache.id, cache.cache_dir, cache.cache_data_backend, cache.max_workers = "hub", str(latents), backend, 2
            cache.debug_log = MagicMock()
            with (
                patch.object(StateTracker, "get_vae_cache_files", return_value=["a.pt", "b.pt"]),
                patch.object(StateTracker, "set_vae_cache_files"),
            ):
                cache.clear_cache()
            self.assertEqual(sorted(path.name for path in latents.iterdir()), [])
            with self.assertRaises(FileNotFoundError):
                backend.delete(str(latents / "a.pt"))
            with self.assertRaises(NotImplementedError):
                backend.delete("0.jpg")
