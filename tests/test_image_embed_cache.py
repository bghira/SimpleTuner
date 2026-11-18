import os
import tempfile
import unittest
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from PIL import Image

from simpletuner.helpers.caching.image_embed import ImageEmbedCache
from simpletuner.helpers.data_backend.base import BaseDataBackend
from simpletuner.helpers.training.state_tracker import StateTracker


class DummyImageBackend:
    def __init__(self, root: str) -> None:
        self.root = root
        self.type = "local"

    def list_files(self, instance_data_dir: str, file_extensions=None):
        results = []
        extensions = tuple(ext.lower() for ext in (file_extensions or []))
        for dirpath, _dirnames, filenames in os.walk(instance_data_dir):
            filtered = []
            for filename in filenames:
                if not extensions or filename.lower().endswith(extensions):
                    filtered.append(os.path.join(dirpath, filename))
            if filtered:
                results.append((dirpath, [], filtered))
        return results

    def read_image(self, filepath: str):
        with Image.open(filepath) as image:
            return image.copy()


class DummyCacheBackend:
    bulk_exists = BaseDataBackend.bulk_exists

    def __init__(self, root: str) -> None:
        self.root = root
        self.type = "local"

    def create_directory(self, path: str) -> None:
        if path:
            os.makedirs(path, exist_ok=True)

    def list_files(self, instance_data_dir: str, file_extensions=None):
        search_root = instance_data_dir or self.root
        results = []
        extensions = tuple(ext.lower() for ext in (file_extensions or []))
        for dirpath, _dirnames, filenames in os.walk(search_root):
            filtered = []
            for filename in filenames:
                if not extensions or filename.lower().endswith(extensions):
                    filtered.append(os.path.join(dirpath, filename))
            if filtered:
                results.append((dirpath, [], filtered))
        return results

    def exists(self, path: str) -> bool:
        return os.path.exists(path)

    def torch_save(self, tensor: torch.Tensor, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(tensor, path)

    def torch_load(self, path: str) -> torch.Tensor:
        return torch.load(path)


class DummyImageProcessor:
    def __call__(self, images, return_tensors="pt"):
        tensors = []
        for image in images:
            arr = np.asarray(image).astype("float32") / 255.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1)
            tensors.append(tensor)
        return {"pixel_values": torch.stack(tensors)}


class DummyImageEncoder:
    def __init__(self):
        self._device = torch.device("cpu")

    def to(self, device):
        self._device = device
        return self

    def eval(self):
        return self

    def __call__(self, pixel_values, output_hidden_states=True):
        batch_size = pixel_values.shape[0]
        embedding = torch.mean(pixel_values, dim=(2, 3)).reshape(batch_size, -1)
        hidden_states = [torch.zeros_like(embedding), embedding]
        return SimpleNamespace(hidden_states=hidden_states)


class DummyModel:
    def __init__(self, pipeline):
        self._pipeline = pipeline

    def get_pipeline(self):
        return self._pipeline

    def get_conditioning_image_embedder(self):
        pipeline = self.get_pipeline()

        class _Embedder:
            def __init__(self, pipe):
                self._pipe = pipe

            def encode(self, images):
                processed = self._pipe.image_processor(images, return_tensors="pt")
                pixel_values = processed.get("pixel_values")
                outputs = self._pipe.image_encoder(pixel_values, output_hidden_states=True)
                hidden_states = getattr(outputs, "hidden_states", None)
                if not hidden_states:
                    raise ValueError("Dummy encoder did not return hidden states.")
                return hidden_states[-1]

        return _Embedder(pipeline)


class TestImageEmbedCache(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.images_dir = os.path.join(self.tempdir.name, "images")
        self.cache_dir = os.path.join(self.tempdir.name, "cache")
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        self._old_args = getattr(StateTracker, "args", None)
        StateTracker.set_args(SimpleNamespace(output_dir=self.tempdir.name))

        self._old_data_backends = StateTracker.data_backends.copy()

    def tearDown(self):
        if self._old_args is not None:
            StateTracker.set_args(self._old_args)
        else:
            StateTracker.set_args(None)
        StateTracker.data_backends = self._old_data_backends
        self.tempdir.cleanup()

    def test_generate_embed_filename_with_hash(self):
        image_backend = DummyImageBackend(self.images_dir)
        cache_backend = DummyCacheBackend(self.cache_dir)
        model = MagicMock()
        accelerator = SimpleNamespace(device=torch.device("cpu"))

        cache = ImageEmbedCache(
            id="dataset",
            dataset_type="image",
            model=model,
            accelerator=accelerator,
            metadata_backend=None,
            image_data_backend=image_backend,
            cache_data_backend=cache_backend,
            instance_data_dir="/data",
            cache_dir="cache",
            hash_filenames=True,
        )

        full_path, _ = cache.generate_embed_filename("/data/sample.png")
        expected_filename = sha256("sample".encode()).hexdigest() + ".pt"
        self.assertEqual(os.path.basename(full_path), expected_filename)
        self.assertEqual(os.path.dirname(full_path), os.path.abspath("cache"))

    def test_native_preparation_and_demand_reads_do_not_retain_inventory(self):
        from simpletuner.helpers.data_backend.webshart import WebshartDataBackend

        refs = ("webshart://0/0/name.jpg", "webshart://1/0/name.jpg")
        source = MagicMock()
        source.is_sample_id = WebshartDataBackend.is_sample_id
        source.normalize_sample_id = WebshartDataBackend.normalize_sample_id
        source.read_image.side_effect = lambda ref: Image.new("RGB", (8, 8), "red" if ref == refs[0] else "blue")
        store = DummyCacheBackend(self.cache_dir)
        store.list_files = MagicMock(side_effect=AssertionError("artifact inventory requested"))
        store.bulk_exists = MagicMock(wraps=store.bulk_exists)
        dataset = MagicMock()
        dataset.iter_preparation_batches.return_value = [("1.0", refs)]
        model = DummyModel(SimpleNamespace(image_processor=DummyImageProcessor(), image_encoder=DummyImageEncoder()))
        cache = ImageEmbedCache(
            "source",
            "image",
            model,
            SimpleNamespace(device=torch.device("cpu")),
            SimpleNamespace(
                native_policy="shard_local", image_artifact_key=lambda ref, kind, model: sha256(ref.encode()).hexdigest()
            ),
            source,
            store,
            cache_dir=self.cache_dir,
            embed_batch_size=2,
        )
        cache.prepare_dataset(dataset)
        self.assertNotEqual(cache.generate_embed_filename(refs[0]), cache.generate_embed_filename(refs[1]))
        self.assertEqual(source.read_image.call_count, 2)
        cache.prepare_dataset(dataset)
        cache.retrieve_from_cache(refs[1])
        self.assertEqual(source.read_image.call_count, 2)
        store.list_files.assert_not_called()
        self.assertEqual(store.bulk_exists.call_count, 2)

    def test_process_and_retrieve_embeddings(self):
        image_path = os.path.join(self.images_dir, "frame.png")
        Image.new("RGB", (8, 8), color=(120, 60, 30)).save(image_path)

        image_backend = DummyImageBackend(self.images_dir)
        cache_backend = DummyCacheBackend(self.cache_dir)
        pipeline = SimpleNamespace(image_processor=DummyImageProcessor(), image_encoder=DummyImageEncoder())
        model = DummyModel(pipeline)
        accelerator = SimpleNamespace(device=torch.device("cpu"))

        cache = ImageEmbedCache(
            id="dataset1",
            dataset_type="image",
            model=model,
            accelerator=accelerator,
            metadata_backend=MagicMock(),
            image_data_backend=image_backend,
            cache_data_backend=cache_backend,
            instance_data_dir=self.images_dir,
            cache_dir=self.cache_dir,
        )

        cache_backend.list_files = MagicMock(side_effect=AssertionError("artifact inventory"))
        dataset = MagicMock()
        dataset.describe.return_value = SimpleNamespace(accepted_samples=1)
        dataset.iter_preparation_batches.return_value = [("square", (image_path,))]
        self.assertEqual(cache.prepare_dataset(dataset), 1)

        cache_file, _ = cache.generate_embed_filename(image_path)
        self.assertTrue(cache_backend.exists(cache_file))
        self.assertIsInstance(cache.retrieve_from_cache(image_path), torch.Tensor)
        cache_backend.list_files.assert_not_called()


if __name__ == "__main__":
    unittest.main()
