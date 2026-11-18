import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from simpletuner.helpers.caching.vae import VAECache
from simpletuner.helpers.data_backend.local import LocalDataBackend
from simpletuner.helpers.training.state_tracker import StateTracker


class _PoolingVae(torch.nn.Module):
    """Stands in for a VAE: an 8x average pool, so a latent's shape shows the size that was encoded."""

    dtype = torch.float32
    device = torch.device("cpu")

    def encode(self, samples):
        return torch.nn.functional.avg_pool2d(samples, 8)


class _Model:
    MAXIMUM_CANVAS_SIZE = None

    def __init__(self):
        self.encoded_with = []

    def get_transforms(self, dataset_type="image"):
        return transforms.ToTensor()

    def pre_vae_encode_transform_sample(self, sample):
        return sample

    def encode_cache_batch(self, vae, samples, metadata_entries=None):
        self.encoded_with.extend(metadata_entries)
        return vae.encode(samples)

    def post_vae_encode_transform_sample(self, sample):
        return sample

    def scale_vae_latents_for_cache(self, latents, vae):
        return latents


class _Geometry:
    """A metadata backend's view of the dataset: the geometry recorded for each path."""

    image_metadata_loaded = True

    def __init__(self, geometry):
        self.geometry = geometry
        self.lookups = []

    def get_metadata_by_filepath(self, filepath):
        self.lookups.append(filepath)
        return self.geometry.get(filepath)


class OnDemandLatentsFollowTheBatchRecords(unittest.TestCase):
    """
    Latents encoded while training are resized as the batch's own records say, through the real
    VAECache, TrainingSample and image pipeline, without asking the metadata backend.
    """

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = self.directory.name
        accelerator = SimpleNamespace(device=torch.device("cpu"), is_local_main_process=True, is_main_process=True)
        self.paths = []
        pixels = np.random.default_rng(7).integers(0, 256, (3, 64, 96, 3), dtype=np.uint8)
        for index in range(3):
            path = os.path.join(root, f"sample-{index}.png")
            Image.fromarray(pixels[index]).save(path)
            self.paths.append(path)
        config = {
            "crop": False,
            "crop_style": "center",
            "crop_aspect": "square",
            "resolution": 64,
            "resolution_type": "pixel",
            "instance_data_dir": root,
        }
        self.model = _Model()
        data_backend = LocalDataBackend(accelerator=accelerator, id="images")
        self.geometry = _Geometry({path: self._record((96, 64)) for path in self.paths})
        self.cache = VAECache(
            id="images",
            model=self.model,
            vae=_PoolingVae(),
            accelerator=accelerator,
            metadata_backend=self.geometry,
            instance_data_dir=root,
            image_data_backend=data_backend,
            cache_dir=os.path.join(root, "vae"),
            vae_cache_ondemand=True,
            vae_cache_disable=True,
        )
        backend = {"id": "images", "data_backend": data_backend, "config": config, "dataset_type": "image"}
        args = SimpleNamespace(model_family="sdxl", aspect_bucket_alignment=8, aspect_bucket_rounding=2)
        for target, name, value in [
            (StateTracker, "data_backends", {"images": backend}),
            (StateTracker, "args", args),
            (StateTracker, "model", self.model),
        ]:
            patcher = patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def _record(size):
        return {
            "original_size": (96, 64),
            "intermediary_size": size,
            "target_size": size,
            "crop_coordinates": (0, 0),
            "aspect_ratio": 1.5,
        }

    def _encode(self, records=None):
        return self.cache.encode_images([None] * len(self.paths), list(self.paths), records=records)

    def test_a_batch_is_encoded_at_the_size_its_records_hold(self):
        # The metadata backend holds 96x64 for these samples; the batch says 48x32.
        records = {path: self._record((48, 32)) for path in self.paths}
        latents = self._encode(records)
        self.assertEqual([tuple(latent.shape) for latent in latents], [(3, 4, 6)] * 3)
        self.assertEqual(self.geometry.lookups, [])
        encoded_with = {entry["filepath"]: entry["metadata"] for entry in self.model.encoded_with}
        self.assertEqual(encoded_with, records)

    def test_a_caller_with_only_paths_gets_the_geometry_the_metadata_backend_holds(self):
        from_lookup = self._encode()
        self.assertEqual([tuple(latent.shape) for latent in from_lookup], [(3, 8, 12)] * 3)
        self.assertEqual(sorted(set(self.geometry.lookups)), sorted(self.paths))
        from_records = self._encode(dict(self.geometry.geometry))
        self.assertFalse(torch.equal(from_lookup[0], from_lookup[1]))
        for looked_up, recorded in zip(from_lookup, from_records):
            torch.testing.assert_close(looked_up, recorded)

    def test_latents_come_back_in_the_order_the_samples_were_asked_for(self):
        alone = [self.cache.encode_images([None], [path])[0] for path in self.paths]
        self.assertFalse(any(torch.equal(alone[0], other) for other in alone[1:]))
        with patch.object(self.model, "encode_cache_batch", wraps=self.model.encode_cache_batch) as encode:
            together = self._encode()
        for latent, expected in zip(together, alone):
            torch.testing.assert_close(latent, expected)
        # The samples share a size, so they are encoded as one batch.
        self.assertEqual([call.args[1].shape[0] for call in encode.call_args_list], [3])


if __name__ == "__main__":
    unittest.main()
