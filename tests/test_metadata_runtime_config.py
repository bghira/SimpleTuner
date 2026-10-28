import json
import unittest
from unittest.mock import Mock, patch

from simpletuner.helpers.metadata.backends.caption import CaptionMetadataBackend
from simpletuner.helpers.metadata.backends.discovery import DiscoveryMetadataBackend
from simpletuner.helpers.metadata.backends.huggingface import HuggingfaceMetadataBackend
from simpletuner.helpers.metadata.backends.parquet import ParquetMetadataBackend
from simpletuner.helpers.metadata.backends.webshart import WebshartMetadataBackend
from simpletuner.helpers.training.state_tracker import StateTracker


class MetadataRuntimeConfigTests(unittest.TestCase):
    def test_reload_keeps_cached_config_separate_from_live_registry(self):
        cached_config = {
            "train_batch_size": 2,
            "repeats": 0,
            "start_epoch": 1,
            "start_step": 0,
            "end_epoch": None,
            "end_step": None,
        }
        live_config = {
            "train_batch_size": 4,
            "repeats": 3,
            "start_epoch": 2,
            "start_step": 100,
            "end_epoch": 5,
            "end_step": 1000,
        }
        indices = {"1.0": ["sample.jpg"]}
        payload = {"config": cached_config, "aspect_ratio_bucket_indices": indices, "ordered_ids": ["caption.txt"]}

        for backend_class in (
            DiscoveryMetadataBackend,
            HuggingfaceMetadataBackend,
            ParquetMetadataBackend,
            WebshartMetadataBackend,
            CaptionMetadataBackend,
        ):
            with self.subTest(backend=backend_class.__name__):
                backend = backend_class.__new__(backend_class)
                backend.id = "runtime-config"
                backend.cache_file = "buckets.json"
                backend.config = {}
                backend.data_backend = Mock()
                backend.data_backend.exists.return_value = True
                backend.data_backend.read.return_value = json.dumps(payload)
                backend.load_image_metadata = Mock()
                backend._load_caption_cache = Mock()
                backend._sync_image_files_with_buckets = Mock()
                registered_backend = {"config": live_config}

                with patch.dict(StateTracker.data_backends, {backend.id: registered_backend}, clear=True):
                    for _ in range(2):
                        backend.reload_cache()
                        self.assertIs(registered_backend["config"], live_config)
                        self.assertEqual(StateTracker.get_data_backend_config(backend.id), live_config)
                        self.assertEqual(backend.config, cached_config)
                    if backend_class is CaptionMetadataBackend:
                        self.assertEqual(backend._ordered_ids, ["caption.txt"])
                    else:
                        self.assertEqual(backend.aspect_ratio_bucket_indices, indices)

                    backend.config = {"resolution": 1024}
                    backend.reload_cache(set_config=False)
                    self.assertEqual(backend.config, {"resolution": 1024})
                    self.assertIs(registered_backend["config"], live_config)
