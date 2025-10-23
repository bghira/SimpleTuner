import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from simpletuner.simpletuner_sdk.server.services import fsdp_service
from simpletuner.simpletuner_sdk.server.services.fsdp_service import FSDP_SERVICE


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.block = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(4, 4),
        )
        self._no_split_modules = ["Linear"]


class _DummyModelClass(_DummyModel):
    @classmethod
    def load_config(cls, path, **kwargs):
        return {"path": path, **kwargs}

    @classmethod
    def from_config(cls, config):
        # Return a fresh instance to ensure module enumeration works as expected
        return cls()


class _DummyModelFamily:
    MODEL_CLASS = _DummyModelClass
    MODEL_SUBFOLDER = None
    DEFAULT_MODEL_FLAVOUR = "default"
    HUGGINGFACE_PATHS = {"default": "dummy-path"}


class TestFSDPService(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.prev_cache_path = FSDP_SERVICE.cache_path
        self.prev_cache = {"entries": dict(FSDP_SERVICE._cache.get("entries", {}))}
        FSDP_SERVICE.cache_path = Path(self.tempdir.name) / "fsdp_cache.json"
        FSDP_SERVICE._cache = {"entries": {}}

    def tearDown(self):
        FSDP_SERVICE.cache_path = self.prev_cache_path
        FSDP_SERVICE._cache = {"entries": dict(self.prev_cache.get("entries", {}))}
        self.tempdir.cleanup()

    def test_detect_block_classes_uses_cache(self):
        with patch.dict(fsdp_service.model_families, {"dummy": _DummyModelFamily}, clear=False):
            result = FSDP_SERVICE.detect_block_classes("dummy", pretrained_model="dummy-path")
            self.assertFalse(result["cached"])
            classes = {entry["class_name"] for entry in result["transformer_classes"]}
            self.assertIn("Linear", classes)
            self.assertGreater(result["total_parameter_count"], 0)

            # Second call should hit cache
            cached = FSDP_SERVICE.detect_block_classes("dummy", pretrained_model="dummy-path")
            self.assertTrue(cached["cached"])
            self.assertEqual(cached["transformer_classes"], result["transformer_classes"])


if __name__ == "__main__":
    unittest.main()
