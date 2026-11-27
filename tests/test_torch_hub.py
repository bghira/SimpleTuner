import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from simpletuner.helpers.utils.torch_hub import load_torch_hub_model


class TestTorchHubCache(unittest.TestCase):
    def setUp(self):
        self.hub_dir = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch.object(torch.hub, "get_dir", return_value=str(self.hub_dir)))

    def cache(self, ref):
        checkout = self.hub_dir / f"example_models_{ref}"
        checkout.mkdir()
        (checkout / "hubconf.py").write_text(f"def encoder():\n    return {ref!r}\n")

    def test_cached_default_refs_work_offline(self):
        with patch.object(torch.hub, "urlopen", side_effect=AssertionError("unexpected network request")):
            self.cache("master")
            self.assertEqual(load_torch_hub_model("example/models", "encoder"), "master")
            self.cache("main")
            self.assertEqual(load_torch_hub_model("example/models", "encoder"), "main")

    def test_explicit_ref_takes_precedence(self):
        self.cache("main")
        self.cache("v1")
        with patch.object(torch.hub, "urlopen", side_effect=AssertionError("unexpected network request")):
            self.assertEqual(load_torch_hub_model("example/models:v1", "encoder"), "v1")

    def test_cold_cache_leaves_ref_discovery_to_torch_hub(self):
        self.cache("unrelated")
        (self.hub_dir / "example_models_main").mkdir()
        with patch.object(torch.hub, "load") as load:
            load_torch_hub_model("example/models", "encoder")
            load.assert_called_once_with("example/models", "encoder")

    def test_force_reload_preserves_branch_discovery_and_arguments(self):
        self.cache("main")
        with patch.object(torch.hub, "load") as load:
            load_torch_hub_model("example/models", "encoder", 3, pretrained=False, force_reload=True)
            load.assert_called_once_with("example/models", "encoder", 3, pretrained=False, force_reload=True)

    def test_local_source_passes_through(self):
        self.cache("v1")
        result = load_torch_hub_model(self.hub_dir / "example_models_v1", "encoder", source="local")
        self.assertEqual(result, "v1")
