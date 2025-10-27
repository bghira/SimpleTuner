import itertools
import sys
import unittest
from types import SimpleNamespace

import torch

if not hasattr(torch, "cuda"):
    torch.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]

class _ContextManager:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False

    def __call__(self, func):
        return func


if not hasattr(torch, "no_grad"):
    torch.no_grad = lambda *args, **kwargs: _ContextManager()  # type: ignore[attr-defined]

if not hasattr(torch, "autocast"):
    torch.autocast = lambda *args, **kwargs: _ContextManager()  # type: ignore[attr-defined]

if not hasattr(torch, "float32"):
    torch.float32 = "float32"  # type: ignore[attr-defined]

if not hasattr(torch, "long"):
    torch.long = "int64"  # type: ignore[attr-defined]

if "torch.distributed" not in sys.modules:
    dist_stub = SimpleNamespace(
        is_available=lambda: False,
        is_initialized=lambda: False,
        get_rank=lambda: 0,
    )
    sys.modules["torch.distributed"] = dist_stub
    torch.distributed = dist_stub  # type: ignore[attr-defined]

if "torch.nn" not in sys.modules:
    torch_nn_functional = SimpleNamespace()
    torch_nn = SimpleNamespace(functional=torch_nn_functional)
    sys.modules["torch.nn.functional"] = torch_nn_functional
    sys.modules["torch.nn"] = torch_nn
    torch.nn = torch_nn  # type: ignore[attr-defined]
    torch.nn.functional = torch_nn_functional  # type: ignore[attr-defined]
elif not hasattr(torch, "nn"):
    torch_nn_functional = SimpleNamespace()
    torch_nn = SimpleNamespace(functional=torch_nn_functional)
    sys.modules["torch.nn.functional"] = torch_nn_functional
    sys.modules["torch.nn"] = torch_nn
    torch.nn = torch_nn  # type: ignore[attr-defined]
    torch.nn.functional = torch_nn_functional  # type: ignore[attr-defined]

from simpletuner.helpers.distillation.self_forcing.distiller import SelfForcingDistillation
from simpletuner.helpers.models.common import PredictionTypes


class _StubCache:
    def __init__(self, payloads):
        self.payloads = payloads
        self.distillation_type = "self_forcing"
        self._index = 0

    def load_next_pair(self):
        if not self.payloads:
            return None, None
        payload = self.payloads[self._index % len(self.payloads)]
        path = f"/tmp/cache_artifact_{self._index}.pt"
        self._index += 1
        return payload, path


class _StubModel:
    def __init__(self):
        self.config = SimpleNamespace(weight_dtype=torch.float32)
        self.accelerator = SimpleNamespace(device="cpu")

    def encode_text_batch(self, text_batch, is_negative_prompt=False):
        batch_size = len(text_batch)
        prompt_embeds = torch.zeros((batch_size, 2), dtype=torch.float32)
        attention_masks = torch.ones((batch_size, 2), dtype=torch.float32)
        return {
            "prompt_embeds": prompt_embeds,
            "attention_masks": attention_masks,
        }

    def prepare_batch(self, batch, state):
        result = dict(batch)
        latent_batch = batch["latent_batch"].to(self.config.weight_dtype)
        result["latents"] = latent_batch.clone()
        result["noise"] = torch.zeros_like(latent_batch)
        result["input_noise"] = torch.zeros_like(latent_batch)
        result["timesteps"] = torch.zeros(latent_batch.shape[0], dtype=torch.long)
        return result


class SelfForcingDistillerTests(unittest.TestCase):
    def setUp(self):
        self.teacher = SimpleNamespace(
            PREDICTION_TYPE=PredictionTypes.EPSILON,
            config=SimpleNamespace(lora_type="none"),
            accelerator=SimpleNamespace(),
        )
        self.model = _StubModel()
        self.distiller = SelfForcingDistillation(self.teacher, noise_scheduler=None)

    def test_consumes_caption_batches(self):
        self.assertTrue(self.distiller.consumes_caption_batches())

    def test_prepare_caption_batch_uses_cache_entries(self):
        payload = {
            "latents": torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            "metadata": {"source": "stub"},
            "noise": torch.zeros((1, 4, 8, 8), dtype=torch.float32),
            "timesteps": torch.tensor([500], dtype=torch.long),
        }
        cache = _StubCache([payload])
        self.distiller._distillation_caches = [cache]
        self.distiller._cache_cycle = itertools.cycle(self.distiller._distillation_caches)

        caption_batch = {
            "captions": ["a photo of a cat"],
            "records": [{"metadata_id": "meta-1"}],
            "data_backend_id": "captions",
        }
        prepared = self.distiller.prepare_caption_batch(caption_batch, self.model, state={})

        self.assertIn("latents", prepared)
        self.assertEqual(tuple(prepared["latents"].shape), (1, 4, 8, 8))
        self.assertEqual(prepared["captions"], ["a photo of a cat"])
        self.assertEqual(prepared["records"], [{"metadata_id": "meta-1"}])
        self.assertIn("distillation_metadata", prepared)
        self.assertEqual(prepared["distillation_metadata"][0]["source"], "stub")
        self.assertIn("artifact_path", prepared["distillation_metadata"][0])


if __name__ == "__main__":
    unittest.main()
