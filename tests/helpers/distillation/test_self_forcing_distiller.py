import itertools
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

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

if not hasattr(torch, "nn"):
    torch_nn_functional = SimpleNamespace()
    torch_nn = SimpleNamespace(functional=torch_nn_functional)
    sys.modules["torch.nn.functional"] = torch_nn_functional
    sys.modules["torch.nn"] = torch_nn
    torch.nn = torch_nn  # type: ignore[attr-defined]
    torch.nn.functional = torch_nn_functional  # type: ignore[attr-defined]

from simpletuner.helpers.distillation.self_forcing.distiller import SelfForcingDistillation
from simpletuner.helpers.models.common import PredictionTypes


class _StubScheduler:
    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.sigmas = torch.ones(1000)

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return clean + noise

    def convert_x0_to_noise(self, x0: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return xt - x0


class _FakeTransformer(torch.nn.Module):
    def __init__(self, hidden_size: int = 1, **_kwargs):
        super().__init__()
        self.config = {"hidden_size": hidden_size}

    def forward(self, latents, timesteps, encoder_hidden_states, return_dict=False):
        return (torch.zeros_like(latents),)


class _StubTeacher:
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING

    def __init__(self, scheduler):
        self.config = SimpleNamespace(lora_type="none", weight_dtype=torch.float32)
        self.accelerator = SimpleNamespace(device=torch.device("cpu"), is_local_main_process=True)
        self.noise_schedule = scheduler
        self._component = _FakeTransformer(hidden_size=1)

    def get_trained_component(self):
        return self._component


class _StubModel:
    def __init__(self, scheduler):
        self.config = SimpleNamespace(
            weight_dtype=torch.float32,
            flow_matching=False,
            offset_noise=False,
            noise_offset=0.0,
            noise_offset_probability=0.0,
            input_perturbation=0.0,
            flow_use_beta_schedule=False,
            flow_use_uniform_schedule=False,
            flow_sigmoid_scale=0.0,
            disable_segmented_timestep_sampling=True,
        )
        self.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.noise_schedule = scheduler

    def encode_text_batch(self, text_batch, is_negative_prompt=False, prompt_contexts=None):
        batch_size = len(text_batch)
        prompt_embeds = torch.zeros((batch_size, 2, 4), dtype=torch.float32)
        attention_masks = torch.ones((batch_size, 2), dtype=torch.float32)
        return {
            "prompt_embeds": prompt_embeds,
            "attention_masks": attention_masks,
            "pooled_prompt_embeds": None,
            "batch_time_ids": None,
        }

    def prepare_batch(self, batch, state):
        result = dict(batch)
        latent_batch = batch["latent_batch"].to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        result["latents"] = latent_batch.clone()
        result["prompt_embeds"] = batch.get("prompt_embeds")
        result["encoder_hidden_states"] = result["prompt_embeds"]
        result.setdefault("noise", torch.zeros_like(latent_batch))
        result.setdefault("input_noise", torch.zeros_like(latent_batch))
        timesteps = result.get("timesteps")
        if timesteps is None:
            timesteps = torch.zeros(latent_batch.shape[0], dtype=torch.long)
        result["timesteps"] = timesteps
        return result


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


class SelfForcingDistillerTests(unittest.TestCase):
    def setUp(self):
        self._original_adamw = getattr(torch.optim, "AdamW", None)
        torch.optim.AdamW = MagicMock(return_value=MagicMock())

        self.scheduler = _StubScheduler()
        self.teacher = _StubTeacher(self.scheduler)
        self.model = _StubModel(self.scheduler)
        self.distiller = SelfForcingDistillation(self.teacher, noise_scheduler=self.scheduler)

    def tearDown(self):
        torch.optim.AdamW = self._original_adamw

    def test_consumes_caption_batches(self):
        self.assertTrue(self.distiller.consumes_caption_batches())

    def test_prepare_caption_batch_uses_cache_entries(self):
        latents = torch.full((1, 4, 8, 8), 0.5, dtype=torch.float32)
        noise = torch.full((1, 4, 8, 8), 0.25, dtype=torch.float32)
        payload = {
            "latents": latents,
            "metadata": {"source": "stub"},
            "noise": noise,
            "input_noise": noise.clone(),
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
        torch.testing.assert_close(prepared["noise"], noise)
        torch.testing.assert_close(prepared["input_noise"], noise)
        torch.testing.assert_close(prepared["timesteps"], torch.tensor([500]))
        torch.testing.assert_close(prepared["noisy_latents"], prepared["latents"] + prepared["input_noise"])
        torch.testing.assert_close(prepared["clean_latents"], prepared["latents"])

        self.assertEqual(prepared["captions"], ["a photo of a cat"])
        self.assertEqual(prepared["records"], [{"metadata_id": "meta-1"}])
        self.assertIn("distillation_metadata", prepared)
        self.assertEqual(prepared["distillation_metadata"][0]["source"], "stub")
        self.assertIn("artifact_path", prepared["distillation_metadata"][0])
        self.assertIn("distillation_cache_entries", prepared)
        self.assertEqual(len(prepared["distillation_cache_entries"]), 1)

    def test_prepare_batch_respects_cache_payload(self):
        latents = torch.zeros((1, 4, 8, 8), dtype=torch.float32)
        noise = torch.ones((1, 4, 8, 8), dtype=torch.float32) * 0.1
        payload = {
            "latents": latents,
            "metadata": {},
            "noise": noise,
            "input_noise": noise.clone(),
            "timesteps": torch.tensor([250], dtype=torch.long),
        }
        cache = _StubCache([payload])
        self.distiller._distillation_caches = [cache]
        self.distiller._cache_cycle = itertools.cycle(self.distiller._distillation_caches)

        caption_batch = {
            "captions": ["video frame"],
            "records": [],
            "data_backend_id": "captions",
        }
        prepared = self.distiller.prepare_caption_batch(caption_batch, self.model, state={})
        processed = self.distiller.prepare_batch(prepared, self.model, state={})

        torch.testing.assert_close(processed["timesteps"], torch.tensor([250], device=processed["latents"].device))
        torch.testing.assert_close(processed["noisy_latents"], processed["latents"] + processed["input_noise"])


if __name__ == "__main__":
    unittest.main()
