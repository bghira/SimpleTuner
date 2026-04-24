import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.auraflow.model import Auraflow


class AuraflowModelTests(unittest.TestCase):
    def setUp(self):
        self.model = Auraflow.__new__(Auraflow)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(base_weight_dtype=torch.float32, twinflow_enabled=False)
        self.model.LATENT_CHANNEL_COUNT = 4
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=2)
        self.model.unwrap_model = MagicMock(return_value=SimpleNamespace(config=SimpleNamespace(patch_size=2)))

    def test_model_predict_uses_crepa_capture_block_override(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(*args, **kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            kwargs["hidden_states_buffer"]["layer_6"] = torch.full((1, 4, 8), 6.0)
            return SimpleNamespace(sample=torch.randn(1, 4, 4, 4))

        self.model.model = MagicMock(side_effect=_forward)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 4, 4, 4),
            "timesteps": torch.tensor([250.0]),
            "encoder_hidden_states": torch.randn(1, 4, 16),
            "crepa_capture_block_index": 6,
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 4, 4, 4))
        self.assertTrue(torch.equal(result["crepa_hidden_states"], torch.full((1, 4, 8), 6.0)))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([0.25], dtype=torch.float32)))

    def test_model_predict_accepts_tokenwise_timesteps(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(*args, **kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            return SimpleNamespace(sample=torch.randn(1, 4, 4, 4))

        self.model.model = MagicMock(side_effect=_forward)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 4, 4, 4),
            "timesteps": torch.tensor([[100.0, 900.0, 250.0, 750.0]], dtype=torch.float32),
            "encoder_hidden_states": torch.randn(1, 4, 16),
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 4, 4, 4))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], prepared_batch["timesteps"] / 1000.0))

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        transformer = SimpleNamespace(config=SimpleNamespace(patch_size=2))
        self.model.model = transformer
        self.model.unwrap_model = MagicMock(return_value=transformer)
        self.model.sample_flow_sigmas = MagicMock(
            return_value=(
                torch.tensor([0.9], dtype=torch.float32),
                torch.tensor([900.0], dtype=torch.float32),
            )
        )
        self.model.config.crepa_self_flow_mask_ratio = 1.0

        batch = {
            "latents": torch.zeros(1, 4, 4, 4, dtype=torch.float32),
            "input_noise": torch.ones(1, 4, 4, 4, dtype=torch.float32),
            "sigmas": torch.tensor([0.1], dtype=torch.float32),
            "timesteps": torch.tensor([100.0], dtype=torch.float32),
        }

        result = self.model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 4))
        self.assertTrue(torch.equal(result["crepa_teacher_timesteps"], torch.tensor([100.0], dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()
