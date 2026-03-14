import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.sd3.model import SD3


class SD3ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = SD3.__new__(SD3)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(
            base_weight_dtype=torch.float32,
            weight_dtype=torch.float32,
            twinflow_enabled=False,
            tread_config=None,
            crepa_self_flow_mask_ratio=0.0,
        )
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=2)
        self.model.sample_flow_sigmas = MagicMock(return_value=(torch.tensor([750.0]), torch.tensor([750.0])))

    def test_model_supports_crepa_self_flow(self):
        self.assertTrue(self.model.supports_crepa_self_flow())

    def test_model_predict_uses_crepa_capture_block_override(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            kwargs["hidden_states_buffer"]["layer_9"] = torch.full((1, 4, 8), 9.0)
            return (torch.randn(1, 16, 4, 4),)

        self.model.model = MagicMock(
            side_effect=_forward,
            config=SimpleNamespace(patch_size=2),
        )
        self.model.unwrap_model = lambda model=None: model or self.model.model

        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 4, 4),
            "timesteps": torch.tensor([250.0]),
            "encoder_hidden_states": torch.randn(1, 3, 64),
            "add_text_embeds": torch.randn(1, 32),
            "crepa_capture_block_index": 9,
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 16, 4, 4))
        self.assertTrue(torch.equal(result["crepa_hidden_states"], torch.full((1, 4, 8), 9.0)))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([250.0], dtype=torch.float32)))

    def test_model_predict_accepts_tokenwise_timesteps(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            return (torch.randn(1, 16, 4, 4),)

        self.model.model = MagicMock(
            side_effect=_forward,
            config=SimpleNamespace(patch_size=2),
        )
        self.model.unwrap_model = lambda model=None: model or self.model.model

        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 4, 4),
            "timesteps": torch.tensor([[100.0, 900.0, 500.0, 700.0]], dtype=torch.float32),
            "encoder_hidden_states": torch.randn(1, 3, 64),
            "add_text_embeds": torch.randn(1, 32),
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 16, 4, 4))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], prepared_batch["timesteps"]))

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        self.model.model = MagicMock(config=SimpleNamespace(patch_size=2))
        self.model.unwrap_model = lambda model=None: model or self.model.model
        batch = {
            "latents": torch.randn(1, 16, 4, 4),
            "input_noise": torch.randn(1, 16, 4, 4),
            "sigmas": torch.tensor([250.0]),
            "timesteps": torch.tensor([250.0]),
        }

        result = self.model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 4))
        self.assertEqual(result["crepa_self_flow_mask"].shape, (1, 2, 2))
        self.assertEqual(result["crepa_teacher_timesteps"].shape, (1,))


if __name__ == "__main__":
    unittest.main()
