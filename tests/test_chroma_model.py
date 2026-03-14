import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.chroma.model import Chroma


class ChromaModelTests(unittest.TestCase):
    def setUp(self):
        self.model = Chroma.__new__(Chroma)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(
            base_weight_dtype=torch.float32,
            weight_dtype=torch.float32,
            twinflow_enabled=False,
            tread_config=None,
            crepa_self_flow_mask_ratio=0.5,
            flux_fast_schedule=False,
            flow_use_beta_schedule=False,
            flow_use_uniform_schedule=False,
        )
        self.model.noise_schedule = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=2)
        self.model.sample_flow_sigmas = MagicMock(
            return_value=(torch.tensor([0.75], dtype=torch.float32), torch.tensor([750.0], dtype=torch.float32))
        )

    def test_model_predict_uses_crepa_capture_block_override(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            kwargs["hidden_states_buffer"]["layer_9"] = torch.full((1, 4, 8), 9.0)
            return (torch.randn(1, 4, 64),)

        self.model.model = MagicMock(side_effect=_forward)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 4, 4),
            "latents": torch.randn(1, 16, 4, 4),
            "timesteps": torch.tensor([500.0], dtype=torch.float32),
            "prompt_embeds": torch.randn(1, 2, 16),
            "encoder_attention_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            "crepa_capture_block_index": 9,
        }

        result = self.model.model_predict(prepared_batch)

        self.assertTrue(torch.equal(result["crepa_hidden_states"], torch.full((1, 4, 8), 9.0)))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([0.5], dtype=torch.float32)))

    def test_model_predict_accepts_tokenwise_timesteps(self):
        self.model.model = MagicMock(return_value=(torch.randn(1, 4, 64),))

        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 4, 4),
            "latents": torch.randn(1, 16, 4, 4),
            "timesteps": torch.tensor([[100.0, 900.0, 250.0, 750.0]], dtype=torch.float32),
            "prompt_embeds": torch.randn(1, 2, 16),
            "encoder_attention_mask": torch.tensor([[1, 1]], dtype=torch.bool),
        }

        self.model.model_predict(prepared_batch)

        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(
            torch.equal(
                transformer_kwargs["timestep"],
                torch.tensor([[0.1, 0.9, 0.25, 0.75]], dtype=torch.float32),
            )
        )

    def test_prepare_crepa_self_flow_batch_creates_packed_token_timesteps(self):
        batch = {
            "latents": torch.zeros(1, 16, 4, 4),
            "input_noise": torch.ones(1, 16, 4, 4),
            "sigmas": torch.tensor([0.25], dtype=torch.float32),
            "timesteps": torch.tensor([250.0], dtype=torch.float32),
        }

        updated = self.model._prepare_crepa_self_flow_batch(batch, state={"global_step": 0})

        self.assertEqual(updated["timesteps"].shape, torch.Size([1, 4]))
        self.assertEqual(updated["crepa_self_flow_mask"].shape, torch.Size([1, 2, 2]))
        self.assertEqual(updated["crepa_teacher_timesteps"].shape, torch.Size([1]))


if __name__ == "__main__":
    unittest.main()
