import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.pixart.model import PixartSigma


class PixArtModelTests(unittest.TestCase):
    def setUp(self):
        self.model = PixartSigma.__new__(PixartSigma)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(base_weight_dtype=torch.float32)
        self.model.LATENT_CHANNEL_COUNT = 4
        self.model.NAME = "PixArt"
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=2)
        self.model.unwrap_model = MagicMock(side_effect=lambda model=None, **kwargs: model)

    def test_model_predict_uses_crepa_capture_block_override(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(*args, **kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            kwargs["hidden_states_buffer"]["layer_7"] = torch.full((1, 4, 8), 7.0)
            return (torch.randn(1, 8, 4, 4),)

        self.model.model = MagicMock(side_effect=_forward)
        self.model.model.config = SimpleNamespace(patch_size=2)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 4, 4, 4),
            "timesteps": torch.tensor([400], dtype=torch.int64),
            "encoder_hidden_states": torch.randn(1, 4, 16),
            "encoder_attention_mask": torch.ones(1, 4),
            "crepa_capture_block_index": 7,
            "resolution": torch.tensor([[4.0, 4.0]], dtype=torch.float32),
            "aspect_ratio": torch.tensor([[1.0]], dtype=torch.float32),
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 4, 4, 4))
        self.assertTrue(torch.equal(result["crepa_hidden_states"], torch.full((1, 4, 8), 7.0)))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([400], dtype=torch.int64)))
        self.assertTrue(torch.equal(transformer_kwargs["added_cond_kwargs"]["resolution"], prepared_batch["resolution"]))

    def test_model_predict_accepts_tokenwise_timesteps(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(*args, **kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            return (torch.randn(1, 8, 4, 4),)

        self.model.model = MagicMock(side_effect=_forward)
        self.model.model.config = SimpleNamespace(patch_size=2)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 4, 4, 4),
            "timesteps": torch.tensor([[100, 900, 200, 800]], dtype=torch.int64),
            "encoder_hidden_states": torch.randn(1, 4, 16),
            "encoder_attention_mask": torch.ones(1, 4),
            "resolution": torch.tensor([[4.0, 4.0]], dtype=torch.float32),
            "aspect_ratio": torch.tensor([[1.0]], dtype=torch.float32),
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 4, 4, 4))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], prepared_batch["timesteps"]))

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

    def test_controlnet_predict_passes_added_cond_kwargs(self):
        self.model.controlnet = MagicMock(return_value=(torch.randn(1, 4, 4, 4),))
        transformer = SimpleNamespace(config=SimpleNamespace(patch_size=2))
        self.model.model = transformer
        self.model.unwrap_model = MagicMock(return_value=transformer)
        self.model.config.weight_dtype = torch.float32
        prepared_batch = {
            "noisy_latents": torch.randn(1, 4, 4, 4),
            "timesteps": torch.tensor([400], dtype=torch.int64),
            "encoder_hidden_states": torch.randn(1, 4, 16),
            "encoder_attention_mask": torch.ones(1, 4),
            "conditioning_latents": torch.randn(1, 4, 4, 4),
            "resolution": torch.tensor([[4.0, 4.0]], dtype=torch.float32),
            "aspect_ratio": torch.tensor([[1.0]], dtype=torch.float32),
        }

        result = self.model.controlnet_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 4, 4, 4))
        kwargs = self.model.controlnet.call_args.kwargs
        self.assertTrue(torch.equal(kwargs["timestep"], torch.tensor([400], dtype=torch.int64)))
        self.assertTrue(torch.equal(kwargs["added_cond_kwargs"]["resolution"], prepared_batch["resolution"]))


if __name__ == "__main__":
    unittest.main()
