import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.longcat_image.model import LongCatImage


class LongCatImageModelTests(unittest.TestCase):
    def setUp(self):
        self.model = LongCatImage.__new__(LongCatImage)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(weight_dtype=torch.float32, base_weight_dtype=torch.float32, model_flavour=None)
        self.model.noise_schedule = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        self.model._is_edit_flavour = lambda: False

    def test_model_supports_crepa_self_flow(self):
        self.assertTrue(self.model.supports_crepa_self_flow())

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(
            weight_dtype=torch.float32,
            base_weight_dtype=torch.float32,
            model_flavour=None,
            crepa_self_flow_mask_ratio=0.5,
        )
        self.model.sample_flow_sigmas = MagicMock(
            return_value=(torch.tensor([0.8], dtype=torch.float32), torch.tensor([800.0], dtype=torch.float32))
        )

        batch = {
            "latents": torch.zeros(1, 16, 4, 4, dtype=torch.float32),
            "input_noise": torch.ones(1, 16, 4, 4, dtype=torch.float32),
            "sigmas": torch.tensor([0.2], dtype=torch.float32),
            "timesteps": torch.tensor([200.0], dtype=torch.float32),
        }
        fake_mask_rand = torch.tensor([[[0.2, 0.7], [0.9, 0.1]]], dtype=torch.float32)

        with patch("torch.rand", return_value=fake_mask_rand):
            result = self.model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 4))
        self.assertEqual(result["sigmas"].shape, (1, 1, 4, 4))
        self.assertEqual(set(result["timesteps"].view(-1).tolist()), {200.0, 800.0})
        self.assertTrue(torch.equal(result["crepa_self_flow_mask"], fake_mask_rand < 0.5))

    def test_model_predict_uses_crepa_capture_block_override(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=3)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_3"] = torch.full((1, 4, 8), 3.0)
            kwargs["hidden_states_buffer"]["layer_7"] = torch.full((1, 4, 8), 7.0)
            return (torch.randn(1, 4, 64),)

        self.model.model = MagicMock(side_effect=_forward)

        prepared_batch = {
            "prompt_embeds": torch.randn(1, 2, 16),
            "timesteps": torch.tensor([500.0]),
            "noisy_latents": torch.randn(1, 16, 4, 4),
            "latents": torch.randn(1, 16, 4, 4),
            "crepa_capture_block_index": 7,
        }

        with patch("simpletuner.helpers.models.longcat_image.model.pack_latents", return_value=torch.randn(1, 4, 16)):
            with patch(
                "simpletuner.helpers.models.longcat_image.model.unpack_latents",
                return_value=torch.randn(1, 16, 4, 4),
            ):
                result = self.model.model_predict(prepared_batch)

        self.assertTrue(torch.equal(result["crepa_hidden_states"], torch.full((1, 4, 8), 7.0)))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([0.5], dtype=torch.float32)))

    def test_model_predict_accepts_tokenwise_timesteps(self):
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.model = MagicMock(return_value=(torch.randn(1, 4, 64),))

        prepared_batch = {
            "prompt_embeds": torch.randn(1, 2, 16),
            "timesteps": torch.tensor([[100.0, 900.0]], dtype=torch.float32),
            "noisy_latents": torch.randn(1, 16, 2, 2),
            "latents": torch.randn(1, 16, 2, 2),
        }

        with patch("simpletuner.helpers.models.longcat_image.model.pack_latents", return_value=torch.randn(1, 2, 16)):
            with patch(
                "simpletuner.helpers.models.longcat_image.model.unpack_latents",
                return_value=torch.randn(1, 16, 2, 2),
            ):
                self.model.model_predict(prepared_batch)

        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([[0.1, 0.9]], dtype=torch.float32)))

    def test_model_predict_edit_mode_appends_clean_conditioning_timesteps(self):
        self.model.config = SimpleNamespace(
            weight_dtype=torch.float32, base_weight_dtype=torch.float32, model_flavour="edit"
        )
        self.model._is_edit_flavour = lambda: True
        hidden_states_buffer = {"layer_7": torch.arange(4 * 8, dtype=torch.float32).view(1, 4, 8)}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=3)
        self.model.model = MagicMock(return_value=(torch.randn(1, 4, 64),))

        prepared_batch = {
            "prompt_embeds": torch.randn(1, 2, 16),
            "timesteps": torch.tensor([[100.0, 900.0]], dtype=torch.float32),
            "noisy_latents": torch.randn(1, 16, 2, 2),
            "latents": torch.randn(1, 16, 2, 2),
            "conditioning_latents": torch.randn(1, 16, 2, 2),
            "crepa_capture_block_index": 7,
        }

        with patch(
            "simpletuner.helpers.models.longcat_image.model.pack_latents",
            side_effect=[torch.randn(1, 2, 16), torch.randn(1, 2, 16)],
        ):
            with patch(
                "simpletuner.helpers.models.longcat_image.model.unpack_latents",
                return_value=torch.randn(1, 16, 2, 2),
            ):
                result = self.model.model_predict(prepared_batch)

        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(
            torch.equal(
                transformer_kwargs["timestep"],
                torch.tensor([[0.1, 0.9, 0.0, 0.0]], dtype=torch.float32),
            )
        )
        self.assertEqual(result["crepa_hidden_states"].shape[1], 2)


if __name__ == "__main__":
    unittest.main()
