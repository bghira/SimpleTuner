import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.cosmos.model import Cosmos2Image
from simpletuner.helpers.training.crepa import CrepaMode


class CosmosModelTests(unittest.TestCase):
    def setUp(self):
        self.model = Cosmos2Image.__new__(Cosmos2Image)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(
            weight_dtype=torch.float32,
            base_weight_dtype=torch.float32,
            crepa_self_flow_mask_ratio=0.5,
        )
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=2)
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.prepare_edm_sigmas = MagicMock(return_value={"sigmas": torch.tensor([0.8], dtype=torch.float32)})
        self.model.model = MagicMock(config=SimpleNamespace(patch_size=(1, 2, 2)))
        self.model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped

    def test_model_supports_crepa_self_flow_and_image_mode(self):
        self.assertTrue(self.model.supports_crepa_self_flow())
        self.assertEqual(self.model.crepa_mode, CrepaMode.IMAGE)

    def test_prepare_crepa_self_flow_batch_builds_tokenwise_student_and_teacher_views(self):
        batch = {
            "latents": torch.zeros(1, 2, 1, 4, 4, dtype=torch.float32),
            "input_noise": torch.ones(1, 2, 1, 4, 4, dtype=torch.float32),
            "sigmas": torch.tensor([0.2], dtype=torch.float32),
            "timesteps": torch.tensor([0.2], dtype=torch.float32),
        }
        fake_mask_rand = torch.tensor([[[[0.2, 0.7], [0.9, 0.1]]]], dtype=torch.float32)

        with patch("torch.rand", return_value=fake_mask_rand):
            result = self.model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 4))
        self.assertEqual(result["sigmas"].shape, (1, 1, 1, 4, 4))
        self.assertEqual(result["crepa_teacher_timesteps"].shape, (1,))
        unique_timesteps = torch.unique(result["timesteps"].view(-1)).cpu()
        torch.testing.assert_close(unique_timesteps, torch.tensor([0.2, 0.8], dtype=torch.float32))
        self.assertTrue(torch.equal(result["crepa_self_flow_mask"], fake_mask_rand < 0.5))

    def test_model_predict_preserves_tokenwise_timesteps_and_capture_override(self):
        captured = torch.randn(1, 4, 8)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_7"] = captured
            return (torch.randn(1, 2, 1, 4, 4),)

        self.model.model = MagicMock(side_effect=_forward, config=SimpleNamespace(patch_size=(1, 2, 2)))

        prepared_batch = {
            "noisy_latents": torch.randn(1, 2, 1, 4, 4),
            "sigmas": torch.full((1, 1, 1, 4, 4), 0.2),
            "timesteps": torch.tensor([[0.1, 0.9, 0.1, 0.9]], dtype=torch.float32),
            "encoder_hidden_states": torch.randn(1, 3, 8),
            "crepa_capture_block_index": 7,
        }

        result = self.model.model_predict(prepared_batch)

        self.assertIs(result["crepa_hidden_states"], captured)
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(
            torch.equal(transformer_kwargs["timestep"], prepared_batch["timesteps"] / (prepared_batch["timesteps"] + 1.0))
        )


if __name__ == "__main__":
    unittest.main()
