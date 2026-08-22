import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.cosmos.model import Cosmos2Image
from simpletuner.helpers.training.crepa import CrepaMode
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class CosmosModelTests(unittest.TestCase):
    def setUp(self):
        self.model = Cosmos2Image.__new__(Cosmos2Image)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(
            weight_dtype=torch.float32,
            base_weight_dtype=torch.float32,
            crepa_self_flow_mask_ratio=0.5,
            twinflow_enabled=False,
            scheduled_sampling_reflexflow=False,
            scheduled_sampling_max_step_offset=0,
            input_perturbation=0.0,
        )
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=2)
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.prepare_edm_sigmas = MagicMock(return_value={"sigmas": torch.tensor([0.8], dtype=torch.float32)})
        self.model.model = MagicMock(config=SimpleNamespace(patch_size=(1, 2, 2)))
        self.model.unwrap_model = lambda model=None, wrapped=None: model if model is not None else wrapped

    def _enable_xm(
        self,
        candidate_count: int = 2,
        training_target: str = "noise",
        selection_scope: str = "sample",
        block_size: int = 0,
    ):
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=candidate_count,
            training_target=training_target,
            selection_scope=selection_scope,
            block_size=block_size,
        )

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

    def test_xm_rejects_route_block_and_block_size(self):
        self._enable_xm(training_target="route")
        with self.assertRaisesRegex(ValueError, "xm_training_target='noise'"):
            self.model._xm_noise_candidates_enabled()

        self._enable_xm(selection_scope="block")
        with self.assertRaisesRegex(ValueError, "xm_selection_scope='sample'"):
            self.model._xm_noise_candidates_enabled()

        self._enable_xm(block_size=2)
        with self.assertRaisesRegex(ValueError, "xm_block_size=0"):
            self.model._xm_noise_candidates_enabled()

    def test_xm_noise_candidates_expand_candidate_major_with_edm_noise(self):
        self._enable_xm(candidate_count=3)
        latents = torch.arange(16, dtype=torch.float32).reshape(2, 1, 2, 2, 2)
        batch = {
            "latents": latents.clone(),
            "noise": torch.zeros_like(latents),
            "input_noise": torch.zeros_like(latents),
            "noisy_latents": latents.clone(),
            "sigmas": torch.tensor([0.25, 0.75], dtype=torch.float32).view(2, 1, 1, 1, 1),
            "timesteps": torch.tensor([0.25, 0.75], dtype=torch.float32),
            "encoder_hidden_states": torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
            "conditioning_pixel_values": torch.ones(2, 1, 2, 2),
            "flowmap_r_timesteps": torch.tensor([0.1, 0.2], dtype=torch.float32),
        }

        self.model._prepare_xm_noise_candidates(batch)

        self.assertEqual(tuple(batch["latents"].shape), (6, 1, 2, 2, 2))
        self.assertTrue(torch.equal(batch["latents"][:2], latents))
        self.assertTrue(torch.equal(batch["latents"][2:4], latents))
        self.assertTrue(torch.equal(batch["encoder_hidden_states"][:2], batch["encoder_hidden_states"][2:4]))
        self.assertTrue(torch.equal(batch["flowmap_r_timesteps"], torch.tensor([0.1, 0.2, 0.1, 0.2, 0.1, 0.2])))
        self.assertFalse(torch.equal(batch["noise"][:2], batch["noise"][2:4]))
        expected_noisy = batch["latents"] + batch["sigmas"] * batch["noise"]
        self.assertTrue(torch.allclose(batch["noisy_latents"], expected_noisy))
        self.assertEqual(batch["xm_candidate_count"], 3)

    def test_xm_loss_selects_winners_and_trims_batch(self):
        self._enable_xm(candidate_count=2)
        latents = torch.zeros(4, 1, 1, 1, 1)
        prediction = torch.tensor([5.0, 0.0, 0.0, 4.0], dtype=torch.float32).view(4, 1, 1, 1, 1)
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        prepared_batch = {
            "latents": latents,
            "noise": torch.zeros_like(latents),
            "noisy_latents": torch.zeros_like(latents),
            "sigmas": torch.ones(4, 1, 1, 1, 1),
            "timesteps": torch.ones(4),
            "encoder_hidden_states": torch.arange(4 * 2 * 2, dtype=torch.float32).reshape(4, 2, 2),
            "xm_candidate_count": 2,
            "xm_original_batch_size": 2,
        }
        model_output = {
            "model_prediction": prediction,
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "xm_candidate_count": 2,
        }

        loss, logs = self.model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], prediction[[2, 1]]))
        self.assertTrue(torch.equal(prepared_batch["latents"], latents[[2, 1]]))
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertNotIn("xm_candidate_count", model_output)
        self.assertNotIn("xm_candidate_count", prepared_batch)
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)


if __name__ == "__main__":
    unittest.main()
