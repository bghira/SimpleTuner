import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.sana.model import Sana
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class SanaModelTests(unittest.TestCase):
    def setUp(self):
        self.model = Sana.__new__(Sana)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(
            base_weight_dtype=torch.float32,
            weighting_scheme="none",
            input_perturbation=0.0,
            twinflow_enabled=False,
            scheduled_sampling_reflexflow=False,
            scheduled_sampling_max_step_offset=0,
            crepa_self_flow=False,
            crepa_feature_source=None,
        )
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=2)
        self.model.unwrap_model = MagicMock(side_effect=lambda model=None, **kwargs: model)
        self.model.diff2flow_bridge = None

    def _enable_xm(self, candidate_count: int = 2):
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=candidate_count,
            training_target="noise",
            selection_scope="sample",
            block_size=0,
        )

    def test_model_predict_uses_crepa_capture_block_override(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            kwargs["hidden_states_buffer"]["layer_5"] = torch.full((1, 4, 8), 5.0)
            return (torch.randn(1, 32, 4, 4),)

        self.model.model = MagicMock(side_effect=_forward)
        self.model.model.config = SimpleNamespace(patch_size=2)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 32, 4, 4),
            "timesteps": torch.tensor([400], dtype=torch.int64),
            "encoder_attention_mask": torch.ones(1, 4),
            "encoder_hidden_states": torch.randn(1, 4, 16),
            "crepa_capture_block_index": 5,
        }

        result = self.model.model_predict(prepared_batch)

        self.assertTrue(torch.equal(result["crepa_hidden_states"], torch.full((1, 4, 8), 5.0)))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([400], dtype=torch.int64)))

    def test_model_predict_accepts_tokenwise_timesteps(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            return (torch.randn(1, 32, 4, 4),)

        self.model.model = MagicMock(side_effect=_forward)
        self.model.model.config = SimpleNamespace(patch_size=2)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 32, 4, 4),
            "timesteps": torch.tensor([[100, 900, 250, 750]], dtype=torch.int64),
            "encoder_attention_mask": torch.ones(1, 4),
            "encoder_hidden_states": torch.randn(1, 4, 16),
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 32, 4, 4))
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
            "latents": torch.zeros(1, 32, 4, 4, dtype=torch.float32),
            "input_noise": torch.ones(1, 32, 4, 4, dtype=torch.float32),
            "sigmas": torch.tensor([0.1], dtype=torch.float32),
            "timesteps": torch.tensor([100.0], dtype=torch.float32),
        }

        result = self.model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 4))
        self.assertTrue(torch.equal(result["crepa_teacher_timesteps"], torch.tensor([100.0], dtype=torch.float32)))

    def test_xm_noise_candidates_expand_candidate_major(self):
        self._enable_xm(candidate_count=3)
        batch = {
            "latents": torch.ones(2, 1, 2, 2),
            "noise": torch.zeros(2, 1, 2, 2),
            "input_noise": torch.zeros(2, 1, 2, 2),
            "noisy_latents": torch.zeros(2, 1, 2, 2),
            "sigmas": torch.full((2, 1, 1, 1), 0.25),
            "timesteps": torch.tensor([250.0, 750.0]),
            "encoder_attention_mask": torch.ones(2, 4),
            "encoder_hidden_states": torch.randn(2, 4, 16),
        }

        self.model._prepare_xm_noise_candidates(batch, family_name="Sana")

        self.assertEqual(tuple(batch["latents"].shape), (6, 1, 2, 2))
        self.assertTrue(torch.equal(batch["timesteps"], torch.tensor([250.0, 750.0, 250.0, 750.0, 250.0, 750.0])))
        expected_noisy = 0.75 * batch["latents"] + 0.25 * batch["noise"]
        self.assertTrue(torch.allclose(batch["noisy_latents"], expected_noisy))
        self.assertEqual(batch["xm_candidate_count"], 3)

    def test_xm_loss_selects_winners_and_trims_hidden_states(self):
        self._enable_xm(candidate_count=2)
        latents = torch.zeros(4, 1, 1, 1)
        noise = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(4, 1, 1, 1)
        target = noise - latents
        prediction = torch.tensor([5.0, 1.0, 2.0, -4.0]).view(4, 1, 1, 1)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "noisy_latents": noise,
            "sigmas": torch.ones(4, 1, 1, 1),
            "timesteps": torch.tensor([100.0, 200.0, 100.0, 200.0]),
        }
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        model_output = {
            "model_prediction": prediction,
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "xm_candidate_count": 2,
        }

        loss, logs = self.model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], target[[2, 1]]))
        self.assertTrue(torch.equal(prepared_batch["noise"], noise[[2, 1]]))
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)

    def test_xm_rejects_crepa_self_flow(self):
        self._enable_xm(candidate_count=2)
        self.model.config.crepa_self_flow = True

        with self.assertRaisesRegex(ValueError, "CREPA self-flow"):
            self.model._validate_xm_support()


if __name__ == "__main__":
    unittest.main()
