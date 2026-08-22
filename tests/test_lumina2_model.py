import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.lumina2.model import Lumina2
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class _DummyTransformer:
    def __init__(self, output=None, forward=None):
        self.config = SimpleNamespace(patch_size=2)
        self.output = output
        self.forward = forward
        self.call_args = None

    def __call__(self, **kwargs):
        self.call_args = SimpleNamespace(kwargs=kwargs)
        if self.forward is not None:
            return self.forward(**kwargs)
        return self.output


class Lumina2ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = Lumina2.__new__(Lumina2)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"), unwrap_model=lambda model: model)
        self.model.config = SimpleNamespace(
            base_weight_dtype=torch.float32,
            controlnet=False,
            crepa_self_flow_mask_ratio=0.5,
            input_perturbation=0.0,
            scheduled_sampling_reflexflow=False,
            scheduled_sampling_max_step_offset=0,
            twinflow_enabled=False,
            crepa_self_flow=False,
            crepa_feature_source=None,
            loss_type="l2",
            huber_schedule="constant",
            huber_c=0.1,
        )
        self.model.noise_schedule = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=2)
        self.model.model = _DummyTransformer()
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
            kwargs["hidden_states_buffer"]["layer_6"] = torch.full((1, 4, 8), 6.0)
            return (torch.ones(1, 16, 4, 4),)

        self.model.model = _DummyTransformer(forward=_forward)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 4, 4),
            "timesteps": torch.tensor([250.0]),
            "prompt_embeds": torch.randn(1, 4, 16),
            "encoder_attention_mask": torch.ones(1, 1, 4),
            "crepa_capture_block_index": 6,
        }

        result = self.model.model_predict(prepared_batch)

        self.assertTrue(torch.equal(result["crepa_hidden_states"], torch.full((1, 4, 8), 6.0)))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([0.75], dtype=torch.float32)))
        self.assertTrue(torch.equal(transformer_kwargs["encoder_attention_mask"], torch.ones(1, 4, dtype=torch.int32)))
        self.assertTrue(torch.equal(result["model_prediction"], -torch.ones(1, 16, 4, 4)))

    def test_model_predict_accepts_tokenwise_timesteps(self):
        self.model.model = _DummyTransformer(output=(torch.ones(1, 16, 4, 4),))

        prepared_batch = {
            "noisy_latents": torch.randn(1, 16, 4, 4),
            "timesteps": torch.tensor([[100.0, 900.0, 250.0, 750.0]], dtype=torch.float32),
            "prompt_embeds": torch.randn(1, 4, 16),
            "encoder_attention_mask": torch.ones(1, 4),
        }

        result = self.model.model_predict(prepared_batch)

        transformer_kwargs = self.model.model.call_args.kwargs
        torch.testing.assert_close(
            transformer_kwargs["timestep"],
            torch.tensor([[0.9, 0.1, 0.75, 0.25]], dtype=torch.float32),
        )
        self.assertTrue(torch.equal(result["model_prediction"], -torch.ones(1, 16, 4, 4)))

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        self.model.sample_flow_sigmas = MagicMock(
            return_value=(torch.tensor([0.75], dtype=torch.float32), torch.tensor([750.0], dtype=torch.float32))
        )

        batch = {
            "latents": torch.zeros(1, 16, 4, 4),
            "input_noise": torch.ones(1, 16, 4, 4),
            "sigmas": torch.tensor([0.25], dtype=torch.float32),
            "timesteps": torch.tensor([250.0], dtype=torch.float32),
        }

        updated = self.model._prepare_crepa_self_flow_batch(batch, state={"global_step": 0})

        self.assertEqual(updated["timesteps"].shape, torch.Size([1, 4]))
        self.assertEqual(updated["crepa_teacher_timesteps"].shape, torch.Size([1]))
        self.assertEqual(updated["crepa_teacher_noisy_latents"].shape, torch.Size([1, 16, 4, 4]))

    def test_xm_noise_candidates_expand_candidate_major(self):
        self._enable_xm(candidate_count=3)
        batch = {
            "latents": torch.ones(2, 1, 2, 2),
            "noise": torch.zeros(2, 1, 2, 2),
            "input_noise": torch.zeros(2, 1, 2, 2),
            "noisy_latents": torch.zeros(2, 1, 2, 2),
            "sigmas": torch.full((2, 1, 1, 1), 0.25),
            "timesteps": torch.tensor([250.0, 750.0]),
            "prompt_embeds": torch.randn(2, 4, 16),
            "encoder_attention_mask": torch.ones(2, 4),
        }

        self.model._prepare_xm_noise_candidates(batch, family_name="Lumina2")

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
