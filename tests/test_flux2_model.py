import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.flux2.model import Flux2
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class _RecordingFlux2Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        timestep=None,
        r_timestep=None,
        img_ids=None,
        txt_ids=None,
        guidance=None,
        return_dict=True,
        **kwargs,
    ):
        del return_dict
        self.last_kwargs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "r_timestep": r_timestep,
            "img_ids": img_ids,
            "txt_ids": txt_ids,
            "guidance": guidance,
            **kwargs,
        }
        return SimpleNamespace(sample=torch.zeros_like(hidden_states))


class Flux2ModelTests(unittest.TestCase):
    def setUp(self):
        self.model = Flux2.__new__(Flux2)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(
            base_weight_dtype=torch.float32,
            flux_guidance_mode="constant",
            flux_guidance_value=1.0,
            twinflow_enabled=False,
            tread_config=None,
            crepa_self_flow_mask_ratio=0.0,
        )
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=2)
        self.model.sample_flow_sigmas = MagicMock(return_value=(torch.tensor([750.0]), torch.tensor([750.0])))

    def _enable_xm(self, candidate_count: int = 2, training_target: str = "noise", selection_scope: str = "sample"):
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=candidate_count,
            training_target=training_target,
            selection_scope=selection_scope,
            block_size=0,
        )

    def test_model_predict_uses_crepa_capture_block_override(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            kwargs["hidden_states_buffer"]["layer_9"] = torch.full((1, 4, 8), 9.0)
            return SimpleNamespace(sample=torch.randn(1, 4, 128))

        self.model.model = MagicMock(side_effect=_forward)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 128, 2, 2),
            "latents": torch.randn(1, 128, 2, 2),
            "timesteps": torch.tensor([250.0]),
            "prompt_embeds": torch.randn(1, 3, 16),
            "crepa_capture_block_index": 9,
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 128, 2, 2))
        self.assertTrue(torch.equal(result["crepa_hidden_states"], torch.full((1, 4, 8), 9.0)))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([0.25], dtype=torch.float32)))

    def test_model_predict_accepts_tokenwise_timesteps(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            return SimpleNamespace(sample=torch.randn(1, 4, 128))

        self.model.model = MagicMock(side_effect=_forward)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 128, 2, 2),
            "latents": torch.randn(1, 128, 2, 2),
            "timesteps": torch.tensor([[100.0, 900.0, 500.0, 700.0]], dtype=torch.float32),
            "prompt_embeds": torch.randn(1, 3, 16),
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 128, 2, 2))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(
            torch.equal(
                transformer_kwargs["timestep"],
                torch.tensor([[0.1, 0.9, 0.5, 0.7]], dtype=torch.float32),
            )
        )

    def test_model_predict_forwards_anyflow_r_timestep(self):
        transformer = _RecordingFlux2Transformer()
        self.model.model = transformer
        self.model.unwrap_model = MagicMock(side_effect=lambda model=None, **_: model)
        r_timesteps = torch.tensor([0.25])

        result = self.model.model_predict(
            {
                "noisy_latents": torch.randn(1, 128, 2, 2),
                "latents": torch.randn(1, 128, 2, 2),
                "timesteps": torch.tensor([250.0]),
                "prompt_embeds": torch.randn(1, 3, 16),
                Flux2.FLOWMAP_R_TIMESTEP_BATCH_KEY: r_timesteps,
            }
        )

        self.assertIs(transformer.last_kwargs["r_timestep"], r_timesteps)
        self.assertEqual(result["model_prediction"].shape, (1, 128, 2, 2))

    def test_model_predict_appends_clean_conditioning_timesteps(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.arange(6 * 8, dtype=torch.float32).view(1, 6, 8)
            return SimpleNamespace(sample=torch.randn(1, 6, 128))

        self.model.model = MagicMock(side_effect=_forward)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 128, 2, 2),
            "latents": torch.randn(1, 128, 2, 2),
            "timesteps": torch.tensor([[100.0, 900.0, 500.0, 700.0]], dtype=torch.float32),
            "prompt_embeds": torch.randn(1, 3, 16),
            "conditioning_packed_latents": torch.randn(1, 2, 128),
            "conditioning_ids": torch.zeros(1, 2, 4),
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 128, 2, 2))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(
            torch.equal(
                transformer_kwargs["timestep"],
                torch.tensor([[0.1, 0.9, 0.5, 0.7, 0.0, 0.0]], dtype=torch.float32),
            )
        )
        self.assertEqual(result["crepa_hidden_states"].shape, (1, 4, 8))

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        batch = {
            "latents": torch.randn(1, 32, 4, 4),
            "input_noise": torch.randn(1, 32, 4, 4),
            "sigmas": torch.tensor([250.0]),
            "timesteps": torch.tensor([250.0]),
        }

        result = self.model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 4))
        self.assertEqual(result["crepa_self_flow_mask"].shape, (1, 2, 2))
        self.assertEqual(result["crepa_teacher_timesteps"].shape, (1,))

    def test_xm_validation_rejects_unsupported_route_target(self):
        self._enable_xm(training_target="route")

        with self.assertRaisesRegex(ValueError, "xm_training_target='noise'"):
            self.model._validate_xm_support()

    def test_xm_validation_rejects_block_selection_scope(self):
        self._enable_xm(selection_scope="block")

        with self.assertRaisesRegex(ValueError, "xm_selection_scope='sample'"):
            self.model._validate_xm_support()

    def test_xm_noise_candidates_expand_packed_conditioning_candidate_major(self):
        self._enable_xm(candidate_count=3)
        batch = {
            "latents": torch.ones(2, 128, 2, 2),
            "noise": torch.zeros(2, 128, 2, 2),
            "input_noise": torch.zeros(2, 128, 2, 2),
            "noisy_latents": torch.zeros(2, 128, 2, 2),
            "sigmas": torch.full((2, 1, 1, 1), 0.25),
            "timesteps": torch.tensor([250.0, 750.0]),
            "prompt_embeds": torch.randn(2, 3, 16),
            "conditioning_packed_latents": torch.arange(2 * 2 * 128, dtype=torch.float32).view(2, 2, 128),
            "conditioning_ids": torch.zeros(2, 2, 4),
            "guidance": torch.tensor([1.5, 2.5]),
        }

        self.model._prepare_xm_noise_candidates(batch)

        self.assertEqual(tuple(batch["latents"].shape), (6, 128, 2, 2))
        self.assertTrue(torch.equal(batch["timesteps"], torch.tensor([250.0, 750.0, 250.0, 750.0, 250.0, 750.0])))
        self.assertTrue(torch.equal(batch["guidance"], torch.tensor([1.5, 2.5, 1.5, 2.5, 1.5, 2.5])))
        self.assertEqual(tuple(batch["conditioning_packed_latents"].shape), (6, 2, 128))
        self.assertTrue(torch.equal(batch["flow_target"], batch["noise"] - batch["latents"]))
        expected_noisy = 0.75 * batch["latents"] + 0.25 * batch["noise"]
        self.assertTrue(torch.allclose(batch["noisy_latents"], expected_noisy))

    def test_model_predict_xm_expands_before_transformer_with_conditioning(self):
        self._enable_xm(candidate_count=2)
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.ones(kwargs["hidden_states"].shape[0], 6, 8)
            return SimpleNamespace(sample=torch.zeros_like(kwargs["hidden_states"]))

        self.model.model = MagicMock(side_effect=_forward)
        prepared_batch = {
            "noisy_latents": torch.randn(2, 128, 2, 2),
            "latents": torch.randn(2, 128, 2, 2),
            "noise": torch.randn(2, 128, 2, 2),
            "sigmas": torch.full((2, 1, 1, 1), 0.5),
            "timesteps": torch.tensor([250.0, 750.0]),
            "prompt_embeds": torch.randn(2, 3, 16),
            "conditioning_packed_latents": torch.randn(2, 2, 128),
            "conditioning_ids": torch.zeros(2, 2, 4),
        }

        result = self.model.model_predict(prepared_batch)

        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertEqual(tuple(transformer_kwargs["hidden_states"].shape), (4, 6, 128))
        self.assertEqual(tuple(transformer_kwargs["encoder_hidden_states"].shape), (4, 3, 16))
        self.assertEqual(tuple(transformer_kwargs["timestep"].shape), (4, 6))
        self.assertEqual(tuple(transformer_kwargs["guidance"].shape), (4,))
        self.assertEqual(tuple(result["model_prediction"].shape), (4, 128, 2, 2))
        self.assertEqual(result["xm_candidate_count"], 2)

    def test_xm_loss_selects_winners_and_trims_nextlat_hidden_states(self):
        self._enable_xm(candidate_count=2)
        self.model.config.loss_type = "l2"
        self.model.config.huber_schedule = "constant"
        self.model.config.huber_c = 0.1
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
            "prompt_embeds": torch.randn(4, 3, 8),
            "conditioning_packed_latents": torch.randn(4, 2, 1),
        }
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        model_output = {
            "model_prediction": prediction,
            "hidden_states_buffer": {"layer_1": hidden.clone()},
            "xm_candidate_count": 2,
        }

        loss, logs = self.model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], target[[2, 1]]))
        self.assertTrue(torch.equal(prepared_batch["noise"], noise[[2, 1]]))
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_1"], hidden[[2, 1]]))
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)


if __name__ == "__main__":
    unittest.main()
