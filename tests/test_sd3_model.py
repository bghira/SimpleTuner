import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.sd3.model import SD3
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


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
            loss_type="l2",
        )
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=2)
        self.model.sample_flow_sigmas = MagicMock(return_value=(torch.tensor([750.0]), torch.tensor([750.0])))
        self.model._twinflow_active = lambda: False

    def test_model_supports_crepa_self_flow(self):
        self.assertTrue(self.model.supports_crepa_self_flow())

    def test_xm_rejects_unsupported_route_target(self):
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=2,
            training_target="route",
            selection_scope="sample",
            block_size=0,
        )

        with self.assertRaisesRegex(ValueError, "xm_training_target='noise'"):
            self.model._xm_noise_candidates_enabled({})

    def test_xm_rejects_block_selection_scope(self):
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=2,
            training_target="noise",
            selection_scope="block",
            block_size=2,
        )

        with self.assertRaisesRegex(ValueError, "xm_selection_scope='sample'"):
            self.model._xm_noise_candidates_enabled({})

    def test_prepare_xm_noise_candidates_expands_candidate_major(self):
        torch.manual_seed(1)
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=2,
            training_target="noise",
            selection_scope="sample",
            block_size=0,
        )
        latents = torch.arange(64, dtype=torch.float32).reshape(2, 2, 4, 4)
        prepared_batch = {
            "latents": latents.clone(),
            "noise": torch.zeros_like(latents),
            "input_noise": torch.zeros_like(latents),
            "noisy_latents": latents.clone(),
            "sigmas": torch.tensor([0.25, 0.75], dtype=torch.float32).view(2, 1, 1, 1),
            "timesteps": torch.tensor([250.0, 750.0], dtype=torch.float32),
            "encoder_hidden_states": torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
            "add_text_embeds": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "conditioning_latents": torch.ones(2, 2, 4, 4),
            "flowmap_r_timesteps": torch.tensor([125.0, 375.0], dtype=torch.float32),
            "metadata": [{"id": 0}, {"id": 1}],
        }

        self.model._prepare_xm_noise_candidates(prepared_batch)

        self.assertEqual(prepared_batch["latents"].shape[0], 4)
        self.assertTrue(torch.equal(prepared_batch["latents"][:2], latents))
        self.assertTrue(torch.equal(prepared_batch["latents"][2:], latents))
        self.assertTrue(
            torch.equal(prepared_batch["encoder_hidden_states"][:2], prepared_batch["encoder_hidden_states"][2:])
        )
        self.assertTrue(torch.equal(prepared_batch["add_text_embeds"][:2], prepared_batch["add_text_embeds"][2:]))
        self.assertTrue(torch.equal(prepared_batch["conditioning_latents"][:2], prepared_batch["conditioning_latents"][2:]))
        self.assertTrue(torch.equal(prepared_batch["flowmap_r_timesteps"], torch.tensor([125.0, 375.0, 125.0, 375.0])))
        self.assertEqual(prepared_batch["metadata"], [{"id": 0}, {"id": 1}, {"id": 0}, {"id": 1}])
        self.assertFalse(torch.equal(prepared_batch["noise"][:2], prepared_batch["noise"][2:]))
        self.assertTrue(torch.equal(prepared_batch["flow_target"], prepared_batch["noise"] - prepared_batch["latents"]))
        self.assertTrue(
            torch.equal(
                prepared_batch["noisy_latents"],
                (1.0 - prepared_batch["sigmas"]) * prepared_batch["latents"]
                + prepared_batch["sigmas"] * prepared_batch["noise"],
            )
        )

    def test_xm_loss_selects_winners_and_shrinks_batch(self):
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=2,
            training_target="noise",
            selection_scope="sample",
            block_size=0,
        )
        flow_target = torch.zeros(4, 1, 1, 4)
        prepared_batch = {
            "latents": torch.zeros(4, 1, 1, 4),
            "noise": torch.zeros(4, 1, 1, 4),
            "flow_target": flow_target,
            "timesteps": torch.ones(4),
            "metadata": [{"id": 0}, {"id": 1}, {"id": 0}, {"id": 1}],
            "xm_candidate_count": 2,
            "xm_original_batch_size": 2,
        }
        model_prediction = flow_target.clone()
        model_prediction[0] = 1.0
        model_prediction[1] = 0.0
        model_prediction[2] = 0.0
        model_prediction[3] = 1.0
        hidden_states = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        model_output = {
            "model_prediction": model_prediction,
            "hidden_states_buffer": {"layer_2": hidden_states.clone()},
            "crepa_hidden_states": hidden_states.clone(),
            "xm_candidate_count": 2,
        }

        loss, logs = self.model._xm_noise_loss_with_logs(
            prepared_batch,
            model_output,
            candidate_count=2,
            apply_conditioning_mask=True,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(model_output["xm_winner_indices"].tolist(), [1, 0])
        self.assertEqual(prepared_batch["latents"].shape[0], 2)
        self.assertEqual(prepared_batch["metadata"], [{"id": 0}, {"id": 1}])
        self.assertNotIn("xm_candidate_count", prepared_batch)
        self.assertEqual(model_output["model_prediction"].shape[0], 2)
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_2"], hidden_states[[2, 1]]))
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)

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
