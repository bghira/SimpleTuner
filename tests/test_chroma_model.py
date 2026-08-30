import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.chroma.model import Chroma
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


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
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=False,
            candidate_count=1,
            training_target="noise",
            selection_scope="sample",
            block_size=0,
        )

    def _enable_xm(
        self,
        *,
        training_target: str = "noise",
        selection_scope: str = "sample",
        block_size: int = 0,
        candidate_count: int = 2,
    ) -> None:
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=candidate_count,
            training_target=training_target,
            selection_scope=selection_scope,
            block_size=block_size,
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

    def test_xm_validation_rejects_unsupported_targets_and_block_size(self):
        self._enable_xm(training_target="route")
        with self.assertRaisesRegex(ValueError, "xm_training_target='noise'"):
            self.model._validate_xm_support()

        self._enable_xm(selection_scope="block")
        with self.assertRaisesRegex(ValueError, "xm_selection_scope='sample'"):
            self.model._validate_xm_support()

        self._enable_xm(block_size=2)
        with self.assertRaisesRegex(ValueError, "xm_block_size"):
            self.model._validate_xm_support()

    def test_xm_noise_candidates_expand_chroma_batch_candidate_major(self):
        self._enable_xm(candidate_count=3)
        latents = torch.arange(2 * 1 * 2 * 2, dtype=torch.float32).view(2, 1, 2, 2)
        candidate_noise = torch.full((6, 1, 2, 2), 4.0)
        batch = {
            "latents": latents.clone(),
            "noise": torch.zeros_like(latents),
            "input_noise": torch.zeros_like(latents),
            "noisy_latents": torch.zeros_like(latents),
            "sigmas": torch.tensor([0.25, 0.75], dtype=torch.float32),
            "timesteps": torch.tensor([250.0, 750.0], dtype=torch.float32),
            "prompt_embeds": torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3),
            "encoder_attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.int64),
            "added_cond_kwargs": {"text_embeds": torch.arange(2 * 5, dtype=torch.float32).view(2, 5)},
            "flowmap_r_timesteps": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "conditioning_latents": torch.full((2, 1, 2, 2), 3.0),
            "conditioning_pixel_values": torch.ones(2, 3, 4, 4),
            "metadata": ["a", "b"],
            "crepa_teacher_sigmas": torch.tensor([0.5, 0.25], dtype=torch.float32),
            "crepa_teacher_timesteps": torch.tensor([500.0, 250.0], dtype=torch.float32),
            "crepa_teacher_noisy_latents": torch.zeros_like(latents),
        }

        with patch("torch.randn_like", return_value=candidate_noise):
            self.model._prepare_xm_noise_candidates(batch)

        self.assertEqual(tuple(batch["latents"].shape), (6, 1, 2, 2))
        self.assertTrue(torch.equal(batch["latents"], latents.repeat(3, 1, 1, 1)))
        self.assertTrue(torch.equal(batch["timesteps"], torch.tensor([250.0, 750.0] * 3)))
        self.assertTrue(torch.equal(batch["flowmap_r_timesteps"], torch.tensor([0.1, 0.2] * 3)))
        self.assertTrue(torch.equal(batch["prompt_embeds"], batch["prompt_embeds"][:2].repeat(3, 1, 1)))
        self.assertEqual(batch["metadata"], ["a", "b", "a", "b", "a", "b"])

        sigma_grid = batch["sigmas"].view(6, 1, 1, 1)
        expected_noisy = (1.0 - sigma_grid) * batch["latents"] + sigma_grid * candidate_noise
        self.assertTrue(torch.allclose(batch["noisy_latents"], expected_noisy))
        self.assertTrue(torch.allclose(batch["flow_target"], candidate_noise - batch["latents"]))
        teacher_grid = batch["crepa_teacher_sigmas"].view(6, 1, 1, 1)
        expected_teacher_noisy = (1.0 - teacher_grid) * batch["latents"] + teacher_grid * candidate_noise
        self.assertTrue(torch.allclose(batch["crepa_teacher_noisy_latents"], expected_teacher_noisy))
        self.assertEqual(batch["xm_candidate_count"], 3)
        self.assertEqual(batch["xm_original_batch_size"], 2)

    def test_xm_model_predict_returns_candidate_count(self):
        self._enable_xm()
        self.model._prepare_xm_noise_candidates = MagicMock()
        self.model._model_predict_single = MagicMock(return_value={"model_prediction": torch.zeros(4, 1, 1, 1)})

        result = self.model.model_predict({"latents": torch.zeros(2, 1, 1, 1)})

        self.model._prepare_xm_noise_candidates.assert_called_once()
        self.assertEqual(result["xm_candidate_count"], 2)

    def test_xm_loss_selects_winners_and_trims_sample_aligned_values(self):
        self._enable_xm()
        latents = torch.zeros(4, 1, 1, 1)
        noise = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(4, 1, 1, 1)
        target = noise - latents
        prediction = torch.tensor([5.0, 1.0, 2.0, -4.0]).view(4, 1, 1, 1)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "input_noise": noise,
            "noisy_latents": noise,
            "sigmas": torch.ones(4),
            "timesteps": torch.tensor([100.0, 200.0, 100.0, 200.0]),
            "metadata": ["a0", "b0", "a1", "b1"],
            "xm_candidate_count": 2,
            "xm_original_batch_size": 2,
        }
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        model_output = {
            "model_prediction": prediction,
            "crepa_hidden_states": hidden.clone(),
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "xm_candidate_count": 2,
        }

        loss, logs = self.model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], target[[2, 1]]))
        self.assertTrue(torch.equal(prepared_batch["noise"], noise[[2, 1]]))
        self.assertEqual(prepared_batch["metadata"], ["a1", "b0"])
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertNotIn("xm_candidate_count", prepared_batch)
        self.assertNotIn("xm_candidate_count", model_output)
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)


if __name__ == "__main__":
    unittest.main()
