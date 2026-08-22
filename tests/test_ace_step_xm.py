import unittest
from types import SimpleNamespace

import torch

from simpletuner.helpers.models.ace_step.model import ACEStep
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class ACEStepXMTests(unittest.TestCase):
    def _model(self, *, training_target: str = "noise", selection_scope: str = "sample") -> ACEStep:
        model = ACEStep.__new__(ACEStep)
        model.config = SimpleNamespace(loss_type="l2")
        model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=2,
            training_target=training_target,
            selection_scope=selection_scope,
            block_size=0,
        )
        model._is_v15_layout_active = lambda: False
        return model

    def test_xm_rejects_unsupported_route_target(self):
        model = self._model(training_target="route")

        with self.assertRaisesRegex(ValueError, "xm_training_target='noise'"):
            model._xm_noise_candidates_enabled()

    def test_xm_rejects_block_selection_scope(self):
        model = self._model(selection_scope="block")

        with self.assertRaisesRegex(ValueError, "xm_selection_scope='sample'"):
            model._xm_noise_candidates_enabled()

    def test_prepare_xm_noise_candidates_expands_candidate_major(self):
        torch.manual_seed(1)
        model = self._model()
        latents = torch.arange(16, dtype=torch.float32).reshape(2, 1, 2, 4)
        prepared_batch = {
            "latents": latents.clone(),
            "noisy_latents": latents.clone(),
            "sigmas": torch.tensor([0.25, 0.75], dtype=torch.float32).view(2, 1, 1, 1),
            "timesteps": torch.tensor([250.0, 750.0], dtype=torch.float32),
            "attention_mask": torch.ones(2, 4, dtype=torch.float32),
            "encoder_hidden_states": torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
            "lyric_token_ids": torch.arange(4, dtype=torch.long).reshape(2, 2),
            "metadata": [{"id": 0}, {"id": 1}],
            "ssl_hidden_states": [[torch.ones(3, 2), torch.zeros(3, 2)]],
        }

        model._prepare_xm_noise_candidates(prepared_batch)

        self.assertEqual(prepared_batch["latents"].shape[0], 4)
        self.assertTrue(torch.equal(prepared_batch["latents"][:2], latents))
        self.assertTrue(torch.equal(prepared_batch["latents"][2:], latents))
        self.assertEqual(prepared_batch["metadata"], [{"id": 0}, {"id": 1}, {"id": 0}, {"id": 1}])
        self.assertEqual(len(prepared_batch["ssl_hidden_states"]), 1)
        self.assertEqual(len(prepared_batch["ssl_hidden_states"][0]), 4)
        self.assertFalse(torch.equal(prepared_batch["noise"][:2], prepared_batch["noise"][2:]))
        self.assertTrue(torch.equal(prepared_batch["flow_target"], prepared_batch["noise"] - prepared_batch["latents"]))

    def test_xm_loss_selects_winners_and_shrinks_hidden_state_buffer(self):
        model = self._model()
        latents = torch.zeros(2, 1, 1, 4)
        prepared_batch = {
            "latents": latents.repeat((2, 1, 1, 1)),
            "attention_mask": torch.ones(4, 4, dtype=torch.float32),
            "xm_candidate_count": 2,
            "xm_original_batch_size": 2,
        }
        model_prediction = prepared_batch["latents"].clone()
        model_prediction[0] = 1.0
        model_prediction[1] = 0.0
        model_prediction[2] = 0.0
        model_prediction[3] = 1.0
        hidden_states = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        model_output = {
            "model_prediction": model_prediction,
            "hidden_states_buffer": {"layer_0": hidden_states.clone()},
            "crepa_hidden_states": hidden_states.clone(),
            "xm_candidate_count": 2,
        }

        loss, logs = model._xm_noise_loss_with_logs(
            prepared_batch,
            model_output,
            candidate_count=2,
            apply_conditioning_mask=True,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(model_output["xm_winner_indices"].tolist(), [1, 0])
        self.assertEqual(prepared_batch["latents"].shape[0], 2)
        self.assertNotIn("xm_candidate_count", prepared_batch)
        self.assertEqual(model_output["model_prediction"].shape[0], 2)
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden_states[[2, 1]]))
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)


if __name__ == "__main__":
    unittest.main()
