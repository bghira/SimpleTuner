import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.longcat_video.model import LongCatVideo
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class LongCatVideoXMTests(unittest.TestCase):
    def _shell(self, candidate_count: int = 2):
        model = LongCatVideo.__new__(LongCatVideo)
        model.config = SimpleNamespace(
            weight_dtype=torch.float32,
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
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.diff2flow_bridge = None
        model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=candidate_count,
            training_target="noise",
            selection_scope="sample",
            block_size=0,
        )
        model._new_hidden_state_buffer = MagicMock(return_value={})
        model.crepa_regularizer = None
        return model

    def test_xm_noise_candidates_expand_video_batch_candidate_major(self):
        model = self._shell(candidate_count=3)
        batch = {
            "latents": torch.ones(2, 1, 1, 2, 2),
            "noise": torch.zeros(2, 1, 1, 2, 2),
            "input_noise": torch.zeros(2, 1, 1, 2, 2),
            "noisy_latents": torch.zeros(2, 1, 1, 2, 2),
            "sigmas": torch.full((2, 1, 1, 1, 1), 0.25),
            "timesteps": torch.tensor([250.0, 750.0]),
            "encoder_hidden_states": torch.randn(2, 4, 8),
            "encoder_attention_mask": torch.ones(2, 4),
            LongCatVideo.FLOWMAP_R_TIMESTEP_BATCH_KEY: torch.tensor([0.1, 0.2]),
        }

        model._prepare_xm_noise_candidates(batch, family_name=LongCatVideo.NAME)

        self.assertEqual(tuple(batch["latents"].shape), (6, 1, 1, 2, 2))
        self.assertTrue(torch.equal(batch["timesteps"], torch.tensor([250.0, 750.0, 250.0, 750.0, 250.0, 750.0])))
        self.assertTrue(
            torch.equal(
                batch[LongCatVideo.FLOWMAP_R_TIMESTEP_BATCH_KEY],
                torch.tensor([0.1, 0.2, 0.1, 0.2, 0.1, 0.2]),
            )
        )
        expected_noisy = 0.75 * batch["latents"] + 0.25 * batch["noise"]
        self.assertTrue(torch.allclose(batch["noisy_latents"], expected_noisy))
        self.assertEqual(batch["xm_candidate_count"], 3)

    def test_xm_loss_selects_winners_and_trims_hidden_states(self):
        model = self._shell(candidate_count=2)
        latents = torch.zeros(4, 1, 1, 1, 1)
        noise = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(4, 1, 1, 1, 1)
        target = noise - latents
        prediction = torch.tensor([5.0, 1.0, 2.0, -4.0]).view(4, 1, 1, 1, 1)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "noisy_latents": noise,
            "sigmas": torch.ones(4, 1, 1, 1, 1),
            "timesteps": torch.tensor([100.0, 200.0, 100.0, 200.0]),
        }
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        model_output = {
            "model_prediction": prediction,
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "xm_candidate_count": 2,
        }

        loss, logs = model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], target[[2, 1]]))
        self.assertTrue(torch.equal(prepared_batch["noise"], noise[[2, 1]]))
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)

    def test_xm_rejects_crepa_self_flow(self):
        model = self._shell(candidate_count=2)
        model.config.crepa_self_flow = True

        with self.assertRaisesRegex(ValueError, "CREPA self-flow"):
            model._validate_xm_support()


if __name__ == "__main__":
    unittest.main()
