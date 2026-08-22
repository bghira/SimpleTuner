import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.minimaxh3.model import MiniMaxH3
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class MiniMaxH3XMTests(unittest.TestCase):
    def _model(
        self,
        *,
        training_target: str = "noise",
        selection_scope: str = "sample",
        block_size: int = 0,
        candidate_count: int = 2,
    ) -> MiniMaxH3:
        model = MiniMaxH3.__new__(MiniMaxH3)
        model.config = SimpleNamespace(
            loss_type="l2",
            audio_loss_weight=1.0,
            twinflow_enabled=False,
            scheduled_sampling_reflexflow=False,
            scheduled_sampling_max_step_offset=0,
            input_perturbation=0.0,
            weight_dtype=torch.float32,
        )
        model.diff2flow_bridge = None
        model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=candidate_count,
            training_target=training_target,
            selection_scope=selection_scope,
            block_size=block_size,
        )
        return model

    def test_xm_rejects_unsupported_modes(self):
        with self.assertRaisesRegex(ValueError, "xm_training_target='noise'"):
            self._model(training_target="route")._xm_noise_candidates_enabled()

        with self.assertRaisesRegex(ValueError, "xm_selection_scope='sample'"):
            self._model(selection_scope="block", block_size=2)._xm_noise_candidates_enabled()

        with self.assertRaisesRegex(ValueError, "xm_block_size"):
            self._model(block_size=2)._xm_noise_candidates_enabled()

    def test_prepare_xm_noise_candidates_expands_h3_batch_candidate_major(self):
        model = self._model(candidate_count=3)
        latents = torch.arange(2 * 2 * 1 * 2 * 2, dtype=torch.float32).view(2, 2, 1, 2, 2)
        audio_latents = torch.arange(2 * 2 * 3 * 2, dtype=torch.float32).view(2, 2, 3, 2)
        video_noise = torch.arange(6 * 2 * 1 * 2 * 2, dtype=torch.float32).view(6, 2, 1, 2, 2)
        audio_noise = torch.arange(6 * 2 * 3 * 2, dtype=torch.float32).view(6, 2, 3, 2) + 100.0
        batch = {
            "latents": latents.clone(),
            "noisy_latents": torch.zeros_like(latents),
            "sigmas": torch.tensor([0.25, 0.75], dtype=torch.float32),
            "timesteps": torch.tensor([250.0, 750.0], dtype=torch.float32),
            "encoder_hidden_states": torch.arange(2 * 2 * 3, dtype=torch.float32).view(2, 2, 3),
            "text_token_tags": torch.tensor([1, 0], dtype=torch.long),
            "text_encoder_output": {"text_token_tags": torch.tensor([1, 0], dtype=torch.long)},
            "conditioning_latents": torch.ones(2, 2, 1, 2, 2),
            "h3_conditioning_noise": torch.full((2, 2, 1, 2, 2), 2.0),
            "audio_latents": audio_latents.clone(),
            "audio_noisy_latents": torch.zeros_like(audio_latents),
            "audio_sigmas": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "audio_timesteps": torch.tensor([0.9, 0.8], dtype=torch.float32),
            "audio_latent_mask": torch.tensor([1.0, 0.0]),
            "flowmap_r_timesteps": torch.tensor([0.15, 0.35], dtype=torch.float32),
            "metadata": [{"id": 0}, {"id": 1}],
            "minimax_h3_target_mode": "av",
        }

        with patch("torch.randn_like", side_effect=[video_noise, audio_noise]):
            model._prepare_xm_noise_candidates(batch)

        self.assertEqual(tuple(batch["latents"].shape), (6, 2, 1, 2, 2))
        self.assertTrue(torch.equal(batch["latents"], latents.repeat(3, 1, 1, 1, 1)))
        self.assertTrue(torch.equal(batch["audio_latents"], audio_latents.repeat(3, 1, 1, 1)))
        self.assertEqual(tuple(batch["encoder_hidden_states"].shape), (6, 2, 3))
        self.assertTrue(torch.equal(batch["text_token_tags"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(batch["text_encoder_output"]["text_token_tags"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(batch["flowmap_r_timesteps"], torch.tensor([0.15, 0.35] * 3)))
        self.assertEqual(batch["metadata"], [{"id": 0}, {"id": 1}, {"id": 0}, {"id": 1}, {"id": 0}, {"id": 1}])
        self.assertFalse(torch.equal(batch["noise"][:2], batch["noise"][2:4]))
        self.assertFalse(torch.equal(batch["audio_noise"][:2], batch["audio_noise"][2:4]))
        video_sigma_grid = batch["sigmas"].view(6, 1, 1, 1, 1)
        expected_video_noisy = (1.0 - video_sigma_grid) * batch["latents"] + video_sigma_grid * video_noise
        self.assertTrue(torch.allclose(batch["noisy_latents"], expected_video_noisy))
        self.assertTrue(torch.allclose(batch["flow_target"], batch["latents"] - video_noise))
        audio_sigma_grid = batch["audio_sigmas"].view(6, 1, 1, 1)
        expected_audio_noisy = (1.0 - audio_sigma_grid) * batch["audio_latents"] + audio_sigma_grid * audio_noise
        self.assertTrue(torch.allclose(batch["audio_noisy_latents"], expected_audio_noisy))
        self.assertTrue(torch.allclose(batch["audio_target"], batch["audio_latents"] - audio_noise))
        self.assertEqual(batch["xm_candidate_count"], 3)

    def test_prepare_xm_noise_candidates_replaces_existing_audio_target(self):
        model = self._model(candidate_count=2)
        latents = torch.zeros(1, 2, 1, 2, 2)
        audio_latents = torch.zeros(1, 2, 3, 2)
        video_noise = torch.ones(2, 2, 1, 2, 2)
        audio_noise = torch.ones(2, 2, 3, 2) * 2.0
        batch = {
            "latents": latents,
            "noisy_latents": torch.zeros_like(latents),
            "sigmas": torch.tensor([0.25], dtype=torch.float32),
            "timesteps": torch.tensor([250.0], dtype=torch.float32),
            "encoder_hidden_states": torch.zeros(1, 2, 3),
            "conditioning_latents": torch.ones(1, 2, 1, 2, 2),
            "audio_latents": audio_latents,
            "audio_noisy_latents": torch.zeros_like(audio_latents),
            "audio_sigmas": torch.tensor([0.1], dtype=torch.float32),
            "audio_timesteps": torch.tensor([0.9], dtype=torch.float32),
            "audio_target": torch.full_like(audio_latents, 99.0),
        }

        with patch("torch.randn_like", side_effect=[video_noise, audio_noise]):
            model._prepare_xm_noise_candidates(batch)

        self.assertTrue(torch.equal(batch["audio_target"], batch["audio_latents"] - audio_noise))

    def test_model_predict_tags_xm_output(self):
        model = self._model(candidate_count=2)
        model._prepare_xm_noise_candidates = MagicMock()
        model._model_predict_for_prepared_batch = MagicMock(return_value={"model_prediction": torch.zeros(2, 1)})

        output = model.model_predict({"latents": torch.zeros(1, 1)})

        model._prepare_xm_noise_candidates.assert_called_once()
        model._model_predict_for_prepared_batch.assert_called_once()
        self.assertEqual(output["xm_candidate_count"], 2)

    def test_xm_loss_selects_winners_and_trims_before_auxiliary_loss(self):
        model = self._model(candidate_count=2)
        latents = torch.zeros(4, 1, 1, 1, 1)
        noise = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32).view(4, 1, 1, 1, 1)
        target = latents - noise
        prediction = torch.tensor([9.0, -1.0, -2.0, 9.0], dtype=torch.float32).view(4, 1, 1, 1, 1)
        audio_latents = torch.zeros(4, 2, 1, 1)
        audio_noise = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32).view(4, 1, 1, 1)
        audio_target = audio_latents - audio_noise
        audio_prediction = audio_target.clone()
        audio_prediction[0] = 4.0
        audio_prediction[3] = 4.0
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).view(4, 3, 2)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "noisy_latents": noise,
            "sigmas": torch.ones(4),
            "timesteps": torch.tensor([100.0, 200.0, 100.0, 200.0]),
            "audio_latents": audio_latents,
            "audio_noise": audio_noise,
            "audio_target": audio_target,
            "audio_latent_mask": torch.ones(4),
            "flowmap_r_timesteps": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            "metadata": ["a0", "b0", "a1", "b1"],
            "xm_candidate_count": 2,
            "xm_original_batch_size": 2,
        }
        model_output = {
            "model_prediction": prediction,
            "audio_prediction": audio_prediction,
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "crepa_hidden_states": hidden.clone(),
            "xm_candidate_count": 2,
        }

        loss, logs = model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], target[[2, 1]]))
        self.assertTrue(torch.equal(model_output["audio_prediction"], audio_target[[2, 1]]))
        self.assertTrue(torch.equal(prepared_batch["flowmap_r_timesteps"], torch.tensor([0.3, 0.2])))
        self.assertEqual(prepared_batch["metadata"], ["a1", "b0"])
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertNotIn("xm_candidate_count", model_output)
        self.assertNotIn("xm_candidate_count", prepared_batch)
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)

        class FakeNextLat:
            enabled = True

            def __init__(self):
                self.hidden_shape = None
                self.prediction_shape = None

            def compute_loss(self, hidden_states_buffer, output):
                self.hidden_shape = tuple(hidden_states_buffer["layer_0"].shape)
                self.prediction_shape = tuple(output["model_prediction"].shape)
                return torch.tensor(0.5), {"nextlat_loss": 0.5}

        nextlat = FakeNextLat()
        model.nextlat_regularizer = nextlat
        model.crepa_regularizer = None
        model.internal_guidance_regularizer = None
        model._twinflow_active = MagicMock(return_value=False)
        aux_loss, aux_logs = model.auxiliary_loss(
            model_output=model_output,
            prepared_batch=prepared_batch,
            loss=loss,
            apply_layersync=False,
            clear_hidden_state_buffer=False,
        )

        self.assertEqual(nextlat.hidden_shape, (2, 3, 2))
        self.assertEqual(nextlat.prediction_shape, (2, 1, 1, 1, 1))
        self.assertAlmostEqual(aux_loss.item(), 0.5)
        self.assertEqual(aux_logs["nextlat_loss"], 0.5)


if __name__ == "__main__":
    unittest.main()
