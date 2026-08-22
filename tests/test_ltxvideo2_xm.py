import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.ltxvideo2.model import LTXVideo2
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class LTXVideo2XMTests(unittest.TestCase):
    def setUp(self):
        self.model = LTXVideo2.__new__(LTXVideo2)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(
            weight_dtype=torch.float32,
            framerate=24,
            twinflow_enabled=False,
            tread_config=None,
            context_parallel_size=1,
            context_parallel_comm_strategy="allgather",
            loss_type="l2",
            huber_schedule="constant",
            huber_c=0.1,
            audio_loss_weight=1.0,
        )
        self.model.model = MagicMock(config=SimpleNamespace(patch_size=1, patch_size_t=1))
        self.model.unwrap_model = lambda model=None, **kwargs: model
        self.model._load_connectors = MagicMock()
        self.model.connectors = MagicMock(
            return_value=(
                torch.randn(4, 3, 16),
                torch.randn(4, 2, 16),
                torch.ones(4, 3),
            )
        )
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.crepa_regularizer = None
        self.model.flow_matching_target = lambda latents, noise: noise - latents

    def _enable_xm(self, *, training_target="noise", selection_scope="sample", block_size=0):
        self.model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=2,
            training_target=training_target,
            selection_scope=selection_scope,
            block_size=block_size,
        )

    def test_validate_xm_support_rejects_unsupported_ltxvideo2_settings(self):
        self._enable_xm(training_target="route")
        with self.assertRaisesRegex(ValueError, "xm_training_target='noise'"):
            self.model._validate_xm_support()

        self._enable_xm(selection_scope="block", block_size=2)
        with self.assertRaisesRegex(ValueError, "xm_selection_scope='sample'"):
            self.model._validate_xm_support()

        self._enable_xm(block_size=2)
        with self.assertRaisesRegex(ValueError, "xm_block_size"):
            self.model._validate_xm_support()

    def test_model_predict_expands_xm_candidates_for_video_audio_and_conditioning(self):
        self._enable_xm()
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(**kwargs):
            kwargs["hidden_states_buffer"]["layer_0"] = torch.arange(4 * 1 * 2, dtype=torch.float32).view(4, 1, 2)
            return torch.zeros(4, 1, 128), torch.zeros(4, 1, 8)

        self.model.model = MagicMock(side_effect=_forward, config=SimpleNamespace(patch_size=1, patch_size_t=1))
        latents = torch.arange(2 * 128, dtype=torch.float32).view(2, 128, 1, 1, 1)
        audio_latents = torch.arange(2 * 8, dtype=torch.float32).view(2, 8, 1, 1)
        video_noise = torch.full((4, 128, 1, 1, 1), 3.0)
        audio_noise = torch.full((4, 8, 1, 1), 5.0)
        prepared_batch = {
            "latents": latents.clone(),
            "noise": torch.zeros_like(latents),
            "input_noise": torch.zeros_like(latents),
            "noisy_latents": torch.zeros_like(latents),
            "sigmas": torch.full((2, 1, 1, 1, 1), 0.25),
            "timesteps": torch.tensor([100.0, 200.0]),
            "audio_latents": audio_latents.clone(),
            "audio_noise": torch.zeros_like(audio_latents),
            "audio_noisy_latents": torch.zeros_like(audio_latents),
            "audio_sigmas": torch.full((2, 1, 1, 1), 0.5),
            "audio_timesteps": torch.tensor([100.0, 200.0]),
            "encoder_hidden_states": torch.randn(2, 4, 16),
            "encoder_attention_mask": torch.ones(2, 4),
            "metadata": ["a", "b"],
        }

        with patch("torch.randn_like", side_effect=[video_noise, audio_noise]):
            result = self.model.model_predict(prepared_batch)

        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertEqual(result["xm_candidate_count"], 2)
        self.assertEqual(tuple(prepared_batch["latents"].shape), (4, 128, 1, 1, 1))
        self.assertTrue(torch.equal(prepared_batch["latents"], latents.repeat(2, 1, 1, 1, 1)))
        self.assertTrue(torch.equal(prepared_batch["noise"], video_noise))
        self.assertTrue(torch.equal(prepared_batch["audio_noise"], audio_noise))
        self.assertTrue(
            torch.allclose(prepared_batch["noisy_latents"], 0.75 * prepared_batch["latents"] + 0.25 * video_noise)
        )
        self.assertTrue(
            torch.allclose(
                prepared_batch["audio_noisy_latents"],
                0.5 * prepared_batch["audio_latents"] + 0.5 * audio_noise,
            )
        )
        self.assertTrue(torch.equal(prepared_batch["flow_target"], video_noise - prepared_batch["latents"]))
        self.assertEqual(prepared_batch["metadata"], ["a", "b", "a", "b"])
        self.assertEqual(tuple(transformer_kwargs["hidden_states"].shape), (4, 1, 128))
        self.assertEqual(tuple(transformer_kwargs["audio_hidden_states"].shape), (4, 1, 8))
        self.assertEqual(tuple(transformer_kwargs["encoder_hidden_states"].shape), (4, 3, 16))
        self.assertEqual(tuple(result["hidden_states_buffer"]["layer_0"].shape), (4, 1, 2))

    def test_xm_loss_selects_winners_before_nextlat_auxiliary_loss(self):
        self._enable_xm()
        latents = torch.zeros(4, 1, 1, 1, 1)
        noise = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(4, 1, 1, 1, 1)
        target = noise - latents
        prediction = torch.tensor([5.0, 1.0, 2.0, -4.0]).view(4, 1, 1, 1, 1)
        audio_latents = torch.zeros(4, 1, 1, 1)
        audio_noise = torch.zeros(4, 1, 1, 1)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "input_noise": noise,
            "noisy_latents": noise,
            "flow_target": target,
            "sigmas": torch.ones(4, 1, 1, 1, 1),
            "timesteps": torch.tensor([100.0, 200.0, 100.0, 200.0]),
            "audio_latents": audio_latents,
            "audio_noise": audio_noise,
            "audio_noisy_latents": audio_noise,
            "metadata": ["a", "b", "a", "b"],
        }
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        model_output = {
            "model_prediction": prediction,
            "audio_prediction": torch.zeros(4, 1, 1, 1),
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "xm_candidate_count": 2,
        }

        loss, logs = self.model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], target[[2, 1]]))
        self.assertTrue(torch.equal(prepared_batch["noise"], noise[[2, 1]]))
        self.assertEqual(prepared_batch["metadata"], ["a", "b"])
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertNotIn("xm_candidate_count", model_output)
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
        self.model.nextlat_regularizer = nextlat
        self.model.crepa_regularizer = None
        self.model._twinflow_active = MagicMock(return_value=False)
        aux_loss, aux_logs = self.model.auxiliary_loss(model_output, prepared_batch, loss)

        self.assertEqual(tuple(nextlat.hidden_shape), (2, 3, 2))
        self.assertEqual(tuple(nextlat.prediction_shape), (2, 1, 1, 1, 1))
        self.assertAlmostEqual(aux_loss.item(), 0.5)
        self.assertEqual(aux_logs["nextlat_loss"], 0.5)


if __name__ == "__main__":
    unittest.main()
