import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.wan_s2v.model import WanS2V
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class _RecordingWanS2VTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        motion_latents,
        audio_embeds,
        image_latents,
        pose_latents,
        r_timestep=None,
        **kwargs,
    ):
        self.last_kwargs = {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "motion_latents": motion_latents,
            "audio_embeds": audio_embeds,
            "image_latents": image_latents,
            "pose_latents": pose_latents,
            "r_timestep": r_timestep,
            **kwargs,
        }
        return (torch.zeros_like(hidden_states),)


class WanS2VXMTests(unittest.TestCase):
    def _wan_s2v_xm_shell(
        self,
        *,
        candidate_count: int = 2,
        training_target: str = "noise",
        selection_scope: str = "sample",
        block_size: int = 0,
    ):
        model = object.__new__(WanS2V)
        model.config = SimpleNamespace(
            input_perturbation=0.0,
            loss_type="l2",
            scheduled_sampling_max_step_offset=0,
            scheduled_sampling_reflexflow=False,
            twinflow_enabled=False,
            weight_dtype=torch.float32,
        )
        model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=candidate_count,
            training_target=training_target,
            selection_scope=selection_scope,
            block_size=block_size,
        )
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.crepa_regularizer = None
        model.internal_guidance_regularizer = None
        model.nextlat_regularizer = None
        model.unwrap_model = MagicMock(side_effect=lambda model=None, **_: model)
        model._new_hidden_state_buffer = MagicMock(return_value={})
        model._twinflow_active = MagicMock(return_value=False)
        return model

    def test_xm_support_rejects_route_block_and_block_size_modes(self):
        with self.assertRaisesRegex(ValueError, "xm_training_target='noise'"):
            self._wan_s2v_xm_shell(training_target="route")._validate_xm_support()

        with self.assertRaisesRegex(ValueError, "xm_selection_scope='sample'"):
            self._wan_s2v_xm_shell(selection_scope="block", block_size=2)._validate_xm_support()

        with self.assertRaisesRegex(ValueError, "xm_block_size"):
            self._wan_s2v_xm_shell(block_size=2)._validate_xm_support()

    def test_prepare_xm_noise_candidates_expands_s2v_conditioning_candidate_major(self):
        model = self._wan_s2v_xm_shell(candidate_count=3)
        latents = torch.arange(2 * 1 * 1 * 2 * 2, dtype=torch.float32).view(2, 1, 1, 2, 2)
        candidate_noise = torch.arange(6 * 1 * 1 * 2 * 2, dtype=torch.float32).view(6, 1, 1, 2, 2)
        first_frame = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).view(2, 3, 4, 4)
        last_frame = first_frame + 100.0
        audio_embeds = torch.arange(2 * 2 * 3 * 1, dtype=torch.float32).view(2, 2, 3, 1)
        conditioning_latents = torch.arange(2 * 1 * 1 * 2 * 2, dtype=torch.float32).view(2, 1, 1, 2, 2) + 200.0
        batch = {
            "latents": latents.clone(),
            "noise": torch.zeros_like(latents),
            "input_noise": torch.zeros_like(latents),
            "noisy_latents": torch.zeros_like(latents),
            "sigmas": torch.tensor([0.25, 0.75]),
            "timesteps": torch.tensor([250.0, 750.0]),
            "encoder_hidden_states": torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3),
            "encoder_attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.float32),
            "audio_embeds": audio_embeds.clone(),
            "conditioning_latents": conditioning_latents.clone(),
            "conditioning_pixel_values_multi": [first_frame.clone(), last_frame.clone()],
            WanS2V.FLOWMAP_R_TIMESTEP_BATCH_KEY: torch.tensor([0.1, 0.2]),
            "metadata": ["sample-a", "sample-b"],
            "s2v_audio_paths": ["a.wav", "b.wav"],
            "s2v_audio_backend_ids": ["audio-a", "audio-b"],
        }

        with patch("torch.randn_like", return_value=candidate_noise):
            model._prepare_xm_noise_candidates(batch)

        self.assertEqual(tuple(batch["latents"].shape), (6, 1, 1, 2, 2))
        self.assertTrue(torch.equal(batch["latents"], latents.repeat(3, 1, 1, 1, 1)))
        self.assertTrue(torch.equal(batch["audio_embeds"], audio_embeds.repeat(3, 1, 1, 1)))
        self.assertTrue(torch.equal(batch["conditioning_latents"], conditioning_latents.repeat(3, 1, 1, 1, 1)))
        self.assertTrue(torch.equal(batch["timesteps"], torch.tensor([250.0, 750.0, 250.0, 750.0, 250.0, 750.0])))
        self.assertTrue(
            torch.equal(batch[WanS2V.FLOWMAP_R_TIMESTEP_BATCH_KEY], torch.tensor([0.1, 0.2, 0.1, 0.2, 0.1, 0.2]))
        )
        self.assertEqual(batch["metadata"], ["sample-a", "sample-b", "sample-a", "sample-b", "sample-a", "sample-b"])
        self.assertEqual(batch["s2v_audio_paths"], ["a.wav", "b.wav", "a.wav", "b.wav", "a.wav", "b.wav"])
        self.assertEqual(len(batch["conditioning_pixel_values_multi"]), 2)
        self.assertTrue(torch.equal(batch["conditioning_pixel_values_multi"][0], first_frame.repeat(3, 1, 1, 1)))
        self.assertTrue(torch.equal(batch["conditioning_pixel_values_multi"][1], last_frame.repeat(3, 1, 1, 1)))
        sigma_grid = batch["sigmas"].view(6, 1, 1, 1, 1)
        expected_noisy = (1.0 - sigma_grid) * batch["latents"] + sigma_grid * candidate_noise
        self.assertTrue(torch.equal(batch["noise"], candidate_noise))
        self.assertTrue(torch.allclose(batch["noisy_latents"], expected_noisy))
        self.assertTrue(torch.equal(batch["flow_target"], candidate_noise - batch["latents"]))
        self.assertEqual(batch["xm_candidate_count"], 3)

    def test_model_predict_returns_xm_candidate_count_and_preserves_s2v_kwargs(self):
        model = self._wan_s2v_xm_shell(candidate_count=2)
        transformer = _RecordingWanS2VTransformer()
        model.model = transformer
        latents = torch.zeros(2, 1, 1, 2, 2)
        candidate_noise = torch.ones(4, 1, 1, 2, 2)
        audio_embeds = torch.arange(2 * 2 * 3 * 1, dtype=torch.float32).view(2, 2, 3, 1)
        conditioning_latents = torch.arange(2 * 1 * 1 * 2 * 2, dtype=torch.float32).view(2, 1, 1, 2, 2)
        batch = {
            "latents": latents,
            "noise": torch.zeros_like(latents),
            "input_noise": torch.zeros_like(latents),
            "noisy_latents": torch.zeros_like(latents),
            "sigmas": torch.tensor([0.25, 0.75]),
            "timesteps": torch.tensor([250.0, 750.0]),
            "encoder_hidden_states": torch.randn(2, 4, 8),
            "audio_embeds": audio_embeds,
            "conditioning_latents": conditioning_latents,
            "force_keep_mask": torch.tensor([[True, False], [False, True]]),
            WanS2V.FLOWMAP_R_TIMESTEP_BATCH_KEY: torch.tensor([0.1, 0.2]),
        }

        with patch("torch.randn_like", return_value=candidate_noise):
            result = model.model_predict(batch)

        self.assertEqual(result["xm_candidate_count"], 2)
        self.assertEqual(tuple(transformer.last_kwargs["hidden_states"].shape), (4, 1, 1, 2, 2))
        self.assertEqual(tuple(transformer.last_kwargs["encoder_hidden_states"].shape), (4, 4, 8))
        self.assertTrue(torch.equal(transformer.last_kwargs["audio_embeds"], audio_embeds.repeat(2, 1, 1, 1)))
        self.assertTrue(torch.equal(transformer.last_kwargs["image_latents"], conditioning_latents.repeat(2, 1, 1, 1, 1)))
        self.assertTrue(torch.equal(transformer.last_kwargs["r_timestep"], torch.tensor([0.1, 0.2, 0.1, 0.2])))
        self.assertTrue(
            torch.equal(
                transformer.last_kwargs["force_keep_mask"],
                torch.tensor([[True, False], [False, True], [True, False], [False, True]]),
            )
        )

    def test_xm_loss_selects_winners_and_trims_s2v_state_before_nextlat(self):
        model = self._wan_s2v_xm_shell(candidate_count=2)
        latents = torch.zeros(4, 1, 1, 1, 1)
        noise = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32).view(4, 1, 1, 1, 1)
        target = noise - latents
        prediction = torch.tensor([5.0, 1.0, 2.0, -4.0], dtype=torch.float32).view(4, 1, 1, 1, 1)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "input_noise": noise,
            "noisy_latents": noise,
            "flow_target": target,
            "sigmas": torch.ones(4),
            "timesteps": torch.tensor([100.0, 200.0, 100.0, 200.0]),
            "metadata": ["a0", "b0", "a1", "b1"],
            "s2v_audio_paths": ["a0.wav", "b0.wav", "a1.wav", "b1.wav"],
            "audio_embeds": torch.zeros(4, 1, 1, 1),
        }
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).view(4, 3, 2)
        model_output = {
            "model_prediction": prediction,
            "crepa_hidden_states": hidden.clone(),
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "xm_candidate_count": 2,
        }

        loss, logs = model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], target[[2, 1]]))
        self.assertTrue(torch.equal(model_output["crepa_hidden_states"], hidden[[2, 1]]))
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertEqual(prepared_batch["metadata"], ["a1", "b0"])
        self.assertEqual(prepared_batch["s2v_audio_paths"], ["a1.wav", "b0.wav"])
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
        model.nextlat_regularizer = nextlat
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
