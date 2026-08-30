import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.common import PredictionTypes
from simpletuner.helpers.models.deepfloyd.model import DeepFloydIF
from simpletuner.helpers.models.sd1x.model import StableDiffusion1
from simpletuner.helpers.models.sdxl.model import SDXL
from simpletuner.helpers.models.stable_cascade.model import StableCascadeStageC
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig


class _NoiseSchedule:
    def __init__(self):
        self.config = SimpleNamespace(prediction_type="epsilon", num_train_timesteps=1000)

    def add_noise(self, latents, noise, timesteps):
        timestep_grid = timesteps.reshape(timesteps.shape[0], *([1] * (latents.ndim - 1))).to(
            device=latents.device,
            dtype=latents.dtype,
        )
        return latents + timestep_grid * noise

    def get_velocity(self, latents, noise, timesteps):
        return noise - latents


class SDUNetXMTests(unittest.TestCase):
    def _model(self, model_cls):
        model = model_cls.__new__(model_cls)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(
            base_weight_dtype=torch.float32,
            weight_dtype=torch.float32,
            loss_type="l2",
            snr_gamma=None,
            snr_weight=1.0,
            twinflow_enabled=False,
            scheduled_sampling_reflexflow=False,
            scheduled_sampling_max_step_offset=0,
            input_perturbation=0.0,
            model_flavour="i-medium-400m",
        )
        model.NAME = model_cls.NAME
        model.PREDICTION_TYPE = PredictionTypes.EPSILON
        model.noise_schedule = _NoiseSchedule()
        model.diff2flow_bridge = None
        model.xm_config = ExplorativeModelingConfig(
            enabled=False,
            candidate_count=1,
            training_target="noise",
            selection_scope="sample",
            block_size=0,
        )
        return model

    def _enable_xm(
        self,
        model,
        *,
        candidate_count=2,
        training_target="noise",
        selection_scope="sample",
        block_size=0,
    ):
        model.xm_config = ExplorativeModelingConfig(
            enabled=True,
            candidate_count=candidate_count,
            training_target=training_target,
            selection_scope=selection_scope,
            block_size=block_size,
        )

    def test_unet_families_reject_unsupported_xm_modes(self):
        for model_cls in (StableDiffusion1, SDXL, DeepFloydIF, StableCascadeStageC):
            with self.subTest(model=model_cls.NAME, mode="route"):
                model = self._model(model_cls)
                self._enable_xm(model, training_target="route")
                with self.assertRaisesRegex(ValueError, "xm_training_target='noise'"):
                    model._xm_noise_candidates_enabled()

            with self.subTest(model=model_cls.NAME, mode="block"):
                model = self._model(model_cls)
                self._enable_xm(model, selection_scope="block")
                with self.assertRaisesRegex(ValueError, "xm_selection_scope='sample'"):
                    model._xm_noise_candidates_enabled()

            with self.subTest(model=model_cls.NAME, mode="block_size"):
                model = self._model(model_cls)
                self._enable_xm(model, block_size=2)
                with self.assertRaisesRegex(ValueError, "xm_block_size"):
                    model._xm_noise_candidates_enabled()

    def test_sd1x_xm_expands_candidate_major_and_trims_winners(self):
        model = self._model(StableDiffusion1)
        self._enable_xm(model)
        latents = torch.arange(8, dtype=torch.float32).reshape(2, 1, 2, 2)
        candidate_noise = torch.full((4, 1, 2, 2), 2.0)
        prepared_batch = {
            "latents": latents.clone(),
            "noise": torch.zeros_like(latents),
            "input_noise": torch.zeros_like(latents),
            "noisy_latents": latents.clone(),
            "timesteps": torch.tensor([0.25, 0.75], dtype=torch.float32),
            "encoder_hidden_states": torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
            "conditioning_pixel_values": torch.ones(2, 3, 4, 4),
            "conditioning_latents": torch.ones(2, 1, 2, 2),
            "flowmap_r_timesteps": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "metadata": [{"id": 0}, {"id": 1}],
        }

        with patch("torch.randn_like", return_value=candidate_noise):
            model._prepare_xm_noise_candidates(prepared_batch)

        self.assertTrue(torch.equal(prepared_batch["latents"], latents.repeat(2, 1, 1, 1)))
        self.assertTrue(torch.equal(prepared_batch["timesteps"], torch.tensor([0.25, 0.75, 0.25, 0.75])))
        self.assertTrue(
            torch.equal(prepared_batch["encoder_hidden_states"][:2], prepared_batch["encoder_hidden_states"][2:])
        )
        self.assertTrue(torch.equal(prepared_batch["flowmap_r_timesteps"], torch.tensor([0.1, 0.2, 0.1, 0.2])))
        self.assertEqual(prepared_batch["metadata"], [{"id": 0}, {"id": 1}, {"id": 0}, {"id": 1}])
        expected_noisy = prepared_batch["latents"] + prepared_batch["timesteps"].view(4, 1, 1, 1) * candidate_noise
        self.assertTrue(torch.equal(prepared_batch["noisy_latents"], expected_noisy))

        noise = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32).view(4, 1, 1, 1)
        loss_batch = {
            "latents": torch.zeros_like(noise),
            "noise": noise,
            "input_noise": noise,
            "noisy_latents": noise,
            "timesteps": torch.tensor([100.0, 200.0, 100.0, 200.0]),
            "metadata": ["a0", "b0", "a1", "b1"],
            "xm_candidate_count": 2,
            "xm_original_batch_size": 2,
        }
        hidden = torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)
        model_output = {
            "model_prediction": torch.tensor([5.0, 1.0, 2.0, -4.0], dtype=torch.float32).view(4, 1, 1, 1),
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "xm_candidate_count": 2,
            "metadata_out": ["c0s0", "c0s1", "c1s0", "c1s1"],
        }

        loss, logs = model.loss_with_logs(loss_batch, model_output)

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(model_output["xm_winner_indices"].tolist(), [1, 0])
        self.assertEqual(loss_batch["metadata"], ["a1", "b0"])
        self.assertEqual(model_output["metadata_out"], ["c1s0", "c0s1"])
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)

    def test_sdxl_xm_expands_added_conditioning_and_controlnet_once(self):
        model = self._model(SDXL)
        self._enable_xm(model)
        model.controlnet = MagicMock(return_value=([torch.zeros(4, 1, 1, 1)], torch.zeros(4, 1, 1, 1)))
        model.model = MagicMock(return_value=(torch.zeros(4, 1, 2, 2),))
        latents = torch.zeros(2, 1, 2, 2)
        prepared_batch = {
            "latents": latents,
            "noise": torch.zeros_like(latents),
            "input_noise": torch.zeros_like(latents),
            "noisy_latents": latents.clone(),
            "timesteps": torch.tensor([0.25, 0.75], dtype=torch.float32),
            "encoder_hidden_states": torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
            "add_text_embeds": torch.arange(8, dtype=torch.float32).reshape(2, 4),
            "added_cond_kwargs": {
                "text_embeds": torch.arange(8, dtype=torch.float32).reshape(2, 4),
                "time_ids": torch.arange(12, dtype=torch.float32).reshape(2, 6),
            },
            "conditioning_pixel_values": torch.ones(2, 3, 4, 4),
            "metadata": ["a", "b"],
        }

        result = model.controlnet_predict(prepared_batch)

        self.assertEqual(result["xm_candidate_count"], 2)
        self.assertEqual(prepared_batch["metadata"], ["a", "b", "a", "b"])
        controlnet_args = model.controlnet.call_args
        self.assertEqual(controlnet_args.args[0].shape[0], 4)
        self.assertEqual(controlnet_args.kwargs["added_cond_kwargs"]["text_embeds"].shape[0], 4)
        self.assertEqual(controlnet_args.kwargs["controlnet_cond"].shape[0], 4)
        unet_kwargs = model.model.call_args.kwargs
        self.assertEqual(unet_kwargs["added_cond_kwargs"]["time_ids"].shape[0], 4)

    def test_sdxl_xm_loss_trims_winners(self):
        model = self._model(SDXL)
        self._enable_xm(model)
        noise = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32).view(4, 1, 1, 1)
        prepared_batch = {
            "latents": torch.zeros_like(noise),
            "noise": noise,
            "input_noise": noise,
            "noisy_latents": noise,
            "timesteps": torch.tensor([100.0, 200.0, 100.0, 200.0]),
            "metadata": [{"id": "c0s0"}, {"id": "c0s1"}, {"id": "c1s0"}, {"id": "c1s1"}],
            "xm_candidate_count": 2,
            "xm_original_batch_size": 2,
        }
        model_output = {
            "model_prediction": torch.tensor([5.0, 1.0, 2.0, -4.0], dtype=torch.float32).view(4, 1, 1, 1),
            "xm_candidate_count": 2,
        }

        loss, logs = model.loss_with_logs(prepared_batch, model_output)

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(model_output["xm_winner_indices"].tolist(), [1, 0])
        self.assertEqual(prepared_batch["metadata"], [{"id": "c1s0"}, {"id": "c0s1"}])
        self.assertNotIn("xm_candidate_count", prepared_batch)
        self.assertNotIn("xm_candidate_count", model_output)
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)

    def test_deepfloyd_xm_model_predict_expands_before_unet_forward(self):
        model = self._model(DeepFloydIF)
        self._enable_xm(model)
        model._apply_flowmap_r_timestep_kwargs = MagicMock()
        model.model = MagicMock(return_value=(torch.zeros(4, 2, 2, 2),))
        latents = torch.zeros(2, 1, 2, 2)
        prepared_batch = {
            "latents": latents,
            "noise": torch.zeros_like(latents),
            "input_noise": torch.zeros_like(latents),
            "noisy_latents": latents.clone(),
            "timesteps": torch.tensor([0.25, 0.75], dtype=torch.float32),
            "encoder_hidden_states": torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
            "metadata": ["a", "b"],
        }

        result = model.model_predict(prepared_batch)

        self.assertEqual(result["xm_candidate_count"], 2)
        self.assertEqual(result["model_prediction"].shape, (4, 1, 2, 2))
        self.assertEqual(model.model.call_args.args[0].shape[0], 4)
        self.assertEqual(model.model.call_args.kwargs["encoder_hidden_states"].shape[0], 4)
        self.assertEqual(prepared_batch["metadata"], ["a", "b", "a", "b"])

    def test_stable_cascade_xm_model_predict_expands_prior_conditioning(self):
        model = self._model(StableCascadeStageC)
        self._enable_xm(model)
        model.model = MagicMock(return_value=(torch.zeros(4, 1, 2, 2),))
        latents = torch.zeros(2, 1, 2, 2)
        prepared_batch = {
            "latents": latents,
            "noise": torch.zeros_like(latents),
            "input_noise": torch.zeros_like(latents),
            "noisy_latents": latents.clone(),
            "timesteps": torch.tensor([250.0, 750.0], dtype=torch.float32),
            "encoder_hidden_states": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
            "added_cond_kwargs": {"text_embeds": torch.arange(8, dtype=torch.float32).reshape(2, 4)},
            "cascade_clip_image_embeds": torch.arange(10, dtype=torch.float32).reshape(2, 1, 5),
            "metadata": ["a", "b"],
        }

        result = model.model_predict(prepared_batch)

        self.assertEqual(result["xm_candidate_count"], 2)
        prior_kwargs = model.model.call_args.kwargs
        self.assertEqual(prior_kwargs["sample"].shape[0], 4)
        self.assertEqual(prior_kwargs["clip_text"].shape[0], 4)
        self.assertEqual(prior_kwargs["clip_text_pooled"].shape, (4, 1, 4))
        self.assertEqual(prior_kwargs["clip_img"].shape[0], 4)
        self.assertEqual(prepared_batch["metadata"], ["a", "b", "a", "b"])


if __name__ == "__main__":
    unittest.main()
