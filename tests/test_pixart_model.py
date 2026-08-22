import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from simpletuner.helpers.models.common import PredictionTypes
from simpletuner.helpers.models.pixart.model import PixartSigma
from simpletuner.helpers.training.explorative_modeling import ExplorativeModelingConfig
from simpletuner.helpers.training.grounding.types import GroundingBatch


class _NoiseSchedule:
    def add_noise(self, latents, noise, timesteps):
        timestep_grid = timesteps.reshape(timesteps.shape[0], *([1] * (latents.ndim - 1))).to(
            device=latents.device,
            dtype=latents.dtype,
        )
        return latents + timestep_grid * noise


class PixArtModelTests(unittest.TestCase):
    def setUp(self):
        self.model = PixartSigma.__new__(PixartSigma)
        self.model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.model.config = SimpleNamespace(
            base_weight_dtype=torch.float32,
            loss_type="l2",
            twinflow_enabled=False,
            scheduled_sampling_reflexflow=False,
            scheduled_sampling_max_step_offset=0,
            input_perturbation=0.0,
        )
        self.model.LATENT_CHANNEL_COUNT = 4
        self.model.NAME = "PixArt"
        self.model.PREDICTION_TYPE = PredictionTypes.EPSILON
        self.model.noise_schedule = _NoiseSchedule()
        self.model._new_hidden_state_buffer = MagicMock(return_value={})
        self.model.crepa_regularizer = SimpleNamespace(enabled=True, block_index=2)
        self.model.unwrap_model = MagicMock(side_effect=lambda model=None, **kwargs: model)

    def _enable_xm(
        self,
        candidate_count: int = 2,
        training_target: str = "noise",
        selection_scope: str = "sample",
    ):
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

        def _forward(*args, **kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            kwargs["hidden_states_buffer"]["layer_7"] = torch.full((1, 4, 8), 7.0)
            return (torch.randn(1, 8, 4, 4),)

        self.model.model = MagicMock(side_effect=_forward)
        self.model.model.config = SimpleNamespace(patch_size=2)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 4, 4, 4),
            "timesteps": torch.tensor([400], dtype=torch.int64),
            "encoder_hidden_states": torch.randn(1, 4, 16),
            "encoder_attention_mask": torch.ones(1, 4),
            "crepa_capture_block_index": 7,
            "resolution": torch.tensor([[4.0, 4.0]], dtype=torch.float32),
            "aspect_ratio": torch.tensor([[1.0]], dtype=torch.float32),
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 4, 4, 4))
        self.assertTrue(torch.equal(result["crepa_hidden_states"], torch.full((1, 4, 8), 7.0)))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], torch.tensor([400], dtype=torch.int64)))
        self.assertTrue(
            torch.equal(
                transformer_kwargs["added_cond_kwargs"]["resolution"],
                prepared_batch["resolution"],
            )
        )

    def test_model_predict_accepts_tokenwise_timesteps(self):
        hidden_states_buffer = {}
        self.model._new_hidden_state_buffer = MagicMock(return_value=hidden_states_buffer)

        def _forward(*args, **kwargs):
            kwargs["hidden_states_buffer"]["layer_2"] = torch.full((1, 4, 8), 2.0)
            return (torch.randn(1, 8, 4, 4),)

        self.model.model = MagicMock(side_effect=_forward)
        self.model.model.config = SimpleNamespace(patch_size=2)

        prepared_batch = {
            "noisy_latents": torch.randn(1, 4, 4, 4),
            "timesteps": torch.tensor([[100, 900, 200, 800]], dtype=torch.int64),
            "encoder_hidden_states": torch.randn(1, 4, 16),
            "encoder_attention_mask": torch.ones(1, 4),
            "resolution": torch.tensor([[4.0, 4.0]], dtype=torch.float32),
            "aspect_ratio": torch.tensor([[1.0]], dtype=torch.float32),
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 4, 4, 4))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertTrue(torch.equal(transformer_kwargs["timestep"], prepared_batch["timesteps"]))

    def test_prepare_crepa_self_flow_batch_creates_tokenwise_timesteps(self):
        transformer = SimpleNamespace(config=SimpleNamespace(patch_size=2))
        self.model.model = transformer
        self.model.unwrap_model = MagicMock(return_value=transformer)
        self.model.sample_flow_sigmas = MagicMock(
            return_value=(
                torch.tensor([0.9], dtype=torch.float32),
                torch.tensor([900.0], dtype=torch.float32),
            )
        )
        self.model.config.crepa_self_flow_mask_ratio = 1.0

        batch = {
            "latents": torch.zeros(1, 4, 4, 4, dtype=torch.float32),
            "input_noise": torch.ones(1, 4, 4, 4, dtype=torch.float32),
            "sigmas": torch.tensor([0.1], dtype=torch.float32),
            "timesteps": torch.tensor([100.0], dtype=torch.float32),
        }

        result = self.model._prepare_crepa_self_flow_batch(batch, state={})

        self.assertEqual(result["timesteps"].shape, (1, 4))
        self.assertTrue(
            torch.equal(
                result["crepa_teacher_timesteps"],
                torch.tensor([100.0], dtype=torch.float32),
            )
        )

    def test_xm_noise_candidates_expand_candidate_major(self):
        torch.manual_seed(1)
        self._enable_xm(candidate_count=3)
        latents = torch.arange(32, dtype=torch.float32).reshape(2, 4, 2, 2)
        grounding_batch = GroundingBatch(
            boxes=torch.arange(16, dtype=torch.float32).reshape(2, 2, 4),
            validity_mask=torch.ones(2, 2),
            spatial_masks=torch.ones(2, 2, 2, 2),
            text_embeds=torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
            image_embeds=torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
            text_masks=torch.ones(2, 2),
            image_masks=torch.ones(2, 2),
            max_entities=2,
        )
        batch = {
            "latents": latents.clone(),
            "noise": torch.zeros_like(latents),
            "input_noise": torch.zeros_like(latents),
            "noisy_latents": latents.clone(),
            "timesteps": torch.tensor([0.25, 0.75], dtype=torch.float32),
            "encoder_attention_mask": torch.ones(2, 4),
            "encoder_hidden_states": torch.arange(16, dtype=torch.float32).reshape(2, 4, 2),
            "resolution": torch.tensor([[4.0, 4.0], [8.0, 4.0]], dtype=torch.float32),
            "aspect_ratio": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
            "flowmap_r_timesteps": torch.tensor([0.1, 0.2], dtype=torch.float32),
            "conditioning_latents": torch.arange(32, dtype=torch.float32).reshape(2, 4, 2, 2),
            "grounding_batch": grounding_batch,
        }

        self.model._prepare_xm_noise_candidates(batch, family_name="PixArt")

        self.assertEqual(tuple(batch["latents"].shape), (6, 4, 2, 2))
        self.assertTrue(torch.equal(batch["latents"][:2], latents))
        self.assertTrue(torch.equal(batch["latents"][2:4], latents))
        self.assertTrue(torch.equal(batch["timesteps"], torch.tensor([0.25, 0.75, 0.25, 0.75, 0.25, 0.75])))
        self.assertTrue(torch.equal(batch["encoder_hidden_states"][:2], batch["encoder_hidden_states"][2:4]))
        self.assertTrue(torch.equal(batch["resolution"][:2], batch["resolution"][2:4]))
        self.assertTrue(torch.equal(batch["aspect_ratio"][:2], batch["aspect_ratio"][2:4]))
        self.assertTrue(
            torch.equal(
                batch["flowmap_r_timesteps"],
                torch.tensor([0.1, 0.2, 0.1, 0.2, 0.1, 0.2]),
            )
        )
        self.assertTrue(torch.equal(batch["conditioning_latents"][:2], batch["conditioning_latents"][2:4]))
        self.assertTrue(torch.equal(batch["grounding_batch"].boxes[:2], grounding_batch.boxes))
        self.assertTrue(torch.equal(batch["grounding_batch"].boxes[2:4], grounding_batch.boxes))
        self.assertFalse(torch.equal(batch["noise"][:2], batch["noise"][2:4]))
        self.assertTrue(
            torch.allclose(
                batch["noisy_latents"],
                batch["latents"] + batch["timesteps"].view(6, 1, 1, 1) * batch["noise"],
            )
        )
        self.assertEqual(batch["xm_candidate_count"], 3)

    def test_model_predict_returns_xm_candidate_count(self):
        self._enable_xm(candidate_count=2)

        def _forward(*args, **kwargs):
            return (torch.randn(4, 8, 2, 2),)

        self.model.model = MagicMock(side_effect=_forward)
        self.model.model.config = SimpleNamespace(patch_size=1)
        prepared_batch = {
            "latents": torch.zeros(2, 4, 2, 2),
            "noise": torch.zeros(2, 4, 2, 2),
            "input_noise": torch.zeros(2, 4, 2, 2),
            "noisy_latents": torch.zeros(2, 4, 2, 2),
            "timesteps": torch.tensor([0.25, 0.75], dtype=torch.float32),
            "encoder_hidden_states": torch.randn(2, 4, 16),
            "encoder_attention_mask": torch.ones(2, 4),
            "resolution": torch.tensor([[4.0, 4.0], [8.0, 4.0]], dtype=torch.float32),
            "aspect_ratio": torch.tensor([[1.0], [2.0]], dtype=torch.float32),
        }

        result = self.model.model_predict(prepared_batch)

        self.assertEqual(result["xm_candidate_count"], 2)
        self.assertEqual(tuple(result["model_prediction"].shape), (4, 4, 2, 2))
        transformer_kwargs = self.model.model.call_args.kwargs
        self.assertEqual(tuple(transformer_kwargs["encoder_hidden_states"].shape), (4, 4, 16))
        self.assertEqual(tuple(transformer_kwargs["added_cond_kwargs"]["resolution"].shape), (4, 2))

    def test_xm_loss_selects_winners_and_trims_hidden_states(self):
        self._enable_xm(candidate_count=2)
        noise = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(4, 1, 1, 1)
        prepared_batch = {
            "latents": torch.zeros(4, 1, 1, 1),
            "noise": noise,
            "noisy_latents": noise,
            "timesteps": torch.tensor([100.0, 200.0, 100.0, 200.0]),
            "xm_candidate_count": 2,
            "xm_original_batch_size": 2,
        }
        prediction = torch.tensor([5.0, 1.0, 2.0, -4.0]).view(4, 1, 1, 1)
        hidden = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
        model_output = {
            "model_prediction": prediction,
            "hidden_states_buffer": {"layer_0": hidden.clone()},
            "xm_candidate_count": 2,
        }

        loss, logs = self.model.loss_with_logs(prepared_batch, model_output)

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertTrue(torch.equal(model_output["xm_winner_indices"], torch.tensor([1, 0])))
        self.assertTrue(torch.equal(model_output["model_prediction"], noise[[2, 1]]))
        self.assertTrue(torch.equal(prepared_batch["noise"], noise[[2, 1]]))
        self.assertTrue(torch.equal(model_output["hidden_states_buffer"]["layer_0"], hidden[[2, 1]]))
        self.assertNotIn("xm_candidate_count", model_output)
        self.assertNotIn("xm_candidate_count", prepared_batch)
        self.assertEqual(logs["xm_candidate_0_wins"], 1.0)
        self.assertEqual(logs["xm_candidate_1_wins"], 1.0)

    def test_xm_rejects_route_and_block_selection(self):
        self._enable_xm(training_target="route")
        with self.assertRaisesRegex(ValueError, "xm_training_target='noise'"):
            self.model._xm_noise_candidates_enabled()

        self._enable_xm(selection_scope="block")
        with self.assertRaisesRegex(ValueError, "xm_selection_scope='sample'"):
            self.model._xm_noise_candidates_enabled()

    def test_controlnet_predict_passes_added_cond_kwargs(self):
        self.model.controlnet = MagicMock(return_value=(torch.randn(1, 4, 4, 4),))
        transformer = SimpleNamespace(config=SimpleNamespace(patch_size=2))
        self.model.model = transformer
        self.model.unwrap_model = MagicMock(return_value=transformer)
        self.model.config.weight_dtype = torch.float32
        prepared_batch = {
            "noisy_latents": torch.randn(1, 4, 4, 4),
            "timesteps": torch.tensor([400], dtype=torch.int64),
            "encoder_hidden_states": torch.randn(1, 4, 16),
            "encoder_attention_mask": torch.ones(1, 4),
            "conditioning_latents": torch.randn(1, 4, 4, 4),
            "resolution": torch.tensor([[4.0, 4.0]], dtype=torch.float32),
            "aspect_ratio": torch.tensor([[1.0]], dtype=torch.float32),
        }

        result = self.model.controlnet_predict(prepared_batch)

        self.assertEqual(result["model_prediction"].shape, (1, 4, 4, 4))
        kwargs = self.model.controlnet.call_args.kwargs
        self.assertTrue(torch.equal(kwargs["timestep"], torch.tensor([400], dtype=torch.int64)))
        self.assertTrue(torch.equal(kwargs["added_cond_kwargs"]["resolution"], prepared_batch["resolution"]))


if __name__ == "__main__":
    unittest.main()
