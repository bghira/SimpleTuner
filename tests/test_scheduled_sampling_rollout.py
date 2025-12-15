import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers import DDPMScheduler
from torch.testing import assert_close

from simpletuner.diff2flow.bridge import DiffusionToFlowBridge
from simpletuner.helpers.models.common import ModelFoundation, PredictionTypes
from simpletuner.helpers.scheduled_sampling.plan import ScheduledSamplingPlan, build_rollout_schedule
from simpletuner.helpers.scheduled_sampling.rollout import apply_scheduled_sampling_rollout


class _DummySampler:
    require_previous = 0

    def sample(self, current, x_pred, *, step, schedule, sigma_transform, noise, previous):
        # Return the provided x_pred as the next state; sufficient to validate rollout wiring.
        return types.SimpleNamespace(final=x_pred)


class _DummyModel:
    PREDICTION_TYPE = PredictionTypes.EPSILON

    def model_predict(self, prepared_batch):
        # Provide a small correction so rollout actually moves the state.
        return torch.full_like(prepared_batch["noisy_latents"], 0.2)


class _DummyFlowScheduler:
    def __init__(self, num_train_timesteps: int = 4):
        self.config = SimpleNamespace(num_train_timesteps=num_train_timesteps)
        self.sigmas = torch.linspace(0.0, 1.0, num_train_timesteps)

    def add_noise(self, latents, noise, timesteps):
        sigma = self.sigmas[timesteps.long()].to(device=latents.device, dtype=latents.dtype)
        while sigma.dim() < latents.dim():
            sigma = sigma.view(-1, *([1] * (latents.dim() - 1)))
        return (1 - sigma) * latents + sigma * noise


class _DummyFlowModel:
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING

    def __init__(self, value: float = 0.1):
        self.value = value

    def model_predict(self, prepared_batch):
        return torch.full_like(prepared_batch["noisy_latents"], self.value)


class ScheduledSamplingRolloutTests(unittest.TestCase):
    def test_diff2flow_bridge_matches_flow_target(self):
        scheduler = DDPMScheduler(num_train_timesteps=8)
        bridge = DiffusionToFlowBridge(alphas_cumprod=scheduler.alphas_cumprod)

        latents = torch.full((1, 4, 2, 2), 0.3, dtype=torch.float32)
        noise = torch.full_like(latents, 0.7)
        timesteps = torch.tensor([3], dtype=torch.long)

        noisy = scheduler.add_noise(latents, noise, timesteps)

        eps_flow = bridge.prediction_to_flow(noise, noisy, timesteps, prediction_type="epsilon")
        assert_close(eps_flow, noise - latents, atol=1e-5, rtol=1e-5)

        v_pred = scheduler.get_velocity(latents, noise, timesteps)
        v_flow = bridge.prediction_to_flow(v_pred, noisy, timesteps, prediction_type="v_prediction")
        assert_close(v_flow, noise - latents, atol=1e-5, rtol=1e-5)

    def test_rollout_schedule_respects_base_timesteps(self):
        torch.manual_seed(0)
        base_ts = torch.tensor([3, 1, 0], dtype=torch.long)
        plan = build_rollout_schedule(
            num_train_timesteps=10,
            batch_size=3,
            max_step_offset=2,
            device="cpu",
            base_timesteps=base_ts,
            strategy="uniform",
            apply_probability=1.0,
        )

        self.assertTrue(torch.equal(plan.target_timesteps, base_ts))
        self.assertTrue(torch.all(plan.source_timesteps >= base_ts))
        self.assertTrue(torch.all(plan.source_timesteps <= base_ts + 2))
        self.assertTrue(torch.equal(plan.rollout_steps, plan.source_timesteps - base_ts))

    @patch("simpletuner.helpers.scheduled_sampling.rollout.make_sampler", return_value=_DummySampler())
    def test_apply_scheduled_sampling_rollout_updates_noisy_latents(self, _sampler_ctor):
        scheduler = DDPMScheduler(num_train_timesteps=6)
        latents = torch.zeros((2, 1, 1, 1), dtype=torch.float32)
        noise = torch.full_like(latents, 0.5)
        target_timesteps = torch.tensor([1, 1], dtype=torch.long)
        source_timesteps = torch.tensor([2, 1], dtype=torch.long)
        rollout_steps = torch.tensor([1, 0], dtype=torch.long)

        initial_noisy = scheduler.add_noise(latents, noise, target_timesteps)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "input_noise": noise,
            "noisy_latents": initial_noisy.clone(),
            "timesteps": target_timesteps.clone(),
            "scheduled_sampling_plan": ScheduledSamplingPlan(
                target_timesteps=target_timesteps,
                source_timesteps=source_timesteps,
                rollout_steps=rollout_steps,
            ),
        }

        model = _DummyModel()
        config = SimpleNamespace(controlnet=False)

        updated = apply_scheduled_sampling_rollout(model, prepared_batch, scheduler, config)

        # Item 0 should be advanced from source(2)->target(1) with our dummy sampler/transform.
        source_noisy = scheduler.add_noise(latents[0:1], noise[0:1], torch.tensor([2], dtype=torch.long))
        alpha_bar = scheduler.alphas_cumprod[2]
        sigma_u = torch.sqrt(1.0 - alpha_bar)
        sigma_v = torch.sqrt(alpha_bar)
        expected_noisy_0 = (source_noisy - sigma_u * 0.2) / sigma_v
        assert_close(updated["noisy_latents"][0], expected_noisy_0.squeeze(0), atol=1e-5, rtol=1e-5)
        self.assertEqual(updated["timesteps"][0].item(), 1)

        # Item 1 had no rollout; it should be unchanged.
        assert_close(updated["noisy_latents"][1], initial_noisy[1], atol=1e-5, rtol=1e-5)
        self.assertEqual(updated["timesteps"][1].item(), target_timesteps[1].item())

    @patch("simpletuner.helpers.scheduled_sampling.rollout.make_sampler", return_value=_DummySampler())
    def test_apply_scheduled_sampling_rollout_mutates_batch_in_place(self, _sampler_ctor):
        scheduler = DDPMScheduler(num_train_timesteps=6)
        latents = torch.zeros((2, 1, 1, 1), dtype=torch.float32)
        noise = torch.full_like(latents, 0.5)
        target_timesteps = torch.tensor([1, 1], dtype=torch.long)
        source_timesteps = torch.tensor([2, 1], dtype=torch.long)
        rollout_steps = torch.tensor([1, 0], dtype=torch.long)

        initial_noisy = scheduler.add_noise(latents, noise, target_timesteps)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "input_noise": noise,
            "noisy_latents": initial_noisy.clone(),
            "timesteps": target_timesteps.clone(),
            "scheduled_sampling_plan": ScheduledSamplingPlan(
                target_timesteps=target_timesteps,
                source_timesteps=source_timesteps,
                rollout_steps=rollout_steps,
            ),
        }

        original_noisy = prepared_batch["noisy_latents"].clone()

        model = _DummyModel()
        config = SimpleNamespace(controlnet=False)

        updated = apply_scheduled_sampling_rollout(model, prepared_batch, scheduler, config)

        self.assertIs(updated, prepared_batch)
        # First item should change due to rollout
        self.assertFalse(torch.equal(prepared_batch["noisy_latents"][0], original_noisy[0]))
        # Second item should remain the same
        assert_close(prepared_batch["noisy_latents"][1], original_noisy[1], atol=1e-5, rtol=1e-5)
        self.assertEqual(prepared_batch["timesteps"][1].item(), target_timesteps[1].item())

    def test_flow_matching_rollout_updates_sigmas_and_caches(self):
        scheduler = _DummyFlowScheduler(num_train_timesteps=4)
        latents = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
        noise = torch.ones_like(latents)
        target_timesteps = torch.tensor([1], dtype=torch.long)
        source_timesteps = torch.tensor([3], dtype=torch.long)
        rollout_steps = torch.tensor([2], dtype=torch.long)

        initial_noisy = scheduler.add_noise(latents, noise, target_timesteps)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "input_noise": noise,
            "sigmas": scheduler.sigmas[target_timesteps].clone(),
            "noisy_latents": initial_noisy.clone(),
            "timesteps": target_timesteps.clone().float(),
            "scheduled_sampling_plan": ScheduledSamplingPlan(
                target_timesteps=target_timesteps,
                source_timesteps=source_timesteps,
                rollout_steps=rollout_steps,
            ),
        }

        model = _DummyFlowModel(value=0.1)
        config = SimpleNamespace(controlnet=False, scheduled_sampling_reflexflow=True)

        updated = apply_scheduled_sampling_rollout(model, prepared_batch, scheduler, config)

        expected = torch.tensor([0.9333], dtype=torch.float32)
        assert_close(updated["noisy_latents"].flatten(), expected, atol=1e-3, rtol=1e-3)
        self.assertEqual(updated["timesteps"][0].item(), 1)
        assert_close(updated["sigmas"][0], scheduler.sigmas[1], atol=1e-6, rtol=1e-6)
        # ReflexFlow caches should be populated
        assert "_reflexflow_clean_pred" in updated
        assert "_reflexflow_biased_pred" in updated


class ReflexFlowDefaultToggleTests(unittest.TestCase):
    def _make_model(self, prediction_type, config):
        class _StubModel(ModelFoundation):
            PREDICTION_TYPE = prediction_type

            def __init__(self, cfg):
                self.config = cfg

            def model_predict(self, prepared_batch, custom_timesteps: list | None = None):
                return prepared_batch

            def _encode_prompts(self, *args, **kwargs):
                return None

            def convert_text_embed_for_pipeline(self, text_encoder_output, pooling_encode_output=None):
                return text_encoder_output

            def convert_negative_text_embed_for_pipeline(self, text_encoder_output, pooling_encode_output=None):
                return text_encoder_output

        return _StubModel(config)

    def test_auto_enables_when_unset_for_flow_matching(self):
        config = SimpleNamespace(scheduled_sampling_max_step_offset=3, scheduled_sampling_reflexflow=None)
        model = self._make_model(PredictionTypes.FLOW_MATCHING, config)

        changed = model._maybe_enable_reflexflow_default()

        self.assertTrue(changed)
        self.assertTrue(config.scheduled_sampling_reflexflow)

    def test_respects_user_opt_out(self):
        config = SimpleNamespace(scheduled_sampling_max_step_offset=3, scheduled_sampling_reflexflow=False)
        model = self._make_model(PredictionTypes.FLOW_MATCHING, config)

        changed = model._maybe_enable_reflexflow_default()

        self.assertFalse(changed)
        self.assertFalse(config.scheduled_sampling_reflexflow)

    def test_skips_non_flow_matching_models(self):
        config = SimpleNamespace(scheduled_sampling_max_step_offset=3, scheduled_sampling_reflexflow=None)
        model = self._make_model(PredictionTypes.EPSILON, config)

        changed = model._maybe_enable_reflexflow_default()

        self.assertFalse(changed)
        self.assertIsNone(config.scheduled_sampling_reflexflow)


class _FlowLossModel(ModelFoundation):
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    NAME = "dummy"
    TEXT_ENCODER_CONFIGURATION = []

    def __init__(self):
        self.config = SimpleNamespace(
            loss_type="l2",
            snr_weight=1.0,
            snr_gamma=None,
            masked_loss_probability=0.0,
            flow_matching=True,
            scheduled_sampling_reflexflow=True,
            scheduled_sampling_reflexflow_alpha=1.0,
            scheduled_sampling_reflexflow_beta1=10.0,
            scheduled_sampling_reflexflow_beta2=1.0,
        )
        self.noise_schedule = SimpleNamespace(config=SimpleNamespace(prediction_type="flow_matching"))
        self.diff2flow_bridge = None

    # Satisfy abstract methods not relevant for this unit test.
    def model_predict(self, prepared_batch, custom_timesteps: list | None = None):
        return prepared_batch

    def _encode_prompts(self, *args, **kwargs):
        return None, None, None, None

    def convert_text_embed_for_pipeline(self, text_encoder_output, pooling_encode_output=None):
        return text_encoder_output

    def convert_negative_text_embed_for_pipeline(self, text_encoder_output, pooling_encode_output=None):
        return text_encoder_output


class ReflexFlowLossTests(unittest.TestCase):
    def test_reflexflow_weighting_and_adr(self):
        model = _FlowLossModel()
        latents = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
        noise = torch.ones_like(latents)
        prepared_batch = {
            "latents": latents,
            "noise": noise,
            "noisy_latents": torch.full_like(latents, 0.3),
            "timesteps": torch.tensor([1.0]),
            "_reflexflow_clean_pred": torch.full_like(latents, 0.6),
            "_reflexflow_biased_pred": torch.full_like(latents, 0.4),
        }
        model_output = {"model_prediction": torch.full_like(latents, 0.7)}

        loss = model.loss(prepared_batch, model_output, apply_conditioning_mask=False)
        # Base loss (0.09) is doubled by FC weighting; ADR term is zero when aligned with the noise-facing flow vector.
        self.assertAlmostEqual(loss.item(), 0.18, places=4)


if __name__ == "__main__":
    unittest.main()
