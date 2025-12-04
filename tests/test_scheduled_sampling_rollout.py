import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers import DDPMScheduler
from torch.testing import assert_close

from simpletuner.diff2flow.bridge import DiffusionToFlowBridge
from simpletuner.helpers.models.common import PredictionTypes
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


if __name__ == "__main__":
    unittest.main()
