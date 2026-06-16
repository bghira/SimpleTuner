import unittest

import torch
from diffusers import FlowMatchEulerDiscreteScheduler as DiffusersFlowMatchEulerDiscreteScheduler

from simpletuner.helpers.models.ace_step.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler as ACEStepFlowMatchEulerDiscreteScheduler,
)
from simpletuner.helpers.training.flow_match import fix_flow_match_euler_schedule_bounds


class TestFlowMatchSchedulerBounds(unittest.TestCase):
    def test_000_diffusers_static_shift_still_has_duplicate_shift_regression(self):
        scheduler = DiffusersFlowMatchEulerDiscreteScheduler(num_train_timesteps=10, shift=3.0)

        initial_sigmas = scheduler.sigmas.clone()
        scheduler.set_timesteps(num_inference_steps=10)

        self.assertFalse(torch.allclose(scheduler.sigmas[:-1].cpu(), initial_sigmas, atol=1e-6))
        self.assertGreater(scheduler.sigmas[-2].item(), initial_sigmas[-1].item())

    def test_diffusers_static_shift_bounds_are_unshifted_for_set_timesteps(self):
        scheduler = DiffusersFlowMatchEulerDiscreteScheduler(num_train_timesteps=10, shift=3.0)
        fix_flow_match_euler_schedule_bounds(scheduler)

        initial_sigmas = scheduler.sigmas.clone()
        scheduler.set_timesteps(num_inference_steps=10)

        self.assertAlmostEqual(scheduler.sigma_min, 0.1, places=6)
        self.assertAlmostEqual(scheduler.sigma_max, 1.0, places=6)
        self.assertTrue(torch.allclose(scheduler.sigmas[:-1].cpu(), initial_sigmas, atol=1e-6))

    def test_ace_step_static_shift_bounds_are_unshifted_for_set_timesteps(self):
        scheduler = ACEStepFlowMatchEulerDiscreteScheduler(num_train_timesteps=10, shift=3.0)

        initial_sigmas = scheduler.sigmas.clone()
        scheduler.set_timesteps(num_inference_steps=10)

        self.assertAlmostEqual(scheduler.sigma_min, 0.1, places=6)
        self.assertAlmostEqual(scheduler.sigma_max, 1.0, places=6)
        self.assertTrue(torch.allclose(scheduler.sigmas[:-1].cpu(), initial_sigmas, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
