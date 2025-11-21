import unittest

import torch

from simpletuner.helpers.models.ace_step.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)


class TestACEStepScheduler(unittest.TestCase):
    def test_step_basic(self):
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=10, shift=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        sample = torch.randn(1, 4, 16, 16)
        model_output = torch.randn(1, 4, 16, 16)
        timestep = scheduler.timesteps[0]

        # Run step
        output = scheduler.step(model_output, timestep, sample, omega=1.0)

        self.assertEqual(output.prev_sample.shape, sample.shape)
        self.assertFalse(torch.isnan(output.prev_sample).any())

    def test_omega_logic(self):
        # Test that omega modifies the output
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=10, shift=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        sample = torch.zeros(1, 4, 16, 16)
        model_output = torch.randn(1, 4, 16, 16)
        timestep = scheduler.timesteps[0]

        # With omega=1.0 (logistic function will change it slightly)
        # But let's just check it runs and produces something.
        out1 = scheduler.step(model_output, timestep, sample, omega=0.5)

        # Reset step index?
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=10, shift=1.0)
        scheduler.set_timesteps(num_inference_steps=10)

        out2 = scheduler.step(model_output, timestep, sample, omega=-0.5)

        self.assertFalse(torch.allclose(out1.prev_sample, out2.prev_sample))


if __name__ == "__main__":
    unittest.main()
