import unittest
from types import SimpleNamespace

import torch

import tests.test_stubs  # noqa: F401
from simpletuner.helpers.models.common import ImageModelFoundation


def _flow_model(custom_timesteps: str, mode: str):
    model = SimpleNamespace(
        config=SimpleNamespace(
            flow_custom_timesteps=custom_timesteps,
            flow_timesteps_mode=mode,
        ),
        accelerator=SimpleNamespace(device=torch.device("cpu")),
    )
    model._normalize_flow_custom_timesteps = ImageModelFoundation._normalize_flow_custom_timesteps.__get__(model)
    model.sample_flow_sigmas = ImageModelFoundation.sample_flow_sigmas.__get__(model)
    return model


class FlowCustomTimestepsTests(unittest.TestCase):
    def test_round_robin_cycles_custom_timesteps(self):
        model = _flow_model("100,200,300", "round-robin")
        batch = {"latents": torch.zeros(2, 1, 2, 2)}

        _, first_timesteps = model.sample_flow_sigmas(batch=batch, state={})
        _, second_timesteps = model.sample_flow_sigmas(batch=batch, state={})

        self.assertTrue(torch.equal(first_timesteps, torch.tensor([100.0, 200.0])))
        self.assertTrue(torch.equal(second_timesteps, torch.tensor([300.0, 100.0])))

    def test_invalid_custom_timestep_mode_raises(self):
        model = _flow_model("100,200", "sequential")
        batch = {"latents": torch.zeros(1, 1, 2, 2)}

        with self.assertRaisesRegex(ValueError, "flow_timesteps_mode"):
            model.sample_flow_sigmas(batch=batch, state={})


if __name__ == "__main__":
    unittest.main()
