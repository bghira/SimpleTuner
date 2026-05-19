import unittest
from tempfile import TemporaryDirectory
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
        accelerator=SimpleNamespace(device=torch.device("cpu"), num_processes=1, process_index=0),
    )
    model._normalize_flow_custom_timesteps = ImageModelFoundation._normalize_flow_custom_timesteps.__get__(model)
    model.reset_flow_custom_timestep_cursor = ImageModelFoundation.reset_flow_custom_timestep_cursor.__get__(model)
    model._flow_custom_timestep_state_path = ImageModelFoundation._flow_custom_timestep_state_path.__get__(model)
    model.save_flow_custom_timestep_state = ImageModelFoundation.save_flow_custom_timestep_state.__get__(model)
    model.load_flow_custom_timestep_state = ImageModelFoundation.load_flow_custom_timestep_state.__get__(model)
    model.sample_flow_sigmas = ImageModelFoundation.sample_flow_sigmas.__get__(model)
    model.prepare_batch_conditions = ImageModelFoundation.prepare_batch_conditions.__get__(model)
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

    def test_round_robin_offsets_distributed_ranks(self):
        rank0 = _flow_model("100,200,300,400,500", "round-robin")
        rank1 = _flow_model("100,200,300,400,500", "round-robin")
        rank0.accelerator.num_processes = 2
        rank1.accelerator.num_processes = 2
        rank1.accelerator.process_index = 1
        batch = {"latents": torch.zeros(2, 1, 2, 2)}

        _, rank0_timesteps = rank0.sample_flow_sigmas(batch=batch, state={"global_step": 0})
        _, rank1_timesteps = rank1.sample_flow_sigmas(batch=batch, state={"global_step": 0})
        _, rank0_next = rank0.sample_flow_sigmas(batch=batch, state={"global_step": 0})

        self.assertTrue(torch.equal(rank0_timesteps, torch.tensor([100.0, 200.0])))
        self.assertTrue(torch.equal(rank1_timesteps, torch.tensor([300.0, 400.0])))
        self.assertTrue(torch.equal(rank0_next, torch.tensor([500.0, 100.0])))

    def test_round_robin_initializes_from_resume_step(self):
        model = _flow_model("100,200,300,400,500", "round-robin")
        model.accelerator.num_processes = 2
        batch = {"latents": torch.zeros(2, 1, 2, 2)}

        _, timesteps = model.sample_flow_sigmas(batch=batch, state={"global_step": 1})

        self.assertTrue(torch.equal(timesteps, torch.tensor([500.0, 100.0])))

    def test_round_robin_resume_reset_overrides_prior_cursor(self):
        model = _flow_model("100,200,300,400,500", "round-robin")
        model.accelerator.num_processes = 2
        batch = {"latents": torch.zeros(2, 1, 2, 2)}

        model.sample_flow_sigmas(batch=batch, state={"global_step": 0})
        model.reset_flow_custom_timestep_cursor(global_step=1)
        _, timesteps = model.sample_flow_sigmas(batch=batch, state={"global_step": 1})

        self.assertTrue(torch.equal(timesteps, torch.tensor([500.0, 100.0])))

    def test_round_robin_checkpoint_restores_microbatch_cursor(self):
        model = _flow_model("100,200,300,400", "round-robin")
        batch = {"latents": torch.zeros(1, 1, 2, 2)}

        model.sample_flow_sigmas(batch=batch, state={"global_step": 0})
        model.sample_flow_sigmas(batch=batch, state={"global_step": 0})
        model.sample_flow_sigmas(batch=batch, state={"global_step": 0})

        with TemporaryDirectory() as tmpdir:
            model.save_flow_custom_timestep_state(tmpdir)
            resumed = _flow_model("100,200,300,400", "round-robin")
            loaded = resumed.load_flow_custom_timestep_state(tmpdir, fallback_global_step=1)
            _, timesteps = resumed.sample_flow_sigmas(batch=batch, state={"global_step": 1})

        self.assertTrue(loaded)
        self.assertTrue(torch.equal(timesteps, torch.tensor([400.0])))

    def test_round_robin_checkpoint_load_falls_back_to_global_step(self):
        model = _flow_model("100,200,300,400", "round-robin")
        batch = {"latents": torch.zeros(1, 1, 2, 2)}

        with TemporaryDirectory() as tmpdir:
            loaded = model.load_flow_custom_timestep_state(tmpdir, fallback_global_step=1)
            _, timesteps = model.sample_flow_sigmas(batch=batch, state={"global_step": 1})

        self.assertFalse(loaded)
        self.assertTrue(torch.equal(timesteps, torch.tensor([200.0])))

    def test_prepare_batch_conditions_selects_reference_strict_latents(self):
        model = _flow_model("100", "fixed-list")
        reference_latents = torch.ones(1, 1, 2, 2)
        loose_latents = torch.zeros_like(reference_latents)
        batch = {
            "conditioning_type": "reference_strict",
            "conditioning_latents": [loose_latents, reference_latents],
            "conditioning_latents_type": ["reference_loose", "reference_strict"],
        }

        result = model.prepare_batch_conditions(batch=batch, state={})

        self.assertTrue(torch.equal(result["conditioning_latents"], reference_latents))
        self.assertEqual(result["conditioning_latents_type"], "reference_strict")


if __name__ == "__main__":
    unittest.main()
