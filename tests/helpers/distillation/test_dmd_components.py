import types
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from simpletuner.helpers.distillation.dmd.distiller import DMDDistiller
from simpletuner.helpers.distillation.self_forcing.pipeline import SelfForcingTrainingPipeline
from simpletuner.helpers.distillation.self_forcing.scheduler import FlowMatchingSchedulerAdapter
from simpletuner.helpers.models.common import PredictionTypes


class _StubScheduler:
    def __init__(self):
        self.sigmas = torch.linspace(1.0, 0.0, steps=1000)
        self.timesteps = torch.arange(1000)
        self.config = SimpleNamespace(num_train_timesteps=1000)


class SchedulerAdapterTests(unittest.TestCase):
    def test_add_noise_and_conversions(self):
        scheduler = FlowMatchingSchedulerAdapter(_StubScheduler())
        clean = torch.zeros(1, 4, 4, 4)
        noise = torch.ones_like(clean)
        timestep = torch.tensor([10])

        noisy = scheduler.add_noise(clean, noise, timestep)
        sigma = scheduler.sigmas[10]
        expected = (1.0 - sigma) * clean + sigma * noise
        torch.testing.assert_close(noisy, expected)

        flow_pred = torch.ones_like(clean) * 0.25
        recon = scheduler.convert_flow_to_x0(flow_pred, noisy, timestep)
        # Convert back to noise and ensure round trip.
        round_trip = scheduler.convert_x0_to_noise(recon, noisy, timestep)
        torch.testing.assert_close(round_trip, flow_pred, atol=1e-4, rtol=1e-4)


class _EchoGenerator:
    def forward(self, noisy_latents, timesteps, conditional_dict):
        flow = torch.zeros_like(noisy_latents)
        pred = noisy_latents - 0.1
        return flow, pred


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = FlowMatchingSchedulerAdapter(_StubScheduler())
        self.generator = _EchoGenerator()
        self.pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=[900, 600, 300],
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=1,
            same_step_across_blocks=False,
        )

    def test_inference_with_trajectory_runs(self):
        noise = torch.zeros(1, 2, 3, 4, 4)
        prompt_embeds = torch.zeros(1, 1, dtype=torch.float32)
        output, denoised_from, denoised_to = self.pipeline.inference_with_trajectory(
            noise=noise,
            conditional_dict={"prompt_embeds": prompt_embeds},
        )

        self.assertEqual(output.shape, noise.shape)
        self.assertIsNotNone(denoised_from)
        self.assertIsNotNone(denoised_to)
        self.assertTrue(torch.all(torch.isfinite(output)))
        self.assertFalse(torch.allclose(output, noise))


class _StubTransformer(nn.Module):
    def __init__(self, **_kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(to_dict=lambda: {})

    def forward(self, hidden_states, timestep, encoder_hidden_states, **_kwargs):
        return (torch.zeros_like(hidden_states),)


class _StubTeacher:
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING

    def __init__(self):
        self.config = SimpleNamespace(weight_dtype=torch.float32)
        self.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self._component = _StubTransformer()

    def get_trained_component(self, *args, **kwargs):
        return self._component

    def model_predict(self, prepared_batch):
        return {"model_prediction": torch.zeros_like(prepared_batch["noisy_latents"])}


class _ConstantWrapper:
    def __init__(self, scale: float):
        self.scale = scale

    def forward(self, noisy_latents, timesteps, conditional_dict):
        return torch.zeros_like(noisy_latents), noisy_latents * self.scale


class _TrainableWrapper:
    def __init__(self, module: nn.Module, scale: float):
        self.module = module
        self.scale = scale

    def forward(self, noisy_latents, timesteps, conditional_dict):
        weight = self.module.weight.view(1, 1, 1, 1, 1) * self.scale
        pred = noisy_latents * weight
        return torch.zeros_like(noisy_latents), pred


class _StubPipeline:
    def __init__(self, latents: torch.Tensor):
        self.latents = latents

    def inference_with_trajectory(self, *args, **kwargs):
        return self.latents, 400, 600


class DMDDistillerComponentTests(unittest.TestCase):
    def setUp(self):
        self.teacher = _StubTeacher()
        self.scheduler = _StubScheduler()
        self.distiller = DMDDistiller(
            teacher_model=self.teacher,
            noise_scheduler=self.scheduler,
            config={"generator_update_interval": 1},
        )
        latents = torch.ones(1, 2, 2, 4, 4)
        self.distiller.pipeline = _StubPipeline(latents)
        self.distiller.generator_wrapper = _ConstantWrapper(scale=0.0)
        self.distiller.real_score_wrapper = _ConstantWrapper(scale=0.5)
        self.distiller.fake_score_wrapper = _TrainableWrapper(self.distiller.fake_score_transformer, scale=0.8)

        self.prepared_batch = {
            "latents": torch.zeros_like(latents),
            "encoder_hidden_states": torch.zeros(1, 1),
            "added_cond_kwargs": {},
        }

    def test_generator_step_returns_loss(self):
        loss, logs = self.distiller._generator_step(self.prepared_batch)
        self.assertGreater(float(loss), 0.0)
        self.assertIn("dmd_generator_loss", logs)

    def test_critic_step_executes(self):
        # Replace optimizer with simple SGD for determinism.
        self.distiller.fake_score_optimizer = torch.optim.SGD(self.distiller.fake_score_transformer.parameters(), lr=1e-3)
        loss, logs = self.distiller._critic_step(self.prepared_batch)
        self.assertGreaterEqual(loss.detach().item(), 0.0)
        self.assertIn("dmd_critic_loss", logs)


if __name__ == "__main__":
    unittest.main()
