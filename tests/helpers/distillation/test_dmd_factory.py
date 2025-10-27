import unittest
from types import SimpleNamespace

import torch

from simpletuner.helpers.distillation.dmd.distiller import DMDDistiller
from simpletuner.helpers.distillation.factory import DistillerFactory
from simpletuner.helpers.models.common import PredictionTypes


class _StubScheduler:
    def __init__(self):
        self.sigmas = torch.linspace(1.0, 0.0, steps=1000)
        self.timesteps = torch.arange(1000)
        self.config = SimpleNamespace(num_train_timesteps=1000)


class _StubTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(to_dict=lambda: {})

    def forward(self, *args, **kwargs):
        return (torch.zeros(1, 1, 1, 1, 1),)


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


class DMDDistillerConfigTests(unittest.TestCase):
    def setUp(self):
        self.teacher = _StubTeacher()
        self.scheduler = _StubScheduler()

    def test_defaults_populated(self):
        distiller = DMDDistiller(
            teacher_model=self.teacher,
            noise_scheduler=self.scheduler,
        )
        config = distiller.config

        self.assertEqual(config["generator_update_interval"], 1)
        self.assertIn("fake_score_guidance_scale", config)
        self.assertIn("num_frame_per_block", config)
        self.assertIn("num_training_frames", config)
        self.assertIn("fake_score_grad_clip", config)

    def test_overrides_applied(self):
        distiller = DMDDistiller(
            teacher_model=self.teacher,
            noise_scheduler=self.scheduler,
            config={
                "generator_update_interval": 4,
                "fake_score_guidance_scale": 1.5,
                "num_training_frames": 16,
            },
        )
        config = distiller.config
        self.assertEqual(config["generator_update_interval"], 4)
        self.assertAlmostEqual(config["fake_score_guidance_scale"], 1.5)
        self.assertEqual(config["num_training_frames"], 16)


class DistillerFactoryDMDDTests(unittest.TestCase):
    def setUp(self):
        self.teacher = _StubTeacher()
        self.scheduler = _StubScheduler()

    def test_factory_merges_distillation_config(self):
        trainer_config = {
            "distillation_config": {
                "dmd": {
                    "generator_update_interval": 6,
                    "fake_score_lr": 2e-5,
                }
            }
        }
        distiller = DistillerFactory.create_distiller(
            method="dmd",
            teacher_model=self.teacher,
            noise_scheduler=self.scheduler,
            config=trainer_config,
            model_type="lora",
            model_family="wan",
            prediction_type="flow_matching",
        )

        self.assertIsInstance(distiller, DMDDistiller)
        self.assertEqual(distiller.config["generator_update_interval"], 6)
        self.assertAlmostEqual(distiller.config["fake_score_lr"], 2e-5)


if __name__ == "__main__":
    unittest.main()

