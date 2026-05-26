import unittest
from types import SimpleNamespace

import torch

import tests.test_stubs  # noqa: F401
from simpletuner.helpers.distillation.anyflow.distiller import AnyFlowDistiller
from simpletuner.helpers.distillation.factory import DistillerFactory
from simpletuner.helpers.models.common import PredictionTypes


class _FlowMapComponent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_enabled = True
        self.flowmap_enabled = False
        self.flowmap_gate_value = None
        self.flowmap_deltatime_type = None

    def enable_flowmap_time_conditioning(self, gate_value: float = 0.25, deltatime_type: str = "r") -> None:
        self.flowmap_enabled = True
        self.flowmap_gate_value = gate_value
        self.flowmap_deltatime_type = deltatime_type

    def enable_lora(self):
        self.adapter_enabled = True

    def disable_lora(self):
        self.adapter_enabled = False


class _FlowModel:
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING

    def __init__(self):
        self.component = _FlowMapComponent()
        self.teacher_adapter_states = []
        self.teacher_timesteps = []
        self.config = SimpleNamespace(lora_type="standard", weight_dtype=torch.float32)
        self.accelerator = SimpleNamespace(device=torch.device("cpu"), num_processes=1)

    def get_trained_component(self, unwrap_model=False):
        return self.component

    def model_predict(self, batch):
        self.teacher_adapter_states.append(self.component.adapter_enabled)
        self.teacher_timesteps.append(batch["timesteps"].detach().clone())
        value = 7.0 if self.component.adapter_enabled else 2.0
        return {"model_prediction": torch.full_like(batch["noisy_latents"], value)}


class _EpsilonModel(_FlowModel):
    PREDICTION_TYPE = PredictionTypes.EPSILON


class _NoFlowMapModel(_FlowModel):
    def __init__(self):
        super().__init__()
        self.component = torch.nn.Linear(1, 1)


def _prepared_batch():
    latents = torch.zeros(2, 1, 2, 2)
    noise = torch.ones_like(latents)
    sigmas = torch.tensor([1.0, 0.5]).view(2, 1, 1, 1)
    timesteps = torch.tensor([1000.0, 500.0])
    return {
        "latents": latents,
        "noise": noise,
        "input_noise": noise.clone(),
        "sigmas": sigmas,
        "timesteps": timesteps,
        "noisy_latents": (1 - sigmas) * latents + sigmas * noise,
    }


class AnyFlowDistillerTests(unittest.TestCase):
    def test_init_enables_flowmap_conditioning(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "gate_value": 0.5, "deltatime_type": "t-r", "target_mode": "linear"},
        )

        self.assertIs(distiller._flowmap_component, model.component)
        self.assertTrue(model.component.flowmap_enabled)
        self.assertEqual(model.component.flowmap_gate_value, 0.5)
        self.assertEqual(model.component.flowmap_deltatime_type, "t-r")

    def test_linear_prepare_batch_sets_r_timesteps_and_target(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "target_mode": "linear", "r_timestep_sampler": "zero"},
        )
        batch = distiller.prepare_batch(_prepared_batch(), model=model, state={})

        self.assertTrue(torch.equal(batch["flowmap_r_timesteps"], torch.zeros(2)))
        self.assertTrue(torch.equal(batch["target"], torch.ones_like(batch["latents"])))
        self.assertIs(batch["flow_target"], batch["target"])

    def test_linear_prepare_batch_preserves_normalized_timestep_parameterization(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "target_mode": "linear", "r_timestep_sampler": "zero"},
        )
        batch = _prepared_batch()
        batch["timesteps"] = torch.tensor([1.0, 0.5])

        batch = distiller.prepare_batch(batch, model=model, state={})

        self.assertTrue(torch.equal(batch["flowmap_r_timesteps"], torch.zeros(2)))
        self.assertTrue(torch.equal(batch["anyflow_timestep_interval"], torch.tensor([1.0, 0.5])))

    def test_online_teacher_target_uses_disabled_adapter(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "target_mode": "online_teacher",
                "r_timestep_sampler": "zero",
                "teacher_rollout_steps": 3,
            },
        )

        batch = distiller.prepare_batch(_prepared_batch(), model=model, state={})

        self.assertTrue(torch.allclose(batch["target"], torch.full_like(batch["latents"], 2.0)))
        self.assertEqual(model.teacher_adapter_states, [False, False, False])
        self.assertTrue(model.component.adapter_enabled)
        self.assertEqual(len(model.teacher_timesteps), 3)

    def test_compute_distill_loss_scales_precomputed_training_loss(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "target_mode": "linear", "loss_weight": 0.5},
        )
        batch = _prepared_batch()
        batch["anyflow_r_timesteps"] = torch.zeros(2)

        loss, logs = distiller.compute_distill_loss(batch, {}, torch.tensor(4.0))

        self.assertEqual(float(loss), 2.0)
        self.assertEqual(logs["anyflow_loss"], 2.0)
        self.assertIn("anyflow_interval", logs)

    def test_zero_length_interval_is_rejected(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "target_mode": "linear", "r_timestep_sampler": "zero"},
        )
        batch = _prepared_batch()
        batch["timesteps"] = torch.tensor([0.0, 500.0])
        batch["sigmas"][0] = 0.0
        batch["noisy_latents"] = (1 - batch["sigmas"]) * batch["latents"] + batch["sigmas"] * batch["noise"]

        with self.assertRaisesRegex(ValueError, "r_timestep < timestep"):
            distiller.prepare_batch(batch, model=model, state={})

    def test_factory_creates_anyflow_distiller(self):
        model = _FlowModel()

        distiller = DistillerFactory.create_distiller(
            "anyflow",
            teacher_model=model,
            noise_scheduler=None,
            config={"distillation_config": {"anyflow": {"target_mode": "linear", "r_timestep_sampler": "zero"}}},
            model_type="lora",
            prediction_type="flow_matching",
        )

        self.assertIsInstance(distiller, AnyFlowDistiller)
        self.assertEqual(distiller.config["target_mode"], "linear")
        self.assertTrue(model.component.flowmap_enabled)

    def test_requires_flow_matching_model(self):
        with self.assertRaisesRegex(ValueError, "flow-matching"):
            AnyFlowDistiller(
                teacher_model=_EpsilonModel(),
                noise_scheduler=None,
                config={"model_type": "lora"},
            )

    def test_requires_flowmap_capable_component(self):
        with self.assertRaisesRegex(ValueError, "FlowMap interval conditioning"):
            AnyFlowDistiller(
                teacher_model=_NoFlowMapModel(),
                noise_scheduler=None,
                config={"model_type": "lora"},
            )


if __name__ == "__main__":
    unittest.main()
