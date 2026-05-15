import unittest
from types import SimpleNamespace

import torch

import tests.test_stubs  # noqa: F401
from simpletuner.helpers.distillation.flow_dpo.distiller import FlowDPODistiller
from simpletuner.helpers.models.common import PredictionTypes


class _Adapter:
    def __init__(self):
        self.multiplier = 1.0
        self.calls = []

    def set_multiplier(self, value):
        self.multiplier = float(value)
        self.calls.append(self.multiplier)


class _FlowModel:
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING

    def __init__(self, adapter: _Adapter):
        self.adapter = adapter
        self.config = SimpleNamespace(lora_type="lycoris")
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            _lycoris_wrapped_network=adapter,
        )

    def model_predict(self, batch):
        target = batch["noise"] - batch["latents"]
        rejected = bool(torch.mean(batch["latents"]).item() > 0.5)
        if self.adapter.multiplier > 0.0:
            offset = 0.40 if rejected else 0.10
        else:
            offset = 0.10 if rejected else 0.30
        return {"model_prediction": target + offset * torch.ones_like(target)}


def _prepared_batch():
    latents = torch.zeros(2, 1, 2, 2)
    rejected_latents = torch.ones_like(latents)
    noise = torch.full_like(latents, 2.0)
    sigmas = torch.full((2, 1, 1, 1), 0.5)
    input_noise = noise.clone()
    return {
        "latents": latents,
        "conditioning_latents": rejected_latents,
        "conditioning_type": "reference_strict",
        "noise": noise,
        "input_noise": input_noise,
        "sigmas": sigmas,
        "timesteps": torch.full((2,), 500.0),
        "noisy_latents": (1 - sigmas) * latents + sigmas * input_noise,
    }


class FlowDPODistillerTests(unittest.TestCase):
    def test_computes_dpo_loss_with_adapter_reference_pass(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)
        distiller = FlowDPODistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "auto_beta": False, "beta": 1.0},
        )
        batch = _prepared_batch()
        model_output = model.model_predict(batch)

        loss, logs = distiller.compute_distill_loss(batch, model_output, torch.tensor(0.0))

        self.assertGreater(float(loss), 0.0)
        self.assertGreater(logs["flow_dpo_margin"], 0.0)
        self.assertIn("flow_dpo_gradient_factor", logs)
        self.assertEqual(adapter.calls, [0.0, 1.0])
        self.assertEqual(adapter.multiplier, 1.0)

    def test_requires_low_rank_training(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)

        with self.assertRaisesRegex(ValueError, "only supports low-rank"):
            FlowDPODistiller(
                teacher_model=model,
                noise_scheduler=None,
                config={"model_type": "full"},
            )

    def test_requires_reference_strict_conditioning(self):
        adapter = _Adapter()
        model = _FlowModel(adapter)
        distiller = FlowDPODistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "auto_beta": False},
        )
        batch = _prepared_batch()
        batch["conditioning_type"] = "reference_loose"
        model_output = model.model_predict(batch)

        with self.assertRaisesRegex(ValueError, "conditioning_type=reference_strict"):
            distiller.compute_distill_loss(batch, model_output, torch.tensor(0.0))


if __name__ == "__main__":
    unittest.main()
