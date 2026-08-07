import unittest
from types import SimpleNamespace

import torch

import tests.test_stubs  # noqa: F401
from simpletuner.helpers.distillation.factory import DistillerFactory
from simpletuner.helpers.distillation.h3_drift.distiller import H3DriftDistiller
from simpletuner.helpers.models.common import PredictionTypes


class _Adapter:
    def __init__(self):
        self.multiplier = 1.0
        self.calls = []

    def set_multiplier(self, value):
        self.multiplier = float(value)
        self.calls.append(self.multiplier)


class _FlowMapComponent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flowmap_enabled = False
        self.flowmap_gate_value = None
        self.flowmap_deltatime_type = None

    def enable_flowmap_time_conditioning(self, gate_value: float = 0.25, deltatime_type: str = "r") -> None:
        self.flowmap_enabled = True
        self.flowmap_gate_value = gate_value
        self.flowmap_deltatime_type = deltatime_type


class _H3Model:
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    NAME = "MiniMax H3"

    def __init__(self, adapter: _Adapter):
        self.adapter = adapter
        self.component = _FlowMapComponent()
        self.raise_when_disabled = False
        self.predict_batches = []
        self.loss_batches = []
        self.config = SimpleNamespace(lora_type="lycoris")
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            num_processes=1,
            _lycoris_wrapped_network=adapter,
        )

    def get_trained_component(self, unwrap_model=False):
        del unwrap_model
        return self.component

    def flow_matching_target_direction(self) -> float:
        return -1.0

    def noiseward_flow_to_prediction(self, flow: torch.Tensor) -> torch.Tensor:
        return -flow

    def prediction_to_noiseward_flow(self, prediction: torch.Tensor) -> torch.Tensor:
        return -prediction

    def get_flow_matching_target(
        self,
        prepared_batch: dict,
        *,
        latents: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        prefer_explicit_target: bool = True,
    ) -> torch.Tensor:
        if prefer_explicit_target and prepared_batch.get("target") is not None:
            return prepared_batch["target"]
        if prefer_explicit_target and prepared_batch.get("flow_target") is not None:
            return prepared_batch["flow_target"]
        if latents is None:
            latents = prepared_batch["latents"]
        if noise is None:
            noise = prepared_batch["noise"]
        return latents - noise

    def model_predict(self, batch):
        if self.raise_when_disabled and self.adapter.multiplier == 0.0:
            raise RuntimeError("reference pass failed")
        self.predict_batches.append(
            {
                "adapter_multiplier": self.adapter.multiplier,
                "has_flowmap": "flowmap_r_timesteps" in batch,
                "has_anyflow": "anyflow_r_timesteps" in batch,
            }
        )
        offset = 3.0 if self.adapter.multiplier > 0.0 else 1.0
        video = torch.zeros(2, 1, 2, 2) + offset
        output = {
            "model_prediction": video,
            "hidden_states_buffer": {"layer": torch.ones(1)},
        }
        if batch.get("include_audio", True):
            output["audio_prediction"] = torch.zeros(2, 2, 3, 2) + offset
        else:
            output["audio_prediction"] = None
        return output

    def loss(self, prepared_batch, model_output, apply_conditioning_mask: bool = True):
        del apply_conditioning_mask
        self.loss_batches.append(
            {
                "has_flowmap": "flowmap_r_timesteps" in prepared_batch,
                "has_anyflow": "anyflow_r_timesteps" in prepared_batch,
            }
        )
        target = self.get_flow_matching_target(prepared_batch)
        return (model_output["model_prediction"].float() - target.float()).square().mean()


class H3DriftDistillerTests(unittest.TestCase):
    def test_computes_reference_prediction_loss_with_adapter_disabled(self):
        adapter = _Adapter()
        model = _H3Model(adapter)
        distiller = H3DriftDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "model_family": "minimaxh3",
                "loss_weight": 0.5,
                "sft_loss_weight": 0.25,
            },
        )
        batch = {"include_audio": True, "audio_latent_mask": torch.tensor([1.0, 0.0])}
        model_output = model.model_predict(batch)

        loss, logs = distiller.compute_distill_loss(batch, model_output, torch.tensor(4.0))

        self.assertAlmostEqual(float(loss), 3.0)
        self.assertAlmostEqual(logs["h3_drift_loss"], 4.0)
        self.assertAlmostEqual(logs["h3_drift_audio_loss"], 4.0)
        self.assertEqual(logs["h3_drift_audio_elements"], 12.0)
        self.assertEqual(adapter.calls, [0.0, 1.0])
        self.assertEqual(adapter.multiplier, 1.0)

    def test_video_only_batch_allows_missing_audio_prediction(self):
        adapter = _Adapter()
        model = _H3Model(adapter)
        distiller = H3DriftDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "model_family": "minimaxh3", "sft_loss_weight": 0.0},
        )
        batch = {"include_audio": False}
        model_output = model.model_predict(batch)

        loss, logs = distiller.compute_distill_loss(batch, model_output, torch.tensor(4.0))

        self.assertAlmostEqual(float(loss), 4.0)
        self.assertEqual(logs["h3_drift_audio_elements"], 0.0)

    def test_reference_pass_failure_reenables_adapter(self):
        adapter = _Adapter()
        model = _H3Model(adapter)
        model.raise_when_disabled = True
        distiller = H3DriftDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "model_family": "minimaxh3"},
        )
        batch = {"include_audio": True}
        model_output = model.model_predict(batch)

        with self.assertRaisesRegex(RuntimeError, "reference pass failed"):
            distiller.compute_distill_loss(batch, model_output, torch.tensor(0.0))

        self.assertEqual(adapter.calls, [0.0, 1.0])
        self.assertEqual(adapter.multiplier, 1.0)

    def test_factory_creates_h3_drift_distiller(self):
        adapter = _Adapter()
        model = _H3Model(adapter)

        distiller = DistillerFactory.create_distiller(
            "h3_drift",
            teacher_model=model,
            noise_scheduler=None,
            config={"distillation_config": {"h3_drift": {"balance": "modality"}}},
            model_type="lora",
            model_family="minimaxh3",
        )

        self.assertIsInstance(distiller, H3DriftDistiller)
        self.assertEqual(distiller.config["balance"], "modality")

    def test_rejects_non_h3_model_family(self):
        adapter = _Adapter()
        model = _H3Model(adapter)
        model.NAME = "Flux"

        with self.assertRaisesRegex(ValueError, "MiniMax-H3"):
            H3DriftDistiller(
                teacher_model=model,
                noise_scheduler=None,
                config={"model_type": "lora", "model_family": "flux"},
            )

    def test_wraps_anyflow_and_preserves_normal_h3_anchor(self):
        adapter = _Adapter()
        model = _H3Model(adapter)
        distiller = H3DriftDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "model_family": "minimaxh3",
                "loss_weight": 0.5,
                "sft_loss_weight": 0.25,
                "inner_distillation_method": "anyflow",
                "inner_distillation_config": {
                    "target_mode": "online_teacher",
                    "r_timestep_sampler": "zero",
                    "loss_weight": 2.0,
                },
            },
        )
        latents = torch.zeros(2, 1, 2, 2)
        noise = torch.ones_like(latents)
        sigmas = torch.tensor([1.0, 0.5]).view(2, 1, 1, 1)
        batch = {
            "latents": latents,
            "noise": noise,
            "input_noise": noise.clone(),
            "sigmas": sigmas,
            "timesteps": torch.tensor([1.0, 0.5]),
            "noisy_latents": (1 - sigmas) * latents + sigmas * noise,
            "include_audio": False,
        }

        prepared = distiller.prepare_batch(batch, model=model, state={})
        model_output = model.model_predict(prepared)
        loss, logs = distiller.compute_distill_loss(prepared, model_output, torch.tensor(4.0))

        self.assertTrue(model.component.flowmap_enabled)
        self.assertTrue(torch.equal(prepared["target"], torch.ones_like(latents)))
        self.assertTrue(
            torch.equal(
                model.get_flow_matching_target(prepared, prefer_explicit_target=False),
                latents - noise,
            )
        )
        self.assertAlmostEqual(float(loss), 14.0)
        self.assertAlmostEqual(logs["anyflow_loss"], 8.0)
        self.assertAlmostEqual(logs["h3_drift_inner_total"], 8.0)
        self.assertAlmostEqual(logs["h3_drift_sft_loss"], 4.0)
        self.assertAlmostEqual(logs["h3_drift_weighted_loss"], 2.0)
        self.assertEqual(adapter.calls, [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(
            [entry["has_flowmap"] for entry in model.predict_batches],
            [False, True, True, False],
        )
        self.assertEqual(model.loss_batches, [{"has_flowmap": False, "has_anyflow": False}])

    def test_zero_drift_weight_skips_reference_pass(self):
        adapter = _Adapter()
        model = _H3Model(adapter)
        distiller = H3DriftDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "model_family": "minimaxh3",
                "loss_weight": 0.0,
                "sft_loss_weight": 0.25,
                "inner_distillation_method": "anyflow",
                "inner_distillation_config": {
                    "target_mode": "online_teacher",
                    "r_timestep_sampler": "zero",
                    "loss_weight": 2.0,
                },
            },
        )
        latents = torch.zeros(2, 1, 2, 2)
        noise = torch.ones_like(latents)
        sigmas = torch.tensor([1.0, 0.5]).view(2, 1, 1, 1)
        batch = {
            "latents": latents,
            "noise": noise,
            "input_noise": noise.clone(),
            "sigmas": sigmas,
            "timesteps": torch.tensor([1.0, 0.5]),
            "noisy_latents": (1 - sigmas) * latents + sigmas * noise,
            "include_audio": False,
        }

        prepared = distiller.prepare_batch(batch, model=model, state={})
        model_output = model.model_predict(prepared)
        loss, logs = distiller.compute_distill_loss(prepared, model_output, torch.tensor(4.0))

        self.assertAlmostEqual(float(loss), 12.0)
        self.assertEqual(logs["h3_drift_loss"], 0.0)
        self.assertEqual(logs["h3_drift_weighted_loss"], 0.0)
        self.assertEqual(logs["h3_drift_video_elements"], 0.0)
        self.assertEqual(logs["h3_drift_audio_elements"], 0.0)
        self.assertEqual(adapter.calls, [0.0, 1.0])
        self.assertEqual(
            [entry["has_flowmap"] for entry in model.predict_batches],
            [False, True, False],
        )
        self.assertEqual(model.loss_batches, [{"has_flowmap": False, "has_anyflow": False}])

    def test_rejects_recursive_inner_h3_drift(self):
        adapter = _Adapter()
        model = _H3Model(adapter)

        with self.assertRaisesRegex(ValueError, "may not wrap"):
            H3DriftDistiller(
                teacher_model=model,
                noise_scheduler=None,
                config={
                    "model_type": "lora",
                    "model_family": "minimaxh3",
                    "inner_distillation_method": "h3_drift",
                },
            )


if __name__ == "__main__":
    unittest.main()
