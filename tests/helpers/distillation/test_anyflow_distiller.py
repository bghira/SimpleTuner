import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from peft import LoraConfig, inject_adapter_in_model
from peft.utils import get_peft_model_state_dict
from safetensors.torch import save_file

import tests.test_stubs  # noqa: F401
from simpletuner.helpers.distillation.anyflow.distiller import AnyFlowDistiller
from simpletuner.helpers.distillation.anyflow.scheduler import AnyFlowValidationScheduler
from simpletuner.helpers.distillation.factory import DistillerFactory
from simpletuner.helpers.models.common import PredictionTypes


class _FlowMapComponent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_enabled = True
        self.active_adapter = "default"
        self.adapter_params = torch.nn.ParameterDict({"default": torch.nn.Parameter(torch.tensor([5.0]))})
        self.peft_config = {"default": SimpleNamespace(name="default")}
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

    def add_adapter(self, adapter_config, adapter_name="default"):
        self.peft_config[adapter_name] = adapter_config
        self.adapter_params[adapter_name] = torch.nn.Parameter(torch.zeros(1))

    def set_adapter(self, adapter_name):
        self.active_adapter = adapter_name
        for name, parameter in self.adapter_params.items():
            parameter.requires_grad_(name == adapter_name)

    def forward(self, timestep=None, r_timestep=None, **kwargs):
        del kwargs
        self.last_timestep = timestep
        self.last_r_timestep = r_timestep
        return (r_timestep,)


class _FlowModel:
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING

    def __init__(self):
        self.component = _FlowMapComponent()
        self.teacher_adapter_states = []
        self.teacher_timesteps = []
        self.config = SimpleNamespace(lora_type="standard", weight_dtype=torch.float32)
        self.accelerator = SimpleNamespace(
            device=torch.device("cpu"),
            is_main_process=True,
            num_processes=1,
            process_index=0,
        )

    def get_trained_component(self, unwrap_model=False):
        return self.component

    def model_predict(self, batch):
        self.teacher_adapter_states.append(self.component.adapter_enabled)
        self.teacher_timesteps.append(batch["timesteps"].detach().clone())
        value = batch["noisy_latents"].new_tensor(2.0)
        if self.component.adapter_enabled:
            value = value + self.component.adapter_params[self.component.active_adapter]
        return {"model_prediction": torch.ones_like(batch["noisy_latents"]) * value}

    def flow_matching_target(self, latents, noise):
        return noise - latents


class _EpsilonModel(_FlowModel):
    PREDICTION_TYPE = PredictionTypes.EPSILON


class _InverseFlowModel(_FlowModel):
    def get_flow_matching_target(
        self,
        prepared_batch,
        *,
        latents=None,
        noise=None,
        prefer_explicit_target=True,
    ):
        if prefer_explicit_target and prepared_batch.get("target") is not None:
            return prepared_batch["target"]
        if prefer_explicit_target and prepared_batch.get("flow_target") is not None:
            return prepared_batch["flow_target"]
        if latents is None:
            latents = prepared_batch["latents"]
        if noise is None:
            noise = prepared_batch["noise"]
        return latents - noise

    def prediction_to_noiseward_flow(self, prediction):
        return -prediction

    def noiseward_flow_to_prediction(self, flow):
        return -flow

    def flow_matching_target(self, latents, noise):
        return latents - noise


class _DatawardTimeFlowModel(_FlowModel):
    def flow_matching_timesteps_from_sigmas(self, sigmas, *, reference_timesteps=None):
        del reference_timesteps
        return 1.0 - sigmas


class _SigmaPredictionFlowModel(_FlowModel):
    def model_predict(self, batch):
        return {"model_prediction": batch["noisy_latents"].clone()}


class _DatawardSigmaPredictionFlowModel(_InverseFlowModel):
    def flow_matching_timesteps_from_sigmas(self, sigmas, *, reference_timesteps=None):
        del reference_timesteps
        return 1.0 - sigmas

    def model_predict(self, batch):
        return {"model_prediction": -batch["noisy_latents"].clone()}


class _DatawardRoleFlowModel(_InverseFlowModel):
    def flow_matching_timesteps_from_sigmas(self, sigmas, *, reference_timesteps=None):
        del reference_timesteps
        return 1.0 - sigmas

    def model_predict(self, batch):
        self.teacher_timesteps.append(batch["timesteps"].detach().clone())
        value = batch["noisy_latents"].new_tensor(2.0)
        if self.component.adapter_enabled:
            value = value + self.component.adapter_params[self.component.active_adapter]
        return {"model_prediction": -torch.ones_like(batch["noisy_latents"]) * value}


class _NoFlowMapModel(_FlowModel):
    def __init__(self):
        super().__init__()
        self.component = torch.nn.Linear(1, 1)


class _ValidationScheduler:
    order = 1

    def __init__(self):
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.timesteps = torch.tensor([1000.0, 500.0])
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])

    def set_timesteps(self, timesteps):
        self.timesteps = torch.as_tensor(timesteps, dtype=torch.float32)

    def step(self, *args, **kwargs):
        self.step_args = args
        self.step_kwargs = kwargs
        return ("stepped",)


class _DatawardValidationScheduler(_ValidationScheduler):
    def __init__(self):
        super().__init__()
        self.timesteps = torch.tensor([0.0, 0.5])


class _TParameterComponent(torch.nn.Module):
    def forward(self, x, t, cap_feats, r_timestep=None):
        del x, cap_feats
        self.last_timestep = t
        self.last_r_timestep = r_timestep
        return (r_timestep,)


class _TimestepRComponent(torch.nn.Module):
    def forward(self, timestep, timestep_r=None):
        self.last_timestep = timestep
        self.last_timestep_r = timestep_r
        return (timestep_r,)


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
        "negative_encoder_hidden_states": torch.zeros(2, 1, 1),
    }


class AnyFlowDistillerTests(unittest.TestCase):
    def test_init_enables_flowmap_conditioning(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "gate_value": 0.5, "deltatime_type": "t-r"},
        )

        self.assertIs(distiller._flowmap_component, model.component)
        self.assertTrue(model.component.flowmap_enabled)
        self.assertEqual(model.component.flowmap_gate_value, 0.5)
        self.assertEqual(model.component.flowmap_deltatime_type, "t-r")

    def test_factory_prepares_flowmap_conditioning_before_adapter_creation(self):
        model = _FlowModel()

        DistillerFactory.prepare_model_for_adapter(
            "anyflow",
            model,
            {
                "distillation_config": {
                    "anyflow": {"gate_value": 0.4, "deltatime_type": "t-r"},
                }
            },
        )

        self.assertTrue(model.component.flowmap_enabled)
        self.assertEqual(model.component.flowmap_gate_value, 0.4)
        self.assertEqual(model.component.flowmap_deltatime_type, "t-r")

    def test_adapter_preparation_rejects_lora_dropout(self):
        model = _FlowModel()
        model.config.lora_dropout = 0.1

        with self.assertRaisesRegex(ValueError, "lora_dropout=0.0"):
            AnyFlowDistiller.prepare_model_for_adapter(model, {})

    def test_factory_reports_guidance_conditioning_for_method_config_shapes(self):
        direct = DistillerFactory.training_batch_requirements(
            "anyflow",
            {"distillation_config": {"anyflow": {"fuse_guidance_scale": 3.0}}},
        )
        nested = DistillerFactory.training_batch_requirements(
            "h3_drift",
            {
                "distillation_config": {
                    "h3_drift": {
                        "inner_distillation_method": "anyflow",
                        "inner_distillation_config": {"fuse_guidance_scale": 3.0},
                    }
                }
            },
        )
        unwrapped = DistillerFactory.training_batch_requirements(
            "anyflow",
            {"distillation_config": {"fuse_guidance_scale": 3.0}},
        )
        real_guidance_only = DistillerFactory.training_batch_requirements(
            "anyflow",
            {"distillation_config": {"fuse_guidance_scale": 1.0, "real_score_guidance_scale": 0.5}},
        )

        self.assertEqual(direct, {"unconditional_text_embeddings"})
        self.assertEqual(unwrapped, direct)
        self.assertEqual(real_guidance_only, direct)
        self.assertEqual(nested, direct)

    def test_removed_legacy_target_modes_are_rejected(self):
        for target_mode in ("online_teacher", "linear"):
            with self.subTest(target_mode=target_mode), self.assertRaisesRegex(ValueError, "target_mode was removed"):
                AnyFlowDistiller(
                    teacher_model=_FlowModel(),
                    noise_scheduler=None,
                    config={"model_type": "lora", "target_mode": target_mode},
                )

    def test_meanflow_schedule_shift_can_override_model_scheduler(self):
        model = _FlowModel()
        scheduler = SimpleNamespace(config=SimpleNamespace(shift=12.0))
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=scheduler,
            config={"model_type": "lora", "schedule_shift": 5.0},
        )

        shifted = distiller._apply_scheduler_shift(torch.tensor([0.5]))

        self.assertTrue(torch.allclose(shifted, torch.tensor([5.0 / 6.0])))

    def test_meanflow_schedule_shift_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "schedule_shift must be greater than zero"):
            AnyFlowDistiller(
                teacher_model=_FlowModel(),
                noise_scheduler=None,
                config={"model_type": "lora", "schedule_shift": 0.0},
            )

    def test_meanflow_non_diffusion_sigma_cap_preserves_diffusion_samples(self):
        model = _FlowModel()
        scheduler = SimpleNamespace(config=SimpleNamespace(shift=5.0))
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=scheduler,
            config={
                "model_type": "lora",
                "diffusion_ratio": 0.5,
                "consistency_ratio": 0.5,
                "meanflow_non_diffusion_max_sigma": 0.95,
            },
        )
        draws = [torch.tensor([0.8, 0.7]), torch.tensor([0.2, 0.4])]

        with patch("torch.rand", side_effect=draws):
            batch = _prepared_batch()
            t_sigmas, _ = distiller._prepare_meanflow_pair(batch, model)

        cap_in_base_coordinates = 19.0 / 24.0
        self.assertAlmostEqual(float(batch["anyflow_t_base"][0]), 0.8)
        self.assertAlmostEqual(float(batch["anyflow_t_base"][1]), 0.7 * cap_in_base_coordinates)
        self.assertGreater(float(t_sigmas[0]), 0.95)
        self.assertLessEqual(float(t_sigmas[1]), 0.95)

    def test_meanflow_non_diffusion_sigma_cap_must_be_in_unit_interval(self):
        for value in (0.0, 1.01):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(
                    ValueError,
                    "meanflow_non_diffusion_max_sigma must be in",
                ),
            ):
                AnyFlowDistiller(
                    teacher_model=_FlowModel(),
                    noise_scheduler=None,
                    config={"model_type": "lora", "meanflow_non_diffusion_max_sigma": value},
                )

    def test_meanflow_central_difference_can_clamp_to_physical_sigma_bounds(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "diffusion_ratio": 0.0,
                "consistency_ratio": 1.0,
                "central_difference_epsilon": 0.005,
                "central_difference_boundary_mode": "clamp",
            },
        )
        draws = [torch.tensor([0.999, 0.998]), torch.tensor([0.2, 0.4])]

        with patch("torch.rand", side_effect=draws):
            distiller.prepare_batch(_prepared_batch(), model=model, state={})

        self.assertEqual(len(model.teacher_timesteps), 2)
        self.assertLessEqual(float(torch.stack(model.teacher_timesteps).max()), 1000.0)

    def test_meanflow_central_difference_boundary_mode_is_validated(self):
        with self.assertRaisesRegex(ValueError, "central_difference_boundary_mode"):
            AnyFlowDistiller(
                teacher_model=_FlowModel(),
                noise_scheduler=None,
                config={"model_type": "lora", "central_difference_boundary_mode": "invalid"},
            )

    def test_chunked_geometry_matches_full_precision_reductions(self):
        left = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[-1.0, 2.0], [0.0, 3.0]]], dtype=torch.bfloat16)
        right = torch.tensor([[[4.0, 3.0], [2.0, 1.0]], [[2.0, -1.0], [4.0, 1.0]]], dtype=torch.bfloat16)

        cosine, norm_ratio = AnyFlowDistiller._chunked_per_sample_geometry(left, right, chunk_elements=2)
        left_flat = left.float().flatten(1)
        right_flat = right.float().flatten(1)
        expected_cosine = torch.nn.functional.cosine_similarity(left_flat, right_flat, dim=1)
        expected_ratio = torch.linalg.vector_norm(left_flat, dim=1) / torch.linalg.vector_norm(right_flat, dim=1)

        self.assertTrue(torch.allclose(cosine, expected_cosine))
        self.assertTrue(torch.allclose(norm_ratio, expected_ratio))

    def test_meanflow_uses_official_interval_mixture_and_central_target(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "diffusion_ratio": 0.5,
                "consistency_ratio": 0.5,
            },
        )
        draws = [
            torch.tensor([0.8, 0.7]),
            torch.tensor([0.2, 0.4]),
        ]

        with patch("torch.rand", side_effect=draws):
            batch = distiller.prepare_batch(_prepared_batch(), model=model, state={})

        self.assertTrue(torch.equal(batch["anyflow_diffusion_mask"], torch.tensor([True, False])))
        self.assertTrue(torch.equal(batch["anyflow_consistency_mask"], torch.tensor([False, True])))
        self.assertTrue(torch.equal(batch["anyflow_arbitrary_mask"], torch.tensor([False, False])))
        self.assertTrue(torch.allclose(batch["anyflow_t_base"], torch.tensor([0.8, 0.7])))
        self.assertTrue(torch.allclose(batch["anyflow_r_base"], torch.tensor([0.8, 0.0])))
        self.assertTrue(torch.allclose(batch["target"], torch.ones_like(batch["latents"])))
        self.assertEqual(len(model.teacher_timesteps), 2)
        self.assertTrue(model.component.adapter_enabled)

    def test_meanflow_can_anchor_only_diffusion_samples_to_frozen_base_prediction(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "diffusion_ratio": 0.5,
                "consistency_ratio": 0.5,
                "diffusion_target": "base_prediction",
                "fuse_guidance_scale": 1.0,
            },
        )
        draws = [
            torch.tensor([0.8, 0.7]),
            torch.tensor([0.2, 0.4]),
        ]

        with patch("torch.rand", side_effect=draws):
            batch = distiller.prepare_batch(_prepared_batch(), model=model, state={})

        expected = torch.ones_like(batch["target"])
        expected[0] = 2.0
        self.assertTrue(torch.equal(batch["target"], expected))
        self.assertEqual(model.teacher_adapter_states, [False, True, True])
        self.assertTrue(model.component.adapter_enabled)
        self.assertTrue(model.component.training)
        self.assertTrue(torch.equal(batch["_anyflow_base_prediction_flow_norm_ratio"], torch.full((2,), 2.0)))

    def test_meanflow_diffusion_target_is_validated(self):
        with self.assertRaisesRegex(ValueError, "diffusion_target"):
            AnyFlowDistiller(
                teacher_model=_FlowModel(),
                noise_scheduler=None,
                config={"model_type": "lora", "diffusion_target": "unknown"},
            )

        with self.assertRaisesRegex(ValueError, "requires fuse_guidance_scale=1.0"):
            AnyFlowDistiller(
                teacher_model=_FlowModel(),
                noise_scheduler=None,
                config={"model_type": "lora", "diffusion_target": "base_prediction"},
            )

    def test_frozen_base_target_uses_base_flow_residual_for_adaptive_weighting(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "diffusion_ratio": 0.5,
                "consistency_ratio": 0.5,
                "diffusion_target": "base_prediction",
                "meanflow_weight_type": "uniform",
                "fuse_guidance_scale": 1.0,
            },
        )
        draws = [
            torch.tensor([0.8, 0.7]),
            torch.tensor([0.2, 0.4]),
        ]

        with patch("torch.rand", side_effect=draws):
            batch = distiller.prepare_batch(_prepared_batch(), model=model, state={})

        prediction = batch["target"].clone()
        prediction[1] += 2.0
        loss = distiller._meanflow_loss(batch, {"model_prediction": prediction})

        self.assertAlmostEqual(float(batch["_anyflow_pre_adaptive_loss"][0]), 0.0)
        self.assertAlmostEqual(float(batch["_anyflow_adaptive_reference_loss"][0]), 1.0)
        self.assertAlmostEqual(float(batch["_anyflow_adaptive_scale"][1]), 0.25, places=5)
        self.assertAlmostEqual(float(batch["_anyflow_post_adaptive_loss"][1]), 1.0, places=5)
        self.assertAlmostEqual(float(loss), 0.5, places=5)

    def test_seeded_meanflow_pair_uses_isolated_rng_stream(self):
        torch.manual_seed(1234)
        expected_next = torch.rand(4)
        torch.manual_seed(1234)
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "seed": 987},
        )

        distiller.prepare_batch(_prepared_batch(), model=model, state={})
        actual_next = torch.rand(4)

        self.assertTrue(torch.equal(actual_next, expected_next))

    def test_meanflow_branch_assignment_uses_data_replica_rank_with_context_parallelism(self):
        model = _FlowModel()
        model.accelerator.num_processes = 8
        model.accelerator.process_index = 6
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "diffusion_ratio": 0.5,
                "consistency_ratio": 0.25,
            },
        )
        draws = [torch.tensor([0.8, 0.7]), torch.tensor([0.2, 0.4])]

        with (
            patch("torch.rand", side_effect=draws),
            patch(
                "simpletuner.helpers.distillation.anyflow.distiller.get_model_replica_data_info",
                return_value=(True, 1, 0, 2, 4),
            ),
        ):
            batch = _prepared_batch()
            distiller._prepare_meanflow_pair(batch, model)

        self.assertTrue(torch.equal(batch["anyflow_diffusion_mask"], torch.tensor([True, True])))
        self.assertTrue(torch.equal(batch["anyflow_consistency_mask"], torch.tensor([False, False])))

    def test_meanflow_warns_once_when_global_batch_omits_enabled_branch(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "diffusion_ratio": 0.5,
                "consistency_ratio": 0.25,
            },
        )
        latents = torch.zeros(3, 1, 2, 2)
        noise = torch.ones_like(latents)
        sigmas = torch.tensor([0.9, 0.6, 0.3]).view(3, 1, 1, 1)
        batch = {
            "latents": latents,
            "noise": noise,
            "input_noise": noise.clone(),
            "sigmas": sigmas,
            "timesteps": torch.tensor([900.0, 600.0, 300.0]),
            "noisy_latents": (1 - sigmas) * latents + sigmas * noise,
        }
        draws = [torch.tensor([0.8, 0.7, 0.6]), torch.tensor([0.2, 0.4, 0.3])] * 2

        with self.assertLogs("AnyFlowDistiller", level="WARNING") as captured, patch("torch.rand", side_effect=draws):
            distiller._prepare_meanflow_pair(batch, model)
            distiller._prepare_meanflow_pair(batch, model)

        self.assertEqual(len(captured.records), 1)
        self.assertIn("zero arbitrary samples", captured.output[0])

    def test_meanflow_central_difference_uses_noise_sigma_coordinate(self):
        model = _SigmaPredictionFlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "diffusion_ratio": 0.0,
                "consistency_ratio": 1.0,
                "central_difference_epsilon": 0.01,
                "fuse_guidance_scale": 1.0,
            },
        )
        draws = [
            torch.tensor([0.8, 0.7]),
            torch.tensor([0.2, 0.4]),
        ]

        with patch("torch.rand", side_effect=draws):
            batch = distiller.prepare_batch(_prepared_batch(), model=model, state={})

        expected = 1.0 - torch.tensor([0.8, 0.7]).view(2, 1, 1, 1)
        self.assertTrue(torch.allclose(batch["target"], expected.expand_as(batch["target"]), atol=1e-5))

    def test_meanflow_preserves_h3_dataward_prediction_convention(self):
        model = _DatawardSigmaPredictionFlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "diffusion_ratio": 0.0,
                "consistency_ratio": 1.0,
                "central_difference_epsilon": 0.01,
                "fuse_guidance_scale": 1.0,
            },
        )
        draws = [
            torch.tensor([0.8, 0.7]),
            torch.tensor([0.2, 0.4]),
        ]

        with patch("torch.rand", side_effect=draws):
            batch = distiller.prepare_batch(_prepared_batch(), model=model, state={})

        expected = -1.0 + torch.tensor([0.8, 0.7]).view(2, 1, 1, 1)
        self.assertTrue(torch.allclose(batch["target"], expected.expand_as(batch["target"]), atol=1e-5))
        self.assertTrue(torch.allclose(batch["timesteps"], torch.tensor([0.2, 0.3])))

    def test_meanflow_rejects_joint_audio_video_batch(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora"},
        )
        batch = _prepared_batch()
        batch["audio_latents"] = torch.zeros(2, 2, 3, 4)

        with self.assertRaisesRegex(ValueError, "joint audio-video"):
            distiller.prepare_batch(batch, model=model, state={})

    def test_meanflow_loss_uses_per_sample_target_and_interval_logs(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "diffusion_ratio": 0.5,
                "consistency_ratio": 0.5,
                "meanflow_weight_type": "uniform",
                "meanflow_adaptive_weighting": False,
                "loss_weight": 0.5,
                "fuse_guidance_scale": 1.0,
            },
        )
        draws = [
            torch.tensor([0.8, 0.7]),
            torch.tensor([0.2, 0.4]),
        ]
        with patch("torch.rand", side_effect=draws):
            batch = distiller.prepare_batch(_prepared_batch(), model=model, state={})
        prediction = batch["target"] + 2.0

        loss, logs = distiller.compute_distill_loss(
            batch,
            {"model_prediction": prediction},
            original_loss=torch.tensor(99.0),
        )

        self.assertAlmostEqual(float(loss), 2.0)
        self.assertAlmostEqual(logs["anyflow_diffusion_fraction"], 0.5)
        self.assertAlmostEqual(logs["anyflow_consistency_fraction"], 0.5)
        self.assertAlmostEqual(logs["anyflow_arbitrary_fraction"], 0.0)
        self.assertAlmostEqual(logs["anyflow_diffusion_target_base_cosine"], 1.0)
        self.assertAlmostEqual(logs["anyflow_diffusion_t_sigma"], 0.8)
        self.assertAlmostEqual(logs["anyflow_consistency_r_sigma"], 0.0)
        self.assertAlmostEqual(logs["anyflow_consistency_target_base_norm_ratio"], 1.0)
        self.assertAlmostEqual(logs["anyflow_diffusion_pre_adaptive_loss"], 4.0)
        self.assertAlmostEqual(logs["anyflow_consistency_adaptive_scale"], 1.0)
        self.assertAlmostEqual(logs["anyflow_consistency_post_adaptive_loss"], 4.0)

    def test_meanflow_fuses_guidance_without_replacing_raw_flow_target(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "fuse_guidance_scale": 3.0,
                "meanflow_weight_type": "uniform",
                "meanflow_adaptive_weighting": False,
            },
        )
        batch = _prepared_batch()
        batch["target"] = torch.ones_like(batch["latents"])
        conditional = torch.full_like(batch["latents"], 10.0, requires_grad=True)

        loss = distiller._meanflow_loss(batch, {"model_prediction": conditional})
        loss.backward()

        # The detached unconditional prediction is 7, so (10 + 2*7) / 3 = 8.
        self.assertAlmostEqual(float(loss.detach()), 49.0)
        self.assertTrue(torch.allclose(conditional.grad, torch.full_like(conditional, 7.0 / 12.0)))
        self.assertTrue(torch.equal(batch["target"], torch.ones_like(batch["target"])))

    def test_meanflow_guidance_requires_unconditional_embeddings(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "fuse_guidance_scale": 3.0},
        )
        batch = _prepared_batch()
        batch.pop("negative_encoder_hidden_states")
        batch["target"] = torch.ones_like(batch["latents"])

        with self.assertRaisesRegex(ValueError, "cached unconditional text embeddings"):
            distiller._meanflow_loss(batch, {"model_prediction": torch.ones_like(batch["latents"])})

    def test_real_score_guidance_uses_negative_tags_and_attention_mask(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "real_score_guidance_scale": 0.5},
        )
        batch = _prepared_batch()
        batch["encoder_hidden_states"] = torch.ones(2, 5, 1)
        batch["negative_encoder_hidden_states"] = torch.zeros(2, 3, 1)
        batch["text_token_tags"] = torch.ones(2, 5, dtype=torch.long)
        batch["negative_text_token_tags"] = torch.full((2, 3), 2, dtype=torch.long)
        batch["encoder_attention_mask"] = torch.ones(2, 5, dtype=torch.long)
        batch["negative_encoder_attention_mask"] = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.long)
        conditional_x0 = torch.ones_like(batch["latents"])
        unconditional_x0 = torch.zeros_like(batch["latents"])

        with patch.object(distiller, "_score_x0", return_value=unconditional_x0) as score_x0:
            result = distiller._apply_real_score_guidance(
                batch,
                batch["noisy_latents"],
                batch["timesteps"],
                conditional_x0,
            )

        score_x0.assert_called_once()
        unconditional_batch = score_x0.call_args.args[0]
        self.assertIs(unconditional_batch["encoder_hidden_states"], batch["negative_encoder_hidden_states"])
        self.assertIs(unconditional_batch["text_token_tags"], batch["negative_text_token_tags"])
        self.assertIs(unconditional_batch["encoder_attention_mask"], batch["negative_encoder_attention_mask"])
        self.assertTrue(torch.equal(result, torch.full_like(conditional_x0, 1.5)))

    def test_unconditional_batch_drops_positive_attention_mask_when_negative_has_none(self):
        batch = _prepared_batch()
        batch["encoder_attention_mask"] = torch.ones(2, 5, dtype=torch.long)

        unconditional_batch = AnyFlowDistiller._unconditional_batch(batch)

        self.assertNotIn("encoder_attention_mask", unconditional_batch)

    def test_unconditional_batch_drops_model_specific_conditioning_aliases(self):
        batch = _prepared_batch()
        batch["prompt_embeds"] = torch.ones(2, 7, 4)
        batch["attention_mask"] = torch.ones(2, 7, dtype=torch.bool)
        batch["attention_masks"] = torch.ones(2, 7, dtype=torch.bool)

        unconditional_batch = AnyFlowDistiller._unconditional_batch(batch)

        self.assertNotIn("prompt_embeds", unconditional_batch)
        self.assertNotIn("attention_mask", unconditional_batch)
        self.assertNotIn("attention_masks", unconditional_batch)

    def test_onpolicy_initializes_separate_discriminator_adapter_and_optimizer(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "stage": "onpolicy", "rollout_step_counts": [2]},
        )

        self.assertIn("anyflow_discriminator", model.component.peft_config)
        self.assertEqual(model.component.active_adapter, "default")
        self.assertIsNotNone(distiller.discriminator_optimizer)
        self.assertEqual(distiller.discriminator_optimizer.defaults["betas"], (0.0, 0.999))

    def test_onpolicy_rollout_integrates_h3_dataward_predictions_in_noise_coordinate(self):
        model = _DatawardRoleFlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "stage": "onpolicy", "rollout_step_counts": [2]},
        )
        batch = _prepared_batch()
        batch["timesteps"] = torch.tensor([0.0, 0.5])

        with patch.object(distiller, "_randn_like", return_value=torch.zeros_like(batch["latents"])):
            with distiller._adapter_role("student"):
                result = distiller._training_rollout(batch, step_count=2, grad_timestep=1)

        self.assertTrue(torch.allclose(result, torch.full_like(result, -7.0)))
        self.assertEqual(len(model.teacher_timesteps), 2)
        self.assertEqual(model.component.active_adapter, "default")

    def test_onpolicy_generator_loss_backpropagates_only_to_student_adapter(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "stage": "onpolicy", "rollout_step_counts": [2]},
        )
        model.component.adapter_params["anyflow_discriminator"].data.fill_(1.0)
        batch = _prepared_batch()

        loss, logs = distiller._onpolicy_generator_loss(batch)
        loss.backward()

        self.assertGreater(float(loss.detach()), 0.0)
        self.assertIsNotNone(model.component.adapter_params["default"].grad)
        self.assertIsNone(model.component.adapter_params["anyflow_discriminator"].grad)
        self.assertEqual(logs["anyflow_rollout_steps"], 2.0)
        self.assertIn(logs["anyflow_rollout_grad_timestep"], (0.0, 1.0))
        self.assertEqual(model.component.active_adapter, "default")

    def test_distributed_rollout_schedule_consumes_rank_local_draws_before_broadcast(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "stage": "onpolicy", "rollout_step_counts": [2, 4, 8]},
        )
        broadcast_values = [0, 1]

        def broadcast_from_rank_zero(tensor, src):
            self.assertEqual(src, 0)
            tensor.fill_(broadcast_values.pop(0))

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.broadcast", side_effect=broadcast_from_rank_zero) as broadcast,
            patch.object(
                distiller,
                "_randint",
                side_effect=[torch.tensor([2], dtype=torch.long), torch.tensor([0], dtype=torch.long)],
            ) as randint,
        ):
            step_count, grad_timestep = distiller._distributed_rollout_schedule()

        self.assertEqual(step_count, 2)
        self.assertEqual(grad_timestep, 1)
        self.assertEqual(broadcast.call_count, 2)
        self.assertEqual(randint.call_count, 2)
        self.assertEqual(randint.call_args_list[0].args, (3, (1,)))
        self.assertEqual(randint.call_args_list[1].args, (2, (1,)))

    def test_onpolicy_discriminator_step_updates_only_discriminator_adapter(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={
                "model_type": "lora",
                "stage": "onpolicy",
                "rollout_step_counts": [1],
                "discriminator_lr": 0.1,
            },
        )
        batch = _prepared_batch()
        student_before = model.component.adapter_params["default"].detach().clone()
        discriminator_before = model.component.adapter_params["anyflow_discriminator"].detach().clone()

        distiller.discriminator_step(batch)

        self.assertTrue(torch.equal(model.component.adapter_params["default"], student_before))
        self.assertFalse(torch.equal(model.component.adapter_params["anyflow_discriminator"], discriminator_before))
        self.assertEqual(model.component.active_adapter, "default")

    def test_onpolicy_discriminator_gradients_are_averaged_across_ddp_ranks(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "stage": "onpolicy", "rollout_step_counts": [1]},
        )
        parameter = distiller._discriminator_parameters[0]
        parameter.grad = torch.full_like(parameter, 3.0)

        def sum_across_two_ranks(gradient, **kwargs):
            self.assertEqual(kwargs["op"], torch.distributed.ReduceOp.SUM)
            gradient.mul_(2.0)

        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch("torch.distributed.all_reduce", side_effect=sum_across_two_ranks) as all_reduce,
        ):
            distiller._sync_discriminator_gradients()

        all_reduce.assert_called_once()
        self.assertTrue(torch.equal(parameter.grad, torch.full_like(parameter, 3.0)))

    def test_onpolicy_discriminator_checkpoint_round_trip(self):
        model = _FlowModel()
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora", "stage": "onpolicy", "rollout_step_counts": [1]},
        )
        model.component.adapter_params["anyflow_discriminator"].data.fill_(3.0)

        with tempfile.TemporaryDirectory() as directory:
            distiller.on_save_checkpoint(12, directory)
            model.component.adapter_params["anyflow_discriminator"].data.zero_()
            distiller.on_load_checkpoint(directory)

            self.assertTrue(torch.equal(model.component.adapter_params["anyflow_discriminator"], torch.tensor([3.0])))
            self.assertTrue((Path(directory) / "anyflow_discriminator.safetensors").is_file())
            self.assertTrue((Path(directory) / "anyflow_discriminator_optim.pt").is_file())

    def test_factory_creates_anyflow_distiller(self):
        model = _FlowModel()

        distiller = DistillerFactory.create_distiller(
            "anyflow",
            teacher_model=model,
            noise_scheduler=None,
            config={"seed": 123, "distillation_config": {"anyflow": {"stage": "forward"}}},
            model_type="lora",
            prediction_type="flow_matching",
        )

        self.assertIsInstance(distiller, AnyFlowDistiller)
        self.assertEqual(distiller.config["stage"], "forward")
        self.assertEqual(distiller._rng_seed, 123)
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

    def test_validation_scheduler_derives_direct_interval_endpoint(self):
        scheduler = AnyFlowValidationScheduler(_ValidationScheduler())

        r_timestep = scheduler.r_timestep_for(torch.tensor([1000.0, 500.0]))

        self.assertTrue(torch.equal(r_timestep, torch.tensor([500.0, 0.0])))

    def test_validation_scheduler_derives_normalized_interval_endpoint(self):
        scheduler = AnyFlowValidationScheduler(_ValidationScheduler())

        r_timestep = scheduler.r_timestep_for(torch.tensor([1.0, 0.5]))

        self.assertTrue(torch.equal(r_timestep, torch.tensor([0.5, 0.0])))

    def test_validation_scheduler_derives_inverted_normalized_interval_endpoint(self):
        scheduler = AnyFlowValidationScheduler(_ValidationScheduler())

        r_timestep = scheduler.r_timestep_for(torch.tensor([0.0, 0.5]))

        self.assertTrue(torch.equal(r_timestep, torch.tensor([0.5, 1.0])))

    def test_validation_scheduler_reuses_inverted_normalized_mapping(self):
        scheduler = AnyFlowValidationScheduler(_ValidationScheduler())

        scheduler.r_timestep_for(torch.tensor([0.0]))
        r_timestep = scheduler.r_timestep_for(torch.tensor([0.5]))

        self.assertTrue(torch.equal(r_timestep, torch.tensor([1.0])))

    def test_validation_scheduler_advances_native_dataward_timestep(self):
        scheduler = AnyFlowValidationScheduler(_DatawardValidationScheduler(), num_train_timesteps=1000)

        first_endpoint = scheduler.r_timestep_for(torch.tensor([0.0]))
        final_endpoint = scheduler.r_timestep_for(torch.tensor([0.5]))

        self.assertTrue(torch.equal(first_endpoint, torch.tensor([0.5])))
        self.assertTrue(torch.equal(final_endpoint, torch.tensor([1.0])))

    def test_validation_scheduler_wraps_t_parameter_component(self):
        scheduler = AnyFlowValidationScheduler(_ValidationScheduler())
        component = _TParameterComponent()
        pipeline = SimpleNamespace(transformer=component)

        scheduler.install_pipeline_hooks(pipeline, component_names=("transformer",))
        output = pipeline.transformer(torch.zeros(1), torch.tensor([1000.0]), torch.zeros(1))

        self.assertIs(output[0], component.last_r_timestep)
        self.assertTrue(torch.equal(component.last_r_timestep, torch.tensor([500.0])))

    def test_validation_scheduler_replaces_none_timestep_r(self):
        scheduler = AnyFlowValidationScheduler(_ValidationScheduler())
        component = _TimestepRComponent()
        pipeline = SimpleNamespace(transformer=component)

        scheduler.install_pipeline_hooks(pipeline, component_names=("transformer",))
        output = pipeline.transformer(timestep=torch.tensor([1000.0]), timestep_r=None)

        self.assertIs(output[0], component.last_timestep_r)
        self.assertTrue(torch.equal(component.last_timestep_r, torch.tensor([500.0])))

    def test_validation_scheduler_preserves_distinct_h3_video_and_audio_intervals(self):
        video_scheduler = _DatawardValidationScheduler()
        audio_scheduler = _DatawardValidationScheduler()
        video_scheduler.timesteps = torch.tensor([0.0, 0.1, 1.0])
        audio_scheduler.timesteps = torch.tensor([0.0, 0.25, 1.0])
        scheduler = AnyFlowValidationScheduler(video_scheduler)
        scheduler._audio_scheduler = AnyFlowValidationScheduler(audio_scheduler)

        timestep, r_timestep, timestep_indices = scheduler.component_timesteps(
            torch.tensor([0.0]),
            {
                "timestep_indices": torch.tensor([0, 0, 0]),
                "video_indices": torch.tensor([1]),
                "audio_indices": torch.tensor([2]),
                "num_condition_video_rows": 0,
                "num_condition_audio_rows": 0,
            },
        )

        row_timesteps = timestep[timestep_indices]
        row_r_timesteps = r_timestep[timestep_indices]
        self.assertTrue(torch.equal(row_timesteps, torch.tensor([0.0, 0.0, 0.0])))
        self.assertAlmostEqual(row_r_timesteps[0].item(), 0.1)
        self.assertAlmostEqual(row_r_timesteps[1].item(), 0.1)
        self.assertAlmostEqual(row_r_timesteps[2].item(), 0.25)

    def test_validation_scheduler_preserves_underlying_set_timesteps_signature(self):
        scheduler = AnyFlowValidationScheduler(_ValidationScheduler())

        self.assertIn("timesteps", inspect.signature(scheduler.set_timesteps).parameters)

    def test_get_scheduler_installs_validation_pipeline_hook(self):
        model = _FlowModel()
        model.pipeline = SimpleNamespace(transformer=model.component, scheduler=_ValidationScheduler())
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora"},
        )

        scheduler = distiller.get_scheduler(model.pipeline.scheduler)
        model.pipeline.transformer(timestep=torch.tensor([1000.0]))

        self.assertIsInstance(scheduler, AnyFlowValidationScheduler)
        self.assertTrue(torch.equal(model.component.last_r_timestep, torch.tensor([500.0])))

    def test_get_scheduler_wraps_conditional_transformer_pipeline(self):
        model = _FlowModel()
        model.pipeline = SimpleNamespace(conditional_transformer=model.component, scheduler=_ValidationScheduler())
        distiller = AnyFlowDistiller(
            teacher_model=model,
            noise_scheduler=None,
            config={"model_type": "lora"},
        )

        scheduler = distiller.get_scheduler(model.pipeline.scheduler)
        model.pipeline.conditional_transformer(timestep=torch.tensor([1000.0]))

        self.assertIsInstance(scheduler, AnyFlowValidationScheduler)
        self.assertTrue(torch.equal(model.component.last_r_timestep, torch.tensor([500.0])))


if __name__ == "__main__":
    unittest.main()


class AnyFlowUnconditionalBatchTests(unittest.TestCase):
    def test_unconditional_batch_drops_stale_mask_and_model_specific_aliases(self):
        batch = _prepared_batch()
        batch["encoder_attention_mask"] = torch.ones(2, 5, dtype=torch.long)
        batch["prompt_embeds"] = torch.ones(2, 7, 4)
        batch["attention_mask"] = torch.ones(2, 7, dtype=torch.bool)
        batch["attention_masks"] = torch.ones(2, 7, dtype=torch.bool)

        unconditional_batch = AnyFlowDistiller._unconditional_batch(batch)

        self.assertIs(unconditional_batch["encoder_hidden_states"], batch["negative_encoder_hidden_states"])
        self.assertNotIn("encoder_attention_mask", unconditional_batch)
        self.assertNotIn("prompt_embeds", unconditional_batch)
        self.assertNotIn("attention_mask", unconditional_batch)
        self.assertNotIn("attention_masks", unconditional_batch)


class AnyFlowDeltaEmbedderTests(unittest.TestCase):
    def test_clone_flowmap_embedder_is_trainable(self):
        from simpletuner.helpers.models.flowmap import clone_flowmap_embedder

        frozen = torch.nn.Linear(4, 4)
        frozen.requires_grad_(False)
        clone = clone_flowmap_embedder(frozen)

        self.assertTrue(all(p.requires_grad for p in clone.parameters()))
        self.assertFalse(any(p.requires_grad for p in frozen.parameters()))

    def test_sidecar_prefixes_cover_all_family_namings(self):
        from simpletuner.helpers.training.adapter import ANYFLOW_SIDECAR_PREFIXES

        for naming in (
            "condition_embedder.delta_embedder.linear_1.weight",
            "delta_adaln_embedder.table",
            "delta_time_embedder.linear_1.weight",
            "delta_timestep_embedder.linear_1.weight",
            "delta_t_embedding.mlp_in.weight",
        ):
            with self.subTest(naming=naming):
                self.assertTrue(naming.startswith(ANYFLOW_SIDECAR_PREFIXES))

    def test_collect_anyflow_sidecar_state(self):
        from simpletuner.helpers.training.save_hooks import _collect_anyflow_sidecar_state

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.delta_t_embedding = torch.nn.Linear(2, 2)
                self.other = torch.nn.Linear(2, 2)

        collected = _collect_anyflow_sidecar_state(Toy())
        self.assertEqual(
            sorted(collected),
            ["delta_t_embedding.bias", "delta_t_embedding.weight"],
        )

    def test_collect_anyflow_sidecar_state_excludes_peft_adapter_aliases(self):
        from simpletuner.helpers.training.save_hooks import _collect_anyflow_sidecar_state

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.delta_t_embedding = torch.nn.Module()
                self.delta_t_embedding.linear_1 = torch.nn.Linear(2, 2)

        model = Toy().requires_grad_(False)
        model = inject_adapter_in_model(
            LoraConfig(r=1, lora_alpha=1, target_modules=["delta_t_embedding.linear_1"]),
            model,
        )
        state = get_peft_model_state_dict(model)
        state.update(_collect_anyflow_sidecar_state(model))

        self.assertEqual(
            sorted(name for name in state if "delta_t_embedding" in name),
            [
                "delta_t_embedding.linear_1.bias",
                "delta_t_embedding.linear_1.lora_A.weight",
                "delta_t_embedding.linear_1.lora_B.weight",
                "delta_t_embedding.linear_1.weight",
            ],
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            save_file(state, str(Path(temp_dir) / "adapter.safetensors"))
