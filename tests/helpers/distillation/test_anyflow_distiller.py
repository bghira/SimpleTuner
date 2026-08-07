import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

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

    def test_removed_legacy_target_modes_are_rejected(self):
        for target_mode in ("online_teacher", "linear"):
            with self.subTest(target_mode=target_mode), self.assertRaisesRegex(ValueError, "target_mode was removed"):
                AnyFlowDistiller(
                    teacher_model=_FlowModel(),
                    noise_scheduler=None,
                    config={"model_type": "lora", "target_mode": target_mode},
                )

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

        with patch("torch.randn_like", return_value=torch.zeros_like(batch["latents"])):
            with distiller._adapter_role("student"):
                result = distiller._training_rollout(batch, step_count=2)

        self.assertTrue(torch.allclose(result, torch.full_like(result, -7.0)))
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
        self.assertEqual(model.component.active_adapter, "default")

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
            config={"distillation_config": {"anyflow": {"stage": "forward"}}},
            model_type="lora",
            prediction_type="flow_matching",
        )

        self.assertIsInstance(distiller, AnyFlowDistiller)
        self.assertEqual(distiller.config["stage"], "forward")
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
