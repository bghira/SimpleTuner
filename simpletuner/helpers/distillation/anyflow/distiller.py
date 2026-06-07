from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.distillation.anyflow.scheduler import AnyFlowValidationScheduler
from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.registry import DistillationRegistry
from simpletuner.helpers.models.flowmap import validate_flowmap_deltatime_type


class AnyFlowDistiller(DistillationBase):
    """Online AnyFlow/FlowMap target preparation for flow-matching models."""

    FLOWMAP_R_TIMESTEP_BATCH_KEY = "flowmap_r_timesteps"

    _DEFAULTS: Dict[str, Any] = {
        "distillation_type": "anyflow",
        "target_mode": "online_teacher",
        "r_timestep_sampler": "uniform",
        "min_interval_ratio": 0.02,
        "teacher_rollout_steps": 1,
        "gate_value": 0.25,
        "deltatime_type": "r",
        "loss_weight": 1.0,
        "timestep_scale": None,
    }
    _TARGET_MODE_ALIASES = {
        "online": "online_teacher",
        "online_teacher": "online_teacher",
        "teacher": "online_teacher",
        "linear": "linear",
        "straight": "linear",
    }

    def __init__(
        self,
        teacher_model,
        student_model=None,
        *,
        noise_scheduler=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        merged_config = dict(self._DEFAULTS)
        if config:
            merged_config.update(config)

        super().__init__(teacher_model, student_model, merged_config)
        self.noise_scheduler = noise_scheduler
        self.num_train_timesteps = self._resolve_timestep_scale(noise_scheduler)

        if not self.is_flow_matching:
            raise ValueError("AnyFlow requires a flow-matching model.")

        self.config["target_mode"] = self._normalize_target_mode(self.config["target_mode"])
        self.config["r_timestep_sampler"] = self._normalize_r_timestep_sampler(self.config["r_timestep_sampler"])
        self.config["deltatime_type"] = validate_flowmap_deltatime_type(
            str(self.config["deltatime_type"]),
            model_name="AnyFlow",
        )
        self.config["teacher_rollout_steps"] = max(1, int(self.config.get("teacher_rollout_steps", 1) or 1))

        min_interval_ratio = float(self.config.get("min_interval_ratio", 0.0) or 0.0)
        if min_interval_ratio < 0.0 or min_interval_ratio >= 1.0:
            raise ValueError("AnyFlow min_interval_ratio must be in [0.0, 1.0).")
        self.config["min_interval_ratio"] = min_interval_ratio

        model_type = str(self.config.get("model_type") or "").lower()
        if self.config["target_mode"] == "online_teacher" and self.low_rank_distillation and model_type not in ("", "lora"):
            raise ValueError("AnyFlow online_teacher mode requires low-rank training or a separate student_model.")

        self._flowmap_component = self._enable_flowmap_time_conditioning()

    def prepare_batch(self, batch: Dict[str, Any], model, state) -> Dict[str, Any]:
        self._validate_prepared_batch(batch)
        del state

        latents = batch["latents"]
        timesteps = batch["timesteps"].to(device=latents.device, dtype=torch.float32)
        if timesteps.ndim != 1:
            raise ValueError(
                "AnyFlow currently expects per-sample scalar flow timesteps. "
                f"Received timestep shape {tuple(timesteps.shape)}."
            )

        t_sigmas = self._scalar_sigmas(batch)
        r_sigmas = self._sample_r_sigmas(t_sigmas)
        self._validate_interval(t_sigmas, r_sigmas)
        r_timesteps = self._timesteps_from_sigmas(r_sigmas, timesteps)
        r_timesteps = r_timesteps.to(device=batch["timesteps"].device, dtype=batch["timesteps"].dtype)

        target = self._base_flow_target(batch)
        if self.config["target_mode"] == "online_teacher":
            target = self._online_teacher_average_velocity(
                prepared_batch=batch,
                t_sigmas=t_sigmas,
                r_sigmas=r_sigmas,
                base_target=target,
            )

        flowmap_key = getattr(model, "FLOWMAP_R_TIMESTEP_BATCH_KEY", self.FLOWMAP_R_TIMESTEP_BATCH_KEY)
        batch[flowmap_key] = r_timesteps
        batch["anyflow_r_timesteps"] = r_timesteps
        batch["anyflow_timestep_interval"] = batch["timesteps"].to(dtype=r_timesteps.dtype) - r_timesteps
        batch["target"] = target.detach()
        batch["flow_target"] = batch["target"]
        return batch

    def compute_distill_loss(
        self,
        prepared_batch: Dict[str, Any],
        model_output: Dict[str, Any],
        original_loss: torch.Tensor,
    ):
        del model_output

        loss = original_loss * float(self.config.get("loss_weight", 1.0))
        r_timesteps = prepared_batch.get("anyflow_r_timesteps", prepared_batch.get(self.FLOWMAP_R_TIMESTEP_BATCH_KEY))
        logs = {
            "anyflow_loss": float(loss.detach()),
            "anyflow_timestep": float(torch.mean(prepared_batch["timesteps"].float()).detach()),
            "total": float(loss.detach()),
        }
        if r_timesteps is not None:
            logs["anyflow_r_timestep"] = float(torch.mean(r_timesteps.float()).detach())
            logs["anyflow_interval"] = float(
                torch.mean((prepared_batch["timesteps"].float() - r_timesteps.float())).detach()
            )
        return loss, logs

    def get_scheduler(self, scheduler=None):
        pipeline = getattr(self.teacher_model, "pipeline", None)
        base_scheduler = scheduler
        if base_scheduler is None and pipeline is not None:
            base_scheduler = getattr(pipeline, "scheduler", None)
        if base_scheduler is None:
            base_scheduler = self.noise_scheduler
        if base_scheduler is None:
            raise ValueError("AnyFlow validation requires an inference scheduler on the validation pipeline.")

        validation_scheduler = AnyFlowValidationScheduler(
            base_scheduler,
            num_train_timesteps=self.num_train_timesteps,
        )
        if pipeline is not None:
            validation_scheduler.install_pipeline_hooks(
                pipeline,
                component_names=self._validation_component_names(),
            )
        return validation_scheduler

    def _validation_component_names(self) -> tuple[str, ...]:
        names: list[str] = []
        model_type = getattr(getattr(self.teacher_model, "MODEL_TYPE", None), "value", None)
        if isinstance(model_type, str) and model_type:
            names.append(model_type)
        for fallback_name in ("transformer", "conditional_transformer", "unet"):
            if fallback_name not in names:
                names.append(fallback_name)
        return tuple(names)

    @staticmethod
    def _normalize_target_mode(value: Any) -> str:
        mode = str(value).strip().lower().replace("-", "_")
        try:
            return AnyFlowDistiller._TARGET_MODE_ALIASES[mode]
        except KeyError as exc:
            raise ValueError("AnyFlow target_mode must be one of: online_teacher, linear.") from exc

    @staticmethod
    def _normalize_r_timestep_sampler(value: Any) -> str:
        sampler = str(value).strip().lower().replace("-", "_")
        if sampler not in {"uniform", "zero"}:
            raise ValueError("AnyFlow r_timestep_sampler must be one of: uniform, zero.")
        return sampler

    def _resolve_timestep_scale(self, noise_scheduler) -> float:
        configured = self.config.get("timestep_scale")
        if configured not in (None, ""):
            return float(configured)
        scheduler_config = getattr(noise_scheduler, "config", None)
        return float(getattr(scheduler_config, "num_train_timesteps", 1000))

    def _enable_flowmap_time_conditioning(self):
        component = self._get_trained_component(self.student_model)
        enable_flowmap = getattr(component, "enable_flowmap_time_conditioning", None)
        if not callable(enable_flowmap):
            raise ValueError(
                "AnyFlow requires model-specific FlowMap interval conditioning. "
                "Add enable_flowmap_time_conditioning() to the trained component before enabling AnyFlow."
            )
        enable_flowmap(
            gate_value=float(self.config.get("gate_value", 0.25)),
            deltatime_type=self.config["deltatime_type"],
        )
        return component

    @staticmethod
    def _get_trained_component(model):
        getter = getattr(model, "get_trained_component", None)
        if callable(getter):
            try:
                return getter(unwrap_model=True)
            except TypeError:
                return getter()
        component = getattr(model, "model", None)
        if component is not None:
            return component
        raise ValueError("AnyFlow requires a model with get_trained_component() or a `.model` component.")

    @staticmethod
    def _validate_prepared_batch(batch: Dict[str, Any]) -> None:
        required_keys = ("latents", "noise", "noisy_latents", "sigmas", "timesteps")
        missing = [key for key in required_keys if key not in batch or batch[key] is None]
        if missing:
            raise ValueError(f"AnyFlow prepared batch is missing required fields: {', '.join(missing)}.")
        for key in required_keys:
            if not torch.is_tensor(batch[key]):
                raise ValueError(f"AnyFlow prepared batch field `{key}` must be a tensor.")

    @staticmethod
    def _broadcast_time(time_tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        while time_tensor.ndim < like.ndim:
            time_tensor = time_tensor.unsqueeze(-1)
        return time_tensor

    @staticmethod
    def _scalar_sigmas(prepared_batch: Dict[str, Any]) -> torch.Tensor:
        sigmas = prepared_batch["sigmas"]
        batch_size = prepared_batch["latents"].shape[0]
        sigmas = sigmas.to(device=prepared_batch["latents"].device, dtype=torch.float32)
        if sigmas.ndim == 0:
            return sigmas.expand(batch_size)
        if sigmas.shape[0] != batch_size:
            raise ValueError(f"AnyFlow expected {batch_size} sigma values, got shape {tuple(sigmas.shape)}.")
        return sigmas.reshape(batch_size, -1)[:, 0].clamp(0.0, 1.0)

    def _timesteps_from_sigmas(self, sigmas: torch.Tensor, reference_timesteps: torch.Tensor) -> torch.Tensor:
        if torch.max(reference_timesteps.detach().float()) <= 1.0:
            return sigmas
        return sigmas * self.num_train_timesteps

    def _sample_r_sigmas(self, t_sigmas: torch.Tensor) -> torch.Tensor:
        sampler = self.config["r_timestep_sampler"]
        if sampler == "zero":
            return torch.zeros_like(t_sigmas)

        min_interval = float(self.config["min_interval_ratio"])
        max_r = (t_sigmas - min_interval).clamp_min(0.0)
        return torch.rand_like(t_sigmas) * max_r

    @staticmethod
    def _validate_interval(t_sigmas: torch.Tensor, r_sigmas: torch.Tensor) -> None:
        interval = t_sigmas - r_sigmas
        eps = torch.finfo(interval.dtype).eps if torch.is_floating_point(interval) else 0.0
        if torch.any(interval <= eps):
            raise ValueError(
                "AnyFlow requires r_timestep < timestep. Avoid timestep 0 in flow_custom_timesteps "
                "or use a sampler that leaves a positive interval."
            )

    @staticmethod
    def _base_flow_target(prepared_batch: Dict[str, Any]) -> torch.Tensor:
        flow_target = prepared_batch.get("flow_target")
        if torch.is_tensor(flow_target):
            return flow_target.to(device=prepared_batch["latents"].device, dtype=prepared_batch["latents"].dtype)
        return prepared_batch["noise"] - prepared_batch["latents"]

    def _online_teacher_average_velocity(
        self,
        *,
        prepared_batch: Dict[str, Any],
        t_sigmas: torch.Tensor,
        r_sigmas: torch.Tensor,
        base_target: torch.Tensor,
    ) -> torch.Tensor:
        start_latents = prepared_batch["noisy_latents"].detach()
        current_latents = start_latents
        current_sigmas = t_sigmas

        try:
            self.toggle_adapter(enable=False)
            with torch.no_grad():
                for next_sigmas in self._rollout_sigma_schedule(t_sigmas, r_sigmas):
                    teacher_batch = self._teacher_batch(prepared_batch, current_latents, current_sigmas)
                    teacher_prediction = self.teacher_model.model_predict(teacher_batch)["model_prediction"]
                    teacher_prediction = teacher_prediction.to(device=current_latents.device, dtype=current_latents.dtype)
                    step = self._broadcast_time(next_sigmas - current_sigmas, current_latents)
                    current_latents = current_latents + step * teacher_prediction
                    current_sigmas = next_sigmas
        finally:
            self.toggle_adapter(enable=True)

        denominator = self._broadcast_time(r_sigmas - t_sigmas, current_latents)
        average_velocity = (current_latents - start_latents) / denominator
        return average_velocity.to(device=base_target.device, dtype=base_target.dtype)

    def _rollout_sigma_schedule(self, t_sigmas: torch.Tensor, r_sigmas: torch.Tensor):
        steps = int(self.config["teacher_rollout_steps"])
        for index in range(steps):
            ratio = float(index + 1) / float(steps)
            yield t_sigmas + (r_sigmas - t_sigmas) * ratio

    def _teacher_batch(
        self,
        prepared_batch: Dict[str, Any],
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> Dict[str, Any]:
        teacher_batch = dict(prepared_batch)
        teacher_batch.pop("target", None)
        teacher_batch.pop("flow_target", None)
        teacher_batch.pop(self.FLOWMAP_R_TIMESTEP_BATCH_KEY, None)
        teacher_batch.pop("anyflow_r_timesteps", None)
        teacher_batch.pop("anyflow_timestep_interval", None)

        teacher_batch["noisy_latents"] = noisy_latents
        teacher_batch["timesteps"] = self._timesteps_from_sigmas(
            sigmas,
            prepared_batch["timesteps"].to(device=sigmas.device, dtype=torch.float32),
        ).to(
            device=prepared_batch["timesteps"].device,
            dtype=prepared_batch["timesteps"].dtype,
        )
        teacher_batch["sigmas"] = self._broadcast_time(sigmas, noisy_latents).to(
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
        return teacher_batch


DistillationRegistry.register(
    "anyflow",
    AnyFlowDistiller,
    requires_distillation_cache=False,
    data_requirements=[[DatasetType.IMAGE, DatasetType.VIDEO]],
    requirement_notes="Requires model-specific FlowMap interval conditioning on the trained component.",
)
