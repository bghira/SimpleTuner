from __future__ import annotations

import copy
import os
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.data_backend.runtime.context_parallel_sync import (
    gather_variable_batch_tensor,
    resolve_distributed_batch_layout,
)
from simpletuner.helpers.distillation.anyflow.scheduler import AnyFlowValidationScheduler
from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.registry import DistillationRegistry
from simpletuner.helpers.models.flowmap import validate_flowmap_deltatime_type

ANYFLOW_DISCRIMINATOR_FILENAME = "anyflow_discriminator.safetensors"
ANYFLOW_DISCRIMINATOR_OPTIMIZER_FILENAME = "anyflow_discriminator_optim.pt"


class AnyFlowDistiller(DistillationBase):
    """NVIDIA AnyFlow forward MeanFlow and on-policy DMD training stages."""

    FLOWMAP_R_TIMESTEP_BATCH_KEY = "flowmap_r_timesteps"
    DISCRIMINATOR_ADAPTER_NAME = "anyflow_discriminator"

    _DEFAULTS: Dict[str, Any] = {
        "distillation_type": "anyflow",
        "stage": "forward",
        "gate_value": 0.25,
        "deltatime_type": "r",
        "loss_weight": 1.0,
        "timestep_scale": None,
        "schedule_shift": None,
        "diffusion_ratio": 0.5,
        "consistency_ratio": 0.25,
        "central_difference_epsilon": 0.005,
        "central_difference_boundary_mode": "extrapolate",
        "fuse_guidance_scale": 3.0,
        "meanflow_weight_type": "beta08",
        "meanflow_adaptive_weighting": True,
        "meanflow_non_diffusion_max_sigma": 1.0,
        "diffusion_target": "flow",
        "cotrain_forward": True,
        "rollout_step_counts": (2, 4, 8, 16, 50),
        "dmd_weight": 1.0,
        "dmd_batch_size": 1,
        "dmd_min_timestep_ratio": 0.0,
        "dmd_max_timestep_ratio": 1.0,
        "real_score_guidance_scale": 0.0,
        "discriminator_lr": 2e-6,
        "discriminator_weight_decay": 0.0,
        "discriminator_betas": (0.0, 0.999),
        "discriminator_eps": 1e-8,
        "discriminator_grad_clip": 1.0,
        "student_adapter_name": None,
        "discriminator_adapter_name": DISCRIMINATOR_ADAPTER_NAME,
    }

    @classmethod
    def prepare_model_for_adapter(cls, model, config: Dict[str, Any]) -> None:
        model_config = getattr(model, "config", None)
        if isinstance(model_config, dict):
            lora_dropout = model_config.get("lora_dropout", 0.0)
        else:
            lora_dropout = getattr(model_config, "lora_dropout", 0.0)
        lora_dropout = float(lora_dropout or 0.0)
        if lora_dropout != 0.0:
            raise ValueError(
                "AnyFlow requires lora_dropout=0.0. Independent dropout masks in the finite-difference forwards "
                "corrupt the derivative target."
            )

        component = cls._get_trained_component(model)
        enable_flowmap = getattr(component, "enable_flowmap_time_conditioning", None)
        if not callable(enable_flowmap):
            raise ValueError(
                "AnyFlow requires model-specific FlowMap interval conditioning. "
                "Add enable_flowmap_time_conditioning() to the trained component before enabling AnyFlow."
            )
        enable_flowmap(
            gate_value=float(config.get("gate_value", cls._DEFAULTS["gate_value"])),
            deltatime_type=str(config.get("deltatime_type", cls._DEFAULTS["deltatime_type"])),
        )

    @classmethod
    def training_batch_requirements(cls, config: Dict[str, Any]) -> set[str]:
        fuse_guidance = float(config.get("fuse_guidance_scale", cls._DEFAULTS["fuse_guidance_scale"]))
        real_guidance = float(config.get("real_score_guidance_scale", cls._DEFAULTS["real_score_guidance_scale"]))
        return {"unconditional_text_embeddings"} if fuse_guidance != 1.0 or real_guidance != 0.0 else set()

    def __init__(
        self,
        teacher_model,
        student_model=None,
        *,
        noise_scheduler=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        if config and "target_mode" in config:
            raise ValueError(
                "AnyFlow target_mode was removed. Use stage='forward' for NVIDIA MeanFlow pretraining or "
                "stage='onpolicy' for NVIDIA's on-policy DMD stage."
            )

        merged_config = dict(self._DEFAULTS)
        if config:
            merged_config.update(config)
        super().__init__(teacher_model, student_model, merged_config)

        self.noise_scheduler = noise_scheduler
        self.num_train_timesteps = self._resolve_timestep_scale(noise_scheduler)
        if not self.is_flow_matching:
            raise ValueError("AnyFlow requires a flow-matching model.")

        self.config["stage"] = self._normalize_stage(self.config["stage"])
        self.config["deltatime_type"] = validate_flowmap_deltatime_type(
            str(self.config["deltatime_type"]),
            model_name="AnyFlow",
        )
        self._validate_forward_config()
        self._flowmap_component = self._enable_flowmap_time_conditioning()
        self._delta_initial_parameters = {
            name: parameter.detach().float().cpu().clone() for name, parameter in self._trainable_delta_parameters()
        }
        self._rng_seed = self._resolve_rng_seed()
        self._rng_generators: Dict[str, torch.Generator] = {}

        self._student_adapter_name: Optional[str] = None
        self._discriminator_adapter_name: Optional[str] = None
        self._current_adapter_role = "student"
        self._discriminator_parameters: list[torch.nn.Parameter] = []
        self.discriminator_optimizer: Optional[torch.optim.Optimizer] = None
        if self.config["stage"] == "onpolicy":
            self._init_onpolicy_stage()

    # ------------------------------------------------------------------
    # DistillationBase API
    # ------------------------------------------------------------------
    def prepare_batch(self, batch: Dict[str, Any], model, state) -> Dict[str, Any]:
        del state
        self._validate_video_batch(batch)
        timesteps = batch["timesteps"]
        if timesteps.ndim != 1:
            raise ValueError(
                "AnyFlow currently expects per-sample scalar flow timesteps. "
                f"Received timestep shape {tuple(timesteps.shape)}."
            )

        t_sigmas, r_sigmas = self._prepare_meanflow_pair(batch, model)
        r_timesteps = self._timesteps_from_sigmas(r_sigmas, timesteps).to(
            device=timesteps.device,
            dtype=timesteps.dtype,
        )
        flowmap_key = getattr(model, "FLOWMAP_R_TIMESTEP_BATCH_KEY", self.FLOWMAP_R_TIMESTEP_BATCH_KEY)
        batch[flowmap_key] = r_timesteps
        batch["anyflow_r_timesteps"] = r_timesteps
        batch["anyflow_timestep_interval"] = (batch["timesteps"].to(r_timesteps.dtype) - r_timesteps).abs()

        base_target = self._base_flow_target(batch, model=model)
        meanflow_base_target = self._meanflow_base_target(
            prepared_batch=batch,
            t_sigmas=t_sigmas,
            base_target=base_target,
        )
        target = self._meanflow_target(
            prepared_batch=batch,
            model=model,
            t_sigmas=t_sigmas,
            r_sigmas=r_sigmas,
            base_target=meanflow_base_target,
        ).detach()
        batch["target"] = target
        batch["flow_target"] = target

        target_base_cosine, target_base_norm_ratio = self._chunked_per_sample_geometry(target, base_target)
        batch["_anyflow_target_base_cosine"] = target_base_cosine
        batch["_anyflow_target_base_norm_ratio"] = target_base_norm_ratio
        return batch

    def compute_distill_loss(
        self,
        prepared_batch: Dict[str, Any],
        model_output: Dict[str, Any],
        original_loss: torch.Tensor,
    ):
        del original_loss
        forward_loss = self._meanflow_loss(prepared_batch, model_output)
        forward_loss = forward_loss * float(self.config["loss_weight"])
        total_loss = forward_loss
        logs = self._forward_logs(prepared_batch, forward_loss)

        if self.config["stage"] == "onpolicy":
            if not bool(self.config["cotrain_forward"]):
                total_loss = forward_loss.new_zeros(())
            dmd_loss, dmd_logs = self._onpolicy_generator_loss(prepared_batch)
            total_loss = total_loss + dmd_loss
            logs.update(dmd_logs)

        logs["total"] = float(total_loss.detach())
        return total_loss, logs

    def discriminator_step(self, prepared_batch: Dict[str, Any], **_: Any) -> None:
        if self.config["stage"] != "onpolicy" or self.discriminator_optimizer is None:
            return

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        with self._adapter_role("discriminator"):
            discriminator_loss = self._onpolicy_discriminator_loss(prepared_batch)
            discriminator_loss.backward()
        self._sync_discriminator_gradients()
        torch.nn.utils.clip_grad_norm_(
            self._discriminator_parameters,
            float(self.config["discriminator_grad_clip"]),
        )
        self.discriminator_optimizer.step()
        self.discriminator_optimizer.zero_grad(set_to_none=True)

    def on_save_checkpoint(self, step: int, ckpt_dir: str) -> None:
        if self.config["stage"] != "onpolicy" or self.discriminator_optimizer is None:
            return
        accelerator = getattr(self.teacher_model, "accelerator", None)
        if accelerator is not None and not bool(getattr(accelerator, "is_main_process", True)):
            return

        os.makedirs(ckpt_dir, exist_ok=True)
        state = self._discriminator_state_dict()
        save_file(state, os.path.join(ckpt_dir, ANYFLOW_DISCRIMINATOR_FILENAME))
        torch.save(
            {"step": int(step), "state": self.discriminator_optimizer.state_dict()},
            os.path.join(ckpt_dir, ANYFLOW_DISCRIMINATOR_OPTIMIZER_FILENAME),
        )

    def on_load_checkpoint(self, ckpt_dir: str) -> None:
        if self.config["stage"] != "onpolicy" or self.discriminator_optimizer is None:
            return

        weight_path = os.path.join(ckpt_dir, ANYFLOW_DISCRIMINATOR_FILENAME)
        if os.path.exists(weight_path):
            self._load_discriminator_state_dict(load_file(weight_path, device="cpu"))

        optimizer_path = os.path.join(ckpt_dir, ANYFLOW_DISCRIMINATOR_OPTIMIZER_FILENAME)
        if os.path.exists(optimizer_path):
            payload = torch.load(optimizer_path, map_location="cpu", weights_only=True)
            self.discriminator_optimizer.load_state_dict(payload["state"])
            self._move_optimizer_state(self.discriminator_optimizer, self._device)

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

    def supports_special_scheduler_validation(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Configuration and adapter roles
    # ------------------------------------------------------------------
    def _resolve_rng_seed(self) -> Optional[int]:
        seed = self.config.get("seed")
        if seed in (None, "", 0):
            seed = getattr(getattr(self.teacher_model, "config", None), "seed", None)
        if seed in (None, "", 0):
            return None

        seed = int(seed)
        seed_for_each_device = self.config.get("seed_for_each_device")
        if seed_for_each_device is None:
            seed_for_each_device = getattr(getattr(self.teacher_model, "config", None), "seed_for_each_device", False)
        if bool(seed_for_each_device):
            accelerator = getattr(self.teacher_model, "accelerator", None)
            seed += int(getattr(accelerator, "process_index", 0) or 0)
        return seed

    @staticmethod
    def _canonical_rng_device(device: torch.device | str) -> torch.device:
        torch_device = torch.device(device)
        if torch_device.type == "cuda" and torch_device.index is None and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch_device

    def _rng_generator(self, device: torch.device | str) -> Optional[torch.Generator]:
        if self._rng_seed is None:
            return None

        torch_device = self._canonical_rng_device(device)
        key = str(torch_device)
        generator = self._rng_generators.get(key)
        if generator is None:
            generator = torch.Generator(device=torch_device)
            generator.manual_seed(self._rng_seed)
            self._rng_generators[key] = generator
        return generator

    def _rand(
        self,
        size: Sequence[int] | torch.Size,
        *,
        device: torch.device | str,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        torch_device = self._canonical_rng_device(device)
        kwargs: Dict[str, Any] = {"device": torch_device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        generator = self._rng_generator(torch_device)
        if generator is not None:
            kwargs["generator"] = generator
        return torch.rand(size, **kwargs)

    def _randn(
        self,
        size: Sequence[int] | torch.Size,
        *,
        device: torch.device | str,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        torch_device = self._canonical_rng_device(device)
        kwargs: Dict[str, Any] = {"device": torch_device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        generator = self._rng_generator(torch_device)
        if generator is not None:
            kwargs["generator"] = generator
        return torch.randn(size, **kwargs)

    def _randn_like(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._randn(tuple(tensor.shape), device=tensor.device, dtype=tensor.dtype)

    def _randint(
        self,
        high: int,
        size: Sequence[int] | torch.Size,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        torch_device = self._canonical_rng_device(device)
        kwargs: Dict[str, Any] = {"device": torch_device, "dtype": dtype}
        generator = self._rng_generator(torch_device)
        if generator is not None:
            kwargs["generator"] = generator
        return torch.randint(high, size, **kwargs)

    @property
    def _device(self) -> torch.device:
        accelerator = getattr(self.teacher_model, "accelerator", None)
        return torch.device(getattr(accelerator, "device", "cpu"))

    @staticmethod
    def _normalize_stage(value: Any) -> str:
        stage = str(value).strip().lower().replace("-", "_")
        aliases = {"forward": "forward", "meanflow": "forward", "pretrain": "forward", "onpolicy": "onpolicy"}
        try:
            return aliases[stage]
        except KeyError as exc:
            raise ValueError("AnyFlow stage must be one of: forward, onpolicy.") from exc

    def _validate_forward_config(self) -> None:
        diffusion_ratio = float(self.config["diffusion_ratio"])
        consistency_ratio = float(self.config["consistency_ratio"])
        if diffusion_ratio < 0.0 or consistency_ratio < 0.0 or diffusion_ratio + consistency_ratio > 1.0:
            raise ValueError("AnyFlow diffusion_ratio and consistency_ratio must be non-negative and sum to at most 1.")
        self.config["diffusion_ratio"] = diffusion_ratio
        self.config["consistency_ratio"] = consistency_ratio

        schedule_shift = self.config.get("schedule_shift")
        if schedule_shift not in (None, ""):
            schedule_shift = float(schedule_shift)
            if schedule_shift <= 0.0:
                raise ValueError("AnyFlow schedule_shift must be greater than zero.")
            self.config["schedule_shift"] = schedule_shift

        epsilon = float(self.config["central_difference_epsilon"])
        if epsilon <= 0.0 or epsilon >= 0.5:
            raise ValueError("AnyFlow central_difference_epsilon must be in (0.0, 0.5).")
        self.config["central_difference_epsilon"] = epsilon

        boundary_mode = str(self.config["central_difference_boundary_mode"]).strip().lower()
        if boundary_mode not in {"extrapolate", "clamp"}:
            raise ValueError("AnyFlow central_difference_boundary_mode must be one of: extrapolate, clamp.")
        self.config["central_difference_boundary_mode"] = boundary_mode

        non_diffusion_max_sigma = float(self.config["meanflow_non_diffusion_max_sigma"])
        if non_diffusion_max_sigma <= 0.0 or non_diffusion_max_sigma > 1.0:
            raise ValueError("AnyFlow meanflow_non_diffusion_max_sigma must be in (0.0, 1.0].")
        self.config["meanflow_non_diffusion_max_sigma"] = non_diffusion_max_sigma

        diffusion_target = str(self.config["diffusion_target"]).strip().lower()
        if diffusion_target not in {"flow", "base_prediction"}:
            raise ValueError("AnyFlow diffusion_target must be one of: flow, base_prediction.")
        if diffusion_target == "base_prediction" and not self.low_rank_distillation:
            raise ValueError("AnyFlow diffusion_target=base_prediction currently requires adapter distillation.")
        self.config["diffusion_target"] = diffusion_target

        guidance_scale = float(self.config["fuse_guidance_scale"])
        if guidance_scale <= 0.0:
            raise ValueError("AnyFlow fuse_guidance_scale must be greater than zero.")
        if diffusion_target == "base_prediction" and guidance_scale != 1.0:
            raise ValueError("AnyFlow diffusion_target=base_prediction requires fuse_guidance_scale=1.0.")
        self.config["fuse_guidance_scale"] = guidance_scale

        weight_type = str(self.config["meanflow_weight_type"]).strip().lower()
        if weight_type not in {"beta08", "uniform"}:
            raise ValueError("AnyFlow meanflow_weight_type must be one of: beta08, uniform.")
        self.config["meanflow_weight_type"] = weight_type

    def _init_onpolicy_stage(self) -> None:
        if not bool(self.config["cotrain_forward"]):
            raise ValueError("NVIDIA AnyFlow on-policy training requires cotrain_forward=true in SimpleTuner.")
        if not self.low_rank_distillation or str(self.config.get("model_type", "lora")).lower() != "lora":
            raise ValueError("AnyFlow on-policy training currently requires a standard PEFT LoRA student.")
        lora_type = str(getattr(self.teacher_model.config, "lora_type", "standard")).lower()
        if lora_type != "standard":
            raise ValueError("AnyFlow on-policy training currently requires lora_type=standard.")

        steps = self._parse_step_counts(self.config["rollout_step_counts"])
        self.config["rollout_step_counts"] = steps
        dmd_batch_size = int(self.config["dmd_batch_size"])
        if dmd_batch_size < 1:
            raise ValueError("AnyFlow dmd_batch_size must be at least 1.")
        self.config["dmd_batch_size"] = dmd_batch_size

        min_ratio = float(self.config["dmd_min_timestep_ratio"])
        max_ratio = float(self.config["dmd_max_timestep_ratio"])
        if min_ratio < 0.0 or max_ratio > 1.0 or max_ratio <= min_ratio:
            raise ValueError("AnyFlow DMD timestep ratios must satisfy 0 <= min < max <= 1.")

        component = self._flowmap_component
        peft_config = getattr(component, "peft_config", None)
        if not isinstance(peft_config, dict) or not peft_config:
            raise ValueError("AnyFlow on-policy training requires an initialized PEFT adapter on the student.")

        configured_student = self.config.get("student_adapter_name")
        self._student_adapter_name = str(configured_student or next(iter(peft_config)))
        if self._student_adapter_name not in peft_config:
            raise ValueError(f"AnyFlow student adapter {self._student_adapter_name!r} does not exist.")

        discriminator_name = str(self.config["discriminator_adapter_name"])
        if discriminator_name == self._student_adapter_name:
            raise ValueError("AnyFlow discriminator_adapter_name must differ from the student adapter name.")
        if discriminator_name not in peft_config:
            component.add_adapter(copy.deepcopy(peft_config[self._student_adapter_name]), adapter_name=discriminator_name)
        self._discriminator_adapter_name = discriminator_name

        self._discriminator_parameters = [
            parameter
            for name, parameter in component.named_parameters()
            if self._parameter_belongs_to_adapter(name, discriminator_name)
        ]
        if not self._discriminator_parameters:
            raise ValueError("AnyFlow could not locate the discriminator adapter parameters.")

        self.discriminator_optimizer = torch.optim.AdamW(
            self._discriminator_parameters,
            lr=float(self.config["discriminator_lr"]),
            betas=tuple(float(value) for value in self.config["discriminator_betas"]),
            weight_decay=float(self.config["discriminator_weight_decay"]),
            eps=float(self.config["discriminator_eps"]),
        )
        component.set_adapter(self._student_adapter_name)

    @staticmethod
    def _parse_step_counts(value: str | Sequence[int]) -> tuple[int, ...]:
        if isinstance(value, str):
            steps = tuple(int(item.strip()) for item in value.split(",") if item.strip())
        else:
            steps = tuple(int(item) for item in value)
        if not steps or any(step < 1 for step in steps):
            raise ValueError("AnyFlow rollout_step_counts must contain positive integers.")
        return steps

    @staticmethod
    def _parameter_belongs_to_adapter(parameter_name: str, adapter_name: str) -> bool:
        return f".{adapter_name}." in parameter_name or parameter_name.endswith(f".{adapter_name}")

    @contextmanager
    def _adapter_role(self, role: str) -> Iterator[None]:
        if self.config["stage"] != "onpolicy":
            yield
            return

        component = self._flowmap_component
        previous_role = self._current_adapter_role
        was_training = component.training
        try:
            self._set_adapter_role(role)
            yield
        finally:
            self._set_adapter_role(previous_role)
            component.train(was_training)

    def _set_adapter_role(self, role: str) -> None:
        component = self._flowmap_component
        if role == "student":
            component.enable_lora()
            component.set_adapter(self._student_adapter_name)
            component.train()
        elif role == "discriminator":
            component.enable_lora()
            component.set_adapter(self._discriminator_adapter_name)
            component.train()
        elif role == "real":
            component.disable_lora()
            component.eval()
        else:
            raise ValueError(f"Unknown AnyFlow adapter role: {role}")
        self._current_adapter_role = role

    # ------------------------------------------------------------------
    # NVIDIA forward MeanFlow stage
    # ------------------------------------------------------------------
    @contextmanager
    def _frozen_base_adapter(self) -> Iterator[None]:
        if self.config["stage"] == "onpolicy":
            with self._adapter_role("real"):
                yield
            return

        component = self._flowmap_component
        was_training = component.training
        try:
            self.toggle_adapter(enable=False)
            component.eval()
            yield
        finally:
            self.toggle_adapter(enable=True)
            component.train(was_training)

    def _meanflow_base_target(
        self,
        *,
        prepared_batch: Dict[str, Any],
        t_sigmas: torch.Tensor,
        base_target: torch.Tensor,
    ) -> torch.Tensor:
        if self.config["diffusion_target"] == "flow":
            return base_target

        with self._frozen_base_adapter(), torch.no_grad():
            base_prediction = self._predict_at_sigmas(
                prepared_batch,
                prepared_batch["noisy_latents"],
                t_sigmas,
                t_sigmas,
            ).detach()

        base_prediction = base_prediction.to(device=base_target.device, dtype=base_target.dtype)
        cosine, norm_ratio = self._chunked_per_sample_geometry(base_prediction, base_target)
        prepared_batch["_anyflow_base_prediction_flow_cosine"] = cosine
        prepared_batch["_anyflow_base_prediction_flow_norm_ratio"] = norm_ratio
        prepared_batch["_anyflow_base_prediction_flow_mse"] = (
            (base_prediction.float() - base_target.float()).square().flatten(1).mean(dim=1).detach()
        )

        diffusion_mask = prepared_batch["anyflow_diffusion_mask"].to(device=base_target.device, dtype=torch.bool)
        diffusion_mask = self._broadcast_time(diffusion_mask, base_target)
        return torch.where(diffusion_mask, base_prediction, base_target)

    def _prepare_meanflow_pair(self, prepared_batch: Dict[str, Any], model) -> tuple[torch.Tensor, torch.Tensor]:
        latents = prepared_batch["latents"]
        batch_size = int(latents.shape[0])
        device = latents.device
        first = self._rand((batch_size,), device=device, dtype=torch.float32)
        second = self._rand((batch_size,), device=device, dtype=torch.float32)
        t_base = torch.maximum(first, second)
        r_base = torch.minimum(first, second)

        accelerator = getattr(model, "accelerator", getattr(self.teacher_model, "accelerator", None))
        batch_layout = resolve_distributed_batch_layout(accelerator, batch_size)
        self._distributed_batch_accelerator = accelerator
        self._distributed_batch_layout = batch_layout
        global_batch_size = batch_layout.global_batch_size
        global_indices = batch_layout.local_batch_offset + torch.arange(batch_size, device=device)
        diffusion_count = round(float(self.config["diffusion_ratio"]) * global_batch_size)
        consistency_count = round(float(self.config["consistency_ratio"]) * global_batch_size)
        effective_diffusion_count = min(diffusion_count, global_batch_size)
        effective_consistency_count = min(consistency_count, global_batch_size - effective_diffusion_count)
        arbitrary_count = global_batch_size - effective_diffusion_count - effective_consistency_count
        if batch_layout.data_rank == 0 and not getattr(self, "_meanflow_branch_mix_logged", False):
            self.logger.info(
                "AnyFlow interval mixture at global batch %d: diffusion=%d, consistency=%d, arbitrary=%d.",
                global_batch_size,
                effective_diffusion_count,
                effective_consistency_count,
                arbitrary_count,
            )
            missing_branches = []
            if float(self.config["diffusion_ratio"]) > 0.0 and effective_diffusion_count == 0:
                missing_branches.append("diffusion")
            if float(self.config["consistency_ratio"]) > 0.0 and effective_consistency_count == 0:
                missing_branches.append("consistency")
            arbitrary_ratio = 1.0 - float(self.config["diffusion_ratio"]) - float(self.config["consistency_ratio"])
            if arbitrary_ratio > 0.0 and arbitrary_count == 0:
                missing_branches.append("arbitrary")
            if missing_branches:
                self.logger.warning(
                    "AnyFlow global batch %d rounds the configured interval mixture to zero %s samples. "
                    "Increase the per-device batch size or data-parallel process count so every enabled branch "
                    "is represented on each optimizer step.",
                    global_batch_size,
                    "/".join(missing_branches),
                )
            self._meanflow_branch_mix_logged = True
        diffusion_mask = global_indices < diffusion_count
        consistency_mask = (global_indices >= diffusion_count) & (global_indices < diffusion_count + consistency_count)
        arbitrary_mask = ~(diffusion_mask | consistency_mask)

        non_diffusion_max_sigma = float(self.config["meanflow_non_diffusion_max_sigma"])
        if non_diffusion_max_sigma < 1.0:
            non_diffusion_max_base = self._invert_scheduler_shift(non_diffusion_max_sigma)
            non_diffusion_mask = ~diffusion_mask
            t_base = torch.where(non_diffusion_mask, t_base * non_diffusion_max_base, t_base)
            r_base = torch.where(non_diffusion_mask, r_base * non_diffusion_max_base, r_base)
        r_base = torch.where(diffusion_mask, t_base, r_base)
        r_base = torch.where(consistency_mask, torch.zeros_like(r_base), r_base)

        t_sigmas = self._apply_scheduler_shift(t_base)
        r_sigmas = self._apply_scheduler_shift(r_base)
        prepared_batch["anyflow_diffusion_mask"] = diffusion_mask
        prepared_batch["anyflow_consistency_mask"] = consistency_mask
        prepared_batch["anyflow_arbitrary_mask"] = arbitrary_mask
        prepared_batch["anyflow_t_base"] = t_base
        prepared_batch["anyflow_r_base"] = r_base
        prepared_batch["anyflow_t_sigmas"] = t_sigmas
        prepared_batch["anyflow_r_sigmas"] = r_sigmas
        self._set_batch_sigma_path(prepared_batch, t_sigmas)
        return t_sigmas, r_sigmas

    def _meanflow_target(
        self,
        *,
        prepared_batch: Dict[str, Any],
        model,
        t_sigmas: torch.Tensor,
        r_sigmas: torch.Tensor,
        base_target: torch.Tensor,
    ) -> torch.Tensor:
        epsilon = float(self.config["central_difference_epsilon"])
        plus_sigmas = t_sigmas + epsilon
        minus_sigmas = t_sigmas - epsilon
        if self.config["central_difference_boundary_mode"] == "clamp":
            plus_sigmas = plus_sigmas.clamp(0.0, 1.0)
            minus_sigmas = minus_sigmas.clamp(0.0, 1.0)

        with torch.no_grad():
            plus_output = model.model_predict(self._batch_at_sigmas(prepared_batch, plus_sigmas))
            plus_prediction = plus_output["model_prediction"].detach()
            self._clear_prediction_buffers(plus_output)
            minus_output = model.model_predict(self._batch_at_sigmas(prepared_batch, minus_sigmas))
            minus_prediction = minus_output["model_prediction"].detach()
            self._clear_prediction_buffers(minus_output)

        denominator = self._broadcast_time(plus_sigmas - minus_sigmas, plus_prediction).to(
            device=plus_prediction.device,
            dtype=plus_prediction.dtype,
        )
        total_derivative = (plus_prediction - minus_prediction) / (denominator * float(self.config["fuse_guidance_scale"]))
        interval = self._broadcast_time(t_sigmas - r_sigmas, total_derivative).to(
            device=total_derivative.device,
            dtype=total_derivative.dtype,
        )
        target = base_target.to(device=total_derivative.device, dtype=total_derivative.dtype) - interval * total_derivative
        return target.to(device=base_target.device, dtype=base_target.dtype)

    def _meanflow_loss(self, prepared_batch: Dict[str, Any], model_output: Dict[str, Any]) -> torch.Tensor:
        prediction = model_output.get("model_prediction")
        target = prepared_batch.get("target")
        if not torch.is_tensor(prediction) or not torch.is_tensor(target):
            raise ValueError("AnyFlow MeanFlow loss requires tensor model_prediction and target values.")
        if prediction.shape != target.shape:
            raise ValueError(
                f"AnyFlow MeanFlow target shape {tuple(target.shape)} does not match prediction {tuple(prediction.shape)}."
            )

        prediction = self._fuse_guidance_prediction(prepared_batch, prediction)
        per_sample = (prediction.float() - target.float()).square().flatten(1).mean(dim=1)
        t_sigmas = self._scalar_sigmas(prepared_batch).to(device=per_sample.device, dtype=per_sample.dtype)
        timestep_weight = self._meanflow_timestep_weight(t_sigmas)
        per_sample = per_sample * timestep_weight
        prepared_batch["_anyflow_pre_adaptive_loss"] = per_sample.detach()
        adaptive_scale = torch.ones_like(per_sample)

        diffusion_mask = prepared_batch.get("anyflow_diffusion_mask")
        if bool(self.config["meanflow_adaptive_weighting"]) and torch.is_tensor(diffusion_mask):
            diffusion_mask = diffusion_mask.to(device=per_sample.device, dtype=torch.bool)
            global_loss = self._gather_detached(per_sample)
            global_diffusion_mask = self._gather_detached(diffusion_mask)
            if bool(global_diffusion_mask.any()):
                diffusion_mean = global_loss[global_diffusion_mask].mean()
                base_prediction_flow_mse = prepared_batch.get("_anyflow_base_prediction_flow_mse")
                if torch.is_tensor(base_prediction_flow_mse):
                    adaptive_reference = (
                        base_prediction_flow_mse.to(
                            device=per_sample.device,
                            dtype=per_sample.dtype,
                        )
                        * timestep_weight
                    )
                    prepared_batch["_anyflow_adaptive_reference_loss"] = adaptive_reference.detach()
                    global_adaptive_reference = self._gather_detached(adaptive_reference)
                    diffusion_mean = torch.maximum(
                        diffusion_mean,
                        global_adaptive_reference[global_diffusion_mask].mean(),
                    )
                non_diffusion_mask = ~diffusion_mask
                if bool(non_diffusion_mask.any()):
                    scale = diffusion_mean / (per_sample.detach()[non_diffusion_mask] + 1e-5)
                    adaptive_scale[non_diffusion_mask] = scale
                    per_sample = per_sample.clone()
                    per_sample[non_diffusion_mask] = per_sample[non_diffusion_mask] * scale
        prepared_batch["_anyflow_adaptive_scale"] = adaptive_scale.detach()
        prepared_batch["_anyflow_post_adaptive_loss"] = per_sample.detach()
        return per_sample.mean()

    def _meanflow_timestep_weight(self, t_sigmas: torch.Tensor) -> torch.Tensor:
        if self.config["meanflow_weight_type"] == "uniform":
            return torch.ones_like(t_sigmas)
        weight = t_sigmas * torch.sqrt((1.0 - t_sigmas).clamp_min(0.0))
        grid = torch.linspace(1.0, 0.0, int(self.num_train_timesteps) + 1, device=t_sigmas.device)[:-1]
        shifted_grid = self._apply_scheduler_shift(grid)
        grid_weight = shifted_grid * torch.sqrt((1.0 - shifted_grid).clamp_min(0.0))
        normalization = grid_weight.numel() / grid_weight.sum().clamp_min(torch.finfo(grid_weight.dtype).eps)
        return weight * normalization.to(dtype=weight.dtype)

    # ------------------------------------------------------------------
    # NVIDIA on-policy DMD / Flow Map Backward Simulation stage
    # ------------------------------------------------------------------
    def _onpolicy_generator_loss(self, prepared_batch: Dict[str, Any]) -> tuple[torch.Tensor, Dict[str, float]]:
        dmd_batch = self._slice_batch(prepared_batch, int(self.config["dmd_batch_size"]))
        step_count, grad_timestep = self._distributed_rollout_schedule()
        with self._adapter_role("student"):
            generated = self._training_rollout(
                dmd_batch,
                step_count=step_count,
                grad_timestep=grad_timestep,
            )

        sigma = self._sample_dmd_sigma(generated.shape[0], logit_normal=False, device=generated.device)
        noise = self._randn_like(generated)
        sigma_broadcast = self._broadcast_time(sigma, generated).to(generated.dtype)
        noisy = ((1.0 - sigma_broadcast) * generated + sigma_broadcast * noise).detach()

        with torch.no_grad():
            with self._adapter_role("discriminator"):
                fake_x0 = self._score_x0(dmd_batch, noisy, sigma)
            with self._adapter_role("real"):
                real_x0 = self._score_x0(dmd_batch, noisy, sigma)
                real_x0 = self._apply_real_score_guidance(dmd_batch, noisy, sigma, real_x0)

            gradient = fake_x0 - real_x0
            dimensions = tuple(range(1, generated.ndim))
            normalizer = torch.abs(generated.detach() - real_x0).mean(dim=dimensions, keepdim=True)
            gradient = torch.nan_to_num(gradient / normalizer)

        target = (generated.double() - gradient.double()).detach()
        loss = float(self.config["dmd_weight"]) * F.mse_loss(generated.double(), target, reduction="mean")
        logs = {
            "anyflow_dmd_loss": float(loss.detach()),
            "anyflow_dmd_gradient_norm": float(gradient.float().abs().mean().detach()),
            "anyflow_dmd_sigma": float(sigma.float().mean().detach()),
            "anyflow_rollout_steps": float(step_count),
            "anyflow_rollout_grad_timestep": float(grad_timestep),
        }
        return loss, logs

    def _onpolicy_discriminator_loss(self, prepared_batch: Dict[str, Any]) -> torch.Tensor:
        dmd_batch = self._slice_batch(prepared_batch, int(self.config["dmd_batch_size"]))
        step_count, grad_timestep = self._distributed_rollout_schedule()
        with torch.no_grad(), self._adapter_role("student"):
            generated = self._training_rollout(
                dmd_batch,
                step_count=step_count,
                grad_timestep=grad_timestep,
            ).detach()

        sigma = self._sample_dmd_sigma(generated.shape[0], logit_normal=True, device=generated.device)
        noise = self._randn_like(generated)
        sigma_broadcast = self._broadcast_time(sigma, generated).to(generated.dtype)
        noisy = (1.0 - sigma_broadcast) * generated + sigma_broadcast * noise
        prediction = self._predict_at_sigmas(dmd_batch, noisy, sigma, sigma)
        target = self._flow_target(generated, noise).to(device=prediction.device, dtype=prediction.dtype)
        return (prediction.float() - target.float()).square().flatten(1).mean(dim=1).mean()

    def _training_rollout(
        self,
        prepared_batch: Dict[str, Any],
        *,
        step_count: int,
        grad_timestep: int,
    ) -> torch.Tensor:
        if grad_timestep < 0 or grad_timestep >= step_count:
            raise ValueError(f"AnyFlow grad_timestep must be in [0, {step_count}), got {grad_timestep}.")

        latents = self._randn_like(prepared_batch["latents"])
        base_sigmas = torch.linspace(1.0, 0.0, step_count + 1, device=latents.device, dtype=torch.float32)
        sigmas = self._apply_scheduler_shift(base_sigmas)

        intervals = (
            (sigmas[0], sigmas[grad_timestep]),
            (sigmas[grad_timestep], sigmas[grad_timestep + 1]),
            (sigmas[grad_timestep + 1], sigmas[-1]),
        )
        for t_value, r_value in intervals:
            if bool(t_value == r_value):
                continue
            t_sigma = t_value.expand(latents.shape[0])
            r_sigma = r_value.expand(latents.shape[0])
            prediction = self._predict_at_sigmas(prepared_batch, latents, t_sigma, r_sigma)
            velocity = self._prediction_to_noiseward_flow(prediction)
            step = self._broadcast_time(r_sigma - t_sigma, latents).to(dtype=latents.dtype)
            latents = latents + step * velocity.to(dtype=latents.dtype)
        return latents

    def _score_x0(
        self,
        prepared_batch: Dict[str, Any],
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        prediction = self._predict_at_sigmas(prepared_batch, noisy_latents, sigmas, sigmas)
        velocity = self._prediction_to_noiseward_flow(prediction)
        sigma = self._broadcast_time(sigmas, noisy_latents).to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        return noisy_latents - sigma * velocity.to(dtype=noisy_latents.dtype)

    def _apply_real_score_guidance(
        self,
        prepared_batch: Dict[str, Any],
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        conditional_x0: torch.Tensor,
    ) -> torch.Tensor:
        guidance = float(self.config["real_score_guidance_scale"])
        if guidance == 0.0:
            return conditional_x0
        negative = prepared_batch.get("negative_encoder_hidden_states")
        if not torch.is_tensor(negative):
            raise ValueError("AnyFlow real_score_guidance_scale requires cached negative_encoder_hidden_states.")
        unconditional_batch = self._unconditional_batch(prepared_batch)
        unconditional_x0 = self._score_x0(unconditional_batch, noisy_latents, sigmas)
        return conditional_x0 + (conditional_x0 - unconditional_x0) * guidance

    def _distributed_rollout_schedule(self) -> tuple[int, int]:
        steps = self.config["rollout_step_counts"]
        device = self._device
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            index = self._randint(len(steps), (1,), device=device, dtype=torch.long)
            torch.distributed.broadcast(index, src=0)
            step_count = int(steps[int(index.item())])
            grad_timestep = self._randint(step_count, (1,), device=device, dtype=torch.long)
            torch.distributed.broadcast(grad_timestep, src=0)
            return step_count, int(grad_timestep.item())

        step_count = int(steps[int(self._randint(len(steps), (1,), device=device, dtype=torch.long).item())])
        grad_timestep = int(self._randint(step_count, (1,), device=device, dtype=torch.long).item())
        return step_count, grad_timestep

    def _sample_dmd_sigma(self, batch_size: int, *, logit_normal: bool, device: torch.device) -> torch.Tensor:
        base = (
            torch.sigmoid(self._randn((batch_size,), device=device))
            if logit_normal
            else self._rand((batch_size,), device=device)
        )
        sigma = self._apply_scheduler_shift(base)
        return sigma.clamp(
            min=float(self.config["dmd_min_timestep_ratio"]),
            max=float(self.config["dmd_max_timestep_ratio"]),
        )

    # ------------------------------------------------------------------
    # Shared model and tensor helpers
    # ------------------------------------------------------------------
    def _predict_at_sigmas(
        self,
        prepared_batch: Dict[str, Any],
        noisy_latents: torch.Tensor,
        t_sigmas: torch.Tensor,
        r_sigmas: torch.Tensor,
    ) -> torch.Tensor:
        batch = dict(prepared_batch)
        batch.pop("target", None)
        batch.pop("flow_target", None)
        batch["noisy_latents"] = noisy_latents
        batch["sigmas"] = self._sigma_for_reference(t_sigmas, prepared_batch.get("sigmas"), noisy_latents)
        batch["timesteps"] = self._timesteps_from_sigmas(t_sigmas, prepared_batch["timesteps"]).to(
            device=prepared_batch["timesteps"].device,
            dtype=prepared_batch["timesteps"].dtype,
        )
        r_timesteps = self._timesteps_from_sigmas(r_sigmas, prepared_batch["timesteps"]).to(
            device=prepared_batch["timesteps"].device,
            dtype=prepared_batch["timesteps"].dtype,
        )
        flowmap_key = getattr(self.teacher_model, "FLOWMAP_R_TIMESTEP_BATCH_KEY", self.FLOWMAP_R_TIMESTEP_BATCH_KEY)
        batch[flowmap_key] = r_timesteps
        batch["anyflow_r_timesteps"] = r_timesteps
        output = self.teacher_model.model_predict(batch)
        prediction = output["model_prediction"]
        self._clear_prediction_buffers(output)
        return prediction

    def _fuse_guidance_prediction(
        self,
        prepared_batch: Dict[str, Any],
        conditional_prediction: torch.Tensor,
    ) -> torch.Tensor:
        guidance_scale = float(self.config["fuse_guidance_scale"])
        if guidance_scale == 1.0:
            return conditional_prediction

        unconditional_batch = self._unconditional_batch(prepared_batch)
        with torch.no_grad():
            output = self.teacher_model.model_predict(unconditional_batch)
            unconditional_prediction = output["model_prediction"].detach()
            self._clear_prediction_buffers(output)
        return (conditional_prediction - (1.0 - guidance_scale) * unconditional_prediction) / guidance_scale

    @staticmethod
    def _unconditional_batch(prepared_batch: Dict[str, Any]) -> Dict[str, Any]:
        negative = prepared_batch.get("negative_encoder_hidden_states")
        if not torch.is_tensor(negative):
            raise ValueError("AnyFlow fuse_guidance_scale != 1 requires cached unconditional text embeddings.")

        batch = dict(prepared_batch)
        batch["encoder_hidden_states"] = negative
        negative_tags = prepared_batch.get("negative_text_token_tags")
        if torch.is_tensor(negative_tags):
            batch["text_token_tags"] = negative_tags
        negative_mask = prepared_batch.get("negative_encoder_attention_mask")
        batch.pop("encoder_attention_mask", None)
        if torch.is_tensor(negative_mask):
            batch["encoder_attention_mask"] = negative_mask
        # Some families read model-specific conditioning aliases ahead of (or instead of) the
        # generic keys: Ideogram prefers `prompt_embeds` when present, and Flux requires it.
        # Swap the aliases to the unconditional tensors rather than popping them.
        if "prompt_embeds" in batch:
            batch["prompt_embeds"] = negative
        for alias in ("attention_mask", "attention_masks"):
            if alias in batch:
                if torch.is_tensor(negative_mask):
                    batch[alias] = negative_mask
                else:
                    batch.pop(alias)
        negative_pooled = prepared_batch.get("negative_add_text_embeds")
        if torch.is_tensor(negative_pooled):
            if "add_text_embeds" in batch:
                batch["add_text_embeds"] = negative_pooled
            added_cond_kwargs = batch.get("added_cond_kwargs")
            if isinstance(added_cond_kwargs, dict) and "text_embeds" in added_cond_kwargs:
                batch["added_cond_kwargs"] = {**added_cond_kwargs, "text_embeds": negative_pooled}
        # Lets families with a dedicated unconditional model (e.g. Ideogram) dispatch to it.
        batch["is_unconditional_pass"] = True
        return batch

    def _slice_batch(self, prepared_batch: Dict[str, Any], requested_batch_size: int) -> Dict[str, Any]:
        source_batch_size = int(prepared_batch["latents"].shape[0])
        batch_size = min(source_batch_size, requested_batch_size)

        def slice_value(value):
            if torch.is_tensor(value) and value.ndim > 0 and int(value.shape[0]) == source_batch_size:
                return value[:batch_size]
            if isinstance(value, dict):
                return {key: slice_value(item) for key, item in value.items()}
            return value

        return {key: slice_value(value) for key, value in prepared_batch.items()}

    def _forward_logs(self, prepared_batch: Dict[str, Any], loss: torch.Tensor) -> Dict[str, float]:
        r_timesteps = prepared_batch["anyflow_r_timesteps"]
        global_timesteps = self._gather_detached(prepared_batch["timesteps"].float().reshape(-1))
        global_r_timesteps = self._gather_detached(r_timesteps.float().reshape(-1))
        global_intervals = self._gather_detached(prepared_batch["anyflow_timestep_interval"].float().reshape(-1))
        logs = {
            "anyflow_forward_loss": float(loss.detach()),
            "anyflow_timestep": float(global_timesteps.mean()),
            "anyflow_r_timestep": float(global_r_timesteps.mean()),
            "anyflow_interval": float(global_intervals.mean()),
            "anyflow_fuse_guidance_scale": float(self.config["fuse_guidance_scale"]),
            "anyflow_meanflow_non_diffusion_max_sigma": float(self.config["meanflow_non_diffusion_max_sigma"]),
            "anyflow_diffusion_target_is_base_prediction": float(self.config["diffusion_target"] == "base_prediction"),
        }
        metric_tensors = {
            "t_sigma": prepared_batch.get("anyflow_t_sigmas"),
            "r_sigma": prepared_batch.get("anyflow_r_sigmas"),
            "target_base_cosine": prepared_batch.get("_anyflow_target_base_cosine"),
            "target_base_norm_ratio": prepared_batch.get("_anyflow_target_base_norm_ratio"),
            "base_prediction_flow_cosine": prepared_batch.get("_anyflow_base_prediction_flow_cosine"),
            "base_prediction_flow_norm_ratio": prepared_batch.get("_anyflow_base_prediction_flow_norm_ratio"),
            "adaptive_reference_loss": prepared_batch.get("_anyflow_adaptive_reference_loss"),
            "pre_adaptive_loss": prepared_batch.get("_anyflow_pre_adaptive_loss"),
            "adaptive_scale": prepared_batch.get("_anyflow_adaptive_scale"),
            "post_adaptive_loss": prepared_batch.get("_anyflow_post_adaptive_loss"),
        }
        global_metrics = {
            name: self._gather_detached(value.float().reshape(-1))
            for name, value in metric_tensors.items()
            if torch.is_tensor(value)
        }
        for branch in ("diffusion", "consistency", "arbitrary"):
            mask = self._gather_detached(prepared_batch[f"anyflow_{branch}_mask"].reshape(-1)).bool()
            logs[f"anyflow_{branch}_fraction"] = float(mask.float().mean())
            if bool(mask.any()):
                for metric_name, values in global_metrics.items():
                    logs[f"anyflow_{branch}_{metric_name}"] = float(values[mask].mean())
        logs.update(self._delta_parameter_logs())
        return logs

    _DELTA_PARAMETER_TAGS = (
        "condition_embedder.delta_embedder",
        "delta_adaln_embedder",
        "delta_time_embed",
        "delta_time_embedder",
        "delta_timestep_embedder",
        "delta_t_embedding",
    )

    def _trainable_delta_parameters(self):
        for name, parameter in self._flowmap_component.named_parameters():
            if ".original_module." in name or not parameter.requires_grad:
                continue
            if any(tag in name for tag in self._DELTA_PARAMETER_TAGS):
                yield name, parameter

    def _delta_parameter_logs(self) -> Dict[str, float]:
        if not self._delta_initial_parameters:
            return {}
        norm_sq = 0.0
        drift_sq = 0.0
        for name, parameter in self._trainable_delta_parameters():
            current = parameter.detach().float()
            norm_sq += float(current.square().sum())
            reference = self._delta_initial_parameters.get(name)
            if reference is not None:
                drift_sq += float((current.cpu() - reference).square().sum())
        return {
            "anyflow_delta_weight_norm": norm_sq**0.5,
            "anyflow_delta_drift": drift_sq**0.5,
        }

    def _validation_component_names(self) -> tuple[str, ...]:
        names: list[str] = []
        model_type = getattr(getattr(self.teacher_model, "MODEL_TYPE", None), "value", None)
        if isinstance(model_type, str) and model_type:
            names.append(model_type)
        for fallback_name in ("transformer", "conditional_transformer", "unet"):
            if fallback_name not in names:
                names.append(fallback_name)
        return tuple(names)

    def _resolve_timestep_scale(self, noise_scheduler) -> float:
        configured = self.config.get("timestep_scale")
        if configured not in (None, ""):
            return float(configured)
        scheduler_config = getattr(noise_scheduler, "config", None)
        if isinstance(scheduler_config, dict):
            return float(scheduler_config.get("num_train_timesteps", 1000))
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
            gate_value=float(self.config["gate_value"]),
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
    def _validate_video_batch(batch: Dict[str, Any]) -> None:
        required_keys = ("latents", "noise", "noisy_latents", "sigmas", "timesteps")
        missing = [key for key in required_keys if key not in batch or batch[key] is None]
        if missing:
            raise ValueError(f"AnyFlow prepared batch is missing required fields: {', '.join(missing)}.")
        for key in required_keys:
            if not torch.is_tensor(batch[key]):
                raise ValueError(f"AnyFlow prepared batch field `{key}` must be a tensor.")
        for key in ("audio_latents", "audio_noisy_latents", "audio_timesteps", "audio_sigmas"):
            if torch.is_tensor(batch.get(key)):
                raise ValueError(
                    "AnyFlow does not yet support joint audio-video batches. MiniMax-H3 audio requires its native "
                    "shift-3 schedule while video uses shift 12."
                )

    @staticmethod
    def _broadcast_time(time_tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        while time_tensor.ndim < like.ndim:
            time_tensor = time_tensor.unsqueeze(-1)
        return time_tensor

    @staticmethod
    @torch.no_grad()
    def _chunked_per_sample_geometry(
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        chunk_elements: int = 262_144,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_flat = left.reshape(left.shape[0], -1)
        right_flat = right.reshape(right.shape[0], -1)
        dot = torch.zeros(left.shape[0], device=left.device, dtype=torch.float32)
        left_squared = torch.zeros_like(dot)
        right_squared = torch.zeros_like(dot)
        for start in range(0, left_flat.shape[1], chunk_elements):
            end = min(start + chunk_elements, left_flat.shape[1])
            left_chunk = left_flat[:, start:end].float()
            right_chunk = right_flat[:, start:end].float()
            dot.add_((left_chunk * right_chunk).sum(dim=1))
            left_squared.add_(left_chunk.square().sum(dim=1))
            right_squared.add_(right_chunk.square().sum(dim=1))
        epsilon = torch.finfo(dot.dtype).eps
        left_norm = left_squared.sqrt()
        right_norm = right_squared.sqrt()
        cosine = dot / (left_norm * right_norm).clamp_min(epsilon)
        norm_ratio = left_norm / right_norm.clamp_min(epsilon)
        return cosine.detach(), norm_ratio.detach()

    @staticmethod
    def _scalar_sigmas(prepared_batch: Dict[str, Any]) -> torch.Tensor:
        sigmas = prepared_batch["sigmas"]
        batch_size = prepared_batch["latents"].shape[0]
        sigmas = sigmas.to(device=prepared_batch["latents"].device, dtype=torch.float32)
        if sigmas.ndim == 0:
            return sigmas.expand(batch_size)
        if sigmas.shape[0] != batch_size:
            raise ValueError(f"AnyFlow expected {batch_size} sigma values, got shape {tuple(sigmas.shape)}.")
        return sigmas.reshape(batch_size, -1)[:, 0]

    def _timesteps_from_sigmas(self, sigmas: torch.Tensor, reference_timesteps: torch.Tensor) -> torch.Tensor:
        converter = getattr(self.teacher_model, "flow_matching_timesteps_from_sigmas", None)
        if callable(converter):
            return converter(sigmas, reference_timesteps=reference_timesteps)
        if torch.max(reference_timesteps.detach().float()) <= 1.0:
            return sigmas
        return sigmas * self.num_train_timesteps

    def _apply_scheduler_shift(self, sigmas: torch.Tensor) -> torch.Tensor:
        shift = self._scheduler_shift_value()
        if shift == 1.0:
            return sigmas
        return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)

    def _invert_scheduler_shift(self, sigmas: float | torch.Tensor) -> float | torch.Tensor:
        shift = self._scheduler_shift_value()
        if shift == 1.0:
            return sigmas
        return sigmas / (shift + (1.0 - shift) * sigmas)

    def _scheduler_shift_value(self) -> float:
        scheduler_config = getattr(self.noise_scheduler, "config", None)
        shift = self.config.get("schedule_shift")
        if shift in (None, ""):
            shift = (
                scheduler_config.get("shift", 1.0)
                if isinstance(scheduler_config, dict)
                else getattr(scheduler_config, "shift", 1.0)
            )
        shift = float(shift or 1.0)
        return shift

    def _set_batch_sigma_path(self, batch: Dict[str, Any], sigmas: torch.Tensor) -> None:
        latents = batch["latents"]
        noise = batch["noise"].to(device=latents.device, dtype=latents.dtype)
        sigma_for_latents = self._broadcast_time(sigmas, latents).to(device=latents.device, dtype=latents.dtype)
        batch["noisy_latents"] = (1.0 - sigma_for_latents) * latents + sigma_for_latents * noise
        batch["sigmas"] = self._sigma_for_reference(sigmas, batch.get("sigmas"), latents)
        batch["timesteps"] = self._timesteps_from_sigmas(sigmas, batch["timesteps"]).to(
            device=batch["timesteps"].device,
            dtype=batch["timesteps"].dtype,
        )

    def _sigma_for_reference(
        self,
        sigmas: torch.Tensor,
        reference_sigmas: Any,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.is_tensor(reference_sigmas):
            return sigmas
        sigma_for_model = sigmas
        while sigma_for_model.ndim < reference_sigmas.ndim:
            sigma_for_model = sigma_for_model.unsqueeze(-1)
        return sigma_for_model.expand_as(reference_sigmas).to(
            device=latents.device,
            dtype=reference_sigmas.dtype,
        )

    def _batch_at_sigmas(self, prepared_batch: Dict[str, Any], sigmas: torch.Tensor) -> Dict[str, Any]:
        batch = dict(prepared_batch)
        self._set_batch_sigma_path(batch, sigmas)
        batch.pop("target", None)
        batch.pop("flow_target", None)
        return batch

    def _base_flow_target(self, prepared_batch: Dict[str, Any], model=None) -> torch.Tensor:
        target_model = model or self.student_model or self.teacher_model
        get_target = getattr(target_model, "get_flow_matching_target", None)
        if callable(get_target):
            target = get_target(prepared_batch, prefer_explicit_target=False)
            return target.to(device=prepared_batch["latents"].device, dtype=prepared_batch["latents"].dtype)
        return self._flow_target(prepared_batch["latents"], prepared_batch["noise"])

    def _flow_target(self, latents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        target_fn = getattr(self.teacher_model, "flow_matching_target", None)
        if callable(target_fn):
            return target_fn(latents, noise)
        return noise - latents

    def _prediction_to_noiseward_flow(self, prediction: torch.Tensor) -> torch.Tensor:
        converter = getattr(self.teacher_model, "prediction_to_noiseward_flow", None)
        return converter(prediction) if callable(converter) else prediction

    @staticmethod
    def _clear_prediction_buffers(output: Dict[str, Any]) -> None:
        hidden_states_buffer = output.get("hidden_states_buffer")
        if isinstance(hidden_states_buffer, dict):
            hidden_states_buffer.clear()

    def _gather_detached(self, tensor: torch.Tensor) -> torch.Tensor:
        accelerator = getattr(
            self,
            "_distributed_batch_accelerator",
            getattr(self.teacher_model, "accelerator", None),
        )
        layout = getattr(self, "_distributed_batch_layout", None)
        if layout is None or layout.local_batch_size != tensor.shape[0]:
            layout = resolve_distributed_batch_layout(accelerator, tensor.shape[0])
        return gather_variable_batch_tensor(tensor, accelerator, layout)

    def _discriminator_state_dict(self) -> Dict[str, torch.Tensor]:
        adapter_name = str(self._discriminator_adapter_name)
        return {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in self._flowmap_component.state_dict().items()
            if self._parameter_belongs_to_adapter(name, adapter_name)
        }

    def _sync_discriminator_gradients(self) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        world_size = torch.distributed.get_world_size()
        for parameter in self._discriminator_parameters:
            if parameter.grad is None:
                continue
            torch.distributed.all_reduce(parameter.grad, op=torch.distributed.ReduceOp.SUM)
            parameter.grad.div_(world_size)

    def _load_discriminator_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        component_state = self._flowmap_component.state_dict()
        unexpected = sorted(set(state) - set(component_state))
        if unexpected:
            raise ValueError(f"Unexpected AnyFlow discriminator checkpoint keys: {unexpected[:5]}")
        for name, tensor in state.items():
            component_state[name].copy_(tensor.to(device=component_state[name].device, dtype=component_state[name].dtype))

    @staticmethod
    def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device=device)


DistillationRegistry.register(
    "anyflow",
    AnyFlowDistiller,
    requires_distillation_cache=False,
    data_requirements=[[DatasetType.IMAGE, DatasetType.VIDEO]],
    requirement_notes="Requires model-specific FlowMap interval conditioning; joint H3 audio-video is not yet supported.",
)
