from __future__ import annotations

import contextlib
import math
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.registry import DistillationRegistry
from simpletuner.helpers.models.common import ModelTypes, PredictionTypes


class SelfTranscendenceProjector(nn.Module):
    """Three-layer projector used by both Self-Transcendence stages."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.SiLU(),
            nn.Linear(intermediate_size, intermediate_size),
            nn.SiLU(),
            nn.Linear(intermediate_size, hidden_size),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)


class SelfTranscendenceDistiller(DistillationBase):
    """Two-stage internal representation guidance for diffusion transformers."""

    PROJECTOR_NAME = "self_transcendence_projector"
    STUDENT_OUTPUT_KEY = "self_transcendence_student_hidden"
    _VALID_STAGES = {"vae", "self"}
    _DEFAULTS: Dict[str, Any] = {
        "stage": "vae",
        "student_block": None,
        "teacher_block": None,
        "weight": 0.5,
        "cfg_scale": 30.0,
        "timestep_min": 0.4,
        "timestep_max": 0.7,
        "stop_step": 0,
        "projector_hidden_dim": 2048,
        "teacher_adapter_path": None,
    }

    @classmethod
    def _normalized_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(cls._DEFAULTS)
        normalized.update(config or {})
        normalized["stage"] = str(normalized["stage"]).strip().lower()
        if normalized["stage"] not in cls._VALID_STAGES:
            raise ValueError("Self-Transcendence stage must be either 'vae' or 'self'.")
        if normalized["student_block"] is None:
            raise ValueError("Self-Transcendence requires student_block.")
        if normalized["stage"] == "self" and normalized["teacher_block"] is None:
            raise ValueError("Self-Transcendence self stage requires teacher_block.")
        normalized["student_block"] = int(normalized["student_block"])
        if normalized["teacher_block"] is not None:
            normalized["teacher_block"] = int(normalized["teacher_block"])
        if normalized["student_block"] < 0 or (normalized["teacher_block"] is not None and normalized["teacher_block"] < 0):
            raise ValueError("Self-Transcendence block indices must be zero-based non-negative integers.")
        normalized["weight"] = float(normalized["weight"])
        normalized["cfg_scale"] = float(normalized["cfg_scale"])
        normalized["timestep_min"] = float(normalized["timestep_min"])
        normalized["timestep_max"] = float(normalized["timestep_max"])
        normalized["stop_step"] = int(normalized["stop_step"] or 0)
        normalized["projector_hidden_dim"] = int(normalized["projector_hidden_dim"])
        teacher_adapter_path = normalized.get("teacher_adapter_path")
        normalized["teacher_adapter_path"] = str(teacher_adapter_path).strip() if teacher_adapter_path else None
        if normalized["weight"] <= 0:
            raise ValueError("Self-Transcendence weight must be greater than zero.")
        if normalized["cfg_scale"] < 1:
            raise ValueError("Self-Transcendence cfg_scale must be at least 1.0.")
        if not 0 <= normalized["timestep_min"] < normalized["timestep_max"] <= 1:
            raise ValueError("Self-Transcendence timestep range must satisfy 0 <= min < max <= 1.")
        if normalized["stop_step"] < 0:
            raise ValueError("Self-Transcendence stop_step must be non-negative.")
        if normalized["projector_hidden_dim"] <= 0:
            raise ValueError("Self-Transcendence projector_hidden_dim must be greater than zero.")
        return normalized

    @classmethod
    def prepare_model_for_adapter(cls, model, config: Dict[str, Any]) -> None:
        settings = cls._normalized_config(config)
        if getattr(model, "MODEL_TYPE", None) is not ModelTypes.TRANSFORMER:
            raise ValueError("Self-Transcendence supports diffusion transformer model families, not UNet models.")
        if getattr(model, "PREDICTION_TYPE", None) is PredictionTypes.AUTOREGRESSIVE_NEXT_TOKEN:
            raise ValueError("Self-Transcendence is not defined for autoregressive model families.")
        if (
            "lora" in str(getattr(model.config, "model_type", "")).lower()
            and str(getattr(model.config, "lora_type", "standard")).lower() == "lycoris"
        ):
            raise ValueError(
                "Self-Transcendence supports full-model and standard PEFT LoRA training; LyCORIS is not supported."
            )

        component = model.get_trained_component(unwrap_model=False)
        if component is None:
            raise ValueError("Self-Transcendence requires a loaded transformer component.")
        hidden_size = cls._infer_hidden_size(component)
        if hidden_size is None:
            raise ValueError(f"Self-Transcendence could not infer the hidden size for {model.NAME}.")
        projector = getattr(component, cls.PROJECTOR_NAME, None)
        if projector is None:
            projector = SelfTranscendenceProjector(hidden_size, settings["projector_hidden_dim"])
            setattr(component, cls.PROJECTOR_NAME, projector)
        projector.to(device=model.accelerator.device, dtype=torch.float32)

    @classmethod
    def training_batch_requirements(cls, config: Dict[str, Any]) -> set[str]:
        settings = cls._normalized_config(config)
        return {"unconditional_text_embeddings"} if settings["stage"] == "self" else set()

    @staticmethod
    def _infer_hidden_size(component: nn.Module) -> Optional[int]:
        config = getattr(component, "config", None)
        if config is None:
            return None
        heads = getattr(config, "num_attention_heads", None)
        head_dim = getattr(config, "attention_head_dim", None)
        if heads is not None and head_dim is not None:
            return int(heads * head_dim)
        for name in ("hidden_size", "model_dim", "d_model", "dim", "inner_dim", "emb_dim", "width"):
            value = getattr(config, name, None)
            if value is not None:
                return int(value)
        for name in ("hidden_size", "model_dim", "inner_dim", "dim"):
            value = getattr(component, name, None)
            if isinstance(value, int):
                return value
        return None

    def __init__(
        self,
        teacher_model,
        student_model=None,
        *,
        noise_scheduler=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        settings = self._normalized_config(config or {})
        super().__init__(teacher_model, student_model, settings)
        self.noise_scheduler = noise_scheduler
        self.stage = settings["stage"]
        self.student_block = settings["student_block"]
        self.teacher_block = settings["teacher_block"]
        self.weight = settings["weight"]
        self.cfg_scale = settings["cfg_scale"]
        self.timestep_min = settings["timestep_min"]
        self.timestep_max = settings["timestep_max"]
        self.stop_step = settings["stop_step"]
        self.teacher_adapter_path = settings["teacher_adapter_path"]
        self._global_step = 0
        self._teacher_parameters: Optional[list[torch.Tensor]] = None

        component = teacher_model.get_trained_component(unwrap_model=False)
        self.projector = getattr(component, self.PROJECTOR_NAME, None)
        if self.projector is None:
            raise ValueError("Self-Transcendence projector was not attached before distiller initialization.")

    def pre_training_step(self, model, step):
        self._global_step = int(step)
        if self.stage != "self" or self._teacher_parameters is not None:
            return
        parameters = self._trainable_parameters()
        if self.teacher_adapter_path:
            self._teacher_parameters = self._snapshot_teacher_adapter(parameters)
        else:
            self._teacher_parameters = [parameter.detach().clone() for parameter in parameters]
        self.logger.info(
            "Captured a fixed Self-Transcendence teacher snapshot with %d trainable tensors.",
            len(self._teacher_parameters),
        )

    def _snapshot_teacher_adapter(self, parameters: list[nn.Parameter]) -> list[torch.Tensor]:
        if str(getattr(self.teacher_model.config, "model_type", "")).lower() != "lora":
            raise ValueError("teacher_adapter_path is only supported for PEFT LoRA Self-Transcendence runs.")
        if str(getattr(self.teacher_model.config, "lora_type", "standard")).lower() != "standard":
            raise ValueError("teacher_adapter_path requires a standard PEFT LoRA.")
        if not os.path.isfile(self.teacher_adapter_path):
            raise FileNotFoundError(f"Self-Transcendence teacher adapter does not exist: {self.teacher_adapter_path}")

        from simpletuner.helpers.training.adapter import load_lora_weights

        student_parameters = [parameter.detach().clone() for parameter in parameters]
        component = self.teacher_model.get_trained_component(unwrap_model=False)
        prefix = getattr(self.teacher_model, "MODEL_SUBFOLDER", None) or self.teacher_model.MODEL_TYPE.value
        try:
            additional_keys, missing_keys = load_lora_weights(
                {prefix: component},
                self.teacher_adapter_path,
                use_dora=bool(getattr(self.teacher_model.config, "use_dora", False)),
            )
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise ValueError(f"Self-Transcendence teacher adapter is missing required tensors: {missing}")
            if additional_keys:
                additional = ", ".join(sorted(additional_keys))
                self.logger.warning("Self-Transcendence teacher adapter contains unused tensors: %s", additional)
            teacher_parameters = [parameter.detach().clone() for parameter in parameters]
        finally:
            for parameter, student in zip(parameters, student_parameters):
                parameter.data.copy_(student)
        return teacher_parameters

    def prepare_model_output(self, model_output: Dict[str, Any]) -> None:
        buffer = model_output.get("hidden_states_buffer")
        if not isinstance(buffer, dict):
            raise ValueError("Self-Transcendence requires a transformer hidden-state buffer.")
        key = f"layer_{self.student_block}"
        student = buffer.get(key)
        if student is None:
            raise ValueError(f"Self-Transcendence could not find student {key}.")
        model_output[self.STUDENT_OUTPUT_KEY] = student

    def compute_distill_loss(self, prepared_batch, model_output, original_loss):
        student = model_output.get(self.STUDENT_OUTPUT_KEY)
        if student is None:
            raise ValueError("Self-Transcendence student hidden state was not preserved before auxiliary losses.")
        student_tokens = self._flatten_hidden(student)
        projected = self.projector(student_tokens.to(dtype=self._projector_dtype()))
        active_weight = 0.0 if self.stop_step and self._global_step >= self.stop_step else self.weight

        if active_weight == 0.0:
            guide_loss = projected.sum() * 0.0
            logs = {"self_transcendence/loss": 0.0, "self_transcendence/weight": 0.0}
        elif self.stage == "vae":
            prediction_target = self.teacher_model.get_prediction_target(prepared_batch)
            target = self._vae_target_tokens(prediction_target, student, projected.shape[1])
            target = target.to(device=projected.device, dtype=projected.dtype)
            prediction = projected[..., : target.shape[-1]]
            guide_loss = self._masked_mse(prediction, target, prepared_batch)
            logs = {
                "self_transcendence/loss": guide_loss.detach().float().item(),
                "self_transcendence/weight": active_weight,
            }
        else:
            target = self._cfg_teacher_hidden(prepared_batch).to(device=projected.device, dtype=projected.dtype)
            if target.shape != projected.shape:
                raise ValueError(
                    "Self-Transcendence student and teacher shapes differ: "
                    f"student={tuple(projected.shape)}, teacher={tuple(target.shape)}."
                )
            guide_loss = self._masked_mse(projected, target, prepared_batch)
            logs = {
                "self_transcendence/loss": guide_loss.detach().float().item(),
                "self_transcendence/weight": active_weight,
                "self_transcendence/teacher_cfg_scale": self.cfg_scale,
            }

        buffer = model_output.get("hidden_states_buffer")
        if isinstance(buffer, dict):
            buffer.clear()
        return original_loss + guide_loss.to(dtype=original_loss.dtype) * active_weight, logs

    def _trainable_parameters(self) -> list[nn.Parameter]:
        component = self.teacher_model.get_trained_component(unwrap_model=False)
        return [parameter for parameter in component.parameters() if parameter.requires_grad]

    @contextlib.contextmanager
    def _fixed_teacher_context(self):
        if self._teacher_parameters is None:
            raise ValueError("Self-Transcendence teacher snapshot was not captured before the training forward.")
        parameters = self._trainable_parameters()
        if len(parameters) != len(self._teacher_parameters):
            raise RuntimeError("Self-Transcendence trainable parameter topology changed after the teacher snapshot.")
        stored = [parameter.detach().clone() for parameter in parameters]
        component = self.teacher_model.get_trained_component(unwrap_model=False)
        was_training = component.training
        try:
            for parameter, teacher in zip(parameters, self._teacher_parameters):
                parameter.data.copy_(teacher.to(device=parameter.device, dtype=parameter.dtype))
            component.eval()
            yield
        finally:
            for parameter, student in zip(parameters, stored):
                parameter.data.copy_(student)
            component.train(was_training)

    def _cfg_teacher_hidden(self, prepared_batch: Dict[str, Any]) -> torch.Tensor:
        cond_batch = dict(prepared_batch)
        uncond_batch = self._unconditional_batch(prepared_batch)
        with self._fixed_teacher_context(), torch.no_grad():
            cond = self._teacher_hidden(cond_batch)
            uncond = self._teacher_hidden(uncond_batch)
        return uncond + self.cfg_scale * (cond - uncond)

    def _teacher_hidden(self, batch: Dict[str, Any]) -> torch.Tensor:
        output = self.teacher_model.model_predict(batch)
        buffer = output.get("hidden_states_buffer")
        if not isinstance(buffer, dict):
            raise ValueError("Self-Transcendence teacher forward did not return a hidden-state buffer.")
        key = f"layer_{self.teacher_block}"
        hidden = buffer.get(key)
        if hidden is None:
            raise ValueError(f"Self-Transcendence teacher forward did not capture {key}.")
        flattened = self._flatten_hidden(hidden).detach()
        buffer.clear()
        return flattened

    @staticmethod
    def _unconditional_batch(prepared_batch: Dict[str, Any]) -> Dict[str, Any]:
        negative = prepared_batch.get("negative_encoder_hidden_states")
        if negative is None:
            negative = prepared_batch.get("negative_prompt_embeds")
        if negative is None:
            raise ValueError("Self-Transcendence self stage requires cached empty-prompt embeddings.")
        batch = dict(prepared_batch)
        batch["encoder_hidden_states"] = negative
        batch["prompt_embeds"] = negative
        replacements = {
            "negative_encoder_attention_mask": "encoder_attention_mask",
            "negative_text_token_tags": "text_token_tags",
        }
        for source, destination in replacements.items():
            if prepared_batch.get(source) is not None:
                batch[destination] = prepared_batch[source]
        added_cond_kwargs = dict(prepared_batch.get("added_cond_kwargs") or {})
        if prepared_batch.get("negative_add_text_embeds") is not None:
            added_cond_kwargs["text_embeds"] = prepared_batch["negative_add_text_embeds"]
        batch["added_cond_kwargs"] = added_cond_kwargs
        return batch

    def _masked_mse(self, prediction: torch.Tensor, target: torch.Tensor, prepared_batch: Dict[str, Any]) -> torch.Tensor:
        per_sample = F.mse_loss(prediction.float(), target.float(), reduction="none").flatten(1).mean(1)
        times = self._normalized_timesteps(prepared_batch, per_sample.shape[0], per_sample.device)
        mask = (times >= self.timestep_min) & (times <= self.timestep_max)
        if not torch.any(mask):
            return prediction.sum() * 0.0
        return per_sample[mask].mean()

    def _normalized_timesteps(self, prepared_batch: Dict[str, Any], batch_size: int, device: torch.device) -> torch.Tensor:
        sigmas = prepared_batch.get("sigmas")
        if sigmas is not None:
            return sigmas.to(device=device, dtype=torch.float32).reshape(batch_size, -1)[:, 0]
        timesteps = prepared_batch.get("timesteps")
        if timesteps is None:
            raise ValueError("Self-Transcendence requires sigmas or timesteps for range masking.")
        count = int(getattr(getattr(self.noise_scheduler, "config", None), "num_train_timesteps", 1000))
        return timesteps.to(device=device, dtype=torch.float32).reshape(batch_size, -1)[:, 0] / max(count - 1, 1)

    @staticmethod
    def _flatten_hidden(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim == 3:
            return hidden
        if hidden.ndim == 4:
            return hidden.reshape(hidden.shape[0], -1, hidden.shape[-1])
        raise ValueError(f"Self-Transcendence requires 3D or 4D hidden states, got {tuple(hidden.shape)}.")

    def _projector_dtype(self) -> torch.dtype:
        return next(self.projector.parameters()).dtype

    @classmethod
    def _vae_target_tokens(cls, latents: torch.Tensor, hidden: torch.Tensor, token_count: int) -> torch.Tensor:
        if latents.ndim not in (3, 4, 5):
            raise ValueError(f"Self-Transcendence VAE guidance does not support latent shape {tuple(latents.shape)}.")
        spatial_shape = tuple(int(value) for value in latents.shape[2:])
        if hidden.ndim == 4 and len(spatial_shape) == 3:
            temporal_tokens = int(hidden.shape[1])
            spatial_tokens = int(hidden.shape[2])
            if temporal_tokens * spatial_tokens != token_count:
                raise ValueError("Self-Transcendence hidden-state token geometry is inconsistent.")
            token_grid = (temporal_tokens, *cls._factor_grid(spatial_shape[1:], spatial_tokens))
        elif hidden.ndim == 4 and len(spatial_shape) == 2:
            token_grid = (int(hidden.shape[1]), int(hidden.shape[2]))
            if math.prod(token_grid) != token_count:
                raise ValueError("Self-Transcendence hidden-state token geometry is inconsistent.")
        else:
            token_grid = cls._factor_grid(spatial_shape, token_count)
        if any(source % grid for source, grid in zip(spatial_shape, token_grid)):
            raise ValueError(f"Self-Transcendence token grid {token_grid} does not divide latent shape {spatial_shape}.")
        patch_shape = tuple(source // grid for source, grid in zip(spatial_shape, token_grid))

        reshape_shape = [latents.shape[0], latents.shape[1]]
        for grid, patch in zip(token_grid, patch_shape):
            reshape_shape.extend((grid, patch))
        patched = latents.float().reshape(reshape_shape)
        grid_axes = [2 + 2 * index for index in range(len(token_grid))]
        patch_axes = [axis + 1 for axis in grid_axes]
        permutation = [0, *grid_axes, 1, *patch_axes]
        return patched.permute(permutation).reshape(latents.shape[0], token_count, -1)

    @staticmethod
    def _factor_grid(source_shape: tuple[int, ...], token_count: int) -> tuple[int, ...]:
        if math.prod(source_shape) == token_count:
            return source_shape
        if len(source_shape) == 1:
            if source_shape[0] % token_count == 0:
                return (token_count,)
        if len(source_shape) == 2:
            height, width = source_shape
            candidates = [
                (factor, token_count // factor)
                for factor in range(1, token_count + 1)
                if token_count % factor == 0 and height % factor == 0 and width % (token_count // factor) == 0
            ]
            if candidates:
                return min(candidates, key=lambda pair: abs((pair[0] / pair[1]) - (height / width)))
        if len(source_shape) == 3:
            frames, height, width = source_shape
            candidates = []
            for temporal in range(1, token_count + 1):
                if token_count % temporal or frames % temporal:
                    continue
                spatial = token_count // temporal
                for grid_h in range(1, spatial + 1):
                    if spatial % grid_h:
                        continue
                    grid_w = spatial // grid_h
                    if height % grid_h or width % grid_w:
                        continue
                    score = abs(temporal / max(frames, 1) - grid_h / max(height, 1))
                    score += abs(grid_h / max(height, 1) - grid_w / max(width, 1))
                    candidates.append((score, (temporal, grid_h, grid_w)))
            if candidates:
                return min(candidates, key=lambda item: item[0])[1]
        raise ValueError(f"Self-Transcendence could not map {token_count} tokens onto latent spatial shape {source_shape}.")


DistillationRegistry.register(
    "self_transcendence",
    SelfTranscendenceDistiller,
    requires_distillation_cache=False,
    data_requirements=[[DatasetType.IMAGE, DatasetType.VIDEO, DatasetType.AUDIO]],
    requirement_notes="Requires a diffusion transformer with latent-token hidden-state capture.",
)
