from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.dmd.distiller import DMDDistiller
from simpletuner.helpers.distillation.registry import DistillationRegistry
from simpletuner.helpers.training.state_tracker import StateTracker

from .generator import SelfForcingODEGenerator


class SelfForcingDistillation(DistillationBase):
    """Self-Forcing variant of DMD that materialises batches from caption + cache inputs."""

    _DMD_DEFAULTS: Dict[str, Any] = {
        "dmd_denoising_steps": "1000,757,522",
        "min_timestep_ratio": 0.02,
        "max_timestep_ratio": 0.98,
        "generator_update_interval": 1,
        "real_score_guidance_scale": 3.0,
        "fake_score_lr": 1e-5,
        "fake_score_weight_decay": 0.01,
        "fake_score_betas": (0.9, 0.999),
        "fake_score_eps": 1e-8,
        "fake_score_grad_clip": 1.0,
        "fake_score_guidance_scale": 0.0,
        "num_frame_per_block": 3,
        "independent_first_frame": False,
        "same_step_across_blocks": False,
        "last_step_only": False,
        "num_training_frames": 21,
        "context_noise": 0,
        "ts_schedule": True,
        "ts_schedule_max": False,
        "min_score_timestep": 0,
        "timestep_shift": 1.0,
    }
    _SELF_FORCING_DEFAULTS: Dict[str, Any] = {
        "distillation_type": "self_forcing",
        "ode_generator": {},
    }

    def __init__(
        self,
        teacher_model,
        student_model=None,
        *,
        noise_scheduler=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        merged_config: Dict[str, Any] = dict(self._DMD_DEFAULTS)
        merged_config.update(self._SELF_FORCING_DEFAULTS)
        if config:
            merged_config.update(config)

        super().__init__(teacher_model, student_model, merged_config)

        if noise_scheduler is None:
            raise ValueError("Self-forcing distillation requires a noise scheduler.")

        self.noise_scheduler = noise_scheduler
        self._ode_generator = SelfForcingODEGenerator(config=merged_config.get("ode_generator", {}))
        self._distillation_caches: List[Any] = []
        self._cache_cycle: Optional[Iterable[Any]] = None

        # Delegate DMD-specific optimisation to the existing distiller implementation.
        self._dmd = DMDDistiller(
            teacher_model=teacher_model,
            student_model=student_model,
            noise_scheduler=noise_scheduler,
            config=dict(merged_config),
        )
        # Keep configs aligned so downstream consumers see the populated DMD defaults.
        self.config = self._dmd.config

    def requires_distillation_cache(self) -> bool:
        return True

    def get_required_distillation_cache_type(self) -> Optional[str]:
        return self.config.get("distillation_type", "self_forcing")

    def get_ode_generator_provider(self):
        return self._ode_generator

    def consumes_caption_batches(self) -> bool:
        return True

    def prepare_caption_batch(self, caption_batch: Dict[str, Any], model, state) -> Dict[str, Any]:
        captions = self._extract_captions(caption_batch)
        if not captions:
            raise ValueError("Caption batch was empty; cannot prepare distillation inputs.")

        cache_entries = [self._next_cache_payload() for _ in captions]
        latents = self._build_latent_batch(cache_entries)

        text_output = model.encode_text_batch(captions, is_negative_prompt=False)
        synthetic_batch = self._build_synthetic_batch(
            captions=captions,
            latents=latents,
            text_output=text_output,
            caption_batch=caption_batch,
            cache_entries=cache_entries,
        )

        prepared = model.prepare_batch(synthetic_batch, state)
        prepared["captions"] = list(captions)
        prepared["records"] = caption_batch.get("records", [])
        prepared["data_backend_id"] = caption_batch.get("data_backend_id")
        metadata = [entry.get("metadata", {}) for entry in cache_entries]
        prepared["distillation_metadata"] = metadata
        prepared["distillation_cache_entries"] = cache_entries

        latents_device = prepared["latents"].device
        latents_dtype = prepared["latents"].dtype

        noise_tensor = synthetic_batch.get("noise")
        if noise_tensor is not None:
            prepared["noise"] = noise_tensor.to(device=latents_device, dtype=latents_dtype)

        input_noise_tensor = synthetic_batch.get("input_noise")
        if input_noise_tensor is not None:
            prepared["input_noise"] = input_noise_tensor.to(device=latents_device, dtype=latents_dtype)
        elif "noise" in prepared:
            prepared["input_noise"] = prepared["noise"]

        timestep_tensor = synthetic_batch.get("timesteps")
        if timestep_tensor is not None:
            prepared["timesteps"] = timestep_tensor.to(device=latents_device, dtype=torch.long).view(-1)

        if self.noise_scheduler is None:
            raise ValueError("Self-forcing distillation requires a noise scheduler.")
        if "timesteps" in prepared and "input_noise" in prepared:
            prepared["noisy_latents"] = self.noise_scheduler.add_noise(
                prepared["latents"].float(),
                prepared["input_noise"].float(),
                prepared["timesteps"],
            ).to(device=latents_device, dtype=latents_dtype)
        prepared["clean_latents"] = prepared["latents"]

        return prepared

    def prepare_batch(self, batch, model, state):
        return self._dmd.prepare_batch(batch, model, state)

    def compute_distill_loss(self, prepared_batch, model_output, original_loss):
        return self._dmd.compute_distill_loss(prepared_batch, model_output, original_loss)

    def pre_training_step(self, model, step):
        return self._dmd.pre_training_step(model, step)

    def post_training_step(self, model, step):
        return self._dmd.post_training_step(model, step)

    def generator_loss_step(self, prepared_batch, model_output, current_loss):
        return self._dmd.generator_loss_step(prepared_batch, model_output, current_loss)

    def discriminator_step(self, prepared_batch: Dict[str, Any], **kwargs):
        return self._dmd.discriminator_step(prepared_batch=prepared_batch, **kwargs)

    def on_load_checkpoint(self, ckpt_dir: str):
        return self._dmd.on_load_checkpoint(ckpt_dir)

    def on_save_checkpoint(self, step: int, ckpt_dir: str):
        return self._dmd.on_save_checkpoint(step, ckpt_dir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_captions(self, caption_batch: Dict[str, Any]) -> List[str]:
        captions = caption_batch.get("captions") or []
        return [str(caption) for caption in captions if caption is not None]

    def _ensure_cache_cycle(self) -> None:
        if self._cache_cycle is not None and self._distillation_caches:
            return

        try:
            backend_map = StateTracker.get_data_backends(
                _type=DatasetType.DISTILLATION_CACHE,
                _types=[DatasetType.DISTILLATION_CACHE],
            )
        except Exception:
            backend_map = {}

        caches = []
        for backend in backend_map.values():
            cache = backend.get("distillation_cache")
            if cache is not None:
                caches.append(cache)

        if not caches:
            raise ValueError("Self-forcing distillation requires at least one distillation_cache backend.")

        self._distillation_caches = caches
        self._cache_cycle = itertools.cycle(self._distillation_caches)

    def _next_cache_payload(self) -> Dict[str, Any]:
        self._ensure_cache_cycle()
        if not self._distillation_caches:
            raise ValueError("No distillation caches were configured for self-forcing distillation.")

        for _ in range(len(self._distillation_caches)):
            cache = next(self._cache_cycle)
            payload, artifact_path = cache.load_next_pair()
            if payload is None:
                continue

            entry = payload if isinstance(payload, dict) else {"value": payload}
            metadata = entry.get("metadata") or {}
            if artifact_path:
                metadata.setdefault("artifact_path", artifact_path)
            metadata.setdefault("distillation_type", cache.distillation_type)
            entry["metadata"] = metadata
            return entry

        raise ValueError("Distillation caches did not yield any artifacts for caption batch preparation.")

    def _build_latent_batch(self, cache_entries: Sequence[Dict[str, Any]]) -> torch.Tensor:
        latents: List[torch.Tensor] = []
        for entry in cache_entries:
            latent_tensor = entry.get("latents")
            if latent_tensor is None:
                raise ValueError("Distillation cache entry missing 'latents' tensor.")
            if not torch.is_tensor(latent_tensor):
                latent_tensor = torch.tensor(latent_tensor, dtype=torch.float32)
            if latent_tensor.ndim == 4 and latent_tensor.shape[0] == 1:
                latent_tensor = latent_tensor.squeeze(0)
            if latent_tensor.ndim != 3:
                raise ValueError(
                    f"Expected cached latents to have shape (C, H, W); received tensor with shape {tuple(latent_tensor.shape)}"
                )
            latents.append(latent_tensor.clone().detach())

        return torch.stack(latents, dim=0)

    def _build_synthetic_batch(
        self,
        *,
        captions: Sequence[str],
        latents: torch.Tensor,
        text_output: Dict[str, Any],
        caption_batch: Dict[str, Any],
        cache_entries: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        synthetic_batch: Dict[str, Any] = {
            "latent_batch": latents,
            "prompts": list(captions),
            "text_encoder_output": text_output,
            "prompt_embeds": text_output.get("prompt_embeds"),
            "add_text_embeds": text_output.get("pooled_prompt_embeds"),
            "batch_time_ids": text_output.get("batch_time_ids"),
            "encoder_attention_mask": text_output.get("attention_masks"),
            "conditioning_pixel_values": None,
            "conditioning_latents": None,
            "conditioning_image_embeds": None,
            "batch_luminance": 0.0,
            "is_regularisation_data": False,
            "is_i2v_data": False,
            "caption_records": caption_batch.get("records", []),
            "distillation_metadata": [entry.get("metadata", {}) for entry in cache_entries],
            "data_backend_id": caption_batch.get("data_backend_id"),
        }

        noise_tensor = self._stack_optional_latent(cache_entries, "noise", latents)
        if noise_tensor is not None:
            synthetic_batch["noise"] = noise_tensor

        input_noise_tensor = self._stack_optional_latent(cache_entries, "input_noise", latents)
        if input_noise_tensor is not None:
            synthetic_batch["input_noise"] = input_noise_tensor
        elif noise_tensor is not None:
            synthetic_batch["input_noise"] = noise_tensor.clone()

        timestep_tensor = self._stack_timesteps(cache_entries)
        if timestep_tensor is not None:
            synthetic_batch["timesteps"] = timestep_tensor

        return synthetic_batch

    def _stack_optional_latent(
        self,
        cache_entries: Sequence[Dict[str, Any]],
        key: str,
        latents: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        tensors: List[torch.Tensor] = []
        expected_shape = latents.shape[1:]
        for entry in cache_entries:
            value = entry.get(key)
            if value is None:
                return None
            tensor = value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.float32)
            if tensor.ndim == len(expected_shape) + 1 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.ndim != len(expected_shape):
                raise ValueError(
                    f"Expected cached '{key}' tensor to have shape {tuple(expected_shape)}; "
                    f"received tensor with shape {tuple(tensor.shape)}"
                )
            if tuple(tensor.shape) != tuple(expected_shape):
                tensor = tensor.reshape(expected_shape)
            tensors.append(tensor.clone().detach())

        if not tensors:
            return None
        return torch.stack(tensors, dim=0)

    def _stack_timesteps(self, cache_entries: Sequence[Dict[str, Any]]) -> Optional[torch.Tensor]:
        timesteps: List[int] = []
        for entry in cache_entries:
            value = entry.get("timesteps")
            if value is None:
                return None
            if torch.is_tensor(value):
                scalar = value.view(-1)[0].item()
            elif isinstance(value, (list, tuple)) and value:
                scalar = value[0]
            else:
                scalar = value
            timesteps.append(int(scalar))
        if not timesteps:
            return None
        return torch.tensor(timesteps, dtype=torch.long)


DistillationRegistry.register(
    "self_forcing",
    SelfForcingDistillation,
    requires_distillation_cache=True,
    distillation_type="self_forcing",
    data_requirements=[DatasetType.CAPTION],
    is_data_generator=True,
    requirement_notes="Requires distillation_cache backend matching distillation_type.",
)
