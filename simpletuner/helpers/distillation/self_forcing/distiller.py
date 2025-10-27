from __future__ import annotations

import itertools
from typing import Any, Dict, Iterable, List, Optional, Sequence

import torch

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.registry import DistillationRegistry
from simpletuner.helpers.training.state_tracker import StateTracker

from .generator import SelfForcingODEGenerator


class SelfForcingDistillation(DistillationBase):
    """Lightweight distiller that demonstrates self-forcing ODE pair generation."""

    def __init__(
        self,
        teacher_model,
        student_model=None,
        *,
        noise_scheduler=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        default_config = {
            "distillation_type": "self_forcing",
            "ode_generator": {},
        }
        if config:
            default_config.update(config)

        super().__init__(teacher_model, student_model, default_config)
        self.noise_scheduler = noise_scheduler
        self._ode_generator = SelfForcingODEGenerator(config=default_config.get("ode_generator", {}))
        self._distillation_caches = []
        self._cache_cycle: Optional[Iterable[Any]] = None

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
        prepared["distillation_metadata"] = [entry.get("metadata", {}) for entry in cache_entries]

        return prepared

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _extract_captions(self, caption_batch: Dict[str, Any]) -> List[str]:
        captions = caption_batch.get("captions") or []
        normalized = [str(caption) for caption in captions if caption is not None]
        return normalized

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
            latents.append(latent_tensor)

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

        # Carry auxiliary payload values through for downstream consumers.
        first_entry = cache_entries[0] if cache_entries else {}
        for key in ("sigmas", "timesteps", "noise", "input_noise", "guidance_scale"):
            if key in first_entry and key not in synthetic_batch:
                synthetic_batch[key] = first_entry[key]

        return synthetic_batch


DistillationRegistry.register(
    "self_forcing",
    SelfForcingDistillation,
    requires_distillation_cache=True,
    distillation_type="self_forcing",
    data_requirements=[DatasetType.CAPTION],
    is_data_generator=True,
    requirement_notes="Requires distillation_cache backend matching distillation_type.",
)
