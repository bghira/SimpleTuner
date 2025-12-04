from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch

logger = logging.getLogger("PerFlowODEGenerator")


class PerFlowODEGenerator:
    """
    Materialises deterministic ODE pairs for PerFlow/ReFlow distillation.
    """

    def __init__(
        self,
        teacher_model,
        scheduler_provider: Callable[[], object],
        config: Optional[Dict[str, Any]] = None,
    ):
        self.teacher_model = teacher_model
        self._scheduler_provider = scheduler_provider
        self.config = config or {}

    def _target_device_dtype(self) -> Tuple[torch.device, torch.dtype]:
        accelerator = getattr(self.teacher_model, "accelerator", None)
        device = getattr(accelerator, "device", None) or torch.device("cpu")
        dtype = getattr(self.teacher_model, "weight_dtype", None)
        if dtype is None:
            dtype = getattr(getattr(self.teacher_model, "config", None), "weight_dtype", None)
        return device, dtype or torch.float32

    def _infer_latent_shape(self) -> Optional[Sequence[int]]:
        pipeline = getattr(self.teacher_model, "pipeline", None)
        unet = getattr(self.teacher_model, "unet", None) or getattr(pipeline, "unet", None)
        if unet is None or not hasattr(unet, "config"):
            return None

        channels = getattr(unet.config, "in_channels", None)
        sample_size = getattr(unet.config, "sample_size", None)
        if channels is None or sample_size is None:
            return None

        if isinstance(sample_size, (list, tuple)) and len(sample_size) >= 2:
            height, width = int(sample_size[0]), int(sample_size[1])
        else:
            height = width = int(sample_size)

        return int(channels), height, width

    def _prepare_noise(
        self,
        batch_size: int,
        latent_shape: Sequence[int],
        device: torch.device,
        dtype: torch.dtype,
        seed: int,
    ) -> Tuple[torch.Tensor, torch.Generator]:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        shape = (batch_size, *latent_shape)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise, generator

    def _build_scheduler_metadata(self, scheduler) -> Dict[str, Any]:
        config = getattr(scheduler, "config", None)
        return {
            "name": scheduler.__class__.__name__,
            "num_train_timesteps": getattr(config, "num_train_timesteps", None),
            "prediction_type": getattr(config, "prediction_type", None),
            "beta_schedule": getattr(config, "beta_schedule", None),
            "beta_start": getattr(config, "beta_start", None),
            "beta_end": getattr(config, "beta_end", None),
            "t_noise": getattr(config, "t_noise", None),
            "t_clean": getattr(config, "t_clean", None),
            "num_time_windows": getattr(config, "num_time_windows", None),
        }

    def _extract_latents(self, output) -> torch.Tensor:
        latents = getattr(output, "latents", None)
        if latents is None:
            latents = getattr(output, "images", None)
        if latents is None and isinstance(output, (list, tuple)) and output:
            latents = output[0]
        if latents is None:
            raise ValueError("Teacher pipeline did not return latents for ODE cache generation.")
        return latents if torch.is_tensor(latents) else torch.tensor(latents)

    def _run_teacher(
        self,
        *,
        scheduler,
        noise: torch.Tensor,
        generator: torch.Generator,
        prompt: Any,
        num_inference_steps: int,
    ) -> torch.Tensor:
        pipeline = getattr(self.teacher_model, "pipeline", None)
        if pipeline is None:
            raise ValueError("PerFlow ODE generation requires an initialized teacher pipeline.")

        original_scheduler = getattr(pipeline, "scheduler", None)
        pipeline.scheduler = scheduler
        try:
            with torch.no_grad():
                output = pipeline(
                    prompt=[prompt] * noise.shape[0] if isinstance(prompt, str) else prompt,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                    output_type="latent",
                    latents=noise.clone(),
                )
        finally:
            if original_scheduler is not None:
                pipeline.scheduler = original_scheduler

        latents = self._extract_latents(output)
        if latents.ndim == noise.ndim - 1:
            latents = latents.unsqueeze(0)
        return latents

    def generate(self, cache, backend_config: Optional[Dict[str, Any]] = None) -> None:
        backend_cfg = backend_config or {}
        cache.discover_all_files()
        if cache.has_cached_pairs() and not self.config.get("force_regenerate", False):
            logger.debug("Distillation cache %s already contains ODE pairs; skipping generation.", cache.id)
            return

        scheduler = self._scheduler_provider() if callable(self._scheduler_provider) else None
        if scheduler is None:
            raise ValueError("PerFlow ODE generator requires a scheduler provider.")

        required_type = self.config.get("distillation_type") or backend_cfg.get("distillation_type")
        if required_type and cache.distillation_type != required_type:
            raise ValueError(
                f"Distillation cache {cache.id} has type '{cache.distillation_type}' "
                f"but PerFlow generation expects '{required_type}'."
            )

        device, dtype = self._target_device_dtype()
        latent_shape = backend_cfg.get("latent_shape", self.config.get("latent_shape"))
        if latent_shape is None:
            latent_shape = self._infer_latent_shape()
        if latent_shape is None:
            raise ValueError("Could not infer latent shape for PerFlow ODE generation; please provide `latent_shape`.")
        latent_shape = tuple(int(dim) for dim in latent_shape)

        num_pairs = int(backend_cfg.get("num_pairs", self.config.get("num_pairs", 1)))
        batch_size = int(backend_cfg.get("batch_size", self.config.get("batch_size", 1)))
        prompt = backend_cfg.get("prompt", self.config.get("prompt", "perflow distillation"))
        base_seed = int(backend_cfg.get("seed", self.config.get("seed", 0)))
        num_inference_steps = int(
            backend_cfg.get(
                "num_inference_steps",
                self.config.get("num_inference_steps", min(50, getattr(scheduler.config, "num_train_timesteps", 50))),
            )
        )

        scheduler_meta = self._build_scheduler_metadata(scheduler)
        scheduler.set_timesteps(num_inference_steps, device=device)

        for pair_idx in range(num_pairs):
            seed = base_seed + pair_idx
            noise, torch_generator = self._prepare_noise(batch_size, latent_shape, device, dtype, seed)
            latents = self._run_teacher(
                scheduler=scheduler,
                noise=noise,
                generator=torch_generator,
                prompt=prompt,
                num_inference_steps=num_inference_steps,
            )

            payload = {
                "metadata": {
                    "generated_at": time.time(),
                    "distillation_type": cache.distillation_type,
                    "seed": seed,
                    "scheduler": scheduler_meta,
                },
                "latents": latents.detach().cpu(),
                "noise": noise.detach().cpu(),
            }

            filename = cache.next_artifact_name(prefix=self.config.get("artifact_prefix", "perflow"))
            artifact_path = cache.write_tensor(filename, payload)
            logger.info("Wrote PerFlow ODE pair to %s", artifact_path)
