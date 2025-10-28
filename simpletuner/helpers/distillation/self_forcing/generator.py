from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger("SelfForcingODEGenerator")


class SelfForcingODEGenerator:
    """Simple placeholder generator that materialises deterministic ODE pairs into the cache."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def generate(self, cache, backend_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Materialise a placeholder ODE entry when the cache is empty.

        The real implementation should mirror realtime-video/scripts/generate_ode_pairs.py, but we
        keep things simple here to exercise the caching pipeline.
        """
        cache.discover_all_files()
        if cache.has_cached_pairs() and not self.config.get("force_regenerate", False):
            logger.debug("Distillation cache %s already contains ODE pairs; skipping generation.", cache.id)
            return

        filename = cache.next_artifact_name(prefix=self.config.get("artifact_prefix", "self_forcing"))
        latent_shape = tuple(self.config.get("latent_shape", (4, 64, 64)))
        if len(latent_shape) != 3:
            logger.warning("Invalid latent_shape %s; defaulting to (4, 64, 64).", latent_shape)
            latent_shape = (4, 64, 64)
        latents = torch.zeros((1, *latent_shape), dtype=torch.float32)
        noise = torch.randn_like(latents)
        default_timestep = int(self.config.get("default_timestep", 750))
        timesteps = torch.full((latents.shape[0],), default_timestep, dtype=torch.long)
        guidance_scale = float(self.config.get("guidance_scale", 6.0))

        payload = {
            "metadata": {
                "generated_at": time.time(),
                "distillation_type": cache.distillation_type,
                "guidance_scale": guidance_scale,
            },
            "latents": latents,
            "noise": noise,
            "input_noise": noise.clone(),
            "timesteps": timesteps,
            "guidance_scale": guidance_scale,
        }

        artifact_path = cache.write_tensor(filename, payload)
        logger.info("Wrote placeholder ODE pair to %s", artifact_path)
