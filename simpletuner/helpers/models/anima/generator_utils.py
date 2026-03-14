# Vendored from diffusers-anima: /src/diffusers-anima/src/diffusers_anima/pipelines/anima/generator_utils.py
# Adapted for SimpleTuner local imports.

"""Generator and noise runtime utilities for AnimaPipeline."""

from __future__ import annotations

import torch

from .sampling import GeneratorInput


def _normalize_generator(
    generator: GeneratorInput | None,
    *,
    batch_size: int,
) -> torch.Generator | list[torch.Generator] | None:
    if generator is None:
        return None

    if isinstance(generator, (list, tuple)):
        return _normalize_generator_sequence(generator, batch_size=batch_size)

    if not isinstance(generator, torch.Generator):
        raise ValueError("`generator` must be a torch.Generator or a list/tuple of torch.Generator instances.")
    return generator


def _generator_device_type(generator: torch.Generator) -> str:
    return generator.device.type if hasattr(generator, "device") else "cpu"


def _normalize_generator_sequence(
    generators: list[torch.Generator] | tuple[torch.Generator, ...],
    *,
    batch_size: int,
) -> list[torch.Generator] | None:
    if batch_size < 1:
        raise ValueError(f"`batch_size` must be >= 1, got {batch_size}.")
    if len(generators) != batch_size:
        raise ValueError(f"`generator` list length must match batch size ({batch_size}), got {len(generators)}.")

    normalized: list[torch.Generator] = []
    first_device: str | None = None
    for item in generators:
        if not isinstance(item, torch.Generator):
            raise ValueError("`generator` list items must be torch.Generator instances.")
        device_type = _generator_device_type(item)
        if first_device is None:
            first_device = device_type
        elif device_type != first_device:
            raise ValueError(
                "`generator` list items must be on the same device type. "
                f"Got mixed devices: {first_device}, {device_type}."
            )
        normalized.append(item)
    return normalized


def _resolve_noise_runtime(
    *,
    execution_device: str,
    generator: GeneratorInput | None,
    batch_size: int,
) -> tuple[
    torch.Generator | list[torch.Generator] | None,
    torch.Generator | list[torch.Generator] | None,
    str,
    torch.dtype,
]:
    """Resolve RNG objects and noise device for latent initialization/sampling."""
    noise_device = execution_device
    noise_dtype = torch.float32

    provided_generator = _normalize_generator(generator, batch_size=batch_size)
    if provided_generator is not None:
        if isinstance(provided_generator, list):
            generator_device = provided_generator[0].device.type if len(provided_generator) > 0 else "cpu"
        else:
            generator_device = provided_generator.device.type if hasattr(provided_generator, "device") else "cpu"
        return provided_generator, provided_generator, generator_device, noise_dtype

    return None, None, noise_device, noise_dtype
