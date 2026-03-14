# Vendored from diffusers-anima: /src/diffusers-anima/src/diffusers_anima/pipelines/anima/image_processing.py
# Adapted for SimpleTuner local imports.

"""Anima image preprocessing and VAE encode/decode utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from diffusers import AutoencoderKL
    from diffusers.utils import BaseOutput

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ImageInput = Image.Image | np.ndarray | torch.Tensor
ImageBatchInput = ImageInput | list[ImageInput] | tuple[ImageInput, ...]

_IMAGE_INPUT_TYPES = (Image.Image, np.ndarray, torch.Tensor, list, tuple)
_IMAGE_BATCH_ITEM_TYPES = (Image.Image, np.ndarray, torch.Tensor)


def _reshape_image_tensor_to_bchw(
    image: np.ndarray | torch.Tensor,
    *,
    input_label: str,
) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image)
    else:
        tensor = image.detach()

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        if tensor.shape[0] in {1, 3} and tensor.shape[-1] not in {1, 3}:
            tensor = tensor.unsqueeze(0)
        elif tensor.shape[-1] in {1, 3}:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        else:
            raise ValueError(f"`{input_label}` must have channel size 1 or 3. Got shape {tuple(tensor.shape)}.")
    elif tensor.ndim == 4:
        if tensor.shape[1] not in {1, 3}:
            if tensor.shape[-1] in {1, 3}:
                tensor = tensor.permute(0, 3, 1, 2)
            else:
                raise ValueError(f"`{input_label}` must have channel size 1 or 3. Got shape {tuple(tensor.shape)}.")
    else:
        raise ValueError(f"`{input_label}` must be 2D/3D/4D. Got shape {tuple(tensor.shape)}.")

    if tensor.shape[0] < 1:
        raise ValueError(f"`{input_label}` batch size must be >= 1. Got {tensor.shape[0]}.")
    return tensor


def _normalize_tensor_to_unit_interval(
    tensor: torch.Tensor,
    *,
    input_label: str,
) -> torch.Tensor:
    tensor = tensor.to(dtype=torch.float32)
    value_min = float(tensor.min().item())
    value_max = float(tensor.max().item())

    if value_min >= 0.0 and value_max <= 1.0:
        normalized = tensor
    elif value_min >= -1.0 and value_max <= 1.0:
        normalized = (tensor + 1.0) / 2.0
    elif value_min >= 0.0 and value_max <= 255.0:
        normalized = tensor / 255.0
    else:
        raise ValueError(f"`{input_label}` value range is unsupported: min={value_min:.4f}, max={value_max:.4f}.")

    return normalized.clamp(0.0, 1.0)


def prepare_init_image_tensor(
    image: ImageBatchInput,
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    """Convert an image input (PIL / ndarray / tensor / list) to a BCHW float tensor in [-1, 1]."""
    if isinstance(image, (list, tuple)):
        if len(image) == 0:
            raise ValueError("`image` list/tuple must not be empty.")
        batch_tensors: list[torch.Tensor] = []
        for item in image:
            if isinstance(item, (list, tuple)):
                raise ValueError("Nested list/tuple in `image` is not supported.")
            batch_tensors.append(
                prepare_init_image_tensor(
                    item,
                    width=width,
                    height=height,
                )
            )
        tensor = torch.cat(batch_tensors, dim=0)
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
        tensor = torch.from_numpy(np.array(pil_image, copy=True)).permute(2, 0, 1).unsqueeze(0)
    else:
        tensor = _reshape_image_tensor_to_bchw(image, input_label="image")
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor.shape[1] != 3:
            raise ValueError(f"`image` must have 1 or 3 channels, got {tensor.shape[1]}.")

        tensor = _normalize_tensor_to_unit_interval(tensor, input_label="image")
        tensor = F.interpolate(
            tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return tensor.mul(2.0).sub(1.0)

    tensor = _normalize_tensor_to_unit_interval(tensor, input_label="image")
    return tensor.mul(2.0).sub(1.0)


def prepare_inpaint_mask_tensor(
    mask_image: ImageBatchInput,
    *,
    width: int,
    height: int,
) -> torch.Tensor:
    """Convert a mask input to a single-channel float tensor in [0, 1]."""
    if isinstance(mask_image, (list, tuple)):
        if len(mask_image) == 0:
            raise ValueError("`mask_image` list/tuple must not be empty.")
        mask_tensors: list[torch.Tensor] = []
        for item in mask_image:
            if isinstance(item, (list, tuple)):
                raise ValueError("Nested list/tuple in `mask_image` is not supported.")
            mask_tensors.append(
                prepare_inpaint_mask_tensor(
                    item,
                    width=width,
                    height=height,
                )
            )
        mask = torch.cat(mask_tensors, dim=0)
    elif isinstance(mask_image, Image.Image):
        pil_mask = mask_image.convert("L").resize((width, height), Image.Resampling.LANCZOS)
        mask = torch.from_numpy(np.array(pil_mask, copy=True)).unsqueeze(0).unsqueeze(0)
    else:
        mask = _reshape_image_tensor_to_bchw(mask_image, input_label="mask_image")
        if mask.shape[1] == 3:
            mask = mask.mean(dim=1, keepdim=True)
        elif mask.shape[1] != 1:
            raise ValueError(f"`mask_image` must have 1 or 3 channels, got {mask.shape[1]}.")
        mask = _normalize_tensor_to_unit_interval(mask, input_label="mask_image")
        mask = F.interpolate(
            mask,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return mask.clamp(0.0, 1.0)

    mask = _normalize_tensor_to_unit_interval(mask, input_label="mask_image")
    return mask.clamp(0.0, 1.0)


def _retrieve_vae_latents(
    encoder_output: "BaseOutput",
    *,
    generator: torch.Generator | list[torch.Generator] | None,
) -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist"):
        if generator is None:
            return encoder_output.latent_dist.sample()
        return encoder_output.latent_dist.sample(generator)  # type: ignore[arg-type]
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Failed to retrieve latents from VAE encoder output.")


def encode_image_to_latents(
    vae: "AutoencoderKL",
    *,
    image_tensor: torch.Tensor,
    execution_device: str,
    model_dtype: torch.dtype,
    generator: torch.Generator | list[torch.Generator] | None,
    sample_dtype: torch.dtype,
) -> torch.Tensor:
    """Encode a BCHW image tensor to latents and apply the Anima latent normalisation."""
    image = image_tensor.to(device=execution_device, dtype=model_dtype).unsqueeze(2)
    with torch.inference_mode():
        encoded = vae.encode(image)
    image_latents = _retrieve_vae_latents(encoded, generator=generator).to(dtype=sample_dtype)

    latents_mean = torch.tensor(
        vae.config.latents_mean,
        dtype=image_latents.dtype,
        device=image_latents.device,
    ).view(1, 16, 1, 1, 1)
    latents_std = (
        1.0
        / torch.tensor(
            vae.config.latents_std,
            dtype=image_latents.dtype,
            device=image_latents.device,
        )
    ).view(1, 16, 1, 1, 1)
    return (image_latents - latents_mean) * latents_std


def _ensure_finite(tensor: torch.Tensor, *, name: str, runtime_dtype: torch.dtype) -> None:
    if torch.isfinite(tensor).all():
        return

    if runtime_dtype == torch.float16:
        raise RuntimeError(
            f"{name} contains NaN/Inf."
            " dtype=float16 is unstable for this model/environment."
            " Use --dtype auto, bfloat16, or float32."
        )
    raise RuntimeError(f"{name} contains NaN/Inf.")


def decode_latents(
    vae: "AutoencoderKL",
    latents: torch.Tensor,
    *,
    runtime_dtype: torch.dtype,
) -> list[Image.Image]:
    """Decode latents to PIL images, applying inverse Anima latent statistics."""
    _ensure_finite(latents, name="latents before decode", runtime_dtype=runtime_dtype)
    latents = latents.to(vae.dtype)
    latents_mean = torch.tensor(vae.config.latents_mean, dtype=latents.dtype, device=latents.device).view(1, 16, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std, dtype=latents.dtype, device=latents.device).view(1, 16, 1, 1, 1)
    latents = latents * latents_std + latents_mean

    with torch.inference_mode():
        image = vae.decode(latents, return_dict=False)[0][:, :, 0]

    _ensure_finite(image, name="VAE decode output", runtime_dtype=runtime_dtype)
    image = image.float().clamp(-1.0, 1.0)
    image = ((image + 1.0) / 2.0).clamp(0.0, 1.0)
    image_np = image.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray((item * 255).round().astype("uint8")) for item in image_np]


def latent_hw(
    height: int,
    width: int,
    vae_scale_factor: int = 8,
    patch_size: int = 2,
) -> tuple[int, int, int, int]:
    """Return ``(height, width, latent_h, latent_w)`` aligned to the spatial step."""
    step = vae_scale_factor * patch_size
    height = step * (height // step)
    width = step * (width // step)
    return height, width, height // vae_scale_factor, width // vae_scale_factor


def align_tensor_batch_size(
    tensor: torch.Tensor,
    *,
    target_batch_size: int,
    input_name: str,
) -> torch.Tensor:
    """Broadcast or repeat ``tensor`` to ``target_batch_size``."""
    current_batch_size = int(tensor.shape[0])
    if current_batch_size == target_batch_size:
        return tensor
    if current_batch_size == 1:
        return tensor.repeat(target_batch_size, *([1] * (tensor.ndim - 1)))
    if target_batch_size % current_batch_size == 0:
        repeat_count = target_batch_size // current_batch_size
        return tensor.repeat_interleave(repeat_count, dim=0)
    raise ValueError(
        f"`{input_name}` batch size ({current_batch_size}) is incompatible with prompt batch size ({target_batch_size})."
    )
