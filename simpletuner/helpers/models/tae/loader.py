"""Helpers for loading Tiny AutoEncoders for validation previews."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Union

import requests
import torch
from diffusers import AutoencoderTiny

from .taehv import TAEHV
from .types import ImageTAESpec, VideoTAESpec

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("cache") / "tae"


def _resolve_device(device: Union[str, torch.device, None]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_dtype(dtype: Union[str, torch.dtype, None]) -> torch.dtype:
    if dtype is None:
        return torch.float16 if torch.cuda.is_available() else torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    candidate = getattr(torch, str(dtype), None)
    if isinstance(candidate, torch.dtype):
        return candidate
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _download_checkpoint(url: str, filename: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    destination = _CACHE_DIR / filename
    if destination.exists():
        return destination
    logger.info("Downloading Tiny AutoEncoder weights from %s", url)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    temp_path = destination.with_suffix(".tmp")
    with open(temp_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1_048_576):
            if chunk:
                handle.write(chunk)
    os.replace(temp_path, destination)
    return destination


class _BaseTAEDecoder:
    is_video: bool = False

    def __init__(self):
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None

    def decode(self, latents: torch.Tensor) -> torch.Tensor:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError


class ImageTAEDecoder(_BaseTAEDecoder):
    is_video = False

    def __init__(self, model: AutoencoderTiny, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.model = model.eval()
        self.device = device
        self.dtype = dtype

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            decoded = self.model.decode(latents, return_dict=False)[0]
        decoded = (decoded / 2 + 0.5).clamp_(0, 1)
        return decoded


class VideoTAEDecoder(_BaseTAEDecoder):
    is_video = True

    def __init__(self, tae: TAEHV, parallel: bool, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.model = tae.eval()
        self.parallel = parallel
        self.device = device
        self.dtype = dtype

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 4:
            latents = latents.unsqueeze(2)
        if latents.ndim != 5:
            raise ValueError(f"Video latents must be rank 5, got shape {latents.shape}")
        # Convert from (B, C, T, H, W) to (B, T, C, H, W) if needed
        if latents.shape[1] == self.model.latent_channels:
            latents = latents.permute(0, 2, 1, 3, 4)
        latents = latents.to(device=self.device, dtype=self.dtype)
        with torch.no_grad():
            frames = self.model.decode_video(latents, parallel=self.parallel, show_progress_bar=False)
        return frames.clamp_(0, 1)


def _instantiate_image_decoder(spec: ImageTAESpec, device: torch.device, dtype: torch.dtype) -> ImageTAEDecoder:
    model = AutoencoderTiny.from_pretrained(
        spec.repo_id,
        subfolder=spec.subfolder,
        variant=spec.variant,
        torch_dtype=dtype,
    )
    model.to(device)
    return ImageTAEDecoder(model, device=device, dtype=dtype)


def _instantiate_video_decoder(spec: VideoTAESpec, device: torch.device, dtype: torch.dtype) -> VideoTAEDecoder:
    checkpoint_path = _download_checkpoint(spec.download_url, spec.filename)
    decoder_kwargs = {
        "checkpoint_path": str(checkpoint_path),
        "decoder_time_upscale": spec.decoder_time_upscale or (True, True),
        "decoder_space_upscale": spec.decoder_space_upscale or (True, True, True),
        "patch_size": spec.patch_size or 1,
        "latent_channels": spec.latent_channels or 16,
    }
    tae = TAEHV(**decoder_kwargs)
    tae.to(device=device, dtype=dtype)
    return VideoTAEDecoder(tae, parallel=spec.parallel_decode, device=device, dtype=dtype)


def load_tae_decoder(
    spec: Union[ImageTAESpec, VideoTAESpec],
    *,
    device: Union[torch.device, str, None] = None,
    dtype: Union[torch.dtype, str, None] = None,
) -> _BaseTAEDecoder:
    """
    Instantiate the requested Tiny AutoEncoder on the given device/dtype.
    """

    device = _resolve_device(device)
    dtype = _resolve_dtype(dtype)
    if isinstance(spec, ImageTAESpec):
        decoder = _instantiate_image_decoder(spec, device, dtype)
    elif isinstance(spec, VideoTAESpec):
        decoder = _instantiate_video_decoder(spec, device, dtype)
    else:  # pragma: no cover - defensive
        raise TypeError(f"Unsupported TAE spec: {spec}")
    decoder.device = device
    decoder.dtype = dtype
    return decoder
