"""Dataclasses describing Tiny AutoEncoder loading options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ImageTAESpec:
    """
    Specification for AutoencoderTiny weights hosted on Hugging Face.

    Attributes:
        repo_id: Hugging Face repo id passed to AutoencoderTiny.from_pretrained.
        subfolder: Optional subfolder containing weights.
        variant: Optional variant identifier (fp16, bf16, etc.).
    """

    repo_id: str
    subfolder: Optional[str] = None
    variant: Optional[str] = None


@dataclass(frozen=True)
class VideoTAESpec:
    """
    Specification for TAEHV-style video decoders hosted as raw checkpoint files.

    Attributes:
        filename: Name of the checkpoint file to download.
        base_url: Base URL where the checkpoint can be downloaded.
        decoder_time_upscale: Optional tuple controlling temporal upscaling flags.
        decoder_space_upscale: Optional tuple controlling spatial upscaling flags.
        patch_size: Optional pixel shuffle patch size override.
        latent_channels: Optional latent channel count override.
        parallel_decode: Whether to decode all frames in parallel (more memory, faster).
        description: Optional human readable description for logging.
    """

    filename: str
    base_url: str = "https://raw.githubusercontent.com/madebyollin/taehv/main"
    decoder_time_upscale: Optional[Tuple[bool, ...]] = None
    decoder_space_upscale: Optional[Tuple[bool, ...]] = None
    patch_size: Optional[int] = None
    latent_channels: Optional[int] = None
    parallel_decode: bool = False
    description: Optional[str] = None
    sha256: Optional[str] = None

    @property
    def download_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/{self.filename}"
