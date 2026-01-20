"""TAEF2 implementation (Tiny AutoEncoder for FLUX.2).

Adapted from https://github.com/madebyollin/taesd (MIT).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def conv(n_in: int, n_out: int, **kwargs) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x) + self.skip(x))


class TAEF2Decoder(nn.Module):
    """Tiny AutoEncoder decoder for FLUX.2 (32 latent channels)."""

    latent_channels: int = 32
    image_channels: int = 3

    def __init__(self, use_midblock_gn: bool = True):
        super().__init__()
        # FLUX.2 architecture uses group norm in midblock for proper distillation
        self.use_midblock_gn = use_midblock_gn

        layers = [
            Clamp(),
            conv(self.latent_channels, 64),
            nn.ReLU(inplace=True),
            Block(64, 64),
            Block(64, 64),
            Block(64, 64),
        ]

        # Add group norm if using midblock_gn (required for flux_2)
        if use_midblock_gn:
            layers.append(nn.GroupNorm(1, 64))

        layers.extend(
            [
                nn.Upsample(scale_factor=2),
                conv(64, 64, bias=False),
                Block(64, 64),
                Block(64, 64),
                Block(64, 64),
                nn.Upsample(scale_factor=2),
                conv(64, 64, bias=False),
                Block(64, 64),
                Block(64, 64),
                Block(64, 64),
                nn.Upsample(scale_factor=2),
                conv(64, 64, bias=False),
                nn.ReLU(inplace=True),
                conv(64, self.image_channels),
            ]
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents).clamp(0, 1)


class TAEF2(nn.Module):
    """Tiny AutoEncoder for FLUX.2.

    This is a lightweight autoencoder for fast preview generation during
    FLUX.2 denoising. It operates on the 32-channel latent space.
    """

    latent_channels: int = 32

    def __init__(self, decoder_path: Optional[str] = None):
        super().__init__()
        self.decoder = TAEF2Decoder(use_midblock_gn=True)

        if decoder_path is not None:
            self._load_checkpoint(decoder_path)

    def _load_checkpoint(self, path: str) -> None:
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(path)
        else:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)

        # The checkpoint may have encoder keys we don't need
        decoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("decoder."):
                decoder_state[key] = value

        self.load_state_dict(decoder_state, strict=False)

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images.

        Args:
            latents: (B, 32, H, W) latent tensor

        Returns:
            (B, 3, H*8, W*8) RGB image tensor in [0, 1] range
        """
        return self.decoder(latents)
