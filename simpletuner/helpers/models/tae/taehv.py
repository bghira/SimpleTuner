"""TAEHV implementation (Tiny AutoEncoder for Hunyuan Video).

Adapted from https://github.com/madebyollin/taehv (Apache-2.0).
"""

from __future__ import annotations

from collections import namedtuple
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

DecoderResult = namedtuple("DecoderResult", ("frame", "memory"))
TWorkItem = namedtuple("TWorkItem", ("input_tensor", "block_index"))


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class MemBlock(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(
            conv(n_in * 2, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
            nn.ReLU(inplace=True),
            conv(n_out, n_out),
        )
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, past):
        return self.act(self.conv(torch.cat([x, past], 1)) + self.skip(x))


class TPool(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f * stride, n_f, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        return self.conv(x.reshape(-1, self.stride * C, H, W))


class TGrow(nn.Module):
    def __init__(self, n_f, stride):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(n_f, n_f * stride, 1, bias=False)

    def forward(self, x):
        _NT, C, H, W = x.shape
        x = self.conv(x)
        return x.reshape(-1, C, H, W)


def apply_model_with_memblocks(
    model: nn.Sequential,
    x: torch.Tensor,
    parallel: bool,
    show_progress_bar: bool,
) -> torch.Tensor:
    """
    Apply a sequential model with memblocks to the given NTCHW input.
    """

    assert x.ndim == 5, f"TAEHV operates on NTCHW tensors, but got {x.ndim}-dim tensor"
    N, T, C, H, W = x.shape
    if parallel:
        x = x.reshape(N * T, C, H, W)
        for block in tqdm(model, disable=not show_progress_bar):
            if isinstance(block, MemBlock):
                NT, C, H, W = x.shape
                T = NT // N
                reshaped = x.reshape(N, T, C, H, W)
                mem = F.pad(reshaped, (0, 0, 0, 0, 0, 0, 1, 0), value=0)[:, :T].reshape(x.shape)
                x = block(x, mem)
            else:
                x = block(x)
        NT, C, H, W = x.shape
        T = NT // N
        x = x.view(N, T, C, H, W)
    else:
        out: List[torch.Tensor] = []
        work_queue: List[TWorkItem] = [TWorkItem(xt, 0) for xt in x.reshape(N, T * C, H, W).chunk(T, dim=1)]
        progress_bar = tqdm(range(T), disable=not show_progress_bar)
        mem: List[Optional[torch.Tensor]] = [None] * len(model)
        while work_queue:
            xt, idx = work_queue.pop(0)
            if idx == 0:
                progress_bar.update(1)
            if idx == len(model):
                out.append(xt)
                continue
            block = model[idx]
            if isinstance(block, MemBlock):
                if mem[idx] is None:
                    xt_new = block(xt, xt * 0)
                    mem[idx] = xt
                else:
                    xt_new = block(xt, mem[idx])
                    mem[idx].copy_(xt)
                work_queue.insert(0, TWorkItem(xt_new, idx + 1))
            elif isinstance(block, TPool):
                if mem[idx] is None:
                    mem[idx] = [xt]
                else:
                    mem[idx].append(xt)
                if len(mem[idx]) == block.stride:
                    combined = torch.cat(mem[idx], 1)
                    xt = block(combined.view(-1, block.stride * combined.shape[1] // block.stride, H, W))
                    mem[idx] = []
                    work_queue.insert(0, TWorkItem(xt, idx + 1))
            elif isinstance(block, TGrow):
                xt = block(xt)
                NT, C, H, W = xt.shape
                for xt_next in reversed(xt.view(N, block.stride * C, H, W).chunk(block.stride, 1)):
                    work_queue.insert(0, TWorkItem(xt_next, idx + 1))
            else:
                xt = block(xt)
                work_queue.insert(0, TWorkItem(xt, idx + 1))
        progress_bar.close()
        x = torch.stack(out, 1)
    return x


class TAEHV(nn.Module):
    """Tiny AutoEncoder for Hunyuan Video."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = "taehv.pth",
        decoder_time_upscale: Iterable[bool] = (True, True),
        decoder_space_upscale: Iterable[bool] = (True, True, True),
        patch_size: int = 1,
        latent_channels: int = 16,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.image_channels = 3
        self.is_cogvideox = checkpoint_path is not None and "taecvx" in checkpoint_path
        if checkpoint_path is not None and "taew2_2" in checkpoint_path:
            # Wan 2.2 5B checkpoints always require patch_size=2 and 48-channel latents.
            # Override caller-provided values to match the architecture.
            self.patch_size, self.latent_channels = 2, 48

        self.encoder = nn.Sequential(
            conv(self.image_channels * self.patch_size**2, 64),
            nn.ReLU(inplace=True),
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 2),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            TPool(64, 1),
            conv(64, 64, stride=2, bias=False),
            MemBlock(64, 64),
            MemBlock(64, 64),
            MemBlock(64, 64),
            conv(64, self.latent_channels),
        )

        n_f = [256, 128, 64, 64]
        decoder_layers: List[nn.Module] = [
            Clamp(),
            conv(self.latent_channels, n_f[0]),
            nn.ReLU(inplace=True),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            MemBlock(n_f[0], n_f[0]),
            nn.Upsample(scale_factor=2 if decoder_space_upscale[0] else 1),
            TGrow(n_f[0], 1),
            conv(n_f[0], n_f[1], bias=False),
        ]
        decoder_layers.extend(
            [
                MemBlock(n_f[1], n_f[1]),
                MemBlock(n_f[1], n_f[1]),
                MemBlock(n_f[1], n_f[1]),
                nn.Upsample(scale_factor=2 if decoder_space_upscale[1] else 1),
                TGrow(n_f[1], 2 if decoder_time_upscale[0] else 1),
                conv(n_f[1], n_f[2], bias=False),
                MemBlock(n_f[2], n_f[2]),
                MemBlock(n_f[2], n_f[2]),
                MemBlock(n_f[2], n_f[2]),
                nn.Upsample(scale_factor=2 if decoder_space_upscale[2] else 1),
                TGrow(n_f[2], 2 if decoder_time_upscale[1] else 1),
                conv(n_f[2], n_f[3], bias=False),
                nn.ReLU(inplace=True),
                conv(n_f[3], self.image_channels * self.patch_size**2),
            ]
        )

        self.decoder = nn.Sequential(*decoder_layers)
        self.frames_to_trim = 2 ** sum(bool(x) for x in decoder_time_upscale) - 1

        if checkpoint_path is not None:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            patched = self.patch_tgrow_layers(state_dict)
            self.load_state_dict(patched, strict=False)

    def patch_tgrow_layers(self, state_dict):
        new_sd = self.state_dict()
        for idx, layer in enumerate(self.decoder):
            if isinstance(layer, TGrow):
                key = f"decoder.{idx}.conv.weight"
                if key in state_dict and key in new_sd and state_dict[key].shape[0] > new_sd[key].shape[0]:
                    state_dict[key] = state_dict[key][-new_sd[key].shape[0] :]
        return state_dict

    @torch.no_grad()
    def encode_video(self, x: torch.Tensor, parallel: bool = True, show_progress_bar: bool = False) -> torch.Tensor:
        if self.patch_size > 1:
            x = F.pixel_unshuffle(x, self.patch_size)
        if x.shape[1] % 4 != 0:
            n_pad = 4 - x.shape[1] % 4
            padding = x[:, -1:].repeat_interleave(n_pad, dim=1)
            x = torch.cat([x, padding], 1)
        return apply_model_with_memblocks(self.encoder, x, parallel, show_progress_bar)

    @torch.no_grad()
    def decode_video(self, x: torch.Tensor, parallel: bool = True, show_progress_bar: bool = False) -> torch.Tensor:
        skip_trim = self.is_cogvideox and x.shape[1] % 2 == 0
        x = apply_model_with_memblocks(self.decoder, x, parallel, show_progress_bar)
        x = x.clamp_(0, 1)
        if self.patch_size > 1:
            x = F.pixel_shuffle(x, self.patch_size)
        if skip_trim:
            return x
        return x[:, self.frames_to_trim :]
