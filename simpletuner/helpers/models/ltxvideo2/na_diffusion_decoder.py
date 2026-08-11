# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DecoderOutput
from diffusers.models.embeddings import get_timestep_embedding
from einops import rearrange
from torch import nn

from simpletuner.helpers.models.ltxvideo2.autoencoder import AutoencoderKLLTX2Video
from simpletuner.helpers.models.ltxvideo2.na_kernels import na3d, rms_rope_

MLP_TOKEN_CHUNK = 65536

LTX_25_DIFFUSION_DECODER_CONFIG = {
    "in_channels": 128,
    "out_channels": 3,
    "patch_size": 4,
    "head_dim": 64,
    "stage_channels": (2048, 1024, 512, 512, 256),
    "stage_depths": (4, 6, 4, 2, 8),
    "stage_kernels": ((3, 7, 7), (3, 7, 7), (3, 5, 5), (3, 5, 5), (11, 11, 11)),
    "upsamples": (((1, 2, 2), 2), ((2, 1, 1), 2), ((2, 2, 2), 1), ((2, 2, 2), 2)),
    "stage5_kernel": (11, 11, 11),
    "timestep_scale_multiplier": 1000.0,
    "default_num_inference_steps": 1,
}


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), weight=weight.to(x.dtype), eps=eps)
    x_f = x.float()
    x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f * weight.float()).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.eps)


def patchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    return rearrange(
        x,
        "b c (f p) (h q) (w r) -> b (c p r q) f h w",
        p=patch_size_t,
        q=patch_size_hw,
        r=patch_size_hw,
    )


def unpatchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    return rearrange(
        x,
        "b (c p r q) f h w -> b c (f p) (h q) (w r)",
        p=patch_size_t,
        q=patch_size_hw,
        r=patch_size_hw,
    )


def default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2 != 0:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    return (d_t, d_hw, d_hw)


def rope_inv_freqs(dim: int, base: float = 10000.0, device: torch.device | None = None) -> torch.Tensor:
    exponents = torch.arange(0, dim, 2, dtype=torch.float64, device=device) / dim
    return (1.0 / torch.pow(torch.tensor(float(base), dtype=torch.float64, device=device), exponents)).to(torch.float32)


def _rope_tables(
    lengths: tuple[int, int, int],
    inv_freqs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    tables = []
    for length, inv in zip(lengths, inv_freqs, strict=True):
        pos = torch.arange(length, dtype=torch.float32, device=device)
        ang = pos[:, None] * inv[None, :]
        tables.append((ang.cos(), ang.sin()))
    return tables


def _rope_matrices_slice(
    tables: list[tuple[torch.Tensor, torch.Tensor]],
    t0: int,
    t1: int,
    h: int,
    w: int,
) -> torch.Tensor:
    parts = []
    for (c, s), sl in zip(tables, (slice(t0, t1), slice(None), slice(None)), strict=True):
        c, s = c[sl], s[sl]
        parts.append(torch.stack([c, -s, s, c], dim=-1).reshape(c.shape[0], 1, 1, c.shape[1], 2, 2))
    ts = t1 - t0
    freqs = torch.cat(
        [
            parts[0].expand(ts, h, w, -1, 2, 2),
            parts[1].transpose(0, 1).expand(ts, h, w, -1, 2, 2),
            parts[2].movedim(0, 2).expand(ts, h, w, -1, 2, 2),
        ],
        dim=3,
    )
    return freqs.reshape(1, ts * h * w, 1, -1, 2, 2)


class NeighborhoodAttention3D(nn.Module):
    def __init__(self, dim: int, kernel_size: tuple[int, int, int], head_dim: int = 64, rope_base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5
        self.rope_split = default_rope_dim_split(head_dim)
        self.rope_base = rope_base

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)

    def forward(self, x: torch.Tensor, pre=None, add_to: torch.Tensor | None = None) -> torch.Tensor:
        batch, t, h, w, _ = x.shape
        inv_freqs = tuple(rope_inv_freqs(d, self.rope_base, device=x.device) for d in self.rope_split)
        tables = _rope_tables((t, h, w), inv_freqs, x.device)
        shape = (batch, t, h, w, self.num_heads, self.head_dim)
        q = torch.empty(shape, dtype=x.dtype, device=x.device)
        k = torch.empty(shape, dtype=x.dtype, device=x.device)
        v = torch.empty(shape, dtype=x.dtype, device=x.device)
        q_weight = (self.q_norm.weight.detach() * self.scale).to(x.dtype)
        k_weight = self.k_norm.weight.detach().to(x.dtype)
        chunk = max(1, 2**25 // max(h * w * self.dim, 1))
        for t0 in range(0, t, chunk):
            t1 = min(t0 + chunk, t)
            sl = x[:, t0:t1] if pre is None else pre(x[:, t0:t1])
            qc, kc, vc = self.qkv(sl).chunk(3, dim=-1)
            cshape = (batch, t1 - t0, h, w, self.num_heads, self.head_dim)
            q[:, t0:t1] = qc.reshape(cshape)
            k[:, t0:t1] = kc.reshape(cshape)
            v[:, t0:t1] = vc.reshape(cshape)
            freqs = _rope_matrices_slice(tables, t0, t1, h, w)
            nt = (t1 - t0) * h * w
            for b in range(batch):
                rms_rope_(
                    q[b, t0:t1].view(1, nt, self.num_heads, self.head_dim),
                    k[b, t0:t1].view(1, nt, self.num_heads, self.head_dim),
                    freqs,
                    q_weight,
                    k_weight,
                )
        out = na3d(q, k, v, list(self.kernel_size), None, 1.0)
        del q, k, v
        out = out.reshape(batch, t, h, w, self.dim)
        res = add_to if add_to is not None else torch.empty_like(out)
        for t0 in range(0, t, chunk):
            t1 = min(t0 + chunk, t)
            if add_to is not None:
                res[:, t0:t1] += self.proj(out[:, t0:t1])
            else:
                res[:, t0:t1] = self.proj(out[:, t0:t1])
        return res


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, pre=None, add_to: torch.Tensor | None = None) -> torch.Tensor:
        _, t, h, w, _ = x.shape
        chunk = max(1, MLP_TOKEN_CHUNK // max(h * w, 1))
        out = add_to if add_to is not None else torch.empty_like(x)
        for t0 in range(0, t, chunk):
            t1 = min(t0 + chunk, t)
            sl = x[:, t0:t1] if pre is None else pre(x[:, t0:t1])
            y = self.w_down(F.silu(self.w_gate(sl)) * self.w_up(sl))
            if add_to is not None:
                out[:, t0:t1] += y
            else:
                out[:, t0:t1] = y
        return out


class NABlock(nn.Module):
    def __init__(self, dim: int, kernel_size: tuple[int, int, int], head_dim: int = 64, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * mlp_ratio) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x, pre=self.norm1, add_to=x)
        return self.mlp(x, pre=self.norm2, add_to=x)


def modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale) + shift


class AdaLNZero(nn.Module):
    NUM_CHUNKS = 7

    def __init__(self, dim: int, t_emb_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(t_emb_dim, self.NUM_CHUNKS * dim, bias=True)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        h = self.proj(F.silu(t_emb))
        return tuple(c[:, None, None, None, :] for c in h.chunk(self.NUM_CHUNKS, dim=-1))


class DiffusionNABlock(nn.Module):
    def __init__(
        self,
        dim: int,
        kernel_size: tuple[int, int, int],
        context_channels: int,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
    ) -> None:
        super().__init__()
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(AdaLNZero.NUM_CHUNKS, dim))
        self.norm1 = RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim=head_dim)
        self.norm2 = RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * mlp_ratio) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)

    def forward(
        self,
        x: torch.Tensor,
        latent_context: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(AdaLNZero.NUM_CHUNKS)
        ]
        chunk = max(1, MLP_TOKEN_CHUNK // max(x.shape[2] * x.shape[3], 1))
        for t0 in range(0, x.shape[1], chunk):
            x[:, t0 : t0 + chunk] += self.context_proj(latent_context[:, t0 : t0 + chunk])
        x = self.attn(x, pre=lambda s: modulate(self.norm1(s), scale_msa, shift_msa), add_to=x)
        return self.mlp(x, pre=lambda s: modulate(self.norm2(s), scale_mlp, shift_mlp), add_to=x)


class LinearPixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels: int, stride: tuple[int, int, int], out_channels_reduction_factor: int = 1) -> None:
        super().__init__()
        self.stride = tuple(stride)
        proj_out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.out_channels = proj_out_channels // math.prod(stride)
        self.proj = nn.Linear(in_channels, proj_out_channels, bias=True)

    def forward(self, x: torch.Tensor, drop_leading_frame: bool = True) -> torch.Tensor:
        batch, t, h, w, _ = x.shape
        p1, p2, p3 = self.stride
        out = torch.empty((batch, t * p1, h * p2, w * p3, self.out_channels), dtype=x.dtype, device=x.device)
        chunk = max(1, MLP_TOKEN_CHUNK // max(h * w, 1))
        for t0 in range(0, t, chunk):
            t1 = min(t0 + chunk, t)
            out[:, t0 * p1 : t1 * p1] = rearrange(
                self.proj(x[:, t0:t1]),
                "b t h w (c p1 p2 p3) -> b (t p1) (h p2) (w p3) c",
                p1=p1,
                p2=p2,
                p3=p3,
            )
        if p1 == 2 and drop_leading_frame:
            out = out[:, 1:]
        return out


class TimestepEmbedder(nn.Module):
    def __init__(self, t_emb_dim: int = 384, freq_dim: int = 256) -> None:
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, t_emb_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim, bias=True),
        )

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        emb = get_timestep_embedding(
            timestep.flatten(),
            self.freq_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=1,
        )
        return self.mlp(emb.to(dtype))


class NADiffusionDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        head_dim: int = 64,
        stage_channels: tuple[int, ...] = (2048, 1024, 512, 512, 256),
        stage_depths: tuple[int, ...] = (4, 6, 4, 2, 8),
        stage_kernels: tuple[tuple[int, int, int], ...] = (
            (3, 7, 7),
            (3, 7, 7),
            (3, 5, 5),
            (3, 5, 5),
            (11, 11, 11),
        ),
        upsamples: tuple[tuple[tuple[int, int, int], int], ...] = (
            ((1, 2, 2), 2),
            ((2, 1, 1), 2),
            ((2, 2, 2), 1),
            ((2, 2, 2), 2),
        ),
        stage5_kernel: tuple[int, int, int] = (11, 11, 11),
        t_emb_dim: int = 384,
        default_num_inference_steps: int = 1,
        timestep_scale_multiplier: float = 1000.0,
        model_output_type: str = "x0",
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.model_output_type = model_output_type
        self.register_buffer(
            "default_inference_timesteps",
            torch.linspace(1.0, 1.0 / default_num_inference_steps, default_num_inference_steps),
            persistent=False,
        )
        self.temporal_upscale = math.prod(s[0] for s, _ in upsamples)
        self.spatial_upscale = math.prod(s[1] for s, _ in upsamples) * patch_size
        self.trailing_pad_latent_frames = (stage_kernels[0][0] // 2) * 2

        self.conv_in = nn.Linear(in_channels, stage_channels[0], bias=True)

        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage_i in range(len(stage_channels) - 1):
            c = stage_channels[stage_i]
            self.det_stages.append(
                nn.ModuleList([NABlock(c, stage_kernels[stage_i], head_dim=head_dim) for _ in range(stage_depths[stage_i])])
            )
            stride, reduction = upsamples[stage_i]
            self.upsamples.append(LinearPixelShuffleUpsample(c, stride, out_channels_reduction_factor=reduction))

        self.t_embedder = TimestepEmbedder(t_emb_dim=t_emb_dim)

        c5 = stage_channels[-1]
        self.context_channels = c5
        noised_pixel_channels = out_channels * (patch_size**2)
        self.conv_in_x_t = nn.Linear(noised_pixel_channels, c5, bias=True)
        self.shared_adaln = AdaLNZero(c5, t_emb_dim)
        self.diff_blocks = nn.ModuleList(
            [DiffusionNABlock(c5, stage5_kernel, context_channels=c5, head_dim=head_dim) for _ in range(stage_depths[-1])]
        )
        self.norm_out = RMSNorm(c5, eps=1e-6)
        self.conv_out = nn.Linear(c5, noised_pixel_channels, bias=True)

    def forward_pre_diffusion(
        self,
        z: torch.Tensor,
        drop_leading_frame: bool = True,
        pad_trailing: bool = True,
    ) -> torch.Tensor:
        n = self.trailing_pad_latent_frames if pad_trailing else 0
        if n > 0:
            z = torch.cat([z, z[:, :, -1:].expand(-1, -1, n, -1, -1)], dim=2)
        x = z.permute(0, 2, 3, 4, 1)
        x = self.conv_in(x)
        for stage_i, blocks in enumerate(self.det_stages):
            for block in blocks:
                x = block(x)
            x = self.upsamples[stage_i](x, drop_leading_frame=drop_leading_frame)
        if n > 0:
            x = x[:, : -(n * self.temporal_upscale)]
        return x

    def forward_diff_step(self, context: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = patchify(x_t, patch_size_hw=self.patch_size, patch_size_t=1)
        x = self.conv_in_x_t(x.permute(0, 2, 3, 4, 1))
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, dtype=x.dtype)
        modulation = self.shared_adaln(t_emb)
        for block in self.diff_blocks:
            x = block(x, context, modulation)
        x = self.norm_out(x)
        x = self.conv_out(x)
        x = x.permute(0, 4, 1, 2, 3)
        return unpatchify(x, patch_size_hw=self.patch_size, patch_size_t=1)

    def forward(
        self,
        z: torch.Tensor,
        generator: torch.Generator | None = None,
        drop_leading_frame: bool = True,
        pad_trailing: bool = True,
    ) -> torch.Tensor:
        context = self.forward_pre_diffusion(z, drop_leading_frame=drop_leading_frame, pad_trailing=pad_trailing)
        batch, t5, h5, w5, _ = context.shape
        pixel_shape = (batch, self.out_channels, t5, h5 * self.patch_size, w5 * self.patch_size)
        x_t = torch.randn(pixel_shape, dtype=z.dtype, device=z.device, generator=generator)

        timesteps = self.default_inference_timesteps.to(z.device)
        num_steps = timesteps.shape[0]
        for i in range(num_steps):
            t_now = timesteps[i].expand(batch)
            model_out = self.forward_diff_step(context, x_t, t_now)
            if self.model_output_type == "x0":
                x0 = model_out
                if i == num_steps - 1:
                    return x0
                velocity = (x_t.float() - x0.float()) / timesteps[i]
            else:
                velocity = model_out.float()
                if i == num_steps - 1:
                    return (x_t.float() - timesteps[i] * velocity).to(z.dtype)
            t_next = timesteps[i + 1] if i + 1 < num_steps else torch.zeros_like(timesteps[i])
            x_t = (x_t.float() - (timesteps[i] - t_next) * velocity).to(z.dtype)
        return x_t


class AutoencoderKLLTX2VideoDiffusionDecoder(AutoencoderKLLTX2Video):
    def __init__(
        self,
        diffusion_decoder_config: dict[str, Any] | None = None,
        diffusion_decoder_model_output_type: str = "x0",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        decoder_config = {
            **LTX_25_DIFFUSION_DECODER_CONFIG,
            **(diffusion_decoder_config or {}),
            "model_output_type": diffusion_decoder_model_output_type,
        }
        decoder_config["stage_channels"] = tuple(decoder_config["stage_channels"])
        decoder_config["stage_depths"] = tuple(decoder_config["stage_depths"])
        decoder_config["stage_kernels"] = tuple(tuple(kernel) for kernel in decoder_config["stage_kernels"])
        decoder_config["upsamples"] = tuple((tuple(stride), reduction) for stride, reduction in decoder_config["upsamples"])
        decoder_config["stage5_kernel"] = tuple(decoder_config["stage5_kernel"])
        self.decoder = NADiffusionDecoder(**decoder_config)
        self.register_to_config(
            diffusion_decoder_config=decoder_config,
            diffusion_decoder_model_output_type=diffusion_decoder_model_output_type,
        )

    def _decode(
        self,
        z: torch.Tensor,
        temb: torch.Tensor | None = None,
        causal: bool | None = None,
        return_dict: bool = True,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        generator = torch.Generator(device=z.device)
        generator.manual_seed(0)
        dec = self.decoder(z, generator=generator)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
