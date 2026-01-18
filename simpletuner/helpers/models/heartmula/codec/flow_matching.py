# Copyright 2025 SimpleTuner contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from vector_quantize_pytorch import ResidualVQ

from .transformer import LlamaTransformer


class FlowMatching(nn.Module):
    def __init__(
        self,
        dim: int = 512,
        codebook_size: int = 8192,
        decay: float = 0.9,
        commitment_weight: float = 1.0,
        threshold_ema_dead_code: int = 2,
        use_cosine_sim: bool = False,
        codebook_dim: int = 32,
        num_quantizers: int = 8,
        attention_head_dim: int = 64,
        in_channels: int = 1024,
        norm_type: str = "ada_norm_single",
        num_attention_heads: int = 24,
        num_layers: int = 24,
        num_layers_2: int = 6,
        out_channels: int = 256,
    ):
        super().__init__()
        self.vq_embed = ResidualVQ(
            dim=dim,
            codebook_size=codebook_size,
            decay=decay,
            commitment_weight=commitment_weight,
            threshold_ema_dead_code=threshold_ema_dead_code,
            use_cosine_sim=use_cosine_sim,
            codebook_dim=codebook_dim,
            num_quantizers=num_quantizers,
        )
        self.cond_feature_emb = nn.Linear(dim, dim)
        self.zero_cond_embedding1 = nn.Parameter(torch.randn(dim))
        self.estimator = LlamaTransformer(
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            norm_type=norm_type,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers,
            num_layers_2=num_layers_2,
            out_channels=out_channels,
        )
        self.latent_dim = out_channels

    def _quantize_condition(self, codes: torch.Tensor) -> torch.Tensor:
        quantized = self.vq_embed.get_output_from_indices(codes.transpose(1, 2))
        quantized = self.cond_feature_emb(quantized)
        quantized = F.interpolate(quantized.permute(0, 2, 1), scale_factor=2, mode="nearest").permute(0, 2, 1)
        return quantized

    def _prepare_masks(
        self,
        batch: int,
        total_frames: int,
        latent_length: int,
        incontext_length: int,
        scenario: str,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.zeros(batch, total_frames, dtype=torch.int64, device=device)
        mask[:, :latent_length] = 2
        if scenario == "other_seg" and incontext_length > 0:
            mask[:, :incontext_length] = 1
        return mask

    @torch.no_grad()
    def inference_codes(
        self,
        codes,
        true_latents,
        latent_length,
        incontext_length,
        guidance_scale=2.0,
        num_steps=20,
        disable_progress=True,
        scenario="start_seg",
    ):
        device = true_latents.device
        dtype = true_latents.dtype
        codes_bestrq_emb = codes[0]
        batch = codes_bestrq_emb.shape[0]
        self.vq_embed.eval()

        conditioning = self._quantize_condition(codes_bestrq_emb)
        total_frames = conditioning.shape[1]

        latents = torch.randn(batch, total_frames, self.latent_dim, device=device, dtype=dtype)
        mask = self._prepare_masks(batch, total_frames, latent_length, incontext_length, scenario, device)
        mask_float = (mask > 0.5).unsqueeze(-1)
        conditioning = mask_float * conditioning + (1 - mask_float) * self.zero_cond_embedding1.view(1, 1, -1)

        incontext_mask = ((mask > 0.5) & (mask < 1.5)).unsqueeze(-1)
        incontext_latents = true_latents * incontext_mask.float()
        context_len = int(incontext_mask.sum(-1)[0].item()) if incontext_length else 0

        t_span = torch.linspace(0, 1, num_steps + 1, device=device)
        latents = self.solve_euler(
            latents,
            incontext_latents,
            context_len,
            t_span,
            conditioning,
            guidance_scale,
            disable_progress=disable_progress,
        )
        if context_len:
            latents[:, :context_len, :] = incontext_latents[:, :context_len, :]
        return latents

    def solve_euler(
        self,
        latents: torch.Tensor,
        incontext_latents: torch.Tensor,
        incontext_length: int,
        t_span: torch.Tensor,
        conditioning: torch.Tensor,
        guidance_scale: float,
        disable_progress: bool = False,
    ) -> torch.Tensor:
        t = t_span[0]
        dt = t_span[1] - t_span[0]
        noise = latents.clone()

        for step in tqdm(range(1, len(t_span)), disable=disable_progress):
            if incontext_length > 0:
                latents[:, :incontext_length, :] = (1 - (1 - 1e-6) * t) * noise[
                    :, :incontext_length, :
                ] + t * incontext_latents[:, :incontext_length, :]

            if guidance_scale > 1.0:
                stacked = torch.cat(
                    [
                        torch.cat([latents, latents], dim=0),
                        torch.cat([incontext_latents, incontext_latents], dim=0),
                        torch.cat([torch.zeros_like(conditioning), conditioning], dim=0),
                    ],
                    dim=2,
                )
                velocity = self.estimator(stacked, timestep=t.unsqueeze(-1).repeat(2))
                uncond, cond = velocity.chunk(2, dim=0)
                velocity = uncond + guidance_scale * (cond - uncond)
            else:
                velocity = self.estimator(
                    torch.cat([latents, incontext_latents, conditioning], dim=2), timestep=t.unsqueeze(-1)
                )

            latents = latents + dt * velocity
            t = t + dt
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return latents
