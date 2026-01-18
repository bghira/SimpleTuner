# Copyright 2025 SimpleTuner contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import math
from typing import Optional

import torch
from transformers.modeling_utils import PreTrainedModel

from .configuration_heartcodec import HeartCodecConfig
from .flow_matching import FlowMatching
from .sq_codec import ScalarModel

CODE_FRAMES_PER_SEC = 12.5
LATENT_FRAMES_PER_SEC = 25.0
MIN_DURATION_SEC = 0.08
HOP_RATIO_NUM = 80
HOP_RATIO_DEN = 93


class HeartCodec(PreTrainedModel):
    config_class = HeartCodecConfig

    def __init__(self, config: HeartCodecConfig):
        super().__init__(config)
        self.config = config
        self.flow_matching = FlowMatching(
            dim=config.dim,
            codebook_size=config.codebook_size,
            decay=config.decay,
            commitment_weight=config.commitment_weight,
            threshold_ema_dead_code=config.threshold_ema_dead_code,
            use_cosine_sim=config.use_cosine_sim,
            codebook_dim=config.codebook_dim,
            num_quantizers=config.num_quantizers,
            attention_head_dim=config.attention_head_dim,
            in_channels=config.in_channels,
            norm_type=config.norm_type,
            num_attention_heads=config.num_attention_heads,
            num_layers=config.num_layers,
            num_layers_2=config.num_layers_2,
            out_channels=config.out_channels,
        )
        self.scalar_model = ScalarModel(
            num_bands=config.num_bands,
            sample_rate=config.sample_rate,
            causal=config.causal,
            num_samples=config.num_samples,
            downsample_factors=config.downsample_factors,
            downsample_kernel_sizes=config.downsample_kernel_sizes,
            upsample_factors=config.upsample_factors,
            upsample_kernel_sizes=config.upsample_kernel_sizes,
            latent_hidden_dim=config.latent_hidden_dim,
            default_kernel_size=config.default_kernel_size,
            delay_kernel_size=config.delay_kernel_size,
            init_channel=config.init_channel,
            res_kernel_size=config.res_kernel_size,
        )
        self.sample_rate = config.sample_rate
        self.post_init()

    def _resolve_device(self, device: Optional[torch.device] = None) -> torch.device:
        if device is not None:
            return torch.device(device)
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @staticmethod
    def _normalize_codes(codes: torch.Tensor, num_codebooks: int) -> torch.Tensor:
        if codes.ndim != 2:
            raise ValueError(
                f"HeartCodec expects codes with shape [codebooks, frames] or [frames, codebooks], got {codes.shape}."
            )
        if codes.shape[0] == num_codebooks and codes.shape[1] == num_codebooks:
            raise ValueError(
                f"HeartCodec codes shape {codes.shape} is ambiguous because both dimensions match num_codebooks "
                f"({num_codebooks}). Provide tokens with distinct frame and codebook dimensions."
            )
        if codes.shape[0] == num_codebooks:
            return codes
        if codes.shape[1] == num_codebooks:
            return codes.transpose(0, 1)
        raise ValueError(f"HeartCodec codes do not match num_codebooks={num_codebooks}: {tuple(codes.shape)}")

    def _prepare_segment_codes(
        self,
        codes: torch.Tensor,
        min_frames: int,
        hop_frames: int,
        overlap_frames: int,
    ) -> torch.Tensor:
        if codes.shape[-1] < min_frames:
            repeated = codes
            while repeated.shape[-1] < min_frames:
                repeated = torch.cat([repeated, repeated], dim=-1)
            codes = repeated[..., :min_frames]
        if hop_frames <= 0:
            hop_frames = min_frames
            overlap_frames = 0
        if (codes.shape[-1] - overlap_frames) % hop_frames != 0:
            target = math.ceil((codes.shape[-1] - overlap_frames) / float(hop_frames)) * hop_frames + overlap_frames
            while codes.shape[-1] < target:
                codes = torch.cat([codes, codes], dim=-1)
            codes = codes[..., :target]
        return codes

    @torch.inference_mode()
    def detokenize(
        self,
        codes: torch.Tensor,
        duration: Optional[float] = 29.76,
        num_steps: int = 10,
        disable_progress: bool = False,
        guidance_scale: float = 1.25,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        device = self._resolve_device(device)
        dtype = next(self.parameters()).dtype

        codes = self._normalize_codes(codes, self.config.num_quantizers)
        codes = codes.unsqueeze(0).to(device=device, dtype=torch.long)

        original_frames = codes.shape[-1]
        if duration is None:
            duration = max(original_frames / CODE_FRAMES_PER_SEC, MIN_DURATION_SEC)

        min_code_frames = max(int(duration * CODE_FRAMES_PER_SEC), 1)
        latent_length = max(int(duration * LATENT_FRAMES_PER_SEC), 1)
        hop_frames = (min_code_frames // HOP_RATIO_DEN) * HOP_RATIO_NUM
        overlap_frames = min_code_frames - hop_frames

        codes = self._prepare_segment_codes(codes, min_code_frames, hop_frames, overlap_frames)
        frames = codes.shape[-1]

        latent_dim = self.flow_matching.latent_dim
        latents_seed = torch.randn(codes.shape[0], latent_length, latent_dim, device=device, dtype=dtype)
        segments = []
        for start in range(0, frames - hop_frames + 1, hop_frames):
            window = codes[:, :, start : start + min_code_frames]
            if not segments or overlap_frames == 0:
                in_context = 0
                scenario = "start_seg"
                seed = latents_seed
            else:
                prev_latents = segments[-1][:, -overlap_frames:, :]
                in_context = prev_latents.shape[1]
                pad = torch.randn(
                    prev_latents.shape[0],
                    latent_length - in_context,
                    latent_dim,
                    device=device,
                    dtype=dtype,
                )
                seed = torch.cat([prev_latents, pad], dim=1)
                scenario = "other_seg"

            latents = self.flow_matching.inference_codes(
                [window],
                seed,
                latent_length,
                in_context,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                disable_progress=disable_progress,
                scenario=scenario,
            )
            segments.append(latents)

        segments = [segment.float() for segment in segments]

        target_len = int(original_frames / CODE_FRAMES_PER_SEC * self.sample_rate)
        segment_samples = max(int(duration * self.sample_rate), 1)
        hop_samples = (segment_samples // HOP_RATIO_DEN) * HOP_RATIO_NUM
        overlap_samples = segment_samples - hop_samples

        output = None
        for latents in segments:
            batch, time_steps, feat = latents.shape
            if feat % 2 != 0:
                raise ValueError(f"HeartCodec latent feature dim must be even, got {feat}.")
            reshaped = latents.reshape(batch, time_steps, 2, feat // 2).permute(0, 2, 1, 3)
            reshaped = reshaped.reshape(batch * 2, time_steps, feat // 2)
            decoded = self.scalar_model.decode(reshaped.transpose(1, 2))
            decoded = decoded.squeeze(1)
            if decoded.dim() == 1:
                decoded = decoded.unsqueeze(0)
            decoded = decoded[:, :segment_samples].detach().cpu()

            if output is None:
                output = decoded
            else:
                if overlap_samples <= 0:
                    output = torch.cat([output, decoded], dim=-1)
                else:
                    fade = torch.linspace(0, 1, overlap_samples, device=output.device)
                    fade = fade.unsqueeze(0)
                    output[:, -overlap_samples:] = (
                        output[:, -overlap_samples:] * (1 - fade) + decoded[:, :overlap_samples] * fade
                    )
                    output = torch.cat([output, decoded[:, overlap_samples:]], dim=-1)

        if output is None:
            raise ValueError("HeartCodec failed to decode audio from tokens.")
        output = output[:, :target_len]
        return output
