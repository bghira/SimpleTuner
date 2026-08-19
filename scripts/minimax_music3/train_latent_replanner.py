#!/usr/bin/env python3
# Copyright 2026 SimpleTuner contributors
# Licensed under the Apache License, Version 2.0
"""Flow-matching latent re-planner for MiniMax Music 3 style transfer.

Overfit mode: memorize the transform from one source song (MERT conditioning)
to its ACE style-swap target (DAV latents). Acceptance: generated latents decode
to audio recognizably matching the target, judged against the DAV round-trip
of the target itself (the codec ceiling).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SAMPLE_RATE = 44_100
MERT_SAMPLE_RATE = 24_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MiniMax Music 3 latent re-planner")
    parser.add_argument("--source-audio", type=Path, required=True)
    parser.add_argument("--target-audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--mert-id", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--mert-layer", type=int, default=7)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-seconds", type=float, default=30.0)
    parser.add_argument("--crop-offset-seconds", type=float, default=0.0)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_audio(path: Path, sample_rate: int, mono: bool) -> torch.Tensor:
    import torchaudio

    audio, native_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T.copy())
    if native_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, native_rate, sample_rate)
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def crop(waveform: torch.Tensor, sample_rate: int, offset_seconds: float, duration_seconds: float) -> torch.Tensor:
    start = int(offset_seconds * sample_rate)
    length = int(duration_seconds * sample_rate)
    if start + length > waveform.shape[-1]:
        raise ValueError(f"crop [{offset_seconds}, +{duration_seconds}]s exceeds audio length")
    return waveform[..., start : start + length]


@torch.no_grad()
def extract_mert_features(
    waveform_24k: torch.Tensor,
    mert,
    processor,
    device: torch.device,
    layer: int,
    all_layers: bool = False,
) -> torch.Tensor:
    inputs = processor(waveform_24k.squeeze(0).numpy(), sampling_rate=MERT_SAMPLE_RATE, return_tensors="pt")
    outputs = mert(inputs["input_values"].to(device), output_hidden_states=True)
    if all_layers:
        return torch.stack([states.squeeze(0).float() for states in outputs.hidden_states], dim=0)
    return outputs.hidden_states[layer].squeeze(0).float()


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        inv_freq = base ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(length, device=device, dtype=torch.float32)
        angles = positions[:, None] * self.inv_freq.to(device)[None, :]
        return angles.cos(), angles.sin()


def apply_rope(states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = states.chunk(2, dim=-1)
    cos = cos[None, None]
    sin = sin[None, None]
    return torch.cat((first * cos - second * sin, first * sin + second * cos), dim=-1)


class ReplannerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.attn_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.mlp_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.adaln = nn.Linear(d_model, 6 * d_model)
        nn.init.zeros_(self.adaln.weight)
        nn.init.zeros_(self.adaln.bias)
        self.layer_cond_proj: nn.Module | None = None

    def enable_layer_conditioning(self, cond_dim: int) -> None:
        self.layer_cond_proj = nn.Linear(cond_dim, self.qkv.in_features, bias=False)
        nn.init.zeros_(self.layer_cond_proj.weight)

    def forward(
        self,
        states: torch.Tensor,
        conditioning: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        layer_conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if layer_conditioning is not None:
            states = states + self.layer_cond_proj(layer_conditioning)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaln(conditioning).chunk(6, dim=-1)
        batch, length, _ = states.shape
        normed = self.attn_norm(states) * (1 + scale_a[:, None]) + shift_a[:, None]
        query, key, value = self.qkv(normed).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        query = apply_rope(query, cos, sin)
        key = apply_rope(key, cos, sin)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        states = states + gate_a[:, None] * self.attn_out(attended)
        normed = self.mlp_norm(states) * (1 + scale_m[:, None]) + shift_m[:, None]
        return states + gate_m[:, None] * self.mlp(normed)


class LatentReplanner(nn.Module):
    def __init__(self, latent_dim: int, cond_dim: int, d_model: int, depth: int, heads: int):
        super().__init__()
        self.proj_in = nn.Linear(latent_dim + cond_dim, d_model)
        self.time_dim = d_model
        self.time_embed = nn.Sequential(nn.Linear(256, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.style_proj: nn.Module | None = None
        self.code_embed: nn.Module | None = None
        self.degraded_in_proj: nn.Module | None = None
        self.rope = RotaryEmbedding(d_model // heads)
        self.blocks = nn.ModuleList(ReplannerBlock(d_model, heads) for _ in range(depth))
        self.out_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj_out = nn.Linear(d_model, latent_dim)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    @staticmethod
    def timestep_features(t: torch.Tensor) -> torch.Tensor:
        half = 128
        freqs = torch.exp(-math.log(10_000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        angles = t[:, None].float() * freqs[None, :] * 1_000.0
        return torch.cat((angles.sin(), angles.cos()), dim=-1)

    def enable_mert_masking(self, cond_dim: int, mert_layer_count: int) -> None:
        """Learned per-layer null vectors substituted at masked reference frames."""
        self.mert_null = nn.Parameter(torch.zeros(mert_layer_count, cond_dim))

    def enable_style_conditioning(self, style_dim: int) -> None:
        """CLAP-style pooled conditioning entering beside the flow timestep."""
        self.style_proj = nn.Sequential(
            nn.Linear(style_dim, self.time_dim), nn.SiLU(), nn.Linear(self.time_dim, self.time_dim)
        )
        self.style_null = nn.Parameter(torch.zeros(style_dim))

    def enable_degraded_latent_conditioning(self, latent_dim: int = 128) -> None:
        """SR3-style: degraded latents concatenated per frame via zero-init additive projection."""
        self.degraded_in_proj = nn.Linear(latent_dim, self.proj_in.out_features, bias=False)
        self.degraded_null = nn.Parameter(torch.zeros(latent_dim))
        nn.init.zeros_(self.degraded_in_proj.weight)

    def enable_code_conditioning(self, vocab_size: int, books: int, code_dim: int = 256) -> None:
        """RVQ-code conditioning: summed book embeddings, zero-init additive input projection."""
        self.code_books = books
        self.code_embed = nn.Embedding(vocab_size, code_dim)
        self.code_in_proj = nn.Linear(code_dim, self.proj_in.out_features, bias=False)
        self.code_null = nn.Parameter(torch.zeros(code_dim))
        nn.init.zeros_(self.code_in_proj.weight)

    def embed_codes(self, codes: torch.Tensor, frame_count: int) -> torch.Tensor:
        """codes [B, Tc, books] (already offset per book) -> [B, frame_count, code_dim]."""
        embedded = self.code_embed(codes).sum(dim=-2)
        return F.interpolate(embedded.transpose(1, 2), size=frame_count, mode="linear", align_corners=True).transpose(1, 2)

    def enable_layer_conditioning(self, cond_dim: int, mert_layer_count: int) -> None:
        """Map DiT block i to MERT hidden-state 1+i, repeating the penultimate layer past the match."""
        self.layer_map = [min(1 + index, mert_layer_count - 2) for index in range(len(self.blocks))]
        for block in self.blocks:
            block.enable_layer_conditioning(cond_dim)

    def forward(
        self,
        noisy_latents: torch.Tensor,
        conditioning: torch.Tensor,
        t: torch.Tensor,
        layer_conditioning: torch.Tensor | None = None,
        style: torch.Tensor | None = None,
        code_conditioning: torch.Tensor | None = None,
        degraded_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        states = self.proj_in(torch.cat((noisy_latents, conditioning), dim=-1))
        if self.degraded_in_proj is not None:
            if degraded_latents is None:
                degraded_latents = self.degraded_null[None, None].expand(states.shape[0], states.shape[1], -1)
            states = states + self.degraded_in_proj(degraded_latents)
        if self.code_embed is not None:
            if code_conditioning is None:
                code_conditioning = self.code_null[None, None].expand(states.shape[0], states.shape[1], -1)
            states = states + self.code_in_proj(code_conditioning)
        time_conditioning = self.time_embed(self.timestep_features(t))
        if self.style_proj is not None:
            if style is None:
                style = self.style_null[None].expand(t.shape[0], -1)
            time_conditioning = time_conditioning + self.style_proj(style)
        cos, sin = self.rope(states.shape[1], states.device)
        for index, block in enumerate(self.blocks):
            block_layers = layer_conditioning[:, self.layer_map[index]] if layer_conditioning is not None else None
            states = block(states, time_conditioning, cos, sin, block_layers)
        return self.proj_out(self.out_norm(states))


@torch.no_grad()
def sample(
    model: LatentReplanner,
    conditioning: torch.Tensor,
    steps: int,
    generator: torch.Generator,
    layer_conditioning: torch.Tensor | None = None,
    style: torch.Tensor | None = None,
    code_conditioning: torch.Tensor | None = None,
    initial_latents: torch.Tensor | None = None,
    edit_strength: float = 1.0,
    degraded_latents: torch.Tensor | None = None,
) -> torch.Tensor:
    noise = torch.randn(
        conditioning.shape[0],
        conditioning.shape[1],
        model.proj_out.out_features,
        generator=generator,
        device="cpu",
    ).to(conditioning.device)
    if not 0.0 < edit_strength <= 1.0:
        raise ValueError("edit_strength must be in (0, 1]")
    if initial_latents is None:
        latents = noise
    else:
        latents = (1.0 - edit_strength) * initial_latents + edit_strength * noise
    schedule = torch.linspace(edit_strength, 0.0, steps + 1, device=conditioning.device)
    for index in range(steps):
        t = schedule[index].expand(latents.shape[0])
        velocity = model(latents, conditioning, t, layer_conditioning, style, code_conditioning, degraded_latents)
        latents = latents - (schedule[index] - schedule[index + 1]) * velocity
    return latents


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    from simpletuner.helpers.models.minimaxmusic.vocoder import MiniMaxMusic3DAV

    source_44k = crop(
        load_audio(args.source_audio, SAMPLE_RATE, mono=False), SAMPLE_RATE, args.crop_offset_seconds, args.crop_seconds
    )
    target_44k = crop(
        load_audio(args.target_audio, SAMPLE_RATE, mono=False), SAMPLE_RATE, args.crop_offset_seconds, args.crop_seconds
    )
    source_24k = crop(
        load_audio(args.source_audio, MERT_SAMPLE_RATE, mono=True),
        MERT_SAMPLE_RATE,
        args.crop_offset_seconds,
        args.crop_seconds,
    )

    dav_path = hf_hub_download(args.model_id, "dav.pth", cache_dir=args.cache_dir)
    dav = MiniMaxMusic3DAV.from_original_dav(dav_path).to(device).eval()
    with torch.no_grad():
        target_latents = dav.encode(target_44k.unsqueeze(0).to(device)).squeeze(0).transpose(0, 1).float()

    mert = AutoModel.from_pretrained(args.mert_id, trust_remote_code=True, cache_dir=args.cache_dir).to(device).eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.mert_id, trust_remote_code=True, cache_dir=args.cache_dir)
    mert_features = extract_mert_features(source_24k, mert, processor, device, args.mert_layer)
    del mert
    torch.cuda.empty_cache()

    frame_count = target_latents.shape[0]
    conditioning = (
        F.interpolate(mert_features.transpose(0, 1).unsqueeze(0), size=frame_count, mode="linear", align_corners=True)
        .squeeze(0)
        .transpose(0, 1)
    )

    latent_mean = target_latents.mean(dim=0, keepdim=True)
    latent_std = target_latents.std(dim=0, keepdim=True).clamp_min(1e-4)
    normalized_target = (target_latents - latent_mean) / latent_std
    cond_mean = conditioning.mean(dim=0, keepdim=True)
    cond_std = conditioning.std(dim=0, keepdim=True).clamp_min(1e-4)
    conditioning = ((conditioning - cond_mean) / cond_std).unsqueeze(0)
    target_batch = normalized_target.unsqueeze(0)

    with torch.no_grad():
        roundtrip = dav.decode(target_latents.transpose(0, 1).unsqueeze(0).to(dav.dtype))
    sf.write(args.output_dir / "target_roundtrip.flac", roundtrip.squeeze(0).float().cpu().T.numpy(), SAMPLE_RATE)
    sf.write(args.output_dir / "source_crop.flac", source_44k.T.numpy(), SAMPLE_RATE)
    sf.write(args.output_dir / "target_crop.flac", target_44k.T.numpy(), SAMPLE_RATE)

    model = LatentReplanner(128, conditioning.shape[-1], args.d_model, args.depth, args.heads).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({"parameters": parameter_count, "frames": frame_count, "cond_dim": conditioning.shape[-1]}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        t = torch.sigmoid(torch.randn(1, generator=generator)).to(device)
        noise = torch.randn(target_batch.shape, generator=generator).to(device)
        noisy = (1.0 - t[:, None, None]) * target_batch.to(device) + t[:, None, None] * noise
        velocity_target = noise - target_batch.to(device)
        velocity = model(noisy, conditioning.to(device), t)
        loss = F.mse_loss(velocity, velocity_target)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == 1:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": round(loss.item(), 6),
                        "grad_norm": round(float(grad_norm), 4),
                        "steps_per_second": round(step / (time.perf_counter() - started), 3),
                    }
                ),
                flush=True,
            )
        if step % args.sample_every == 0 or step == args.steps:
            model.eval()
            generated = sample(
                model, conditioning.to(device), args.sample_steps, torch.Generator(device="cpu").manual_seed(args.seed)
            )
            model.train()
            denormalized = generated.squeeze(0) * latent_std.to(device) + latent_mean.to(device)
            cosine = F.cosine_similarity(denormalized.flatten(), target_latents.to(device).flatten(), dim=0)
            frame_cosine = F.cosine_similarity(denormalized, target_latents.to(device), dim=-1).mean()
            with torch.no_grad():
                audio = dav.decode(denormalized.transpose(0, 1).unsqueeze(0).to(dav.dtype))
            sf.write(args.output_dir / f"generated_step{step}.flac", audio.squeeze(0).float().cpu().T.numpy(), SAMPLE_RATE)
            torch.save(
                {
                    "model": model.state_dict(),
                    "latent_mean": latent_mean,
                    "latent_std": latent_std,
                    "cond_mean": cond_mean,
                    "cond_std": cond_std,
                    "args": vars(args)
                    | {
                        "source_audio": str(args.source_audio),
                        "target_audio": str(args.target_audio),
                        "output_dir": str(args.output_dir),
                    },
                },
                args.output_dir / f"checkpoint_step{step}.pt",
            )
            print(
                json.dumps(
                    {
                        "step": step,
                        "sample_global_cosine": round(cosine.item(), 6),
                        "sample_frame_cosine": round(frame_cosine.item(), 6),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
