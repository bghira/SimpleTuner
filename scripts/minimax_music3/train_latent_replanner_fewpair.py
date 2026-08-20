#!/usr/bin/env python3
# Copyright 2026 SimpleTuner contributors
# Licensed under the Apache License, Version 2.0
"""Few-pair selection test for the latent re-planner.

One small model, N source→target transforms. Acceptance: the conditioning
selects the right transform — cosine confusion matrix with a high diagonal
and low off-diagonals.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.minimax_music3.train_latent_replanner import (  # noqa: E402
    MERT_SAMPLE_RATE,
    SAMPLE_RATE,
    LatentReplanner,
    crop,
    extract_mert_features,
    load_audio,
    sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-pair latent re-planner selection test")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--pair-ids", required=True, help="comma-separated audio file names present in both dirs")
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
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sample-every", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--decode-pairs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=0, help="pairs per step; 0 = all pairs every step")
    parser.add_argument(
        "--mert-per-layer", action="store_true", help="feed MERT layers 1..penultimate into matching DiT blocks"
    )
    parser.add_argument("--holdout-count", type=int, default=0, help="pairs excluded from training, eval-only")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pair_ids = [item.strip() for item in args.pair_ids.split(",") if item.strip()]
    if len(pair_ids) < 2:
        raise ValueError("few-pair test needs at least two pairs")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    from simpletuner.helpers.models.minimaxmusic.vocoder import MiniMaxMusic3DAV

    mert = AutoModel.from_pretrained(args.mert_id, trust_remote_code=True, cache_dir=args.cache_dir).to(device).eval()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.mert_id, trust_remote_code=True, cache_dir=args.cache_dir)
    mert_features = []
    for pair_id in pair_ids:
        source_24k = crop(
            load_audio(args.source_dir / pair_id, MERT_SAMPLE_RATE, mono=True),
            MERT_SAMPLE_RATE,
            args.crop_offset_seconds,
            args.crop_seconds,
        )
        mert_features.append(
            extract_mert_features(source_24k, mert, processor, device, args.mert_layer, all_layers=args.mert_per_layer).cpu()
        )
    del mert
    torch.cuda.empty_cache()

    dav_path = hf_hub_download(args.model_id, "dav.pth", cache_dir=args.cache_dir)
    dav = MiniMaxMusic3DAV.from_original_dav(dav_path).to(device).eval()
    target_latents = []
    for pair_id in pair_ids:
        target_44k = crop(
            load_audio(args.target_dir / pair_id, SAMPLE_RATE, mono=False),
            SAMPLE_RATE,
            args.crop_offset_seconds,
            args.crop_seconds,
        )
        with torch.no_grad():
            target_latents.append(dav.encode(target_44k.unsqueeze(0).to(device)).squeeze(0).transpose(0, 1).float())

    frame_count = min(latents.shape[0] for latents in target_latents)
    targets = torch.stack([latents[:frame_count] for latents in target_latents])
    if args.mert_per_layer:
        stacked = torch.stack(
            [
                F.interpolate(features.transpose(1, 2), size=frame_count, mode="linear", align_corners=True).transpose(1, 2)
                for features in mert_features
            ]
        )  # [pairs, mert_layers, frames, dim]
        layer_mean = stacked.mean(dim=(0, 2), keepdim=True)
        layer_std = stacked.std(dim=(0, 2), keepdim=True).clamp_min(1e-4)
        layer_conditioning = ((stacked - layer_mean) / layer_std).to(torch.float16)
        conditioning = stacked[:, args.mert_layer].clone()
        del stacked
    else:
        layer_conditioning = None
        conditioning = torch.stack(
            [
                F.interpolate(features.transpose(0, 1).unsqueeze(0), size=frame_count, mode="linear", align_corners=True)
                .squeeze(0)
                .transpose(0, 1)
                for features in mert_features
            ]
        )

    latent_mean = targets.mean(dim=(0, 1), keepdim=True)
    latent_std = targets.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
    normalized_targets = (targets - latent_mean) / latent_std
    cond_mean = conditioning.mean(dim=(0, 1), keepdim=True)
    cond_std = conditioning.std(dim=(0, 1), keepdim=True).clamp_min(1e-4)
    conditioning = (conditioning - cond_mean) / cond_std

    model = LatentReplanner(128, conditioning.shape[-1], args.d_model, args.depth, args.heads)
    if args.mert_per_layer:
        model.enable_layer_conditioning(conditioning.shape[-1], layer_conditioning.shape[1])
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({"parameters": parameter_count, "pairs": len(pair_ids), "frames": frame_count}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    total = len(pair_ids)
    if not 0 <= args.holdout_count < total:
        raise ValueError("holdout-count must leave at least one training pair")
    train_count = total - args.holdout_count
    batch = args.batch_size if args.batch_size > 0 else train_count
    if batch > train_count:
        raise ValueError("batch-size exceeds training pair count")
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        indices = torch.randperm(train_count, generator=generator)[:batch]
        t = torch.sigmoid(torch.randn(batch, generator=generator)).to(device)
        clean = normalized_targets[indices].to(device)
        noise = torch.randn(clean.shape, generator=generator).to(device)
        noisy = (1.0 - t[:, None, None]) * clean + t[:, None, None] * noise
        step_layers = layer_conditioning[indices].to(device).float() if layer_conditioning is not None else None
        velocity = model(noisy, conditioning[indices].to(device), t, step_layers)
        loss = F.mse_loss(velocity, noise - clean)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if step % 100 == 0 or step == 1:
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
            chunks = []
            for start in range(0, total, 16):
                chunks.append(
                    sample(
                        model,
                        conditioning[start : start + 16].to(device),
                        args.sample_steps,
                        torch.Generator(device="cpu").manual_seed(args.seed + start),
                        (
                            layer_conditioning[start : start + 16].to(device).float()
                            if layer_conditioning is not None
                            else None
                        ),
                    )
                )
            generated = torch.cat(chunks, dim=0)
            model.train()
            denormalized = generated * latent_std.to(device) + latent_mean.to(device)
            flat_generated = F.normalize(denormalized.flatten(1), dim=1)
            flat_targets = F.normalize(targets.to(device).flatten(1), dim=1)
            confusion = flat_generated @ flat_targets.T
            diagonal = confusion.diagonal()
            off_diagonal = confusion - torch.eye(total, device=device) * confusion.diagonal()
            train_diagonal = diagonal[:train_count]
            report = {
                "step": step,
                "train_diag_mean": round(train_diagonal.mean().item(), 4),
                "train_diag_min": round(train_diagonal.min().item(), 4),
                "offdiag_max": round(off_diagonal.max().item(), 4),
                "offdiag_mean": round((off_diagonal.sum() / (total * (total - 1))).item(), 4),
            }
            if args.holdout_count:
                holdout_diagonal = diagonal[train_count:]
                report["holdout_diag_mean"] = round(holdout_diagonal.mean().item(), 4)
                report["holdout_diag_min"] = round(holdout_diagonal.min().item(), 4)
                report["holdout_diag_max"] = round(holdout_diagonal.max().item(), 4)
            print(json.dumps(report), flush=True)
            if step == args.steps:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "latent_mean": latent_mean,
                        "latent_std": latent_std,
                        "cond_mean": cond_mean,
                        "cond_std": cond_std,
                        "pair_ids": pair_ids,
                        "train_count": train_count,
                    },
                    args.output_dir / f"checkpoint_step{step}.pt",
                )
                for index in range(min(args.decode_pairs, batch)):
                    with torch.no_grad():
                        audio = dav.decode(denormalized[index].transpose(0, 1).unsqueeze(0).to(dav.dtype))
                        reference_audio = dav.decode(targets[index].to(device).transpose(0, 1).unsqueeze(0).to(dav.dtype))
                    stem = Path(pair_ids[index]).stem[:8]
                    sf.write(
                        args.output_dir / f"pair{index}_{stem}_generated.flac",
                        audio.squeeze(0).float().cpu().T.numpy(),
                        SAMPLE_RATE,
                    )
                    sf.write(
                        args.output_dir / f"pair{index}_{stem}_roundtrip.flac",
                        reference_audio.squeeze(0).float().cpu().T.numpy(),
                        SAMPLE_RATE,
                    )


if __name__ == "__main__":
    main()
