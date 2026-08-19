#!/usr/bin/env python3
# Copyright 2026 SimpleTuner contributors
# Licensed under the Apache License, Version 2.0
"""DDP latent re-planner trainer with online MERT/DAV extraction.

Per-layer (n->n) MERT conditioning into a d768-class DiT, random 30s crops,
checkpoints and held-out renders written to disk at a fixed cadence.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.minimax_music3.train_latent_replanner import (  # noqa: E402
    MERT_SAMPLE_RATE,
    SAMPLE_RATE,
    LatentReplanner,
    load_audio,
    sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DDP latent re-planner training")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--pair-ids-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--mert-id", default="m-a-p/MERT-v1-95M")
    parser.add_argument("--mert-input-layer", type=int, default=7)
    parser.add_argument("--clap-id", default="laion/larger_clap_music")
    parser.add_argument("--codes-dir", type=Path, help="precomputed RVQ codes dir; enables the code stream")
    parser.add_argument("--codes-per-crop", type=int, default=1024)
    parser.add_argument("--style-dropout", type=float, default=0.15)
    parser.add_argument(
        "--identity-rate", type=float, default=0.0, help="probability a sample trains as identity (target := source)"
    )
    parser.add_argument("--mert-full-mask-rate", type=float, default=0.05)
    parser.add_argument(
        "--mert-span-mask-rate", type=float, default=0.5, help="fraction of samples receiving partial span masks"
    )
    parser.add_argument("--mert-max-mask-fraction", type=float, default=0.3)
    parser.add_argument("--code-dropout", type=float, default=0.15)
    parser.add_argument("--warm-start", type=Path, help="load model weights only (strict=False), fresh optimizer")
    parser.add_argument("--cache-dir")
    parser.add_argument("--crop-seconds", type=float, default=30.0)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--sample-steps", type=int, default=32)
    parser.add_argument("--holdout-count", type=int, default=64)
    parser.add_argument("--eval-holdout-samples", type=int, default=32)
    parser.add_argument("--render-count", type=int, default=2)
    parser.add_argument("--stats-samples", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--degrade-rate",
        type=float,
        default=0.0,
        help="probability a sample trains as restore (degraded input, clean target)",
    )
    parser.add_argument("--objective", choices=("fm", "ddpm", "bridge"), default="fm")
    parser.add_argument(
        "--degraded-latent-stream", action="store_true", help="feed DAV(degraded) latents as SR3-style conditioning"
    )
    parser.add_argument(
        "--clap-from-source",
        action="store_true",
        help="condition CLAP on the (possibly degraded) source instead of the target",
    )
    parser.add_argument(
        "--eval-degrade", action="store_true", help="apply seeded degradation to holdout sources (refiner evaluation)"
    )
    parser.add_argument(
        "--task-conditioning", action="store_true", help="learned task embedding with dropout to trained null"
    )
    parser.add_argument("--task-dropout", type=float, default=0.15)
    return parser.parse_args()


class PairCropDataset(Dataset):
    """Yields aligned random crops of (source 24k mono, target 44.1k stereo) raw audio."""

    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        pair_ids: list[str],
        crop_seconds: float,
        deterministic: bool,
        codes_dir: Path | None = None,
        codes_per_crop: int = 1024,
        identity_rate: float = 0.0,
        degrade_rate: float = 0.0,
        eval_degrade: bool = False,
    ):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.pair_ids = pair_ids
        self.crop_seconds = crop_seconds
        self.deterministic = deterministic
        self.codes_dir = codes_dir
        self.codes_per_crop = codes_per_crop
        self.identity_rate = identity_rate
        self.degrade_rate = degrade_rate
        self.eval_degrade = eval_degrade

    def __len__(self) -> int:
        return len(self.pair_ids)

    def __getitem__(self, index: int) -> dict:
        for hop in range(5):
            try:
                return self._load_item((index + hop) % len(self.pair_ids))
            except Exception:
                if hop == 4:
                    raise
        raise RuntimeError("unreachable")

    def _load_item(self, index: int) -> dict:
        pair_id = self.pair_ids[index]
        draw = random.random() if not self.deterministic else 1.0
        identity = draw < self.identity_rate
        restore = self.identity_rate <= draw < self.identity_rate + self.degrade_rate
        if self.deterministic and self.eval_degrade:
            restore = True
        source = load_audio(self.source_dir / pair_id, MERT_SAMPLE_RATE, mono=True)
        target_path = (self.source_dir if (identity or restore) else self.target_dir) / pair_id
        target = load_audio(target_path, SAMPLE_RATE, mono=False)
        if target.shape[0] == 1:
            target = target.repeat(2, 1)
        source_seconds = source.shape[-1] / MERT_SAMPLE_RATE
        target_seconds = target.shape[-1] / SAMPLE_RATE
        max_offset = min(source_seconds, target_seconds) - self.crop_seconds
        if max_offset < 0:
            raise ValueError(f"{pair_id} is shorter than the crop window")
        if self.deterministic:
            offset = 0.0
        else:
            offset = random.uniform(0.0, max_offset)
        source_start = int(offset * MERT_SAMPLE_RATE)
        target_start = int(offset * SAMPLE_RATE)
        source_length = int(self.crop_seconds * MERT_SAMPLE_RATE)
        target_length = int(self.crop_seconds * SAMPLE_RATE)
        source_crop = source[..., source_start : source_start + source_length]
        target_crop = target[..., target_start : target_start + target_length]
        degraded_crop = None
        if restore:
            if self.deterministic:
                state = random.getstate()
                random.seed(1000 + index)
            degraded_crop = degrade_source(target_crop, SAMPLE_RATE)
            import torchaudio

            source_crop = torchaudio.functional.resample(
                degraded_crop.mean(dim=0, keepdim=True), SAMPLE_RATE, MERT_SAMPLE_RATE
            )
            expected = int(self.crop_seconds * MERT_SAMPLE_RATE)
            if source_crop.shape[-1] < expected:
                source_crop = F.pad(source_crop, (0, expected - source_crop.shape[-1]))
            source_crop = source_crop[..., :expected]
            if self.deterministic:
                random.setstate(state)
        item = {
            "task": 1 if identity else (2 if restore else 0),
            "source": source_crop,
            "target": target_crop,
            "degraded": degraded_crop if degraded_crop is not None else target_crop,
        }
        if self.codes_dir is not None:
            payload = torch.load(self.codes_dir / f"{pair_id.replace('/', '__')}.pt", weights_only=True)
            codes = payload["codes"]
            duration = float(payload["duration"])
            code_start = offset / duration * codes.shape[0]
            code_span = self.crop_seconds / duration * codes.shape[0]
            positions = torch.linspace(code_start, min(code_start + code_span, codes.shape[0] - 1), self.codes_per_crop)
            item["codes"] = codes[positions.long()].long()
        return item


def degrade_source(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Random restoration-task degradation chain: bandwidth crush, noise, bit crush, clipping."""
    import torchaudio

    degraded = waveform
    if random.random() < 0.8:
        low_rate = random.choice((4_000, 8_000, 12_000))
        degraded = torchaudio.functional.resample(
            torchaudio.functional.resample(degraded, sample_rate, low_rate), low_rate, sample_rate
        )
        if degraded.shape[-1] < waveform.shape[-1]:
            degraded = F.pad(degraded, (0, waveform.shape[-1] - degraded.shape[-1]))
        degraded = degraded[..., : waveform.shape[-1]]
    if random.random() < 0.5:
        snr_db = random.uniform(15.0, 35.0)
        signal_power = degraded.square().mean().clamp_min(1e-8)
        noise = torch.randn_like(degraded) * (signal_power / (10 ** (snr_db / 10))).sqrt()
        degraded = degraded + noise
    if random.random() < 0.3:
        levels = 2 ** random.choice((8, 9, 10))
        degraded = (degraded * levels).round() / levels
    if random.random() < 0.3:
        drive = random.uniform(1.5, 4.0)
        degraded = torch.tanh(degraded * drive) / drive
    return degraded


def collate_crops(items: list[dict]) -> dict:
    batch = {
        "source": torch.stack([item["source"] for item in items]),
        "target": torch.stack([item["target"] for item in items]),
    }
    if "codes" in items[0]:
        batch["codes"] = torch.stack([item["codes"] for item in items])
    if "task" in items[0]:
        batch["task"] = torch.tensor([item["task"] for item in items], dtype=torch.long)
    if "degraded" in items[0]:
        batch["degraded"] = torch.stack([item["degraded"] for item in items])
    return batch


class OnlineExtractor:
    """Frozen MERT (all layers) + DAV encoder living on the training GPU."""

    def __init__(self, args, device: torch.device):
        from huggingface_hub import hf_hub_download
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        from simpletuner.helpers.models.minimaxmusic.vocoder import MiniMaxMusic3DAV

        self.device = device
        self.mert = (
            AutoModel.from_pretrained(args.mert_id, trust_remote_code=True, cache_dir=args.cache_dir).to(device).eval()
        )
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            args.mert_id, trust_remote_code=True, cache_dir=args.cache_dir
        )
        dav_path = hf_hub_download(args.model_id, "dav.pth", cache_dir=args.cache_dir)
        self.dav = MiniMaxMusic3DAV.from_original_dav(dav_path).to(device).eval()
        self.mert.requires_grad_(False)
        self.dav.requires_grad_(False)
        from transformers import ClapModel, ClapProcessor

        self.clap = ClapModel.from_pretrained(args.clap_id, cache_dir=args.cache_dir).to(device).eval()
        self.clap.requires_grad_(False)
        self.clap_processor = ClapProcessor.from_pretrained(args.clap_id, cache_dir=args.cache_dir)

    @torch.no_grad()
    def clap_audio_embedding(self, target: torch.Tensor) -> torch.Tensor:
        """Pooled CLAP audio embedding of the center 10s of each 44.1k stereo target crop."""
        import torchaudio

        mono = target.mean(dim=1)
        center = mono.shape[-1] // 2
        half = min(5 * SAMPLE_RATE, center)
        clip = mono[..., center - half : center + half]
        clip = torchaudio.functional.resample(clip.cpu(), SAMPLE_RATE, 48_000)
        inputs = self.clap_processor(audio=[row.numpy() for row in clip], sampling_rate=48_000, return_tensors="pt")
        features = self.clap.get_audio_features(input_features=inputs["input_features"].to(self.device))
        if not torch.is_tensor(features):
            features = features.pooler_output
        if features.shape[-1] != self.clap.config.projection_dim:
            features = self.clap.audio_projection(features)
        return features.float()

    @torch.no_grad()
    def __call__(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (latents [B,T,128], mert_layers [B,L,T,768]) at the DAV frame rate."""
        target = batch["target"].to(self.device)
        latents = self.dav.encode(target).transpose(1, 2).float()
        self.last_degraded_latents = None
        if "degraded" in batch:
            self.last_degraded_latents = self.dav.encode(batch["degraded"].to(self.device)).transpose(1, 2).float()
        frame_count = latents.shape[1]
        inputs = self.processor(
            [waveform.squeeze(0).numpy() for waveform in batch["source"]],
            sampling_rate=MERT_SAMPLE_RATE,
            return_tensors="pt",
        )
        outputs = self.mert(inputs["input_values"].to(self.device), output_hidden_states=True)
        layers = torch.stack(outputs.hidden_states, dim=1).float()  # [B, L, Tm, 768]
        batch_size, layer_count, mert_frames, dim = layers.shape
        layers = (
            F.interpolate(
                layers.reshape(batch_size * layer_count, mert_frames, dim).transpose(1, 2),
                size=frame_count,
                mode="linear",
                align_corners=True,
            )
            .transpose(1, 2)
            .reshape(batch_size, layer_count, frame_count, dim)
        )
        return latents, layers


@torch.no_grad()
def bridge_sample(
    model, conditioning, steps, layer_conditioning, degraded_latents, direction="restore", style=None, code_conditioning=None
):
    """Deterministic Euler along the clean<->degraded bridge. restore: t 1->0 from degraded; degrade: t 0->1 from clean."""
    latents = degraded_latents.clone()
    if direction == "restore":
        schedule = torch.linspace(1.0, 0.0, steps + 1, device=latents.device)
    else:
        schedule = torch.linspace(0.0, 1.0, steps + 1, device=latents.device)
    for index in range(steps):
        t = schedule[index].expand(latents.shape[0])
        velocity = model(latents, conditioning, t, layer_conditioning, style, code_conditioning)
        latents = latents - (schedule[index] - schedule[index + 1]) * velocity
    return latents


def ddpm_alpha_bar(t: torch.Tensor) -> torch.Tensor:
    return torch.cos(t.clamp(0.0, 0.999) * torch.pi / 2).square()


@torch.no_grad()
def ddpm_ancestral_sample(
    model, conditioning, steps, generator, layer_conditioning=None, style=None, code_conditioning=None
):
    device = conditioning.device
    latents = torch.randn(
        conditioning.shape[0], conditioning.shape[1], model.proj_out.out_features, generator=generator, device="cpu"
    ).to(device)
    times = torch.linspace(1.0, 0.0, steps + 1, device=device)
    for index in range(steps):
        t = times[index].expand(latents.shape[0])
        t_next = times[index + 1]
        abar = ddpm_alpha_bar(t)[:, None, None]
        abar_next = ddpm_alpha_bar(t_next.expand(latents.shape[0]))[:, None, None]
        v = model(latents, conditioning, t, layer_conditioning, style, code_conditioning)
        x0 = abar.sqrt() * latents - (1 - abar).sqrt() * v
        eps = (1 - abar).sqrt() * latents + abar.sqrt() * v
        if index == steps - 1:
            latents = x0
        else:
            alpha_step = (abar / abar_next).clamp(max=1.0)
            var = ((1 - abar_next) / (1 - abar)) * (1 - alpha_step)
            mean = abar_next.sqrt() * x0 + (1 - abar_next - var).clamp_min(0.0).sqrt() * eps
            latents = mean + var.sqrt() * torch.randn(
                latents.shape, generator=torch.Generator().manual_seed(int(index) + 7), device="cpu"
            ).to(device)
    return latents


def barrier(active: bool) -> None:
    if active:
        dist.barrier()


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    device = torch.device("cuda", local_rank)
    main_process = rank == 0
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    pair_ids = [line.strip() for line in args.pair_ids_file.read_text().replace(",", "\n").splitlines() if line.strip()]
    if len(pair_ids) <= args.holdout_count:
        raise ValueError("holdout-count leaves no training pairs")
    train_ids = pair_ids[: -args.holdout_count]
    holdout_ids = pair_ids[-args.holdout_count :]

    dataset = PairCropDataset(
        args.source_dir,
        args.target_dir,
        train_ids,
        args.crop_seconds,
        deterministic=False,
        codes_dir=args.codes_dir,
        codes_per_crop=args.codes_per_crop,
        identity_rate=args.identity_rate,
        degrade_rate=args.degrade_rate,
    )
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        collate_fn=collate_crops,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    extractor = OnlineExtractor(args, device)

    stats_path = args.output_dir / "stats.pt"
    if main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if not stats_path.exists():
            stats_dataset = PairCropDataset(
                args.source_dir, args.target_dir, train_ids[: args.stats_samples], args.crop_seconds, deterministic=True
            )
            stats_loader = DataLoader(stats_dataset, batch_size=4, collate_fn=collate_crops)
            latent_sum = latent_sq = layer_sum = layer_sq = None
            count = layer_frames = 0
            for stats_batch in stats_loader:
                latents, layers = extractor(stats_batch)
                flat = latents.reshape(-1, latents.shape[-1])
                flat_layers = layers.permute(1, 0, 2, 3).reshape(layers.shape[1], -1, layers.shape[-1])
                if latent_sum is None:
                    latent_sum = flat.sum(0)
                    latent_sq = flat.square().sum(0)
                    layer_sum = flat_layers.sum(1)
                    layer_sq = flat_layers.square().sum(1)
                else:
                    latent_sum += flat.sum(0)
                    latent_sq += flat.square().sum(0)
                    layer_sum += flat_layers.sum(1)
                    layer_sq += flat_layers.square().sum(1)
                count += flat.shape[0]
                layer_frames += flat_layers.shape[1]
            latent_mean = latent_sum / count
            latent_std = (latent_sq / count - latent_mean.square()).clamp_min(1e-8).sqrt()
            layer_mean = layer_sum / layer_frames
            layer_std = (layer_sq / layer_frames - layer_mean.square()).clamp_min(1e-8).sqrt()
            torch.save(
                {
                    "latent_mean": latent_mean.cpu(),
                    "latent_std": latent_std.cpu(),
                    "layer_mean": layer_mean.cpu(),
                    "layer_std": layer_std.cpu(),
                },
                stats_path,
            )
    barrier(distributed)
    stats = torch.load(stats_path, map_location=device, weights_only=True)
    latent_mean = stats["latent_mean"][None, None]
    latent_std = stats["latent_std"][None, None]
    layer_mean = stats["layer_mean"][None, :, None]
    layer_std = stats["layer_std"][None, :, None]

    model = LatentReplanner(128, 768, args.d_model, args.depth, args.heads)
    model.enable_layer_conditioning(768, 13)
    model.enable_mert_masking(768, 13)
    model.enable_style_conditioning(512)
    if args.degraded_latent_stream or args.objective == "bridge":
        model.enable_degraded_latent_conditioning(128)
    if args.task_conditioning:
        model.enable_task_conditioning(3)
    if args.codes_dir is not None:
        codes_meta = json.loads((args.codes_dir / "meta.json").read_text())
        model.enable_code_conditioning(codes_meta["total_vocab"], codes_meta["books"])
    model = model.to(device)
    if args.warm_start is not None:
        payload = torch.load(args.warm_start, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"warm-start has unexpected keys: {unexpected}")
        if main_process:
            print(json.dumps({"warm_start_missing_keys": len(missing)}), flush=True)
    if main_process:
        print(
            json.dumps(
                {
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "train_pairs": len(train_ids),
                    "holdout_pairs": len(holdout_ids),
                    "world_size": world_size,
                    "global_batch": args.batch_size * world_size,
                    "layer_map": model.layer_map,
                }
            ),
            flush=True,
        )
    wrapped = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True) if distributed else model
    optimizer = torch.optim.AdamW(wrapped.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_step = 0
    if args.resume is not None:
        payload = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = int(payload["step"])

    holdout_batch = None
    holdout_latents = holdout_layers = None
    if main_process:
        holdout_dataset = PairCropDataset(
            args.source_dir,
            args.target_dir,
            holdout_ids[: args.eval_holdout_samples],
            args.crop_seconds,
            deterministic=True,
            codes_dir=args.codes_dir,
            codes_per_crop=args.codes_per_crop,
            eval_degrade=args.eval_degrade,
        )
        holdout_items = [holdout_dataset[index] for index in range(len(holdout_dataset))]
        holdout_batch = collate_crops(holdout_items)
        holdout_latents, holdout_layers = extractor(holdout_batch)
        holdout_degraded = None
        if extractor.last_degraded_latents is not None and (args.degraded_latent_stream or args.objective == "bridge"):
            holdout_degraded = ((extractor.last_degraded_latents - latent_mean) / latent_std).cpu()
        if args.clap_from_source:
            import torchaudio as _ta2

            holdout_clap_input = _ta2.functional.resample(holdout_batch["source"], MERT_SAMPLE_RATE, SAMPLE_RATE).repeat(
                1, 2, 1
            )
            holdout_style = extractor.clap_audio_embedding(holdout_clap_input)
        else:
            holdout_style = extractor.clap_audio_embedding(holdout_batch["target"])
        holdout_codes = None
        if args.codes_dir is not None:
            holdout_codes = torch.stack(
                [
                    model.embed_codes(holdout_batch["codes"][index : index + 1].to(device), holdout_latents.shape[1])
                    .squeeze(0)
                    .cpu()
                    for index in range(holdout_batch["codes"].shape[0])
                ]
            )
        holdout_layers = ((holdout_layers - layer_mean) / layer_std).to(torch.float16).cpu()

    def lr_scale(step: int) -> float:
        return min(1.0, step / max(1, args.warmup_steps))

    step = start_step
    task_loss_sums = [0.0, 0.0, 0.0]
    task_loss_counts = [0, 0, 0]
    started = time.perf_counter()
    data_iterator = iter(loader)
    epoch = 0
    while step < args.steps:
        try:
            batch = next(data_iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iterator = iter(loader)
            batch = next(data_iterator)
        latents, layers = extractor(batch)
        clean = (latents - latent_mean) / latent_std
        layers = (layers - layer_mean) / layer_std
        batch_size = clean.shape[0]
        frame_total = layers.shape[2]
        for row in range(batch_size):
            draw = torch.rand(())
            if draw < args.mert_full_mask_rate:
                layers[row] = model.mert_null[:, None, :]
            elif draw < args.mert_full_mask_rate + args.mert_span_mask_rate:
                span = int(torch.randint(1, max(2, int(frame_total * args.mert_max_mask_fraction)), (1,)))
                start = int(torch.randint(0, frame_total - span, (1,)))
                layers[row, :, start : start + span] = model.mert_null[:, None, :]
        conditioning = layers[:, args.mert_input_layer]
        if args.clap_from_source:
            import torchaudio as _ta

            clap_input = _ta.functional.resample(batch["source"], MERT_SAMPLE_RATE, SAMPLE_RATE).repeat(1, 2, 1)
            style = extractor.clap_audio_embedding({"target": clap_input}["target"])
        else:
            style = extractor.clap_audio_embedding(batch["target"])
        style_keep = (torch.rand(batch_size, device=device) >= args.style_dropout).float()[:, None]
        style = style_keep * style + (1.0 - style_keep) * model.style_null[None]
        code_conditioning = None
        if args.codes_dir is not None:
            code_conditioning = model.embed_codes(batch["codes"].to(device), clean.shape[1])
            code_keep = (torch.rand(batch_size, device=device) >= args.code_dropout).float()[:, None, None]
            code_conditioning = code_keep * code_conditioning + (1.0 - code_keep) * model.code_null[None, None]
        degraded_latents = None
        if extractor.last_degraded_latents is not None and (args.degraded_latent_stream or args.objective == "bridge"):
            degraded_latents = (extractor.last_degraded_latents - latent_mean) / latent_std
        noise = torch.randn_like(clean)
        if args.objective == "bridge":
            if degraded_latents is None:
                raise RuntimeError("bridge objective requires degraded latents")
            t = torch.rand(batch_size, device=device)
            noisy = (1.0 - t[:, None, None]) * clean + t[:, None, None] * degraded_latents
            prediction_target = degraded_latents - clean
        elif args.objective == "ddpm":
            t = torch.rand(batch_size, device=device)
            abar = ddpm_alpha_bar(t)[:, None, None]
            noisy = abar.sqrt() * clean + (1 - abar).sqrt() * noise
            prediction_target = abar.sqrt() * noise - (1 - abar).sqrt() * clean
        else:
            t = torch.sigmoid(torch.randn(batch_size, device=device))
            noisy = (1.0 - t[:, None, None]) * clean + t[:, None, None] * noise
            prediction_target = noise - clean
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * lr_scale(step + 1)
        stream_latents = degraded_latents if args.degraded_latent_stream and args.objective != "bridge" else None
        task_ids = None
        if args.task_conditioning and "task" in batch:
            task_ids = batch["task"].to(device)
            null_mask = torch.rand(batch_size, device=device) < args.task_dropout
            task_ids = torch.where(null_mask, torch.full_like(task_ids, 3), task_ids)
        velocity = wrapped(noisy, conditioning, t, layers, style, code_conditioning, stream_latents, task_ids)
        per_sample = (velocity - prediction_target).square().mean(dim=(1, 2))
        loss = per_sample.mean()
        if "task" in batch:
            tasks = batch["task"].to(device)
            for task_id in (0, 1, 2):
                mask = tasks == task_id
                if mask.any():
                    task_loss_sums[task_id] += per_sample[mask].sum().item()
                    task_loss_counts[task_id] += int(mask.sum())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        if main_process and (step % 50 == 0 or step == start_step + 1):
            record = {
                "step": step,
                "loss": round(loss.item(), 6),
                "grad_norm": round(float(grad_norm), 4),
                "steps_per_second": round((step - start_step) / (time.perf_counter() - started), 3),
                "epoch": epoch,
            }
            for task_id, name in ((0, "loss_transfer"), (1, "loss_identity"), (2, "loss_restore")):
                if task_loss_counts[task_id]:
                    record[name] = round(task_loss_sums[task_id] / task_loss_counts[task_id], 6)
            task_loss_sums = [0.0, 0.0, 0.0]
            task_loss_counts = [0, 0, 0]
            print(json.dumps(record), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            if main_process:
                model.eval()
                chunks = []
                for start in range(0, holdout_layers.shape[0], 8):
                    chunk_layers = holdout_layers[start : start + 8].to(device).float()
                    chunks.append(
                        sample(
                            model,
                            chunk_layers[:, args.mert_input_layer],
                            args.sample_steps,
                            torch.Generator(device="cpu").manual_seed(args.seed + start),
                            chunk_layers,
                            style=holdout_style[start : start + 8],
                            code_conditioning=(
                                holdout_codes[start : start + 8].to(device).float() if holdout_codes is not None else None
                            ),
                        )
                    )
                model.train()
                generated = torch.cat(chunks, dim=0)
                denormalized = generated * latent_std + latent_mean
                flat_generated = F.normalize(denormalized.flatten(1), dim=1)
                flat_targets = F.normalize(holdout_latents.flatten(1), dim=1)
                confusion = flat_generated @ flat_targets.T
                diagonal = confusion.diagonal()
                count = confusion.shape[0]
                print(
                    json.dumps(
                        {
                            "step": step,
                            "holdout_diag_mean": round(diagonal.mean().item(), 4),
                            "holdout_diag_min": round(diagonal.min().item(), 4),
                            "holdout_diag_max": round(diagonal.max().item(), 4),
                            "holdout_offdiag_mean": round(
                                ((confusion.sum() - diagonal.sum()) / (count * (count - 1))).item(), 4
                            ),
                        }
                    ),
                    flush=True,
                )
                checkpoint_dir = args.output_dir / f"checkpoint-{step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "args": {k: str(v) for k, v in vars(args).items()},
                    },
                    checkpoint_dir / "state.pt",
                )
                for index in range(min(args.render_count, count)):
                    with torch.no_grad():
                        audio = extractor.dav.decode(
                            denormalized[index].transpose(0, 1).unsqueeze(0).to(extractor.dav.dtype)
                        )
                    sf.write(
                        checkpoint_dir / f"holdout{index}_generated.flac",
                        audio.squeeze(0).float().cpu().T.numpy(),
                        SAMPLE_RATE,
                    )
                    if step == args.eval_every:
                        with torch.no_grad():
                            reference = extractor.dav.decode(
                                holdout_latents[index].transpose(0, 1).unsqueeze(0).to(extractor.dav.dtype)
                            )
                        sf.write(
                            args.output_dir / f"holdout{index}_roundtrip.flac",
                            reference.squeeze(0).float().cpu().T.numpy(),
                            SAMPLE_RATE,
                        )
                        sf.write(
                            args.output_dir / f"holdout{index}_source.flac",
                            holdout_batch["source"][index].T.numpy(),
                            MERT_SAMPLE_RATE,
                        )
            barrier(distributed)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
