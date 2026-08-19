#!/usr/bin/env python3
# Copyright 2026 SimpleTuner contributors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, Qwen3ForCausalLM, get_polynomial_decay_schedule_with_warmup

from simpletuner.helpers.models.minimaxmusic.encoders import _clean_caption, _normalize_lyrics
from simpletuner.helpers.models.minimaxmusic.reference_control import (
    MiniMaxMusic3ReferenceControlAdapter,
    ReferenceControlConfig,
    create_qwen_oftv2_adapter,
    embed_rvq_frames,
)
from simpletuner.helpers.models.minimaxmusic.rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder

AUDIO_CODE_OFFSET = 151_675
AUDIO_VOCAB_SIZE = 16_384
PROMPT_TEMPLATE = (
    "<|im_start|><|caption_start|>{caption}<|caption_end|>" "<|lyrics_start|>{lyrics}<|lyrics_end|><|im_end|><|audio_start|>"
)


class CachedStylePairDataset(Dataset):
    def __init__(
        self,
        cache_dir: Path,
        crop_frames: int,
        reference_context_frames: int,
        *,
        split: str = "train",
        validation_fraction: float = 0.1,
        clip_id: str | None = None,
        clip_ids: tuple[str, ...] | None = None,
        fixed_crop_start: int | None = None,
        target_context_frames: int = 0,
    ):
        if split not in {"train", "validation"}:
            raise ValueError("split must be 'train' or 'validation'")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if clip_id is not None and clip_ids is not None:
            raise ValueError("clip_id and clip_ids are mutually exclusive")
        paths = sorted(cache_dir.glob("shard-*/*.safetensors"))
        if clip_ids is not None:
            if not clip_ids or len(set(clip_ids)) != len(clip_ids):
                raise ValueError("clip_ids must be non-empty and unique")
            paths_by_id = {path.stem: path for path in paths}
            missing = [value for value in clip_ids if value not in paths_by_id]
            if missing:
                raise ValueError(f"Cached style pairs are missing {len(missing)} requested clip IDs")
            self.paths = [paths_by_id[value] for value in clip_ids]
        elif clip_id is None:
            threshold = int(validation_fraction * 2**64)
            self.paths = [
                path
                for path in paths
                if (int.from_bytes(hashlib.sha256(path.stem.encode("utf-8")).digest()[:8], "big") < threshold)
                == (split == "validation")
            ]
        else:
            self.paths = [path for path in paths if path.stem == clip_id]
        if not self.paths:
            selection = (
                "the requested clip subset"
                if clip_ids is not None
                else (f"clip {clip_id}" if clip_id is not None else f"the {split} split")
            )
            raise ValueError(f"No cached style pairs found for {selection} under {cache_dir}")
        if crop_frames < 2:
            raise ValueError("crop_frames must be at least 2")
        if reference_context_frames < 0:
            raise ValueError("reference_context_frames must be non-negative")
        if target_context_frames < 0:
            raise ValueError("target_context_frames must be non-negative")
        if fixed_crop_start is not None and fixed_crop_start < 0:
            raise ValueError("fixed_crop_start must be non-negative")
        self.crop_frames = crop_frames
        self.reference_context_frames = reference_context_frames
        self.target_context_frames = target_context_frames
        self.deterministic_crops = split == "validation"
        self.fixed_crop_start = fixed_crop_start

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        path = self.paths[index]
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
            target_length = handle.get_slice("target_codes").get_shape()[0]
            reference_length = handle.get_slice("reference_codes").get_shape()[0]
            frame_count = min(self.crop_frames, target_length)
            max_start = target_length - frame_count
            if self.fixed_crop_start is not None:
                if self.fixed_crop_start > max_start:
                    raise ValueError(f"fixed crop start {self.fixed_crop_start} exceeds maximum {max_start} for {path.stem}")
                target_start = self.fixed_crop_start
            else:
                target_start = (
                    int.from_bytes(hashlib.sha256(path.stem.encode("utf-8")).digest()[8:16], "big") % (max_start + 1)
                    if self.deterministic_crops
                    else random.randint(0, max_start)
                )
            target_end = target_start + frame_count
            sequence_start = max(0, target_start - self.target_context_frames)
            loss_start = target_start - sequence_start
            query_positions = torch.arange(sequence_start, target_end, dtype=torch.float32)
            if target_length > 1 and reference_length > 1:
                query_positions *= (reference_length - 1) / (target_length - 1)
            reference_start = max(
                0,
                math.floor(query_positions[0].item()) - self.reference_context_frames,
            )
            reference_end = min(
                reference_length,
                math.ceil(query_positions[-1].item()) + self.reference_context_frames + 1,
            )
            target_codes = handle.get_slice("target_codes")[sequence_start:target_end].long()
            reference_codes = handle.get_slice("reference_codes")[reference_start:reference_end].long()
        return {
            "clip_id": metadata["clip_id"],
            "metadata": metadata,
            "prompt": metadata["prompt"],
            "lyrics": metadata["lyrics"],
            "target_codes": target_codes,
            "loss_start": loss_start,
            "reference_codes": reference_codes,
            "query_positions": query_positions,
            "key_positions": torch.arange(reference_start, reference_end, dtype=torch.float32),
        }


def parse_layer_indices(value: str) -> tuple[int, ...]:
    try:
        indices = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layer indices must be comma-separated integers") from exc
    if not indices:
        raise argparse.ArgumentTypeError("at least one layer index is required")
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MiniMax Music 3 aligned reference control")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--model-id", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-frames", type=int, default=256)
    parser.add_argument("--target-context-frames", type=int, default=0)
    parser.add_argument("--control-dim", type=int, default=512)
    parser.add_argument("--control-heads", type=int, default=8)
    parser.add_argument("--control-layers", type=parse_layer_indices, default=(5, 11, 17, 23, 29, 35))
    parser.add_argument("--reference-window-frames", type=int, default=100)
    parser.add_argument("--reference-dropout", type=float, default=0.1)
    parser.add_argument("--feedback-corruption-rate", type=float, default=0.0)
    parser.add_argument("--feedback-sampling-top-k", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--oft-learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--overfit-clip-id")
    parser.add_argument("--overfit-start-frame", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--disable-oftv2", action="store_true")
    parser.add_argument("--oft-block-size", type=int, default=64)
    parser.add_argument("--wandb-project")
    return parser.parse_args()


def tokenize_prompt(tokenizer, prompt: str, lyrics: str, device: torch.device) -> torch.Tensor:
    text = PROMPT_TEMPLATE.format(caption=_clean_caption(prompt), lyrics=_normalize_lyrics(lyrics))
    return tokenizer(text, return_tensors="pt")["input_ids"].to(device)


@torch.no_grad()
def greedy_warmup_codes(language_model, depth_decoder, text_ids: torch.Tensor) -> torch.Tensor:
    text_embeds = language_model.model.embed_tokens(text_ids)
    hidden = language_model.model(inputs_embeds=text_embeds, use_cache=False).last_hidden_state[:, -1]
    semantic_logits = F.linear(
        hidden,
        language_model.lm_head.weight[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + AUDIO_VOCAB_SIZE],
    )
    semantic = semantic_logits.argmax(dim=-1)
    sequence = [depth_decoder.projection(hidden).unsqueeze(1)]
    semantic_embed = language_model.model.embed_tokens(semantic + AUDIO_CODE_OFFSET)
    sequence.append(depth_decoder.projection(semantic_embed).unsqueeze(1))
    codes = [semantic]
    for codebook in range(1, depth_decoder.config.num_codebooks):
        depth_hidden = depth_decoder(torch.cat(sequence, dim=1))[:, -1]
        code = depth_decoder.audio_heads[codebook - 1](depth_hidden).argmax(dim=-1)
        codes.append(code)
        if codebook < depth_decoder.config.num_codebooks - 1:
            embedding = depth_decoder.audio_embeddings(code + (codebook - 1) * depth_decoder.config.audio_vocab_size)
            sequence.append(depth_decoder.projection(embedding).unsqueeze(1))
    return torch.stack(codes, dim=-1).unsqueeze(1)


def depth_losses(
    depth_decoder,
    language_model,
    hidden_states: torch.Tensor,
    target_codes: torch.Tensor,
    *,
    checkpoint_decoder: bool = False,
) -> list:
    batch_size, frame_count, hidden_size = hidden_states.shape
    hidden = hidden_states.reshape(batch_size * frame_count, hidden_size)
    codes = target_codes.reshape(batch_size * frame_count, -1)
    sequence = [depth_decoder.projection(hidden).unsqueeze(1)]
    semantic_embed = language_model.model.embed_tokens(codes[:, 0] + AUDIO_CODE_OFFSET)
    sequence.append(depth_decoder.projection(semantic_embed).unsqueeze(1))
    for codebook in range(1, depth_decoder.config.num_codebooks - 1):
        embedding = depth_decoder.audio_embeddings(
            codes[:, codebook] + (codebook - 1) * depth_decoder.config.audio_vocab_size
        )
        sequence.append(depth_decoder.projection(embedding).unsqueeze(1))
    depth_inputs = torch.cat(sequence, dim=1)
    depth_hidden = (
        checkpoint(depth_decoder, depth_inputs, use_reentrant=False) if checkpoint_decoder else depth_decoder(depth_inputs)
    )
    return [
        F.cross_entropy(
            depth_decoder.audio_heads[codebook - 1](depth_hidden[:, codebook]).float(),
            codes[:, codebook],
        )
        for codebook in range(1, depth_decoder.config.num_codebooks)
    ]


@torch.no_grad()
def sample_codes_from_hidden(
    depth_decoder,
    language_model,
    hidden_states: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    batch_size, frame_count, hidden_size = hidden_states.shape
    hidden = hidden_states.reshape(batch_size * frame_count, hidden_size)
    semantic_logits = F.linear(
        hidden,
        language_model.lm_head.weight[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + AUDIO_VOCAB_SIZE],
    ).float()

    def sample(logits: torch.Tensor) -> torch.Tensor:
        values, indices = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
        selected = torch.multinomial(values.softmax(dim=-1), 1)
        return indices.gather(-1, selected).squeeze(-1)

    semantic = sample(semantic_logits)
    sequence = [depth_decoder.projection(hidden).unsqueeze(1)]
    semantic_embed = language_model.model.embed_tokens(semantic + AUDIO_CODE_OFFSET)
    sequence.append(depth_decoder.projection(semantic_embed).unsqueeze(1))
    depth_hidden, past_key_values = depth_decoder.forward_with_cache(torch.cat(sequence, dim=1))
    codes = [semantic]
    for codebook in range(1, depth_decoder.config.num_codebooks):
        code = sample(depth_decoder.audio_heads[codebook - 1](depth_hidden[:, -1]).float())
        codes.append(code)
        if codebook < depth_decoder.config.num_codebooks - 1:
            embedding = depth_decoder.audio_embeddings(code + (codebook - 1) * depth_decoder.config.audio_vocab_size)
            depth_hidden, past_key_values = depth_decoder.forward_with_cache(
                depth_decoder.projection(embedding).unsqueeze(1),
                past_key_values,
            )
    return torch.stack(codes, dim=-1).reshape(batch_size, frame_count, -1)


def splice_sampled_feedback(
    clean_feedback: torch.Tensor,
    sampled_codes: torch.Tensor,
    *,
    loss_start: int,
    corruption_rate: float,
) -> tuple[torch.Tensor, float]:
    if clean_feedback.shape != sampled_codes.shape:
        raise ValueError("clean_feedback and sampled_codes must have the same shape")
    if not 0.0 <= corruption_rate <= 1.0:
        raise ValueError("corruption_rate must be in [0, 1]")
    if not 0 <= loss_start < clean_feedback.shape[1]:
        raise ValueError("loss_start must index clean_feedback")
    first_corruptible = loss_start + 1
    if corruption_rate == 0.0 or first_corruptible >= clean_feedback.shape[1]:
        return clean_feedback, 0.0
    mask = torch.zeros(clean_feedback.shape[:2], dtype=torch.bool, device=clean_feedback.device)
    mask[:, first_corruptible:] = (
        torch.rand(mask[:, first_corruptible:].shape, device=clean_feedback.device) < corruption_rate
    )
    sampled_feedback = torch.cat((clean_feedback[:, :1], sampled_codes[:, :-1]), dim=1)
    feedback = torch.where(mask.unsqueeze(-1), sampled_feedback, clean_feedback)
    return feedback, mask[:, first_corruptible:].float().mean().item()


def conditioned_hidden_states(
    language_model,
    depth_decoder,
    text_embeds: torch.Tensor,
    feedback_codes: torch.Tensor,
    reference_memory: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
) -> torch.Tensor:
    feedback_embeds = embed_rvq_frames(language_model, depth_decoder, feedback_codes)
    inputs_embeds = torch.cat((text_embeds, feedback_embeds), dim=1)
    query_start = text_embeds.shape[1]
    output = language_model.model(
        inputs_embeds=inputs_embeds,
        use_cache=False,
        reference_memory=reference_memory,
        reference_query_positions=query_positions,
        reference_key_positions=key_positions,
        reference_query_start=query_start,
    )
    return output.last_hidden_state[:, query_start : query_start + feedback_codes.shape[1]]


def train_step(
    language_model,
    depth_decoder,
    adapter,
    tokenizer,
    sample: dict,
    device: torch.device,
    reference_dropout: float,
    feedback_corruption_rate: float = 0.0,
    feedback_sampling_top_k: int = 50,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_codes = sample["target_codes"].to(device).unsqueeze(0)
    loss_start = int(sample.get("loss_start", 0))
    reference_codes = sample["reference_codes"].to(device).unsqueeze(0)
    query_positions = sample["query_positions"].to(device).unsqueeze(0)
    key_positions = sample["key_positions"].to(device).unsqueeze(0)
    text_ids = tokenize_prompt(tokenizer, sample["prompt"], sample["lyrics"], device)
    with torch.no_grad():
        warmup_codes = greedy_warmup_codes(language_model, depth_decoder, text_ids)
        feedback_codes = torch.cat((warmup_codes, target_codes[:, :-1]), dim=1)
        text_embeds = language_model.model.embed_tokens(text_ids)
        reference_embeds = embed_rvq_frames(language_model, depth_decoder, reference_codes)
    reference_memory = adapter.encode_reference(reference_embeds)
    if torch.rand((), device=device).item() < reference_dropout:
        reference_memory = reference_memory * 0.0
    corruption_fraction = 0.0
    if feedback_corruption_rate > 0.0:
        with torch.no_grad():
            clean_hidden = conditioned_hidden_states(
                language_model,
                depth_decoder,
                text_embeds,
                feedback_codes,
                reference_memory,
                query_positions,
                key_positions,
            )
            sampled_codes = sample_codes_from_hidden(
                depth_decoder,
                language_model,
                clean_hidden,
                feedback_sampling_top_k,
            )
            feedback_codes, corruption_fraction = splice_sampled_feedback(
                feedback_codes,
                sampled_codes,
                loss_start=loss_start,
                corruption_rate=feedback_corruption_rate,
            )
    hidden = conditioned_hidden_states(
        language_model,
        depth_decoder,
        text_embeds,
        feedback_codes,
        reference_memory,
        query_positions,
        key_positions,
    )
    hidden = hidden[:, loss_start:]
    target_codes = target_codes[:, loss_start:]
    semantic_logits = F.linear(
        hidden,
        language_model.lm_head.weight[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + AUDIO_VOCAB_SIZE],
    ).float()
    semantic_loss = F.cross_entropy(semantic_logits.flatten(0, 1), target_codes[..., 0].flatten())
    residual_losses = depth_losses(depth_decoder, language_model, hidden, target_codes)
    head_losses = [semantic_loss, *residual_losses]
    loss = torch.stack(head_losses).mean()
    with torch.no_grad():
        metrics = {
            "loss": loss.item(),
            "semantic_loss": semantic_loss.item(),
            "semantic_top1": (semantic_logits.argmax(dim=-1) == target_codes[..., 0]).float().mean().item(),
            "feedback_corruption_fraction": corruption_fraction,
        }
        metrics.update({f"codebook_{index}_loss": value.item() for index, value in enumerate(head_losses)})
    return loss, metrics


def save_checkpoint(output_dir: Path, step: int, adapter, oft_network, optimizer, scheduler) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(adapter.state_dict(), str(checkpoint_dir / "reference_control.safetensors"))
    (checkpoint_dir / "reference_control.json").write_text(
        json.dumps(adapter.config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    if oft_network is not None:
        oft_network.save_weights(checkpoint_dir / "qwen_oftv2.safetensors", torch.float32, {})
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        checkpoint_dir / "training_state.pt",
    )


def load_initial_weights(checkpoint_dir: Path, adapter, oft_network) -> None:
    config_path = checkpoint_dir / "reference_control.json"
    adapter_path = checkpoint_dir / "reference_control.safetensors"
    for path in (config_path, adapter_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_config = ReferenceControlConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    if checkpoint_config != adapter.config:
        raise ValueError("init-checkpoint reference-control topology does not match the requested topology")
    adapter.load_state_dict(load_file(str(adapter_path)), strict=True)
    oft_path = checkpoint_dir / "qwen_oftv2.safetensors"
    if oft_network is None:
        if oft_path.exists():
            raise ValueError("init-checkpoint contains OFTv2 weights but --disable-oftv2 was requested")
    else:
        if not oft_path.is_file():
            raise FileNotFoundError(oft_path)
        oft_network.load_weights(str(oft_path))


@torch.no_grad()
def evaluate(
    language_model,
    depth_decoder,
    adapter,
    tokenizer,
    loader,
    device: torch.device,
    max_samples: int,
) -> dict[str, float]:
    language_model.eval()
    adapter.eval()
    reference_totals = {"loss": 0.0, "semantic_top1": 0.0}
    null_totals = {"loss": 0.0, "semantic_top1": 0.0}
    count = 0
    for sample in loader:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, reference_metrics = train_step(
                language_model,
                depth_decoder,
                adapter,
                tokenizer,
                sample,
                device,
                reference_dropout=0.0,
            )
            _, null_metrics = train_step(
                language_model,
                depth_decoder,
                adapter,
                tokenizer,
                sample,
                device,
                reference_dropout=1.0,
            )
        for key in reference_totals:
            reference_totals[key] += reference_metrics[key]
            null_totals[key] += null_metrics[key]
        count += 1
        if count >= max_samples:
            break
    language_model.train()
    adapter.train()
    reference_loss = reference_totals["loss"] / count
    null_loss = null_totals["loss"] / count
    reference_top1 = reference_totals["semantic_top1"] / count
    null_top1 = null_totals["semantic_top1"] / count
    return {
        "validation/loss": reference_loss,
        "validation/null_loss": null_loss,
        "validation/reference_gain": null_loss - reference_loss,
        "validation/semantic_top1": reference_top1,
        "validation/null_semantic_top1": null_top1,
        "validation/semantic_top1_gain": reference_top1 - null_top1,
        "validation/samples": count,
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.reference_dropout < 1.0:
        raise ValueError("reference-dropout must be in [0, 1)")
    if not 0.0 <= args.feedback_corruption_rate <= 1.0:
        raise ValueError("feedback-corruption-rate must be in [0, 1]")
    if not 1 <= args.feedback_sampling_top_k <= 1024:
        raise ValueError("feedback-sampling-top-k must be between 1 and 1024")
    if args.target_context_frames < 0:
        raise ValueError("target-context-frames must be non-negative")
    if args.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation must be positive")
    if args.overfit_start_frame < 0:
        raise ValueError("overfit-start-frame must be non-negative")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    dataset = CachedStylePairDataset(
        args.cache_dir,
        args.crop_frames,
        args.reference_window_frames,
        split="train",
        validation_fraction=args.validation_fraction,
        clip_id=args.overfit_clip_id,
        fixed_crop_start=args.overfit_start_frame if args.overfit_clip_id is not None else None,
        target_context_frames=args.target_context_frames,
    )
    validation_dataset = CachedStylePairDataset(
        args.cache_dir,
        args.crop_frames,
        args.reference_window_frames,
        split="validation",
        validation_fraction=args.validation_fraction,
        clip_id=args.overfit_clip_id,
        fixed_crop_start=args.overfit_start_frame if args.overfit_clip_id is not None else None,
        target_context_frames=args.target_context_frames,
    )
    loader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=args.num_workers)
    validation_loader = DataLoader(validation_dataset, batch_size=None, shuffle=False, num_workers=0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, subfolder="tokenizer")
    language_model = Qwen3ForCausalLM.from_pretrained(
        args.model_id,
        subfolder="language_model",
        dtype=torch.bfloat16,
    ).to(device)
    depth_decoder = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
        args.model_id,
        subfolder="rvq_depth_decoder",
        torch_dtype=torch.bfloat16,
    ).to(device)
    language_model.requires_grad_(False)
    depth_decoder.requires_grad_(False)
    language_model.train()
    depth_decoder.eval()
    oft_network = None
    if not args.disable_oftv2:
        oft_network = create_qwen_oftv2_adapter(language_model, block_size=args.oft_block_size).to(device)
    control_config = ReferenceControlConfig(
        hidden_size=language_model.config.hidden_size,
        control_dim=args.control_dim,
        num_heads=args.control_heads,
        layer_indices=args.control_layers,
        window_frames=args.reference_window_frames,
    )
    adapter = MiniMaxMusic3ReferenceControlAdapter(control_config).to(device)
    adapter.install(language_model)
    if args.init_checkpoint is not None:
        load_initial_weights(args.init_checkpoint, adapter, oft_network)
    language_model.gradient_checkpointing_enable()
    parameter_groups = [
        {"params": list(adapter.parameters()), "lr": args.learning_rate},
    ]
    if oft_network is not None:
        parameter_groups.append({"params": list(oft_network.parameters()), "lr": args.oft_learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        power=1.0,
    )
    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    step = 0
    micro_step = 0
    started = time.perf_counter()
    baseline_metrics = evaluate(
        language_model,
        depth_decoder,
        adapter,
        tokenizer,
        validation_loader,
        device,
        args.eval_samples,
    )
    baseline_metrics["step"] = 0
    print(json.dumps(baseline_metrics), flush=True)
    if wandb_run is not None:
        wandb_run.log(baseline_metrics, step=0)
    while step < args.max_steps:
        for sample in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss, metrics = train_step(
                    language_model,
                    depth_decoder,
                    adapter,
                    tokenizer,
                    sample,
                    device,
                    args.reference_dropout,
                    args.feedback_corruption_rate,
                    args.feedback_sampling_top_k,
                )
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation:
                continue
            parameters = [parameter for group in parameter_groups for parameter in group["params"]]
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            metrics.update(
                {
                    "step": step,
                    "grad_norm": float(grad_norm),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "steps_per_second": step / (time.perf_counter() - started),
                }
            )
            if device.type == "cuda":
                metrics["peak_vram_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
                metrics["peak_reserved_vram_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
            print(json.dumps(metrics), flush=True)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            if step % args.eval_every == 0 or step == args.max_steps:
                evaluation_metrics = evaluate(
                    language_model,
                    depth_decoder,
                    adapter,
                    tokenizer,
                    validation_loader,
                    device,
                    args.eval_samples,
                )
                evaluation_metrics["step"] = step
                print(json.dumps(evaluation_metrics), flush=True)
                if wandb_run is not None:
                    wandb_run.log(evaluation_metrics, step=step)
            if step % args.save_every == 0 or step == args.max_steps:
                save_checkpoint(args.output_dir, step, adapter, oft_network, optimizer, scheduler)
            if step >= args.max_steps:
                break
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
