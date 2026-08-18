#!/usr/bin/env python3
# Copyright 2026 SimpleTuner contributors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, Qwen3ForCausalLM, get_polynomial_decay_schedule_with_warmup

from scripts.minimax_music3.train_reference_control import (
    AUDIO_CODE_OFFSET,
    AUDIO_VOCAB_SIZE,
    CachedStylePairDataset,
    depth_losses,
    parse_layer_indices,
    sample_codes_from_hidden,
    splice_sampled_feedback,
    tokenize_prompt,
)
from simpletuner.helpers.models.minimaxmusic.reference_control import (
    ControlLoRAConfig,
    MiniMaxMusic3ControlLoRAAdapter,
    create_qwen_lokr_adapter,
    embed_rvq_frames,
)
from simpletuner.helpers.models.minimaxmusic.rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder

CHECKPOINT_FORMAT = "minimax-music3-control-lora-v1"
CONTROL_INPUT_MODES = ("additive-hint", "reference-delta")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MiniMax Music 3 shared-block ControlLoRA")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--model-id", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-frames", type=int, default=50)
    parser.add_argument("--overfit-clip-id")
    parser.add_argument("--overfit-ids-csv", type=Path)
    parser.add_argument("--overfit-start-frame", type=int, default=0)
    parser.add_argument("--random-crops", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--control-layers", type=parse_layer_indices, default=tuple(range(36)))
    parser.add_argument("--residual-rank", type=int, default=16)
    parser.add_argument("--lokr-rank", type=int, default=16)
    parser.add_argument("--lokr-alpha", type=float, default=16.0)
    parser.add_argument("--hint-scale", type=float, default=1.0)
    parser.add_argument("--control-input-mode", choices=CONTROL_INPUT_MODES, default="reference-delta")
    parser.add_argument("--reference-dropout", type=float, default=0.1)
    parser.add_argument("--feedback-corruption-rate", type=float, default=0.5)
    parser.add_argument("--feedback-sampling-top-k", type=int, default=1)
    parser.add_argument("--semantic-loss-weight", type=float, default=16.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lokr-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-depth-decoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb-project")
    return parser.parse_args()


@dataclass(frozen=True)
class DistributedContext:
    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def initialize_distributed(device_name: str) -> DistributedContext:
    distributed_keys = {name: os.environ.get(name) for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE")}
    configured = {name for name, value in distributed_keys.items() if value is not None}
    if configured and len(configured) != len(distributed_keys):
        raise ValueError("distributed launch requires RANK, LOCAL_RANK, and WORLD_SIZE")
    if not configured:
        return DistributedContext(device=torch.device(device_name))
    if not torch.cuda.is_available():
        raise RuntimeError("ControlLoRA distributed training requires CUDA")
    rank = int(distributed_keys["RANK"])
    local_rank = int(distributed_keys["LOCAL_RANK"])
    world_size = int(distributed_keys["WORLD_SIZE"])
    if world_size < 2:
        raise ValueError("torchrun WORLD_SIZE must be at least 2")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return DistributedContext(torch.device("cuda", local_rank), rank, local_rank, world_size)


def reduce_metrics(metrics: dict[str, float], context: DistributedContext) -> dict[str, float]:
    if not context.enabled:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], dtype=torch.float64, device=context.device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values /= context.world_size
    return dict(zip(keys, values.tolist(), strict=True))


def load_clip_ids_csv(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "clip_id" not in reader.fieldnames:
            raise ValueError("overfit IDs CSV must contain a clip_id column")
        clip_ids = tuple(row["clip_id"].strip() for row in reader if row["clip_id"].strip())
    if not clip_ids or len(set(clip_ids)) != len(clip_ids):
        raise ValueError("overfit IDs CSV must contain non-empty unique clip IDs")
    return clip_ids


def collate_control_samples(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("cannot collate an empty ControlLoRA batch")
    target_shapes = {tuple(sample["target_codes"].shape) for sample in samples}
    query_shapes = {tuple(sample["query_positions"].shape) for sample in samples}
    loss_starts = {int(sample.get("loss_start", 0)) for sample in samples}
    if len(target_shapes) != 1 or len(query_shapes) != 1 or len(loss_starts) != 1:
        raise ValueError("ControlLoRA batches require equal target windows and loss starts")
    max_reference_frames = max(sample["reference_codes"].shape[0] for sample in samples)
    reference_codes = []
    reference_attention_mask = []
    key_positions = []
    for sample in samples:
        pad_frames = max_reference_frames - sample["reference_codes"].shape[0]
        reference_codes.append(
            torch.cat((sample["reference_codes"], sample["reference_codes"][-1:].expand(pad_frames, -1)), dim=0)
        )
        reference_attention_mask.append(
            torch.cat(
                (
                    torch.ones(sample["reference_codes"].shape[0], dtype=torch.long),
                    torch.zeros(pad_frames, dtype=torch.long),
                )
            )
        )
        key_positions.append(torch.cat((sample["key_positions"], sample["key_positions"][-1:].expand(pad_frames)), dim=0))
    return {
        "clip_id": [sample["clip_id"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
        "prompt": [sample["prompt"] for sample in samples],
        "lyrics": [sample["lyrics"] for sample in samples],
        "target_codes": torch.stack([sample["target_codes"] for sample in samples]),
        "loss_start": loss_starts.pop(),
        "reference_codes": torch.stack(reference_codes),
        "reference_attention_mask": torch.stack(reference_attention_mask),
        "query_positions": torch.stack([sample["query_positions"] for sample in samples]),
        "key_positions": torch.stack(key_positions),
    }


def batched_text_embeddings(language_model, tokenizer, prompts, lyrics, device: torch.device):
    prompts = [prompts] if isinstance(prompts, str) else prompts
    lyrics = [lyrics] if isinstance(lyrics, str) else lyrics
    if len(prompts) != len(lyrics):
        raise ValueError("prompt and lyrics batch sizes differ")
    sequences = [tokenize_prompt(tokenizer, prompt, lyric, device)[0] for prompt, lyric in zip(prompts, lyrics)]
    if tokenizer.pad_token_id is None:
        raise ValueError("ControlLoRA batching requires a tokenizer pad token")
    max_length = max(sequence.shape[0] for sequence in sequences)
    text_ids = torch.full(
        (len(sequences), max_length),
        tokenizer.pad_token_id,
        dtype=sequences[0].dtype,
        device=device,
    )
    attention_mask = torch.zeros((len(sequences), max_length), dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        text_ids[index, -sequence.shape[0] :] = sequence
        attention_mask[index, -sequence.shape[0] :] = 1
    return language_model.model.embed_tokens(text_ids), attention_mask


def aligned_reference_hint(
    reference_embeddings: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
) -> torch.Tensor:
    if reference_embeddings.ndim != 3:
        raise ValueError("reference_embeddings must have shape [batch, frames, hidden_size]")
    if query_positions.ndim != 2 or key_positions.ndim != 2:
        raise ValueError("query_positions and key_positions must have shape [batch, frames]")
    if reference_embeddings.shape[:2] != key_positions.shape:
        raise ValueError("reference embeddings and key positions do not share a timeline")
    if query_positions.shape[0] != reference_embeddings.shape[0]:
        raise ValueError("query and reference batch sizes differ")
    if not bool((key_positions[:, 1:] >= key_positions[:, :-1]).all()):
        raise ValueError("key positions must be monotonic")

    right = torch.searchsorted(key_positions.contiguous(), query_positions.contiguous(), right=False)
    right = right.clamp(max=key_positions.shape[1] - 1)
    left = (right - 1).clamp(min=0)
    left_position = key_positions.gather(1, left)
    right_position = key_positions.gather(1, right)
    denominator = right_position - left_position
    weight = torch.where(
        denominator > 0,
        (query_positions - left_position) / denominator,
        torch.zeros_like(query_positions),
    ).to(reference_embeddings.dtype)
    hidden_size = reference_embeddings.shape[-1]
    left_values = reference_embeddings.gather(1, left.unsqueeze(-1).expand(-1, -1, hidden_size))
    right_values = reference_embeddings.gather(1, right.unsqueeze(-1).expand(-1, -1, hidden_size))
    return torch.lerp(left_values, right_values, weight.unsqueeze(-1))


def reference_warmup_codes(
    reference_codes: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
) -> torch.Tensor:
    distance = (key_positions - query_positions[:, :1]).abs()
    indices = distance.argmin(dim=1)
    batch_indices = torch.arange(reference_codes.shape[0], device=reference_codes.device)
    return reference_codes[batch_indices, indices].unsqueeze(1)


def control_lora_hidden_states(
    language_model,
    depth_decoder,
    adapter,
    lokr_network,
    text_embeddings: torch.Tensor,
    feedback_codes: torch.Tensor,
    reference_codes: torch.Tensor,
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    *,
    hint_scale: float,
    null_reference: bool,
    control_input_mode: str = "additive-hint",
    text_attention_mask: torch.Tensor | None = None,
    reference_attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if control_input_mode not in CONTROL_INPUT_MODES:
        raise ValueError(f"unsupported control input mode: {control_input_mode}")
    feedback_embeddings = embed_rvq_frames(language_model, depth_decoder, feedback_codes)
    reference_embeddings = embed_rvq_frames(language_model, depth_decoder, reference_codes)
    main_inputs = torch.cat((text_embeddings, feedback_embeddings), dim=1)
    query_start = text_embeddings.shape[1]
    main_attention_mask = None
    if text_attention_mask is not None:
        if text_attention_mask.shape != text_embeddings.shape[:2]:
            raise ValueError("text attention mask does not match text embeddings")
        main_attention_mask = torch.cat(
            (
                text_attention_mask,
                torch.ones(
                    feedback_embeddings.shape[:2],
                    dtype=text_attention_mask.dtype,
                    device=text_attention_mask.device,
                ),
            ),
            dim=1,
        )

    lokr_network.set_multiplier(1.0)
    if control_input_mode == "additive-hint":
        hint = aligned_reference_hint(reference_embeddings, query_positions, key_positions)
        if null_reference:
            hint = hint * 0.0
        control_inputs = torch.cat((text_embeddings, feedback_embeddings + hint_scale * hint), dim=1)
        control_output = language_model.model(
            inputs_embeds=control_inputs,
            attention_mask=main_attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        control_hidden_states = control_output.hidden_states[1:]
        control_scale = 1.0
    else:
        if reference_attention_mask is None:
            reference_attention_mask = torch.ones(
                reference_embeddings.shape[:2],
                dtype=torch.long,
                device=reference_embeddings.device,
            )
        if reference_attention_mask.shape != reference_embeddings.shape[:2]:
            raise ValueError("reference attention mask does not match reference embeddings")
        conditioned_reference = reference_embeddings * (0.0 if null_reference else 1.0)
        null_reference_embeddings = torch.zeros_like(reference_embeddings)
        conditioned_inputs = torch.cat((text_embeddings, conditioned_reference), dim=1)
        null_inputs = torch.cat((text_embeddings, null_reference_embeddings), dim=1)
        control_inputs = torch.cat((conditioned_inputs, null_inputs), dim=0)
        control_attention_mask = None
        if text_attention_mask is not None:
            reference_sequence_mask = torch.cat((text_attention_mask, reference_attention_mask), dim=1)
            control_attention_mask = torch.cat((reference_sequence_mask, reference_sequence_mask), dim=0)
        control_output = language_model.model(
            inputs_embeds=control_inputs,
            attention_mask=control_attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        control_hidden_states = []
        for layer_hidden_states in control_output.hidden_states[1:]:
            conditioned_hidden, null_hidden = layer_hidden_states.chunk(2, dim=0)
            reference_delta = conditioned_hidden[:, query_start:] - null_hidden[:, query_start:]
            control_hidden_states.append(aligned_reference_hint(reference_delta, query_positions, key_positions))
        control_hidden_states = tuple(control_hidden_states)
        control_scale = hint_scale
    if len(control_hidden_states) != len(language_model.model.layers):
        raise RuntimeError("control pass did not return one hidden state per Qwen block")
    lokr_network.set_multiplier(0.0)
    try:
        output = language_model.model(
            inputs_embeds=main_inputs,
            attention_mask=main_attention_mask,
            use_cache=False,
            control_hidden_states=control_hidden_states,
            control_query_start=query_start,
            control_scale=control_scale,
        )
    finally:
        lokr_network.set_multiplier(1.0)
    return output.last_hidden_state[:, query_start:]


class ControlLoRATrainingModel(nn.Module):
    def __init__(self, language_model, depth_decoder, adapter, lokr_network):
        super().__init__()
        self.language_model = language_model
        self.depth_decoder = depth_decoder
        self.adapter = adapter
        self.lokr_network = lokr_network

    def forward(
        self,
        text_embeddings: torch.Tensor,
        feedback_codes: torch.Tensor,
        reference_codes: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        *,
        hint_scale: float,
        null_reference: bool,
        control_input_mode: str,
        text_attention_mask: torch.Tensor | None,
        reference_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return control_lora_hidden_states(
            self.language_model,
            self.depth_decoder,
            self.adapter,
            self.lokr_network,
            text_embeddings,
            feedback_codes,
            reference_codes,
            query_positions,
            key_positions,
            hint_scale=hint_scale,
            null_reference=null_reference,
            control_input_mode=control_input_mode,
            text_attention_mask=text_attention_mask,
            reference_attention_mask=reference_attention_mask,
        )


def unwrap_training_model(training_model) -> ControlLoRATrainingModel:
    return training_model.module if isinstance(training_model, DistributedDataParallel) else training_model


def train_step(
    training_model,
    tokenizer,
    sample: dict,
    device: torch.device,
    args,
    *,
    force_null_reference: bool = False,
    disable_corruption: bool = False,
    disable_reference_dropout: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    components = unwrap_training_model(training_model)
    language_model = components.language_model
    depth_decoder = components.depth_decoder
    target_codes = sample["target_codes"].to(device)
    reference_codes = sample["reference_codes"].to(device)
    reference_attention_mask = sample.get("reference_attention_mask")
    if reference_attention_mask is not None:
        reference_attention_mask = reference_attention_mask.to(device)
    query_positions = sample["query_positions"].to(device)
    key_positions = sample["key_positions"].to(device)
    if target_codes.ndim == 2:
        target_codes = target_codes.unsqueeze(0)
        reference_codes = reference_codes.unsqueeze(0)
        if reference_attention_mask is not None:
            reference_attention_mask = reference_attention_mask.unsqueeze(0)
        query_positions = query_positions.unsqueeze(0)
        key_positions = key_positions.unsqueeze(0)
    if reference_attention_mask is None:
        reference_attention_mask = torch.ones(reference_codes.shape[:2], dtype=torch.long, device=device)
    loss_start = int(sample.get("loss_start", 0))
    with torch.no_grad():
        text_embeddings, text_attention_mask = batched_text_embeddings(
            language_model,
            tokenizer,
            sample["prompt"],
            sample["lyrics"],
            device,
        )
        warmup_codes = reference_warmup_codes(reference_codes, query_positions, key_positions)
        clean_feedback = torch.cat((warmup_codes, target_codes[:, :-1]), dim=1)
    null_reference = force_null_reference or (
        not disable_reference_dropout and torch.rand((), device=device).item() < args.reference_dropout
    )
    feedback_codes = clean_feedback
    corruption_fraction = 0.0
    if args.feedback_corruption_rate > 0.0 and not disable_corruption:
        with torch.no_grad():
            clean_hidden = training_model(
                text_embeddings,
                clean_feedback,
                reference_codes,
                query_positions,
                key_positions,
                hint_scale=args.hint_scale,
                null_reference=null_reference,
                control_input_mode=args.control_input_mode,
                text_attention_mask=text_attention_mask,
                reference_attention_mask=reference_attention_mask,
            )
            sampled_codes = sample_codes_from_hidden(
                depth_decoder,
                language_model,
                clean_hidden,
                args.feedback_sampling_top_k,
            )
            feedback_codes, corruption_fraction = splice_sampled_feedback(
                clean_feedback,
                sampled_codes,
                loss_start=loss_start,
                corruption_rate=args.feedback_corruption_rate,
            )
    hidden = training_model(
        text_embeddings,
        feedback_codes,
        reference_codes,
        query_positions,
        key_positions,
        hint_scale=args.hint_scale,
        null_reference=null_reference,
        control_input_mode=args.control_input_mode,
        text_attention_mask=text_attention_mask,
        reference_attention_mask=reference_attention_mask,
    )
    hidden = hidden[:, loss_start:]
    target_codes = target_codes[:, loss_start:]
    semantic_logits = F.linear(
        hidden,
        language_model.lm_head.weight[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + AUDIO_VOCAB_SIZE],
    ).float()
    semantic_loss = F.cross_entropy(semantic_logits.flatten(0, 1), target_codes[..., 0].flatten())
    acoustic_losses = depth_losses(
        depth_decoder,
        language_model,
        hidden,
        target_codes,
        checkpoint_decoder=args.checkpoint_depth_decoder and torch.is_grad_enabled(),
    )
    loss = (args.semantic_loss_weight * semantic_loss + torch.stack(acoustic_losses).sum()) / (
        args.semantic_loss_weight + len(acoustic_losses)
    )
    with torch.no_grad():
        metrics = {
            "loss": loss.item(),
            "semantic_loss": semantic_loss.item(),
            "semantic_top1": (semantic_logits.argmax(dim=-1) == target_codes[..., 0]).float().mean().item(),
            "semantic_first_top1": (semantic_logits[:, 0].argmax(dim=-1) == target_codes[:, 0, 0]).float().mean().item(),
            "feedback_corruption_fraction": corruption_fraction,
            "null_reference": float(null_reference),
        }
        metrics.update(
            {f"codebook_{index}_loss": value.item() for index, value in enumerate((semantic_loss, *acoustic_losses))}
        )
    return loss, metrics


@torch.no_grad()
def evaluate(training_model, tokenizer, sample, device, args) -> dict:
    components = unwrap_training_model(training_model)
    language_model = components.language_model
    adapter = components.adapter
    was_training = language_model.training
    language_model.eval()
    adapter.eval()
    values = {}
    for name, null_reference in (("reference", False), ("null", True)):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, values[name] = train_step(
                training_model,
                tokenizer,
                sample,
                device,
                args,
                force_null_reference=null_reference,
                disable_corruption=True,
                disable_reference_dropout=True,
            )
    language_model.train(was_training)
    adapter.train(was_training)
    return {
        "validation/reference_loss": values["reference"]["loss"],
        "validation/null_loss": values["null"]["loss"],
        "validation/reference_semantic_top1": values["reference"]["semantic_top1"],
        "validation/null_semantic_top1": values["null"]["semantic_top1"],
        "validation/reference_semantic_first_top1": values["reference"]["semantic_first_top1"],
        "validation/null_semantic_first_top1": values["null"]["semantic_first_top1"],
    }


def save_checkpoint(output_dir: Path, step: int, adapter, lokr_network, optimizer, scheduler, args) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(adapter.state_dict(), checkpoint_dir / "control_lora.safetensors")
    lokr_network.save_weights(checkpoint_dir / "qwen_control_lokr.safetensors", torch.float32, {})
    config = {
        "format": CHECKPOINT_FORMAT,
        "adapter": adapter.config.to_dict(),
        "lokr_rank": args.lokr_rank,
        "lokr_alpha": args.lokr_alpha,
        "crop_frames": args.crop_frames,
        "hint_scale": args.hint_scale,
        "control_input_mode": args.control_input_mode,
        "reference_dropout": args.reference_dropout,
        "feedback_corruption_rate": args.feedback_corruption_rate,
        "feedback_sampling_top_k": args.feedback_sampling_top_k,
        "semantic_loss_weight": args.semantic_loss_weight,
        "batch_size": args.batch_size,
        "global_batch_size": args.batch_size * args.world_size * args.gradient_accumulation,
        "world_size": args.world_size,
        "gradient_accumulation": args.gradient_accumulation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "checkpoint_depth_decoder": args.checkpoint_depth_decoder,
        "random_crops": args.random_crops,
        "training_clip_ids": list(args.training_clip_ids),
    }
    (checkpoint_dir / "control_lora.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        checkpoint_dir / "training_state.pt",
    )


def load_checkpoint(checkpoint_dir: Path, adapter, lokr_network, args) -> None:
    config_path = checkpoint_dir / "control_lora.json"
    adapter_path = checkpoint_dir / "control_lora.safetensors"
    lokr_path = checkpoint_dir / "qwen_control_lokr.safetensors"
    for path in (config_path, adapter_path, lokr_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported ControlLoRA checkpoint format: {config.get('format')}")
    if ControlLoRAConfig.from_dict(config["adapter"]) != adapter.config:
        raise ValueError("ControlLoRA checkpoint adapter topology does not match")
    if int(config["lokr_rank"]) != args.lokr_rank or float(config["lokr_alpha"]) != args.lokr_alpha:
        raise ValueError("ControlLoRA checkpoint LoKr topology does not match")
    checkpoint_mode = config.get("control_input_mode", "additive-hint")
    if checkpoint_mode != args.control_input_mode:
        raise ValueError(f"ControlLoRA checkpoint input mode {checkpoint_mode!r} does not match {args.control_input_mode!r}")
    adapter.load_state_dict(load_file(adapter_path), strict=True)
    load_state = lokr_network.load_weights(str(lokr_path))
    if load_state:
        raise RuntimeError(f"ControlLoRA LoKr checkpoint load mismatch: {load_state}")


def main() -> None:
    args = parse_args()
    context = initialize_distributed(args.device)
    args.world_size = context.world_size
    if (args.overfit_clip_id is None) == (args.overfit_ids_csv is None):
        raise ValueError("provide exactly one of --overfit-clip-id or --overfit-ids-csv")
    if args.crop_frames < 2:
        raise ValueError("crop-frames must be at least 2")
    if args.overfit_start_frame < 0:
        raise ValueError("overfit-start-frame must be non-negative")
    if args.residual_rank < 1 or args.lokr_rank < 1 or args.lokr_alpha <= 0.0:
        raise ValueError("adapter rank and alpha values must be positive")
    if args.hint_scale <= 0.0:
        raise ValueError("hint-scale must be positive")
    if not 0.0 <= args.reference_dropout < 1.0:
        raise ValueError("reference-dropout must be in [0, 1)")
    if not 0.0 <= args.feedback_corruption_rate <= 1.0:
        raise ValueError("feedback-corruption-rate must be in [0, 1]")
    if not 1 <= args.feedback_sampling_top_k <= 1024:
        raise ValueError("feedback-sampling-top-k must be between 1 and 1024")
    if args.semantic_loss_weight <= 0.0:
        raise ValueError("semantic-loss-weight must be positive")
    if args.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation must be positive")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = context.device
    clip_ids = load_clip_ids_csv(args.overfit_ids_csv) if args.overfit_ids_csv is not None else None
    args.training_clip_ids = clip_ids if clip_ids is not None else (args.overfit_clip_id,)
    dataset = CachedStylePairDataset(
        args.cache_dir,
        args.crop_frames,
        reference_context_frames=1,
        clip_id=args.overfit_clip_id,
        clip_ids=clip_ids,
        fixed_crop_start=None if args.random_crops else args.overfit_start_frame,
    )
    global_micro_batch_size = args.batch_size * context.world_size
    if len(dataset) % global_micro_batch_size:
        raise ValueError(
            f"dataset size {len(dataset)} is not divisible by global micro-batch size {global_micro_batch_size}"
        )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if context.enabled
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_control_samples,
        generator=torch.Generator().manual_seed(args.seed + context.rank),
    )
    if len(loader) % args.gradient_accumulation:
        raise ValueError("per-rank dataloader length must be divisible by gradient accumulation")
    validation_start = context.rank * args.batch_size
    validation_indices = range(validation_start, validation_start + args.batch_size)
    validation_sample = collate_control_samples([dataset[index] for index in validation_indices])
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
    lokr_network = create_qwen_lokr_adapter(
        language_model,
        rank=args.lokr_rank,
        alpha=args.lokr_alpha,
    ).to(device)
    adapter = MiniMaxMusic3ControlLoRAAdapter(
        ControlLoRAConfig(
            hidden_size=language_model.config.hidden_size,
            residual_rank=args.residual_rank,
            layer_indices=args.control_layers,
        )
    ).to(device)
    adapter.install(language_model)
    if args.init_checkpoint is not None:
        load_checkpoint(args.init_checkpoint, adapter, lokr_network, args)
    if args.gradient_checkpointing:
        language_model.gradient_checkpointing_enable()
    else:
        language_model.gradient_checkpointing_disable()
    language_model.train()
    depth_decoder.eval()
    training_module = ControlLoRATrainingModel(language_model, depth_decoder, adapter, lokr_network)
    training_model = (
        DistributedDataParallel(
            training_module,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
        )
        if context.enabled
        else training_module
    )
    torch.manual_seed(args.seed + context.rank)
    random.seed(args.seed + context.rank)
    parameter_groups = [
        {"params": list(adapter.parameters()), "lr": args.learning_rate},
        {"params": list(lokr_network.parameters()), "lr": args.lokr_learning_rate},
    ]
    parameters = [parameter for group in parameter_groups for parameter in group["params"]]
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
        power=1.0,
    )
    trainable_parameters = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    steps_per_epoch = len(loader) // args.gradient_accumulation
    if context.is_main_process:
        print(
            json.dumps(
                {
                    "trainable_parameters": trainable_parameters,
                    "training_files": len(dataset),
                    "batch_size_per_rank": args.batch_size,
                    "global_batch_size": args.batch_size * context.world_size * args.gradient_accumulation,
                    "world_size": context.world_size,
                    "steps_per_epoch": steps_per_epoch,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "checkpoint_depth_decoder": args.checkpoint_depth_decoder,
                }
            ),
            flush=True,
        )
    wandb_run = None
    if args.wandb_project and context.is_main_process:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))
    if context.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    if context.enabled:
        dist.barrier()
    baseline = evaluate(training_model, tokenizer, validation_sample, device, args)
    baseline = reduce_metrics(baseline, context)
    baseline["step"] = 0
    if context.is_main_process:
        print(json.dumps(baseline), flush=True)
    if wandb_run is not None:
        wandb_run.log(baseline, step=0)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    step = 0
    micro_step = 0
    data_epoch = 0
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(data_epoch)
        for sample in loader:
            sync_gradients = (micro_step + 1) % args.gradient_accumulation == 0
            synchronization = (
                nullcontext()
                if sync_gradients or not isinstance(training_model, DistributedDataParallel)
                else training_model.no_sync()
            )
            with synchronization:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    loss, metrics = train_step(
                        training_model,
                        tokenizer,
                        sample,
                        device,
                        args,
                    )
                    scaled_loss = loss / args.gradient_accumulation
                scaled_loss.backward()
            micro_step += 1
            if not sync_gradients:
                continue
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            metrics = reduce_metrics(metrics, context)
            metrics.update(
                {
                    "step": step,
                    "epoch": step / steps_per_epoch,
                    "grad_norm": float(grad_norm),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "lokr_learning_rate": scheduler.get_last_lr()[1],
                    "steps_per_second": step / (time.perf_counter() - started),
                }
            )
            if device.type == "cuda":
                metrics["peak_vram_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            if context.is_main_process:
                print(json.dumps(metrics), flush=True)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            if step % args.eval_every == 0 or step == args.max_steps:
                validation = evaluate(training_model, tokenizer, validation_sample, device, args)
                validation = reduce_metrics(validation, context)
                validation["step"] = step
                if context.is_main_process:
                    print(json.dumps(validation), flush=True)
                if wandb_run is not None:
                    wandb_run.log(validation, step=step)
            if step % args.save_every == 0 or step == args.max_steps:
                if context.is_main_process:
                    save_checkpoint(args.output_dir, step, adapter, lokr_network, optimizer, scheduler, args)
                if context.enabled:
                    dist.barrier()
            if step >= args.max_steps:
                break
        data_epoch += 1
    if wandb_run is not None:
        wandb_run.finish()
    if context.enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
