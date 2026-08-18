#!/usr/bin/env python3
# Copyright 2026 SimpleTuner contributors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import contextlib
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Qwen3ForCausalLM, get_polynomial_decay_schedule_with_warmup

from scripts.minimax_music3.train_reference_control import (
    AUDIO_CODE_OFFSET,
    AUDIO_VOCAB_SIZE,
    CachedStylePairDataset,
    sample_codes_from_hidden,
    splice_sampled_feedback,
    tokenize_prompt,
)
from simpletuner.helpers.models.minimaxmusic.reference_control import (
    create_qwen_lokr_adapter,
    create_qwen_oftv2_adapter,
    embed_rvq_frames,
    prefix_adapter_checkpoint_filename,
)
from simpletuner.helpers.models.minimaxmusic.rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder

PREFIX_FORMAT = "rvq-prefix-text-target-v3"
TEACHER_FORMAT = "target-prefix-text-target-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MiniMax Music 3 reference-prefix distillation with LyCORIS")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--crop-frames", type=int, default=256)
    parser.add_argument("--overfit-clip-id", required=True)
    parser.add_argument("--overfit-start-frame", type=int, default=0)
    parser.add_argument("--feedback-corruption-rate", type=float, default=0.25)
    parser.add_argument("--feedback-sampling-top-k", type=int, default=50)
    parser.add_argument("--sequential-feedback-rollout", action="store_true")
    parser.add_argument("--warmup-mode", choices=("generated", "reference-first"), default="generated")
    parser.add_argument("--reference-dropout", type=float, default=0.1)
    parser.add_argument("--teacher-kl-weight", type=float, default=0.25)
    parser.add_argument("--teacher-kl-temperature", type=float, default=2.0)
    parser.add_argument("--hidden-alignment-weight", type=float, default=0.1)
    parser.add_argument("--prefix-anchor-frames", type=int, default=1)
    parser.add_argument("--prefix-anchor-weight", type=float, default=1.0)
    parser.add_argument("--semantic-loss-weight", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--adapter-type", choices=("oftv2", "lokr"), default="oftv2")
    parser.add_argument("--oft-block-size", type=int, default=64)
    parser.add_argument("--lokr-rank", type=int, default=16)
    parser.add_argument("--lokr-alpha", type=float, default=16.0)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--wandb-project")
    return parser.parse_args()


def special_token_id(tokenizer, token: str) -> int:
    token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"{token} must encode to exactly one token, found {len(token_ids)}")
    return token_ids[0]


def reference_prefix_embeddings(
    language_model,
    depth_decoder,
    tokenizer,
    reference_codes: torch.Tensor,
    *,
    null_reference: bool,
) -> torch.Tensor:
    if reference_codes.ndim != 3:
        raise ValueError("reference_codes must have shape [batch, frames, codebooks]")
    batch_size = reference_codes.shape[0]
    device = reference_codes.device
    start_id = special_token_id(tokenizer, "<|audio_start|>")
    end_id = special_token_id(tokenizer, "<|audio_end|>")
    delimiters = torch.tensor((start_id, end_id), device=device).repeat(batch_size, 1)
    start = language_model.model.embed_tokens(delimiters[:, :1])
    end = language_model.model.embed_tokens(delimiters[:, 1:])
    frames = embed_rvq_frames(language_model, depth_decoder, reference_codes)
    if null_reference:
        frames = frames * 0.0
    return torch.cat((start, frames, end), dim=1)


def conditioned_hidden_states(
    language_model,
    depth_decoder,
    prefix_embeddings: torch.Tensor | None,
    text_embeddings: torch.Tensor,
    feedback_codes: torch.Tensor,
) -> torch.Tensor:
    feedback_embeddings = embed_rvq_frames(language_model, depth_decoder, feedback_codes)
    parts = [text_embeddings, feedback_embeddings]
    if prefix_embeddings is not None:
        parts.insert(0, prefix_embeddings)
    output = language_model.model(inputs_embeds=torch.cat(parts, dim=1), use_cache=False)
    return output.last_hidden_state[:, -feedback_codes.shape[1] :]


def semantic_logits(language_model, hidden_states: torch.Tensor) -> torch.Tensor:
    return F.linear(
        hidden_states,
        language_model.lm_head.weight[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + AUDIO_VOCAB_SIZE],
    ).float()


@torch.no_grad()
def greedy_prefix_warmup_codes(
    language_model,
    depth_decoder,
    prefix_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
) -> torch.Tensor:
    hidden = language_model.model(
        inputs_embeds=torch.cat((prefix_embeddings, text_embeddings), dim=1),
        use_cache=False,
    ).last_hidden_state[:, -1:]
    return sample_codes_from_hidden(depth_decoder, language_model, hidden, top_k=1)


def prefix_warmup_codes(
    language_model,
    depth_decoder,
    prefix_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    conditioning_codes: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if mode == "reference-first":
        return conditioning_codes[:, :1]
    if mode == "generated":
        return greedy_prefix_warmup_codes(language_model, depth_decoder, prefix_embeddings, text_embeddings)
    raise ValueError(f"Unsupported warmup mode: {mode}")


@torch.no_grad()
def sequential_feedback_codes(
    language_model,
    depth_decoder,
    prefix_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    warmup_codes: torch.Tensor,
    frame_count: int,
    top_k: int,
) -> torch.Tensor:
    was_training = language_model.training
    language_model.eval()
    try:
        warmup_embeddings = embed_rvq_frames(language_model, depth_decoder, warmup_codes)
        output = language_model.model(
            inputs_embeds=torch.cat((prefix_embeddings, text_embeddings, warmup_embeddings), dim=1),
            use_cache=True,
        )
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1:]
        generated = []
        for _ in range(frame_count - 1):
            codes = sample_codes_from_hidden(depth_decoder, language_model, last_hidden, top_k)
            generated.append(codes)
            feedback = embed_rvq_frames(language_model, depth_decoder, codes)
            output = language_model.model(
                inputs_embeds=feedback,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1:]
    finally:
        language_model.train(was_training)
    return torch.cat((warmup_codes, *generated), dim=1)


def frame_loss_weights(
    frame_count: int,
    anchor_frames: int,
    anchor_weight: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    if not 1 <= anchor_frames <= frame_count:
        raise ValueError("anchor_frames must be between 1 and frame_count")
    if anchor_weight < 1.0:
        raise ValueError("anchor_weight must be at least 1")
    weights = torch.ones(frame_count, device=device)
    weights[:anchor_frames] = anchor_weight
    return weights


def weighted_frame_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if values.ndim != 2 or weights.shape != (values.shape[1],):
        raise ValueError("values must have shape [batch, frames] matching weights")
    return (values * weights.unsqueeze(0)).sum() / (weights.sum() * values.shape[0])


def weighted_head_mean(
    semantic_loss: torch.Tensor,
    depth_losses: list[torch.Tensor],
    semantic_weight: float,
) -> torch.Tensor:
    if semantic_weight <= 0.0:
        raise ValueError("semantic_weight must be positive")
    return (semantic_weight * semantic_loss + torch.stack(depth_losses).sum()) / (semantic_weight + len(depth_losses))


def depth_outputs(
    depth_decoder,
    language_model,
    hidden_states: torch.Tensor,
    target_codes: torch.Tensor,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    batch_size, frame_count, hidden_size = hidden_states.shape
    hidden = hidden_states.reshape(batch_size * frame_count, hidden_size)
    codes = target_codes.reshape(batch_size * frame_count, -1)
    sequence = [depth_decoder.projection(hidden).unsqueeze(1)]
    semantic = language_model.model.embed_tokens(codes[:, 0] + AUDIO_CODE_OFFSET)
    sequence.append(depth_decoder.projection(semantic).unsqueeze(1))
    logits = []
    depth_hiddens = []
    for codebook in range(1, depth_decoder.config.num_codebooks):
        depth_hidden = depth_decoder(torch.cat(sequence, dim=1))[:, -1]
        depth_hiddens.append(depth_hidden)
        logits.append(depth_decoder.audio_heads[codebook - 1](depth_hidden).float())
        if codebook < depth_decoder.config.num_codebooks - 1:
            embedding = depth_decoder.audio_embeddings(
                codes[:, codebook] + (codebook - 1) * depth_decoder.config.audio_vocab_size
            )
            sequence.append(depth_decoder.projection(embedding).unsqueeze(1))
    combined = torch.cat(depth_hiddens, dim=-1).reshape(batch_size, frame_count, -1)
    return logits, combined


@contextlib.contextmanager
def base_model_teacher(language_model, oft_network):
    was_training = language_model.training
    oft_network.set_multiplier(0.0)
    language_model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        oft_network.set_multiplier(1.0)
        language_model.train(was_training)


def teacher_targets(
    language_model,
    depth_decoder,
    oft_network,
    tokenizer,
    text_ids: torch.Tensor,
    target_codes: torch.Tensor,
    warmup_mode: str = "generated",
) -> tuple[torch.Tensor, torch.Tensor]:
    with base_model_teacher(language_model, oft_network):
        text_embeddings = language_model.model.embed_tokens(text_ids)
        target_prefix = reference_prefix_embeddings(
            language_model,
            depth_decoder,
            tokenizer,
            target_codes,
            null_reference=False,
        )
        warmup_codes = prefix_warmup_codes(
            language_model,
            depth_decoder,
            target_prefix,
            text_embeddings,
            target_codes,
            warmup_mode,
        )
        feedback_codes = torch.cat((warmup_codes, target_codes[:, :-1]), dim=1)
        hidden = conditioned_hidden_states(
            language_model,
            depth_decoder,
            target_prefix,
            text_embeddings,
            feedback_codes,
        )
        logits = semantic_logits(language_model, hidden)
        _, depth_hidden = depth_outputs(depth_decoder, language_model, hidden, target_codes)
        frame_hidden = torch.cat((hidden.float(), depth_hidden.float()), dim=-1)
    return logits, frame_hidden


def distillation_step(
    language_model,
    depth_decoder,
    oft_network,
    tokenizer,
    sample: dict,
    device: torch.device,
    *,
    feedback_corruption_rate: float,
    feedback_sampling_top_k: int,
    reference_dropout: float,
    teacher_kl_weight: float,
    teacher_kl_temperature: float,
    hidden_alignment_weight: float,
    prefix_anchor_frames: int,
    prefix_anchor_weight: float,
    semantic_loss_weight: float,
    warmup_mode: str,
    sequential_feedback_rollout: bool,
    force_null_reference: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    target_codes = sample["target_codes"].to(device).unsqueeze(0)
    reference_codes = sample["reference_codes"].to(device).unsqueeze(0)
    text_ids = tokenize_prompt(tokenizer, sample["prompt"], sample["lyrics"], device)
    teacher_logits, teacher_frame_hidden = teacher_targets(
        language_model,
        depth_decoder,
        oft_network,
        tokenizer,
        text_ids,
        target_codes,
        warmup_mode,
    )
    null_reference = force_null_reference or torch.rand((), device=device).item() < reference_dropout
    with torch.no_grad():
        text_embeddings = language_model.model.embed_tokens(text_ids)
        prefix_embeddings = reference_prefix_embeddings(
            language_model,
            depth_decoder,
            tokenizer,
            reference_codes,
            null_reference=null_reference,
        )
        warmup_codes = prefix_warmup_codes(
            language_model,
            depth_decoder,
            prefix_embeddings,
            text_embeddings,
            reference_codes,
            warmup_mode,
        )
        clean_feedback = torch.cat((warmup_codes, target_codes[:, :-1]), dim=1)
    feedback_codes = clean_feedback
    corruption_fraction = 0.0
    if sequential_feedback_rollout:
        feedback_codes = sequential_feedback_codes(
            language_model,
            depth_decoder,
            prefix_embeddings,
            text_embeddings,
            warmup_codes,
            target_codes.shape[1],
            feedback_sampling_top_k,
        )
        corruption_fraction = 1.0
    elif feedback_corruption_rate > 0.0:
        with torch.no_grad():
            clean_hidden = conditioned_hidden_states(
                language_model,
                depth_decoder,
                prefix_embeddings,
                text_embeddings,
                clean_feedback,
            )
            sampled_codes = sample_codes_from_hidden(
                depth_decoder,
                language_model,
                clean_hidden,
                feedback_sampling_top_k,
            )
            feedback_codes, corruption_fraction = splice_sampled_feedback(
                clean_feedback,
                sampled_codes,
                loss_start=0,
                corruption_rate=feedback_corruption_rate,
            )
    hidden = conditioned_hidden_states(
        language_model,
        depth_decoder,
        prefix_embeddings,
        text_embeddings,
        feedback_codes,
    )
    student_logits = semantic_logits(language_model, hidden)
    student_depth_logits, student_depth_hidden = depth_outputs(depth_decoder, language_model, hidden, target_codes)
    weights = frame_loss_weights(
        target_codes.shape[1],
        prefix_anchor_frames,
        prefix_anchor_weight,
        device=device,
    )
    semantic_per_frame = F.cross_entropy(
        student_logits.flatten(0, 1),
        target_codes[..., 0].flatten(),
        reduction="none",
    ).reshape(target_codes.shape[:2])
    semantic_loss = weighted_frame_mean(semantic_per_frame, weights)
    depth_losses = [
        weighted_frame_mean(
            F.cross_entropy(logits, target_codes[..., codebook].flatten(), reduction="none").reshape(target_codes.shape[:2]),
            weights,
        )
        for codebook, logits in enumerate(student_depth_logits, start=1)
    ]
    code_loss = weighted_head_mean(semantic_loss, depth_losses, semantic_loss_weight)
    temperature = teacher_kl_temperature
    kl_per_frame = (
        F.kl_div(
            F.log_softmax(student_logits.flatten(0, 1) / temperature, dim=-1),
            F.softmax(teacher_logits.flatten(0, 1) / temperature, dim=-1),
            reduction="none",
        )
        .sum(dim=-1)
        .reshape(target_codes.shape[:2])
        * temperature**2
    )
    kl_loss = kl_per_frame.mean()
    student_frame_hidden = torch.cat((hidden.float(), student_depth_hidden.float()), dim=-1)
    hidden_alignment_loss = (1.0 - F.cosine_similarity(student_frame_hidden, teacher_frame_hidden, dim=-1)).mean()
    loss = code_loss + teacher_kl_weight * kl_loss + hidden_alignment_weight * hidden_alignment_loss
    with torch.no_grad():
        target_semantic = target_codes[..., 0]
        metrics = {
            "loss": loss.item(),
            "code_loss": code_loss.item(),
            "semantic_loss": semantic_loss.item(),
            "teacher_kl_loss": kl_loss.item(),
            "hidden_alignment_loss": hidden_alignment_loss.item(),
            "hidden_cosine": 1.0 - hidden_alignment_loss.item(),
            "semantic_top1": (student_logits.argmax(dim=-1) == target_semantic).float().mean().item(),
            "semantic_first_top1": (student_logits[:, 0].argmax(dim=-1) == target_semantic[:, 0]).float().mean().item(),
            "semantic_anchor_top1": (
                student_logits[:, :prefix_anchor_frames].argmax(dim=-1) == target_semantic[:, :prefix_anchor_frames]
            )
            .float()
            .mean()
            .item(),
            "teacher_semantic_top1": (teacher_logits.argmax(dim=-1) == target_semantic).float().mean().item(),
            "student_teacher_top1": (student_logits.argmax(dim=-1) == teacher_logits.argmax(dim=-1)).float().mean().item(),
            "feedback_corruption_fraction": corruption_fraction,
            "null_reference": float(null_reference),
        }
        metrics.update(
            {f"codebook_{index}_loss": value.item() for index, value in enumerate((semantic_loss, *depth_losses))}
        )
    return loss, metrics


@torch.no_grad()
def evaluate(language_model, depth_decoder, oft_network, tokenizer, sample: dict, device: torch.device, args) -> dict:
    was_training = language_model.training
    language_model.eval()
    values = {}
    for name, null_reference in (("reference", False), ("null", True)):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, metrics = distillation_step(
                language_model,
                depth_decoder,
                oft_network,
                tokenizer,
                sample,
                device,
                feedback_corruption_rate=0.0,
                feedback_sampling_top_k=args.feedback_sampling_top_k,
                reference_dropout=0.0,
                teacher_kl_weight=args.teacher_kl_weight,
                teacher_kl_temperature=args.teacher_kl_temperature,
                hidden_alignment_weight=args.hidden_alignment_weight,
                prefix_anchor_frames=args.prefix_anchor_frames,
                prefix_anchor_weight=args.prefix_anchor_weight,
                semantic_loss_weight=args.semantic_loss_weight,
                warmup_mode=args.warmup_mode,
                sequential_feedback_rollout=False,
                force_null_reference=null_reference,
            )
        values[name] = metrics
    language_model.train(was_training)
    return {
        "validation/reference_loss": values["reference"]["loss"],
        "validation/null_loss": values["null"]["loss"],
        "validation/reference_semantic_top1": values["reference"]["semantic_top1"],
        "validation/null_semantic_top1": values["null"]["semantic_top1"],
        "validation/reference_semantic_first_top1": values["reference"]["semantic_first_top1"],
        "validation/null_semantic_first_top1": values["null"]["semantic_first_top1"],
        "validation/reference_semantic_anchor_top1": values["reference"]["semantic_anchor_top1"],
        "validation/null_semantic_anchor_top1": values["null"]["semantic_anchor_top1"],
        "validation/reference_hidden_cosine": values["reference"]["hidden_cosine"],
        "validation/null_hidden_cosine": values["null"]["hidden_cosine"],
        "validation/teacher_semantic_top1": values["reference"]["teacher_semantic_top1"],
    }


def create_prefix_adapter(language_model, args):
    if args.adapter_type == "oftv2":
        return create_qwen_oftv2_adapter(language_model, block_size=args.oft_block_size)
    return create_qwen_lokr_adapter(language_model, rank=args.lokr_rank, alpha=args.lokr_alpha)


def validate_init_checkpoint(args) -> None:
    if args.init_checkpoint is None:
        return
    config_path = args.init_checkpoint / "prefix_distillation.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint_adapter_type = config.get("adapter_type", "oftv2")
    if checkpoint_adapter_type != args.adapter_type:
        raise ValueError(f"init checkpoint uses {checkpoint_adapter_type}, but --adapter-type is {args.adapter_type}")
    if args.adapter_type == "oftv2" and int(config["oft_block_size"]) != args.oft_block_size:
        raise ValueError("init checkpoint OFTv2 block size does not match --oft-block-size")
    if args.adapter_type == "lokr":
        if int(config["lokr_rank"]) != args.lokr_rank or float(config["lokr_alpha"]) != args.lokr_alpha:
            raise ValueError("init checkpoint LoKr rank/alpha do not match the requested topology")


def save_checkpoint(output_dir: Path, step: int, adapter_network, optimizer, scheduler, args) -> None:
    checkpoint_dir = output_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_network.save_weights(
        checkpoint_dir / prefix_adapter_checkpoint_filename(args.adapter_type),
        torch.float32,
        {},
    )
    config = {
        "format": PREFIX_FORMAT,
        "teacher_format": TEACHER_FORMAT,
        "adapter_type": args.adapter_type,
        "oft_block_size": args.oft_block_size,
        "lokr_rank": args.lokr_rank,
        "lokr_alpha": args.lokr_alpha,
        "crop_frames": args.crop_frames,
        "feedback_corruption_rate": args.feedback_corruption_rate,
        "feedback_sampling_top_k": args.feedback_sampling_top_k,
        "reference_dropout": args.reference_dropout,
        "teacher_kl_weight": args.teacher_kl_weight,
        "teacher_kl_temperature": args.teacher_kl_temperature,
        "hidden_alignment_weight": args.hidden_alignment_weight,
        "prefix_anchor_frames": args.prefix_anchor_frames,
        "prefix_anchor_weight": args.prefix_anchor_weight,
        "semantic_loss_weight": args.semantic_loss_weight,
        "warmup_mode": args.warmup_mode,
        "sequential_feedback_rollout": args.sequential_feedback_rollout,
        "init_checkpoint": args.init_checkpoint.name if args.init_checkpoint is not None else None,
    }
    (checkpoint_dir / "prefix_distillation.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    torch.save(
        {"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
        checkpoint_dir / "training_state.pt",
    )


def main() -> None:
    args = parse_args()
    if args.crop_frames < 2:
        raise ValueError("crop-frames must be at least 2")
    if args.overfit_start_frame < 0:
        raise ValueError("overfit-start-frame must be non-negative")
    if not 0.0 <= args.feedback_corruption_rate <= 1.0:
        raise ValueError("feedback-corruption-rate must be in [0, 1]")
    if args.sequential_feedback_rollout and args.feedback_corruption_rate != 1.0:
        raise ValueError("sequential-feedback-rollout requires feedback-corruption-rate=1")
    if not 1 <= args.feedback_sampling_top_k <= 1024:
        raise ValueError("feedback-sampling-top-k must be between 1 and 1024")
    if not 0.0 <= args.reference_dropout < 1.0:
        raise ValueError("reference-dropout must be in [0, 1)")
    if args.teacher_kl_weight < 0.0 or args.hidden_alignment_weight < 0.0:
        raise ValueError("distillation weights must be non-negative")
    if args.teacher_kl_temperature <= 0.0:
        raise ValueError("teacher-kl-temperature must be positive")
    if not 1 <= args.prefix_anchor_frames <= args.crop_frames:
        raise ValueError("prefix-anchor-frames must be between 1 and crop-frames")
    if args.prefix_anchor_weight < 1.0:
        raise ValueError("prefix-anchor-weight must be at least 1")
    if args.semantic_loss_weight <= 0.0:
        raise ValueError("semantic-loss-weight must be positive")
    if args.lokr_rank <= 0 or args.lokr_alpha <= 0.0:
        raise ValueError("LoKr rank and alpha must be positive")
    if args.gradient_accumulation < 1:
        raise ValueError("gradient-accumulation must be positive")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    dataset = CachedStylePairDataset(
        args.cache_dir,
        args.crop_frames,
        reference_context_frames=0,
        clip_id=args.overfit_clip_id,
        fixed_crop_start=args.overfit_start_frame,
    )
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=args.num_workers)
    sample = dataset[0]
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
    validate_init_checkpoint(args)
    oft_network = create_prefix_adapter(language_model, args).to(device)
    if args.init_checkpoint is not None:
        weights_path = args.init_checkpoint / prefix_adapter_checkpoint_filename(args.adapter_type)
        if not weights_path.is_file():
            raise FileNotFoundError(weights_path)
        load_state = oft_network.load_weights(str(weights_path))
        if load_state:
            raise RuntimeError(f"LyCORIS checkpoint load mismatch: {load_state}")
    language_model.gradient_checkpointing_enable()
    language_model.train()
    depth_decoder.eval()
    parameters = list(oft_network.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
    print(json.dumps({"adapter_type": args.adapter_type, "trainable_parameters": trainable_parameters}), flush=True)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
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
    baseline = evaluate(language_model, depth_decoder, oft_network, tokenizer, sample, device, args)
    baseline["step"] = 0
    print(json.dumps(baseline), flush=True)
    if wandb_run is not None:
        wandb_run.log(baseline, step=0)
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    step = 0
    micro_step = 0
    while step < args.max_steps:
        for sample in loader:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss, metrics = distillation_step(
                    language_model,
                    depth_decoder,
                    oft_network,
                    tokenizer,
                    sample,
                    device,
                    feedback_corruption_rate=args.feedback_corruption_rate,
                    feedback_sampling_top_k=args.feedback_sampling_top_k,
                    reference_dropout=args.reference_dropout,
                    teacher_kl_weight=args.teacher_kl_weight,
                    teacher_kl_temperature=args.teacher_kl_temperature,
                    hidden_alignment_weight=args.hidden_alignment_weight,
                    prefix_anchor_frames=args.prefix_anchor_frames,
                    prefix_anchor_weight=args.prefix_anchor_weight,
                    semantic_loss_weight=args.semantic_loss_weight,
                    warmup_mode=args.warmup_mode,
                    sequential_feedback_rollout=args.sequential_feedback_rollout,
                )
                scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation:
                continue
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
            print(json.dumps(metrics), flush=True)
            if wandb_run is not None:
                wandb_run.log(metrics, step=step)
            if step % args.eval_every == 0 or step == args.max_steps:
                validation = evaluate(language_model, depth_decoder, oft_network, tokenizer, sample, device, args)
                validation["step"] = step
                print(json.dumps(validation), flush=True)
                if wandb_run is not None:
                    wandb_run.log(validation, step=step)
            if step % args.save_every == 0 or step == args.max_steps:
                save_checkpoint(args.output_dir, step, oft_network, optimizer, scheduler, args)
            if step >= args.max_steps:
                break
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
