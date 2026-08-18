#!/usr/bin/env python3
# Copyright 2026 SimpleTuner contributors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3ForCausalLM

from scripts.minimax_music3.eval_prefix_distillation import generate_frame, sustained_relock_frame
from scripts.minimax_music3.train_control_lora import (
    CHECKPOINT_FORMAT,
    control_lora_hidden_states,
    evaluate,
    load_checkpoint,
    reference_warmup_codes,
)
from scripts.minimax_music3.train_reference_control import (
    AUDIO_CODE_OFFSET,
    AUDIO_VOCAB_SIZE,
    CachedStylePairDataset,
    tokenize_prompt,
)
from simpletuner.helpers.models.minimaxmusic.reference_control import (
    ControlLoRAConfig,
    MiniMaxMusic3ControlLoRAAdapter,
    create_qwen_lokr_adapter,
)
from simpletuner.helpers.models.minimaxmusic.rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder

FRAME_RATE = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MiniMax Music 3 shared-block ControlLoRA")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seconds", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--perturb-frame", type=int)
    parser.add_argument("--control-strength", type=float, default=1.0)
    return parser.parse_args()


def checkpoint_args(config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        lokr_rank=int(config["lokr_rank"]),
        lokr_alpha=float(config["lokr_alpha"]),
        hint_scale=float(config["hint_scale"]),
        reference_dropout=float(config["reference_dropout"]),
        feedback_corruption_rate=float(config["feedback_corruption_rate"]),
        feedback_sampling_top_k=int(config["feedback_sampling_top_k"]),
        semantic_loss_weight=float(config["semantic_loss_weight"]),
    )


def load_components(args, device: torch.device):
    config_path = args.checkpoint_dir / "control_lora.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported ControlLoRA checkpoint format: {config.get('format')}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, subfolder="tokenizer", cache_dir=args.cache_dir)
    language_model = Qwen3ForCausalLM.from_pretrained(
        args.model_id,
        subfolder="language_model",
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
    ).to(device)
    depth_decoder = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
        args.model_id,
        subfolder="rvq_depth_decoder",
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
    ).to(device)
    language_model.requires_grad_(False)
    depth_decoder.requires_grad_(False)
    runtime_args = checkpoint_args(config)
    lokr_network = create_qwen_lokr_adapter(
        language_model,
        rank=runtime_args.lokr_rank,
        alpha=runtime_args.lokr_alpha,
    ).to(device)
    adapter = MiniMaxMusic3ControlLoRAAdapter(ControlLoRAConfig.from_dict(config["adapter"])).to(
        device=device,
        dtype=torch.bfloat16,
    )
    adapter.install(language_model)
    load_checkpoint(args.checkpoint_dir, adapter, lokr_network, runtime_args)
    adapter.set_multiplier(args.control_strength)
    adapter_dtype = next(adapter.parameters()).dtype
    model_dtype = next(language_model.parameters()).dtype
    if adapter_dtype != model_dtype:
        raise ValueError(f"ControlLoRA dtype {adapter_dtype} does not match language model dtype {model_dtype}")
    language_model.eval()
    depth_decoder.eval()
    adapter.eval()
    return SimpleNamespace(
        tokenizer=tokenizer,
        language_model=language_model,
        depth_decoder=depth_decoder,
        adapter=adapter,
        lokr_network=lokr_network,
        config=config,
        runtime_args=runtime_args,
    )


@torch.inference_mode()
def rollout(
    components,
    sample: dict,
    frame_count: int,
    *,
    null_reference: bool,
    sample_codes: bool,
    top_k: int,
    seed: int,
    perturb_frame: int | None,
):
    language_model = components.language_model
    depth_decoder = components.depth_decoder
    device = next(language_model.parameters()).device
    target_codes = sample["target_codes"][:frame_count].to(device).unsqueeze(0)
    reference_codes = sample["reference_codes"].to(device).unsqueeze(0)
    query_positions = sample["query_positions"][:frame_count].to(device).unsqueeze(0)
    key_positions = sample["key_positions"].to(device).unsqueeze(0)
    text_ids = tokenize_prompt(components.tokenizer, sample["prompt"], sample["lyrics"], device)
    text_embeddings = language_model.model.embed_tokens(text_ids)
    warmup_codes = reference_warmup_codes(reference_codes, query_positions, key_positions)
    generated_codes = []
    target_log_probs = []
    target_top1 = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    started = time.perf_counter()
    for frame_index in range(frame_count):
        feedback_codes = torch.cat(
            (warmup_codes, *[codes.to(device).unsqueeze(1) for codes in generated_codes]),
            dim=1,
        )
        hidden = control_lora_hidden_states(
            language_model,
            depth_decoder,
            components.adapter,
            components.lokr_network,
            text_embeddings,
            feedback_codes,
            reference_codes,
            query_positions[:, : frame_index + 1],
            key_positions,
            hint_scale=components.runtime_args.hint_scale,
            null_reference=null_reference,
        )[:, -1]
        logits = F.linear(
            hidden,
            language_model.lm_head.weight[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + AUDIO_VOCAB_SIZE],
        ).float()
        target = target_codes[0, frame_index, 0]
        target_log_probs.append(logits.log_softmax(dim=-1)[0, target].cpu())
        target_top1.append((logits.argmax(dim=-1)[0] == target).cpu())
        codes, _ = generate_frame(
            language_model,
            depth_decoder,
            hidden,
            sample=sample_codes,
            top_k=top_k,
            generator=generator,
            semantic_override_logits=logits,
        )
        if frame_index == perturb_frame:
            codes = codes.clone()
            codes[0, 0] = (target + 1) % AUDIO_VOCAB_SIZE
        generated_codes.append(codes.cpu())
    return SimpleNamespace(
        generated_codes=torch.cat(generated_codes, dim=0),
        target_log_probs=torch.stack(target_log_probs),
        target_top1=torch.stack(target_top1),
        seconds=time.perf_counter() - started,
    )


def rollout_metrics(trace, target_codes: torch.Tensor) -> dict:
    target = target_codes[: trace.generated_codes.shape[0]]
    matches = trace.generated_codes[:, 0] == target[:, 0]
    misses = (~matches).nonzero().flatten()
    reacquisitions = (~matches[:-1] & matches[1:]).sum()
    return {
        "seconds": round(trace.seconds, 3),
        "semantic_target_top1": matches.float().mean().item(),
        "acoustic_target_top1": (trace.generated_codes[:, 1:] == target[:, 1:]).float().mean().item(),
        "first_semantic_miss": int(misses[0]) if misses.numel() else None,
        "semantic_match_frames": matches.nonzero().flatten().tolist(),
        "semantic_reacquisitions": int(reacquisitions),
        "semantic_sustained_relock_2": sustained_relock_frame(matches, 2),
        "semantic_sustained_relock_4": sustained_relock_frame(matches, 4),
        "target_semantic_log_prob_mean": trace.target_log_probs.mean().item(),
        "target_semantic_top1_mean": trace.target_top1.float().mean().item(),
    }


def main() -> None:
    args = parse_args()
    if args.max_seconds <= 0.0:
        raise ValueError("max-seconds must be positive")
    if not 1 <= args.top_k <= AUDIO_VOCAB_SIZE:
        raise ValueError(f"top-k must be between 1 and {AUDIO_VOCAB_SIZE}")
    if args.control_strength < 0.0:
        raise ValueError("control-strength must be non-negative")
    device = torch.device(args.device)
    components = load_components(args, device)
    dataset = CachedStylePairDataset(
        args.pair_cache,
        crop_frames=int(components.config["crop_frames"]),
        reference_context_frames=1,
        clip_id=args.clip_id,
        fixed_crop_start=0,
    )
    sample = dataset[0]
    frame_count = min(sample["target_codes"].shape[0], round(args.max_seconds * FRAME_RATE))
    if args.perturb_frame is not None and not 0 <= args.perturb_frame < frame_count:
        raise ValueError(f"perturb-frame must be between 0 and {frame_count - 1}")
    teacher_forced = evaluate(
        components.language_model,
        components.depth_decoder,
        components.adapter,
        components.lokr_network,
        components.tokenizer,
        sample,
        device,
        components.runtime_args,
    )
    traces = {
        "reference": rollout(
            components,
            sample,
            frame_count,
            null_reference=False,
            sample_codes=args.sample,
            top_k=args.top_k,
            seed=args.seed,
            perturb_frame=args.perturb_frame,
        ),
        "null": rollout(
            components,
            sample,
            frame_count,
            null_reference=True,
            sample_codes=args.sample,
            top_k=args.top_k,
            seed=args.seed,
            perturb_frame=args.perturb_frame,
        ),
    }
    results = {
        "clip_id": args.clip_id,
        "frames": frame_count,
        "sample": args.sample,
        "top_k": args.top_k,
        "perturb_frame": args.perturb_frame,
        "control_strength": args.control_strength,
        "checkpoint_config": components.config,
        "teacher_forced": teacher_forced,
        "modes": {name: rollout_metrics(trace, sample["target_codes"]) for name, trace in traces.items()},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for name, trace in traces.items():
        save_file(
            {
                "generated_codes": trace.generated_codes.to(torch.int16),
                "target_log_probs": trace.target_log_probs,
                "target_top1": trace.target_top1,
            },
            args.output_dir / f"{name}.safetensors",
        )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
