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
    ControlLoRATrainingModel,
    batched_text_embeddings,
    control_lora_hidden_states,
    evaluate,
    load_checkpoint,
    load_clip_ids_csv,
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
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--clip-id")
    selection.add_argument("--clip-ids-csv", type=Path)
    parser.add_argument("--reference-clip-id")
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
    parser.add_argument("--reference-frame-shift", type=int, default=0)
    parser.add_argument("--teacher-forced-only", action="store_true")
    return parser.parse_args()


def checkpoint_args(config: dict) -> SimpleNamespace:
    return SimpleNamespace(
        lokr_rank=int(config["lokr_rank"]),
        lokr_alpha=float(config["lokr_alpha"]),
        hint_scale=float(config["hint_scale"]),
        control_input_mode=config.get("control_input_mode", "additive-hint"),
        reference_dropout=float(config["reference_dropout"]),
        feedback_corruption_rate=float(config["feedback_corruption_rate"]),
        feedback_sampling_top_k=int(config["feedback_sampling_top_k"]),
        semantic_loss_weight=float(config["semantic_loss_weight"]),
        checkpoint_depth_decoder=False,
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
    training_model = ControlLoRATrainingModel(language_model, depth_decoder, adapter, lokr_network)
    return SimpleNamespace(
        tokenizer=tokenizer,
        language_model=language_model,
        depth_decoder=depth_decoder,
        adapter=adapter,
        lokr_network=lokr_network,
        training_model=training_model,
        config=config,
        runtime_args=runtime_args,
    )


@torch.inference_mode()
def teacher_forced_semantic_trace(components, sample: dict, *, null_reference: bool) -> SimpleNamespace:
    language_model = components.language_model
    depth_decoder = components.depth_decoder
    device = next(language_model.parameters()).device
    target_codes = sample["target_codes"].to(device).unsqueeze(0)
    reference_codes = sample["reference_codes"].to(device).unsqueeze(0)
    query_positions = sample["query_positions"].to(device).unsqueeze(0)
    key_positions = sample["key_positions"].to(device).unsqueeze(0)
    loss_start = int(sample.get("loss_start", 0))
    text_embeddings, text_attention_mask = batched_text_embeddings(
        language_model,
        components.tokenizer,
        sample["prompt"],
        sample["lyrics"],
        device,
    )
    warmup_codes = reference_warmup_codes(reference_codes, query_positions, key_positions)
    feedback_codes = torch.cat((warmup_codes, target_codes[:, :-1]), dim=1)
    hidden = components.training_model(
        text_embeddings,
        feedback_codes,
        reference_codes,
        query_positions,
        key_positions,
        hint_scale=components.runtime_args.hint_scale,
        null_reference=null_reference,
        control_input_mode=components.runtime_args.control_input_mode,
        text_attention_mask=text_attention_mask,
        reference_attention_mask=torch.ones(reference_codes.shape[:2], dtype=torch.long, device=device),
    )[:, loss_start:]
    target_semantic = target_codes[:, loss_start:, 0]
    logits = F.linear(
        hidden,
        language_model.lm_head.weight[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + AUDIO_VOCAB_SIZE],
    ).float()
    cross_entropy = F.cross_entropy(
        logits.flatten(0, 1),
        target_semantic.flatten(),
        reduction="none",
    ).view_as(target_semantic)
    return SimpleNamespace(
        cross_entropy=cross_entropy.squeeze(0).cpu(),
        correct=(logits.argmax(dim=-1) == target_semantic).squeeze(0).cpu(),
        target_semantic=target_semantic.squeeze(0).cpu(),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def shifted_reference_sample(sample: dict, frame_shift: int) -> dict:
    if frame_shift == 0:
        return sample
    shifted = dict(sample)
    shifted["reference_codes"] = torch.roll(sample["reference_codes"], shifts=frame_shift, dims=0)
    return shifted


def substituted_reference_sample(sample: dict, reference_sample: dict) -> dict:
    query_positions = sample["query_positions"]
    key_positions = reference_sample["key_positions"]
    if query_positions.numel() < 2 or key_positions.numel() < 2:
        raise ValueError("reference substitution requires at least two query and key positions")
    query_fraction = (query_positions - query_positions[0]) / (query_positions[-1] - query_positions[0])
    substituted = dict(sample)
    substituted["reference_codes"] = reference_sample["reference_codes"]
    substituted["key_positions"] = key_positions
    substituted["query_positions"] = key_positions[0] + query_fraction * (key_positions[-1] - key_positions[0])
    return substituted


def summarize_teacher_forced_pair(
    reference_ce: torch.Tensor,
    null_ce: torch.Tensor,
    reference_correct: torch.Tensor,
    null_correct: torch.Tensor,
    semantic_transition: torch.Tensor,
) -> dict:
    tensors = (reference_ce, null_ce, reference_correct, null_correct, semantic_transition)
    if any(tensor.ndim != 1 for tensor in tensors):
        raise ValueError("teacher-forced trace tensors must be one-dimensional")
    if len({tensor.shape[0] for tensor in tensors}) != 1:
        raise ValueError("teacher-forced trace tensors must have equal lengths")
    if reference_ce.numel() == 0:
        raise ValueError("teacher-forced trace tensors cannot be empty")
    reference_correct = reference_correct.bool()
    null_correct = null_correct.bool()
    semantic_transition = semantic_transition.bool()
    null_wrong = ~null_correct
    fixes = null_wrong & reference_correct
    regressions = null_correct & ~reference_correct
    ce_gain = null_ce.float() - reference_ce.float()

    def subset_metrics(mask: torch.Tensor) -> dict:
        frame_count = int(mask.sum())
        if frame_count == 0:
            return {"frames": 0, "ce_gain_mean": None, "ce_gain_positive_fraction": None}
        values = ce_gain[mask]
        return {
            "frames": frame_count,
            "ce_gain_mean": values.mean().item(),
            "ce_gain_positive_fraction": (values > 0).float().mean().item(),
        }

    frame_count = reference_ce.shape[0]
    null_wrong_count = int(null_wrong.sum())
    null_correct_count = int(null_correct.sum())
    fix_count = int(fixes.sum())
    regression_count = int(regressions.sum())
    return {
        "frames": frame_count,
        "reference_top1": reference_correct.float().mean().item(),
        "null_top1": null_correct.float().mean().item(),
        "top1_gain": (reference_correct.float().mean() - null_correct.float().mean()).item(),
        "null_wrong_frames": null_wrong_count,
        "null_correct_frames": null_correct_count,
        "reference_fixes": fix_count,
        "reference_regressions": regression_count,
        "conditional_fix_rate": _ratio(fix_count, null_wrong_count),
        "conditional_regression_rate": _ratio(regression_count, null_correct_count),
        "net_available_headroom_gain": _ratio(fix_count - regression_count, null_wrong_count),
        "reference_ce_mean": reference_ce.float().mean().item(),
        "null_ce_mean": null_ce.float().mean().item(),
        "ce_gain_mean": ce_gain.mean().item(),
        "ce_gain_median": ce_gain.median().item(),
        "ce_gain_positive_fraction": (ce_gain > 0).float().mean().item(),
        "null_wrong": subset_metrics(null_wrong),
        "semantic_transition": subset_metrics(semantic_transition),
        "semantic_stable": subset_metrics(~semantic_transition),
    }


def plot_teacher_forced_pair(trace: dict[str, torch.Tensor], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    ce_gain = (trace["null_ce"] - trace["reference_ce"]).float().numpy()
    null_wrong = (~trace["null_correct"].bool()).numpy()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(ce_gain, bins=80, alpha=0.7, label="all frames")
    if null_wrong.any():
        axis.hist(ce_gain[null_wrong], bins=80, alpha=0.55, label="null-wrong frames")
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_xlabel("null CE - reference CE (positive favors reference)")
    axis.set_ylabel("frames")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "teacher_forced_ce_gain_histogram.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(ce_gain, linewidth=0.6, alpha=0.8)
    transition_indices = trace["semantic_transition"].nonzero().flatten().numpy()
    if transition_indices.size:
        axis.scatter(
            transition_indices,
            ce_gain[transition_indices],
            s=5,
            alpha=0.5,
            label="target semantic-code transition",
        )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("concatenated deterministic validation frame")
    axis.set_ylabel("null CE - reference CE")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "teacher_forced_ce_gain_timeline.png", dpi=160)
    plt.close(figure)


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
            control_input_mode=components.runtime_args.control_input_mode,
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
    if args.clip_ids_csv is not None and not args.teacher_forced_only:
        raise ValueError("--clip-ids-csv requires --teacher-forced-only")
    if args.reference_clip_id is not None and args.clip_id is None:
        raise ValueError("--reference-clip-id requires --clip-id")
    device = torch.device(args.device)
    components = load_components(args, device)
    clip_ids = load_clip_ids_csv(args.clip_ids_csv) if args.clip_ids_csv is not None else None
    dataset = CachedStylePairDataset(
        args.pair_cache,
        crop_frames=int(components.config["crop_frames"]),
        reference_context_frames=1,
        split="validation" if clip_ids is not None else "train",
        clip_id=args.clip_id,
        clip_ids=clip_ids,
        fixed_crop_start=0 if args.clip_id is not None else None,
    )
    replacement_reference = None
    if args.reference_clip_id is not None:
        replacement_dataset = CachedStylePairDataset(
            args.pair_cache,
            crop_frames=int(components.config["crop_frames"]),
            reference_context_frames=1,
            split="train",
            clip_id=args.reference_clip_id,
            fixed_crop_start=0,
        )
        replacement_reference = replacement_dataset[0]
    paired_values: dict[str, list[torch.Tensor]] = {
        "clip_index": [],
        "frame_index": [],
        "target_semantic": [],
        "reference_ce": [],
        "null_ce": [],
        "reference_correct": [],
        "null_correct": [],
        "semantic_transition": [],
    }
    per_clip = []
    for clip_index in range(len(dataset)):
        trace_sample = dataset[clip_index]
        if replacement_reference is not None:
            trace_sample = substituted_reference_sample(trace_sample, replacement_reference)
        trace_sample = shifted_reference_sample(trace_sample, args.reference_frame_shift)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            reference_trace = teacher_forced_semantic_trace(components, trace_sample, null_reference=False)
            null_trace = teacher_forced_semantic_trace(components, trace_sample, null_reference=True)
        target_semantic = reference_trace.target_semantic
        semantic_transition = torch.zeros_like(target_semantic, dtype=torch.bool)
        semantic_transition[1:] = target_semantic[1:] != target_semantic[:-1]
        frame_count = target_semantic.shape[0]
        clip_summary = summarize_teacher_forced_pair(
            reference_trace.cross_entropy,
            null_trace.cross_entropy,
            reference_trace.correct,
            null_trace.correct,
            semantic_transition,
        )
        per_clip.append({"clip_id": trace_sample["clip_id"], **clip_summary})
        paired_values["clip_index"].append(torch.full((frame_count,), clip_index, dtype=torch.int32))
        paired_values["frame_index"].append(torch.arange(frame_count, dtype=torch.int32))
        paired_values["target_semantic"].append(target_semantic.to(torch.int16))
        paired_values["reference_ce"].append(reference_trace.cross_entropy)
        paired_values["null_ce"].append(null_trace.cross_entropy)
        paired_values["reference_correct"].append(reference_trace.correct)
        paired_values["null_correct"].append(null_trace.correct)
        paired_values["semantic_transition"].append(semantic_transition)
    paired_trace = {name: torch.cat(values) for name, values in paired_values.items()}
    paired_summary = summarize_teacher_forced_pair(
        paired_trace["reference_ce"],
        paired_trace["null_ce"],
        paired_trace["reference_correct"],
        paired_trace["null_correct"],
        paired_trace["semantic_transition"],
    )
    paired_summary["transition_definition"] = "target semantic code differs from the preceding frame"
    paired_summary["per_clip"] = per_clip
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_file(
        paired_trace,
        args.output_dir / "teacher_forced_reference_null.safetensors",
        metadata={"clip_ids": json.dumps([values["clip_id"] for values in per_clip])},
    )
    plot_teacher_forced_pair(paired_trace, args.output_dir)
    if args.teacher_forced_only:
        results = {
            "control_strength": args.control_strength,
            "reference_frame_shift": args.reference_frame_shift,
            "reference_clip_id": args.reference_clip_id,
            "checkpoint_config": components.config,
            "teacher_forced_reference_null": paired_summary,
        }
        (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(results, indent=2), flush=True)
        return

    sample = dataset[0]
    if replacement_reference is not None:
        sample = substituted_reference_sample(sample, replacement_reference)
    sample = shifted_reference_sample(sample, args.reference_frame_shift)
    frame_count = min(sample["target_codes"].shape[0], round(args.max_seconds * FRAME_RATE))
    if args.perturb_frame is not None and not 0 <= args.perturb_frame < frame_count:
        raise ValueError(f"perturb-frame must be between 0 and {frame_count - 1}")
    teacher_forced = evaluate(
        components.training_model,
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
        "reference_frame_shift": args.reference_frame_shift,
        "reference_clip_id": args.reference_clip_id,
        "checkpoint_config": components.config,
        "teacher_forced": teacher_forced,
        "teacher_forced_reference_null": paired_summary,
        "modes": {name: rollout_metrics(trace, sample["target_codes"]) for name, trace in traces.items()},
    }
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
