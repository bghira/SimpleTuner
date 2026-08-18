#!/usr/bin/env python3
# Copyright 2026 SimpleTuner contributors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from diffusers import ModularPipeline
from safetensors.torch import save_file
from transformers import AutoTokenizer, Qwen3ForCausalLM

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_DIR = REPO_ROOT / "model_cards" / "collection"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COLLECTION_DIR))

from benchmark_reference_feedback import place_components, render_traces  # noqa: E402
from minimax_music3_reference_adapter import install_diffusers_reference_adapter  # noqa: E402

from scripts.minimax_music3.eval_reference_control import copy_pair_audio  # noqa: E402
from scripts.minimax_music3.train_prefix_distillation import (  # noqa: E402
    AUDIO_CODE_OFFSET,
    AUDIO_VOCAB_SIZE,
    PREFIX_FORMAT,
    TEACHER_FORMAT,
    CachedStylePairDataset,
    conditioned_hidden_states,
    distillation_step,
    prefix_warmup_codes,
    reference_prefix_embeddings,
    semantic_logits,
    teacher_targets,
    tokenize_prompt,
)
from simpletuner.helpers.models.minimaxmusic.reference_control import (  # noqa: E402
    create_qwen_lokr_adapter,
    create_qwen_oftv2_adapter,
    embed_rvq_frames,
    prefix_adapter_checkpoint_filename,
)
from simpletuner.helpers.models.minimaxmusic.rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder  # noqa: E402

FRAME_RATE = 25.0


def parse_guidance_scales(value: str) -> tuple[float, ...]:
    scales = tuple(float(item) for item in value.split(",") if item.strip())
    if not scales or any(scale < 0.0 for scale in scales):
        raise argparse.ArgumentTypeError("reference guidance scales must be a comma-separated list of non-negative values")
    return scales


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MiniMax Music 3 reference-prefix distillation")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-seconds", type=float, default=10.24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--disable-kv-cache", action="store_true")
    parser.add_argument("--reference-guidance-scales", type=parse_guidance_scales, default=())
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--ar-device", default="cuda:0")
    parser.add_argument("--render-device", default="cuda:1")
    parser.add_argument("--skip-render", action="store_true")
    return parser.parse_args()


def select_code(logits: torch.Tensor, *, sample: bool, top_k: int, generator: torch.Generator) -> torch.Tensor:
    if not sample:
        return logits.argmax(dim=-1)
    values, indices = torch.topk(logits.float(), min(top_k, logits.shape[-1]), dim=-1)
    selected = torch.multinomial(values.softmax(dim=-1).cpu(), 1, generator=generator).to(indices.device)
    return indices.gather(-1, selected).squeeze(-1)


def generate_frame(
    language_model,
    depth_decoder,
    last_hidden: torch.Tensor,
    *,
    sample: bool,
    top_k: int,
    generator: torch.Generator,
    semantic_override_logits: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    semantic = select_code(
        (
            semantic_override_logits
            if semantic_override_logits is not None
            else semantic_logits(language_model, last_hidden.unsqueeze(1))[:, 0]
        ),
        sample=sample,
        top_k=top_k,
        generator=generator,
    )
    sequence = [depth_decoder.projection(last_hidden).unsqueeze(1)]
    semantic_embedding = language_model.model.embed_tokens(semantic + AUDIO_CODE_OFFSET)
    sequence.append(depth_decoder.projection(semantic_embedding).unsqueeze(1))
    codes = [semantic]
    depth_hiddens = []
    for codebook in range(1, depth_decoder.config.num_codebooks):
        depth_hidden = depth_decoder(torch.cat(sequence, dim=1))[:, -1]
        depth_hiddens.append(depth_hidden)
        code = select_code(
            depth_decoder.audio_heads[codebook - 1](depth_hidden).float(),
            sample=sample,
            top_k=top_k,
            generator=generator,
        )
        codes.append(code)
        if codebook < depth_decoder.config.num_codebooks - 1:
            embedding = depth_decoder.audio_embeddings(code + (codebook - 1) * depth_decoder.config.audio_vocab_size)
            sequence.append(depth_decoder.projection(embedding).unsqueeze(1))
    frame_hidden = torch.cat((last_hidden.float(), torch.cat(depth_hiddens, dim=-1).float()), dim=-1)
    return torch.stack(codes, dim=-1), frame_hidden


@torch.inference_mode()
def rollout(
    pipeline,
    oft_network,
    sample_data: dict,
    frame_count: int,
    *,
    null_reference: bool,
    sample_codes: bool,
    top_k: int,
    seed: int,
    disable_kv_cache: bool,
    warmup_mode: str,
    reference_guidance_scale: float | None = None,
) -> SimpleNamespace:
    if null_reference and reference_guidance_scale is not None:
        raise ValueError("reference guidance requires the conditioned reference rollout")
    language_model = pipeline.language_model
    depth_decoder = pipeline.rvq_depth_decoder
    device = next(language_model.parameters()).device
    target_codes = sample_data["target_codes"][:frame_count].to(device).unsqueeze(0)
    reference_codes = sample_data["reference_codes"].to(device).unsqueeze(0)
    text_ids = tokenize_prompt(pipeline.tokenizer, sample_data["prompt"], sample_data["lyrics"], device)
    text_embeddings = language_model.model.embed_tokens(text_ids)
    prefix_embeddings = reference_prefix_embeddings(
        language_model,
        depth_decoder,
        pipeline.tokenizer,
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
    warmup_embeddings = embed_rvq_frames(language_model, depth_decoder, warmup_codes)
    output = language_model.model(
        inputs_embeds=torch.cat((prefix_embeddings, text_embeddings, warmup_embeddings), dim=1),
        use_cache=not disable_kv_cache,
    )
    past_key_values = output.past_key_values
    last_hidden = output.last_hidden_state[:, -1]
    null_prefix_embeddings = None
    null_past_key_values = None
    null_last_hidden = None
    if reference_guidance_scale is not None:
        null_prefix_embeddings = reference_prefix_embeddings(
            language_model,
            depth_decoder,
            pipeline.tokenizer,
            reference_codes,
            null_reference=True,
        )
        null_output = language_model.model(
            inputs_embeds=torch.cat((null_prefix_embeddings, text_embeddings, warmup_embeddings), dim=1),
            use_cache=not disable_kv_cache,
        )
        null_past_key_values = null_output.past_key_values
        null_last_hidden = null_output.last_hidden_state[:, -1]
    clean_feedback = torch.cat((warmup_codes, target_codes[:, :-1]), dim=1)
    teacher_forced_hidden = conditioned_hidden_states(
        language_model,
        depth_decoder,
        prefix_embeddings,
        text_embeddings,
        clean_feedback,
    )
    prefill_hidden_cosine = F.cosine_similarity(last_hidden.float(), teacher_forced_hidden[:, 0].float()).item()
    teacher_forced_first_logits = semantic_logits(language_model, teacher_forced_hidden[:, :1])[:, 0]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    generated_codes = []
    frame_hiddens = []
    target_log_probs = []
    target_top1 = []
    first_top1 = None
    started = time.perf_counter()
    for frame_index in range(frame_count):
        logits = semantic_logits(language_model, last_hidden.unsqueeze(1))[:, 0]
        if reference_guidance_scale is not None:
            null_logits = semantic_logits(language_model, null_last_hidden.unsqueeze(1))[:, 0]
            logits = null_logits + reference_guidance_scale * (logits - null_logits)
        if frame_index == 0:
            first_top1 = (logits.argmax(dim=-1)[0] == target_codes[0, 0, 0]).item()
        target_log_probs.append(logits.log_softmax(dim=-1)[0, target_codes[0, frame_index, 0]].cpu())
        target_top1.append((logits.argmax(dim=-1)[0] == target_codes[0, frame_index, 0]).cpu())
        codes, frame_hidden = generate_frame(
            language_model,
            depth_decoder,
            last_hidden,
            sample=sample_codes,
            top_k=top_k,
            generator=generator,
            semantic_override_logits=logits,
        )
        generated_codes.append(codes.cpu())
        frame_hiddens.append(frame_hidden.cpu())
        if frame_index + 1 < frame_count:
            if disable_kv_cache:
                generated_feedback = torch.cat(
                    (warmup_codes, *[value.to(device).unsqueeze(1) for value in generated_codes]),
                    dim=1,
                )
                feedback = embed_rvq_frames(language_model, depth_decoder, generated_feedback)
                output = language_model.model(
                    inputs_embeds=torch.cat((prefix_embeddings, text_embeddings, feedback), dim=1),
                    use_cache=False,
                )
                if reference_guidance_scale is not None:
                    null_output = language_model.model(
                        inputs_embeds=torch.cat((null_prefix_embeddings, text_embeddings, feedback), dim=1),
                        use_cache=False,
                    )
            else:
                feedback = embed_rvq_frames(language_model, depth_decoder, codes.unsqueeze(1))
                output = language_model.model(inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True)
                past_key_values = output.past_key_values
                if reference_guidance_scale is not None:
                    null_output = language_model.model(
                        inputs_embeds=feedback,
                        past_key_values=null_past_key_values,
                        use_cache=True,
                    )
                    null_past_key_values = null_output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]
            if reference_guidance_scale is not None:
                null_last_hidden = null_output.last_hidden_state[:, -1]
    return SimpleNamespace(
        generated_codes=torch.cat(generated_codes, dim=0),
        frame_hiddens=torch.cat(frame_hiddens, dim=0).unsqueeze(0).to(torch.bfloat16),
        target_semantic_log_probs=torch.stack(target_log_probs),
        target_semantic_top1=torch.stack(target_top1),
        warmup_feedback_codes=warmup_codes.cpu(),
        prefill_hidden_cosine=prefill_hidden_cosine,
        teacher_forced_first_top1=(
            first_top1
            if reference_guidance_scale is not None
            else (teacher_forced_first_logits.argmax(dim=-1)[0] == target_codes[0, 0, 0]).item()
        ),
        ar_seconds=time.perf_counter() - started,
    )


def sustained_relock_frame(matches: torch.Tensor, consecutive_frames: int) -> int | None:
    misses = (~matches).nonzero().flatten()
    if not misses.numel():
        return None
    for start in range(int(misses[0]) + 1, matches.shape[0] - consecutive_frames + 1):
        if bool(matches[start : start + consecutive_frames].all()):
            return start
    return None


def trace_metrics(trace, target_codes: torch.Tensor, teacher_frame_hidden: torch.Tensor) -> dict:
    target = target_codes[: trace.generated_codes.shape[0]]
    agreement = (trace.generated_codes == target).float().mean(dim=0)
    c0_matches = trace.generated_codes[:, 0] == target[:, 0]
    misses = (~c0_matches).nonzero().flatten()
    reacquisitions = (~c0_matches[:-1] & c0_matches[1:]).sum()
    target_top1_reacquisitions = (~trace.target_semantic_top1[:-1] & trace.target_semantic_top1[1:]).sum()
    cosine = F.cosine_similarity(trace.frame_hiddens.float(), teacher_frame_hidden.float(), dim=-1).flatten()
    return {
        "ar_seconds": round(trace.ar_seconds, 3),
        "semantic_target_top1": agreement[0].item(),
        "acoustic_target_top1": agreement[1:].mean().item(),
        "per_codebook_target_top1": agreement.tolist(),
        "first_semantic_miss": int(misses[0]) if misses.numel() else None,
        "semantic_match_frames": c0_matches.nonzero().flatten().tolist(),
        "semantic_reacquisitions": int(reacquisitions),
        "semantic_sustained_relock_2": sustained_relock_frame(c0_matches, 2),
        "semantic_sustained_relock_4": sustained_relock_frame(c0_matches, 4),
        "target_semantic_log_prob_mean": trace.target_semantic_log_probs.mean().item(),
        "target_semantic_top1_mean": trace.target_semantic_top1.float().mean().item(),
        "target_semantic_top1_reacquisitions": int(target_top1_reacquisitions),
        "teacher_hidden_cosine_mean": cosine.mean().item(),
        "teacher_hidden_cosine_p05": torch.quantile(cosine, 0.05).item(),
        "prefill_hidden_cosine": trace.prefill_hidden_cosine,
        "teacher_forced_first_top1": trace.teacher_forced_first_top1,
        "warmup_codes": trace.warmup_feedback_codes[0, 0].tolist() if trace.warmup_feedback_codes is not None else None,
    }


def load_prefix_config(checkpoint_dir: Path) -> dict:
    config_path = checkpoint_dir / "prefix_distillation.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") != PREFIX_FORMAT:
        raise ValueError(f"Unsupported prefix checkpoint format: {config.get('format')}")
    if config.get("teacher_format") != TEACHER_FORMAT:
        raise ValueError(f"Unsupported teacher format: {config.get('teacher_format')}")
    return config


def load_prefix_checkpoint(pipeline, checkpoint_dir: Path, ar_device: torch.device, config: dict):
    adapter_type = config.get("adapter_type", "oftv2")
    weights_path = checkpoint_dir / prefix_adapter_checkpoint_filename(adapter_type)
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    if adapter_type == "oftv2":
        oft_network = create_qwen_oftv2_adapter(
            pipeline.language_model,
            block_size=int(config["oft_block_size"]),
        )
    elif adapter_type == "lokr":
        oft_network = create_qwen_lokr_adapter(
            pipeline.language_model,
            rank=int(config["lokr_rank"]),
            alpha=float(config["lokr_alpha"]),
        )
    else:
        raise ValueError(f"Unsupported prefix adapter type: {adapter_type}")
    load_state = oft_network.load_weights(str(weights_path))
    if load_state:
        raise RuntimeError(f"LyCORIS checkpoint load mismatch: {load_state}")
    oft_network.to(ar_device)
    oft_network.eval()
    return oft_network, config


def load_ar_components(args, ar_device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, subfolder="tokenizer", cache_dir=args.cache_dir)
    language_model = Qwen3ForCausalLM.from_pretrained(
        args.model_id,
        subfolder="language_model",
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
    ).to(ar_device)
    depth_decoder = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
        args.model_id,
        subfolder="rvq_depth_decoder",
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
    ).to(ar_device)
    language_model.eval()
    depth_decoder.eval()
    return SimpleNamespace(
        tokenizer=tokenizer,
        language_model=language_model,
        rvq_depth_decoder=depth_decoder,
    )


@torch.inference_mode()
def teacher_forced_metrics(pipeline, oft_network, sample_data: dict, device: torch.device, config: dict) -> dict:
    modes = {}
    text_ids = tokenize_prompt(pipeline.tokenizer, sample_data["prompt"], sample_data["lyrics"], device)
    text_embeddings = pipeline.language_model.model.embed_tokens(text_ids)
    reference_codes = sample_data["reference_codes"].to(device).unsqueeze(0)
    for name, null_reference in (("reference", False), ("null", True)):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            _, metrics = distillation_step(
                pipeline.language_model,
                pipeline.rvq_depth_decoder,
                oft_network,
                pipeline.tokenizer,
                sample_data,
                device,
                feedback_corruption_rate=0.0,
                feedback_sampling_top_k=int(config["feedback_sampling_top_k"]),
                reference_dropout=0.0,
                teacher_kl_weight=float(config["teacher_kl_weight"]),
                teacher_kl_temperature=float(config["teacher_kl_temperature"]),
                hidden_alignment_weight=float(config["hidden_alignment_weight"]),
                prefix_anchor_frames=int(config["prefix_anchor_frames"]),
                prefix_anchor_weight=float(config["prefix_anchor_weight"]),
                semantic_loss_weight=float(config.get("semantic_loss_weight", 1.0)),
                warmup_mode=config.get("warmup_mode", "generated"),
                sequential_feedback_rollout=False,
                force_null_reference=null_reference,
            )
            prefix_embeddings = reference_prefix_embeddings(
                pipeline.language_model,
                pipeline.rvq_depth_decoder,
                pipeline.tokenizer,
                reference_codes,
                null_reference=null_reference,
            )
            warmup_codes = prefix_warmup_codes(
                pipeline.language_model,
                pipeline.rvq_depth_decoder,
                prefix_embeddings,
                text_embeddings,
                reference_codes,
                config.get("warmup_mode", "generated"),
            )
        metrics["warmup_codes"] = warmup_codes[0, 0].tolist()
        modes[name] = metrics
    return modes


def main() -> None:
    args = parse_args()
    if args.max_seconds <= 0.0:
        raise ValueError("max-seconds must be positive")
    if not 1 <= args.top_k <= AUDIO_VOCAB_SIZE:
        raise ValueError(f"top-k must be between 1 and {AUDIO_VOCAB_SIZE}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_prefix_config(args.checkpoint_dir)
    dataset = CachedStylePairDataset(
        args.pair_cache,
        crop_frames=int(config["crop_frames"]),
        reference_context_frames=0,
        clip_id=args.clip_id,
        fixed_crop_start=0,
    )
    sample_data = dataset[0]
    frame_count = min(sample_data["target_codes"].shape[0], round(args.max_seconds * FRAME_RATE))
    ar_device = torch.device(args.ar_device)
    render_device = torch.device(args.render_device)
    if args.skip_render:
        pipeline = load_ar_components(args, ar_device)
    else:
        install_diffusers_reference_adapter()
        pipeline = ModularPipeline.from_pretrained(args.model_id, cache_dir=args.cache_dir)
        pipeline.load_components(dtype=torch.bfloat16)
        place_components(pipeline, ar_device, render_device)
    oft_network, config = load_prefix_checkpoint(pipeline, args.checkpoint_dir, ar_device, config)
    checkpoint_teacher_forced = teacher_forced_metrics(pipeline, oft_network, sample_data, ar_device, config)
    traces = {
        "reference": rollout(
            pipeline,
            oft_network,
            sample_data,
            frame_count,
            null_reference=False,
            sample_codes=args.sample,
            top_k=args.top_k,
            seed=args.seed,
            disable_kv_cache=args.disable_kv_cache,
            warmup_mode=config.get("warmup_mode", "generated"),
        ),
        "null": rollout(
            pipeline,
            oft_network,
            sample_data,
            frame_count,
            null_reference=True,
            sample_codes=args.sample,
            top_k=args.top_k,
            seed=args.seed,
            disable_kv_cache=args.disable_kv_cache,
            warmup_mode=config.get("warmup_mode", "generated"),
        ),
    }
    for scale in args.reference_guidance_scales:
        traces[f"reference_cfg_{scale:g}"] = rollout(
            pipeline,
            oft_network,
            sample_data,
            frame_count,
            null_reference=False,
            sample_codes=args.sample,
            top_k=args.top_k,
            seed=args.seed,
            disable_kv_cache=args.disable_kv_cache,
            warmup_mode=config.get("warmup_mode", "generated"),
            reference_guidance_scale=scale,
        )
    all_target_codes = sample_data["target_codes"].to(ar_device).unsqueeze(0)
    text_ids = tokenize_prompt(pipeline.tokenizer, sample_data["prompt"], sample_data["lyrics"], ar_device)
    teacher_logits, teacher_frame_hidden = teacher_targets(
        pipeline.language_model,
        pipeline.rvq_depth_decoder,
        oft_network,
        pipeline.tokenizer,
        text_ids,
        all_target_codes,
        config.get("warmup_mode", "generated"),
    )
    teacher_logits = teacher_logits[:, :frame_count]
    teacher_frame_hidden = teacher_frame_hidden[:, :frame_count]
    target_codes = all_target_codes[:, :frame_count]
    traces["teacher_target_replay"] = SimpleNamespace(
        generated_codes=target_codes[0].cpu(),
        frame_hiddens=teacher_frame_hidden.cpu().to(torch.bfloat16),
        target_semantic_log_probs=F.log_softmax(teacher_logits[0], dim=-1)
        .gather(-1, target_codes[0, :, :1])
        .squeeze(-1)
        .cpu(),
        target_semantic_top1=torch.ones(frame_count, dtype=torch.bool),
        warmup_feedback_codes=None,
        prefill_hidden_cosine=1.0,
        teacher_forced_first_top1=True,
        ar_seconds=0.0,
    )
    metrics = {
        "clip_id": args.clip_id,
        "frames": frame_count,
        "seconds": frame_count / FRAME_RATE,
        "sample": args.sample,
        "top_k": args.top_k,
        "disable_kv_cache": args.disable_kv_cache,
        "reference_guidance_scales": args.reference_guidance_scales,
        "checkpoint_config": config,
        "checkpoint_teacher_forced": checkpoint_teacher_forced,
        "modes": {
            name: trace_metrics(trace, sample_data["target_codes"], teacher_frame_hidden.cpu())
            for name, trace in traces.items()
        },
    }
    for name, trace in traces.items():
        save_file(
            {
                "generated_codes": trace.generated_codes,
                "frame_hiddens": trace.frame_hiddens,
                "target_semantic_log_probs": trace.target_semantic_log_probs,
                "target_semantic_top1": trace.target_semantic_top1,
            },
            str(args.output_dir / f"{name}.safetensors"),
        )
    copy_pair_audio(sample_data["metadata"], args.output_dir, args.cache_dir)
    if not args.skip_render:
        metrics["render"] = render_traces(
            pipeline,
            traces,
            args.output_dir,
            args.num_inference_steps,
            args.seed,
        )
    (args.output_dir / "results.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    del oft_network
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
