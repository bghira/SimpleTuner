#!/usr/bin/env python3
# Copyright 2026 SimpleTuner contributors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from diffusers import ModularPipeline
from safetensors import safe_open
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_DIR = REPO_ROOT / "model_cards" / "collection"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COLLECTION_DIR))

from minimax_music3_reference_adapter import (  # noqa: E402
    AR_CFG_SCALE,
    AR_TOP_K,
    AUDIO_CODE_OFFSET,
    AUDIO_END_TOKEN_ID,
    FRAME_RATE,
    SEMANTIC_VOCAB_SIZE,
    ReferenceRolloutTrace,
    _embed_official_codes,
    _sample_official_depth_codes,
    _sample_top_k,
    build_text_ids,
    install_diffusers_reference_adapter,
)

from scripts.minimax_music3.cache_style_pairs import member_index, read_member, shard_path  # noqa: E402
from scripts.minimax_music3.train_reference_control import greedy_warmup_codes, train_step  # noqa: E402
from simpletuner.helpers.models.minimaxmusic.reference_control import (  # noqa: E402
    MiniMaxMusic3ReferenceControlAdapter,
    ReferenceControlConfig,
    create_qwen_oftv2_adapter,
    embed_rvq_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a MiniMax Music 3 reference-control checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path)
    parser.add_argument("--reference-audio", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--lyrics-file", type=Path)
    parser.add_argument("--source-caption", default="user-provided reference audio")
    parser.add_argument("--swap-clip-id")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clip-id")
    parser.add_argument("--model-id", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=AR_CFG_SCALE)
    parser.add_argument("--top-k", type=int, default=AR_TOP_K)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--oft-block-size", type=int, default=64)
    parser.add_argument("--ar-device", default="cuda:0")
    parser.add_argument("--render-device", default="cuda:1")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--skip-pair-audio", action="store_true")
    parser.add_argument("--training-compatible-warmup", action="store_true")
    parser.add_argument("--greedy-rollout", action="store_true")
    parser.add_argument("--run-recovery-diagnostics", action="store_true")
    parser.add_argument("--feedback-probabilities", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--relock-consecutive-frames", type=int, default=4)
    return parser.parse_args()


def parse_probabilities(value: str) -> tuple[float, ...]:
    try:
        probabilities = tuple(float(item) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("feedback-probabilities must be comma-separated numbers") from exc
    if not probabilities or any(probability < 0.0 or probability > 1.0 for probability in probabilities):
        raise ValueError("feedback-probabilities must contain values in [0, 1]")
    return probabilities


def is_validation_clip(clip_id: str, validation_fraction: float) -> bool:
    threshold = int(validation_fraction * 2**64)
    digest = hashlib.sha256(clip_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") < threshold


def find_pair_by_clip_id(pair_cache: Path, clip_id: str) -> Path:
    matches = list(pair_cache.glob(f"shard-*/{clip_id}.safetensors"))
    if len(matches) != 1:
        raise ValueError(f"Expected one cached pair for {clip_id}, found {len(matches)}")
    return matches[0]


def select_pair_path(
    pair_cache: Path,
    clip_id: str | None,
    validation_fraction: float,
) -> Path:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation-fraction must be in (0, 1)")
    if clip_id is not None:
        path = find_pair_by_clip_id(pair_cache, clip_id)
        if not is_validation_clip(clip_id, validation_fraction):
            raise ValueError(f"Clip {clip_id} is not in the configured validation split")
        return path
    for path in sorted(pair_cache.glob("shard-*/*.safetensors")):
        if is_validation_clip(path.stem, validation_fraction):
            return path
    raise ValueError(f"No validation pairs found under {pair_cache}")


def load_pair(path: Path) -> tuple[dict[str, str], torch.Tensor, torch.Tensor]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        reference_codes = handle.get_tensor("reference_codes").long()
        target_codes = handle.get_tensor("target_codes").long()
    if metadata is None:
        raise ValueError(f"Cached pair {path.name} has no metadata")
    if metadata.get("clip_id") != path.stem:
        raise ValueError(f"Cached pair {path.name} has inconsistent clip metadata")
    return metadata, reference_codes, target_codes


def encode_reference_audio(
    path: Path,
    *,
    device: torch.device,
    cache_dir: str | None,
) -> torch.Tensor:
    from minimax_music3_reference_adapter import MiniMaxMusic3ReferenceAdapter

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    encoder = MiniMaxMusic3ReferenceAdapter.from_pretrained(cache_dir=cache_dir)
    codes = encoder.predict_codes(
        torch.from_numpy(audio.T.copy()),
        sample_rate,
        device=device,
        offload_after=True,
        dav_chunk_seconds=30.0,
    )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return codes.long()


def mapped_reference_positions(
    output_frame_count: int,
    target_frame_count: int,
    reference_frame_count: int,
) -> torch.Tensor:
    if output_frame_count < 1:
        raise ValueError("output_frame_count must be positive")
    if target_frame_count < output_frame_count:
        raise ValueError("output_frame_count exceeds the target timeline")
    positions = torch.arange(output_frame_count, dtype=torch.float32)
    if target_frame_count > 1 and reference_frame_count > 1:
        positions *= (reference_frame_count - 1) / (target_frame_count - 1)
    return positions


def load_checkpoint(pipeline, checkpoint_dir: Path, ar_device: torch.device, oft_block_size: int):
    oft_path = checkpoint_dir / "qwen_oftv2.safetensors"
    config_path = checkpoint_dir / "reference_control.json"
    adapter_path = checkpoint_dir / "reference_control.safetensors"
    for path in (oft_path, config_path, adapter_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    oft_network = create_qwen_oftv2_adapter(pipeline.language_model, block_size=oft_block_size)
    oft_network.load_weights(str(oft_path))
    oft_network.to(ar_device)
    oft_network.eval()
    config = ReferenceControlConfig.from_dict(json.loads(config_path.read_text(encoding="utf-8")))
    adapter = MiniMaxMusic3ReferenceControlAdapter(config)
    adapter.load_state_dict(load_file(str(adapter_path)), strict=True)
    adapter.to(ar_device)
    adapter.install(pipeline.language_model)
    adapter.eval()
    return adapter, oft_network


def _sample_warmup_codes(
    language_model,
    depth_decoder,
    last_hidden: torch.Tensor,
    generator: torch.Generator,
    cfg_scale: float,
    top_k: int,
) -> torch.Tensor:
    vocab_mask = torch.ones(language_model.config.vocab_size, dtype=torch.bool, device=last_hidden.device)
    vocab_mask[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE] = False
    vocab_mask[AUDIO_END_TOKEN_ID] = False
    logits = language_model.lm_head(last_hidden).float().masked_fill(vocab_mask, -torch.inf)
    conditioned, unconditioned = logits[:1], logits[1:2]
    guided = unconditioned + (conditioned - unconditioned) * cfg_scale
    threshold = torch.topk(conditioned, top_k, dim=-1).values[..., -1, None]
    token = _sample_top_k(guided.masked_fill(conditioned < threshold, -torch.inf), generator, top_k)
    if int(token.item()) == AUDIO_END_TOKEN_ID:
        raise ValueError("the selected seed ended during the required AR warm-up frame")
    semantic = (token - AUDIO_CODE_OFFSET).repeat(2)
    codes, _ = _sample_official_depth_codes(
        language_model,
        depth_decoder,
        last_hidden,
        semantic,
        generator,
        cfg_scale=cfg_scale,
        top_k=top_k,
    )
    return codes


@torch.inference_mode()
def rollout_reference_control(
    pipeline,
    adapter: MiniMaxMusic3ReferenceControlAdapter,
    reference_codes: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    prompt: str,
    lyrics: str,
    seed: int,
    cfg_scale: float,
    top_k: int,
    use_reference: bool,
    training_compatible_warmup: bool,
    greedy_rollout: bool,
    target_codes: torch.Tensor | None,
    target_feedback_probability: float = 0.0,
    teacher_force_before_frame: int | None = None,
) -> ReferenceRolloutTrace:
    if not 0.0 <= target_feedback_probability <= 1.0:
        raise ValueError("target_feedback_probability must be in [0, 1]")
    if teacher_force_before_frame is not None and teacher_force_before_frame < 1:
        raise ValueError("teacher_force_before_frame must be positive")
    if target_codes is None and (target_feedback_probability > 0.0 or teacher_force_before_frame is not None):
        raise ValueError("target feedback requires target_codes")
    if teacher_force_before_frame is not None and target_feedback_probability > 0.0:
        raise ValueError("teacher_force_before_frame and target_feedback_probability are mutually exclusive")
    language_model = pipeline.language_model
    depth_decoder = pipeline.rvq_depth_decoder
    device = next(language_model.parameters()).device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    feedback_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    text_ids = build_text_ids(pipeline.tokenizer, prompt, lyrics, device)
    text_output = language_model.model(
        inputs_embeds=language_model.model.embed_tokens(text_ids),
        use_cache=True,
    )
    past = text_output.past_key_values
    last_hidden = text_output.last_hidden_state[:, -1]
    if training_compatible_warmup:
        warmup_codes = greedy_warmup_codes(language_model, depth_decoder, text_ids[:1])[:, 0].repeat(2, 1)
    else:
        warmup_codes = _sample_warmup_codes(
            language_model,
            depth_decoder,
            last_hidden,
            generator,
            cfg_scale,
            top_k,
        )

    source_codes = reference_codes.to(device=device, dtype=torch.long).unsqueeze(0)
    source_embeds = embed_rvq_frames(language_model, depth_decoder, source_codes)
    encoded_reference = adapter.encode_reference(source_embeds)
    conditional_memory = encoded_reference if use_reference else encoded_reference * 0.0
    reference_memory = torch.cat((conditional_memory, encoded_reference * 0.0), dim=0)
    key_positions = torch.arange(source_codes.shape[1], device=device, dtype=torch.float32).repeat(2, 1)
    query_positions = query_positions.to(device=device, dtype=torch.float32)
    if target_codes is not None:
        target_codes = target_codes[: query_positions.shape[0]].to(device=device, dtype=torch.long)

    def feed_codes(codes: torch.Tensor, frame_index: int):
        position = query_positions[frame_index].reshape(1, 1).repeat(2, 1)
        return language_model.model(
            inputs_embeds=_embed_official_codes(language_model, depth_decoder, codes).unsqueeze(1),
            past_key_values=past,
            use_cache=True,
            reference_memory=reference_memory,
            reference_query_positions=position,
            reference_key_positions=key_positions,
            reference_query_start=0,
        )

    output = feed_codes(warmup_codes, 0)
    past = output.past_key_values
    last_hidden = output.last_hidden_state[:, -1]
    hidden_frames = []
    generated_frames = []
    target_log_probs = []
    target_top1 = []
    target_feedback = []
    for frame_index in range(query_positions.shape[0]):
        token_logits = language_model.lm_head(last_hidden).float()
        conditioned = token_logits[:1, AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE]
        unconditioned = token_logits[1:2, AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + SEMANTIC_VOCAB_SIZE]
        guided = unconditioned + (conditioned - unconditioned) * cfg_scale
        if target_codes is not None:
            target = target_codes[frame_index, 0]
            target_log_probs.append(conditioned.log_softmax(dim=-1)[0, target].cpu())
            target_top1.append((conditioned.argmax(dim=-1)[0] == target).cpu())
        semantic = (guided.argmax(dim=-1) if greedy_rollout else _sample_top_k(guided, generator, top_k)).repeat(2)
        sampled_codes, depth_hidden = _sample_official_depth_codes(
            language_model,
            depth_decoder,
            last_hidden,
            semantic,
            generator,
            cfg_scale=cfg_scale,
            top_k=top_k,
            greedy=greedy_rollout,
        )
        hidden_frames.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1).cpu())
        generated_frames.append(sampled_codes[:1].cpu())
        use_target_feedback = False
        if frame_index + 1 < query_positions.shape[0]:
            if teacher_force_before_frame is not None:
                use_target_feedback = frame_index < teacher_force_before_frame - 1
            elif target_feedback_probability > 0.0:
                use_target_feedback = target_feedback_probability >= 1.0 or (
                    torch.rand((), generator=feedback_generator).item() < target_feedback_probability
                )
            if use_target_feedback:
                feedback_codes = target_codes[frame_index].unsqueeze(0).repeat(2, 1)
            else:
                feedback_codes = sampled_codes
                if (
                    teacher_force_before_frame is not None
                    and frame_index == teacher_force_before_frame - 1
                    and torch.equal(feedback_codes[:1], target_codes[frame_index].reshape(1, -1))
                ):
                    feedback_codes = feedback_codes.clone()
                    feedback_codes[:, -1] = (feedback_codes[:, -1] + 1) % depth_decoder.config.audio_vocab_size
            output = feed_codes(feedback_codes, frame_index + 1)
            past = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]
        target_feedback.append(use_target_feedback)

    return ReferenceRolloutTrace(
        frame_hiddens=torch.cat(hidden_frames, dim=0).unsqueeze(0),
        generated_codes=torch.cat(generated_frames, dim=0),
        reference_feedback=torch.tensor(target_feedback, dtype=torch.bool),
        warmup_feedback_codes=warmup_codes[:1].cpu(),
        target_semantic_log_probs=torch.stack(target_log_probs) if target_log_probs else None,
        target_semantic_top1=torch.stack(target_top1) if target_top1 else None,
    )


def copy_pair_audio(metadata: dict[str, str], output_dir: Path, cache_dir: str | None) -> None:
    shard_index = int(metadata["shard_index"])
    clip_id = metadata["clip_id"]
    repositories = (
        (metadata["source_repo"], "suno-various-94k", "source.mp3"),
        (metadata["target_repo"], "suno-various-94k-style-pairs", "ace_target.mp3"),
    )
    for repo_id, prefix, output_name in repositories:
        path = shard_path(repo_id, prefix, shard_index, cache_dir)
        with tarfile.open(path) as archive:
            sample = member_index(archive).get(clip_id)
            if sample is None or ".mp3" not in sample:
                raise ValueError(f"Audio for {clip_id} is missing from {repo_id}")
            (output_dir / output_name).write_bytes(read_member(archive, sample[".mp3"]))


def place_components(pipeline, ar_device: torch.device, render_device: torch.device) -> None:
    pipeline.language_model.to(ar_device)
    pipeline.rvq_depth_decoder.to(ar_device)
    pipeline.condition_encoder.to(render_device)
    pipeline.transformer.to(render_device)
    pipeline.vocoder.to(render_device)
    for name in ("language_model", "rvq_depth_decoder", "condition_encoder", "transformer", "vocoder"):
        getattr(pipeline, name).eval()


def audio_health(audio: torch.Tensor) -> dict[str, float]:
    values = audio.float()
    return {
        "rms": round(values.square().mean().sqrt().item(), 6),
        "peak": round(values.abs().max().item(), 6),
        "near_silence_fraction": round((values.abs() < 1e-4).float().mean().item(), 8),
    }


@torch.inference_mode()
def render_traces(
    pipeline,
    traces: dict[str, ReferenceRolloutTrace],
    output_dir: Path,
    steps: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    language_model = pipeline.language_model
    depth_decoder = pipeline.rvq_depth_decoder
    pipeline.language_model = None
    pipeline.rvq_depth_decoder = None
    results = {}
    try:
        for name, trace in traces.items():
            started = time.perf_counter()
            output = pipeline(
                frame_hiddens=trace.frame_hiddens,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                num_inference_steps=steps,
                output_type="pt",
            )
            audio = output.audios[0]
            sf.write(output_dir / f"{name}.flac", audio.float().cpu().T.numpy(), pipeline.sampling_rate)
            results[name] = {"seconds": round(time.perf_counter() - started, 3), **audio_health(audio)}
    finally:
        pipeline.language_model = language_model
        pipeline.rvq_depth_decoder = depth_decoder
    return results


def code_metrics(
    traces: dict[str, ReferenceRolloutTrace],
    reference_codes: torch.Tensor,
    target_codes: torch.Tensor | None,
    query_positions: torch.Tensor,
) -> dict[str, Any]:
    nearest_reference = reference_codes[query_positions.round().long()]
    generated_reference = traces["reference"].generated_codes
    generated_null = traces["null"].generated_codes

    def agreement(left: torch.Tensor, right: torch.Tensor) -> list[float]:
        return [round(value, 6) for value in (left == right).float().mean(dim=0).tolist()]

    metrics = {
        "reference_vs_null": agreement(generated_reference, generated_null),
        "reference_vs_source": agreement(generated_reference, nearest_reference),
        "null_vs_source": agreement(generated_null, nearest_reference),
    }
    if target_codes is not None:
        clipped_target = target_codes[: query_positions.shape[0]]
        metrics["reference_vs_ace_target"] = agreement(generated_reference, clipped_target)
        metrics["null_vs_ace_target"] = agreement(generated_null, clipped_target)
    return metrics


def build_teacher_forced_sample(
    metadata: dict[str, str],
    target_codes: torch.Tensor,
    reference_codes: torch.Tensor,
    frame_count: int,
    window_frames: int,
) -> dict[str, Any]:
    query_positions = mapped_reference_positions(frame_count, target_codes.shape[0], reference_codes.shape[0])
    reference_end = min(
        reference_codes.shape[0],
        int(torch.ceil(query_positions[-1]).item()) + window_frames + 1,
    )
    return {
        "clip_id": metadata["clip_id"],
        "prompt": metadata["prompt"],
        "lyrics": metadata["lyrics"],
        "target_codes": target_codes[:frame_count],
        "reference_codes": reference_codes[:reference_end],
        "query_positions": query_positions,
        "key_positions": torch.arange(reference_end, dtype=torch.float32),
    }


@torch.no_grad()
def teacher_forced_reference_comparison(
    pipeline,
    adapter: MiniMaxMusic3ReferenceControlAdapter,
    metadata: dict[str, str],
    target_codes: torch.Tensor,
    reference_codes: torch.Tensor,
    swapped_reference_codes: torch.Tensor,
    frame_count: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    samples = {
        "reference": build_teacher_forced_sample(
            metadata,
            target_codes,
            reference_codes,
            frame_count,
            adapter.config.window_frames,
        ),
        "swapped": build_teacher_forced_sample(
            metadata,
            target_codes,
            swapped_reference_codes,
            frame_count,
            adapter.config.window_frames,
        ),
    }
    results = {}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for name, sample in samples.items():
            _, metrics = train_step(
                pipeline.language_model,
                pipeline.rvq_depth_decoder,
                adapter,
                pipeline.tokenizer,
                sample,
                device,
                reference_dropout=0.0,
            )
            results[name] = metrics
        _, null_metrics = train_step(
            pipeline.language_model,
            pipeline.rvq_depth_decoder,
            adapter,
            pipeline.tokenizer,
            samples["reference"],
            device,
            reference_dropout=1.0,
        )
    results["null"] = null_metrics
    return results


def rollout_diagnostics(
    traces: dict[str, ReferenceRolloutTrace],
    target_codes: torch.Tensor,
) -> dict[str, Any]:
    diagnostics = {}
    for name, trace in traces.items():
        if trace.target_semantic_log_probs is None or trace.target_semantic_top1 is None:
            raise ValueError("target diagnostics are unavailable for this rollout")
        target = target_codes[: trace.generated_codes.shape[0]]
        semantic_match = (trace.generated_codes[:, 0] == target[:, 0]).float()
        book_agreement = (trace.generated_codes == target).float().mean(dim=-1)
        misses = (~trace.target_semantic_top1).nonzero().flatten()
        first_miss = int(misses[0]) if misses.numel() else None
        reacquisitions = int((~trace.target_semantic_top1[:-1] & trace.target_semantic_top1[1:]).sum())
        diagnostics[name] = {
            "target_semantic_log_prob_mean": trace.target_semantic_log_probs.mean().item(),
            "target_semantic_top1_mean": trace.target_semantic_top1.float().mean().item(),
            "generated_semantic_agreement_mean": semantic_match.mean().item(),
            "generated_all_book_agreement_mean": book_agreement.mean().item(),
            "first_target_top1_miss": first_miss,
            "target_top1_reacquisitions": reacquisitions,
            "per_frame_target_semantic_log_prob": trace.target_semantic_log_probs.tolist(),
            "per_frame_target_semantic_top1": trace.target_semantic_top1.tolist(),
            "per_frame_generated_book_agreement": book_agreement.tolist(),
        }
    return diagnostics


def sustained_relock_latency(
    target_top1: torch.Tensor,
    perturbation_frame: int,
    consecutive_frames: int,
) -> int | None:
    if target_top1.ndim != 1:
        raise ValueError("target_top1 must be one-dimensional")
    if not 0 <= perturbation_frame < target_top1.shape[0]:
        raise ValueError("perturbation_frame is outside target_top1")
    if consecutive_frames < 1:
        raise ValueError("consecutive_frames must be positive")
    last_start = target_top1.shape[0] - consecutive_frames
    for start in range(perturbation_frame, last_start + 1):
        if bool(target_top1[start : start + consecutive_frames].all()):
            return start - perturbation_frame
    return None


@torch.inference_mode()
def run_recovery_diagnostics(
    pipeline,
    adapter: MiniMaxMusic3ReferenceControlAdapter,
    reference_codes: torch.Tensor,
    target_codes: torch.Tensor,
    query_positions: torch.Tensor,
    metadata: dict[str, str],
    *,
    seed: int,
    cfg_scale: float,
    top_k: int,
    probabilities: tuple[float, ...],
    consecutive_frames: int,
    output_dir: Path,
) -> dict[str, Any]:
    dose_response = {}
    for probability in probabilities:
        trace = rollout_reference_control(
            pipeline,
            adapter,
            reference_codes,
            query_positions,
            prompt=metadata["prompt"],
            lyrics=metadata["lyrics"],
            seed=seed,
            cfg_scale=cfg_scale,
            top_k=top_k,
            use_reference=True,
            training_compatible_warmup=True,
            greedy_rollout=True,
            target_codes=target_codes,
            target_feedback_probability=probability,
        )
        summary = rollout_diagnostics({"rollout": trace}, target_codes)["rollout"]
        feedback = trace.reference_feedback[:-1]
        dose_response[str(probability)] = {
            "target_feedback_fraction": feedback.float().mean().item() if feedback.numel() else 0.0,
            "target_semantic_log_prob_mean": summary["target_semantic_log_prob_mean"],
            "target_semantic_top1_mean": summary["target_semantic_top1_mean"],
            "generated_semantic_agreement_mean": summary["generated_semantic_agreement_mean"],
            "generated_all_book_agreement_mean": summary["generated_all_book_agreement_mean"],
        }

    if query_positions.shape[0] < consecutive_frames + 2:
        raise ValueError("recovery diagnostics require more frames than the sustained re-lock window")
    perturbation_frame = query_positions.shape[0] // 2
    perturbed = rollout_reference_control(
        pipeline,
        adapter,
        reference_codes,
        query_positions,
        prompt=metadata["prompt"],
        lyrics=metadata["lyrics"],
        seed=seed,
        cfg_scale=cfg_scale,
        top_k=top_k,
        use_reference=True,
        training_compatible_warmup=True,
        greedy_rollout=True,
        target_codes=target_codes,
        teacher_force_before_frame=perturbation_frame,
    )
    if perturbed.target_semantic_top1 is None or perturbed.target_semantic_log_probs is None:
        raise ValueError("perturbed rollout did not produce target diagnostics")
    save_file(
        {
            "generated_codes": perturbed.generated_codes,
            "target_feedback": perturbed.reference_feedback,
            "target_semantic_log_probs": perturbed.target_semantic_log_probs,
            "target_semantic_top1": perturbed.target_semantic_top1,
        },
        str(output_dir / "recovery_perturbed.safetensors"),
    )
    perturbed_summary = rollout_diagnostics({"rollout": perturbed}, target_codes)["rollout"]
    perturbed_summary["perturbation_frame"] = perturbation_frame
    perturbed_summary["sustained_relock_frames"] = consecutive_frames
    perturbed_summary["sustained_relock_latency"] = sustained_relock_latency(
        perturbed.target_semantic_top1,
        perturbation_frame,
        consecutive_frames,
    )
    return {"dose_response": dose_response, "induced_perturbation": perturbed_summary}


def plot_recovery_dose_response(diagnostics: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(
        diagnostics["dose_response"].values(),
        key=lambda values: values["target_feedback_fraction"],
    )
    feedback = [values["target_feedback_fraction"] for values in rows]
    top1 = [values["target_semantic_top1_mean"] for values in rows]
    agreement = [values["generated_semantic_agreement_mean"] for values in rows]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(feedback, top1, marker="o", label="target c0 top-1")
    axis.plot(feedback, agreement, marker="o", label="generated c0 agreement")
    axis.set_xlabel("actual target-feedback fraction")
    axis.set_ylabel("fraction")
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_rollout_diagnostics(diagnostics: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for name, values in diagnostics.items():
        agreement = np.asarray(values["per_frame_generated_book_agreement"], dtype=np.float32)
        log_prob = np.asarray(values["per_frame_target_semantic_log_prob"], dtype=np.float32)
        window = min(16, agreement.size)
        kernel = np.ones(window, dtype=np.float32) / window
        axes[0].plot(np.convolve(agreement, kernel, mode="same"), label=name)
        axes[1].plot(log_prob, label=name)
    axes[0].set_ylabel("8-book target agreement\n(16-frame mean)")
    axes[1].set_ylabel("target c0 log probability")
    axes[1].set_xlabel("rollout frame")
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    feedback_probabilities = parse_probabilities(args.feedback_probabilities)
    if args.max_seconds <= 0:
        raise ValueError("max-seconds must be positive")
    if not 1 <= args.top_k <= SEMANTIC_VOCAB_SIZE:
        raise ValueError(f"top-k must be between 1 and {SEMANTIC_VOCAB_SIZE}")
    if args.relock_consecutive_frames < 1:
        raise ValueError("relock-consecutive-frames must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ar_device = torch.device(args.ar_device)
    if args.reference_audio is not None:
        if args.pair_cache is not None or args.clip_id is not None or args.swap_clip_id is not None:
            raise ValueError("reference-audio cannot be combined with pair-cache, clip-id, or swap-clip-id")
        if args.prompt is None or args.lyrics_file is None:
            raise ValueError("reference-audio requires prompt and lyrics-file")
        reference_codes = encode_reference_audio(
            args.reference_audio,
            device=ar_device,
            cache_dir=args.cache_dir,
        )
        target_codes = None
        metadata = {
            "clip_id": args.reference_audio.stem,
            "source_caption": args.source_caption,
            "prompt": args.prompt,
            "lyrics": args.lyrics_file.read_text(encoding="utf-8"),
        }
        shutil.copyfile(args.reference_audio, args.output_dir / f"source{args.reference_audio.suffix.lower()}")
        target_frame_count = reference_codes.shape[0]
    else:
        if args.pair_cache is None:
            raise ValueError("pair-cache is required when reference-audio is not provided")
        pair_path = select_pair_path(args.pair_cache, args.clip_id, args.validation_fraction)
        metadata, reference_codes, target_codes = load_pair(pair_path)
        if args.swap_clip_id is not None:
            swap_path = find_pair_by_clip_id(args.pair_cache, args.swap_clip_id)
            _, swapped_reference_codes, _ = load_pair(swap_path)
        else:
            swap_path = next(
                path for path in sorted(args.pair_cache.glob("shard-*/*.safetensors")) if path.stem != metadata["clip_id"]
            )
            _, swapped_reference_codes, _ = load_pair(swap_path)
        target_frame_count = target_codes.shape[0]
        if not args.skip_pair_audio:
            copy_pair_audio(metadata, args.output_dir, args.cache_dir)
    frame_count = min(target_frame_count, max(1, round(args.max_seconds * FRAME_RATE)))
    query_positions = mapped_reference_positions(frame_count, target_frame_count, reference_codes.shape[0])

    install_diffusers_reference_adapter()
    pipeline = ModularPipeline.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    pipeline.load_components(dtype=torch.bfloat16)
    render_device = torch.device(args.render_device)
    place_components(pipeline, ar_device, render_device)
    adapter, oft_network = load_checkpoint(pipeline, args.checkpoint_dir, ar_device, args.oft_block_size)

    traces = {}
    timings = {}
    for name, enabled in (("reference", True), ("null", False)):
        started = time.perf_counter()
        with torch.autocast(device_type=ar_device.type, dtype=torch.bfloat16):
            traces[name] = rollout_reference_control(
                pipeline,
                adapter,
                reference_codes,
                query_positions,
                prompt=metadata["prompt"],
                lyrics=metadata["lyrics"],
                seed=args.seed,
                cfg_scale=args.cfg_scale,
                top_k=args.top_k,
                use_reference=enabled,
                training_compatible_warmup=args.training_compatible_warmup,
                greedy_rollout=args.greedy_rollout,
                target_codes=target_codes,
            )
        timings[name] = round(time.perf_counter() - started, 3)
        trace_tensors = {
            "generated_codes": traces[name].generated_codes,
            "frame_hiddens": traces[name].frame_hiddens,
            "warmup_feedback_codes": traces[name].warmup_feedback_codes,
        }
        if traces[name].target_semantic_log_probs is not None:
            trace_tensors["target_semantic_log_probs"] = traces[name].target_semantic_log_probs
            trace_tensors["target_semantic_top1"] = traces[name].target_semantic_top1
        save_file(trace_tensors, str(args.output_dir / f"{name}.safetensors"))

    results: dict[str, Any] = {
        "clip_id": metadata["clip_id"],
        "source_caption": metadata["source_caption"],
        "target_prompt": metadata["prompt"],
        "frames": frame_count,
        "seconds": frame_count / FRAME_RATE,
        "seed": args.seed,
        "cfg_scale": args.cfg_scale,
        "top_k": args.top_k,
        "training_compatible_warmup": args.training_compatible_warmup,
        "greedy_rollout": args.greedy_rollout,
        "ar_seconds": timings,
        "codebook_top1": code_metrics(traces, reference_codes, target_codes, query_positions),
        "reference_gates": [round(control.gate.item(), 6) for control in adapter.controls],
    }
    if target_codes is not None:
        results["rollout_diagnostics"] = rollout_diagnostics(traces, target_codes)
        plot_rollout_diagnostics(results["rollout_diagnostics"], args.output_dir / "rollout_diagnostics.png")
        results["teacher_forced_reference_comparison"] = teacher_forced_reference_comparison(
            pipeline,
            adapter,
            metadata,
            target_codes,
            reference_codes,
            swapped_reference_codes,
            frame_count,
            ar_device,
        )
        if args.run_recovery_diagnostics:
            with torch.autocast(device_type=ar_device.type, dtype=torch.bfloat16):
                results["recovery_diagnostics"] = run_recovery_diagnostics(
                    pipeline,
                    adapter,
                    reference_codes,
                    target_codes,
                    query_positions,
                    metadata,
                    seed=args.seed,
                    cfg_scale=args.cfg_scale,
                    top_k=args.top_k,
                    probabilities=feedback_probabilities,
                    consecutive_frames=args.relock_consecutive_frames,
                    output_dir=args.output_dir,
                )
            plot_recovery_dose_response(
                results["recovery_diagnostics"],
                args.output_dir / "recovery_dose_response.png",
            )
    if not args.skip_render:
        results["render"] = render_traces(
            pipeline,
            traces,
            args.output_dir,
            args.num_inference_steps,
            args.seed,
        )
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    del oft_network
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
