#!/usr/bin/env python3
# Copyright 2026 SimpleTuner contributors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from diffusers import ModularPipeline
from safetensors.torch import load_file, save_file

REPO_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_DIR = REPO_ROOT / "model_cards" / "collection"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COLLECTION_DIR))

from benchmark_reference_feedback import place_components, render_traces  # noqa: E402
from minimax_music3_reference_adapter import install_diffusers_reference_adapter  # noqa: E402

from scripts.minimax_music3.eval_control_lora import (  # noqa: E402
    AUDIO_CFG_TOKEN_ID,
    checkpoint_args,
    classifier_free_guidance_logits,
    generate_cfg_frame,
)
from scripts.minimax_music3.eval_prefix_distillation import generate_frame  # noqa: E402
from scripts.minimax_music3.train_control_lora import (  # noqa: E402
    CHECKPOINT_FORMAT,
    aligned_reference_hint,
    load_checkpoint,
    reference_delta_control_hidden_states,
    reference_warmup_codes,
)
from scripts.minimax_music3.train_reference_control import (  # noqa: E402
    AUDIO_CODE_OFFSET,
    AUDIO_VOCAB_SIZE,
    CachedStylePairDataset,
    tokenize_prompt,
)
from simpletuner.helpers.models.minimaxmusic.reference_control import (  # noqa: E402
    ControlLoRAConfig,
    MiniMaxMusic3ControlLoRAAdapter,
    create_qwen_lokr_adapter,
    embed_rvq_frames,
)

FRAME_RATE = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate audio with a MiniMax Music 3 ControlLoRA checkpoint")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="MiniMaxAI/MiniMax-Music3")
    parser.add_argument("--cache-dir")
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--control-strength", type=float, default=0.5)
    parser.add_argument("--cfg-scale", type=float)
    parser.add_argument("--control-unconditional-branch", action="store_true")
    parser.add_argument("--generation-mode", choices=("reference", "null", "both"), default="reference")
    parser.add_argument("--ar-device", default="cuda:0")
    parser.add_argument("--render-device", default="cuda:1")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--verify-prefix-codes", type=Path)
    return parser.parse_args()


def load_pipeline_and_checkpoint(args: argparse.Namespace, ar_device: torch.device, render_device: torch.device):
    config_path = args.checkpoint_dir / "control_lora.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported ControlLoRA checkpoint format: {config.get('format')}")

    install_diffusers_reference_adapter()
    pipeline = ModularPipeline.from_pretrained(args.model_id, cache_dir=args.cache_dir)
    pipeline.load_components(dtype=torch.bfloat16)
    place_components(pipeline, ar_device, render_device)
    runtime_args = checkpoint_args(config)
    lokr_network = create_qwen_lokr_adapter(
        pipeline.language_model,
        rank=runtime_args.lokr_rank,
        alpha=runtime_args.lokr_alpha,
    ).to(ar_device)
    adapter = MiniMaxMusic3ControlLoRAAdapter(ControlLoRAConfig.from_dict(config["adapter"])).to(
        device=ar_device,
        dtype=torch.bfloat16,
    )
    adapter.install(pipeline.language_model)
    load_checkpoint(args.checkpoint_dir, adapter, lokr_network, runtime_args)
    adapter.set_multiplier(args.control_strength)
    pipeline.language_model.eval()
    pipeline.rvq_depth_decoder.eval()
    adapter.eval()
    return SimpleNamespace(
        pipeline=pipeline,
        adapter=adapter,
        lokr_network=lokr_network,
        runtime_args=runtime_args,
        config=config,
    )


def controlled_forward(
    language_model,
    lokr_network,
    main_inputs: torch.Tensor,
    control_inputs: torch.Tensor,
    *,
    main_past=None,
    control_past=None,
    control_query_start: int,
):
    lokr_network.set_multiplier(1.0)
    control_output = language_model.model(
        inputs_embeds=control_inputs,
        past_key_values=control_past,
        use_cache=True,
        output_hidden_states=True,
    )
    control_hidden_states = control_output.hidden_states[1:]
    if len(control_hidden_states) != len(language_model.model.layers):
        raise RuntimeError("control pass did not return one hidden state per Qwen block")
    lokr_network.set_multiplier(0.0)
    try:
        main_output = language_model.model(
            inputs_embeds=main_inputs,
            past_key_values=main_past,
            use_cache=True,
            control_hidden_states=control_hidden_states,
            control_query_start=control_query_start,
        )
    finally:
        lokr_network.set_multiplier(1.0)
    return main_output, control_output


def main_forward(
    language_model,
    lokr_network,
    main_inputs: torch.Tensor,
    control_hidden_states: tuple[torch.Tensor, ...],
    *,
    main_past=None,
    control_query_start: int,
    control_scale: float,
):
    lokr_network.set_multiplier(0.0)
    try:
        return language_model.model(
            inputs_embeds=main_inputs,
            past_key_values=main_past,
            use_cache=True,
            control_hidden_states=control_hidden_states,
            control_query_start=control_query_start,
            control_scale=control_scale,
        )
    finally:
        lokr_network.set_multiplier(1.0)


def uncontrolled_forward(language_model, lokr_network, inputs: torch.Tensor, *, past=None):
    lokr_network.set_multiplier(0.0)
    try:
        return language_model.model(inputs_embeds=inputs, past_key_values=past, use_cache=True)
    finally:
        lokr_network.set_multiplier(1.0)


@torch.inference_mode()
def cached_rollout(
    components,
    sample: dict,
    frame_count: int,
    *,
    null_reference: bool,
    sample_codes: bool,
    top_k: int,
    seed: int,
    cfg_scale: float | None,
    control_unconditional_branch: bool,
):
    pipeline = components.pipeline
    language_model = pipeline.language_model
    depth_decoder = pipeline.rvq_depth_decoder
    device = next(language_model.parameters()).device
    reference_codes = sample["reference_codes"].to(device).unsqueeze(0)
    query_positions = sample["query_positions"][:frame_count].to(device).unsqueeze(0)
    key_positions = sample["key_positions"].to(device).unsqueeze(0)
    text_ids = tokenize_prompt(pipeline.tokenizer, sample["prompt"], sample["lyrics"], device)
    text_embeddings = language_model.model.embed_tokens(text_ids)
    unconditional_text_embeddings = None
    if cfg_scale is not None:
        unconditional_ids = text_ids.clone()
        unconditional_ids[:, 1:-2] = AUDIO_CFG_TOKEN_ID
        unconditional_text_embeddings = language_model.model.embed_tokens(unconditional_ids)
    reference_embeddings = embed_rvq_frames(language_model, depth_decoder, reference_codes)
    warmup_codes = reference_warmup_codes(reference_codes, query_positions, key_positions)
    warmup_embeddings = embed_rvq_frames(language_model, depth_decoder, warmup_codes)
    main_inputs = torch.cat((text_embeddings, warmup_embeddings), dim=1)
    control_past = None
    unconditional_control_states = None
    if components.runtime_args.control_input_mode == "additive-hint":
        hints = aligned_reference_hint(reference_embeddings, query_positions, key_positions)
        if null_reference:
            hints = hints * 0.0
        control_inputs = torch.cat(
            (text_embeddings, warmup_embeddings + components.runtime_args.hint_scale * hints[:, :1]),
            dim=1,
        )
        main_output, control_output = controlled_forward(
            language_model,
            components.lokr_network,
            main_inputs,
            control_inputs,
            control_query_start=text_embeddings.shape[1],
        )
        control_past = control_output.past_key_values
        control_states = None
    elif components.runtime_args.control_input_mode == "reference-delta":
        components.lokr_network.set_multiplier(1.0)
        control_states = reference_delta_control_hidden_states(
            language_model,
            text_embeddings,
            reference_embeddings,
            query_positions,
            key_positions,
            null_reference=null_reference,
            text_attention_mask=None,
            reference_attention_mask=None,
        )
        if control_unconditional_branch:
            unconditional_control_states = reference_delta_control_hidden_states(
                language_model,
                unconditional_text_embeddings,
                reference_embeddings,
                query_positions,
                key_positions,
                null_reference=null_reference,
                text_attention_mask=None,
                reference_attention_mask=None,
            )
        main_output = main_forward(
            language_model,
            components.lokr_network,
            main_inputs,
            tuple(states[:, :1] for states in control_states),
            control_query_start=text_embeddings.shape[1],
            control_scale=components.runtime_args.hint_scale,
        )
        hints = None
    else:
        raise ValueError(f"Unsupported ControlLoRA input mode: {components.runtime_args.control_input_mode}")
    main_past = main_output.past_key_values
    hidden = main_output.last_hidden_state[:, -1]
    unconditional_past = None
    unconditional_hidden = None
    if unconditional_text_embeddings is not None:
        unconditional_inputs = torch.cat((unconditional_text_embeddings, warmup_embeddings), dim=1)
        if unconditional_control_states is None:
            unconditional_output = uncontrolled_forward(
                language_model,
                components.lokr_network,
                unconditional_inputs,
            )
        else:
            unconditional_output = main_forward(
                language_model,
                components.lokr_network,
                unconditional_inputs,
                tuple(states[:, :1] for states in unconditional_control_states),
                control_query_start=unconditional_text_embeddings.shape[1],
                control_scale=components.runtime_args.hint_scale,
            )
        unconditional_past = unconditional_output.past_key_values
        unconditional_hidden = unconditional_output.last_hidden_state[:, -1]
    generated_codes = []
    frame_hiddens = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    started = time.perf_counter()
    for frame_index in range(frame_count):
        conditioned_logits = F.linear(
            hidden,
            language_model.lm_head.weight[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + AUDIO_VOCAB_SIZE],
        ).float()
        if unconditional_hidden is None:
            codes, frame_hidden = generate_frame(
                language_model,
                depth_decoder,
                hidden,
                sample=sample_codes,
                top_k=top_k,
                generator=generator,
                semantic_override_logits=conditioned_logits,
            )
        else:
            unconditioned_logits = F.linear(
                unconditional_hidden,
                language_model.lm_head.weight[AUDIO_CODE_OFFSET : AUDIO_CODE_OFFSET + AUDIO_VOCAB_SIZE],
            ).float()
            logits = classifier_free_guidance_logits(conditioned_logits, unconditioned_logits, cfg_scale)
            codes, frame_hidden = generate_cfg_frame(
                language_model,
                depth_decoder,
                hidden,
                unconditional_hidden,
                logits,
                cfg_scale=cfg_scale,
                sample=sample_codes,
                top_k=top_k,
                generator=generator,
            )
        generated_codes.append(codes.cpu())
        frame_hiddens.append(frame_hidden.cpu())
        if (frame_index + 1) % 25 == 0 or frame_index + 1 == frame_count:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "generated_frames": frame_index + 1,
                        "total_frames": frame_count,
                        "frames_per_second": round((frame_index + 1) / elapsed, 3),
                    }
                ),
                flush=True,
            )
        if frame_index + 1 == frame_count:
            break
        feedback = embed_rvq_frames(language_model, depth_decoder, codes.unsqueeze(1))
        if control_states is None:
            main_output, control_output = controlled_forward(
                language_model,
                components.lokr_network,
                feedback,
                feedback + components.runtime_args.hint_scale * hints[:, frame_index + 1 : frame_index + 2],
                main_past=main_past,
                control_past=control_past,
                control_query_start=0,
            )
            control_past = control_output.past_key_values
        else:
            main_output = main_forward(
                language_model,
                components.lokr_network,
                feedback,
                tuple(states[:, frame_index + 1 : frame_index + 2] for states in control_states),
                main_past=main_past,
                control_query_start=0,
                control_scale=components.runtime_args.hint_scale,
            )
        main_past = main_output.past_key_values
        hidden = main_output.last_hidden_state[:, -1]
        if unconditional_past is not None:
            if unconditional_control_states is None:
                unconditional_output = uncontrolled_forward(
                    language_model,
                    components.lokr_network,
                    feedback,
                    past=unconditional_past,
                )
            else:
                unconditional_output = main_forward(
                    language_model,
                    components.lokr_network,
                    feedback,
                    tuple(states[:, frame_index + 1 : frame_index + 2] for states in unconditional_control_states),
                    main_past=unconditional_past,
                    control_query_start=0,
                    control_scale=components.runtime_args.hint_scale,
                )
            unconditional_past = unconditional_output.past_key_values
            unconditional_hidden = unconditional_output.last_hidden_state[:, -1]
    return SimpleNamespace(
        generated_codes=torch.cat(generated_codes, dim=0),
        frame_hiddens=torch.cat(frame_hiddens, dim=0).unsqueeze(0).to(torch.bfloat16),
        seconds=time.perf_counter() - started,
    )


def verify_prefix(trace, expected_path: Path) -> dict:
    expected = load_file(expected_path)["generated_codes"].long()
    compared_frames = min(expected.shape[0], trace.generated_codes.shape[0])
    agreement = (trace.generated_codes[:compared_frames] == expected[:compared_frames]).float().mean(dim=0)
    result = {
        "frames": compared_frames,
        "semantic_agreement": agreement[0].item(),
        "acoustic_agreement": agreement[1:].mean().item(),
        "per_codebook_agreement": agreement.tolist(),
    }
    if not bool((trace.generated_codes[:compared_frames] == expected[:compared_frames]).all()):
        raise RuntimeError(f"cached rollout does not match the expected prefix: {result}")
    return result


def main() -> None:
    args = parse_args()
    if args.max_seconds <= 0.0:
        raise ValueError("max-seconds must be positive")
    if not 1 <= args.top_k <= AUDIO_VOCAB_SIZE:
        raise ValueError(f"top-k must be between 1 and {AUDIO_VOCAB_SIZE}")
    if args.control_strength < 0.0:
        raise ValueError("control-strength must be non-negative")
    if args.cfg_scale is not None and (not math.isfinite(args.cfg_scale) or args.cfg_scale < 0.0):
        raise ValueError("cfg-scale must be finite and non-negative")
    if args.control_unconditional_branch and args.cfg_scale is None:
        raise ValueError("--control-unconditional-branch requires --cfg-scale")
    if args.control_unconditional_branch:
        config = json.loads((args.checkpoint_dir / "control_lora.json").read_text(encoding="utf-8"))
        if config.get("control_input_mode", "additive-hint") != "reference-delta":
            raise ValueError("controlled CFG currently requires a reference-delta checkpoint")
    frame_count = round(args.max_seconds * FRAME_RATE)
    dataset = CachedStylePairDataset(
        args.pair_cache,
        crop_frames=frame_count,
        reference_context_frames=1,
        clip_id=args.clip_id,
        fixed_crop_start=0,
    )
    sample = dataset[0]
    frame_count = min(frame_count, sample["target_codes"].shape[0])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    components = load_pipeline_and_checkpoint(args, torch.device(args.ar_device), torch.device(args.render_device))
    modes = {
        "reference": (("reference", False),),
        "null": (("null", True),),
        "both": (("reference", False), ("null", True)),
    }[args.generation_mode]
    traces = {}
    mode_results = {}
    for name, null_reference in modes:
        trace = cached_rollout(
            components,
            sample,
            frame_count,
            null_reference=null_reference,
            sample_codes=args.sample,
            top_k=args.top_k,
            seed=args.seed,
            cfg_scale=args.cfg_scale,
            control_unconditional_branch=args.control_unconditional_branch,
        )
        traces[name] = trace
        save_file(
            {
                "generated_codes": trace.generated_codes.to(torch.int16),
                "frame_hiddens": trace.frame_hiddens,
            },
            args.output_dir / f"{name}.safetensors",
        )
        mode_results[name] = {"ar_seconds": round(trace.seconds, 3)}
    if args.verify_prefix_codes and "reference" not in traces:
        raise ValueError("--verify-prefix-codes requires reference generation")
    verification = verify_prefix(traces["reference"], args.verify_prefix_codes) if args.verify_prefix_codes else None
    results = {
        "clip_id": args.clip_id,
        "frames": frame_count,
        "seconds": frame_count / FRAME_RATE,
        "sample": args.sample,
        "top_k": args.top_k,
        "control_strength": args.control_strength,
        "cfg_scale": args.cfg_scale,
        "control_unconditional_branch": args.control_unconditional_branch,
        "modes": mode_results,
        "verification": verification,
        "checkpoint_config": components.config,
    }
    if not args.skip_render:
        render_results = render_traces(
            components.pipeline,
            traces,
            args.output_dir,
            args.num_inference_steps,
            args.seed,
        )
        for name, metrics in render_results.items():
            results["modes"][name]["render"] = metrics
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
