#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

if __package__ in (None, ""):
    import _bootstrap  # noqa: F401

from scripts.prompt2effect.base_weights import load_base_weights
from scripts.prompt2effect.lora_utils import save_generated_lora
from scripts.prompt2effect.model import Prompt2EffectConfig, Prompt2EffectHyperNetwork
from scripts.prompt2effect.text_encoder import FrozenTransformersTextEncoder, resolve_device, resolve_dtype


def _module_types_for_layers(schema: dict) -> list[str]:
    return [layer["module_type"] for layer in schema["layers"]]


def _load_checkpoint(path: str | Path, device: torch.device) -> dict:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt2Effect checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or checkpoint.get("format") != "simpletuner-prompt2effect-hypernetwork":
        raise ValueError(f"Not a Prompt2Effect hypernetwork checkpoint: {path}")
    return checkpoint


@torch.no_grad()
def generate_lora(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    base_dtype = resolve_dtype(args.base_weight_dtype)
    output_dtype = resolve_dtype(args.output_dtype)
    checkpoint = _load_checkpoint(args.checkpoint, device)
    schema = checkpoint["schema"]
    config = Prompt2EffectConfig.from_dict(checkpoint["model_config"])
    model = Prompt2EffectHyperNetwork(config).to(device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    text_encoder_model = args.text_encoder_model or checkpoint["text_encoder_model"]
    text_encoder = FrozenTransformersTextEncoder(text_encoder_model, device=device, dtype=dtype)
    encoded = text_encoder.encode([args.prompt], max_length=args.max_prompt_length)

    base_weights = load_base_weights(
        args.base_model or schema["base_model"],
        schema["layers"],
        component_subfolder=args.component_subfolder or schema.get("component_subfolder"),
        revision=args.base_revision or schema.get("base_revision"),
        cache_dir=args.cache_dir,
        dtype=base_dtype,
    )
    if args.base_weights_device == "training":
        base_weights = [weight.to(device=device) for weight in base_weights]

    predictions = model(
        encoded.hidden_states,
        base_weights,
        module_types_for_layer=_module_types_for_layers(schema),
        text_attention_mask=encoded.attention_mask,
    )
    single_predictions = [
        {
            "A": prediction["A"][0],
            "B": prediction["B"][0],
        }
        for prediction in predictions
    ]
    output_path = save_generated_lora(
        args.output,
        single_predictions,
        schema["layers"],
        component_prefix=schema["component_prefix"],
        rank=int(schema["rank"]),
        dtype=output_dtype,
        metadata={
            "format": "simpletuner-prompt2effect-generated-lora",
            "prompt": args.prompt,
            "model_family": schema["model_family"],
            "base_model": args.base_model or schema["base_model"],
            "checkpoint": str(Path(args.checkpoint).expanduser()),
        },
    )
    print(f"Saved generated Prompt2Effect LoRA to {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a PEFT LoRA from a trained Prompt2Effect hypernetwork.")
    parser.add_argument("--checkpoint", required=True, help="Prompt2Effect hypernetwork checkpoint.")
    parser.add_argument("--prompt", required=True, help="Effect prompt to synthesize into LoRA weights.")
    parser.add_argument("--output", required=True, help="Output .safetensors path or directory.")
    parser.add_argument("--text_encoder_model", default=None, help="Override text encoder model from checkpoint.")
    parser.add_argument("--base_model", default=None, help="Override base model repo/path from checkpoint schema.")
    parser.add_argument("--base_revision", default=None, help="Override base model HF revision from checkpoint schema.")
    parser.add_argument("--component_subfolder", default=None, help="Override base model component subfolder from schema.")
    parser.add_argument("--cache_dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--device", default="auto", help="Generation device, or auto.")
    parser.add_argument("--dtype", default="bf16", help="Text encoder dtype: fp32, fp16, or bf16.")
    parser.add_argument("--base_weight_dtype", default="fp32", help="Base weight dtype in memory: fp32, fp16, or bf16.")
    parser.add_argument("--output_dtype", default="fp16", help="Generated LoRA dtype: fp32, fp16, or bf16.")
    parser.add_argument(
        "--base_weights_device",
        choices=("cpu", "training"),
        default="cpu",
        help="Keep base weights on CPU or move them to the generation device.",
    )
    parser.add_argument("--max_prompt_length", type=int, default=128)
    return parser


def main() -> None:
    generate_lora(build_parser().parse_args())


if __name__ == "__main__":
    main()
