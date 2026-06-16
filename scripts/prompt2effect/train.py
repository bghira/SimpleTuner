#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in (None, ""):
    import _bootstrap  # noqa: F401

from scripts.prompt2effect.base_weights import load_base_weights
from scripts.prompt2effect.data import Prompt2EffectTargetDataset, collate_prompt2effect_batch
from scripts.prompt2effect.model import Prompt2EffectConfig, Prompt2EffectHyperNetwork, prompt2effect_loss
from scripts.prompt2effect.schema import load_schema
from scripts.prompt2effect.text_encoder import FrozenTransformersTextEncoder, resolve_device, resolve_dtype


def _module_types_for_layers(schema: dict) -> list[str]:
    return [layer["module_type"] for layer in schema["layers"]]


def _build_config(args: argparse.Namespace, schema: dict, text_hidden_dim: int) -> Prompt2EffectConfig:
    layers = schema["layers"]
    module_types = sorted({layer["module_type"] for layer in layers})
    layer_shapes = [(int(layer["out_dim"]), int(layer["in_dim"])) for layer in layers]
    return Prompt2EffectConfig(
        rank=int(schema["rank"]),
        hidden_dim=args.hidden_dim,
        text_hidden_dim=text_hidden_dim,
        compressed_tokens=args.compressed_tokens,
        num_heads=args.num_heads,
        num_layers=args.layers,
        dropout=args.dropout,
        layer_count=len(layers),
        module_types=module_types,
        layer_shapes=layer_shapes,
    )


def _save_checkpoint(
    output_dir: Path,
    *,
    model: Prompt2EffectHyperNetwork,
    optimizer: torch.optim.Optimizer,
    schema: dict,
    text_encoder_model: str,
    step: int,
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint-{step}.pt"
    torch.save(
        {
            "format": "simpletuner-prompt2effect-hypernetwork",
            "step": step,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "schema": schema,
            "text_encoder_model": text_encoder_model,
            "train_args": vars(args),
        },
        checkpoint_path,
    )
    latest_path = output_dir / "prompt2effect_hypernetwork.pt"
    torch.save(
        {
            "format": "simpletuner-prompt2effect-hypernetwork",
            "step": step,
            "model_config": model.config.to_dict(),
            "model_state_dict": model.state_dict(),
            "schema": schema,
            "text_encoder_model": text_encoder_model,
            "train_args": vars(args),
        },
        latest_path,
    )
    return checkpoint_path


def train_prompt2effect(args: argparse.Namespace) -> Path:
    prepared_dir = Path(args.prepared_dir).expanduser()
    schema = load_schema(prepared_dir)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    base_dtype = resolve_dtype(args.base_weight_dtype)
    output_dir = Path(args.output_dir).expanduser()

    dataset = Prompt2EffectTargetDataset(prepared_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_prompt2effect_batch,
        drop_last=False,
    )
    if len(dataloader) == 0:
        raise ValueError("Prompt2Effect dataloader is empty.")

    text_encoder_model = args.text_encoder_model
    text_encoder = FrozenTransformersTextEncoder(text_encoder_model, device=device, dtype=dtype)
    probe = text_encoder.encode([dataset.samples[0]["effect_prompt"]], max_length=args.max_prompt_length)
    config = _build_config(args, schema, probe.hidden_dim)
    model = Prompt2EffectHyperNetwork(config).to(device=device)

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    module_types_for_layer = _module_types_for_layers(schema)

    step = 0
    model.train()
    progress = tqdm(total=args.max_train_steps, desc="Prompt2Effect steps", dynamic_ncols=True)
    while step < args.max_train_steps:
        for batch in dataloader:
            step += 1
            encoded = text_encoder.encode(batch["effect_prompts"], max_length=args.max_prompt_length)
            targets = [
                {key: value.to(device=device) for key, value in layer_target.items()} for layer_target in batch["targets"]
            ]
            predictions = model(
                encoded.hidden_states,
                base_weights,
                module_types_for_layer=module_types_for_layer,
                text_attention_mask=encoded.attention_mask,
            )
            loss = prompt2effect_loss(predictions, targets, eps=args.loss_epsilon)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            progress.update(1)
            progress.set_postfix(loss=f"{loss.item():.5f}")
            if args.checkpointing_steps > 0 and step % args.checkpointing_steps == 0:
                _save_checkpoint(
                    output_dir,
                    model=model,
                    optimizer=optimizer,
                    schema=schema,
                    text_encoder_model=text_encoder_model,
                    step=step,
                    args=args,
                )
            if step >= args.max_train_steps:
                break
    progress.close()
    final_path = _save_checkpoint(
        output_dir,
        model=model,
        optimizer=optimizer,
        schema=schema,
        text_encoder_model=text_encoder_model,
        step=step,
        args=args,
    )
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"final_checkpoint": str(final_path), "steps": step}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Saved Prompt2Effect hypernetwork checkpoint to {final_path}")
    return final_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a Prompt2Effect LoRA-generating hypernetwork.")
    parser.add_argument("--prepared_dir", required=True, help="Directory produced by scripts/prompt2effect/prepare.py.")
    parser.add_argument("--output_dir", required=True, help="Directory for hypernetwork checkpoints.")
    parser.add_argument("--text_encoder_model", required=True, help="Frozen Transformers text encoder for effect prompts.")
    parser.add_argument("--base_model", default=None, help="Override base model repo/path from schema.")
    parser.add_argument("--base_revision", default=None, help="Override base model HF revision from schema.")
    parser.add_argument("--component_subfolder", default=None, help="Override base model component subfolder from schema.")
    parser.add_argument("--cache_dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--device", default="auto", help="Training device, or auto.")
    parser.add_argument("--dtype", default="bf16", help="Text encoder dtype: fp32, fp16, or bf16.")
    parser.add_argument("--base_weight_dtype", default="fp32", help="Base weight dtype in memory: fp32, fp16, or bf16.")
    parser.add_argument(
        "--base_weights_device",
        choices=("cpu", "training"),
        default="cpu",
        help="Keep base weights on CPU or move them to the training device.",
    )
    parser.add_argument("--max_prompt_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--loss_epsilon", type=float, default=1e-8)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--compressed_tokens", type=int, default=16)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--checkpointing_steps", type=int, default=100)
    return parser


def main() -> None:
    train_prompt2effect(build_parser().parse_args())


if __name__ == "__main__":
    main()
