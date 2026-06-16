#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

if __package__ in (None, ""):
    import _bootstrap  # noqa: F401

from scripts.prompt2effect.base_weights import infer_base_layers
from scripts.prompt2effect.lora_utils import (
    canonicalize_delta,
    discover_lora_modules,
    load_safetensors,
    lora_delta,
    metadata_json,
)
from scripts.prompt2effect.registry import (
    module_matches_target,
    module_type_for_name,
    normalize_target_modules,
    resolve_family_spec,
    resolve_model_repo,
)
from scripts.prompt2effect.schema import TARGETS_FILENAME, save_schema


@dataclass(frozen=True)
class ManifestEntry:
    id: str
    effect_prompt: str
    lora_path: str
    metadata: dict[str, Any]


def load_manifest(path: str | Path) -> list[ManifestEntry]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt2Effect manifest not found: {path}")
    entries: list[ManifestEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in manifest line {line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Manifest line {line_number} must be a JSON object.")
            entry_id = payload.get("id")
            prompt = payload.get("effect_prompt", payload.get("prompt"))
            lora_path = payload.get("lora_path", payload.get("adapter_path"))
            if not isinstance(entry_id, str) or not entry_id.strip():
                raise ValueError(f"Manifest line {line_number} is missing a non-empty `id`.")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f"Manifest line {line_number} is missing a non-empty `effect_prompt`.")
            if not isinstance(lora_path, str) or not lora_path.strip():
                raise ValueError(f"Manifest line {line_number} is missing a non-empty `lora_path`.")
            metadata = {
                k: v for k, v in payload.items() if k not in {"id", "effect_prompt", "prompt", "lora_path", "adapter_path"}
            }
            entries.append(
                ManifestEntry(
                    id=entry_id.strip(),
                    effect_prompt=prompt.strip(),
                    lora_path=str(Path(lora_path).expanduser()),
                    metadata=metadata,
                )
            )
    if not entries:
        raise ValueError(f"Prompt2Effect manifest is empty: {path}")
    return entries


def _filtered_modules(
    state_dict: dict[str, torch.Tensor],
    *,
    component_prefix: str,
    target_modules: list[str],
) -> dict[str, Any]:
    modules = discover_lora_modules(state_dict, component_prefix=component_prefix)
    filtered = {
        module_name: module for module_name, module in modules.items() if module_matches_target(module_name, target_modules)
    }
    if not filtered:
        targets = "all linear modules" if not target_modules else ", ".join(target_modules)
        raise ValueError(f"No LoRA modules matched Prompt2Effect target modules: {targets}.")
    return filtered


def _validate_sample_modules(
    entry: ManifestEntry,
    modules: dict[str, Any],
    expected_modules: dict[str, Any],
) -> None:
    module_names = set(modules)
    expected_names = set(expected_modules)
    if module_names != expected_names:
        missing = sorted(expected_names - module_names)
        extra = sorted(module_names - expected_names)
        raise ValueError(
            f"LoRA `{entry.lora_path}` does not match the Prompt2Effect schema. "
            f"Missing modules: {missing[:8]}; extra modules: {extra[:8]}."
        )
    for module_name, expected in expected_modules.items():
        actual = modules[module_name]
        expected_shape = (expected.in_dim, expected.out_dim)
        actual_shape = (actual.in_dim, actual.out_dim)
        if expected_shape != actual_shape:
            raise ValueError(
                f"LoRA `{entry.lora_path}` module `{module_name}` has in/out {actual_shape}, " f"expected {expected_shape}."
            )


def prepare_prompt2effect_targets(args: argparse.Namespace) -> Path:
    spec = resolve_family_spec(args.model_family)
    target_modules = normalize_target_modules(args.target_modules, spec)
    base_model = resolve_model_repo(spec, args.base_model, args.model_flavour)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_manifest(args.manifest)
    first_state, first_metadata = load_safetensors(entries[0].lora_path)
    first_modules = _filtered_modules(
        first_state,
        component_prefix=spec.component_prefix,
        target_modules=target_modules,
    )

    rank = int(args.rank or next(iter(first_modules.values())).rank)
    if rank <= 0:
        raise ValueError("Prompt2Effect rank must be greater than zero.")

    module_names = sorted(first_modules)
    base_layers = infer_base_layers(
        base_model,
        module_names,
        component_prefix=spec.component_prefix,
        component_subfolder=args.component_subfolder or spec.component_subfolder,
        revision=args.base_revision,
        cache_dir=args.cache_dir,
    )

    layers: list[dict[str, Any]] = []
    for layer_idx, module_name in enumerate(module_names):
        module = first_modules[module_name]
        base_key, (out_dim, in_dim) = base_layers[module_name]
        if (module.out_dim, module.in_dim) != (out_dim, in_dim):
            raise ValueError(
                f"LoRA module `{module_name}` has update shape {(module.out_dim, module.in_dim)}, "
                f"but base tensor `{base_key}` has shape {(out_dim, in_dim)}."
            )
        layers.append(
            {
                "index": layer_idx,
                "module_name": module_name,
                "module_type": module_type_for_name(module_name, target_modules),
                "base_key": base_key,
                "out_dim": out_dim,
                "in_dim": in_dim,
            }
        )

    target_state: dict[str, torch.Tensor] = {}
    samples = []
    for sample_idx, entry in enumerate(entries):
        state_dict = first_state if sample_idx == 0 else load_safetensors(entry.lora_path)[0]
        modules = (
            first_modules
            if sample_idx == 0
            else _filtered_modules(
                state_dict,
                component_prefix=spec.component_prefix,
                target_modules=target_modules,
            )
        )
        _validate_sample_modules(entry, modules, first_modules)
        samples.append(asdict(entry))
        for layer in layers:
            module = modules[layer["module_name"]]
            delta = lora_delta(state_dict, module)
            a_star, b_star = canonicalize_delta(delta, rank=rank)
            layer_idx = int(layer["index"])
            target_state[f"samples.{sample_idx}.layers.{layer_idx}.A"] = a_star.to(dtype=torch.float32).contiguous()
            target_state[f"samples.{sample_idx}.layers.{layer_idx}.B"] = b_star.to(dtype=torch.float32).contiguous()

    targets_path = output_dir / TARGETS_FILENAME
    save_file(
        target_state,
        str(targets_path),
        metadata={
            "format": "simpletuner-prompt2effect-targets",
            "model_family": spec.family,
            "rank": str(rank),
            "target_modules": metadata_json(target_modules or "all-linear"),
            "base_model": base_model,
        },
    )

    schema = {
        "format": "simpletuner-prompt2effect-schema",
        "version": 1,
        "model_family": spec.family,
        "model_label": spec.label,
        "model_flavour": args.model_flavour or spec.default_model_flavour,
        "base_model": base_model,
        "base_revision": args.base_revision,
        "component_prefix": spec.component_prefix,
        "component_subfolder": args.component_subfolder or spec.component_subfolder,
        "rank": rank,
        "target_modules": target_modules,
        "layers": layers,
        "samples": samples,
        "source_lora_metadata": first_metadata,
    }
    save_schema(output_dir, schema)
    print(f"Prepared {len(samples)} Prompt2Effect samples with {len(layers)} layers at {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare Prompt2Effect SVD-canonical LoRA targets.")
    parser.add_argument("--manifest", required=True, help="JSONL file with id, effect_prompt, and lora_path fields.")
    parser.add_argument("--output_dir", required=True, help="Directory for schema.json and targets.safetensors.")
    parser.add_argument("--model_family", required=True, choices=("ltxvideo2", "wan", "hunyuanvideo"))
    parser.add_argument("--base_model", default=None, help="Base model repo/path. Defaults to the selected family flavour.")
    parser.add_argument("--model_flavour", default=None, help="Known model flavour for the selected family.")
    parser.add_argument("--base_revision", default=None, help="Optional base model HF revision.")
    parser.add_argument("--cache_dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--component_subfolder", default=None, help="Base model component subfolder. Defaults per family.")
    parser.add_argument(
        "--target_modules", default="default", help="Comma-separated PEFT target modules, default, or all-linear."
    )
    parser.add_argument("--rank", type=int, default=None, help="Prompt2Effect target rank. Defaults to the first LoRA rank.")
    return parser


def main() -> None:
    prepare_prompt2effect_targets(build_parser().parse_args())


if __name__ == "__main__":
    main()
