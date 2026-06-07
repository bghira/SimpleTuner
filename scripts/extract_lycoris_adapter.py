#!/usr/bin/env python3
"""Extract a LyCORIS/LoCon adapter approximation from two safetensors model components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from extract_adapter_common import (
    compile_optional_regex,
    dtype_from_name,
    normalize_subfolder,
    normalize_target_modules,
    resolve_tensor_source,
    save_safetensors_with_metadata,
    should_extract_key,
    svd_low_rank,
)

SUPPORTED_ALGOS = ("lora", "locon")


def validate_lycoris_state_dict(state_dict: dict[str, torch.Tensor]) -> None:
    try:
        from lycoris.modules import get_module
    except ImportError as exc:
        raise ImportError("The lycoris package is required to validate LyCORIS adapter extraction output.") from exc

    module_names = sorted({key.split(".", 1)[0] for key in state_dict})
    for module_name in module_names:
        lyco_type, _ = get_module(state_dict, module_name)
        if lyco_type is None:
            raise RuntimeError(f"Generated LyCORIS module `{module_name}` is not recognized by the lycoris library.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a LyCORIS-compatible adapter from the weight delta between a base model component and a "
            "target model component. Inputs may be local .safetensors files, local Diffusers folders, or remote "
            "Hugging Face Diffusers repos."
        )
    )
    parser.add_argument("base_model", help="Base model component: .safetensors, Diffusers folder, or HF repo id.")
    parser.add_argument("target_model", help="Target model component: .safetensors, Diffusers folder, or HF repo id.")
    parser.add_argument("output", help="Output .safetensors path or directory.")
    parser.add_argument("--rank", type=int, required=True, help="LyCORIS linear_dim/rank to extract.")
    parser.add_argument("--alpha", type=float, default=None, help="LyCORIS linear_alpha. Defaults to rank.")
    parser.add_argument(
        "--algo",
        default="locon",
        choices=SUPPORTED_ALGOS,
        help="LyCORIS algorithm to encode. Extraction currently supports LoRA/LoCon SVD factors.",
    )
    parser.add_argument("--algorithm", choices=("svd",), default="svd", help="Low-rank extraction algorithm.")
    parser.add_argument(
        "--component-subfolder",
        default="transformer",
        help="Diffusers component subfolder for both models. Use 'none' for direct component folders.",
    )
    parser.add_argument("--base-subfolder", default=None, help="Override component subfolder for the base model.")
    parser.add_argument("--target-subfolder", default=None, help="Override component subfolder for the target model.")
    parser.add_argument("--base-revision", default=None, help="Optional HF revision for the base model.")
    parser.add_argument("--target-revision", default=None, help="Optional HF revision for the target model.")
    parser.add_argument("--cache-dir", default=None, help="Optional Hugging Face cache directory.")
    parser.add_argument("--prefix", default="lycoris", help="LyCORIS state-dict prefix.")
    parser.add_argument(
        "--target-modules",
        default="default",
        help=(
            "Comma-separated module suffixes to extract, 'default' for to_q,to_k,to_v,to_out.0, "
            "or 'all-linear' for every linear weight."
        ),
    )
    parser.add_argument("--include-conv", action="store_true", help="Also extract Conv1d/2d/3d weights.")
    parser.add_argument("--include", default=None, help="Optional regex that tensor keys must match.")
    parser.add_argument("--exclude", default=None, help="Optional regex for tensor keys to skip.")
    parser.add_argument("--device", default="cpu", help="Device used for SVD, e.g. cpu, cuda, mps.")
    parser.add_argument("--dtype", default="float16", help="Output dtype: float32, float16, or bfloat16.")
    parser.add_argument(
        "--min-delta-norm",
        type=float,
        default=0.0,
        help="Skip tensors whose delta L2 norm is less than or equal to this value.",
    )
    parser.add_argument(
        "--skip-mismatched",
        action="store_true",
        help="Skip common tensor keys whose shapes differ instead of raising an error.",
    )
    return parser


def lycoris_module_name(prefix: str, module_name: str) -> str:
    if prefix:
        return f"{prefix}_{module_name}".replace(".", "_")
    return module_name.replace(".", "_")


def extract(args: argparse.Namespace) -> Path:
    alpha = float(args.rank if args.alpha is None else args.alpha)
    dtype = dtype_from_name(args.dtype)
    component_subfolder = normalize_subfolder(args.component_subfolder)
    base_subfolder = normalize_subfolder(args.base_subfolder) or component_subfolder
    target_subfolder = normalize_subfolder(args.target_subfolder) or component_subfolder
    target_modules = normalize_target_modules(args.target_modules)
    include = compile_optional_regex(args.include)
    exclude = compile_optional_regex(args.exclude)

    base = resolve_tensor_source(
        args.base_model,
        label="base",
        subfolder=base_subfolder,
        revision=args.base_revision,
        cache_dir=args.cache_dir,
    )
    target = resolve_tensor_source(
        args.target_model,
        label="target",
        subfolder=target_subfolder,
        revision=args.target_revision,
        cache_dir=args.cache_dir,
    )

    state_dict: dict[str, torch.Tensor] = {}
    skipped_shape = 0
    skipped_filter = 0
    skipped_zero = 0
    with base, target:
        common_keys = sorted(base.keys & target.keys)
        if not common_keys:
            raise ValueError("The base and target sources do not share any tensor keys.")

        for key in common_keys:
            base_tensor = base.get_tensor(key)
            target_tensor = target.get_tensor(key)
            if base_tensor.shape != target_tensor.shape:
                if args.skip_mismatched:
                    skipped_shape += 1
                    continue
                raise ValueError(
                    f"Shape mismatch for `{key}`: base {tuple(base_tensor.shape)} " f"vs target {tuple(target_tensor.shape)}"
                )
            if not should_extract_key(
                key,
                base_tensor,
                target_modules=target_modules,
                include=include,
                exclude=exclude,
                include_conv=args.include_conv,
            ):
                skipped_filter += 1
                continue

            delta = target_tensor.to(dtype=torch.float32) - base_tensor.to(dtype=torch.float32)
            if args.min_delta_norm > 0:
                delta_norm = torch.linalg.vector_norm(delta).item()
                if delta_norm <= args.min_delta_norm:
                    skipped_zero += 1
                    continue

            down, up = svd_low_rank(delta, rank=args.rank, alpha=alpha, device=args.device)
            module_name = lycoris_module_name(args.prefix, key.removesuffix(".weight"))
            state_dict[f"{module_name}.lora_down.weight"] = down.to(dtype=dtype).contiguous()
            state_dict[f"{module_name}.lora_up.weight"] = up.to(dtype=dtype).contiguous()
            state_dict[f"{module_name}.alpha"] = torch.tensor(alpha, dtype=dtype)

    if not state_dict:
        raise ValueError("No tensors were extracted. Check --target-modules, --include/--exclude, and --min-delta-norm.")

    lycoris_config = {
        "algo": args.algo,
        "multiplier": 1.0,
        "linear_dim": args.rank,
        "linear_alpha": alpha,
    }
    metadata = {
        "format": "simpletuner-lycoris-extract",
        "algorithm": args.algorithm,
        "rank": str(args.rank),
        "alpha": str(alpha),
        "base_model": args.base_model,
        "target_model": args.target_model,
        "prefix": args.prefix,
        "target_modules": json.dumps(target_modules or "all-linear"),
        "component_subfolder": component_subfolder or "",
        "lycoris_config": json.dumps(lycoris_config),
    }
    validate_lycoris_state_dict(state_dict)
    output_path = save_safetensors_with_metadata(state_dict, args.output, metadata)
    print(
        f"Saved {len(state_dict) // 3} LyCORIS {args.algo} modules to {output_path} "
        f"(skipped: filtered={skipped_filter}, zero={skipped_zero}, shape={skipped_shape})."
    )
    return output_path


def main() -> None:
    args = build_parser().parse_args()
    extract(args)


if __name__ == "__main__":
    main()
