#!/usr/bin/env python
"""Extract Cosmos3 generation weights into a standalone SimpleTuner component."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_cosmos3_reasoner import (
    _cast_state_dict,
    _clear_previous_weights,
    _contains_local_path,
    _parse_dtype,
    _resolve_file,
    _resolve_shard,
    _validate_no_local_paths,
    _write_sharded_safetensors,
)

REASONER_KEY_PATTERNS = (
    re.compile(r"^embed_tokens\."),
    re.compile(r"^lm_head\."),
    re.compile(r"^norm\."),
    re.compile(r"^layers\.\d+\.input_layernorm\."),
    re.compile(r"^layers\.\d+\.post_attention_layernorm\."),
    re.compile(r"^layers\.\d+\.mlp\."),
    re.compile(r"^layers\.\d+\.self_attn\.to_q\."),
    re.compile(r"^layers\.\d+\.self_attn\.to_k\."),
    re.compile(r"^layers\.\d+\.self_attn\.to_v\."),
    re.compile(r"^layers\.\d+\.self_attn\.to_out\."),
    re.compile(r"^layers\.\d+\.self_attn\.norm_q\."),
    re.compile(r"^layers\.\d+\.self_attn\.norm_k\."),
)
GENERATOR_CONFIG_KEYS = (
    "attention_bias",
    "attention_dropout",
    "head_dim",
    "hidden_size",
    "intermediate_size",
    "base_fps",
    "enable_fps_modulation",
    "latent_channel",
    "unified_3d_mrope_reset_spatial_ids",
    "unified_3d_mrope_temporal_modality_margin",
    "latent_patch_size",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "patch_latent_dim",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "action_dim",
    "action_gen",
    "num_embodiment_domains",
    "sound_dim",
    "sound_gen",
    "sound_latent_fps",
    "timestep_scale",
    "vocab_size",
    "hidden_act",
    "qk_norm_for_text",
    "use_und_k_norm_for_gen",
    "rope_axes_dim",
)
REQUIRED_KEY_PATTERNS = (
    re.compile(r"^proj_in\."),
    re.compile(r"^proj_out\."),
    re.compile(r"^time_embedder\."),
    re.compile(r"^layers\.\d+\.self_attn\.add_q_proj\."),
    re.compile(r"^layers\.\d+\.self_attn\.add_k_proj\."),
    re.compile(r"^layers\.\d+\.self_attn\.add_v_proj\."),
    re.compile(r"^layers\.\d+\.self_attn\.to_add_out\."),
    re.compile(r"^layers\.\d+\.mlp_moe_gen\."),
    re.compile(r"^norm_moe_gen\."),
)


def _is_generator_key(key: str) -> bool:
    return not any(pattern.match(key) for pattern in REASONER_KEY_PATTERNS)


def _validate_selected_keys(keys: list[str]) -> None:
    if not keys:
        raise ValueError("No Cosmos3 generator keys were selected.")
    excluded = [key for key in keys if any(pattern.match(key) for pattern in REASONER_KEY_PATTERNS)]
    if excluded:
        raise ValueError(f"Reasoner keys were selected for the generator component: {excluded[:20]}")
    for pattern in REQUIRED_KEY_PATTERNS:
        if not any(pattern.match(key) for key in keys):
            raise ValueError(f"Generator component is missing required key pattern {pattern.pattern!r}.")


def _load_generator_state_dict(
    source_repo: str,
    index_filename: str,
    weights_filename: str,
    revision: str | None,
) -> dict[str, torch.Tensor]:
    local_index = Path(source_repo) / index_filename
    if Path(source_repo).exists() and not local_index.is_file():
        weights_path = _resolve_file(source_repo, weights_filename, revision)
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            keys = [key for key in handle.keys() if _is_generator_key(key)]
            _validate_selected_keys(keys)
            return {key: handle.get_tensor(key) for key in keys}

    try:
        index_path = _resolve_file(source_repo, index_filename, revision)
    except EntryNotFoundError:
        weights_path = _resolve_file(source_repo, weights_filename, revision)
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            keys = [key for key in handle.keys() if _is_generator_key(key)]
            _validate_selected_keys(keys)
            return {key: handle.get_tensor(key) for key in keys}

    with open(index_path, "r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Index file {index_path} does not contain a weight_map object.")

    selected_keys = [key for key in weight_map if _is_generator_key(key)]
    _validate_selected_keys(selected_keys)

    keys_by_shard: dict[str, list[str]] = {}
    for key in selected_keys:
        keys_by_shard.setdefault(weight_map[key], []).append(key)

    state_dict = {}
    for shard_name, shard_keys in keys_by_shard.items():
        shard_path = _resolve_shard(source_repo, index_path, index_filename, shard_name, revision)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            present = set(handle.keys())
            for key in shard_keys:
                if key not in present:
                    raise KeyError(f"Expected key {key} in shard {shard_path}.")
                state_dict[key] = handle.get_tensor(key)
    return state_dict


def _load_generator_config(source_repo: str, config_filename: str, revision: str | None) -> dict:
    config_path = _resolve_file(source_repo, config_filename, revision)
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    return {key: raw_config[key] for key in GENERATOR_CONFIG_KEYS if key in raw_config}


def _write_generator_component(
    *,
    source_repo: str,
    source_model_id: str,
    source_revision: str | None,
    revision: str | None,
    output_dir: Path,
    dtype: torch.dtype,
    index_filename: str,
    weights_filename: str,
    config_filename: str,
    max_shard_size: str,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_weights(output_dir)
    state_dict = _cast_state_dict(
        _load_generator_state_dict(source_repo, index_filename, weights_filename, revision),
        dtype=dtype,
    )
    keys = sorted(state_dict)
    _validate_selected_keys(keys)

    config = _load_generator_config(source_repo, config_filename, revision)
    config.update(
        {
            "component": "cosmos3_generator",
            "source_model_id": source_model_id,
            "source_revision": source_revision or revision,
            "dtype": "bfloat16" if dtype is torch.bfloat16 else str(dtype).replace("torch.", ""),
            "load_reasoning_layers": False,
        }
    )
    _write_sharded_safetensors(
        state_dict,
        output_dir,
        max_shard_size=max_shard_size,
        filename_pattern="diffusion_pytorch_model{suffix}.safetensors",
        index_filename="diffusion_pytorch_model.safetensors.index.json",
        metadata={
            "format": "pt",
            "simpletuner_component": "cosmos3_generator",
            "source_model_id": source_model_id,
            "source_revision": source_revision or revision or "",
        },
    )
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(output_dir / "README.md", "w", encoding="utf-8") as handle:
        handle.write(
            "# Cosmos3 Generator Component\n\n"
            "This repository contains the Cosmos3 generation-path transformer weights used by SimpleTuner. "
            "The in-transformer reasoner weights are omitted; use the matching reasoner component for K/V entries.\n"
        )
    _validate_no_local_paths(output_dir)
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", help="Cosmos3 model repo or local snapshot directory.")
    parser.add_argument("output_dir", help="Directory where the component files will be written.")
    parser.add_argument("--source_model_id", default=None, help="Public source model id to record in component metadata.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision for source_repo.")
    parser.add_argument("--source_revision", default=None, help="Revision value to record in component config.")
    parser.add_argument("--dtype", default="bf16", help="Output floating-point dtype: bf16, fp16, or fp32.")
    parser.add_argument(
        "--index_filename",
        default="transformer/diffusion_pytorch_model.safetensors.index.json",
        help="Transformer safetensors index path inside source_repo.",
    )
    parser.add_argument(
        "--weights_filename",
        default="transformer/diffusion_pytorch_model.safetensors",
        help="Single transformer safetensors path inside source_repo.",
    )
    parser.add_argument(
        "--config_filename",
        default="transformer/config.json",
        help="Transformer config path inside source_repo.",
    )
    parser.add_argument("--max_shard_size", default="2GB", help="Maximum output shard size.")
    parser.add_argument("--push_to_hub", default=None, help="Optional target Hub repo id.")
    args = parser.parse_args()

    source_model_id = args.source_model_id
    if source_model_id in (None, "", "None"):
        if Path(args.source_repo).exists():
            raise ValueError("--source_model_id is required when source_repo is a local path.")
        source_model_id = args.source_repo
    if _contains_local_path(source_model_id):
        raise ValueError("--source_model_id must not be a local filesystem path.")
    source_revision = args.source_revision
    if source_revision in (None, "", "None") and args.revision not in (None, "", "None"):
        source_revision = args.revision
    if source_revision in (None, "", "None") and not Path(args.source_repo).exists():
        source_revision = HfApi().model_info(args.source_repo, revision=args.revision).sha

    output_dir = Path(args.output_dir)
    keys = _write_generator_component(
        source_repo=args.source_repo,
        source_model_id=source_model_id,
        source_revision=source_revision,
        revision=args.revision,
        output_dir=output_dir,
        dtype=_parse_dtype(args.dtype),
        index_filename=args.index_filename,
        weights_filename=args.weights_filename,
        config_filename=args.config_filename,
        max_shard_size=args.max_shard_size,
    )
    print(f"Wrote {len(keys)} tensors to {output_dir}.")

    if args.push_to_hub:
        api = HfApi()
        api.create_repo(args.push_to_hub, repo_type="model", exist_ok=True)
        commit = api.upload_folder(
            repo_id=args.push_to_hub,
            repo_type="model",
            folder_path=str(output_dir),
            commit_message="Add Cosmos3 generator component",
            delete_patterns=["model*.safetensors", "model.safetensors.index.json"],
        )
        print(commit)


if __name__ == "__main__":
    main()
