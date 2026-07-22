#!/usr/bin/env python
"""Extract Cosmos3 reasoning weights into a standalone SimpleTuner component."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from posixpath import dirname as posix_dirname
from posixpath import join as posix_join

import torch
from huggingface_hub import HfApi, hf_hub_download, split_torch_state_dict_into_shards
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open
from safetensors.torch import save_file

REASONER_KEY_PATTERNS = (
    re.compile(r"^embed_tokens\."),
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
    re.compile(r"^layers\.\d+\.self_attn\.k_norm_und_for_gen\."),
)
EXCLUDED_KEY_FRAGMENTS = (
    "moe_gen",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "audio_",
    "action_",
    "proj_in",
    "proj_out",
    "time_",
    "lm_head",
)
CONFIG_KEYS = (
    "attention_bias",
    "head_dim",
    "hidden_size",
    "intermediate_size",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "rms_norm_eps",
    "rope_scaling",
    "rope_theta",
    "vocab_size",
    "hidden_act",
    "qk_norm_for_text",
    "use_und_k_norm_for_gen",
    "rope_axes_dim",
)
LOCAL_PATH_PATTERNS = (
    re.compile(r"/Users/"),
    re.compile(r"/home/"),
    re.compile(r"/private/"),
    re.compile(r"^[A-Za-z]:\\"),
)


def _is_reasoner_key(key: str) -> bool:
    return any(pattern.match(key) for pattern in REASONER_KEY_PATTERNS)


def _validate_selected_keys(keys: list[str]) -> None:
    if not keys:
        raise ValueError("No Cosmos3 reasoner keys were selected.")
    excluded = [key for key in keys if any(fragment in key for fragment in EXCLUDED_KEY_FRAGMENTS)]
    if excluded:
        raise ValueError(f"Excluded Cosmos3 keys were selected: {excluded[:20]}")


def _resolve_file(source_repo: str, filename: str, revision: str | None) -> Path:
    local_file = Path(source_repo) / filename
    if local_file.is_file():
        return local_file
    return Path(hf_hub_download(repo_id=source_repo, filename=filename, revision=revision))


def _resolve_shard(source_repo: str, index_path: Path, index_filename: str, shard_name: str, revision: str | None) -> Path:
    if Path(source_repo).exists():
        local_shard = index_path.parent / shard_name
        if local_shard.is_file():
            return local_shard
    shard_dir = posix_dirname(index_filename)
    shard_repo_path = posix_join(shard_dir, shard_name) if shard_dir else shard_name
    return Path(hf_hub_download(repo_id=source_repo, filename=shard_repo_path, revision=revision))


def _load_transformer_config(source_repo: str, config_filename: str, revision: str | None) -> dict:
    config_path = _resolve_file(source_repo, config_filename, revision)
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    return {key: raw_config[key] for key in CONFIG_KEYS if key in raw_config}


def _load_reasoner_state_dict(source_repo: str, index_filename: str, weights_filename: str, revision: str | None) -> dict:
    local_index = Path(source_repo) / index_filename
    if Path(source_repo).exists() and not local_index.is_file():
        weights_path = _resolve_file(source_repo, weights_filename, revision)
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            keys = [key for key in handle.keys() if _is_reasoner_key(key)]
            _validate_selected_keys(keys)
            return {key: handle.get_tensor(key) for key in keys}

    try:
        index_path = _resolve_file(source_repo, index_filename, revision)
    except EntryNotFoundError:
        weights_path = _resolve_file(source_repo, weights_filename, revision)
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            keys = [key for key in handle.keys() if _is_reasoner_key(key)]
            _validate_selected_keys(keys)
            return {key: handle.get_tensor(key) for key in keys}

    with open(index_path, "r", encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Index file {index_path} does not contain a weight_map object.")

    selected_keys = [key for key in weight_map if _is_reasoner_key(key)]
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


def _cast_state_dict(state_dict: dict[str, torch.Tensor], dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {key: value.to(dtype=dtype) if torch.is_floating_point(value) else value for key, value in state_dict.items()}


def _contains_local_path(value: object) -> bool:
    text = str(value)
    return any(pattern.search(text) for pattern in LOCAL_PATH_PATTERNS)


def _validate_no_local_paths(output_dir: Path) -> None:
    for filename in (
        "config.json",
        "README.md",
        "model.safetensors.index.json",
        "diffusion_pytorch_model.safetensors.index.json",
    ):
        path = output_dir / filename
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            if _contains_local_path(text):
                raise ValueError(f"{filename} contains a local filesystem path.")
    safetensor_paths = sorted(output_dir.glob("*.safetensors"))
    if not safetensor_paths:
        raise ValueError("No safetensors files were written.")
    for path in safetensor_paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        if any(_contains_local_path(value) for value in metadata.values()):
            raise ValueError(f"{path.name} metadata contains a local filesystem path.")


def _clear_previous_weights(output_dir: Path) -> None:
    for pattern in ("model*.safetensors", "diffusion_pytorch_model*.safetensors"):
        for path in output_dir.glob(pattern):
            path.unlink()
    for filename in ("model.safetensors.index.json", "diffusion_pytorch_model.safetensors.index.json"):
        index_path = output_dir / filename
        if index_path.exists():
            index_path.unlink()


def _write_sharded_safetensors(
    state_dict: dict[str, torch.Tensor],
    output_dir: Path,
    *,
    max_shard_size: str,
    metadata: dict[str, str],
    filename_pattern: str = "model{suffix}.safetensors",
    index_filename: str = "model.safetensors.index.json",
) -> list[str]:
    split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern=filename_pattern,
        max_shard_size=max_shard_size,
    )
    for filename, tensor_names in split.filename_to_tensors.items():
        save_file(
            {tensor_name: state_dict[tensor_name] for tensor_name in tensor_names},
            output_dir / filename,
            metadata=metadata,
        )
    if split.is_sharded:
        index = {
            "metadata": split.metadata,
            "weight_map": split.tensor_to_filename,
        }
        with open(output_dir / index_filename, "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2, sort_keys=True)
            handle.write("\n")
    return sorted(split.filename_to_tensors)


def _write_component(
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
        _load_reasoner_state_dict(source_repo, index_filename, weights_filename, revision),
        dtype=dtype,
    )
    keys = sorted(state_dict)
    _validate_selected_keys(keys)

    config = _load_transformer_config(source_repo, config_filename, revision)
    config.update(
        {
            "component": "cosmos3_reasoner",
            "source_model_id": source_model_id,
            "source_revision": source_revision or revision,
            "dtype": "bfloat16" if dtype is torch.bfloat16 else str(dtype).replace("torch.", ""),
        }
    )
    _write_sharded_safetensors(
        state_dict,
        output_dir,
        max_shard_size=max_shard_size,
        metadata={
            "format": "pt",
            "simpletuner_component": "cosmos3_reasoner",
            "source_model_id": source_model_id,
            "source_revision": source_revision or revision or "",
        },
    )
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(output_dir / "README.md", "w", encoding="utf-8") as handle:
        handle.write(
            "# Cosmos3 Reasoner Component\n\n"
            "This repository contains the frozen Cosmos3 understanding-path weights used by SimpleTuner "
            "to produce cached per-layer reasoner K/V tensors for training.\n"
        )
    _validate_no_local_paths(output_dir)
    return keys


def _parse_dtype(value: str) -> torch.dtype:
    normalized = value.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {value!r}.")


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
    keys = _write_component(
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
            commit_message="Add Cosmos3 reasoner component",
        )
        print(commit)


if __name__ == "__main__":
    main()
