#!/usr/bin/env python
"""Extract Ideogram 4 text projection weights into a standalone component."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from posixpath import dirname as posix_dirname
from posixpath import join as posix_join

import requests
from huggingface_hub import hf_hub_download, hf_hub_url
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.utils import build_hf_headers
from safetensors import safe_open
from safetensors.torch import save_file

from simpletuner.helpers.models.ideogram.quantized_loading import FP8_SCALE_SUFFIX
from simpletuner.helpers.models.ideogram.text_projection import TEXT_PROJECTION_KEY_PREFIXES, Ideogram4TextProjectionConfig


def _resolve_index(source_repo: str, index_filename: str, revision: str | None) -> Path:
    local_index = Path(source_repo) / index_filename
    if local_index.is_file():
        return local_index
    return Path(hf_hub_download(repo_id=source_repo, filename=index_filename, revision=revision))


def _resolve_file(source_repo: str, filename: str, revision: str | None) -> Path:
    local_file = Path(source_repo) / filename
    if local_file.is_file():
        return local_file
    return Path(hf_hub_download(repo_id=source_repo, filename=filename, revision=revision))


def _resolve_shard(
    source_repo: str,
    index_path: Path,
    index_filename: str,
    shard_name: str,
    revision: str | None,
) -> Path:
    local_source = Path(source_repo)
    if local_source.exists():
        local_shard = index_path.parent / shard_name
        if local_shard.is_file():
            return local_shard

    shard_dir = posix_dirname(index_filename)
    shard_repo_path = posix_join(shard_dir, shard_name) if shard_dir else shard_name
    return Path(hf_hub_download(repo_id=source_repo, filename=shard_repo_path, revision=revision))


def _load_projection_from_safetensors(weights_path: Path) -> dict:
    state_dict = {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        selected_keys = [
            key for key in handle.keys() if any(key.startswith(prefix) for prefix in TEXT_PROJECTION_KEY_PREFIXES)
        ]
        if not selected_keys:
            raise ValueError(f"No Ideogram text projection keys were found in {weights_path}.")
        for key in selected_keys:
            state_dict[key] = handle.get_tensor(key)
    return state_dict


def _read_remote_range(url: str, headers: dict[str, str], start: int, end: int) -> bytes:
    response_headers = {**headers, "Range": f"bytes={start}-{end}"}
    with requests.get(
        url,
        headers=response_headers,
        stream=True,
        timeout=(10, 600),
    ) as response:
        if response.status_code != 206:
            raise RuntimeError(f"Expected HTTP 206 for range {start}-{end} from {url}, got {response.status_code}.")
        data = b"".join(response.iter_content(chunk_size=1024 * 1024))

    expected_size = end - start + 1
    if len(data) != expected_size:
        raise RuntimeError(f"Expected {expected_size} bytes for range {start}-{end} from {url}, got {len(data)}.")
    return data


def _read_remote_safetensors_header(source_repo: str, filename: str, revision: str | None) -> tuple[dict, int]:
    url = hf_hub_url(repo_id=source_repo, filename=filename, revision=revision)
    headers = build_hf_headers(token=None)
    header_size_bytes = _read_remote_range(url, headers, 0, 7)
    header_size = struct.unpack("<Q", header_size_bytes)[0]
    header_bytes = _read_remote_range(url, headers, 8, 8 + header_size - 1)
    header = json.loads(header_bytes.decode("utf-8"))
    return header, header_size


def _read_remote_safetensors_tensors(
    source_repo: str,
    filename: str,
    selected_keys: list[str] | None,
    revision: str | None,
) -> list[tuple[str, dict, bytes]]:
    url = hf_hub_url(repo_id=source_repo, filename=filename, revision=revision)
    headers = build_hf_headers(token=None)
    header, header_size = _read_remote_safetensors_header(source_repo, filename, revision)
    if selected_keys is None:
        selected_keys = [key for key in header if any(key.startswith(prefix) for prefix in TEXT_PROJECTION_KEY_PREFIXES)]
    missing = [key for key in selected_keys if key not in header]
    if missing:
        raise KeyError(f"Expected keys missing from {filename}: {missing[:10]}")
    if not selected_keys:
        raise ValueError(f"No Ideogram text projection keys were found in {filename}.")

    data_start = 8 + header_size
    tensors = []
    for key in selected_keys:
        tensor_info = header[key]
        start, end = tensor_info["data_offsets"]
        tensor_bytes = _read_remote_range(url, headers, data_start + start, data_start + end - 1)
        tensors.append((key, tensor_info, tensor_bytes))
    return tensors


def _write_safetensors_subset(output_path: Path, tensors: list[tuple[str, dict, bytes]]) -> list[str]:
    header = {}
    data_blocks = []
    offset = 0
    for key, tensor_info, tensor_bytes in tensors:
        end = offset + len(tensor_bytes)
        header[key] = {
            "dtype": tensor_info["dtype"],
            "shape": tensor_info["shape"],
            "data_offsets": [offset, end],
        }
        data_blocks.append(tensor_bytes)
        offset = end

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(output_path, "wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for data_block in data_blocks:
            handle.write(data_block)
    return list(header)


def _load_projection_state_dict(
    source_repo: str,
    index_filename: str,
    weights_filename: str,
    revision: str | None,
) -> dict:
    try:
        index_path = _resolve_index(source_repo, index_filename, revision)
    except EntryNotFoundError:
        weights_path = _resolve_file(source_repo, weights_filename, revision)
        return _load_projection_from_safetensors(weights_path)

    with open(index_path, "r", encoding="utf-8") as handle:
        index = json.load(handle)

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Index file {index_path} does not contain a weight_map object.")

    selected_keys = [key for key in weight_map if any(key.startswith(prefix) for prefix in TEXT_PROJECTION_KEY_PREFIXES)]
    if not selected_keys:
        raise ValueError(f"No Ideogram text projection keys were found in {index_path}.")

    state_dict = {}
    keys_by_shard: dict[str, list[str]] = {}
    for key in selected_keys:
        keys_by_shard.setdefault(weight_map[key], []).append(key)

    for shard_name, keys in keys_by_shard.items():
        shard_path = _resolve_shard(source_repo, index_path, index_filename, shard_name, revision)
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            present = set(handle.keys())
            for key in keys:
                if key not in present:
                    raise KeyError(f"Expected key {key} in shard {shard_path}.")
                state_dict[key] = handle.get_tensor(key)
    return state_dict


def _extract_remote_projection_state_dict(
    source_repo: str,
    index_filename: str,
    weights_filename: str,
    revision: str | None,
    output_path: Path,
) -> list[str]:
    try:
        index_path = _resolve_index(source_repo, index_filename, revision)
    except EntryNotFoundError:
        tensors = _read_remote_safetensors_tensors(source_repo, weights_filename, None, revision)
        return _write_safetensors_subset(output_path, tensors)

    with open(index_path, "r", encoding="utf-8") as handle:
        index = json.load(handle)

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Index file {index_path} does not contain a weight_map object.")

    selected_keys = [key for key in weight_map if any(key.startswith(prefix) for prefix in TEXT_PROJECTION_KEY_PREFIXES)]
    if not selected_keys:
        raise ValueError(f"No Ideogram text projection keys were found in {index_path}.")

    keys_by_shard: dict[str, list[str]] = {}
    for key in selected_keys:
        keys_by_shard.setdefault(weight_map[key], []).append(key)

    shard_dir = posix_dirname(index_filename)
    tensors = []
    for shard_name, keys in keys_by_shard.items():
        shard_repo_path = posix_join(shard_dir, shard_name) if shard_dir else shard_name
        tensors.extend(_read_remote_safetensors_tensors(source_repo, shard_repo_path, keys, revision))
    return _write_safetensors_subset(output_path, tensors)


def _extract_projection_state_dict(
    source_repo: str,
    index_filename: str,
    weights_filename: str,
    revision: str | None,
    output_path: Path,
) -> list[str]:
    if Path(source_repo).exists():
        state_dict = _load_projection_state_dict(source_repo, index_filename, weights_filename, revision)
        save_file(state_dict, output_path)
        return list(state_dict)
    return _extract_remote_projection_state_dict(
        source_repo,
        index_filename,
        weights_filename,
        revision,
        output_path,
    )


def _detect_quantization_from_keys(keys: list[str]) -> str:
    if any(".quant_state.bitsandbytes__" in key for key in keys):
        return "nf4"
    if any(key.endswith(FP8_SCALE_SUFFIX) for key in keys):
        return "fp8"
    return "none"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", help="Ideogram 4 model repo or local snapshot directory.")
    parser.add_argument("output_dir", help="Directory where the component files will be written.")
    parser.add_argument(
        "--index_filename",
        default="transformer/diffusion_pytorch_model.safetensors.index.json",
        help="Transformer safetensors index path inside source_repo. Falls back to --weights_filename when missing.",
    )
    parser.add_argument(
        "--weights_filename",
        default="transformer/diffusion_pytorch_model.safetensors",
        help="Single transformer safetensors path inside source_repo.",
    )
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision for source_repo.")
    parser.add_argument("--source_revision", default=None, help="Revision value to record in component config.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    keys = _extract_projection_state_dict(
        args.source_repo,
        args.index_filename,
        args.weights_filename,
        args.revision,
        output_dir / "model.safetensors",
    )
    quantization = _detect_quantization_from_keys(keys)
    config = Ideogram4TextProjectionConfig(
        source_repo=args.source_repo,
        source_revision=args.source_revision or args.revision,
        quantization=quantization,
    )

    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(output_dir / "README.md", "w", encoding="utf-8") as handle:
        handle.write(
            "# Ideogram 4 Text Projection Component\n\n"
            "This repository contains only `llm_cond_norm` and `llm_cond_proj` "
            "from an Ideogram 4 transformer checkpoint. SimpleTuner uses it to "
            "project Qwen hidden-state stacks before writing text embed cache files.\n"
        )

    print(f"Wrote {len(keys)} tensors to {output_dir} ({quantization}).")


if __name__ == "__main__":
    main()
