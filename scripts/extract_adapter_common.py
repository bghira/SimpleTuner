#!/usr/bin/env python3
"""Shared helpers for adapter extraction scripts."""

from __future__ import annotations

import json
import re
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from safetensors.torch import safe_open, save_file

WEIGHT_FILENAMES = ("diffusion_pytorch_model.safetensors", "model.safetensors")
INDEX_FILENAMES = (
    "diffusion_pytorch_model.safetensors.index.json",
    "model.safetensors.index.json",
)


def normalize_subfolder(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().strip("/")
    if value.lower() in {"", ".", "none", "null"}:
        return None
    return value


def parse_csv(value: Optional[str]) -> list[str]:
    if value in (None, "", "none", "None"):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype `{name}`. Use float32, float16, or bfloat16.") from exc


@dataclass
class TensorSource:
    label: str
    files_by_key: dict[str, Path]
    _exit_stack: Optional[ExitStack] = field(default=None, init=False, repr=False)
    _handles: Optional[dict[Path, Any]] = field(default=None, init=False, repr=False)

    def __enter__(self) -> "TensorSource":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @property
    def keys(self) -> set[str]:
        return set(self.files_by_key.keys())

    def close(self) -> None:
        if self._exit_stack is not None:
            self._exit_stack.close()
        self._exit_stack = None
        self._handles = None

    def _ensure_open(self) -> None:
        if self._exit_stack is None:
            self._exit_stack = ExitStack()
            self._handles = {}

    def _handle_for_path(self, path: Path):
        self._ensure_open()
        assert self._exit_stack is not None
        assert self._handles is not None
        handle = self._handles.get(path)
        if handle is None:
            handle = self._exit_stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            self._handles[path] = handle
        return handle

    def get_tensor(self, key: str) -> torch.Tensor:
        try:
            path = self.files_by_key[key]
        except KeyError as exc:
            raise KeyError(f"{self.label} does not contain tensor `{key}`.") from exc
        return self._handle_for_path(path).get_tensor(key)


def _index_from_safetensor_files(label: str, files: Iterable[Path]) -> TensorSource:
    files_by_key: dict[str, Path] = {}
    for file_path in files:
        with safe_open(file_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in files_by_key:
                    raise ValueError(f"Duplicate tensor key `{key}` found while reading {label}.")
                files_by_key[key] = file_path
    if not files_by_key:
        raise ValueError(f"No tensors found in {label}.")
    return TensorSource(label=label, files_by_key=files_by_key)


def _source_from_index_file(label: str, index_path: Path, component_dir: Path) -> TensorSource:
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError(f"Index file {index_path} is missing a `weight_map` object.")

    files_by_key: dict[str, Path] = {}
    for key, rel_file in weight_map.items():
        rel_path = Path(rel_file)
        candidate = component_dir / rel_path
        if not candidate.exists():
            candidate = component_dir / rel_path.name
        if not candidate.exists():
            raise FileNotFoundError(f"Index {index_path} references missing safetensors shard {rel_file}.")
        files_by_key[key] = candidate
    return TensorSource(label=label, files_by_key=files_by_key)


def _source_from_local_dir(label: str, root: Path, subfolder: Optional[str]) -> TensorSource:
    component_dir = root / subfolder if subfolder else root
    if not component_dir.is_dir():
        raise FileNotFoundError(f"Component directory not found: {component_dir}")

    for index_name in INDEX_FILENAMES:
        index_path = component_dir / index_name
        if index_path.exists():
            return _source_from_index_file(label, index_path, component_dir)

    for filename in WEIGHT_FILENAMES:
        candidate = component_dir / filename
        if candidate.exists():
            return _index_from_safetensor_files(label, [candidate])

    safetensors_files = sorted(component_dir.glob("*.safetensors"))
    if len(safetensors_files) == 1:
        return _index_from_safetensor_files(label, safetensors_files)
    if len(safetensors_files) > 1:
        raise ValueError(
            f"Multiple safetensors files found in {component_dir}, but no supported index file exists. "
            f"Expected one of: {', '.join(INDEX_FILENAMES)}."
        )
    raise FileNotFoundError(f"No supported safetensors weights found in {component_dir}.")


def _source_from_hub_repo(
    label: str,
    repo_id: str,
    subfolder: Optional[str],
    revision: Optional[str],
    cache_dir: Optional[str],
) -> TensorSource:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError as exc:
        raise ImportError("huggingface_hub is required to read remote model repositories.") from exc

    repo_files = set(list_repo_files(repo_id, revision=revision))
    prefix = f"{subfolder}/" if subfolder else ""

    for index_name in INDEX_FILENAMES:
        index_repo_path = f"{prefix}{index_name}"
        if index_repo_path not in repo_files:
            continue
        index_path = Path(hf_hub_download(repo_id, index_repo_path, revision=revision, cache_dir=cache_dir))
        with index_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Remote index {index_repo_path} is missing a `weight_map` object.")

        downloaded: dict[str, Path] = {}
        files_by_key: dict[str, Path] = {}
        for key, rel_file in weight_map.items():
            repo_file = rel_file if rel_file in repo_files else f"{prefix}{Path(rel_file).name}"
            if repo_file not in repo_files:
                raise FileNotFoundError(f"Remote index {index_repo_path} references missing shard {rel_file}.")
            if repo_file not in downloaded:
                downloaded[repo_file] = Path(hf_hub_download(repo_id, repo_file, revision=revision, cache_dir=cache_dir))
            files_by_key[key] = downloaded[repo_file]
        return TensorSource(label=label, files_by_key=files_by_key)

    for filename in WEIGHT_FILENAMES:
        repo_file = f"{prefix}{filename}"
        if repo_file in repo_files:
            path = Path(hf_hub_download(repo_id, repo_file, revision=revision, cache_dir=cache_dir))
            return _index_from_safetensor_files(label, [path])

    candidates = sorted(file for file in repo_files if file.startswith(prefix) and file.endswith(".safetensors"))
    if len(candidates) == 1:
        path = Path(hf_hub_download(repo_id, candidates[0], revision=revision, cache_dir=cache_dir))
        return _index_from_safetensor_files(label, [path])
    if len(candidates) > 1:
        raise ValueError(
            f"Remote repository {repo_id} has multiple safetensors files under `{prefix}` but no supported index file."
        )
    raise FileNotFoundError(f"No supported safetensors weights found in {repo_id} under `{prefix}`.")


def resolve_tensor_source(
    model_ref: str,
    *,
    label: str,
    subfolder: Optional[str],
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> TensorSource:
    expanded = Path(model_ref).expanduser()
    subfolder = normalize_subfolder(subfolder)
    if expanded.is_file():
        if expanded.suffix != ".safetensors":
            raise ValueError(f"Only .safetensors files are supported as direct file inputs: {expanded}")
        return _index_from_safetensor_files(label, [expanded])
    if expanded.is_dir():
        return _source_from_local_dir(label, expanded, subfolder)
    if str(model_ref).endswith(".safetensors"):
        raise FileNotFoundError(f"Safetensors file not found: {expanded}")
    return _source_from_hub_repo(label, model_ref, subfolder, revision, cache_dir)


def key_matches_module(key: str, target_modules: list[str]) -> bool:
    if not target_modules:
        return True
    module_name = key.removesuffix(".weight")
    return any(module_name == target or module_name.endswith(f".{target}") for target in target_modules)


def normalize_target_modules(raw: str) -> list[str]:
    value = raw.strip()
    if value == "all-linear":
        return []
    if value == "default":
        return ["to_q", "to_k", "to_v", "to_out.0"]
    return parse_csv(value)


def should_extract_key(
    key: str,
    tensor: torch.Tensor,
    *,
    target_modules: list[str],
    include: Optional[re.Pattern[str]],
    exclude: Optional[re.Pattern[str]],
    include_conv: bool = False,
) -> bool:
    if not key.endswith(".weight"):
        return False
    if tensor.ndim == 2:
        pass
    elif include_conv and tensor.ndim in {3, 4, 5}:
        pass
    else:
        return False
    if not key_matches_module(key, target_modules):
        return False
    if include is not None and include.search(key) is None:
        return False
    if exclude is not None and exclude.search(key) is not None:
        return False
    return True


def compile_optional_regex(value: Optional[str]) -> Optional[re.Pattern[str]]:
    if value in (None, "", "none", "None"):
        return None
    return re.compile(str(value))


def svd_low_rank(
    delta: torch.Tensor,
    *,
    rank: int,
    alpha: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rank <= 0:
        raise ValueError("rank must be greater than zero.")
    if alpha <= 0:
        raise ValueError("alpha must be greater than zero.")

    original_shape = tuple(delta.shape)
    if delta.ndim == 2:
        out_dim, in_dim = delta.shape
        flat = delta
        down_shape = (rank, in_dim)
        up_shape = (out_dim, rank)
    elif delta.ndim in {3, 4, 5}:
        out_dim = delta.shape[0]
        in_dim = delta.shape[1]
        kernel_shape = tuple(delta.shape[2:])
        flat = delta.reshape(out_dim, -1)
        down_shape = (rank, in_dim, *kernel_shape)
        up_shape = (out_dim, rank, *([1] * len(kernel_shape)))
    else:
        raise ValueError(f"Cannot decompose tensor with shape {original_shape}.")

    matrix = flat.to(device=device, dtype=torch.float32)
    max_rank = min(matrix.shape)
    effective_rank = min(rank, max_rank)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    scale_correction = float(rank) / float(alpha)

    down_flat = torch.zeros((rank, matrix.shape[1]), device=device, dtype=torch.float32)
    up = torch.zeros((matrix.shape[0], rank), device=device, dtype=torch.float32)
    down_flat[:effective_rank, :] = vh[:effective_rank, :]
    up[:, :effective_rank] = u[:, :effective_rank] * s[:effective_rank].unsqueeze(0) * scale_correction

    return down_flat.reshape(down_shape).cpu(), up.reshape(up_shape).cpu()


def save_safetensors_with_metadata(state_dict: dict[str, torch.Tensor], output: str, metadata: dict[str, str]) -> Path:
    output_path = Path(output).expanduser()
    if output_path.suffix != ".safetensors":
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "pytorch_lora_weights.safetensors"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(state_dict, str(output_path), metadata=metadata)
    return output_path
