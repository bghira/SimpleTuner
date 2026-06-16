from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

try:
    from extract_adapter_common import normalize_subfolder, resolve_tensor_source
except ImportError:  # pragma: no cover - direct module import path is set by script bootstrap.
    from scripts.extract_adapter_common import normalize_subfolder, resolve_tensor_source


def candidate_base_keys(module_name: str, *, component_prefix: str) -> list[str]:
    candidates = [f"{module_name}.weight"]
    prefix = f"{component_prefix}."
    if module_name.startswith(prefix):
        candidates.append(f"{module_name.removeprefix(prefix)}.weight")
    else:
        candidates.append(f"{prefix}{module_name}.weight")
    return list(dict.fromkeys(candidates))


def resolve_base_key(base_keys: set[str], module_name: str, *, component_prefix: str) -> str:
    for candidate in candidate_base_keys(module_name, component_prefix=component_prefix):
        if candidate in base_keys:
            return candidate
    raise KeyError(
        f"Could not find a base model tensor for LoRA module `{module_name}`. Tried: "
        + ", ".join(candidate_base_keys(module_name, component_prefix=component_prefix))
    )


def load_base_weights(
    base_model: str,
    layers: list[dict[str, Any]],
    *,
    component_subfolder: str | None,
    revision: str | None,
    cache_dir: str | None,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    source = resolve_tensor_source(
        base_model,
        label="base",
        subfolder=normalize_subfolder(component_subfolder),
        revision=revision,
        cache_dir=cache_dir,
    )
    weights: list[torch.Tensor] = []
    with source:
        for layer in layers:
            key = layer["base_key"]
            if key not in source.keys:
                raise KeyError(f"Base model does not contain required Prompt2Effect tensor `{key}`.")
            tensor = source.get_tensor(key)
            if tensor.ndim != 2:
                raise ValueError(f"Base tensor `{key}` must be rank-2, got {tuple(tensor.shape)}.")
            expected_shape = (int(layer["out_dim"]), int(layer["in_dim"]))
            if tuple(tensor.shape) != expected_shape:
                raise ValueError(
                    f"Base tensor `{key}` has shape {tuple(tensor.shape)}, expected {expected_shape} from schema."
                )
            weights.append(tensor.to(dtype=dtype).contiguous())
    return weights


def infer_base_layers(
    base_model: str,
    module_names: list[str],
    *,
    component_prefix: str,
    component_subfolder: str | None,
    revision: str | None,
    cache_dir: str | None,
) -> dict[str, tuple[str, tuple[int, int]]]:
    source = resolve_tensor_source(
        base_model,
        label="base",
        subfolder=normalize_subfolder(component_subfolder),
        revision=revision,
        cache_dir=cache_dir,
    )
    result: dict[str, tuple[str, tuple[int, int]]] = {}
    with source:
        base_keys = source.keys
        for module_name in module_names:
            key = resolve_base_key(base_keys, module_name, component_prefix=component_prefix)
            tensor = source.get_tensor(key)
            if tensor.ndim != 2:
                raise ValueError(
                    f"Prompt2Effect currently supports linear base tensors only; `{key}` is {tuple(tensor.shape)}."
                )
            out_dim, in_dim = tensor.shape
            result[module_name] = (key, (int(out_dim), int(in_dim)))
    return result


def resolve_existing_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.exists():
        raise FileNotFoundError(f"Path not found: {resolved}")
    return resolved
