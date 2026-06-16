from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open, save_file

LORA_A_SUFFIXES = (".lora_A.weight", ".lora.down.weight", ".lora_down.weight")
LORA_B_SUFFIXES = (".lora_B.weight", ".lora.up.weight", ".lora_up.weight")
ALPHA_SUFFIXES = (".alpha", ".lora_alpha")


@dataclass(frozen=True)
class LoRAModule:
    module_name: str
    original_module_name: str
    down_key: str
    up_key: str
    alpha_key: str | None
    rank: int
    in_dim: int
    out_dim: int


def _strip_component_prefix(module_name: str, component_prefix: str) -> str:
    prefix = f"{component_prefix}."
    if module_name.startswith(prefix):
        return module_name.removeprefix(prefix)
    return module_name


def _find_suffix(key: str, suffixes: tuple[str, ...]) -> str | None:
    for suffix in suffixes:
        if key.endswith(suffix):
            return suffix
    return None


def _as_float(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def load_safetensors(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"LoRA safetensors file not found: {path}")
    state_dict: dict[str, torch.Tensor] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            state_dict[key] = handle.get_tensor(key)
    return state_dict, metadata


def discover_lora_modules(state_dict: dict[str, torch.Tensor], *, component_prefix: str) -> dict[str, LoRAModule]:
    down_by_module: dict[str, tuple[str, str]] = {}
    up_by_module: dict[str, tuple[str, str]] = {}
    alpha_by_module: dict[str, str] = {}

    for key in state_dict:
        suffix = _find_suffix(key, LORA_A_SUFFIXES)
        if suffix is not None:
            original_module = key[: -len(suffix)]
            module = _strip_component_prefix(original_module, component_prefix)
            down_by_module[module] = (key, original_module)
            continue

        suffix = _find_suffix(key, LORA_B_SUFFIXES)
        if suffix is not None:
            original_module = key[: -len(suffix)]
            module = _strip_component_prefix(original_module, component_prefix)
            up_by_module[module] = (key, original_module)
            continue

        suffix = _find_suffix(key, ALPHA_SUFFIXES)
        if suffix is not None:
            original_module = key[: -len(suffix)]
            module = _strip_component_prefix(original_module, component_prefix)
            alpha_by_module[module] = key

    modules: dict[str, LoRAModule] = {}
    missing_up = sorted(set(down_by_module) - set(up_by_module))
    missing_down = sorted(set(up_by_module) - set(down_by_module))
    if missing_up or missing_down:
        raise ValueError(
            "LoRA file has incomplete A/B pairs. " f"Missing B for: {missing_up[:8]}; missing A for: {missing_down[:8]}."
        )

    for module_name, (down_key, original_module_name) in sorted(down_by_module.items()):
        up_key, _ = up_by_module[module_name]
        down = state_dict[down_key]
        up = state_dict[up_key]
        if down.ndim != 2 or up.ndim != 2:
            raise ValueError(
                f"Prompt2Effect currently supports linear PEFT LoRA weights only; `{module_name}` has "
                f"A shape {tuple(down.shape)} and B shape {tuple(up.shape)}."
            )
        rank, in_dim = down.shape
        out_dim, up_rank = up.shape
        if rank != up_rank:
            raise ValueError(f"Rank mismatch for `{module_name}`: A shape {tuple(down.shape)} vs B shape {tuple(up.shape)}.")
        modules[module_name] = LoRAModule(
            module_name=module_name,
            original_module_name=original_module_name,
            down_key=down_key,
            up_key=up_key,
            alpha_key=alpha_by_module.get(module_name),
            rank=int(rank),
            in_dim=int(in_dim),
            out_dim=int(out_dim),
        )
    if not modules:
        raise ValueError("No PEFT LoRA modules were found in the provided state dict.")
    return modules


def lora_delta(state_dict: dict[str, torch.Tensor], module: LoRAModule) -> torch.Tensor:
    down = state_dict[module.down_key].to(dtype=torch.float32)
    up = state_dict[module.up_key].to(dtype=torch.float32)
    alpha = float(module.rank)
    if module.alpha_key is not None:
        alpha = _as_float(state_dict[module.alpha_key])
    return (up @ down) * (alpha / float(module.rank))


def canonicalize_delta(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    if delta.ndim != 2:
        raise ValueError(f"Prompt2Effect canonicalization expects a rank-2 delta, got {tuple(delta.shape)}.")
    if rank <= 0:
        raise ValueError("rank must be greater than zero.")

    matrix = delta.to(dtype=torch.float32)
    out_dim, in_dim = matrix.shape
    effective_rank = min(rank, out_dim, in_dim)
    u, s, vh = torch.linalg.svd(matrix, full_matrices=False)

    if effective_rank > 0:
        u = u[:, :effective_rank].clone()
        s = s[:effective_rank].clone()
        vh = vh[:effective_rank, :].clone()
        for idx in range(effective_rank):
            pivot = torch.argmax(torch.abs(u[:, idx]))
            if u[pivot, idx] < 0:
                u[:, idx] = -u[:, idx]
                vh[idx, :] = -vh[idx, :]
        sqrt_s = torch.sqrt(s.clamp_min(0.0))
        b_star = u * sqrt_s.unsqueeze(0)
        a_star = sqrt_s.unsqueeze(1) * vh
    else:
        b_star = torch.empty((out_dim, 0), dtype=torch.float32)
        a_star = torch.empty((0, in_dim), dtype=torch.float32)

    if effective_rank < rank:
        padded_b = torch.zeros((out_dim, rank), dtype=torch.float32)
        padded_a = torch.zeros((rank, in_dim), dtype=torch.float32)
        padded_b[:, :effective_rank] = b_star
        padded_a[:effective_rank, :] = a_star
        return padded_a, padded_b
    return a_star.contiguous(), b_star.contiguous()


def normalized_frobenius_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diff = pred.float() - target.float()
    numerator = diff.square().flatten(1).sum(dim=1)
    denominator = target.float().square().flatten(1).sum(dim=1).clamp_min(eps)
    return (numerator / denominator).mean()


def save_generated_lora(
    output_path: str | Path,
    predictions: list[dict[str, torch.Tensor]],
    layers: list[dict[str, Any]],
    *,
    component_prefix: str,
    rank: int,
    dtype: torch.dtype,
    metadata: dict[str, str] | None = None,
) -> Path:
    output_path = Path(output_path).expanduser()
    if output_path.suffix != ".safetensors":
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / "pytorch_lora_weights.safetensors"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict: dict[str, torch.Tensor] = {}
    alpha = torch.tensor(float(rank), dtype=dtype)
    for prediction, layer in zip(predictions, layers):
        module_name = layer["module_name"]
        prefix = f"{component_prefix}.{module_name}" if component_prefix else module_name
        state_dict[f"{prefix}.lora_A.weight"] = prediction["A"].detach().cpu().to(dtype=dtype).contiguous()
        state_dict[f"{prefix}.lora_B.weight"] = prediction["B"].detach().cpu().to(dtype=dtype).contiguous()
        state_dict[f"{prefix}.alpha"] = alpha.clone()

    normalized_metadata = {str(k): str(v) for k, v in (metadata or {}).items()}
    save_file(state_dict, str(output_path), metadata=normalized_metadata)
    return output_path


def metadata_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
