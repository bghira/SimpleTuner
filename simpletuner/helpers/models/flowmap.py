import copy
from typing import Any

import torch


def clone_flowmap_embedder(embedder: torch.nn.Module) -> torch.nn.Module:
    return copy.deepcopy(embedder)


def validate_flowmap_deltatime_type(deltatime_type: str, *, model_name: str) -> str:
    if deltatime_type not in ("r", "t-r"):
        raise ValueError(f"{model_name} FlowMap deltatime_type must be 'r' or 't-r'.")
    return deltatime_type


def set_flowmap_gate(module: torch.nn.Module, gate_value: float, *, buffer_name: str = "flowmap_delta_emb_gate") -> None:
    gate = getattr(module, buffer_name)
    gate.data = torch.tensor([float(gate_value)], device=gate.device, dtype=gate.dtype)


def register_flowmap_config(module: Any, gate_value: float, deltatime_type: str) -> None:
    module.register_to_config(gate_value=float(gate_value), deltatime_type=deltatime_type)
    default_values = module.config.get("_use_default_values", []) or []
    default_values = [key for key in default_values if key not in ("gate_value", "deltatime_type")]
    module.register_to_config(_use_default_values=default_values)


def prepare_flowmap_delta_timestep(
    timestep: torch.Tensor,
    r_timestep: torch.Tensor,
    deltatime_type: str,
    *,
    model_name: str,
) -> torch.Tensor:
    validate_flowmap_deltatime_type(deltatime_type, model_name=model_name)
    if not torch.is_tensor(r_timestep):
        raise TypeError(f"{model_name} FlowMap r_timestep must be a torch.Tensor, got {type(r_timestep).__name__}.")

    r_timestep = r_timestep.to(device=timestep.device, dtype=timestep.dtype)
    if timestep.ndim == 0:
        if r_timestep.ndim == 0:
            pass
        elif r_timestep.numel() == 1:
            r_timestep = r_timestep.reshape(())
        else:
            raise ValueError(
                f"{model_name} FlowMap expected scalar r_timestep for scalar timesteps, got shape {tuple(r_timestep.shape)}."
            )
    elif timestep.ndim == 1:
        if r_timestep.ndim == 0:
            r_timestep = r_timestep.expand(timestep.shape[0])
        elif r_timestep.ndim == 1:
            if r_timestep.shape[0] == 1:
                r_timestep = r_timestep.expand(timestep.shape[0])
            elif r_timestep.shape[0] != timestep.shape[0]:
                raise ValueError(
                    f"{model_name} FlowMap expected 1 or {timestep.shape[0]} r_timestep values, got {r_timestep.shape[0]}."
                )
        else:
            raise ValueError(
                f"{model_name} FlowMap expected scalar or 1D r_timestep for batch timesteps, got shape {tuple(r_timestep.shape)}."
            )
    elif timestep.ndim == 2:
        if r_timestep.ndim == 0:
            r_timestep = r_timestep.expand(timestep.shape)
        elif r_timestep.ndim == 1:
            if r_timestep.shape[0] == 1:
                r_timestep = r_timestep.expand(timestep.shape[0])
            elif r_timestep.shape[0] != timestep.shape[0]:
                raise ValueError(
                    f"{model_name} FlowMap expected 1 or {timestep.shape[0]} batch r_timestep values, got {r_timestep.shape[0]}."
                )
            r_timestep = r_timestep[:, None].expand(-1, timestep.shape[1])
        elif r_timestep.ndim == 2:
            if r_timestep.shape[1] != timestep.shape[1]:
                raise ValueError(
                    f"{model_name} FlowMap tokenwise r_timestep expected sequence length {timestep.shape[1]}, got {r_timestep.shape[1]}."
                )
            if r_timestep.shape[0] == 1:
                r_timestep = r_timestep.expand(timestep.shape[0], -1)
            elif r_timestep.shape[0] != timestep.shape[0]:
                raise ValueError(
                    f"{model_name} FlowMap tokenwise r_timestep expected batch size {timestep.shape[0]}, got {r_timestep.shape[0]}."
                )
        else:
            raise ValueError(
                f"{model_name} FlowMap expected scalar, 1D, or 2D r_timestep, got shape {tuple(r_timestep.shape)}."
            )
    else:
        raise ValueError(f"{model_name} FlowMap expected scalar, 1D, or 2D timesteps, got shape {tuple(timestep.shape)}.")

    if deltatime_type == "r":
        return r_timestep
    return timestep - r_timestep


def flowmap_timestep_embedding(
    *,
    time_proj: Any,
    timestep_embedder: torch.nn.Module,
    timestep: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    original_shape = timestep.shape
    timestep_flat = timestep.reshape(-1)
    timesteps_proj = time_proj(timestep_flat)
    embedding = timestep_embedder(timesteps_proj.to(dtype=dtype))
    if timestep.ndim > 1:
        embedding = embedding.view(*original_shape, -1)
    return embedding


def blend_flowmap_embeddings(
    base_embedding: torch.Tensor,
    delta_embedding: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    gate = gate.to(device=base_embedding.device, dtype=base_embedding.dtype)
    return (1.0 - gate) * base_embedding + gate * delta_embedding.to(
        device=base_embedding.device, dtype=base_embedding.dtype
    )
