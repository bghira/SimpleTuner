from __future__ import annotations

from typing import Literal

import torch


def context_parallel_config(parallel_config):
    return getattr(parallel_config, "context_parallel_config", None)


def context_parallel_enabled(parallel_config) -> bool:
    return context_parallel_config(parallel_config) is not None


def shard_cp_tensor(tensor: torch.Tensor, parallel_config, split_dim: int = 1) -> torch.Tensor:
    context_config = context_parallel_config(parallel_config)
    if context_config is None:
        return tensor

    from diffusers.hooks.context_parallel import PartitionAnythingSharder

    return PartitionAnythingSharder.shard_anything(tensor, split_dim, context_config._flattened_mesh)


def unshard_cp_tensor(tensor: torch.Tensor, parallel_config, split_dim: int = 1) -> torch.Tensor:
    context_config = context_parallel_config(parallel_config)
    if context_config is None:
        return tensor

    from diffusers.hooks.context_parallel import PartitionAnythingSharder

    return PartitionAnythingSharder.unshard_anything(tensor, split_dim, context_config._flattened_mesh)


def prepare_cp_attention_mask(
    attention_mask: torch.Tensor | None,
    sequence_length: int,
    parallel_config,
    *,
    model_name: str,
    crop: Literal["left", "right"] = "right",
) -> torch.Tensor | None:
    if attention_mask is None:
        return None
    if attention_mask.ndim == 2:
        attention_mask = attention_mask[:, None, None, :]

    context_config = context_parallel_config(parallel_config)
    if context_config is None or getattr(context_config, "ulysses_degree", 1) <= 1:
        return attention_mask

    key_length = sequence_length * context_config.ulysses_degree
    mask_length = attention_mask.shape[-1]
    if mask_length < key_length:
        raise ValueError(
            f"{model_name} Ulysses attention mask length {mask_length} is shorter than attention key length {key_length}."
        )
    if mask_length > key_length:
        if crop == "left":
            attention_mask = attention_mask[..., :key_length]
        else:
            attention_mask = attention_mask[..., -key_length:]
    return attention_mask
