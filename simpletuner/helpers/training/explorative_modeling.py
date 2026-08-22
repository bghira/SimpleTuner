from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, cast

import torch

XMTrainingTarget = Literal["noise", "route"]
XMSelectionScope = Literal["sample", "block"]


@dataclass(frozen=True)
class ExplorativeModelingConfig:
    enabled: bool
    candidate_count: int
    training_target: XMTrainingTarget
    selection_scope: XMSelectionScope
    block_size: int

    @classmethod
    def from_config(cls, config) -> "ExplorativeModelingConfig":
        def config_value(name: str, default):
            if isinstance(config, dict):
                return config.get(name, default)
            try:
                values = vars(config)
            except TypeError:
                values = None
            if values is not None:
                return values.get(name, default)
            return getattr(config, name, default)

        enabled = bool(config_value("xm_enabled", False))
        candidate_count = int(config_value("xm_candidate_count", 1) or 1)
        training_target = str(config_value("xm_training_target", "noise") or "noise")
        selection_scope = str(config_value("xm_selection_scope", "sample") or "sample")
        block_size = int(config_value("xm_block_size", 0) or 0)

        if training_target not in ("noise", "route"):
            raise ValueError("xm_training_target must be 'noise' or 'route'.")
        if selection_scope not in ("sample", "block"):
            raise ValueError("xm_selection_scope must be 'sample' or 'block'.")
        if enabled and candidate_count < 2:
            raise ValueError("xm_candidate_count must be at least 2 when XM is enabled.")
        if block_size < 0:
            raise ValueError("xm_block_size must be non-negative.")
        if selection_scope == "block" and block_size == 1:
            raise ValueError("xm_block_size=1 would select winners per token; use sample scope or a larger block.")

        return cls(
            enabled=enabled,
            candidate_count=candidate_count,
            training_target=cast(XMTrainingTarget, training_target),
            selection_scope=cast(XMSelectionScope, selection_scope),
            block_size=block_size,
        )


def reduce_loss_to_samples(loss: torch.Tensor) -> torch.Tensor:
    if loss.ndim == 0:
        return loss.reshape(1)
    return loss.float().mean(dim=tuple(range(1, loss.ndim)))


def select_min_candidate_loss(candidate_losses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate_losses.ndim != 2:
        raise ValueError(f"XM candidate losses must have shape [candidates, batch], got {tuple(candidate_losses.shape)}.")
    selected_loss, winner_indices = candidate_losses.min(dim=0)
    return selected_loss.mean(), winner_indices


def repeat_batch_for_candidates(value, candidate_count: int):
    if not torch.is_tensor(value):
        return value
    if value.ndim == 0:
        return value
    repeat_shape = [candidate_count] + [1] * (value.ndim - 1)
    return value.repeat(repeat_shape)


def reshape_candidate_batch(value: torch.Tensor, candidate_count: int) -> torch.Tensor:
    if candidate_count < 1:
        raise ValueError("candidate_count must be positive.")
    if value.shape[0] % candidate_count != 0:
        raise ValueError(f"Tensor batch dimension {value.shape[0]} is not divisible by candidate_count={candidate_count}.")
    batch_size = value.shape[0] // candidate_count
    return value.reshape(candidate_count, batch_size, *value.shape[1:])


def select_winning_candidates(value: torch.Tensor, winner_indices: torch.Tensor, candidate_count: int) -> torch.Tensor:
    candidate_view = reshape_candidate_batch(value, candidate_count)
    if winner_indices.ndim != 1 or winner_indices.shape[0] != candidate_view.shape[1]:
        raise ValueError(
            "winner_indices must have shape [batch] matching the candidate-expanded tensor's original batch size."
        )
    gather_index = winner_indices.to(device=value.device, dtype=torch.long)
    batch_positions = torch.arange(candidate_view.shape[1], device=value.device)
    return candidate_view[gather_index, batch_positions]


def blockwise_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    block_size: int = 0,
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"Expected logits with shape [batch, seq, vocab], got {tuple(logits.shape)}.")
    if targets.shape != logits.shape[:2]:
        raise ValueError(f"Targets must have shape {tuple(logits.shape[:2])}, got {tuple(targets.shape)}.")

    flat_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).reshape(targets.shape)
    valid = targets.ne(ignore_index)
    if block_size <= 0:
        denom = valid.sum(dim=1).clamp_min(1)
        return (flat_loss * valid.to(flat_loss.dtype)).sum(dim=1) / denom

    pad = (-targets.shape[1]) % block_size
    if pad:
        flat_loss = torch.nn.functional.pad(flat_loss, (0, pad))
        valid = torch.nn.functional.pad(valid, (0, pad))
    block_loss = flat_loss.reshape(flat_loss.shape[0], -1, block_size)
    block_valid = valid.reshape(valid.shape[0], -1, block_size)
    denom = block_valid.sum(dim=2).clamp_min(1)
    per_block = (block_loss * block_valid.to(block_loss.dtype)).sum(dim=2) / denom
    has_block = block_valid.any(dim=2)
    sample_denom = has_block.sum(dim=1).clamp_min(1)
    return (per_block * has_block.to(per_block.dtype)).sum(dim=1) / sample_denom


def route_usage_histogram(winner_indices: torch.Tensor, candidate_count: int) -> Optional[torch.Tensor]:
    if winner_indices.numel() == 0:
        return None
    return torch.bincount(winner_indices.to(dtype=torch.long), minlength=candidate_count).float()
