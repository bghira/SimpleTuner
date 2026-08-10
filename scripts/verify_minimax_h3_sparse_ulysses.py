#!/usr/bin/env python3
"""Verify MiniMax-H3 sparse Ulysses attention against the unsharded operator."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from simpletuner.helpers.models.minimaxh3.sparse_attention import (
    MiniMaxH3SparseAttentionConfig,
    MiniMaxH3SparseAttentionLayout,
    minimax_h3_sparse_attention,
    minimax_h3_sparse_attention_ulysses,
)


def _maximum_across_ranks(value: float, device: torch.device) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def _run_case(
    *,
    name: str,
    video_kv_fraction: float,
    share_across_heads: bool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> tuple[float, float]:
    torch.manual_seed(1234)
    target_start = 129
    target_shape = (2, 8, 16)
    live_sequence_length = target_start + 2 * 8 * 16
    layout = MiniMaxH3SparseAttentionLayout(
        target_start=target_start,
        target_shape=target_shape,
        trailing_padding=(-live_sequence_length) % world_size,
    )
    config = MiniMaxH3SparseAttentionConfig(
        mode="moba3d",
        block_shape=(1, 8, 16),
        video_kv_fraction=video_kv_fraction,
        share_across_heads=share_across_heads,
    )
    sequence_length = live_sequence_length + layout.trailing_padding
    local_sequence_length = sequence_length // world_size
    heads = 56
    if heads % world_size != 0:
        raise ValueError(f"H3's {heads} attention heads do not divide by CP size {world_size}.")
    full_tensors = [
        torch.randn(
            1,
            sequence_length,
            heads,
            64,
            device=device,
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        for _ in range(3)
    ]

    baseline = minimax_h3_sparse_attention(
        *full_tensors,
        layout=layout,
        config=config,
    )
    shard = slice(
        rank * local_sequence_length,
        (rank + 1) * local_sequence_length,
    )
    local_tensors = [tensor.detach()[:, shard].clone().requires_grad_(True) for tensor in full_tensors]
    distributed = minimax_h3_sparse_attention_ulysses(
        *local_tensors,
        layout=layout,
        config=config,
        process_group=dist.group.WORLD,
    )

    global_indices = torch.arange(
        shard.start,
        shard.stop,
        device=device,
    )
    local_live = global_indices < sequence_length - layout.trailing_padding
    output_error = (distributed[:, local_live].float() - baseline[:, shard][:, local_live].float()).abs().max()

    live_rows = sequence_length - layout.trailing_padding
    denominator = baseline[:, :live_rows].numel()
    baseline_loss = baseline[:, :live_rows].float().square().sum() / denominator
    distributed_loss = distributed[:, local_live].float().square().sum() / denominator
    baseline_gradients = torch.autograd.grad(baseline_loss, full_tensors)
    distributed_gradients = torch.autograd.grad(distributed_loss, local_tensors)
    gradient_error = max(
        (distributed_gradient.float() - baseline_gradient[:, shard].float()).abs().max().item()
        for distributed_gradient, baseline_gradient in zip(distributed_gradients, baseline_gradients)
    )

    output_error = _maximum_across_ranks(output_error.item(), device)
    gradient_error = _maximum_across_ranks(gradient_error, device)
    if rank == 0:
        print(
            f"{name}: max_output_error={output_error:.8f}, " f"max_gradient_error={gradient_error:.8f}",
            flush=True,
        )
    return output_error, gradient_error


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if world_size < 2:
        raise ValueError("Run this verifier with at least two processes.")

    cases = (
        ("dense-budget", 1.0, False),
        ("routed-per-head", 0.5, False),
        ("routed-shared-heads", 0.5, True),
    )
    errors = [
        _run_case(
            name=name,
            video_kv_fraction=fraction,
            share_across_heads=shared,
            rank=rank,
            world_size=world_size,
            device=device,
        )
        for name, fraction, shared in cases
    ]
    if any(output > 1e-2 or gradient > 1e-3 for output, gradient in errors):
        raise RuntimeError("Sparse Ulysses verification exceeded its error tolerance.")
    if rank == 0:
        print("MiniMax-H3 sparse Ulysses verification passed.", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
