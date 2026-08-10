#!/usr/bin/env python3
"""Benchmark the experimental MiniMax-H3 sparse attention operator."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

from simpletuner.helpers.models.minimaxh3.sparse_attention import (
    MiniMaxH3SparseAttentionConfig,
    MiniMaxH3SparseAttentionLayout,
    minimax_h3_sparse_attention,
)


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.replace("x", ",").split(","))
    if len(parts) != 3 or min(parts) <= 0:
        raise argparse.ArgumentTypeError("shape must contain three positive integers")
    return parts


def parse_fractions(value: str) -> list[float]:
    fractions = [float(part) for part in value.split(",")]
    if not fractions or any(fraction <= 0.0 or fraction > 1.0 for fraction in fractions):
        raise argparse.ArgumentTypeError("fractions must be in (0, 1]")
    return fractions


def benchmark(
    name: str,
    operation,
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    warmup: int,
    iterations: int,
    backward: bool,
) -> dict:
    timings = []
    for iteration in range(warmup + iterations):
        for tensor in tensors:
            tensor.grad = None
        torch.cuda.synchronize()
        if iteration == warmup:
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        output = operation(*tensors)
        if backward:
            output.float().square().mean().backward()
        torch.cuda.synchronize()
        if iteration >= warmup:
            timings.append(time.perf_counter() - start)
    return {
        "name": name,
        "mean_ms": statistics.mean(timings) * 1000.0,
        "median_ms": statistics.median(timings) * 1000.0,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-shape", type=parse_shape, default=(10, 30, 30))
    parser.add_argument("--block-shape", type=parse_shape, default=(1, 8, 16))
    parser.add_argument("--prefix-tokens", type=int, default=512)
    parser.add_argument("--heads", type=int, default=56)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--fractions", type=parse_fractions, default=[1.0, 0.75, 0.5, 0.25])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(17)
    torch.set_float32_matmul_precision("high")
    target_tokens = args.target_shape[0] * args.target_shape[1] * args.target_shape[2]
    sequence_length = args.prefix_tokens + target_tokens
    layout = MiniMaxH3SparseAttentionLayout(
        target_start=args.prefix_tokens,
        target_shape=args.target_shape,
    )
    tensors = tuple(
        torch.randn(
            1,
            sequence_length,
            args.heads,
            args.head_dim,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=not args.forward_only,
        )
        for _ in range(3)
    )

    def dense_attention(query, key, value):
        return F.scaled_dot_product_attention(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
        ).permute(0, 2, 1, 3)

    results = [
        benchmark(
            "dense_sdpa",
            dense_attention,
            tensors,
            warmup=args.warmup,
            iterations=args.iterations,
            backward=not args.forward_only,
        )
    ]
    for fraction in args.fractions:
        config = MiniMaxH3SparseAttentionConfig(
            mode="moba3d",
            block_shape=args.block_shape,
            video_kv_fraction=fraction,
        )

        def sparse_attention(query, key, value, sparse_config=config):
            return minimax_h3_sparse_attention(
                query,
                key,
                value,
                layout=layout,
                config=sparse_config,
            )

        operation = torch.compile(sparse_attention, dynamic=False) if args.compile else sparse_attention
        results.append(
            benchmark(
                f"moba3d_{fraction:.2f}",
                operation,
                tensors,
                warmup=args.warmup,
                iterations=args.iterations,
                backward=not args.forward_only,
            )
        )
    print(
        json.dumps(
            {
                "shape": {
                    "sequence": sequence_length,
                    "prefix": args.prefix_tokens,
                    "target": args.target_shape,
                    "block": args.block_shape,
                    "heads": args.heads,
                    "head_dim": args.head_dim,
                },
                "backward": not args.forward_only,
                "compiled": args.compile,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
