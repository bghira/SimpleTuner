#!/usr/bin/env python
"""Benchmark raw PyTorch bf16, fp8, and int8 matmul kernels.

This intentionally excludes TorchAO module wrappers and quantization casts from
the timed region. It measures only native PyTorch matmul calls:

  bf16: A @ B
  fp8: torch._scaled_mm(A_fp8, B_fp8_col_major, ...)
  int8: torch._int_mm(A_int8, B_int8)
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--m", type=int, default=2048)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run eager and compiled variants.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def cuda_time_ms(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        # Keep the result live until after the event is recorded.
        if out.numel() == 0:
            raise RuntimeError("empty output")
        timings.append(start.elapsed_time(end))
    torch.cuda.synchronize()
    return timings


def summarize(name: str, timings: list[float], flops: int) -> None:
    mean_ms = statistics.mean(timings)
    median_ms = statistics.median(timings)
    min_ms = min(timings)
    max_ms = max(timings)
    tflops = flops / (median_ms / 1000.0) / 1e12
    print(
        f"{name:18s} median={median_ms:8.3f} ms mean={mean_ms:8.3f} ms "
        f"min={min_ms:8.3f} ms max={max_ms:8.3f} ms approx={tflops:8.2f} TFLOP/s"
    )


def maybe_compile(name: str, fn: Callable[[], torch.Tensor], enabled: bool) -> Callable[[], torch.Tensor]:
    if not enabled:
        return fn
    compiled = torch.compile(fn, mode="max-autotune", fullgraph=True)
    # Trigger compilation outside measured warmup loop.
    compiled()
    torch.cuda.synchronize()
    print(f"compiled {name}")
    return compiled


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.device != "cuda":
        raise RuntimeError("This benchmark uses CUDA events and requires --device cuda")
    if not hasattr(torch, "_scaled_mm"):
        raise RuntimeError("torch._scaled_mm is unavailable")
    if not hasattr(torch, "_int_mm"):
        raise RuntimeError("torch._int_mm is unavailable")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    m, n, k = args.m, args.n, args.k
    print(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "shape": (m, k, n),
            "warmup": args.warmup,
            "iters": args.iters,
        }
    )

    # B is column-major for fp8 _scaled_mm. bf16 matmul can consume the same
    # strided B, so we use it there too to keep shapes/layouts aligned.
    a_bf16 = torch.randn(m, k, device=device, dtype=torch.bfloat16)
    b_bf16_col = torch.randn(n, k, device=device, dtype=torch.bfloat16).t()
    a_fp8 = a_bf16.to(torch.float8_e4m3fn)
    b_fp8_col = b_bf16_col.to(torch.float8_e4m3fn)
    scale_a = torch.ones((), device=device, dtype=torch.float32)
    scale_b = torch.ones((), device=device, dtype=torch.float32)
    a_int8 = torch.randint(-128, 127, (m, k), device=device, dtype=torch.int8)
    b_int8_col = torch.randint(-128, 127, (n, k), device=device, dtype=torch.int8).t()

    print(
        {
            "a_bf16_stride": a_bf16.stride(),
            "b_bf16_stride": b_bf16_col.stride(),
            "a_fp8_stride": a_fp8.stride(),
            "b_fp8_stride": b_fp8_col.stride(),
            "a_int8_stride": a_int8.stride(),
            "b_int8_stride": b_int8_col.stride(),
        }
    )

    def bf16_mm() -> torch.Tensor:
        return a_bf16 @ b_bf16_col

    def fp8_scaled_mm() -> torch.Tensor:
        return torch._scaled_mm(
            a_fp8,
            b_fp8_col,
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.bfloat16,
        )

    def int8_mm() -> torch.Tensor:
        return torch._int_mm(a_int8, b_int8_col)

    variants = [
        ("bf16", bf16_mm),
        ("fp8_scaled_mm", fp8_scaled_mm),
        ("int8_int_mm", int8_mm),
    ]
    run_compile_values = [False, True] if args.all else [args.compile]
    flops = 2 * m * n * k
    for compile_enabled in run_compile_values:
        for name, fn in variants:
            run_name = f"{name}/{'compiled' if compile_enabled else 'eager'}"
            bench_fn = maybe_compile(run_name, fn, compile_enabled)
            timings = cuda_time_ms(bench_fn, args.warmup, args.iters)
            summarize(run_name, timings, flops)


if __name__ == "__main__":
    main()
