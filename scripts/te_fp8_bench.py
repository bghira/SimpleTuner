#!/usr/bin/env python
"""Benchmark TransformerEngine Linear FP8 training kernels.

This intentionally stays outside SimpleTuner's trainer. It answers the first
question we need before wiring a new precision preset: does TransformerEngine
FP8 beat PyTorch BF16 for the GEMM shapes we care about on this host?
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
from collections.abc import Callable, Iterator

import torch
import torch.nn as nn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--ffn-hidden", type=int, default=11008)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recipe", choices=["delayed", "current"], default="delayed")
    parser.add_argument("--fp8-format", choices=["hybrid", "e4m3"], default="hybrid")
    parser.add_argument("--freeze-weights", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--all", action="store_true", help="Run PyTorch BF16, TE BF16, and TE FP8.")
    return parser.parse_args()


def cuda_time_ms(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        out = fn()
        if out.numel() == 0:
            raise RuntimeError("empty output")
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        end.synchronize()
        if out.numel() == 0:
            raise RuntimeError("empty output")
        timings.append(start.elapsed_time(end))
    torch.cuda.synchronize()
    return timings


def summarize(name: str, timings: list[float]) -> None:
    print(
        f"{name:16s} median={statistics.median(timings):8.3f} ms "
        f"mean={statistics.mean(timings):8.3f} ms min={min(timings):8.3f} ms max={max(timings):8.3f} ms"
    )


class TorchMLP(nn.Module):
    def __init__(self, hidden: int, ffn_hidden: int, layers: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        blocks = []
        for _ in range(layers):
            blocks.extend(
                [
                    nn.Linear(hidden, ffn_hidden, bias=False, dtype=dtype, device=device),
                    nn.GELU(),
                    nn.Linear(ffn_hidden, hidden, bias=False, dtype=dtype, device=device),
                ]
            )
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_te_mlp(hidden: int, ffn_hidden: int, layers: int, dtype: torch.dtype, device: torch.device) -> nn.Module:
    import transformer_engine.pytorch as te

    blocks = []
    for _ in range(layers):
        blocks.extend(
            [
                te.Linear(hidden, ffn_hidden, bias=False, params_dtype=dtype, device=device),
                nn.GELU(),
                te.Linear(ffn_hidden, hidden, bias=False, params_dtype=dtype, device=device),
            ]
        )
    return nn.Sequential(*blocks)


def make_te_recipe(args: argparse.Namespace):
    from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, Format

    fp8_format = Format.HYBRID if args.fp8_format == "hybrid" else Format.E4M3
    if args.recipe == "current":
        return Float8CurrentScaling(fp8_format=fp8_format)
    return DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")


def te_autocast(enabled: bool, recipe) -> Iterator[None]:
    if not enabled:
        return contextlib.nullcontext()
    import transformer_engine.pytorch as te

    return te.autocast(enabled=True, recipe=recipe)


def maybe_compile(name: str, fn: Callable[[], torch.Tensor], enabled: bool) -> Callable[[], torch.Tensor]:
    if not enabled:
        return fn
    compiled = torch.compile(fn, mode="max-autotune", fullgraph=False)
    compiled()
    torch.cuda.synchronize()
    print(f"compiled {name}")
    return compiled


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    dtype = torch.bfloat16
    shape = (args.batch, args.seq, args.hidden)

    print(
        {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
            "shape": shape,
            "ffn_hidden": args.ffn_hidden,
            "layers": args.layers,
            "freeze_weights": args.freeze_weights,
            "compile": args.compile,
        }
    )

    try:
        import transformer_engine
        import transformer_engine.pytorch as te

        print({"transformer_engine": getattr(transformer_engine, "__version__", "unknown")})
        print({"te_fp8_available": te.is_fp8_available(return_reason=True)})
    except ImportError as exc:
        raise ImportError(
            "TransformerEngine is not installed. Try: "
            ".venv/bin/pip install --no-build-isolation 'transformer_engine[pytorch]'"
        ) from exc

    recipe = make_te_recipe(args)
    variants = ["torch-bf16", "te-bf16", "te-fp8"] if args.all else ["te-fp8"]
    for variant in variants:
        torch.cuda.empty_cache()
        x = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
        if variant == "torch-bf16":
            model = TorchMLP(args.hidden, args.ffn_hidden, args.layers, dtype, device)
            ctx = contextlib.nullcontext()
        else:
            model = make_te_mlp(args.hidden, args.ffn_hidden, args.layers, dtype, device)
            ctx = te_autocast(variant == "te-fp8", recipe)
        if args.freeze_weights:
            for param in model.parameters():
                param.requires_grad_(False)

        def step() -> torch.Tensor:
            if x.grad is not None:
                x.grad = None
            for param in model.parameters():
                param.grad = None
            with ctx:
                out = model(x)
                loss = out.float().square().mean()
            loss.backward()
            return loss.detach()

        bench = maybe_compile(variant, step, args.compile)
        timings = cuda_time_ms(bench, args.warmup, args.iters)
        summarize(variant, timings)


if __name__ == "__main__":
    main()
