#!/usr/bin/env python
"""Small TorchAO training repro for SimpleTuner quantization experiments.

Examples:
  .venv/bin/python scripts/torchao_mre.py --mode bf16 --profile
  .venv/bin/python scripts/torchao_mre.py --mode fp8-training --profile --compile
  .venv/bin/python scripts/torchao_mre.py --mode int8-training --profile
  .venv/bin/python scripts/torchao_mre.py --mode int8-training --profile --simpletuner-workaround
"""

from __future__ import annotations

import argparse
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


@dataclass
class StepStats:
    index: int
    elapsed_ms: float
    loss: float


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        nn.init.normal_(self.lora_a.weight, std=0.02)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scale


class TinyMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, depth: int, lora_rank: int | None):
        super().__init__()
        layers: list[nn.Module] = []
        for idx in range(depth):
            in_dim = dim if idx == 0 else hidden_dim
            out_dim = dim if idx == depth - 1 else hidden_dim
            linear: nn.Module = nn.Linear(in_dim, out_dim, bias=False)
            if lora_rank is not None:
                linear = LoRALinear(linear, rank=lora_rank)
            layers.append(linear)
            if idx != depth - 1:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["bf16", "fp8-training", "int8-training"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--quantize-lora", action="store_true")
    parser.add_argument("--full-finetune", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--simpletuner-workaround", action="store_true")
    parser.add_argument("--channels-last-input", action="store_true")
    parser.add_argument("--print-param-types", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def make_linear_filter(quantize_lora: bool):
    def linear_filter(mod: nn.Module, fqn: str) -> bool:
        if not isinstance(mod, nn.Linear):
            return False
        if not quantize_lora and (".lora_a" in fqn or ".lora_b" in fqn):
            return False
        return mod.in_features % 16 == 0 and mod.out_features % 16 == 0

    return linear_filter


def quantize_model(model: nn.Module, mode: str, simpletuner_workaround: bool, quantize_lora: bool) -> nn.Module:
    if mode == "bf16":
        return model
    if simpletuner_workaround:
        import simpletuner.helpers.training.quantisation.torchao_workarounds  # noqa: F401

    linear_filter = make_linear_filter(quantize_lora)
    if mode == "fp8-training":
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training

        return convert_to_float8_training(
            model,
            module_filter_fn=linear_filter,
            config=Float8LinearConfig(pad_inner_dim=True),
        )
    if mode == "int8-training":
        from torchao.prototype.quantized_training import int8_weight_only_quantized_training
        from torchao.quantization import quantize_

        quantize_(model, int8_weight_only_quantized_training(), filter_fn=linear_filter)
        return model
    raise ValueError(f"Unknown mode: {mode}")


def trainable_params(model: nn.Module) -> Iterable[nn.Parameter]:
    return (param for param in model.parameters() if param.requires_grad)


def freeze_non_lora_params(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if ".lora_a." in name or ".lora_b." in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)


def print_param_types(model: nn.Module) -> None:
    print("parameter types:")
    for idx, (name, param) in enumerate(model.named_parameters()):
        if idx >= 20:
            remaining = sum(1 for _ in model.named_parameters()) - idx
            print(f"  ... {remaining} more parameters")
            break
        print(
            " ",
            name,
            {
                "parameter_cls": type(param).__name__,
                "data_cls": type(param.data).__name__,
                "dtype": str(param.dtype),
                "requires_grad": param.requires_grad,
                "shape": tuple(param.shape),
            },
        )


def make_batch(args: argparse.Namespace, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(args.batch_size, args.tokens, args.dim, device=args.device, dtype=dtype)
    y = torch.randn(args.batch_size, args.tokens, args.dim, device=args.device, dtype=dtype)
    if args.channels_last_input:
        side = int(args.tokens**0.5)
        if side * side != args.tokens:
            raise ValueError("--channels-last-input requires --tokens to be a perfect square")
        x = (
            x.reshape(args.batch_size, side, side, args.dim)
            .to(memory_format=torch.channels_last)
            .reshape(args.batch_size, args.tokens, args.dim)
        )
        y = (
            y.reshape(args.batch_size, side, side, args.dim)
            .to(memory_format=torch.channels_last)
            .reshape(args.batch_size, args.tokens, args.dim)
        )
    return x, y


def maybe_profile(enabled: bool, active_steps: int):
    if not enabled:
        return nullcontext()
    from torch.profiler import ProfilerActivity, profile, schedule

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return profile(
        activities=activities,
        schedule=schedule(wait=0, warmup=0, active=active_steps, repeat=1),
        record_shapes=True,
        profile_memory=True,
    )


def synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    dtype = dtype_from_name(args.dtype)
    lora_rank = None if args.full_finetune else args.lora_rank
    model = TinyMLP(args.dim, args.hidden_dim, args.depth, lora_rank=lora_rank).to(args.device, dtype=dtype)
    model = quantize_model(model, args.mode, args.simpletuner_workaround, args.quantize_lora)
    if not args.full_finetune:
        freeze_non_lora_params(model)
    if args.print_param_types:
        print_param_types(model)
    if args.compile:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
    optimizer = torch.optim.AdamW(trainable_params(model), lr=args.lr)
    x, target = make_batch(args, dtype)

    param_count = sum(param.numel() for param in model.parameters())
    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(
        "config:",
        {
            "mode": args.mode,
            "dtype": str(dtype),
            "compile": args.compile,
            "batch_size": args.batch_size,
            "tokens": args.tokens,
            "dim": args.dim,
            "hidden_dim": args.hidden_dim,
            "depth": args.depth,
            "trainable_params": trainable_count,
            "total_params": param_count,
            "simpletuner_workaround": args.simpletuner_workaround,
            "channels_last_input": args.channels_last_input,
            "quantize_lora": args.quantize_lora,
        },
    )

    stats: list[StepStats] = []
    profile_start = args.warmup_steps
    profile_end = profile_start + args.profile_steps
    profiler_cm = maybe_profile(args.profile, args.profile_steps)
    profiler = None

    for step in range(args.steps):
        if args.profile and step == profile_start:
            profiler = profiler_cm.__enter__()
        synchronize(args.device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = torch.nn.functional.mse_loss(out.float(), target.float())
        loss.backward()
        optimizer.step()
        synchronize(args.device)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        stats.append(StepStats(index=step, elapsed_ms=elapsed_ms, loss=float(loss.detach().item())))
        if profiler is not None:
            profiler.step()
        if args.profile and step + 1 == profile_end:
            profiler_cm.__exit__(None, None, None)
            profiler = None
        print(f"step={step:02d} elapsed_ms={elapsed_ms:.3f} loss={stats[-1].loss:.6f}")

    if profiler is not None:
        profiler_cm.__exit__(None, None, None)

    measured = [stat.elapsed_ms for stat in stats[args.warmup_steps :]]
    print(
        "summary:",
        {
            "mean_ms_after_warmup": round(statistics.mean(measured), 3),
            "median_ms_after_warmup": round(statistics.median(measured), 3),
            "min_ms_after_warmup": round(min(measured), 3),
            "max_ms_after_warmup": round(max(measured), 3),
            "final_loss": round(stats[-1].loss, 6),
        },
    )
    if args.profile and profiler_cm.profiler is not None:
        print(
            profiler_cm.profiler.key_averages().table(
                sort_by="cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total",
                row_limit=30,
            )
        )


if __name__ == "__main__":
    main()
