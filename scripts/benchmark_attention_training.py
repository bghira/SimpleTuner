"""Time production transformer-block optimizer steps with rotating attention backends."""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase


class DiffusersBenchmarkBlock(torch.nn.Module):
    def __init__(self, family):
        super().__init__()
        self.family = family
        if family == "sdxl":
            from diffusers.models.attention import BasicTransformerBlock

            self.block = BasicTransformerBlock(1280, 20, 64, cross_attention_dim=2048)
        else:
            from diffusers.models.transformers.transformer_flux import FluxTransformerBlock

            self.block = FluxTransformerBlock(3072, 24, 128)

    def forward(self, x, context, *conditioning):
        if self.family == "sdxl":
            return self.block(x, encoder_hidden_states=context)
        context_output, image_output = self.block(x, context, *conditioning)
        return torch.cat((context_output, image_output), dim=1)


def make_case(family, sequence, lora, seed):
    torch.manual_seed(seed)
    hidden = {"ideogram": 4608, "minimax": 5376, "sdxl": 1280, "flux": 3072}[family]
    x = torch.randn(1, sequence, hidden, device="cuda", requires_grad=True)
    target = torch.randn_like(x)
    rows = torch.arange(sequence, device="cuda")
    positions = torch.stack((rows // 256, rows // 16 % 16, rows % 16), -1)[None]
    if family == "ideogram":
        from simpletuner.helpers.models.ideogram.transformer import Ideogram4MRoPE, Ideogram4TransformerBlock

        model = Ideogram4TransformerBlock(hidden, 12288, 18, 1e-5, 512).cuda().train()
        cos, sin = [tensor.bfloat16() for tensor in Ideogram4MRoPE(256, 5000000, (24, 20, 20)).cuda()(positions)]
        arguments = (rows[None] // 256, cos, sin, torch.randn(1, 1, 512, device="cuda"))
    elif family == "minimax":
        from simpletuner.helpers.models.minimaxh3.transformer import MiniMaxH3RotaryPosEmbed, MiniMaxH3TransformerBlock

        model = MiniMaxH3TransformerBlock(hidden, 56, 128, 14336, 2688, 1e-5, 1e-5).cuda().train()
        arguments = (
            torch.randn(1, 2688, device="cuda"),
            torch.zeros(1, sequence, device="cuda", dtype=torch.long),
            MiniMaxH3RotaryPosEmbed().cuda()(positions),
        )
    else:
        model = DiffusersBenchmarkBlock(family).cuda().train()
        if family == "sdxl":
            arguments = (torch.randn(1, 77, 2048, device="cuda"),)
        else:
            from diffusers.models.transformers.transformer_flux import FluxPosEmbed

            context = torch.randn(1, 512, hidden, device="cuda")
            ids = torch.cat((torch.zeros(512, 3, device="cuda"), positions[0]), dim=0)
            rotary = FluxPosEmbed(10000, [16, 56, 56]).cuda()(ids)
            arguments = (context, torch.randn(1, hidden, device="cuda"), rotary)
            target = torch.randn(1, sequence + 512, hidden, device="cuda")
    if lora:
        model = get_peft_model(model.bfloat16(), LoraConfig(r=16, lora_alpha=16, target_modules="all-linear"))
        x = x.detach().bfloat16().requires_grad_()
        target = target.bfloat16()
    return model, x, target, arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("ideogram", "minimax", "sdxl", "flux"), required=True)
    parser.add_argument("--seq", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--checkpoint", action="store_true")
    parser.add_argument("--compile", action="store_true", help="Compile the production block with fullgraph=True.")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("diffusers", "kohaku-fa", "kohaku-fa-auto", "cudnn", "native-flash", "native-efficient", "native-math"),
        default=("diffusers", "kohaku-fa"),
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument(
        "--learning-rate", type=float, default=0.0, help="Zero holds weights fixed for matched timing; AdamW still runs."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This benchmark requires CUDA.")
    if min(args.seq, args.warmup, args.steps, args.repeats) < 1:
        parser.error("Sequence length, warmup, steps and repeats must be positive.")
    model, x, target, arguments = make_case(args.family, args.seq, args.lora, args.seed)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, fused=True)
    if args.compile:
        model = torch.compile(model, fullgraph=True)

    def run():
        optimizer.zero_grad(set_to_none=True)
        x.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = checkpoint(model, x, *arguments, use_reentrant=False) if args.checkpoint else model(x, *arguments)
            loss = (output.float() - target).square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach()

    samples = []
    try:
        for repeat in range(args.repeats):
            offset = repeat % len(args.backends)
            order = args.backends[offset:] + args.backends[:offset]
            for backend in order:
                AttentionBackendController.apply(SimpleNamespace(attention_mechanism=backend), AttentionPhase.TRAIN)
                for _ in range(args.warmup):
                    initial = run()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                started = time.perf_counter()
                begin.record()
                for _ in range(args.steps):
                    final = run()
                end.record()
                end.synchronize()
                wall_ms = (time.perf_counter() - started) * 1000 / args.steps
                if not torch.isfinite(final) or any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
                    raise RuntimeError(f"Nonfinite loss or gradients with {backend}.")
                row = {
                    "backend": backend,
                    "repeat": repeat,
                    "wall_ms_per_step": wall_ms,
                    "cuda_ms_per_step": begin.elapsed_time(end) / args.steps,
                    "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "initial_loss": float(initial),
                    "final_loss": float(final),
                }
                samples.append(row)
                print(json.dumps(row), flush=True)
                AttentionBackendController.restore_default()
    finally:
        AttentionBackendController.restore_default()

    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "family": args.family,
        "shape": list(x.shape),
        "lora": args.lora,
        "checkpoint": args.checkpoint,
        "compiled": args.compile,
        "learning_rate": args.learning_rate,
        "weights": "random initialization; production block dimensions; same model and inputs reused across backends",
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "steps_per_sample": args.steps,
        "backend_order_policy": "Rotate backends by one position each repetition.",
        "samples": samples,
        "median_wall_ms_per_step": {
            backend: statistics.median(row["wall_ms_per_step"] for row in samples if row["backend"] == backend)
            for backend in args.backends
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
