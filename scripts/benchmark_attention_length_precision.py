"""Compare attention precision at aligned and unaligned sequence lengths."""

import argparse
import json
import math
import sys
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_attention_precision import centered_sdpa, evaluate, fp64_reference, make_inputs, metrics

LENGTHS = (256, 257, 261, 1000, 1024, 1029, 4096, 4100)
BACKENDS = ("auto", "math", "cudnn", "efficient", "flex", "kohaku-fa", "kohaku-fa-centered")


@lru_cache(maxsize=1)
def compiled_flex():
    from torch.nn.attention.flex_attention import flex_attention

    # Static shapes preserve the kernel choices this benchmark investigates.
    return torch.compile(flex_attention, fullgraph=True, dynamic=False)


def flex_sdpa(q, k, v, *, attn_mask=None, is_causal=False, enable_gqa=False):
    score_mod = None
    if attn_mask is not None or is_causal:
        mask = None if attn_mask is None else attn_mask.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])

        def score_mod(score, batch, head, query, key):
            visible = query >= key if is_causal else torch.ones_like(query, dtype=torch.bool)
            if mask is not None:
                visible = visible & mask[batch, head, query.clamp(max=q.shape[2] - 1), key.clamp(max=k.shape[2] - 1)]
            return torch.where(visible, score, -float("inf"))

    # The dtype/length/mask sweep exceeds Dynamo's default per-frame limit.
    # Restore the caller's configuration after each call.
    with torch._dynamo.config.patch(recompile_limit=128):
        return compiled_flex()(q, k, v, score_mod=score_mod, enable_gqa=enable_gqa)


def backend(name):
    if name == "kohaku-fa-centered":
        return centered_sdpa, nullcontext()
    if name == "flex":
        return flex_sdpa, nullcontext()
    if name == "kohaku-fa":
        from simpletuner.helpers.training.kohaku_fa_cute import scaled_dot_product_attention

        return scaled_dot_product_attention, nullcontext()
    kinds = {
        "math": torch.nn.attention.SDPBackend.MATH,
        "cudnn": torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
        "efficient": torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
    }
    context = torch.nn.attention.sdpa_kernel(kinds[name]) if name in kinds else nullcontext()
    return torch.nn.functional.scaled_dot_product_attention, context


def sweep(*, lengths, dtypes, modes, backends, device, dim=64, heads=2, logit_scale=1e5, seed=0):
    rows = []
    for dtype_name in dtypes:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
        for length in lengths:
            q, k, v, dout = make_inputs(logit_scale, device=device, dtype=dtype, seq=length, dim=dim, seed=seed, heads=heads)
            maximum = float((q.double() @ k.double().transpose(-1, -2)).abs().max() / math.sqrt(dim))
            for mode in modes:
                mask = torch.arange(length, device=device) < max(1, 3 * length // 4) if mode == "padding" else None
                causal = mode == "causal"
                reference = fp64_reference((q, k, v), dout, mask, causal)
                for name in backends:
                    row = dict(
                        backend=name,
                        dtype=dtype_name,
                        mode=mode,
                        sequence_length=length,
                        shape=list(q.shape),
                        max_absolute_logit=maximum,
                    )
                    try:
                        fn, context = backend(name)
                        with context:
                            got = evaluate(fn, (q, k, v), dout, mask, causal)
                        row["metrics"] = {n: metrics(g, r) for n, g, r in zip(("out", "dq", "dk", "dv"), got, reference)}
                        row["status"] = "ok"
                    except (RuntimeError, ValueError, ImportError) as error:
                        row.update(status="error", error=str(error))
                    rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seq", nargs="+", type=int, default=list(LENGTHS))
    parser.add_argument("--dtypes", nargs="+", choices=("fp16", "bf16"), default=["fp16", "bf16"])
    parser.add_argument("--modes", nargs="+", choices=("dense", "causal", "padding"), default=["dense"])
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=["flex", "cudnn", "kohaku-fa"])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--logit-scale", type=float, default=1e5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(*args.seq, args.dim, args.heads) < 1 or not math.isfinite(args.logit_scale) or args.logit_scale < 0:
        parser.error("sequence lengths, dim and heads must be positive; logit scale must be finite and nonnegative")
    device = torch.device(args.device)
    report = dict(
        torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda,
        cudnn_version=torch.backends.cudnn.version(),
        device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        seed=args.seed,
        requested_logit_scale=args.logit_scale,
        rows=sweep(
            lengths=args.seq,
            dtypes=args.dtypes,
            modes=args.modes,
            backends=args.backends,
            device=device,
            dim=args.dim,
            heads=args.heads,
            logit_scale=args.logit_scale,
            seed=args.seed,
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for row in report["rows"]:
        error = row["metrics"]["dq"]["relative_error"] if row["status"] == "ok" else row["error"]
        print(f"{row['dtype']} S={row['sequence_length']} {row['mode']} {row['backend']}: {error}")


if __name__ == "__main__":
    main()
