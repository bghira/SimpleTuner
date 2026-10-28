"""Compare directed attention regressions with centered FP64 equations on identical inputs."""

import argparse
import dataclasses
import importlib.metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_attention_precision import fp64_reference, metrics


@dataclass(frozen=True)
class Case:
    name: str
    queries: int = 1
    keys: int = 128
    dim: int = 64
    shift: float = -128.0
    heads: int = 2
    kvheads: int = 2
    scale: float | None = None
    causal: bool = False
    mask: str = "none"
    layout: str = "sbhd"
    literal: bool = False


def cases(extended=True):
    result = [
        Case("issue160", queries=2, keys=2, dim=8, heads=1, kvheads=1, literal=True),
        Case("issue1072"),
        Case("zero_shift_control", shift=0),
        Case("key97_control", keys=97),
    ]
    result.extend(Case(f"kv_tail_{length}", queries=65, keys=length) for length in (63, 64, 65, 127, 129, 191, 192, 193))
    if not extended:
        return result
    result.extend(
        Case(f"negative_lse_square_{length}", queries=length, keys=length) for length in (64, 192, 1000, 1216, 1280)
    )
    result.extend(
        Case(f"single_query_{length}", keys=length) for length in (1, 63, 64, 65, 127, 129, 192, 193, 255, 256, 257)
    )
    result.extend(Case(f"shift_{abs(shift)}", queries=65, keys=192, shift=shift) for shift in (-64, -96, -512, -1024, 128))
    result.extend(Case(f"head_dim_{dim}", queries=33, keys=129, dim=dim) for dim in (80, 128, 256, 512))
    result.extend(
        [
            Case("negative_scale", queries=65, keys=193, scale=-0.125),
            Case("zero_scale", queries=65, keys=193, scale=0),
            Case("causal_tail", queries=65, keys=193, causal=True),
            Case("single_query_causal", causal=True),
            Case("gqa_tail", queries=65, keys=193, heads=4),
            Case("bhsd_tail", queries=65, keys=193, layout="bhsd"),
            Case("padding_mask", queries=65, keys=193, mask="padding"),
            Case("fully_masked_row", queries=65, keys=193, mask="empty_row"),
            Case("fully_masked_tile", queries=65, keys=193, mask="late_keys"),
        ]
    )
    return result


def make_case(case, *, device, dtype, seed=123):
    generator = torch.Generator(device=device).manual_seed(seed)

    def tensor(length, heads):
        shape = (length, 1, heads, case.dim) if case.layout == "sbhd" else (1, heads, length, case.dim)
        result = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        return result.permute(1, 2, 0, 3) if case.layout == "sbhd" else result

    q, k, v = tensor(case.queries, case.heads), tensor(case.keys, case.kvheads), tensor(case.keys, case.kvheads)
    dout = tensor(case.queries, case.heads) * 0.05
    if case.literal:
        # Rounded numeric fixture from NVIDIA/cudnn-frontend#160.
        values = (
            [
                [-1.7109, 0.7227, -1.5234, -2.1250, 2.4844, 0.4277, 2.4844, 0.3008],
                [-0.9023, 1.1094, -1.6094, -1.8359, -4.2500, 1.4531, 1.3281, 1.1172],
            ],
            [
                [-0.6367, 0.3945, -0.8867, -0.6250, 70.0000, 1.2109, 1.2969, -0.1934],
                [-0.7031, 1.2031, -0.7383, -0.7188, 70.5000, 1.9297, 0.7891, 1.2500],
            ],
            [
                [0.7422, 0.1875, -0.5664, 0.2383, -0.5195, 0.3789, 0.1260, -1.3047],
                [-0.7031, -0.5859, 0.3613, -0.2852, 0.1729, 0.4023, 1.0312, -1.0000],
            ],
            [
                [3.8126e-9, -1.3970e-8, 5.7335e-9, -1.0303e-8, -4.9477e-9, 1.3388e-8, -1.1292e-8, -1.4115e-9],
                [-1.3504e-8, -1.0594e-8, 1.2282e-8, 1.4144e-8, 5.0932e-9, 1.3853e-8, 1.4727e-8, 2.0606e-8],
            ],
        )
        for target, value in zip((q, k, v, dout), values):
            target.copy_(torch.tensor(value, device=device, dtype=dtype)[None, None])
    else:
        q[..., 0] = case.dim**0.5
        k[..., 0] = -case.shift if case.scale is not None and case.scale < 0 else case.shift
    mask = None
    if case.mask != "none":
        mask = torch.ones(case.queries, case.keys, dtype=torch.bool, device=device)
        if case.mask == "padding":
            mask[:, -1] = False
        elif case.mask == "empty_row":
            mask[0] = False
        elif case.mask == "late_keys":
            mask[:, :128] = False
        else:
            raise ValueError("Unknown mask fixture")
    return (q, k, v), dout, mask


@torch.no_grad()
def logits_statistics(inputs, mask=None, *, causal=False, scale=None):
    q, k, _ = inputs
    keys = k.double().repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    scores = q.double() @ keys.transpose(-1, -2)
    scores *= q.shape[-1] ** -0.5 if scale is None else scale
    if mask is not None:
        scores.masked_fill_(~mask, -torch.inf)
    if causal:
        visible = torch.arange(k.shape[2], device=q.device)[None] <= torch.arange(q.shape[2], device=q.device)[:, None]
        scores.masked_fill_(~visible, -torch.inf)
    maxima, lse = scores.amax(-1), scores.logsumexp(-1)
    valid = torch.isfinite(lse)
    return {
        "row_max_min": float(maxima[valid].min()),
        "row_max_max": float(maxima[valid].max()),
        "lse_min": float(lse[valid].min()),
        "lse_max": float(lse[valid].max()),
        "fully_masked_rows": int((~valid).sum()),
    }


def execute(backend, case, inputs, dout, mask):
    variables = [value.detach().clone().requires_grad_() for value in inputs]
    if backend == "kohakufa-cute":
        from simpletuner.helpers.training.kohaku_fa_cute import attention

        output = attention(*variables, mask=mask, causal=case.causal, scale=case.scale)
        return output, *torch.autograd.grad(output, variables, dout)
    if backend.startswith("frontend"):
        return frontend(case, inputs, dout, mask, padding=backend.endswith("padding"))
    from torch.nn.attention import SDPBackend, sdpa_kernel

    selected = {"cudnn": SDPBackend.CUDNN_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION, "math": SDPBackend.MATH}
    with sdpa_kernel(selected[backend]):
        output = torch.nn.functional.scaled_dot_product_attention(
            *variables,
            attn_mask=mask,
            is_causal=case.causal,
            scale=case.scale,
            enable_gqa=case.heads != case.kvheads,
        )
        return output, *torch.autograd.grad(output, variables, dout)


def frontend(case, inputs, dout, mask, *, padding=False):
    import cudnn

    if mask is not None:
        raise NotImplementedError("Direct frontend comparison uses unmasked fixtures only")
    q, k, v = inputs
    dtype = cudnn.data_type.BFLOAT16 if q.dtype == torch.bfloat16 else cudnn.data_type.HALF
    scale = case.dim**-0.5 if case.scale is None else case.scale

    def graph():
        return cudnn.pygraph(
            io_data_type=dtype, intermediate_data_type=cudnn.data_type.FLOAT, compute_data_type=cudnn.data_type.FLOAT
        )

    def options(g, bindings):
        if not padding:
            return {}
        values = [
            torch.full((1, 1, 1, 1), length, device=q.device, dtype=torch.int32) for length in (case.queries, case.keys)
        ]
        attrs = [g.tensor_like(value) for value in values]
        bindings.update(zip(attrs, values))
        return dict(use_padding_mask=True, seq_len_q=attrs[0], seq_len_kv=attrs[1])

    def mark(attr, tensor):
        attr.set_output(True).set_dim(tensor.shape).set_stride(tensor.stride())

    def run(g, bindings):
        g.validate()
        g.build_operation_graph()
        g.create_execution_plans([cudnn.heur_mode.A])
        g.check_support()
        g.build_plans()
        workspace = torch.empty(g.get_workspace_size(), device=q.device, dtype=torch.uint8)
        g.execute(bindings, workspace)
        torch.cuda.synchronize()

    forward = graph()
    attrs = [forward.tensor_like(value) for value in inputs]
    bindings = dict(zip(attrs, inputs))
    output_attr, stats_attr = forward.sdpa(
        q=attrs[0],
        k=attrs[1],
        v=attrs[2],
        generate_stats=True,
        attn_scale=scale,
        use_causal_mask=case.causal,
        **options(forward, bindings),
    )
    output = torch.empty_like(q)
    stats = torch.empty(1, case.heads, case.queries, 1, device=q.device, dtype=torch.float32)
    mark(output_attr, output)
    mark(stats_attr, stats)
    stats_attr.set_data_type(cudnn.data_type.FLOAT)
    bindings.update({output_attr: output, stats_attr: stats})
    run(forward, bindings)
    if not torch.isfinite(output).all() or not torch.isfinite(stats).all():
        raise FloatingPointError("Direct frontend saved output/statistics are nonfinite")
    backward = graph()
    saved = dict(q=q, k=k, v=v, o=output, dO=dout, stats=stats)
    attrs = {name: backward.tensor_like(value) for name, value in saved.items()}
    bindings = {attrs[name]: value for name, value in saved.items()}
    grads = backward.sdpa_backward(**attrs, attn_scale=scale, use_causal_mask=case.causal, **options(backward, bindings))
    gradients = [torch.empty_like(value) for value in inputs]
    for attr, value in zip(grads, gradients):
        mark(attr, value)
    bindings.update(zip(grads, gradients))
    run(backward, bindings)
    return output, *gradients


def assess(actual, expected, dtype):
    def acceptable(row, got):
        return row["finite"] and (
            row["relative_error"] < (0.05 if dtype == torch.bfloat16 else 0.015)
            if row["reference_norm"] > 0
            else int(got.count_nonzero()) == 0
        )

    result = {}
    passed = True
    for name, got, want in zip(("out", "dq", "dk", "dv"), actual, expected):
        row = metrics(got, want)
        row["nan_elements"] = int(torch.isnan(got).sum())
        row["inf_elements"] = int(torch.isinf(got).sum())
        # Norm-relative comparisons cannot hide tiny nonzero gradients under
        # a large absolute tolerance. Zero-reference cases require actual zero.
        row["passed"] = acceptable(row, got)
        row["edge_rows"] = {}
        for edge, region in (("first", slice(0, 1)), ("last", slice(-1, None))):
            value, reference = got[..., region, :], want[..., region, :]
            edge_result = metrics(value, reference)
            edge_result["passed"] = acceptable(edge_result, value)
            row["edge_rows"][edge] = edge_result
            row["passed"] &= edge_result["passed"]
        passed &= row["passed"]
        result[name] = row
    return {"status": "pass" if passed else "numerical_failure", "tensors": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("kohakufa-cute", "cudnn", "flash", "math", "frontend", "frontend-padding"),
        default=["kohakufa-cute", "cudnn", "flash"],
    )
    parser.add_argument("--case", action="append", help="Restrict to these named fixtures")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    selected = [case for case in cases(extended=not args.quick) if not args.case or case.name in args.case]
    if not selected:
        parser.error("No selected fixtures")
    report = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "torch_cudnn": torch.backends.cudnn.version(),
        "dependencies": {
            name: importlib.metadata.version(name) for name in ("nvidia-cutlass-dsl", "apache-tvm-ffi", "triton")
        },
        "reference": "Centered FP64 equations on identical rounded inputs; norm-relative gradient checks",
        "sources": [
            "https://github.com/NVIDIA/cudnn-frontend/issues/160",
            "https://github.com/NVIDIA/cudnn-frontend/issues/1072",
            "https://github.com/NVIDIA/cudnn-frontend/pull/628",
        ],
        "rows": [],
    }
    if any(name.startswith("frontend") for name in args.backends):
        import cudnn

        report["frontend"] = str(cudnn.__version__)
        report["frontend_backend"] = cudnn.backend_version()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for dtype in (torch.bfloat16, torch.float16):
        for case in selected:
            inputs, dout, mask = make_case(case, device="cuda", dtype=dtype)
            expected = fp64_reference(inputs, dout, mask, causal=case.causal, scale=case.scale)
            assert all(torch.isfinite(value).all() for value in (*inputs, dout, *expected))
            row = {
                "case": dataclasses.asdict(case),
                "dtype": str(dtype),
                "statistics": logits_statistics(inputs, mask, causal=case.causal, scale=case.scale),
                "backends": {},
            }
            for backend in args.backends:
                try:
                    actual = execute(backend, case, inputs, dout, mask)
                    result = assess(actual, expected, dtype)
                    del actual
                except FloatingPointError as exc:
                    result = {"status": "numerical_failure", "exception": type(exc).__name__}
                except (RuntimeError, NotImplementedError) as exc:
                    # A selected CuTe failure is never classified as fallback.
                    result = {
                        "status": "execution_error" if backend == "kohakufa-cute" else "unsupported_or_error",
                        "exception": type(exc).__name__,
                    }
                row["backends"][backend] = result
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(str(dtype), case.name, {name: value["status"] for name, value in row["backends"].items()}, flush=True)
    failures = sum(row["backends"].get("kohakufa-cute", {}).get("status", "pass") != "pass" for row in report["rows"])
    report["cute_failures"] = failures
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return bool(failures)


if __name__ == "__main__":
    sys.exit(main())
