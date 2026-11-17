"""Optional qualified SM80 schedule; other calls retain native CuTe dispatch."""

import math
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch


@lru_cache(maxsize=1)
def available():
    """Require the measured runtime and exact optional source dependency."""
    if sys.version_info[:2] != (3, 13):
        return False
    try:
        version("flash-attn-4")
        for package, qualified in (
            ("nvidia-cutlass-dsl", "4.8.1"),
            ("apache-tvm-ffi", "0.1.15.post0"),
            ("quack-kernels", "0.6.5"),
            ("torch-c-dlpack-ext", "0.1.5"),
        ):
            if version(package) != qualified:
                return False
    except PackageNotFoundError:
        return False
    if torch.__version__.split("+")[0] != "2.15.0" or torch.version.cuda != "13.2":
        return False
    from ._private_api import interface_matches

    return interface_matches()


def compatible(q, k, v, mask, dropout, causal, scale, gqa):
    """Select only the full-device shape and arithmetic families qualified here."""
    from . import _l40s, _sm120, rtx4090

    if (
        isinstance(q, torch.Tensor)
        and getattr(q, "ndim", None) == 4
        and q.device.type == "cuda"
        and q.shape == (1, 8, 2048, 128)
        and causal
        and mask is None
        and dropout == 0
        and not torch.compiler.is_compiling()
        and not torch.are_deterministic_algorithms_enabled()
        and (
            scale is None
            or (isinstance(scale, (float, int)) and math.isfinite(scale) and 1.1754943508222875e-38 <= scale <= 2e38)
        )
        and available()
    ):
        from ..native import extension

        if extension().a100_medium_metadata_compatible(q, k, v):
            return all(torch.autograd.forward_ad.unpack_dual(t).tangent is None for t in (q, k, v))

    if (
        getattr(q, "ndim", None) == 4
        and q.device.type == "cuda"
        and not torch.compiler.is_compiling()
        and _l40s.is_device(q.device)
    ):
        return _l40s.compatible(q, k, v, mask, dropout, causal, scale, gqa)
    if (
        getattr(q, "ndim", None) == 4
        and q.device.type == "cuda"
        and not torch.compiler.is_compiling()
        and rtx4090._short_device(q.device.index)
    ):
        return rtx4090.automatic_compatible(q, k, v, mask, dropout, causal, scale, gqa)
    if _sm120.compatible(q, k, v, mask, dropout, causal, scale, gqa):
        return available()
    if (
        torch.compiler.is_compiling()
        or mask is not None
        or dropout != 0
        or torch.are_deterministic_algorithms_enabled()
        or getattr(q, "ndim", None) != 4
        or q.shape != k.shape
        or k.shape != v.shape
        or q.device.type != "cuda"
        or q.device != k.device
        or q.device != v.device
        or q.device.index != torch.cuda.current_device()
        or q.dtype not in (torch.float16, torch.bfloat16)
        or q.dtype != k.dtype
        or q.dtype != v.dtype
        or q.shape[0] != 1
        or not (
            (q.shape[-1] in (64, 128) and 8 <= q.shape[1] <= 16 and 2048 <= q.shape[2] <= 4096)
            or (
                q.shape[-1] == 256
                and ((q.shape[1] in (8, 16) and q.shape[2] == 4096) or (q.shape[1] == 16 and q.shape[2] == 1024))
            )
        )
        or not all(t.is_contiguous() for t in (q, k, v))
        or any(t.data_ptr() % 16 for t in (q, k, v))
        or torch.cuda.get_device_capability(q.device) != (8, 0)
        or torch.cuda.get_device_properties(q.device).multi_processor_count != 108
    ):
        return False
    # The current native runtime also qualifies this medium training family.
    # Other medium causal families retain their preceding selection.
    if causal and q.shape[-1] == 128 and q.shape[1] < 16 and q.shape[2] < 4096:
        if not (
            q.shape[1] == 8
            and q.shape[2] == 2048
            and torch.is_grad_enabled()
            and all(getattr(t, "requires_grad", False) for t in (q, k, v))
        ):
            return False
        if any(torch.autograd.forward_ad.unpack_dual(t).tangent is not None for t in (q, k, v)):
            return False
    if scale is not None and (
        not isinstance(scale, (float, int)) or not math.isfinite(scale) or not 1.1754943508222875e-38 <= scale <= 2e38
    ):
        return False
    return available()


def attention(q, k, v, *, causal=False, scale=None):
    from . import _l40s, _sm120, rtx4090
    from ._host import native_runner
    from ._schedules import a100_schedule

    if q.device.type == "cuda" and _l40s.is_device(q.device):
        return _l40s.attention(q, k, v, causal=causal, scale=scale)
    if q.device.type == "cuda" and rtx4090._short_device(q.device.index):
        return rtx4090.attention(q, k, v, causal=causal, scale=scale)
    if _sm120.target_device(q.device):
        dimension = q.shape[-1]
        if dimension == 256:
            return _sm120.d256_runner(causal, q.shape[2])(q, k, v, causal=causal, scale=scale)
        return native_runner(_sm120.schedule(dimension, causal))(q, k, v, causal=causal, scale=scale)
    runner = native_runner(a100_schedule(q.shape[-1], q.shape[2], causal, heads=q.shape[1]))
    return runner(q, k, v, causal=causal, scale=scale)
