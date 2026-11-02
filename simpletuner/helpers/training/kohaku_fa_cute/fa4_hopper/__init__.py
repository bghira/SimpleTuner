"""Experimental full-H200 pipeline with explicit benchmark entry points."""

import math
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import torch


@lru_cache(maxsize=1)
def available():
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
    if (
        torch.compiler.is_compiling()
        or torch.are_deterministic_algorithms_enabled()
        or mask is not None
        or dropout != 0
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
        or q.shape[1] != 16
        or q.shape[2] != 4096
        or q.shape[3] not in (64, 128)
        or not all(tensor.is_contiguous() for tensor in (q, k, v))
        or torch.cuda.get_device_capability(q.device) != (9, 0)
    ):
        return False
    properties = torch.cuda.get_device_properties(q.device)
    if (
        properties.name not in ("NVIDIA H200", "NVIDIA H200 NVL")
        or properties.multi_processor_count != 132
        or properties.shared_memory_per_block_optin != 232448
    ):
        return False
    if scale is not None and (
        not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or not torch.finfo(torch.float32).tiny <= scale <= 2e38
    ):
        return False
    return available()


def attention(q, k, v, *, causal=False, scale=None):
    if scale is not None and (
        not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or not torch.finfo(torch.float32).tiny <= scale <= 2e38
    ):
        raise ValueError("The experimental Hopper pipeline requires a positive normal FP32 scale.")
    from ._attention import attention as run

    return run(q, k, v, causal=causal, scale=scale)
