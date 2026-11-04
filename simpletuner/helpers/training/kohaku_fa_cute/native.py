"""Native autograd and TVM FFI dispatch for the CuTe attention kernels."""

import atexit
from functools import lru_cache
from hashlib import sha256
from itertools import count
from pathlib import Path

import triton
import tvm_ffi
from torch.utils.cpp_extension import load
from tvm_ffi import libinfo

from . import kernels

_names = count()


def compile_kernel(mode, tensors, causal, masked, configuration=None):
    if mode in ("short_d128_stats_forward", "short_d128_stats_backward"):
        if mode.endswith("backward"):
            from .short_key_stats_owned import compile_kernel as compile_short_d128
        else:
            from .short_reciprocal import compile_kernel as compile_short_d128

        function = compile_short_d128(mode.rsplit("_", 1)[1], tensors, causal)
    elif mode in ("short_d128_forward", "short_d128_backward"):
        from .short_reciprocal import compile_kernel as compile_short_d128

        function = compile_short_d128(mode.rsplit("_", 1)[1], tensors, causal)
    elif mode in ("short_key32_forward", "short_key32_backward"):
        from .short_key32_owned import compile_kernel as compile_short_key32

        function = compile_short_key32(mode.rsplit("_", 1)[1], tensors, causal)
    elif mode in ("short_forward", "short_backward", "shortopt_forward", "shortopt_backward"):
        from .short_owned import compile_kernel as compile_short

        narrow, prefetch = False, False
        if mode.startswith("shortopt_"):
            from .fa4_sm80 import _l40s, rtx4090

            policy = _l40s if _l40s.is_device(tensors[0].device) else rtx4090
            narrow, prefetch = policy._short_flags(tensors[0].dtype, causal)
        function = compile_short(mode.split("_", 1)[1], tensors, causal, narrow_product=narrow, prefetch_statistics=prefetch)
    elif mode in ("sm80_forward", "sm80_backward", "warp_forward", "warp_backward", "warp_inference"):
        from .fa4_sm80._native import compile_kernel as compile_sm80

        function = compile_sm80(mode.split("_", 1)[1], tensors, causal, configuration)
    elif mode in ("hopper_forward", "hopper_backward"):
        from .fa4_hopper._native import compile_kernel as compile_hopper

        function = compile_hopper(mode.split("_", 1)[1], tensors, causal, configuration)
    else:
        descriptors = tuple(kernels.descriptor(tensor) for tensor in tensors)
        function = kernels.compile_kernel(tensors[0].device.index, mode, causal, masked, descriptors)
    name = f"simpletuner.cute.native.{next(_names)}"
    tvm_ffi.register_global_func(name, function.__tvm_ffi_object__())
    return name


@lru_cache(maxsize=1)
def extension():
    library = Path(libinfo.find_libtvm_ffi()).parent
    source = Path(__file__).with_suffix(".cpp")
    # Checkouts and copied source trees can retain older modification times.
    # Give each C++ revision its own build cache rather than trusting mtimes.
    name = "simpletuner_cute_attention_native_" + sha256(source.read_bytes()).hexdigest()[:16]
    module = load(
        name=name,
        sources=[str(source)],
        extra_include_paths=[
            libinfo.find_include_path(),
            libinfo.find_dlpack_include_path(),
            str(Path(triton.__file__).parent / "backends/nvidia/include"),
        ],
        extra_cflags=["-O2"],
        extra_ldflags=[f"-L{library}", "-ltvm_ffi", "-lc10_cuda", f"-Wl,-rpath,{library}"],
    )
    atexit.register(module.clear_cache)
    return module
