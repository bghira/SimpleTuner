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


def compile_kernel(mode, tensors, causal, masked):
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
