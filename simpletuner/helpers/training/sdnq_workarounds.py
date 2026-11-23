import logging
import sys
from contextlib import contextmanager

import torch

logger = logging.getLogger(__name__)

_PATCHED_FROM_FLOAT_ATTR = "_simpletuner_patched_from_float"
_PATCHED_GET_HADAMARD_ATTR = "_simpletuner_fake_safe_get_hadamard"
_PATCHED_ROTATE_HADAMARD_ATTR = "_simpletuner_dynamic_rotate_hadamard"
_PATCHED_INT8_INPUTS_ATTR = "_simpletuner_dynamic_int8_inputs"
_PATCHED_SCALED_MM_ATTR = "_simpletuner_allocator_safe_scaled_mm"
_TRITON_ALLOCATOR_ATTR = "_simpletuner_sdnq_allocator"


def _detect_fake_mode(tensor) -> object | None:
    try:
        from torch._guards import detect_fake_mode

        return detect_fake_mode(tensor)
    except Exception:
        return None


def _install_triton_allocator() -> bool:
    try:
        import triton
        from triton.runtime import _allocation
    except ImportError:
        return False

    allocator_state = getattr(_allocation, "_allocator", None)
    null_allocator = getattr(_allocation, "NullAllocator", None)
    set_allocator = getattr(triton, "set_allocator", None)
    if allocator_state is None or null_allocator is None or not callable(set_allocator):
        return False
    try:
        current_allocator = allocator_state.get()
    except (AttributeError, LookupError):
        return False
    if not isinstance(current_allocator, null_allocator):
        return False

    def torch_cuda_allocator(size: int, alignment: int, stream: int | None):
        del alignment, stream
        return torch.empty(size, dtype=torch.uint8, device=torch.device("cuda", torch.cuda.current_device()))

    setattr(torch_cuda_allocator, _TRITON_ALLOCATOR_ATTR, True)
    set_allocator(torch_cuda_allocator)
    logger.debug("Registered the PyTorch CUDA caching allocator for SDNQ Triton scratch buffers.")
    return True


def capture_sdnq_allocator_context():
    allocation = sys.modules.get("triton.runtime._allocation")
    allocator_state = getattr(allocation, "_allocator", None)
    if allocator_state is None:
        return None
    allocator = allocator_state.get()
    if not getattr(allocator, _TRITON_ALLOCATOR_ATTR, False):
        return None

    @contextmanager
    def use_allocator():
        token = allocator_state.set(allocator)
        try:
            yield
        finally:
            allocator_state.reset(token)

    return use_allocator


def _patch_sdnq_scaled_mm_allocator_context() -> bool:
    try:
        from sdnq.kernels.triton_scaled_mm import sdnq_scaled_mm
    except ImportError:
        return False

    backend_fns = getattr(sdnq_scaled_mm, "_backend_fns", None)
    if not isinstance(backend_fns, dict):
        return False

    backend = backend_fns.get(None)
    if backend is None or getattr(backend, _PATCHED_SCALED_MM_ATTR, False):
        return False

    def allocator_safe_scaled_mm(*args, **kwargs):
        # Triton's allocator is stored in a ContextVar. Activation checkpoint
        # recomputation can run in a fresh context, so startup registration is
        # insufficient for SDNQ's tensor-descriptor scratch allocations.
        _install_triton_allocator()
        return backend(*args, **kwargs)

    setattr(allocator_safe_scaled_mm, _PATCHED_SCALED_MM_ATTR, True)
    backend_fns[None] = allocator_safe_scaled_mm
    logger.debug("Patched the SDNQ scaled-matmul backend to register its Triton allocator per execution context.")
    return True


def _patch_sdnq_int8_hadamard_rotation() -> None:
    from sdnq.common import compile_func, use_torch_compile

    if not use_torch_compile:
        return
    try:
        from sdnq.quant_utils import rotate_hadamard
        from sdnq.training.layers.linear.linear_int8 import linear_int8_dynamic

        original = linear_int8_dynamic.rotate_hadamard_compiled
    except (ImportError, AttributeError):
        return
    if getattr(original, _PATCHED_ROTATE_HADAMARD_ATTR, False):
        return

    def rotate_int8_backward(weight: torch.Tensor, *, hadamard: torch.Tensor) -> torch.Tensor:
        # Give this variable-shape rotation its own Dynamo frame instead of
        # sharing the frame used to compile SDNQ's context-decorated functions.
        return rotate_hadamard(weight, hadamard=hadamard)

    compiled_rotation = compile_func(rotate_int8_backward, fullgraph=True, dynamic=True)
    setattr(compiled_rotation, _PATCHED_ROTATE_HADAMARD_ATTR, True)
    linear_int8_dynamic.rotate_hadamard_compiled = compiled_rotation


def _patch_sdnq_int8_inputs() -> None:
    from sdnq.common import compile_func, use_torch_compile

    if not use_torch_compile:
        return
    try:
        from sdnq.training.layers.linear.linear_int8 import linear_int8, linear_int8_dynamic
    except ImportError:
        return

    def wrap_inputs(prepare, dynamic_weights):
        # Distinct code objects keep forward and backward out of the shared context-wrapper frame.
        if dynamic_weights:

            def prepare_int8_dynamic_inputs(*args, **kwargs):
                return prepare(*args, **kwargs)

            return prepare_int8_dynamic_inputs

        def prepare_int8_inputs(*args, **kwargs):
            return prepare(*args, **kwargs)

        return prepare_int8_inputs

    for module, name, dynamic_weights in (
        (linear_int8, "get_int8_matmul_inputs", False),
        (linear_int8_dynamic, "get_int8_matmul_dynamic_inputs", True),
    ):
        if getattr(module, "include_mm_kernel_in_compile", True):
            continue
        original = getattr(module, name, None)
        prepare = getattr(original, "_torchdynamo_orig_callable", None)
        if prepare is None or getattr(original, _PATCHED_INT8_INPUTS_ATTR, False):
            continue
        compiled = compile_func(wrap_inputs(prepare, dynamic_weights), fullgraph=True, dynamic=True)
        setattr(compiled, _PATCHED_INT8_INPUTS_ATTR, True)
        setattr(module, name, compiled)


def apply_sdnq_workarounds() -> None:
    try:
        import sdnq.dequantizer as sdnq_dequantizer
        import sdnq.quant_utils as sdnq_quant_utils
        from sdnq.training.tensor import SDNQTensor
    except ImportError:
        return

    _install_triton_allocator()
    _patch_sdnq_scaled_mm_allocator_context()
    _patch_sdnq_int8_hadamard_rotation()
    _patch_sdnq_int8_inputs()

    if not getattr(SDNQTensor, _PATCHED_FROM_FLOAT_ATTR, False):
        original_from_float = SDNQTensor.from_float

        def from_float_dequantizing_existing_sdnq(weight, *args, **kwargs):
            if isinstance(weight, SDNQTensor) and _detect_fake_mode(weight) is None:
                weight = weight.dequantize()
            return original_from_float(weight, *args, **kwargs)

        SDNQTensor.from_float = staticmethod(from_float_dequantizing_existing_sdnq)
        setattr(SDNQTensor, _PATCHED_FROM_FLOAT_ATTR, True)
        logger.debug("Patched SDNQTensor.from_float to dequantize existing SDNQTensor inputs before requantizing.")

    get_hadamard = sdnq_quant_utils.get_hadamard
    if not getattr(get_hadamard, _PATCHED_GET_HADAMARD_ATTR, False):

        def fake_safe_get_hadamard(n, dtype=None, device=None):
            if torch.compiler.is_compiling():
                # SDNQ compiles weight quantization separately. Dynamo already
                # executes tensor construction under FakeTensorMode there, and
                # returning the mode object through the FX graph is unsupported.
                return sdnq_quant_utils.build_hadamard(n, dtype=dtype, device=device)
            fake_mode = _detect_fake_mode(None)
            if fake_mode is None:
                return get_hadamard(n, dtype=dtype, device=device)
            # SDNQ's global cache may contain a real CUDA tensor from eager
            # quantization. Reusing it while Dynamo traces FakeTensors makes
            # the Hadamard matmul mix real and fake inputs.
            with fake_mode:
                return sdnq_quant_utils.build_hadamard(n, dtype=dtype, device=device)

        setattr(fake_safe_get_hadamard, _PATCHED_GET_HADAMARD_ATTR, True)
        sdnq_quant_utils.get_hadamard = fake_safe_get_hadamard
        sdnq_dequantizer.get_hadamard = fake_safe_get_hadamard
        logger.debug("Patched SDNQ Hadamard lookup to avoid real tensors during FakeTensor tracing.")
