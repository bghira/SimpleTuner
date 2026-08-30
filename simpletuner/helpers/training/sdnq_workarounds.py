import logging

import torch

logger = logging.getLogger(__name__)

_PATCHED_FROM_FLOAT_ATTR = "_simpletuner_patched_from_float"
_PATCHED_GET_HADAMARD_ATTR = "_simpletuner_fake_safe_get_hadamard"
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


def apply_sdnq_workarounds() -> None:
    try:
        import sdnq.dequantizer as sdnq_dequantizer
        import sdnq.quant_utils as sdnq_quant_utils
        from sdnq.training.tensor import SDNQTensor
    except ImportError:
        return

    _install_triton_allocator()
    _patch_sdnq_scaled_mm_allocator_context()

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
