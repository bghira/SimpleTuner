import logging

import optimum
import torch
import torch.utils.cpp_extension
from optimum.quanto.tensor.packed import PackedTensor
from optimum.quanto.tensor.weights.qbits import WeightQBitsTensor
from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor

_quanto_workarounds_logger = logging.getLogger("simpletuner.quanto_workarounds")

# ============================================================================
# PyTorch cpp_extension fix for ROCm/HIP on systems where ROCM_HOME=/usr
# ============================================================================
# When ROCM_HOME is /usr (common on Gentoo and some distros), PyTorch adds
# -isystem /usr/include to the compile flags. This breaks GCC's #include_next
# mechanism used by C++ standard library headers like <cstdlib>.
#
# PyTorch already has this fix for CUDA (see cpp_extension.py line 1520-1521)
# but not for HIP/ROCm. This monkeypatch adds the same protection.

_original_include_paths = torch.utils.cpp_extension.include_paths


def _patched_include_paths(device_type: str = "cpu") -> list:
    """Patched include_paths that filters out /usr/include for HIP/ROCm."""
    paths = _original_include_paths(device_type)

    # Filter out /usr/include - it breaks #include_next in GCC's C++ headers
    # when passed as -isystem. The compiler already knows about /usr/include.
    filtered = [p for p in paths if p != "/usr/include"]

    if len(filtered) < len(paths):
        _quanto_workarounds_logger.debug("Filtered /usr/include from cpp_extension include_paths to fix GCC #include_next")

    return filtered


torch.utils.cpp_extension.include_paths = _patched_include_paths

# ============================================================================

_TORCH_TENSOR_DATA_DESCRIPTOR = torch.Tensor.data

# TinyGemmWeightQBitsTensor is only available on CUDA
_TinyGemmWeightQBitsTensor = None
_TinyGemmPackedTensor = None
if torch.cuda.is_available() and torch.version.cuda:
    try:
        from optimum.quanto.tensor.weights.tinygemm.packed import TinyGemmPackedTensor as _TinyGemmPackedTensor
        from optimum.quanto.tensor.weights.tinygemm.qbits import TinyGemmWeightQBitsTensor as _TinyGemmWeightQBitsTensor
    except ImportError:
        pass

if torch.cuda.is_available():
    # the marlin fp8 kernel needs some help with dtype casting for some reason
    # see: https://github.com/huggingface/optimum-quanto/pull/296#issuecomment-2380719201
    if torch.device("cuda").type == "cuda" and torch.version.cuda:
        from optimum.quanto.library.extensions.cuda import ext as quanto_ext

        # Save the original operator
        original_gemm_f16f8_marlin = torch.ops.quanto.gemm_f16f8_marlin

        def fp8_marlin_gemm_wrapper(
            a: torch.Tensor,
            b_q_weight: torch.Tensor,
            b_scales: torch.Tensor,
            workspace: torch.Tensor,
            num_bits: int,
            size_m: int,
            size_n: int,
            size_k: int,
        ) -> torch.Tensor:
            # Ensure 'a' has the correct dtype
            a = a.to(b_scales.dtype)
            # Call the original operator
            return original_gemm_f16f8_marlin(
                a,
                b_q_weight,
                b_scales,
                workspace,
                num_bits,
                size_m,
                size_n,
                size_k,
            )

        # Monkey-patch the operator
        torch.ops.quanto.gemm_f16f8_marlin = fp8_marlin_gemm_wrapper

    class TinyGemmQBitsLinearFunction(optimum.quanto.tensor.function.QuantizedLinearFunction):
        @staticmethod
        def forward(ctx, input, other, bias):
            ctx.save_for_backward(input, other)
            if type(input) is not torch.Tensor:
                input = input.dequantize()
            in_features = input.shape[-1]
            out_features = other.shape[0]
            output_shape = input.shape[:-1] + (out_features,)
            output = torch._weight_int4pack_mm(
                input.view(-1, in_features).to(dtype=other.dtype),
                other._data._data,
                other._group_size,
                other._scale_shift,
            )
            output = output.view(output_shape)
            if bias is not None:
                output = output + bias
            return output

    from optimum.quanto.tensor.weights import tinygemm

    tinygemm.qbits.TinyGemmQBitsLinearFunction = TinyGemmQBitsLinearFunction


class WeightQBytesLinearFunction(optimum.quanto.tensor.function.QuantizedLinearFunction):
    @staticmethod
    def forward(ctx, input, other, bias=None):
        ctx.save_for_backward(input, other)
        input_device = getattr(input, "device", None)
        if input_device is None and hasattr(input, "_data"):
            input_device = input._data.device

        if input_device is not None and hasattr(other, "_data"):
            backing_data = other._data
            backing_scale = getattr(other, "_scale", None)
            if backing_data.device != input_device:
                other._data = backing_data.to(input_device, non_blocking=True)
            if backing_scale is not None and hasattr(backing_scale, "device") and backing_scale.device != input_device:
                other._scale = backing_scale.to(input_device, non_blocking=True)

        if isinstance(input, optimum.quanto.tensor.QBytesTensor):
            output = torch.ops.quanto.qbytes_mm(input._data, other._data, input._scale * other._scale)
        else:
            in_features = input.shape[-1]
            out_features = other.shape[0]
            output_shape = input.shape[:-1] + (out_features,)
            output = torch.ops.quanto.qbytes_mm(input.reshape(-1, in_features), other._data, other._scale)
            output = output.view(output_shape)
        if bias is not None:
            # Move bias to output device if needed (ramtorch keeps bias on CPU)
            if bias.device != output.device:
                bias = bias.to(output.device, non_blocking=True)
            output = output + bias
        return output


optimum.quanto.tensor.weights.qbytes.WeightQBytesLinearFunction = WeightQBytesLinearFunction


# Save original forward
_original_qlf_forward = optimum.quanto.tensor.function.QuantizedLinearFunction.forward

# Track whether we've already fallen back to dequantize mode (to avoid repeated warnings)
_quanto_native_failed = False


def _move_qbits_tensor_to_device(tensor, device):
    """Move a WeightQBitsTensor's internal components to the specified device."""
    if not hasattr(tensor, "_data"):
        return tensor

    if tensor._data.device != device:
        tensor._data = tensor._data.to(device, non_blocking=True)
    if hasattr(tensor, "_scale") and tensor._scale is not None and tensor._scale.device != device:
        tensor._scale = tensor._scale.to(device, non_blocking=True)
    if hasattr(tensor, "_shift") and tensor._shift is not None and tensor._shift.device != device:
        tensor._shift = tensor._shift.to(device, non_blocking=True)
    if hasattr(tensor, "_scale_shift") and tensor._scale_shift is not None and tensor._scale_shift.device != device:
        tensor._scale_shift = tensor._scale_shift.to(device, non_blocking=True)

    return tensor


def _dequantize_fallback_forward(ctx, input, other, bias, input_device):
    """Fallback path: dequantize on CPU, move to GPU, standard matmul."""
    ctx.save_for_backward(input, other)
    if type(input) is not torch.Tensor:
        input = input.dequantize()
    weight_dequant = other.dequantize().to(input_device, non_blocking=True)
    in_features = input.shape[-1]
    out_features = other.shape[0]
    output_shape = input.shape[:-1] + (out_features,)
    output = torch.matmul(input.view(-1, in_features), weight_dequant.t())
    output = output.view(output_shape)
    if bias is not None:
        # Move bias to output device if needed (ramtorch keeps bias on CPU)
        if bias.device != output.device:
            bias = bias.to(output.device, non_blocking=True)
        output = output + bias
    return output


def _device_aware_qlf_forward(ctx, input, other, bias=None):
    """Patched QuantizedLinearFunction.forward that handles device mismatch.

    When ramtorch is used with quanto, weights may be on CPU while input is on GPU.
    First tries to move the quantized tensor's internals to GPU and use native quanto.
    If that fails (e.g., HIP/CUDA extension compilation failure), falls back to
    dequantizing on CPU and using standard matmul.
    """
    global _quanto_native_failed

    input_device = input.device if hasattr(input, "device") else None

    if input_device is not None and hasattr(other, "_data"):
        other_device = other._data.device if hasattr(other._data, "device") else None
        if other_device is not None and other_device != input_device:
            # Device mismatch detected - need to handle it

            if _quanto_native_failed:
                # Already know native path fails, go straight to fallback
                return _dequantize_fallback_forward(ctx, input, other, bias, input_device)

            # Try native path first: move quantized tensor to GPU
            try:
                _move_qbits_tensor_to_device(other, input_device)
                return _original_qlf_forward(ctx, input, other, bias)
            except Exception as e:
                # Native path failed - likely HIP/CUDA extension compilation error
                _quanto_native_failed = True
                _quanto_workarounds_logger.error(
                    "Quanto native extension failed (possibly HIP/CUDA compilation error). "
                    "Falling back to dequantize-and-matmul path. This may use more memory per layer."
                )
                _quanto_workarounds_logger.debug("Quanto native extension error details:", exc_info=True)

                # Move tensors back to CPU to avoid partial state
                try:
                    _move_qbits_tensor_to_device(other, other_device)
                except Exception:
                    pass

                return _dequantize_fallback_forward(ctx, input, other, bias, input_device)

    return _original_qlf_forward(ctx, input, other, bias)


optimum.quanto.tensor.function.QuantizedLinearFunction.forward = _device_aware_qlf_forward


def reshape_qlf_backward(ctx, gO):
    # another one where we need .reshape instead of .view
    input_gO = other_gO = bias_gO = None
    input, other = ctx.saved_tensors
    out_features, in_features = other.shape
    if ctx.needs_input_grad[0]:
        # grad(A@(B.t()) = gO => grad(A) = gO@(B.t().t()) = gO@B
        input_gO = torch.matmul(gO, other)
    if ctx.needs_input_grad[1]:
        # grad(B@A.t()) = gO.t() => grad(B) = gO.t()@(A.t().t()) = gO.t()@A
        other_gO = torch.matmul(
            gO.reshape(-1, out_features).t(),
            input.to(gO.dtype).reshape(-1, in_features),
        )
    if ctx.needs_input_grad[2]:
        # Bias gradient is the sum on all dimensions but the last one
        dim = tuple(range(gO.ndim - 1))
        bias_gO = gO.sum(dim)
    return input_gO, other_gO, bias_gO


optimum.quanto.tensor.function.QuantizedLinearFunction.backward = reshape_qlf_backward


def _bridge_storage_accessors(tensor_cls, data_attr: str) -> None:
    if getattr(tensor_cls, "_simpletuner_storage_bridge_applied", False):
        return

    def _backing_tensor(self):
        backing = getattr(self, data_attr, None)
        if backing is None:
            raise AttributeError(f"{tensor_cls.__name__} is missing expected backing tensor '{data_attr}'")
        return backing

    def _data_ptr(self):
        return _backing_tensor(self).data_ptr()

    def _untyped_storage(self):
        return _backing_tensor(self).untyped_storage()

    def _storage(self):
        return _backing_tensor(self).storage()

    tensor_cls.data_ptr = _data_ptr  # type: ignore[assignment]
    tensor_cls.untyped_storage = _untyped_storage  # type: ignore[assignment]
    tensor_cls.storage = _storage  # type: ignore[assignment]
    tensor_cls._simpletuner_storage_bridge_applied = True  # type: ignore[attr-defined]


_bridge_storage_accessors(WeightQBytesTensor, "_data")
_bridge_storage_accessors(WeightQBitsTensor, "_data")
_bridge_storage_accessors(PackedTensor, "_data")
if _TinyGemmPackedTensor is not None:
    _bridge_storage_accessors(_TinyGemmPackedTensor, "_data")


def _sync_tinygemm_internal_devices(tensor) -> None:
    """Ensure TinyGemmWeightQBitsTensor internal components are on the same device.

    When diffusers group_offloading moves internal tensors independently, they can
    end up on different devices. This breaks __tensor_unflatten__ which asserts
    that _data and _scale_shift are on the same device. We sync them to _data's device.
    """
    if _TinyGemmWeightQBitsTensor is None:
        return
    if not isinstance(tensor, _TinyGemmWeightQBitsTensor):
        return

    data_device = tensor._data.device
    scale_shift_device = tensor._scale_shift.device

    if data_device != scale_shift_device:
        tensor._scale_shift = tensor._scale_shift.to(data_device, non_blocking=True)


def _mirror_tensor_data_property(tensor_cls, attrs: tuple[str, ...]) -> None:
    if getattr(tensor_cls, "_simpletuner_data_bridge_applied", False):
        return

    def _data_get(self):
        _sync_tinygemm_internal_devices(self)
        return _TORCH_TENSOR_DATA_DESCRIPTOR.__get__(self, type(self))

    def _data_set(self, value):
        _TORCH_TENSOR_DATA_DESCRIPTOR.__set__(self, value)
        for attr in attrs:
            if hasattr(value, attr) and hasattr(self, attr):
                setattr(self, attr, getattr(value, attr))

    tensor_cls.data = property(_data_get, _data_set)  # type: ignore[assignment]
    tensor_cls._simpletuner_data_bridge_applied = True  # type: ignore[attr-defined]


_mirror_tensor_data_property(WeightQBytesTensor, ("_data", "_scale", "activation_qtype", "_axis", "_qtype"))
_mirror_tensor_data_property(WeightQBitsTensor, ("_data", "_scale", "_shift", "_axis", "_qtype"))
if _TinyGemmWeightQBitsTensor is not None:
    _mirror_tensor_data_property(_TinyGemmWeightQBitsTensor, ("_data", "_scale_shift", "_axis", "_qtype", "_group_size"))
if _TinyGemmPackedTensor is not None:
    _mirror_tensor_data_property(_TinyGemmPackedTensor, ("_data",))
