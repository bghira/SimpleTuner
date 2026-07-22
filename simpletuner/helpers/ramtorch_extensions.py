"""
RamTorch Extensions

Extends ramtorch with CPU-bouncing implementations for layer types beyond Linear:
- Embedding
- Conv1d, Conv2d, Conv3d
- LayerNorm, RMSNorm

These follow the same pattern as ramtorch's CPUBouncingLinear: weights stay in
CPU RAM and are streamed to GPU on-demand during forward pass.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

from simpletuner.helpers.ramtorch import profiling as ramtorch_profile

# Per-device state for async transfers (shared with ramtorch)
_DEVICE_STATE = {}


def _to_cpu_pinned(tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if tensor.device.type == "cpu" and torch.cuda.is_available() and not tensor.is_pinned():
        tensor = tensor.pin_memory()
    return tensor


def _device_obj(device):
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int) and torch.cuda.is_available():
        return torch.device("cuda", device)
    return torch.device(device)


def _tensor_version(tensor: torch.Tensor | None):
    return tensor._version if tensor is not None else None


def _tensor_versions(tensors):
    return tuple(_tensor_version(tensor) for tensor in tensors)


def _prefetch_forward_tensors(module: nn.Module, *tensors: torch.Tensor | None) -> bool:
    device = _device_obj(module.device)
    if device.type != "cuda":
        return False

    versions = _tensor_versions(tensors)
    key = id(module)
    existing = getattr(module, "_ramtorch_forward_prefetch", None)
    if existing is not None and existing["versions"] == versions:
        ramtorch_profile.record_prefetch_existing(
            "extensions",
            key,
            device,
            ramtorch_profile.tensor_bytes(tensors),
        )
        return False

    allowed, bytes_to_prefetch, free_before, total_before = ramtorch_profile.should_prefetch(
        "extensions",
        key,
        device,
        tensors,
    )
    if not allowed:
        return False

    state = _get_device_state(device)
    transfer_stream = state["transfer_stream"]
    with torch.cuda.stream(transfer_stream):
        with record_function("forward_weight_bias_prefetch"):
            copied = tuple(tensor.to(device, non_blocking=True) if tensor is not None else None for tensor in tensors)
        event = torch.cuda.Event()
        event.record()

    module._ramtorch_forward_prefetch = {
        "event": event,
        "versions": versions,
        "tensors": copied,
    }
    ramtorch_profile.record_prefetch_enqueued(
        "extensions",
        key,
        device,
        bytes_to_prefetch,
        free_before=free_before,
        total_before=total_before,
    )
    return True


def _consume_forward_prefetch(module: nn.Module, *tensors: torch.Tensor | None):
    prefetched = getattr(module, "_ramtorch_forward_prefetch", None)
    if prefetched is None:
        return None

    delattr(module, "_ramtorch_forward_prefetch")
    device = _device_obj(module.device)
    key = id(module)
    if prefetched["versions"] != _tensor_versions(tensors):
        ramtorch_profile.record_prefetch_stale("extensions", key, device)
        return None

    with torch.cuda.device(device):
        torch.cuda.current_stream(device).wait_event(prefetched["event"])
    ramtorch_profile.record_prefetch_consumed("extensions", key, device)
    return prefetched["tensors"]


def _transfer_forward_tensors(module: nn.Module, *tensors: torch.Tensor | None):
    prefetched = _consume_forward_prefetch(module, *tensors)
    if prefetched is not None:
        return prefetched

    device = _device_obj(module.device)
    if device.type != "cuda":
        return tuple(tensor.to(device) if tensor is not None else None for tensor in tensors)

    ramtorch_profile.record_fallback_forward_transfer(device, tensors)
    state = _get_device_state(device)
    transfer_stream = state["transfer_stream"]

    with torch.cuda.stream(transfer_stream):
        with record_function("forward_weight_bias_transfer"):
            copied = tuple(tensor.to(device, non_blocking=True) if tensor is not None else None for tensor in tensors)

    with torch.cuda.device(device):
        torch.cuda.current_stream(device).wait_stream(transfer_stream)
    return copied


# thanks to Nerogar for fast stochastic pytorch implementation
# https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
def to_stochastic(
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    device: torch.device | str = None,
    non_blocking: bool = False,
) -> torch.Tensor:
    """
    Apply stochastic rounding for float32 → bfloat16 conversions.

    Stochastic rounding randomly rounds up or down based on the fractional
    part of the number, which eliminates systematic rounding bias and
    improves training stability especially for gradient updates.

    Args:
        tensor: Input tensor to convert
        target_dtype: Target dtype for conversion
        device: Target device (optional)
        non_blocking: Whether to use non-blocking transfer

    Returns:
        Tensor converted to target_dtype, using stochastic rounding if
        converting from float32 to bfloat16, otherwise standard rounding.
    """
    if tensor is None:
        return None

    # Only use stochastic rounding for float32 → bfloat16 downcasts
    if tensor.dtype == torch.float32 and target_dtype == torch.bfloat16:
        with torch.no_grad():
            # create a random 16 bit integer
            result = torch.randint_like(
                tensor,
                dtype=torch.int32,
                low=0,
                high=(1 << 16),
            )

            # add the random number to the lower 16 bit of the mantissa
            result.add_(tensor.view(dtype=torch.int32))

            # mask off the lower 16 bit of the mantissa
            result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

            # Convert the randomized float32 to bfloat16 (truncating the lower bits)
            return result.view(dtype=torch.float32).to(dtype=torch.bfloat16, device=device, non_blocking=non_blocking)

    # Standard deterministic rounding for all other cases
    return tensor.to(dtype=target_dtype, device=device, non_blocking=non_blocking)


def _get_device_state(device):
    """Get or initialize per-device state for async transfers."""
    if isinstance(device, str):
        device = torch.device(device)

    if device not in _DEVICE_STATE:
        with torch.cuda.device(device):
            _DEVICE_STATE[device] = {
                "transfer_stream": torch.cuda.Stream(device=device),
                "buffers": {},
                "clock": 0,
            }
    return _DEVICE_STATE[device]


class CPUBouncingEmbedding(nn.Module):
    """
    Embedding layer with CPU-stored weights that bounce to GPU on demand.

    Drop-in replacement for nn.Embedding but weights are kept in CPU RAM
    and transferred to GPU only during forward pass.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = None,
        max_norm: float = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        device="cuda",
        dtype=None,
        _weight=None,
        embed_scale: float = None,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.device = device

        if dtype is None:
            dtype = torch.float32

        self.is_ramtorch = True

        if _weight is not None:
            self.weight = nn.Parameter(_weight.to("cpu").pin_memory())
        else:
            self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="cpu").pin_memory())
            nn.init.normal_(self.weight)

        self.weight.is_ramtorch = True

        # Support for scaled word embeddings (e.g., Gemma3TextScaledWordEmbedding)
        if embed_scale is not None:
            self.register_buffer("embed_scale", torch.tensor(embed_scale), persistent=False)
        else:
            self.embed_scale = None

    def _apply(self, fn):
        """Override _apply to allow dtype changes but prevent device moves."""
        dummy = torch.tensor(0.0, device="cpu", dtype=self.weight.dtype)
        result = fn(dummy)
        if result.dtype != dummy.dtype:
            self.weight.data = _to_cpu_pinned(self.weight.data, dtype=result.dtype)
        return self

    def cuda(self, device=None):
        return self

    def cpu(self):
        return self

    def prefetch_forward(self) -> bool:
        return _prefetch_forward_tensors(self, self.weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        (weight_gpu,) = _transfer_forward_tensors(self, self.weight)

        # Apply stochastic rounding when autocast is enabled
        if torch.is_autocast_enabled():
            autocast_dtype = torch.get_autocast_gpu_dtype()
            weight_gpu = to_stochastic(weight_gpu, autocast_dtype, device=self.device)

        output = F.embedding(
            input_ids,
            weight_gpu,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )

        # Apply embed_scale if present (for scaled word embeddings like Gemma3)
        if self.embed_scale is not None:
            output = output * self.embed_scale.to(output.dtype)

        return output


class CPUBouncingConv2d(nn.Module):
    """
    Conv2d layer with CPU-stored weights that bounce to GPU on demand.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device="cuda",
        dtype=None,
        _weight=None,
        _bias=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        # padding can be int, tuple, or string ('valid', 'same')
        if isinstance(padding, str):
            self.padding = padding
        elif isinstance(padding, tuple):
            self.padding = padding
        else:
            self.padding = (padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.device = device

        if dtype is None:
            dtype = torch.float32

        self.is_ramtorch = True

        if _weight is not None:
            self.weight = nn.Parameter(_weight.to("cpu").pin_memory())
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    out_channels,
                    in_channels // groups,
                    *self.kernel_size,
                    dtype=dtype,
                    device="cpu",
                ).pin_memory()
            )
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        self.weight.is_ramtorch = True

        if bias:
            if _bias is not None:
                self.bias = nn.Parameter(_bias.to("cpu").pin_memory())
            else:
                self.bias = nn.Parameter(torch.zeros(out_channels, dtype=dtype, device="cpu").pin_memory())
            self.bias.is_ramtorch = True
        else:
            self.register_parameter("bias", None)

    def _apply(self, fn):
        dummy = torch.tensor(0.0, device="cpu", dtype=self.weight.dtype)
        result = fn(dummy)
        if result.dtype != dummy.dtype:
            self.weight.data = _to_cpu_pinned(self.weight.data, dtype=result.dtype)
            if self.bias is not None:
                self.bias.data = _to_cpu_pinned(self.bias.data, dtype=result.dtype)
        return self

    def cuda(self, device=None):
        return self

    def cpu(self):
        return self

    def prefetch_forward(self) -> bool:
        return _prefetch_forward_tensors(self, self.weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_gpu, bias_gpu = _transfer_forward_tensors(self, self.weight, self.bias)

        # Apply stochastic rounding when autocast is enabled
        if torch.is_autocast_enabled():
            autocast_dtype = torch.get_autocast_gpu_dtype()
            weight_gpu = to_stochastic(weight_gpu, autocast_dtype, device=self.device)
            if bias_gpu is not None:
                bias_gpu = to_stochastic(bias_gpu, autocast_dtype, device=self.device)

        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(x, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight_gpu,
                bias_gpu,
                self.stride,
                (0, 0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(x, weight_gpu, bias_gpu, self.stride, self.padding, self.dilation, self.groups)


class CPUBouncingConv3d(nn.Module):
    """
    Conv3d layer with CPU-stored weights that bounce to GPU on demand.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device="cuda",
        dtype=None,
        _weight=None,
        _bias=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        # padding can be int, tuple, or string ('valid', 'same')
        if isinstance(padding, str):
            self.padding = padding
        elif isinstance(padding, tuple):
            self.padding = padding
        else:
            self.padding = (padding, padding, padding)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation, dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        self.device = device

        if dtype is None:
            dtype = torch.float32

        self.is_ramtorch = True

        if _weight is not None:
            self.weight = nn.Parameter(_weight.to("cpu").pin_memory())
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    out_channels,
                    in_channels // groups,
                    *self.kernel_size,
                    dtype=dtype,
                    device="cpu",
                ).pin_memory()
            )
            nn.init.kaiming_uniform_(self.weight, a=5**0.5)

        self.weight.is_ramtorch = True

        if bias:
            if _bias is not None:
                self.bias = nn.Parameter(_bias.to("cpu").pin_memory())
            else:
                self.bias = nn.Parameter(torch.zeros(out_channels, dtype=dtype, device="cpu").pin_memory())
            self.bias.is_ramtorch = True
        else:
            self.register_parameter("bias", None)

    def _apply(self, fn):
        dummy = torch.tensor(0.0, device="cpu", dtype=self.weight.dtype)
        result = fn(dummy)
        if result.dtype != dummy.dtype:
            self.weight.data = _to_cpu_pinned(self.weight.data, dtype=result.dtype)
            if self.bias is not None:
                self.bias.data = _to_cpu_pinned(self.bias.data, dtype=result.dtype)
        return self

    def cuda(self, device=None):
        return self

    def cpu(self):
        return self

    def prefetch_forward(self) -> bool:
        return _prefetch_forward_tensors(self, self.weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_gpu, bias_gpu = _transfer_forward_tensors(self, self.weight, self.bias)

        # Apply stochastic rounding when autocast is enabled
        if torch.is_autocast_enabled():
            autocast_dtype = torch.get_autocast_gpu_dtype()
            weight_gpu = to_stochastic(weight_gpu, autocast_dtype, device=self.device)
            if bias_gpu is not None:
                bias_gpu = to_stochastic(bias_gpu, autocast_dtype, device=self.device)

        return F.conv3d(x, weight_gpu, bias_gpu, self.stride, self.padding, self.dilation, self.groups)


class CPUBouncingLayerNorm(nn.Module):
    """
    LayerNorm with CPU-stored weights that bounce to GPU on demand.
    """

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device="cuda",
        dtype=None,
        _weight=None,
        _bias=None,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.device = device

        if dtype is None:
            dtype = torch.float32

        self.is_ramtorch = True

        if elementwise_affine:
            if _weight is not None:
                self.weight = nn.Parameter(_weight.to("cpu").pin_memory())
            else:
                self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=dtype, device="cpu").pin_memory())
            self.weight.is_ramtorch = True

            if bias:
                if _bias is not None:
                    self.bias = nn.Parameter(_bias.to("cpu").pin_memory())
                else:
                    self.bias = nn.Parameter(torch.zeros(normalized_shape, dtype=dtype, device="cpu").pin_memory())
                self.bias.is_ramtorch = True
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def _apply(self, fn):
        dummy_dtype = self.weight.dtype if self.weight is not None else torch.float32
        dummy = torch.tensor(0.0, device="cpu", dtype=dummy_dtype)
        result = fn(dummy)
        if self.weight is not None and result.dtype != dummy.dtype:
            self.weight.data = _to_cpu_pinned(self.weight.data, dtype=result.dtype)
            if self.bias is not None:
                self.bias.data = _to_cpu_pinned(self.bias.data, dtype=result.dtype)
        return self

    def cuda(self, device=None):
        return self

    def cpu(self):
        return self

    def prefetch_forward(self) -> bool:
        if not self.elementwise_affine:
            return False
        return _prefetch_forward_tensors(self, self.weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.elementwise_affine:
            return F.layer_norm(x, self.normalized_shape, None, None, self.eps)

        weight_gpu, bias_gpu = _transfer_forward_tensors(self, self.weight, self.bias)

        # Apply stochastic rounding when autocast is enabled
        if torch.is_autocast_enabled():
            autocast_dtype = torch.get_autocast_gpu_dtype()
            weight_gpu = to_stochastic(weight_gpu, autocast_dtype, device=self.device)
            if bias_gpu is not None:
                bias_gpu = to_stochastic(bias_gpu, autocast_dtype, device=self.device)

        return F.layer_norm(x, self.normalized_shape, weight_gpu, bias_gpu, self.eps)


class CPUBouncingRMSNorm(nn.Module):
    """
    RMSNorm with CPU-stored weights that bounce to GPU on demand.
    """

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = False,
        device="cuda",
        dtype=None,
        _weight=None,
        _bias=None,
        use_weight_addition: bool = False,
    ):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.use_weight_addition = use_weight_addition
        self.device = device

        if dtype is None:
            dtype = torch.float32

        self.is_ramtorch = True

        if elementwise_affine:
            if _weight is not None:
                weight = _weight.to("cpu").pin_memory()
            else:
                init = torch.zeros if use_weight_addition else torch.ones
                weight = init(self.normalized_shape, dtype=dtype, device="cpu").pin_memory()
            self.weight = nn.Parameter(weight)
            self.weight.is_ramtorch = True

            if bias:
                if _bias is not None:
                    bias_tensor = _bias.to("cpu").pin_memory()
                else:
                    bias_tensor = torch.zeros(self.normalized_shape, dtype=dtype, device="cpu").pin_memory()
                self.bias = nn.Parameter(bias_tensor)
                self.bias.is_ramtorch = True
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def _apply(self, fn):
        dummy_dtype = self.weight.dtype if self.weight is not None else torch.float32
        dummy = torch.tensor(0.0, device="cpu", dtype=dummy_dtype)
        result = fn(dummy)
        if self.weight is not None and result.dtype != dummy.dtype:
            self.weight.data = _to_cpu_pinned(self.weight.data, dtype=result.dtype)
            if self.bias is not None:
                self.bias.data = _to_cpu_pinned(self.bias.data, dtype=result.dtype)
        return self

    def cuda(self, device=None):
        return self

    def cpu(self):
        return self

    def prefetch_forward(self) -> bool:
        if not self.elementwise_affine:
            return False
        return _prefetch_forward_tensors(self, self.weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Do all computations in float32 for precision (matches Gemma3RMSNorm behavior)
        # Only convert back to original dtype at the very end
        original_dtype = x.dtype
        x_float = x.float()
        dims = tuple(range(-len(self.normalized_shape), 0))
        variance = x_float.pow(2).mean(dim=dims, keepdim=True)
        output = x_float * torch.rsqrt(variance + self.eps)

        if not self.elementwise_affine:
            return output.to(dtype=original_dtype)

        weight_gpu, bias_gpu = _transfer_forward_tensors(self, self.weight, self.bias)

        # Apply stochastic rounding when autocast is enabled
        if torch.is_autocast_enabled():
            autocast_dtype = torch.get_autocast_gpu_dtype()
            weight_gpu = to_stochastic(weight_gpu, autocast_dtype, device=self.device)
            if bias_gpu is not None:
                bias_gpu = to_stochastic(bias_gpu, autocast_dtype, device=self.device)

        if self.use_weight_addition:
            output = output * (1.0 + weight_gpu.float())
        else:
            output = output * weight_gpu.float()

        if bias_gpu is not None:
            output = output + bias_gpu.float()

        return output.to(dtype=original_dtype)


def _is_rmsnorm_module(module: nn.Module) -> bool:
    rmsnorm_cls = getattr(nn, "RMSNorm", None)
    if rmsnorm_cls is not None and isinstance(module, rmsnorm_cls):
        return True
    name = module.__class__.__name__.lower()
    return name.endswith("rmsnorm") and "perchannel" not in name


def _rmsnorm_use_weight_addition(module: nn.Module) -> bool:
    for attr in ("unit_offset", "use_unit_offset", "use_weight_addition", "add_unit_offset"):
        if getattr(module, attr, False):
            return True
    return "gemma" in module.__class__.__name__.lower()


def replace_module_with_ramtorch(
    module: nn.Module,
    device: str = "cuda",
    include_embedding: bool = True,
    include_conv: bool = True,
    include_layernorm: bool = True,
    include_rmsnorm: bool = True,
) -> int:
    """
    Replace supported layer types with CPU-bouncing versions.

    Args:
        module: Root module to process
        device: Target device for computation
        include_embedding: Replace nn.Embedding layers
        include_conv: Replace nn.Conv2d and nn.Conv3d layers
        include_layernorm: Replace nn.LayerNorm layers
        include_rmsnorm: Replace RMSNorm-like layers

    Returns:
        Number of modules replaced
    """
    replaced = 0

    for name, child in list(module.named_children()):
        # Check for Embedding
        if include_embedding and isinstance(child, nn.Embedding) and not isinstance(child, CPUBouncingEmbedding):
            # Extract embed_scale from scaled word embeddings (e.g., Gemma3TextScaledWordEmbedding)
            embed_scale = None
            if hasattr(child, "embed_scale") and child.embed_scale is not None:
                embed_scale = child.embed_scale.item() if child.embed_scale.numel() == 1 else child.embed_scale.tolist()

            new_layer = CPUBouncingEmbedding(
                num_embeddings=child.num_embeddings,
                embedding_dim=child.embedding_dim,
                padding_idx=child.padding_idx,
                max_norm=child.max_norm,
                norm_type=child.norm_type,
                scale_grad_by_freq=child.scale_grad_by_freq,
                sparse=child.sparse,
                device=device,
                dtype=child.weight.dtype,
                _weight=child.weight.data,
                embed_scale=embed_scale,
            )
            setattr(module, name, new_layer)
            replaced += 1
            continue

        # Check for Conv2d
        if include_conv and isinstance(child, nn.Conv2d) and not isinstance(child, CPUBouncingConv2d):
            new_layer = CPUBouncingConv2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
                device=device,
                dtype=child.weight.dtype,
                _weight=child.weight.data,
                _bias=child.bias.data if child.bias is not None else None,
            )
            setattr(module, name, new_layer)
            replaced += 1
            continue

        # Check for Conv3d
        if include_conv and isinstance(child, nn.Conv3d) and not isinstance(child, CPUBouncingConv3d):
            new_layer = CPUBouncingConv3d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                groups=child.groups,
                bias=child.bias is not None,
                padding_mode=child.padding_mode,
                device=device,
                dtype=child.weight.dtype,
                _weight=child.weight.data,
                _bias=child.bias.data if child.bias is not None else None,
            )
            setattr(module, name, new_layer)
            replaced += 1
            continue

        # Check for RMSNorm
        if include_rmsnorm and _is_rmsnorm_module(child) and not isinstance(child, CPUBouncingRMSNorm):
            weight = getattr(child, "weight", None)
            bias = getattr(child, "bias", None) if hasattr(child, "bias") else None
            normalized_shape = getattr(child, "normalized_shape", None)
            if normalized_shape is None:
                if weight is not None:
                    normalized_shape = weight.shape
                else:
                    normalized_shape = getattr(child, "hidden_size", None) or getattr(child, "dim", None)
            if normalized_shape is not None:
                use_weight_addition = _rmsnorm_use_weight_addition(child)
                new_layer = CPUBouncingRMSNorm(
                    normalized_shape=normalized_shape,
                    eps=getattr(child, "eps", 1e-6),
                    elementwise_affine=getattr(child, "elementwise_affine", weight is not None),
                    bias=bias is not None,
                    device=device,
                    dtype=weight.dtype if weight is not None else None,
                    _weight=weight.data if weight is not None else None,
                    _bias=bias.data if bias is not None else None,
                    use_weight_addition=use_weight_addition,
                )
                setattr(module, name, new_layer)
                replaced += 1
                continue

        # Check for LayerNorm
        if include_layernorm and isinstance(child, nn.LayerNorm) and not isinstance(child, CPUBouncingLayerNorm):
            new_layer = CPUBouncingLayerNorm(
                normalized_shape=child.normalized_shape,
                eps=child.eps,
                elementwise_affine=child.elementwise_affine,
                bias=child.bias is not None,
                device=device,
                dtype=child.weight.dtype if child.weight is not None else None,
                _weight=child.weight.data if child.weight is not None else None,
                _bias=child.bias.data if child.bias is not None else None,
            )
            setattr(module, name, new_layer)
            replaced += 1
            continue

        # Recurse into children
        replaced += replace_module_with_ramtorch(
            child,
            device=device,
            include_embedding=include_embedding,
            include_conv=include_conv,
            include_layernorm=include_layernorm,
            include_rmsnorm=include_rmsnorm,
        )

    return replaced


def add_ramtorch_sync_hooks(module: nn.Module) -> list:
    """
    Add synchronization hooks after ramtorch layers to fix race conditions.

    Ramtorch uses ping-pong buffering with only 2 buffers, which can cause
    race conditions when more than 2 consecutive layers are processed.
    This adds CUDA synchronization after each ramtorch layer to ensure
    deterministic execution.

    Args:
        module: Root module to add hooks to

    Returns:
        List of hook handles (call .remove() on each to remove hooks)
    """
    hooks = []

    def make_sync_hook():
        def hook(mod, inp, out):
            ramtorch_profile.record_sync_hook()
            torch.cuda.synchronize()
            return out

        return hook

    for name, child in module.named_modules():
        # Add sync hooks after all ramtorch layers
        if getattr(child, "is_ramtorch", False) or "CPUBouncing" in child.__class__.__name__:
            h = child.register_forward_hook(make_sync_hook())
            hooks.append(h)

    return hooks


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() not in {"0", "false", "off", "no", "disabled"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


class _RamTorchPrefetchOrderRuntime:
    def __init__(self, component_key: str, module_by_id: dict[str, nn.Module], traversal_successors: dict[str, str]):
        from simpletuner.helpers.training.state_tracker import StateTracker

        self.component_key = component_key
        self.module_by_id = module_by_id
        self.traversal_successors = traversal_successors
        self.state_tracker = StateTracker
        self.learned_order_enabled = _env_bool("SIMPLETUNER_RAMTORCH_PREFETCH_LEARNED_ORDER", True)
        self.backward_preserve_enabled = _env_bool("SIMPLETUNER_RAMTORCH_PRESERVE_BACKWARD", True)
        self.backward_preserve_max_entries = _env_int("SIMPLETUNER_RAMTORCH_PRESERVE_BACKWARD_MAX_ENTRIES", 2)
        self.backward_preserve_max_bytes = _env_int("SIMPLETUNER_RAMTORCH_PRESERVE_BACKWARD_MAX_BYTES", 0)
        self.backward_preserve_min_free_ratio = _env_float(
            "SIMPLETUNER_RAMTORCH_PRESERVE_BACKWARD_MIN_FREE_RATIO",
            0.05,
        )
        self.backward_preserve_min_free_bytes = _env_int(
            "SIMPLETUNER_RAMTORCH_PRESERVE_BACKWARD_MIN_FREE_BYTES",
            0,
        )
        self.backward_preserve_min_total_bytes = _env_int(
            "SIMPLETUNER_RAMTORCH_PRESERVE_BACKWARD_MIN_TOTAL_BYTES",
            0,
        )
        self.min_observations = _env_int("SIMPLETUNER_RAMTORCH_PREFETCH_LEARNED_MIN_OBSERVATIONS", 3)
        self.min_confidence = _env_float("SIMPLETUNER_RAMTORCH_PREFETCH_LEARNED_MIN_CONFIDENCE", 0.80)
        self.active_depth = 0
        self.previous_module_id: str | None = None
        self.predicted_successors: dict[str, str] = {}
        if self.learned_order_enabled:
            self.state_tracker.ensure_ramtorch_prefetch_orders_loaded()
            self.state_tracker.configure_ramtorch_prefetch_component(
                self.component_key,
                list(self.module_by_id),
            )

    def begin_forward(self) -> None:
        self.active_depth += 1
        if self.active_depth == 1:
            self.previous_module_id = None
            self.predicted_successors.clear()

    def end_forward(self) -> None:
        if self.active_depth <= 0:
            return
        if self.active_depth == 1:
            for previous_id, predicted_id in list(self.predicted_successors.items()):
                ramtorch_profile.record_prefetch_prediction(predicted_id, None)
                if self._record_transition(previous_id, None):
                    ramtorch_profile.record_prefetch_successor_update()
            self.predicted_successors.clear()
            self.previous_module_id = None
        self.active_depth -= 1

    def module_entered(self, module_id: str) -> None:
        if self.active_depth <= 0:
            return
        previous_id = self.previous_module_id
        if previous_id is not None:
            predicted_id = self.predicted_successors.pop(previous_id, None)
            ramtorch_profile.record_prefetch_prediction(predicted_id, module_id)
            if self._record_transition(previous_id, module_id):
                ramtorch_profile.record_prefetch_successor_update()
        self.previous_module_id = module_id

    def choose_successor(self, module_id: str) -> tuple[str | None, nn.Module | None, str | None]:
        if self.learned_order_enabled and self.state_tracker.ramtorch_prefetch_disabled(
            self.component_key,
            module_id,
        ):
            return None, None, "disabled"

        if self.learned_order_enabled:
            learned_id = self.state_tracker.get_ramtorch_prefetch_successor(self.component_key, module_id)
            if learned_id in self.module_by_id:
                return learned_id, self.module_by_id[learned_id], "learned"

        traversal_id = self.traversal_successors.get(module_id)
        if traversal_id in self.module_by_id:
            return traversal_id, self.module_by_id[traversal_id], "traversal"
        return None, None, None

    def _successor_id_for_policy(self, module_id: str) -> str | None:
        if self.learned_order_enabled and self.state_tracker.ramtorch_prefetch_disabled(
            self.component_key,
            module_id,
        ):
            return None

        if self.learned_order_enabled:
            learned_id = self.state_tracker.get_ramtorch_prefetch_successor(self.component_key, module_id)
            if learned_id in self.module_by_id:
                return learned_id

        traversal_id = self.traversal_successors.get(module_id)
        return traversal_id if traversal_id in self.module_by_id else None

    def _module_bytes(self, module_id: str) -> int:
        module = self.module_by_id.get(module_id)
        if module is None:
            return 0
        byte_fn = getattr(module, "ramtorch_forward_bytes", None)
        if callable(byte_fn):
            try:
                return int(byte_fn())
            except Exception:
                return 0
        return 0

    def _module_device(self, module_id: str) -> torch.device | None:
        module = self.module_by_id.get(module_id)
        if module is None:
            return None

        raw_device = getattr(module, "device", None)
        if raw_device is not None:
            try:
                return _device_obj(raw_device)
            except Exception:
                return None

        try:
            parameter = next(module.parameters())
        except StopIteration:
            return None
        except Exception:
            return None
        return parameter.device

    def _preserve_memory_policy_allows(self, module_id: str, bytes_to_preserve: int) -> bool:
        min_ratio = max(float(self.backward_preserve_min_free_ratio), 0.0)
        min_bytes = max(int(self.backward_preserve_min_free_bytes), 0)
        min_total_bytes = max(int(self.backward_preserve_min_total_bytes), 0)
        if min_ratio == 0.0 and min_bytes == 0 and min_total_bytes == 0:
            return True

        device = self._module_device(module_id)
        if device is None or device.type != "cuda":
            return True
        if not torch.cuda.is_available():
            return False

        try:
            with torch.cuda.device(device):
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        except Exception:
            return False

        free_ratio = (float(free_bytes) / float(total_bytes)) if total_bytes else 0.0
        if int(total_bytes) < min_total_bytes or free_ratio < min_ratio or int(free_bytes) < min_bytes:
            ramtorch_profile.record_backward_preserve_skipped_policy(device, bytes_to_preserve)
            return False
        return True

    def should_preserve_for_backward(self, module_id: str) -> bool:
        if not self.backward_preserve_enabled:
            return False
        max_entries = max(int(self.backward_preserve_max_entries), 0)
        max_bytes = max(int(self.backward_preserve_max_bytes), 0)
        if max_entries == 0 and max_bytes == 0:
            return False

        remaining_entries = 0
        suffix_bytes = self._module_bytes(module_id)
        seen = {module_id}
        next_id = self._successor_id_for_policy(module_id)
        while next_id and next_id not in seen:
            seen.add(next_id)
            remaining_entries += 1
            suffix_bytes += self._module_bytes(next_id)
            if max_entries > 0 and remaining_entries >= max_entries:
                return False
            if max_bytes > 0 and suffix_bytes > max_bytes:
                return False
            next_id = self._successor_id_for_policy(next_id)
        bytes_to_preserve = self._module_bytes(module_id)
        return self._preserve_memory_policy_allows(module_id, bytes_to_preserve)

    def predicted(self, module_id: str, successor_id: str) -> None:
        self.predicted_successors[module_id] = successor_id

    def _record_transition(self, previous_id: str, actual_id: str | None) -> bool:
        if not self.learned_order_enabled:
            return False
        return self.state_tracker.record_ramtorch_prefetch_transition(
            self.component_key,
            previous_id,
            actual_id,
            min_observations=self.min_observations,
            min_confidence=self.min_confidence,
        )


def add_ramtorch_prefetch_hooks(module: nn.Module, component_label: str | None = None) -> list:
    """
    Prefetch the next RamTorch layer's forward weights.

    Traversal order is used as the initial guess. Runtime hook observations then
    learn actual module successors, including terminal modules that should stop
    prefetching at the end of a forward pass.

    Returns an empty list if any RamTorch module lacks ``prefetch_forward`` so
    callers can keep using the older synchronization path for compatibility.
    """
    if not ramtorch_profile.prefetch_hooks_allowed():
        return []

    ramtorch_modules: list[tuple[str, nn.Module]] = []
    missing_prefetch = 0

    for name, child in module.named_modules():
        is_ramtorch_module = getattr(child, "is_ramtorch", False) or "CPUBouncing" in child.__class__.__name__
        if not is_ramtorch_module:
            continue
        if callable(getattr(child, "prefetch_forward", None)):
            ramtorch_modules.append((name or "<root>", child))
        else:
            missing_prefetch += 1

    if missing_prefetch or len(ramtorch_modules) < 2:
        return []

    component_key = component_label or module.__class__.__qualname__
    module_by_id = {module_id: child for module_id, child in ramtorch_modules}
    traversal_successors = {
        current_id: next_id for (current_id, _current), (next_id, _next) in zip(ramtorch_modules, ramtorch_modules[1:])
    }
    runtime = _RamTorchPrefetchOrderRuntime(component_key, module_by_id, traversal_successors)

    hooks = []
    hooks.append(module.register_forward_pre_hook(lambda _mod, _inp: runtime.begin_forward()))
    hooks.append(module.register_forward_hook(lambda _mod, _inp, _out: runtime.end_forward()))

    for module_id, current in ramtorch_modules:

        def pre_hook(_mod, _inp, current_id=module_id):
            runtime.module_entered(current_id)
            return None

        def hook(_mod, _inp, _out, current_id=module_id, current_module=current):
            successor_id, target, source = runtime.choose_successor(current_id)
            if target is None or successor_id is None:
                ramtorch_profile.record_hook_prefetch_skipped_learned_order()
                preserve_fn = getattr(current_module, "preserve_forward_for_backward", None)
                if runtime.should_preserve_for_backward(current_id) and callable(preserve_fn):
                    preserve_fn(
                        max_entries=runtime.backward_preserve_max_entries,
                        max_bytes=runtime.backward_preserve_max_bytes,
                    )
                return None
            runtime.predicted(current_id, successor_id)
            ramtorch_profile.record_prefetch_successor_source(source or "traversal")
            success = target.prefetch_forward()
            ramtorch_profile.record_hook_prefetch(bool(success))
            preserve_fn = getattr(current_module, "preserve_forward_for_backward", None)
            if runtime.should_preserve_for_backward(current_id) and callable(preserve_fn):
                preserve_fn(
                    max_entries=runtime.backward_preserve_max_entries,
                    max_bytes=runtime.backward_preserve_max_bytes,
                )
            return None

        hooks.append(current.register_forward_pre_hook(pre_hook))
        hooks.append(current.register_forward_hook(hook))

    return hooks


def remove_ramtorch_sync_hooks(hooks: list) -> None:
    """Remove synchronization hooks added by add_ramtorch_sync_hooks."""
    for h in hooks:
        h.remove()


def get_ramtorch_target_device(model: nn.Module) -> torch.device | None:
    """Return the target GPU device from a model's ramtorch modules, or None.

    Returns the device of the first ramtorch module found.  All ramtorch
    modules within a model share the same target device because
    ``replace_linear_layers_with_ramtorch`` applies a single ``device``
    argument to every replaced layer.

    Returns ``None`` when the model contains no ramtorch modules.
    """
    for m in model.modules():
        if getattr(m, "is_ramtorch", False):
            dev = m.device
            return torch.device(dev) if isinstance(dev, str) else dev
    return None


_model_device_patched = False


def patch_model_device_for_ramtorch():
    """
    Patch ModelMixin.device so it returns the ramtorch target GPU device
    instead of CPU when ramtorch modules are present.

    This single patch fixes:
    - DiffusionPipeline._execution_device (delegates to pipeline.device -> model.device)
    - Direct self.transformer.device / self.unet.device references in pipeline code
    """
    global _model_device_patched
    if _model_device_patched:
        return
    _model_device_patched = True

    from diffusers import ModelMixin

    original_device = ModelMixin.device

    @property
    def device(self) -> torch.device:
        dev = get_ramtorch_target_device(self)
        if dev is not None:
            return dev
        return original_device.fget(self)

    ModelMixin.device = device
