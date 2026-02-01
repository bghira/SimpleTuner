"""
RamTorch Extensions

Extends ramtorch with CPU-bouncing implementations for layer types beyond Linear:
- Embedding
- Conv1d, Conv2d, Conv3d
- LayerNorm, RMSNorm

These follow the same pattern as ramtorch's CPUBouncingLinear: weights stay in
CPU RAM and are streamed to GPU on-demand during forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Per-device state for async transfers (shared with ramtorch)
_DEVICE_STATE = {}


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
            self.weight.data = self.weight.data.to(dtype=result.dtype)
        return self

    def cuda(self, device=None):
        return self

    def cpu(self):
        return self

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        state = _get_device_state(self.device)
        transfer_stream = state["transfer_stream"]

        with torch.cuda.stream(transfer_stream):
            weight_gpu = self.weight.to(self.device, non_blocking=True)

        torch.cuda.current_stream().wait_stream(transfer_stream)

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
            self.weight.data = self.weight.data.to(dtype=result.dtype)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(dtype=result.dtype)
        return self

    def cuda(self, device=None):
        return self

    def cpu(self):
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = _get_device_state(self.device)
        transfer_stream = state["transfer_stream"]

        with torch.cuda.stream(transfer_stream):
            weight_gpu = self.weight.to(self.device, non_blocking=True)
            bias_gpu = self.bias.to(self.device, non_blocking=True) if self.bias is not None else None

        torch.cuda.current_stream().wait_stream(transfer_stream)

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
            self.weight.data = self.weight.data.to(dtype=result.dtype)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(dtype=result.dtype)
        return self

    def cuda(self, device=None):
        return self

    def cpu(self):
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = _get_device_state(self.device)
        transfer_stream = state["transfer_stream"]

        with torch.cuda.stream(transfer_stream):
            weight_gpu = self.weight.to(self.device, non_blocking=True)
            bias_gpu = self.bias.to(self.device, non_blocking=True) if self.bias is not None else None

        torch.cuda.current_stream().wait_stream(transfer_stream)

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
        dummy = torch.tensor(0.0, device="cpu", dtype=torch.float32)
        result = fn(dummy)
        if self.weight is not None and result.dtype != dummy.dtype:
            self.weight.data = self.weight.data.to(dtype=result.dtype)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(dtype=result.dtype)
        return self

    def cuda(self, device=None):
        return self

    def cpu(self):
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.elementwise_affine:
            return F.layer_norm(x, self.normalized_shape, None, None, self.eps)

        state = _get_device_state(self.device)
        transfer_stream = state["transfer_stream"]

        with torch.cuda.stream(transfer_stream):
            weight_gpu = self.weight.to(self.device, non_blocking=True)
            bias_gpu = self.bias.to(self.device, non_blocking=True) if self.bias is not None else None

        torch.cuda.current_stream().wait_stream(transfer_stream)

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
        dummy = torch.tensor(0.0, device="cpu", dtype=torch.float32)
        result = fn(dummy)
        if self.weight is not None and result.dtype != dummy.dtype:
            self.weight.data = self.weight.data.to(dtype=result.dtype)
            if self.bias is not None:
                self.bias.data = self.bias.data.to(dtype=result.dtype)
        return self

    def cuda(self, device=None):
        return self

    def cpu(self):
        return self

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

        state = _get_device_state(self.device)
        transfer_stream = state["transfer_stream"]

        with torch.cuda.stream(transfer_stream):
            weight_gpu = self.weight.to(self.device, non_blocking=True)
            bias_gpu = self.bias.to(self.device, non_blocking=True) if self.bias is not None else None

        torch.cuda.current_stream().wait_stream(transfer_stream)

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
            torch.cuda.synchronize()
            return out

        return hook

    for name, child in module.named_modules():
        # Add sync hooks after all ramtorch layers
        if getattr(child, "is_ramtorch", False) or "CPUBouncing" in child.__class__.__name__:
            h = child.register_forward_hook(make_sync_hook())
            hooks.append(h)

    return hooks


def remove_ramtorch_sync_hooks(hooks: list) -> None:
    """Remove synchronization hooks added by add_ramtorch_sync_hooks."""
    for h in hooks:
        h.remove()
