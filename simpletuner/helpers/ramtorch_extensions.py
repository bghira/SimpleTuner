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

        return F.embedding(
            input_ids,
            weight_gpu,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse,
        )


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
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
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
        self.padding = padding if isinstance(padding, tuple) else (padding, padding, padding)
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
        dims = tuple(range(-len(self.normalized_shape), 0))
        variance = x.float().pow(2).mean(dim=dims, keepdim=True)
        output = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)

        if not self.elementwise_affine:
            return output

        state = _get_device_state(self.device)
        transfer_stream = state["transfer_stream"]

        with torch.cuda.stream(transfer_stream):
            weight_gpu = self.weight.to(self.device, non_blocking=True)
            bias_gpu = self.bias.to(self.device, non_blocking=True) if self.bias is not None else None

        torch.cuda.current_stream().wait_stream(transfer_stream)

        if self.use_weight_addition:
            output = output * (1.0 + weight_gpu.float())
        else:
            output = output * weight_gpu

        if bias_gpu is not None:
            output = output + bias_gpu

        return output.to(dtype=x.dtype)


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
