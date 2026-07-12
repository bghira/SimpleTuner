from __future__ import annotations

import importlib
import inspect
import os
import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import torch

from simpletuner.helpers.logging import get_logger

try:
    from diffusers.models.attention_dispatch import AttentionBackendName, _check_attention_backend_requirements
    from diffusers.models.attention_dispatch import attention_backend as diffusers_attention_backend
except Exception:  # pragma: no cover - diffusers is a hard dependency but we guard for older installs
    AttentionBackendName = None
    diffusers_attention_backend = None
    _check_attention_backend_requirements = None

if TYPE_CHECKING:
    from sparse_linear_attention import SparseLinearAttention

logger = get_logger("AttentionBackend")


@dataclass(frozen=True)
class MetalFlashAttentionProfile:
    target_precision: Optional[int] = None
    quant_mode: int = 0


_METAL_FLASH_ATTENTION_PROFILES = {
    "metal-flash-attention": MetalFlashAttentionProfile(),
    "metal-sdpa": MetalFlashAttentionProfile(),
    "umfa": MetalFlashAttentionProfile(),
    "universal-metal-flash-attention": MetalFlashAttentionProfile(),
    "metal-flash-attention-int8": MetalFlashAttentionProfile(target_precision=3, quant_mode=2),
    "metal-flash-attention-int4": MetalFlashAttentionProfile(target_precision=4, quant_mode=2),
}

_METAL_FLASH_ATTENTION_ALIASES = set(_METAL_FLASH_ATTENTION_PROFILES)

_DIFFUSERS_BACKEND_TARGETS: Dict[str, str] = {
    "flash": "flash",
    "flash-attn": "flash",
    "flash_attn": "flash",
    "flash-hub": "flash_hub",
    "flash_attn_hub": "flash_hub",
    "flash-attn-hub": "flash_hub",
    "flash-attn-2": "flash_varlen",
    "flash_attn_2": "flash_varlen",
    "flash-attn-varlen": "flash_varlen",
    "flash_attn_varlen": "flash_varlen",
    "flash-varlen": "flash_varlen",
    "flash-varlen-hub": "flash_varlen_hub",
    "flash-attn-varlen-hub": "flash_varlen_hub",
    "flash_attn_varlen_hub": "flash_varlen_hub",
    "flash-attn-3": "_flash_3",
    "flash_attn_3": "_flash_3",
    "flash3": "_flash_3",
    "flash3-hub": "_flash_3_hub",
    "flash-attn-3-hub": "_flash_3_hub",
    "flash_attn_3_hub": "_flash_3_hub",
    "flash-attn-3-varlen": "_flash_varlen_3",
    "flash_attn_3_varlen": "_flash_varlen_3",
    "flash3-varlen": "_flash_varlen_3",
    "flash3-varlen-hub": "_flash_3_varlen_hub",
    "flash-attn-3-varlen-hub": "_flash_3_varlen_hub",
    "flash_attn_3_varlen_hub": "_flash_3_varlen_hub",
    "flash_attn3": "_flash_3",
    "flash_attn3_varlen": "_flash_varlen_3",
    "flash4-hub": "flash_4_hub",
    "flash-attn-4-hub": "flash_4_hub",
    "flash_attn_4_hub": "flash_4_hub",
    "flex": "flex",
    "flex-attn": "flex",
    "native": "native",
    "cudnn": "_native_cudnn",
    "native-cudnn": "_native_cudnn",
    "native-efficient": "_native_efficient",
    "native-flash": "_native_flash",
    "native-math": "_native_math",
    "native-npu": "_native_npu",
    "native-xla": "_native_xla",
}

_DIFFUSERS_BACKEND_ALIASES: Dict[str, AttentionBackendName] = {}
if AttentionBackendName is not None:
    for alias, target in _DIFFUSERS_BACKEND_TARGETS.items():
        try:
            _DIFFUSERS_BACKEND_ALIASES[alias] = AttentionBackendName(target)
        except Exception:
            continue


class AttentionBackendMode(str, Enum):
    TRAINING = "training"
    INFERENCE = "inference"
    TRAINING_AND_INFERENCE = "training+inference"

    @property
    def allows_training(self) -> bool:
        return self in (AttentionBackendMode.TRAINING, AttentionBackendMode.TRAINING_AND_INFERENCE)

    @property
    def allows_inference(self) -> bool:
        return self in (AttentionBackendMode.INFERENCE, AttentionBackendMode.TRAINING_AND_INFERENCE)

    @classmethod
    def from_raw(cls, raw_value: Any) -> "AttentionBackendMode":
        if isinstance(raw_value, cls):
            return raw_value
        if raw_value in (None, "", "None"):
            return cls.INFERENCE
        normalized = str(raw_value).strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError(
                f"Unsupported attention backend mode '{raw_value}'. Expected one of: "
                f"{', '.join(member.value for member in cls)}"
            ) from exc


class AttentionPhase(Enum):
    TRAIN = "train"
    EVAL = "eval"


@dataclass(frozen=True)
class PackedAttentionCapabilities:
    fixed_qkvpacked: bool
    varlen_qkvpacked: bool
    varlen_unpacked: bool


class PackedAttentionBackend:
    """
    Dispatch QKV-packed attention through direct FlashAttention or Hugging Face Kernels providers.

    The model processors own projection, normalization, and RoPE ordering. This class only owns kernel
    selection, bool-mask unpadding, and padding the visible-token output back to batch-major layout.
    """

    def __init__(self, name: str, module: Any):
        self.name = name
        self.module = module
        self.fixed_qkvpacked_func = getattr(module, "flash_attn_qkvpacked_func", None)
        self.varlen_qkvpacked_func = getattr(module, "flash_attn_varlen_qkvpacked_func", None)
        self.varlen_unpacked_func = getattr(module, "flash_attn_varlen_func", None)
        self.capabilities = PackedAttentionCapabilities(
            fixed_qkvpacked=self.fixed_qkvpacked_func is not None,
            varlen_qkvpacked=self.varlen_qkvpacked_func is not None,
            varlen_unpacked=self.varlen_unpacked_func is not None,
        )

    def qkvpacked(
        self,
        qkv: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        causal: bool = False,
        dropout_p: float = 0.0,
        softmax_scale: Optional[float] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            if self.fixed_qkvpacked_func is None:
                raise RuntimeError(f"Packed attention backend '{self.name}' does not provide fixed qkvpacked attention.")
            return _call_qkvpacked_kernel(
                self.fixed_qkvpacked_func,
                qkv,
                causal=causal,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
            )

        if self.varlen_qkvpacked_func is None and self.varlen_unpacked_func is None:
            raise RuntimeError(
                f"Packed attention backend '{self.name}' does not provide varlen attention for masked inputs."
            )

        qkv_unpad, indices, cu_seqlens, max_seqlen, batch_size, padded_seqlen = _unpad_qkv(qkv, attention_mask)
        if self.varlen_qkvpacked_func is not None:
            output_unpad = _call_varlen_qkvpacked_kernel(
                self.varlen_qkvpacked_func,
                qkv_unpad,
                cu_seqlens,
                max_seqlen,
                causal=causal,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
            )
        else:
            query, key, value = qkv_unpad.unbind(dim=1)
            output_unpad = _call_varlen_unpacked_kernel(
                self.varlen_unpacked_func,
                query.contiguous(),
                key.contiguous(),
                value.contiguous(),
                cu_seqlens,
                max_seqlen,
                causal=causal,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
            )
        return _pad_varlen_output(output_unpad, indices, batch_size, padded_seqlen)


_PACKED_BACKEND_ALIASES = {
    "flash-attn-2": ("direct-fa2", "flash_attn"),
    "flash_attn_2": ("direct-fa2", "flash_attn"),
    "flash2": ("direct-fa2", "flash_attn"),
    "flash-attn-2-hub": ("hub-fa2", "kernels-community/flash-attn2"),
    "flash_attn_2_hub": ("hub-fa2", "kernels-community/flash-attn2"),
    "flash2-hub": ("hub-fa2", "kernels-community/flash-attn2"),
    "flash-attn-varlen-hub": ("hub-fa2", "kernels-community/flash-attn2"),
    "flash_attn_varlen_hub": ("hub-fa2", "kernels-community/flash-attn2"),
    "flash-varlen-hub": ("hub-fa2", "kernels-community/flash-attn2"),
    "flash-attn-3": ("direct-fa3", "flash_attn_interface"),
    "flash_attn_3": ("direct-fa3", "flash_attn_interface"),
    "flash3": ("direct-fa3", "flash_attn_interface"),
    "flash-attn-3-hub": ("hub-fa3", "kernels-community/flash-attn3"),
    "flash_attn_3_hub": ("hub-fa3", "kernels-community/flash-attn3"),
    "flash3-hub": ("hub-fa3", "kernels-community/flash-attn3"),
    "flash-attn-3-varlen": ("direct-fa3", "flash_attn_interface"),
    "flash_attn_3_varlen": ("direct-fa3", "flash_attn_interface"),
    "flash3-varlen": ("direct-fa3", "flash_attn_interface"),
    "flash-attn-3-varlen-hub": ("hub-fa3", "kernels-community/flash-attn3"),
    "flash_attn_3_varlen_hub": ("hub-fa3", "kernels-community/flash-attn3"),
    "flash3-varlen-hub": ("hub-fa3", "kernels-community/flash-attn3"),
    "flash-attn-4": ("direct-fa4", "flash_attn.cute"),
    "flash_attn_4": ("direct-fa4", "flash_attn.cute"),
    "flash4": ("direct-fa4", "flash_attn.cute"),
    "flash-attn-4-hub": ("hub-fa4", "kernels-community/flash-attn4"),
    "flash_attn_4_hub": ("hub-fa4", "kernels-community/flash-attn4"),
    "flash4-hub": ("hub-fa4", "kernels-community/flash-attn4"),
}


def _normalize_backend_key(value: str) -> str:
    return value.replace("_", "-")


@lru_cache(maxsize=32)
def get_packed_attention_backend(
    preferred_backend: Optional[str] = None,
    require_varlen_qkvpacked: bool = False,
) -> PackedAttentionBackend:
    backend_key = _select_packed_backend(preferred_backend, require_varlen_qkvpacked=require_varlen_qkvpacked)
    provider, target = _PACKED_BACKEND_ALIASES[backend_key]
    if provider.startswith("hub-"):
        try:
            from kernels import get_kernel
        except ImportError as exc:
            raise RuntimeError("The 'kernels' package is required for Hugging Face Hub attention kernels.") from exc

        module = get_kernel(target)
        return PackedAttentionBackend(backend_key, module)

    import importlib

    try:
        module = importlib.import_module(target)
    except ImportError as exc:
        raise RuntimeError(f"Could not import packed attention backend '{target}'.") from exc
    return PackedAttentionBackend(backend_key, module)


def _select_packed_backend(preferred_backend: Optional[str], *, require_varlen_qkvpacked: bool = False) -> str:
    if preferred_backend:
        normalized = _normalize_backend_key(preferred_backend.strip().lower())
        raw = preferred_backend.strip().lower()
        if raw in _PACKED_BACKEND_ALIASES:
            return raw
        if normalized in _PACKED_BACKEND_ALIASES:
            return normalized
        if normalized in ("auto", "automatic"):
            return _auto_packed_backend(require_varlen_qkvpacked=require_varlen_qkvpacked)
        raise ValueError(f"Unsupported packed attention backend '{preferred_backend}'.")
    return _auto_packed_backend(require_varlen_qkvpacked=require_varlen_qkvpacked)


def _auto_packed_backend(*, require_varlen_qkvpacked: bool = False) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("Packed FlashAttention backends require CUDA.")
    major, minor = torch.cuda.get_device_capability()
    if require_varlen_qkvpacked and major >= 8:
        return "flash2-hub"
    if major >= 10:
        return "flash4-hub"
    if major >= 9:
        return "flash3-hub"
    if major == 8:
        return "flash2-hub"
    raise RuntimeError(f"No packed FlashAttention backend is configured for CUDA capability {major}.{minor}.")


def _normalize_bool_mask(attention_mask: torch.Tensor, batch_size: int, seqlen: int) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(
            f"Packed varlen attention expects a [batch, sequence] bool-compatible mask, got {tuple(attention_mask.shape)}."
        )
    if attention_mask.shape != (batch_size, seqlen):
        raise ValueError(
            f"Packed varlen attention mask shape {tuple(attention_mask.shape)} does not match qkv shape "
            f"{(batch_size, seqlen)}."
        )
    return attention_mask.to(dtype=torch.bool)


def _unpad_qkv(qkv: torch.Tensor, attention_mask: torch.Tensor):
    if qkv.ndim != 5 or qkv.shape[2] != 3:
        raise ValueError(f"Expected qkv shape [batch, sequence, 3, heads, head_dim], got {tuple(qkv.shape)}.")
    batch_size, seqlen = qkv.shape[:2]
    mask = _normalize_bool_mask(attention_mask, batch_size, seqlen).to(device=qkv.device)
    lengths = mask.sum(dim=1, dtype=torch.int32)
    if torch.any(lengths == 0):
        raise ValueError("Packed varlen attention received an empty sequence in the attention mask.")

    indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    qkv_unpad = qkv.reshape(batch_size * seqlen, *qkv.shape[2:]).index_select(0, indices)
    cu_seqlens = torch.zeros(batch_size + 1, device=qkv.device, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
    max_seqlen = int(lengths.max().item())
    return qkv_unpad.contiguous(), indices, cu_seqlens, max_seqlen, batch_size, seqlen


def _pad_varlen_output(output_unpad: torch.Tensor, indices: torch.Tensor, batch_size: int, seqlen: int) -> torch.Tensor:
    output = output_unpad.new_zeros((batch_size * seqlen, *output_unpad.shape[1:]))
    output.index_copy_(0, indices, output_unpad)
    return output.view(batch_size, seqlen, *output_unpad.shape[1:])


def _call_qkvpacked_kernel(func, qkv: torch.Tensor, *, causal: bool, dropout_p: float, softmax_scale: Optional[float]):
    try:
        output = func(qkv, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal)
    except TypeError:
        output = func(qkv, causal=causal)
    if isinstance(output, tuple):
        return output[0]
    return output


def _call_varlen_qkvpacked_kernel(
    func,
    qkv_unpad: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    causal: bool,
    dropout_p: float,
    softmax_scale: Optional[float],
):
    try:
        output = func(
            qkv_unpad,
            cu_seqlens,
            max_seqlen,
            dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    except TypeError:
        output = func(
            qkv_unpad,
            cu_seqlens,
            max_seqlen,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
        )
    if isinstance(output, tuple):
        return output[0]
    return output


def _call_varlen_unpacked_kernel(
    func,
    query_unpad: torch.Tensor,
    key_unpad: torch.Tensor,
    value_unpad: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    *,
    causal: bool,
    dropout_p: float,
    softmax_scale: Optional[float],
):
    kwargs = {
        "softmax_scale": softmax_scale,
        "causal": causal,
    }
    if "dropout_p" in inspect.signature(func).parameters:
        kwargs["dropout_p"] = dropout_p
    output = func(
        query_unpad,
        key_unpad,
        value_unpad,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        **kwargs,
    )
    if isinstance(output, tuple):
        return output[0]
    return output


def is_sageattention_available() -> bool:
    """Check if sageattention package is importable."""
    try:
        import sageattention  # noqa: F401

        return True
    except ImportError:
        return False


@lru_cache(maxsize=8)
def get_metal_flash_attention_unavailable_reason(backend: str = "metal-flash-attention") -> Optional[str]:
    """Return why the UMFA PyTorch custom-op backend cannot be used, or None when usable."""
    backend_key = _normalize_backend_key(str(backend or "metal-flash-attention").strip().lower())
    profile = _METAL_FLASH_ATTENTION_PROFILES.get(backend_key)
    if profile is None:
        return f"Unsupported Metal Flash Attention backend '{backend}'."
    try:
        package = importlib.import_module("pytorch_custom_op_ffi")
    except ImportError:
        return "Could not import the UMFA PyTorch custom-op package."

    availability_check = getattr(package, "is_metal_sdpa_available", None)
    if not callable(availability_check):
        return "The installed UMFA PyTorch package does not expose is_metal_sdpa_available()."

    try:
        if not availability_check():
            return "UMFA Metal SDPA is installed but not available on this host."
    except Exception as exc:
        return f"Failed to query UMFA Metal SDPA availability: {exc}"

    try:
        extension = importlib.import_module("metal_sdpa_extension")
    except ImportError:
        return "Could not import metal_sdpa_extension from the UMFA PyTorch custom-op package."

    if profile.target_precision is None:
        if not callable(getattr(extension, "metal_flash_attention_autograd", None)):
            return "metal_sdpa_extension does not expose metal_flash_attention_autograd()."
        return _metal_flash_attention_runtime_error()

    if not callable(getattr(extension, "metal_quantized_flash_attention_autograd", None)):
        return "metal_sdpa_extension does not expose metal_quantized_flash_attention_autograd()."

    return _metal_quantized_flash_attention_runtime_error(profile.target_precision, profile.quant_mode)


def is_metal_flash_attention_available(backend: str = "metal-flash-attention") -> bool:
    """Check if the UMFA PyTorch custom-op backend can run correctly on this host."""
    return get_metal_flash_attention_unavailable_reason(backend) is None


@lru_cache(maxsize=1)
def _metal_flash_attention_runtime_error() -> Optional[str]:
    script = """
import math
import torch
import metal_sdpa_extension

if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
    raise SystemExit("MPS is not available")


def reference_attention_cpu(query, key, value, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, value)


def reference_attention(query, key, value, scale=None):
    query = query.detach().cpu().float()
    key = key.detach().cpu().float()
    value = value.detach().cpu().float()
    return reference_attention_cpu(query, key, value, scale)


def reference_attention_grads(query, key, value):
    ref_query = query.detach().cpu().float().requires_grad_(True)
    ref_key = key.detach().cpu().float().requires_grad_(True)
    ref_value = value.detach().cpu().float().requires_grad_(True)
    ref_output = reference_attention_cpu(ref_query, ref_key, ref_value)
    ref_loss = ref_output.square().mean()
    ref_loss.backward()
    return ref_query.grad, ref_key.grad, ref_value.grad


def check(shape, rtol, atol, transposed=False, queued_producer=False):
    torch.manual_seed(42)
    if transposed:
        batch, heads, seq, dim = shape
        query = torch.randn((batch, seq, heads, dim), dtype=torch.float32, device="mps").transpose(1, 2)
        key = torch.randn((batch, seq, heads, dim), dtype=torch.float32, device="mps").transpose(1, 2)
        value = torch.randn((batch, seq, heads, dim), dtype=torch.float32, device="mps").transpose(1, 2)
    else:
        query = torch.randn(shape, dtype=torch.float32, device="mps")
        key = torch.randn(shape, dtype=torch.float32, device="mps")
        value = torch.randn(shape, dtype=torch.float32, device="mps")

    if queued_producer:
        query = query.mul(1.0)
        key = key.mul(1.0)
        value = value.mul(1.0)
    else:
        torch.mps.synchronize()
        expected = reference_attention(query, key, value)

    observed = metal_sdpa_extension.metal_flash_attention_autograd(
        query,
        key,
        value,
        False,
        0.0,
    )
    torch.mps.synchronize()
    if queued_producer:
        expected = reference_attention(query, key, value)

    if observed.shape != query.shape:
        raise SystemExit(f"Unexpected output shape for {shape}: {tuple(observed.shape)}")
    if observed.dtype != expected.dtype:
        raise SystemExit(f"Unexpected output dtype for {shape}: {observed.dtype}")
    if not torch.isfinite(observed).all().item():
        raise SystemExit(f"UMFA Metal SDPA produced non-finite values for {shape}")

    expected_cpu = expected.float()
    observed_cpu = observed.float().cpu()
    if not torch.allclose(observed_cpu, expected_cpu, rtol=rtol, atol=atol):
        diff = (observed_cpu - expected_cpu).abs()
        raise SystemExit(
            "UMFA Metal SDPA parity failed for "
            f"shape={shape} dtype=torch.float32: "
            f"max_abs={diff.max().item():.6g}, mean_abs={diff.mean().item():.6g}"
        )


def check_autograd(shape, rtol, atol):
    torch.manual_seed(43)
    query = torch.randn(shape, dtype=torch.float32, device="mps", requires_grad=True)
    key = torch.randn(shape, dtype=torch.float32, device="mps", requires_grad=True)
    value = torch.randn(shape, dtype=torch.float32, device="mps", requires_grad=True)
    torch.mps.synchronize()
    expected_grads = reference_attention_grads(query, key, value)

    observed = metal_sdpa_extension.metal_flash_attention_autograd(
        query,
        key,
        value,
        False,
        0.0,
    )
    torch.mps.synchronize()

    if not observed.requires_grad or observed.grad_fn is None:
        raise SystemExit("UMFA Metal SDPA output is detached; autograd is required for training.")

    loss = observed.square().mean()
    loss.backward()
    torch.mps.synchronize()

    for name, tensor, expected in (
        ("query", query, expected_grads[0]),
        ("key", key, expected_grads[1]),
        ("value", value, expected_grads[2]),
    ):
        if tensor.grad is None:
            raise SystemExit(f"UMFA Metal SDPA did not produce a {name} gradient.")
        if not torch.isfinite(tensor.grad).all().item():
            raise SystemExit(f"UMFA Metal SDPA produced non-finite {name} gradients.")
        observed_grad = tensor.grad.detach().cpu().float()
        if not torch.allclose(observed_grad, expected, rtol=rtol, atol=atol):
            diff = (observed_grad - expected).abs()
            raise SystemExit(
                "UMFA Metal SDPA gradient parity failed for "
                f"shape={shape} tensor={name}: "
                f"max_abs={diff.max().item():.6g}, mean_abs={diff.mean().item():.6g}"
            )


check((1, 4, 64, 64), 1e-4, 1e-4)
check_autograd((1, 4, 64, 64), 1e-3, 1e-3)
check((1, 24, 512, 128), 1e-4, 1e-4)
check((1, 24, 512, 128), 1e-4, 1e-4, transposed=True)
check((1, 24, 512, 128), 1e-4, 1e-4, queued_producer=True)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "UMFA Metal SDPA runtime check timed out."

    if result.returncode == 0:
        return None

    detail = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    message = "UMFA Metal SDPA runtime check failed."
    if detail:
        message = f"{message}\n{detail[-2000:]}"
    return message


@lru_cache(maxsize=4)
def _metal_quantized_flash_attention_runtime_error(target_precision: int, quant_mode: int) -> Optional[str]:
    script = f"""
import torch
import torch.nn.functional as F
import metal_sdpa_extension

TARGET_PRECISION = {target_precision}
QUANT_MODE = {quant_mode}
OUTPUT_ATOL = 0.05 if TARGET_PRECISION == 3 else 0.5

if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
    raise SystemExit("MPS is not available")

selected_impl = getattr(metal_sdpa_extension, "metal_quantized_flash_attention_autograd", None)
if not callable(selected_impl):
    raise SystemExit("metal_sdpa_extension does not expose metal_quantized_flash_attention_autograd().")


def check(seed, heads):
    torch.manual_seed(seed)
    shape = (1, heads, 64, 32)
    query_base = torch.randn(shape, dtype=torch.float32, device="mps") * 0.05
    key_base = torch.randn(shape, dtype=torch.float32, device="mps") * 0.05
    value_base = torch.randn(shape, dtype=torch.float32, device="mps") * 0.05

    query_ref = query_base.detach().clone().requires_grad_(True)
    key_ref = key_base.detach().clone().requires_grad_(True)
    value_ref = value_base.detach().clone().requires_grad_(True)
    query = query_base.detach().clone().requires_grad_(True)
    key = key_base.detach().clone().requires_grad_(True)
    value = value_base.detach().clone().requires_grad_(True)

    expected = F.scaled_dot_product_attention(
        query_ref,
        key_ref,
        value_ref,
        dropout_p=0.0,
        is_causal=False,
    )
    observed = selected_impl(
        query,
        key,
        value,
        False,
        0.0,
        TARGET_PRECISION,
        QUANT_MODE,
    )
    torch.mps.synchronize()

    if observed.shape != query.shape:
        raise SystemExit("Unexpected quantized UMFA output shape: " + str(tuple(observed.shape)))
    if observed.dtype != torch.float32:
        raise SystemExit("Unexpected quantized UMFA output dtype: " + str(observed.dtype))
    if not observed.requires_grad or observed.grad_fn is None:
        raise SystemExit("Quantized UMFA output is detached; autograd is required for training.")
    if not torch.isfinite(observed).all().item():
        raise SystemExit("Quantized UMFA produced non-finite output values.")

    output_diff = (observed.detach() - expected.detach()).abs().max().item()
    if output_diff > OUTPUT_ATOL:
        raise SystemExit(
            "Quantized UMFA output drift is too high: "
            + "target_precision="
            + str(TARGET_PRECISION)
            + " heads="
            + str(heads)
            + " seed="
            + str(seed)
            + " max_abs="
            + str(output_diff)
        )

    loss = observed.square().mean()
    loss.backward()
    torch.mps.synchronize()

    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if tensor.grad is None:
            raise SystemExit("Quantized UMFA did not produce a " + name + " gradient.")
        if not torch.isfinite(tensor.grad).all().item():
            raise SystemExit("Quantized UMFA produced non-finite " + name + " gradients.")


for heads in (4, 8):
    for seed in range(8):
        check(seed, heads)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            check=False,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "Quantized UMFA Metal SDPA runtime check timed out."

    if result.returncode == 0:
        return None

    detail = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    message = "Quantized UMFA Metal SDPA runtime check failed."
    if detail:
        message = f"{message}\n{detail[-2000:]}"
    return message


def xformers_compute_capability_error() -> Optional[str]:
    """Return an error message if xformers is unsupported on the current GPU, else None.

    xformers does not support compute capability 9.0+ (Hopper architecture).
    """
    if not torch.cuda.is_available():
        return None
    major, _ = torch.cuda.get_device_capability()
    if major >= 9:
        return (
            f"xformers is not supported on GPUs with compute capability 9.0+ "
            f"(detected {major}.x). Use a different attention mechanism such as 'diffusers'."
        )
    return None


SLAKey = Tuple[int, str, Optional[int], torch.dtype]


class AttentionBackendController:
    _active_backend: Optional[str] = None
    _active_phase: AttentionPhase = AttentionPhase.TRAIN
    _sla_cache: Dict[SLAKey, "SparseLinearAttention"] = {}
    _sla_settings: Dict[str, Any] | None = None
    _optimizer: Optional[Any] = None
    _parameter_sink: Optional[list] = None
    _sink_param_ids: set[int] = set()
    _optimizer_param_ids: set[int] = set()
    _attention_logit_consumer: Optional[Callable[[Dict[str, torch.Tensor]], None]] = None
    _sla_state_store: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
    _sla_state_filename: str = "sla_attention.pt"
    _diffusers_backend_context = None
    _diffusers_backend_name: Optional[str] = None
    _metal_flash_attention_health_checked: set[str] = set()

    @classmethod
    def apply(cls, config, phase: AttentionPhase) -> None:
        cls._active_phase = phase
        backend_value = getattr(config, "attention_mechanism", "diffusers")
        if not isinstance(backend_value, str):
            backend_value = "diffusers"
        backend = (backend_value or "diffusers").strip().lower()
        if not backend:
            backend = "diffusers"
        backend_alias = backend.replace("_", "-")

        if AttentionBackendName is None and backend_alias in _DIFFUSERS_BACKEND_TARGETS:
            message = (
                f"Attention backend '{backend_alias}' requires a newer diffusers release. "
                "Please upgrade diffusers to use this backend."
            )
            logger.error(message)
            raise RuntimeError(message)

        if cls._is_sageattention_backend(backend):
            cls._configure_sageattention(config, backend, phase)
            return

        if cls._is_metal_flash_attention_backend(backend_alias):
            cls._enable_metal_flash_attention(backend_alias)
            return

        if backend == "sla":
            cls._enable_sla(config, phase)
            return

        diffusers_backend = cls._resolve_diffusers_backend(backend_alias)
        if diffusers_backend is not None:
            cls._enable_diffusers_backend(backend_alias, diffusers_backend)
            return

        cls.restore_default()

    @classmethod
    def restore_default(cls) -> None:
        cls._disable_diffusers_backend()
        functional = torch.nn.functional
        if hasattr(functional, "scaled_dot_product_attention_sdpa"):
            functional.scaled_dot_product_attention = functional.scaled_dot_product_attention_sdpa
        cls._active_backend = None
        cls._active_phase = AttentionPhase.TRAIN

    @classmethod
    def _store_sdpa_reference(cls) -> None:
        functional = torch.nn.functional
        if not hasattr(functional, "scaled_dot_product_attention_sdpa"):
            setattr(functional, "scaled_dot_product_attention_sdpa", functional.scaled_dot_product_attention)

    @classmethod
    def _call_original_sdpa(cls, query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa):
        functional = torch.nn.functional
        original = getattr(functional, "scaled_dot_product_attention_sdpa", functional.scaled_dot_product_attention)
        return original(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    @staticmethod
    def _normalize_backend_key(value: str) -> str:
        return value.replace("_", "-")

    @classmethod
    def _resolve_diffusers_backend(cls, backend: str) -> Optional[AttentionBackendName]:
        if AttentionBackendName is None or not _DIFFUSERS_BACKEND_ALIASES:
            return None
        normalized = cls._normalize_backend_key(backend)
        return _DIFFUSERS_BACKEND_ALIASES.get(normalized)

    @classmethod
    def _load_metal_flash_attention_extension(cls, backend: str = "metal-flash-attention"):
        backend_key = _normalize_backend_key(str(backend or "metal-flash-attention").strip().lower())
        profile = _METAL_FLASH_ATTENTION_PROFILES.get(backend_key)
        if profile is None:
            message = f"Unsupported Metal Flash Attention backend '{backend}'."
            logger.error(message)
            raise RuntimeError(message)
        try:
            package = importlib.import_module("pytorch_custom_op_ffi")
        except ImportError as exc:
            message = (
                "Could not import the UMFA PyTorch custom-op package. " f"Install it to use --attention_mechanism={backend}."
            )
            logger.error(message)
            raise RuntimeError(message) from exc

        availability_check = getattr(package, "is_metal_sdpa_available", None)
        if not callable(availability_check):
            message = "The installed UMFA PyTorch package does not expose is_metal_sdpa_available()."
            logger.error(message)
            raise RuntimeError(message)

        try:
            available = bool(availability_check())
        except Exception as exc:
            message = f"Failed to query UMFA Metal SDPA availability: {exc}"
            logger.error(message)
            raise RuntimeError(message) from exc

        if not available:
            message = "UMFA Metal SDPA is installed but not available on this host."
            logger.error(message)
            raise RuntimeError(message)

        try:
            extension = importlib.import_module("metal_sdpa_extension")
        except ImportError as exc:
            message = "Could not import metal_sdpa_extension from the UMFA PyTorch custom-op package."
            logger.error(message)
            raise RuntimeError(message) from exc

        selected_impl_name = (
            "metal_flash_attention_autograd"
            if profile.target_precision is None
            else "metal_quantized_flash_attention_autograd"
        )
        selected_impl = getattr(extension, selected_impl_name, None)
        if not callable(selected_impl):
            message = f"metal_sdpa_extension does not expose {selected_impl_name}()."
            logger.error(message)
            raise RuntimeError(message)

        cls._check_metal_flash_attention_runtime(backend)
        return extension

    @classmethod
    def _check_metal_flash_attention_runtime(cls, backend: str = "metal-flash-attention") -> None:
        backend_key = _normalize_backend_key(str(backend or "metal-flash-attention").strip().lower())
        if backend_key in cls._metal_flash_attention_health_checked:
            return

        profile = _METAL_FLASH_ATTENTION_PROFILES.get(backend_key, MetalFlashAttentionProfile())
        if profile.target_precision is None:
            runtime_error = _metal_flash_attention_runtime_error()
        else:
            runtime_error = _metal_quantized_flash_attention_runtime_error(profile.target_precision, profile.quant_mode)
        if runtime_error:
            logger.error(runtime_error)
            raise RuntimeError(runtime_error)

        cls._metal_flash_attention_health_checked.add(backend_key)

    @staticmethod
    def _is_metal_flash_attention_backend(backend: str) -> bool:
        return backend in _METAL_FLASH_ATTENTION_ALIASES

    @classmethod
    def _enable_metal_flash_attention(cls, backend: str) -> None:
        if cls._active_backend == backend:
            return

        cls._disable_diffusers_backend()
        profile = _METAL_FLASH_ATTENTION_PROFILES.get(backend, MetalFlashAttentionProfile())
        extension = cls._load_metal_flash_attention_extension(backend)
        if profile.target_precision is None:
            selected_impl = extension.metal_flash_attention_autograd
        else:
            selected_impl = extension.metal_quantized_flash_attention_autograd
        cls._store_sdpa_reference()

        def wrapper(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            enable_gqa=False,
        ):
            if cls._metal_flash_attention_should_fallback(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa):
                return cls._call_original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

            try:
                if profile.target_precision is None:
                    return selected_impl(
                        query,
                        key,
                        value,
                        is_causal,
                        0.0 if scale is None else scale,
                    )
                return selected_impl(
                    query,
                    key,
                    value,
                    is_causal,
                    0.0 if scale is None else scale,
                    profile.target_precision,
                    profile.quant_mode,
                )
            except Exception as exc:
                logger.error(
                    "Could not run Metal Flash Attention (%s). Falling back to PyTorch SDPA.",
                    exc,
                )
                return cls._call_original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

        torch.nn.functional.scaled_dot_product_attention = wrapper
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention_metal_flash"):
            setattr(torch.nn.functional, "scaled_dot_product_attention_metal_flash", wrapper)
        cls._active_backend = backend

    @staticmethod
    def _metal_flash_attention_should_fallback(query, key, value, attn_mask, dropout_p, is_causal, enable_gqa) -> bool:
        if dropout_p not in (0, 0.0, None):
            return True
        if is_causal:
            return True
        if enable_gqa:
            return True
        if not all(isinstance(tensor, torch.Tensor) for tensor in (query, key, value)):
            return True
        if query.device.type != "mps" or key.device.type != "mps" or value.device.type != "mps":
            return True
        if query.ndim != 4:
            return True
        if query.shape[1] < 4 or query.shape[2] < 64:
            return True
        if query.dtype != torch.float32:
            return True
        if key.dtype != query.dtype or value.dtype != query.dtype:
            return True
        if attn_mask is not None:
            return True
        return False

    @classmethod
    def _enable_diffusers_backend(cls, backend_key: str, backend_enum: AttentionBackendName) -> None:
        if cls._diffusers_backend_name == backend_key:
            return
        if diffusers_attention_backend is None or _check_attention_backend_requirements is None:
            message = f"Diffusers attention backend helpers are unavailable. Upgrade diffusers to at least 0.35 to use {backend_key}."
            logger.error(message)
            raise RuntimeError(message)

        try:
            _check_attention_backend_requirements(backend_enum)
        except Exception as exc:  # pragma: no cover - exercised only when backend requirements fail
            message = f"Attention backend '{backend_key}' is unavailable: {exc}"
            logger.error(message)
            raise RuntimeError(message) from exc

        cls._disable_diffusers_backend()
        try:
            context = diffusers_attention_backend(backend_enum)
            context.__enter__()
        except Exception as exc:
            message = f"Failed to enable attention backend '{backend_key}': {exc}"
            logger.error(message)
            raise RuntimeError(message) from exc

        cls._diffusers_backend_context = context
        cls._diffusers_backend_name = backend_key
        cls._active_backend = backend_key

    @classmethod
    def _disable_diffusers_backend(cls) -> None:
        if cls._diffusers_backend_context is None:
            return
        try:
            cls._diffusers_backend_context.__exit__(None, None, None)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.debug("Failed to exit attention backend context cleanly (%s). Ignoring.", exc)
        finally:
            cls._diffusers_backend_context = None
            cls._diffusers_backend_name = None

    @staticmethod
    def _is_sageattention_backend(backend: str) -> bool:
        return backend.startswith("sageattention")

    @classmethod
    def _configure_sageattention(cls, config, backend: str, phase: AttentionPhase) -> None:
        cls._disable_diffusers_backend()
        usage_value = getattr(config, "sageattention_usage", AttentionBackendMode.INFERENCE)
        usage = AttentionBackendMode.from_raw(usage_value)
        setattr(config, "sageattention_usage", usage)

        should_enable = (phase == AttentionPhase.TRAIN and usage.allows_training) or (
            phase == AttentionPhase.EVAL and usage.allows_inference
        )

        if should_enable:
            if (
                phase == AttentionPhase.TRAIN
                and usage.allows_training
                and not getattr(config, "_sageattention_training_warned", False)
            ):
                logger.info("Using %s for training. This is an unsupported, experimental configuration.", backend)
                setattr(config, "_sageattention_training_warned", True)
            cls._enable_sageattention(backend, usage)
        else:
            cls.restore_default()

    @classmethod
    def _enable_sageattention(cls, backend: str, usage: AttentionBackendMode) -> None:
        if cls._active_backend == backend:
            return

        cls._disable_diffusers_backend()

        try:
            from sageattention import (
                sageattn,
                sageattn_qk_int8_pv_fp8_cuda,
                sageattn_qk_int8_pv_fp16_cuda,
                sageattn_qk_int8_pv_fp16_triton,
            )
        except ImportError as exc:
            message = f"Could not import SageAttention. Please install it to use --attention_mechanism={backend}."
            logger.error(message)
            logger.error(repr(exc))
            raise RuntimeError(message) from exc

        sageattn_functions = {
            "sageattention": sageattn,
            "sageattention-int8-fp16-triton": sageattn_qk_int8_pv_fp16_triton,
            "sageattention-int8-fp16-cuda": sageattn_qk_int8_pv_fp16_cuda,
            "sageattention-int8-fp8-cuda": sageattn_qk_int8_pv_fp8_cuda,
        }

        if backend not in sageattn_functions:
            message = f"Unsupported SageAttention backend '{backend}'."
            logger.error(message)
            raise ValueError(message)

        selected_impl = sageattn_functions[backend]
        cls._store_sdpa_reference()

        def wrapper(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            enable_gqa=False,
        ):
            try:
                return selected_impl(query, key, value, is_causal=is_causal)
            except Exception as exc:
                logger.error(
                    "Could not run SageAttention with %s (%s). Falling back to PyTorch SDPA.",
                    backend,
                    exc,
                )
                return cls._call_original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

        torch.nn.functional.scaled_dot_product_attention = wrapper
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention_sage"):
            setattr(torch.nn.functional, "scaled_dot_product_attention_sage", wrapper)

        if usage.allows_training:
            logger.warning(
                "Using %s for attention calculations during training. Your attention layers will not be trained. "
                "To disable SageAttention, remove or set --attention_mechanism to a different value.",
                backend,
            )

        cls._active_backend = backend

    @classmethod
    def _enable_sla(cls, config, phase: AttentionPhase) -> None:
        if cls._active_backend == "sla":
            return

        cls._disable_diffusers_backend()

        try:
            from sparse_linear_attention import SparseLinearAttention
        except ImportError as exc:
            message = "Could not import SparseLinearAttention. Install it to use --attention_mechanism=sla."
            logger.error(message)
            logger.error(repr(exc))
            raise RuntimeError(message) from exc

        cls._store_sdpa_reference()
        defaults = {
            "topk": 0.2,
            "feature_map": "softmax",
            "blkq": 64,
            "blkk": 64,
            "tie_feature_map_qk": True,
        }

        user_config = getattr(config, "sla_config", None)
        if not isinstance(user_config, dict):
            user_config = {}

        def _get_value(key: str, attr_name: str, fallback, caster):
            candidate = user_config.get(key, getattr(config, attr_name, None))
            if candidate in (None, "", "None"):
                candidate = fallback
            try:
                return caster(candidate)
            except Exception as exc:
                raise ValueError(f"Invalid SLA setting '{key}': {candidate}") from exc

        def _to_bool(value):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)

        settings = {
            "topk": float(_get_value("topk", "sla_topk", defaults["topk"], float)),
            "feature_map": str(_get_value("feature_map", "sla_feature_map", defaults["feature_map"], str)),
            "blkq": int(_get_value("blkq", "sla_blkq", defaults["blkq"], int)),
            "blkk": int(_get_value("blkk", "sla_blkk", defaults["blkk"], int)),
            "tie_feature_map_qk": bool(
                _to_bool(
                    _get_value("tie_feature_map_qk", "sla_tie_feature_map_qk", defaults["tie_feature_map_qk"], lambda v: v)
                )
            ),
        }

        if cls._sla_settings != settings:
            cls._sla_cache.clear()
            cls._sink_param_ids.clear()
            cls._optimizer_param_ids.clear()
            cls._sla_state_store.clear()
            cls._sla_settings = settings

        def wrapper(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=None,
            enable_gqa=False,
        ):
            if cls._sla_should_fallback(query, attn_mask, dropout_p, is_causal, enable_gqa):
                return cls._call_original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

            try:
                module = cls._get_sla_module(SparseLinearAttention, query)
            except Exception as exc:
                logger.error("Failed to prepare SLA module (%s). Falling back to PyTorch SDPA.", exc)
                return cls._call_original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

            try:
                return module(query, key, value)
            except Exception as exc:
                logger.error("Failed to execute SLA (%s). Falling back to PyTorch SDPA.", exc)
                return cls._call_original_sdpa(query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa)

        torch.nn.functional.scaled_dot_product_attention = wrapper
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention_sla"):
            setattr(torch.nn.functional, "scaled_dot_product_attention_sla", wrapper)
        cls._active_backend = "sla"
        cls._active_phase = phase

    @classmethod
    def _sla_should_fallback(cls, query, attn_mask, dropout_p, is_causal, enable_gqa) -> bool:
        if query.device.type != "cuda":
            return True
        if query.ndim != 4:
            return True
        if attn_mask is not None:
            return True
        if dropout_p not in (0, 0.0, None):
            return True
        if is_causal:
            return True
        if enable_gqa:
            return True
        if query.dtype not in (torch.float16, torch.bfloat16):
            return True
        return False

    @classmethod
    def _get_sla_module(cls, module_cls, query: torch.Tensor):
        if cls._sla_settings is None:
            raise RuntimeError("SLA settings have not been initialised.")
        head_dim = query.shape[-1]
        device = query.device
        cache_key: SLAKey = (head_dim, device.type, getattr(device, "index", None), query.dtype)
        module = cls._sla_cache.get(cache_key)
        if module is None:
            module = module_cls(
                head_dim=head_dim,
                topk=cls._sla_settings["topk"],
                feature_map=cls._sla_settings["feature_map"],
                BLKQ=cls._sla_settings["blkq"],
                BLKK=cls._sla_settings["blkk"],
                use_bf16=query.dtype == torch.bfloat16,
                tie_feature_map_qk=cls._sla_settings.get("tie_feature_map_qk", True),
            )
            module.to(device)
            cls._force_proj_fp32(module)
            cls._sla_cache[cache_key] = module
            cls._apply_state_to_module(module, head_dim, query.dtype)
            cls._sync_parameter_consumers()
        module.train(cls._active_phase == AttentionPhase.TRAIN)
        return module

    @staticmethod
    def _force_proj_fp32(module) -> None:
        proj = getattr(module, "proj_l", None)
        if proj is None:
            return
        try:
            device = next(proj.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        module.proj_l = proj.to(device=device, dtype=torch.float32)

    @classmethod
    def attach_parameter_sink(cls, sink: list) -> None:
        cls._parameter_sink = sink
        cls._sync_parameter_consumers()

    @classmethod
    def bind_optimizer(cls, optimizer: Any) -> None:
        cls._optimizer = optimizer
        cls._sync_parameter_consumers()

    @classmethod
    def get_trainable_parameters(cls) -> list[torch.nn.Parameter]:
        if cls._active_backend != "sla" or not cls._sla_cache:
            return []
        params: list[torch.nn.Parameter] = []
        for module in cls._sla_cache.values():
            params.extend(param for param in module.parameters() if param.requires_grad)
        return params

    @classmethod
    def _sync_parameter_consumers(cls) -> None:
        params = cls.get_trainable_parameters()
        if not params:
            return

        if cls._parameter_sink is not None:
            sink_candidates = [param for param in params if id(param) not in cls._sink_param_ids]
            if sink_candidates:
                cls._parameter_sink.extend(sink_candidates)
                cls._sink_param_ids.update(id(param) for param in sink_candidates)

        if cls._optimizer is not None:
            opt_candidates = [param for param in params if id(param) not in cls._optimizer_param_ids]
            if opt_candidates:
                add_group = getattr(cls._optimizer, "add_param_group", None)
                if callable(add_group):
                    add_group({"params": opt_candidates})
                    cls._optimizer_param_ids.update(id(param) for param in opt_candidates)
                else:
                    logger.error(
                        "Optimizer %s does not support add_param_group; SLA parameters cannot be registered automatically.",
                        type(cls._optimizer).__name__,
                    )
        cls._update_state_store_from_cache()

    @classmethod
    def register_attention_logit_consumer(cls, consumer: Callable[[Dict[str, torch.Tensor]], None]) -> None:
        cls._attention_logit_consumer = consumer

    @classmethod
    def publish_attention_max_logits(cls, logits: Dict[str, torch.Tensor]) -> None:
        if cls._attention_logit_consumer is None:
            return
        try:
            cls._attention_logit_consumer(logits)
        except Exception:
            logger.exception("Failed to forward attention max logits to consumer.")

    @classmethod
    def lookup_param_name(cls, param: Optional[torch.nn.Parameter]) -> str:
        if param is None or cls._optimizer is None:
            return ""
        mapping = getattr(cls._optimizer, "_param_to_name", None)
        if not isinstance(mapping, dict):
            return ""
        return mapping.get(id(param), "")

    @classmethod
    def _store_key(cls, head_dim: int, dtype: torch.dtype) -> Tuple[int, str]:
        return head_dim, cls._dtype_token(dtype)

    @staticmethod
    def _dtype_token(dtype: torch.dtype) -> str:
        if dtype == torch.bfloat16:
            return "bf16"
        if dtype == torch.float16:
            return "fp16"
        if dtype == torch.float32:
            return "fp32"
        return str(dtype).replace("torch.", "")

    @staticmethod
    def _dtype_from_token(token: str) -> torch.dtype:
        mapping = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        return mapping.get(token, getattr(torch, token, torch.bfloat16))

    @classmethod
    def _update_state_store_from_cache(cls) -> None:
        if not cls._sla_cache:
            return
        for cache_key, module in cls._sla_cache.items():
            head_dim, _, _, dtype = cache_key
            store_key = cls._store_key(head_dim, dtype)
            cls._sla_state_store[store_key] = {name: tensor.detach().cpu() for name, tensor in module.state_dict().items()}

    @classmethod
    def _apply_state_to_module(cls, module, head_dim: int, dtype: torch.dtype) -> None:
        store_key = cls._store_key(head_dim, dtype)
        state_dict = cls._sla_state_store.get(store_key)
        if not state_dict:
            return
        device = next(module.parameters()).device
        mapped = {name: tensor.to(device=device) for name, tensor in state_dict.items()}
        try:
            module.load_state_dict(mapped, strict=False)
        except Exception as exc:
            logger.warning("Failed to load SLA state for head_dim=%s dtype=%s (%s)", head_dim, dtype, exc)

    @classmethod
    def _serialize_state(cls) -> Optional[Dict[str, Any]]:
        cls._update_state_store_from_cache()
        if not cls._sla_state_store:
            return None
        serialized: Dict[str, Dict[str, torch.Tensor]] = {}
        for (head_dim, dtype_token), tensor_dict in cls._sla_state_store.items():
            key = f"{head_dim}:{dtype_token}"
            serialized[key] = {name: tensor.cpu() for name, tensor in tensor_dict.items()}
        return {
            "settings": cls._sla_settings,
            "state": serialized,
            "version": 1,
        }

    @classmethod
    def _load_state_store(cls, payload: Dict[str, Any]) -> None:
        serialized = payload.get("state") or {}
        restored: Dict[Tuple[int, str], Dict[str, torch.Tensor]] = {}
        for key, tensor_dict in serialized.items():
            try:
                head_dim_str, dtype_token = key.split(":", 1)
                head_dim = int(head_dim_str)
            except ValueError:
                logger.warning("Invalid SLA state key '%s'; skipping.", key)
                continue
            restored[(head_dim, dtype_token)] = {name: tensor.cpu() for name, tensor in tensor_dict.items()}
        cls._sla_state_store = restored
        saved_settings = payload.get("settings")
        if saved_settings and cls._sla_settings and cls._sla_settings != saved_settings:
            logger.warning(
                "SLA runtime settings differ from checkpoint settings. Runtime=%s, Checkpoint=%s. Proceeding with runtime configuration.",
                cls._sla_settings,
                saved_settings,
            )
        elif saved_settings and not cls._sla_settings:
            cls._sla_settings = saved_settings
        cls._apply_store_to_cache()

    @classmethod
    def _apply_store_to_cache(cls) -> None:
        if not cls._sla_cache:
            return
        for cache_key, module in cls._sla_cache.items():
            head_dim, _, _, dtype = cache_key
            cls._apply_state_to_module(module, head_dim, dtype)

    @classmethod
    def on_save_checkpoint(cls, directory: str, *, is_main_process: bool = True) -> None:
        if not is_main_process:
            return
        payload = cls._serialize_state()
        if not payload:
            return
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, cls._sla_state_filename)
        try:
            torch.save(payload, path)
        except Exception as exc:
            logger.error("Failed to save SLA attention state to %s (%s)", path, exc)

    @classmethod
    def on_load_checkpoint(cls, directory: str) -> None:
        path = os.path.join(directory, cls._sla_state_filename)
        if not os.path.exists(path):
            return
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as exc:
            logger.error("Failed to load SLA attention state from %s (%s)", path, exc)
            return
        if not isinstance(payload, dict):
            logger.error("Invalid SLA attention payload type: %s", type(payload))
            return
        cls._load_state_store(payload)


__all__ = [
    "AttentionBackendController",
    "AttentionBackendMode",
    "AttentionPhase",
]
