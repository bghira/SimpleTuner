"""
Tinygrad-based attention processors for Flux models.

This module provides drop-in replacements for standard PyTorch attention operations
using tinygrad, with proper ROCm/AMDGPU support.

Key features:
- Full autograd support with custom backward pass
- Attention mask support
- Automatic dtype handling (including bfloat16 support)
- Zero-copy tensor conversion when possible
- Support for both single and dual attention blocks
- ROCm/AMDGPU detection and proper backend selection
"""

import torch
from torch import Tensor, FloatTensor
from tinygrad import Tensor as TinyTensor
from tinygrad import dtypes as tinygrad_dtypes
from einops import rearrange
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.attention_processor import Attention
import numpy as np


def detect_device_backend(tensor):
    """
    Detect the appropriate tinygrad backend based on the PyTorch tensor device.
    Properly handles ROCm which masquerades as CUDA in PyTorch.
    """
    if tensor.is_cuda:
        # Check if this is actually ROCm/AMD GPU
        try:
            # ROCm provides torch.version.hip
            if hasattr(torch.version, "hip") and torch.version.hip is not None:
                return "HIP"
            # Additional check using device properties
            device_props = torch.cuda.get_device_properties(tensor.device)
            # AMD GPUs often have specific naming patterns
            if any(
                amd_identifier in device_props.name.lower()
                for amd_identifier in ["amd", "radeon", "vega", "navi", "rdna"]
            ):
                return "HIP"
        except (AttributeError, RuntimeError):
            pass
        # Default to CUDA if not ROCm
        return "CUDA"
    elif tensor.device.type == "mps":
        return "METAL"
    else:
        return "CPU"


def safe_from_torch_dtype(torch_dtype):
    """
    Safely convert PyTorch dtype to tinygrad dtype with fallbacks.
    """
    dtype_map = {
        torch.float32: tinygrad_dtypes.float32,
        torch.float16: tinygrad_dtypes.float16,
        torch.bfloat16: tinygrad_dtypes.bfloat16,
        torch.float64: tinygrad_dtypes.float32,  # Fallback float64 to float32
        torch.int32: tinygrad_dtypes.int32,
        torch.int64: tinygrad_dtypes.int64,
        torch.int8: tinygrad_dtypes.int8,
        torch.uint8: tinygrad_dtypes.uint8,
        torch.bool: tinygrad_dtypes.bool,
    }

    if torch_dtype in dtype_map:
        return dtype_map[torch_dtype]

    # Try the built-in converter as fallback
    try:
        from tinygrad.dtype import _from_torch_dtype

        return _from_torch_dtype(torch_dtype)
    except (KeyError, ImportError) as e:
        print(
            f"Warning: Unsupported dtype {torch_dtype} for tinygrad, falling back to float32. Error: {e}"
        )
        return tinygrad_dtypes.float32


class TinyGradAttentionFunction(torch.autograd.Function):
    """Custom autograd function to handle gradient flow between PyTorch and tinygrad."""

    @staticmethod
    def forward(ctx, query, key, value, scale, attention_mask=None):
        original_dtype = query.dtype
        ctx.save_for_backward(query, key, value)

        # Detect appropriate backend (handles ROCm properly)
        device = detect_device_backend(query)

        # Synchronize before conversion
        if query.device.type == "mps":
            torch.mps.synchronize()
        elif query.is_cuda:
            torch.cuda.synchronize()

        # Save context
        ctx.scale = scale
        ctx.device = device
        ctx.torch_device = query.device
        ctx.original_dtype = original_dtype
        ctx.has_mask = attention_mask is not None

        # Handle bfloat16 by converting to float16 for tinygrad
        if query.dtype == torch.bfloat16:
            query = query.to(torch.float16).contiguous()
            key = key.to(torch.float16).contiguous()
            value = value.to(torch.float16).contiguous()
            if attention_mask is not None:
                attention_mask = attention_mask.to(torch.float16).contiguous()

        # Convert to tinygrad
        q_tiny = TinyTensor.from_blob(
            query.data_ptr(),
            query.shape,
            dtype=safe_from_torch_dtype(query.dtype),
            device=device,
        )
        k_tiny = TinyTensor.from_blob(
            key.data_ptr(),
            key.shape,
            dtype=safe_from_torch_dtype(key.dtype),
            device=device,
        )
        v_tiny = TinyTensor.from_blob(
            value.data_ptr(),
            value.shape,
            dtype=safe_from_torch_dtype(value.dtype),
            device=device,
        )

        # Enable gradients on tinygrad tensors
        q_tiny.requires_grad = True
        k_tiny.requires_grad = True
        v_tiny.requires_grad = True

        # Compute attention scores: Q @ K^T
        k_transposed = k_tiny.transpose(-2, -1)
        scores = (q_tiny @ k_transposed) * scale

        # Apply attention mask if provided
        if attention_mask is not None:
            mask_tiny = TinyTensor.from_blob(
                attention_mask.data_ptr(),
                attention_mask.shape,
                dtype=safe_from_torch_dtype(attention_mask.dtype),
                device=device,
            )
            scores = scores + mask_tiny

        # Softmax with numerical stability
        scores_max = scores.max(axis=-1, keepdim=True)
        scores_stable = scores - scores_max
        scores_exp = scores_stable.exp()
        scores_sum = scores_exp.sum(axis=-1, keepdim=True)
        attention_weights = scores_exp / scores_sum

        # Apply attention to values
        output = attention_weights @ v_tiny

        # Realize the computation
        output = output.realize()

        # Store tinygrad tensors for backward
        ctx.q_tiny = q_tiny
        ctx.k_tiny = k_tiny
        ctx.v_tiny = v_tiny
        ctx.output_tiny = output
        ctx.attention_weights = attention_weights

        # Convert output back to PyTorch
        output_np = output.numpy()
        output_torch = torch.from_numpy(output_np).to(ctx.torch_device)

        # Convert back to original dtype if needed
        if original_dtype == torch.bfloat16:
            output_torch = output_torch.to(torch.bfloat16)

        return output_torch

    @staticmethod
    def backward(ctx, grad_output):
        # Retrieve saved tensors
        query, key, value = ctx.saved_tensors
        scale = ctx.scale

        # Convert grad_output to tinygrad
        grad_tiny = TinyTensor(
            grad_output.cpu().numpy(), device=ctx.device, requires_grad=False
        )

        # Get saved tinygrad tensors
        q_tiny = ctx.q_tiny
        k_tiny = ctx.k_tiny
        v_tiny = ctx.v_tiny
        attention_weights = ctx.attention_weights

        # Backward through attention: grad_output @ V^T
        v_transposed = v_tiny.transpose(-2, -1)
        grad_attention_weights = grad_tiny @ v_transposed

        # Backward through values: attention_weights^T @ grad_output
        attention_weights_t = attention_weights.transpose(-2, -1)
        grad_v = attention_weights_t @ grad_tiny

        # Backward through softmax (simplified for efficiency)
        softmax_grad_sum = (grad_attention_weights * attention_weights).sum(
            axis=-1, keepdim=True
        )
        grad_scores = (grad_attention_weights - softmax_grad_sum) * attention_weights

        # Scale gradient
        grad_scores = grad_scores * scale

        # Backward through Q @ K^T
        grad_q = grad_scores @ k_tiny  # grad_scores @ K
        grad_k = grad_scores.transpose(-2, -1) @ q_tiny  # grad_scores^T @ Q

        # Realize gradients
        grad_q = grad_q.realize()
        grad_k = grad_k.realize()
        grad_v = grad_v.realize()

        # Convert gradients back to PyTorch
        grad_q_torch = torch.from_numpy(grad_q.numpy()).to(
            ctx.torch_device, dtype=ctx.original_dtype
        )
        grad_k_torch = torch.from_numpy(grad_k.numpy()).to(
            ctx.torch_device, dtype=ctx.original_dtype
        )
        grad_v_torch = torch.from_numpy(grad_v.numpy()).to(
            ctx.torch_device, dtype=ctx.original_dtype
        )

        return grad_q_torch, grad_k_torch, grad_v_torch, None, None


def tinygrad_attention(query, key, value, scale, attention_mask=None):
    """Wrapper function for the custom autograd function."""
    return TinyGradAttentionFunction.apply(query, key, value, scale, attention_mask)


class FluxTinygradAttnProcessor:
    """
    Attention processor for SINGLE attention (self-attention only).
    Uses tinygrad for scaled dot-product attention computation.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        attn,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor = None,
        attention_mask: FloatTensor = None,
        image_rotary_emb: Tensor = None,
    ) -> Tensor:
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size, _, _ = hidden_states.shape

        # Standard Q, K, V projections
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape for multi-head attention
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply normalization if needed
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Compute attention with tinygrad
        scale = head_dim**-0.5
        hidden_states = tinygrad_attention(query, key, value, scale, attention_mask)

        # Reshape back
        hidden_states = rearrange(hidden_states, "B H L D -> B L (H D)")
        hidden_states = hidden_states.to(query.dtype)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        return hidden_states


class FluxTinygradDualAttnProcessor:
    """
    Attention processor for DUAL attention blocks.
    Handles both self-attention and cross-attention in a single pass.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        attn,
        hidden_states: FloatTensor,
        encoder_hidden_states: FloatTensor = None,
        attention_mask: FloatTensor = None,
        image_rotary_emb: Tensor = None,
    ) -> FloatTensor:
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # Self-attention projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Cross-attention projections
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)
        encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
            batch_size, -1, attn.heads, head_dim
        ).transpose(1, 2)

        if attn.norm_added_q is not None:
            encoder_hidden_states_query_proj = attn.norm_added_q(
                encoder_hidden_states_query_proj
            )
        if attn.norm_added_k is not None:
            encoder_hidden_states_key_proj = attn.norm_added_k(
                encoder_hidden_states_key_proj
            )

        # Concatenate encoder and self-attention BEFORE RoPE
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        # Apply RoPE AFTER concatenation (RoPE is sized for concatenated sequence)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Compute attention
        scale = head_dim**-0.5
        hidden_states = tinygrad_attention(query, key, value, scale, attention_mask)

        # Reshape
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # Split encoder and self outputs
        encoder_seq_len = encoder_hidden_states.shape[1]
        encoder_hidden_states, hidden_states = (
            hidden_states[:, :encoder_seq_len],
            hidden_states[:, encoder_seq_len:],
        )

        # Output projections
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)  # dropout
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # Reshape if needed
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        return hidden_states, encoder_hidden_states


def set_tinygrad_attention_processors(model, use_tinygrad=True, verbose=True):
    """
    Set tinygrad attention processors for all Attention modules in a model.

    Args:
        model: The model containing Attention modules (e.g., a Flux transformer)
        use_tinygrad: If True, use tinygrad processors. If False, use default processors.
        verbose: If True, print detailed information including detected backend

    Example:
        # Enable tinygrad attention with ROCm support
        set_tinygrad_attention_processors(model)

        # Disable (go back to default)
        set_tinygrad_attention_processors(model, use_tinygrad=False)
    """
    if not use_tinygrad:
        # Reset to default processors
        model.set_attn_processor({})
        if verbose:
            print("Reset to default attention processors")
        return

    # Detect backend if verbose
    if verbose:
        # Try to detect the backend from any parameter tensor
        try:
            sample_param = next(model.parameters())
            backend = detect_device_backend(sample_param)
            print(f"Detected tinygrad backend: {backend}")
            if backend == "HIP":
                print("  ROCm/AMD GPU detected - using HIP backend")
        except StopIteration:
            print("Warning: No parameters found in model")

    attn_procs = {}

    for name, module in model.named_modules():
        if isinstance(module, Attention):
            # The key needs to end with '.processor' for set_attn_processor
            processor_key = f"{name}.processor"

            # Check if this is a dual attention block (has encoder projection layers)
            if hasattr(module, "add_q_proj"):
                attn_procs[processor_key] = FluxTinygradDualAttnProcessor()
                if verbose:
                    print(f"  Setting dual attention processor for: {name}")
            else:
                attn_procs[processor_key] = FluxTinygradAttnProcessor()
                if verbose:
                    print(f"  Setting single attention processor for: {name}")

    if attn_procs:
        if verbose:
            print(f"Setting {len(attn_procs)} tinygrad attention processors")
        model.set_attn_processor(attn_procs)
    else:
        if verbose:
            print("No Attention modules found in the model")
