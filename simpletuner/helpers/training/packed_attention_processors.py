from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, FusedAttnProcessor2_0

from simpletuner.helpers.training.attention_backend import get_packed_attention_backend


def attention_mask_to_keep_mask(
    attention_mask: Optional[torch.Tensor],
    batch_size: int,
    sequence_length: int,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None
    if attention_mask.dtype == torch.bool:
        keep_mask = attention_mask
    else:
        keep_mask = attention_mask >= -9000.0

    if keep_mask.ndim == 1:
        keep_mask = keep_mask.unsqueeze(0).expand(batch_size, sequence_length)
    elif keep_mask.ndim == 2:
        pass
    elif keep_mask.ndim == 3:
        keep_mask = keep_mask.any(dim=1)
    elif keep_mask.ndim == 4:
        keep_mask = keep_mask.any(dim=(1, 2))
    else:
        raise ValueError(f"Unsupported packed attention mask shape: {tuple(attention_mask.shape)}")

    if keep_mask.shape != (batch_size, sequence_length):
        raise ValueError(
            f"Packed attention mask shape {tuple(keep_mask.shape)} does not match " f"({batch_size}, {sequence_length})."
        )
    return keep_mask


def run_packed_qkv_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    preferred_backend: Optional[str],
    *,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError(
            "Packed qkv attention requires query, key, and value to share shape; "
            f"got {tuple(query.shape)}, {tuple(key.shape)}, {tuple(value.shape)}."
        )
    keep_mask = attention_mask_to_keep_mask(attention_mask, query.shape[0], query.shape[1])
    backend = get_packed_attention_backend(preferred_backend, require_varlen_qkvpacked=keep_mask is not None)
    qkv = torch.stack([query, key, value], dim=2).contiguous()
    return backend.qkvpacked(qkv, attention_mask=keep_mask, causal=False, softmax_scale=softmax_scale)


class PackedFusedAttnProcessor2_0:
    def __init__(self, preferred_backend: Optional[str] = None):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("PackedFusedAttnProcessor2_0 requires PyTorch 2.0 or newer.")
        self.preferred_backend = preferred_backend
        self.cross_attention_processor = FusedAttnProcessor2_0()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            return self.cross_attention_processor(
                attn, hidden_states, encoder_hidden_states, attention_mask, temb, *args, **kwargs
            )

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        qkv = attn.to_qkv(hidden_states)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = run_packed_qkv_attention(query, key, value, attention_mask, self.preferred_backend)
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


class PackedJointAttnProcessor2_0:
    def __init__(self, preferred_backend: Optional[str] = None):
        self.preferred_backend = preferred_backend

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sample_input_shape = hidden_states.shape
        context_input_shape = encoder_hidden_states.shape
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, sample_channel, sample_height, sample_width = sample_input_shape
            hidden_states = hidden_states.view(batch_size, sample_channel, sample_height * sample_width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, context_channel, context_height, context_width = context_input_shape
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, context_channel, context_height * context_width
            ).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]
        context_sequence_length = encoder_hidden_states.shape[1]
        qkv = attn.to_qkv(hidden_states)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        encoder_qkv = attn.to_added_qkv(encoder_hidden_states)
        encoder_query, encoder_key, encoder_value = torch.chunk(encoder_qkv, 3, dim=-1)

        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        hidden_states = run_packed_qkv_attention(query, key, value, attention_mask, self.preferred_backend)
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)
        sample_sequence_length = hidden_states.shape[1] - context_sequence_length
        hidden_states, encoder_hidden_states = (
            hidden_states[:, :sample_sequence_length],
            hidden_states[:, sample_sequence_length:],
        )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, sample_channel, sample_height, sample_width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                batch_size, context_channel, context_height, context_width
            )
        return hidden_states, encoder_hidden_states


class PackedAuraFlowAttnProcessor2_0:
    def __init__(self, preferred_backend: Optional[str] = None):
        self.preferred_backend = preferred_backend

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        qkv = attn.to_qkv(hidden_states)
        query, key, value = torch.chunk(qkv, 3, dim=-1)
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if encoder_hidden_states is not None:
            encoder_qkv = attn.to_added_qkv(encoder_hidden_states)
            encoder_query, encoder_key, encoder_value = torch.chunk(encoder_qkv, 3, dim=-1)
            encoder_query = encoder_query.view(batch_size, -1, attn.heads, head_dim)
            encoder_key = encoder_key.view(batch_size, -1, attn.heads, head_dim)
            encoder_value = encoder_value.view(batch_size, -1, attn.heads, head_dim)
            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        hidden_states = run_packed_qkv_attention(
            query,
            key,
            value,
            None,
            self.preferred_backend,
            softmax_scale=attn.scale,
        )
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim).to(query.dtype)

        if encoder_hidden_states is not None:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, encoder_hidden_states.shape[1] :],
                hidden_states[:, : encoder_hidden_states.shape[1]],
            )

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        return hidden_states
