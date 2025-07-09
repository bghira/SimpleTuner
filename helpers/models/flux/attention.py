import torch
from torch import Tensor, FloatTensor
from torch.nn import functional as F
from einops import rearrange
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb

try:
    from flash_attn_interface import flash_attn_func, flash_attn_qkvpacked_func
except:
    pass


def fa3_sdpa(
    q,
    k,
    v,
):
    # flash attention 3 sdpa drop-in replacement
    q, k, v = [x.permute(0, 2, 1, 3) for x in [q, k, v]]
    out = flash_attn_func(q, k, v)[0]
    return out.permute(0, 2, 1, 3)


class FluxSingleAttnProcessor3_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

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

        batch_size, _, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = fa3_sdpa(query, key, value)
        hidden_states = rearrange(hidden_states, "B H L D -> B L (H D)")

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        return hidden_states


class FluxAttnProcessor3_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxAttnProcessor3_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

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

        # `sample` projections.
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

        # `context` projections.
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

        # attention
        query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
        key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = fa3_sdpa(query, key, value)
        hidden_states = rearrange(hidden_states, "B H L D -> B L (H D)")

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        return hidden_states, encoder_hidden_states


class FluxFusedFlashAttnProcessor3(object):
    """
    True fused QKV Flash Attention 3 processor for Flux models.
    Keeps QKV tensors packed through the entire attention computation.
    """

    def __init__(self):
        self.flash_attn_qkvpacked_func = None
        try:
            from flash_attn_interface import flash_attn_qkvpacked_func

            self.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
        except ImportError:
            raise ImportError(
                "FluxFusedFlashAttnProcessor3 requires flash-attn library. "
                "Please see this link for Hopper and Blackwell instructions: https://github.com/bghira/SimpleTuner/blob/main/INSTALL.md#nvidia-hopper--blackwell-follow-up-steps"
            )

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

        context_input_ndim = (
            encoder_hidden_states.ndim if encoder_hidden_states is not None else None
        )
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size = (
            encoder_hidden_states.shape[0]
            if encoder_hidden_states is not None
            else hidden_states.shape[0]
        )
        seq_len = hidden_states.shape[1]

        # Fused QKV projection
        qkv = attn.to_qkv(hidden_states)  # (batch, seq_len, 3 * inner_dim)
        inner_dim = qkv.shape[-1] // 3
        head_dim = inner_dim // attn.heads

        # Reshape to packed format: (batch, seq_len, 3, heads, head_dim)
        qkv = qkv.view(batch_size, seq_len, 3, attn.heads, head_dim)

        # Apply norms if needed (requires temporary unpacking)
        if attn.norm_q is not None or attn.norm_k is not None:
            q, k, v = qkv.unbind(dim=2)  # Each is (batch, seq_len, heads, head_dim)
            q = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            if attn.norm_q is not None:
                q = attn.norm_q(q)
            if attn.norm_k is not None:
                k = attn.norm_k(k)

            # Repack: back to (batch, seq_len, 3, heads, head_dim)
            qkv = torch.stack(
                [q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)], dim=2
            )

        # Handle encoder states if present
        if encoder_hidden_states is not None:
            encoder_seq_len = encoder_hidden_states.shape[1]

            # Fused encoder QKV
            encoder_qkv = attn.to_added_qkv(encoder_hidden_states)
            encoder_qkv = encoder_qkv.view(
                batch_size, encoder_seq_len, 3, attn.heads, head_dim
            )

            # Apply norms if needed
            if attn.norm_added_q is not None or attn.norm_added_k is not None:
                enc_q, enc_k, enc_v = encoder_qkv.unbind(dim=2)
                enc_q = enc_q.transpose(1, 2)
                enc_k = enc_k.transpose(1, 2)
                enc_v = enc_v.transpose(1, 2)

                if attn.norm_added_q is not None:
                    enc_q = attn.norm_added_q(enc_q)
                if attn.norm_added_k is not None:
                    enc_k = attn.norm_added_k(enc_k)

                encoder_qkv = torch.stack(
                    [
                        enc_q.transpose(1, 2),
                        enc_k.transpose(1, 2),
                        enc_v.transpose(1, 2),
                    ],
                    dim=2,
                )

            # Concatenate along sequence dimension
            qkv = torch.cat(
                [encoder_qkv, qkv], dim=1
            )  # (batch, encoder_seq + seq, 3, heads, head_dim)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            q, k, v = qkv.unbind(dim=2)  # Each is (batch, seq_len, heads, head_dim)

            # Transpose to (batch, heads, seq_len, head_dim) for RoPE
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Apply RoPE to q and k
            q = apply_rotary_emb(q, image_rotary_emb)
            k = apply_rotary_emb(k, image_rotary_emb)

            # Transpose back and repack
            qkv = torch.stack(
                [q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)], dim=2
            )

        # Flash Attention 3 with packed QKV
        # Input shape: (batch, seq_len, 3, heads, head_dim)
        # Output shape: (batch, seq_len, heads, head_dim)
        hidden_states = self.flash_attn_qkvpacked_func(
            qkv,
            causal=False,
            # Don't pass num_heads_q for standard MHA
        )

        # Reshape output: (batch, seq_len, heads, head_dim) -> (batch, seq_len, heads * head_dim)
        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(qkv.dtype)

        # Split and process outputs
        if encoder_hidden_states is not None:
            encoder_seq_len = encoder_hidden_states.shape[1]
            encoder_hidden_states = hidden_states[:, :encoder_seq_len]
            hidden_states = hidden_states[:, encoder_seq_len:]

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
        else:
            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )
            return hidden_states


class FluxFusedSDPAProcessor:
    """
    Fused QKV processor using PyTorch's scaled_dot_product_attention.
    Uses fused projections but splits for attention computation.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxFusedSDPAProcessor requires PyTorch 2.0+ for scaled_dot_product_attention"
            )

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

        context_input_ndim = (
            encoder_hidden_states.ndim if encoder_hidden_states is not None else None
        )
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(
                batch_size, channel, height * width
            ).transpose(1, 2)

        batch_size = (
            encoder_hidden_states.shape[0]
            if encoder_hidden_states is not None
            else hidden_states.shape[0]
        )

        # Single attention case (no encoder states)
        if encoder_hidden_states is None:
            # Use fused QKV projection
            qkv = attn.to_qkv(hidden_states)  # (batch, seq_len, 3 * inner_dim)
            inner_dim = qkv.shape[-1] // 3
            head_dim = inner_dim // attn.heads
            seq_len = hidden_states.shape[1]

            # Split and reshape
            qkv = qkv.view(batch_size, seq_len, 3, attn.heads, head_dim)
            query, key, value = qkv.unbind(
                dim=2
            )  # Each is (batch, seq_len, heads, head_dim)

            # Transpose to (batch, heads, seq_len, head_dim)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            # Apply norms if needed
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            # Apply RoPE if needed
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

            # SDPA
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )

            # Reshape back
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)

            if input_ndim == 4:
                hidden_states = hidden_states.transpose(-1, -2).reshape(
                    batch_size, channel, height, width
                )

            return hidden_states

        # Joint attention case (with encoder states)
        else:
            # Process self-attention QKV
            qkv = attn.to_qkv(hidden_states)
            inner_dim = qkv.shape[-1] // 3
            head_dim = inner_dim // attn.heads
            seq_len = hidden_states.shape[1]

            qkv = qkv.view(batch_size, seq_len, 3, attn.heads, head_dim)
            query, key, value = qkv.unbind(dim=2)

            # Transpose to (batch, heads, seq_len, head_dim)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            # Apply norms if needed
            if attn.norm_q is not None:
                query = attn.norm_q(query)
            if attn.norm_k is not None:
                key = attn.norm_k(key)

            # Process encoder QKV
            encoder_seq_len = encoder_hidden_states.shape[1]
            encoder_qkv = attn.to_added_qkv(encoder_hidden_states)
            encoder_qkv = encoder_qkv.view(
                batch_size, encoder_seq_len, 3, attn.heads, head_dim
            )
            encoder_query, encoder_key, encoder_value = encoder_qkv.unbind(dim=2)

            # Transpose to (batch, heads, seq_len, head_dim)
            encoder_query = encoder_query.transpose(1, 2)
            encoder_key = encoder_key.transpose(1, 2)
            encoder_value = encoder_value.transpose(1, 2)

            # Apply encoder norms if needed
            if attn.norm_added_q is not None:
                encoder_query = attn.norm_added_q(encoder_query)
            if attn.norm_added_k is not None:
                encoder_key = attn.norm_added_k(encoder_key)

            # Concatenate encoder and self-attention
            query = torch.cat([encoder_query, query], dim=2)
            key = torch.cat([encoder_key, key], dim=2)
            value = torch.cat([encoder_value, value], dim=2)

            # Apply RoPE if needed
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

            # SDPA
            hidden_states = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
            )

            # Reshape: (batch, heads, seq_len, head_dim) -> (batch, seq_len, heads * head_dim)
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
            hidden_states = hidden_states.to(query.dtype)

            # Split encoder and self outputs
            encoder_hidden_states = hidden_states[:, :encoder_seq_len]
            hidden_states = hidden_states[:, encoder_seq_len:]

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


class FluxSingleFusedSDPAProcessor:
    """
    Fused QKV processor for single attention (no encoder states).
    Simpler version for self-attention only blocks.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxSingleFusedSDPAProcessor requires PyTorch 2.0+ for scaled_dot_product_attention"
            )

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

        batch_size, seq_len, _ = hidden_states.shape

        # Use fused QKV projection
        qkv = attn.to_qkv(hidden_states)  # (batch, seq_len, 3 * inner_dim)
        inner_dim = qkv.shape[-1] // 3
        head_dim = inner_dim // attn.heads

        # Split and reshape in one go
        qkv = qkv.view(batch_size, seq_len, 3, attn.heads, head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Now each is (batch, heads, seq_len, head_dim)

        # Apply norms if needed
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # SDPA
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        # Reshape back
        hidden_states = rearrange(hidden_states, "B H L D -> B L (H D)")
        hidden_states = hidden_states.to(query.dtype)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        return hidden_states
