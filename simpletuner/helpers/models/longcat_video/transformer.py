import logging
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin

logger = logging.getLogger(__name__)


def _rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _broadcat(tensors, dim=-1):
    """
    Broadcast then concatenate a list of tensors along the given dimension.

    This implementation explicitly computes a target shape for all non-concatenated
    axes and validates that each tensor can be broadcast to that shape before
    concatenation.
    """
    if not tensors:
        raise ValueError("No tensors provided to _broadcat.")
    nd = tensors[0].dim()
    dim = dim if dim >= 0 else nd + dim
    if dim < 0 or dim >= nd:
        raise ValueError(f"Invalid concat dimension {dim} for tensors with {nd} dims.")

    # Target shape for broadcast axes (leave concat axis None)
    target_shape = []
    for axis in range(nd):
        if axis == dim:
            target_shape.append(None)
        else:
            target_shape.append(max(t.shape[axis] for t in tensors))

    expanded = []
    for t in tensors:
        if t.dim() != nd:
            raise ValueError(f"Tensor rank mismatch in _broadcat: expected {nd} dims, got {t.dim()}")
        shape = list(target_shape)
        shape[dim] = t.shape[dim]
        # Validate broadcastability
        for axis in range(nd):
            if axis == dim:
                continue
            if t.shape[axis] not in (1, shape[axis]):
                raise ValueError(
                    f"Cannot broadcast tensor with shape {t.shape} to target shape {tuple(shape)} at axis {axis}"
                )
        expanded.append(t.expand(*shape))

    return torch.cat(expanded, dim=dim)


class RMSNorm_FP32(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LayerNorm_FP32(nn.LayerNorm):
    def __init__(self, dim, eps, elementwise_affine):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        out = F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            None if self.weight is None else self.weight.float(),
            None if self.bias is None else self.bias.float(),
            self.eps,
        ).to(origin_dtype)
        return out


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding."""

    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (2, 4, 4),
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer=None,
        flatten: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        return x


def modulate_fp32(norm_func, x, shift, scale):
    assert shift.dtype == torch.float32 and scale.dtype == torch.float32
    dtype = x.dtype
    x = norm_func(x.to(torch.float32))
    x = x * (scale + 1) + shift
    x = x.to(dtype)
    return x


class FeedForwardSwiGLU(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, t_embed_dim, frequency_embedding_size=256):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, t_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(t_embed_dim, t_embed_dim, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class CaptionEmbedder(nn.Module):
    """
    Projects text encoder outputs into the transformer hidden size.
    """

    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.y_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_size, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    def forward(self, caption):
        caption = self.y_proj(caption)
        return caption


class FinalLayer_FP32(nn.Module):
    """
    The final projection layer of the transformer.
    """

    def __init__(self, hidden_size, num_patch, out_channels, adaln_tembed_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_patch = num_patch
        self.out_channels = out_channels
        self.adaln_tembed_dim = adaln_tembed_dim

        self.norm_final = LayerNorm_FP32(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_tembed_dim, 2 * hidden_size, bias=True))

    def forward(self, x, t, latent_shape):
        assert t.dtype == torch.float32
        B, N, C = x.shape
        T, _, _ = latent_shape

        with torch.autocast(device_type=t.device.type, dtype=torch.float32, enabled=t.device.type == "cuda"):
            shift, scale = self.adaLN_modulation(t).unsqueeze(2).chunk(2, dim=-1)  # [B, T, 1, C]
            x = modulate_fp32(self.norm_final, x.view(B, T, -1, C), shift, scale).view(B, N, C)
            x = self.linear(x)
        return x


class RotaryPositionalEmbedding(nn.Module):
    """
    3D rotary positional embedding with cached frequencies.
    """

    def __init__(self, head_dim: int, cp_split_hw: Optional[List[int]] = None):
        super().__init__()
        assert head_dim % 8 == 0, "Dim must be a multiple of 8 for 3D RoPE."
        self.head_dim = head_dim
        self.cp_split_hw = cp_split_hw or [1, 1]
        self.base = 10000
        self.freqs_dict = {}

    def register_grid_size(self, grid_size: Tuple[int, int, int]):
        if grid_size not in self.freqs_dict:
            self.freqs_dict[grid_size] = self.precompute_freqs_cis_3d(grid_size)

    def precompute_freqs_cis_3d(self, grid_size: Tuple[int, int, int]):
        num_frames, height, width = grid_size
        dim_t = self.head_dim - 4 * (self.head_dim // 6)
        dim_h = 2 * (self.head_dim // 6)
        dim_w = 2 * (self.head_dim // 6)
        freqs_t = 1.0 / (self.base ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (self.base ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (self.base ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))
        grid_t = torch.linspace(0, num_frames - 1, num_frames, dtype=torch.float32)
        grid_h = torch.linspace(0, height - 1, height, dtype=torch.float32)
        grid_w = torch.linspace(0, width - 1, width, dtype=torch.float32)
        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)
        freqs_t = freqs_t.repeat_interleave(2, dim=-1)
        freqs_h = freqs_h.repeat_interleave(2, dim=-1)
        freqs_w = freqs_w.repeat_interleave(2, dim=-1)
        freqs = _broadcat((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)
        freqs = freqs.reshape(num_frames * height * width, -1)
        return freqs

    def forward(self, q, k, grid_size: Tuple[int, int, int]):
        if grid_size not in self.freqs_dict:
            self.register_grid_size(grid_size)

        freqs_cis = self.freqs_dict[grid_size].to(q.device)
        if q.device.type == "mps" and freqs_cis.dtype == torch.float64:
            freqs_cis = freqs_cis.float()
        q_, k_ = q.float(), k.float()
        freqs_cis = freqs_cis.float()
        cos, sin = freqs_cis.cos(), freqs_cis.sin()
        cos, sin = cos.view(1, 1, -1, cos.shape[-1]), sin.view(1, 1, -1, sin.shape[-1])
        q_ = (q_ * cos) + (_rotate_half(q_) * sin)
        k_ = (k_ * cos) + (_rotate_half(k_) * sin)

        return q_.type_as(q), k_.type_as(k)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        cp_split_hw: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers
        self.enable_bsa = enable_bsa
        self.bsa_params = bsa_params
        self.cp_split_hw = cp_split_hw or [1, 1]

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

        self.rope_3d = RotaryPositionalEmbedding(self.head_dim, cp_split_hw=self.cp_split_hw)

    def _flash_or_sdp_attention(self, q, k, v, shape):
        B, H, SQ, _ = q.shape
        attn_output = None

        if self.enable_bsa and shape[0] > 1:
            try:  # pragma: no cover - optional dependency
                from simpletuner.helpers.models.longcat_video.block_sparse_attention.bsa_interface import flash_attn_bsa_3d

                _, H_spatial, W_spatial = shape
                H_spatial //= self.cp_split_hw[0]
                W_spatial //= self.cp_split_hw[1]
                Tq = SQ // (H_spatial * W_spatial)
                latent_shape_q = (Tq, H_spatial, W_spatial)
                attn_output = flash_attn_bsa_3d(q, k, v, latent_shape_q, latent_shape_q, **(self.bsa_params or {}))
            except Exception:
                attn_output = None

        if attn_output is None:
            try:
                attn_output = dispatch_attention_fn(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    backend=getattr(self, "_attention_backend", None),
                )
            except Exception:
                attn_output = None

        if attn_output is None and self.enable_flashattn3:
            try:  # pragma: no cover - optional dependency
                from flash_attn_interface import flash_attn_func

                q_flash = q.permute(0, 2, 1, 3).contiguous()
                k_flash = k.permute(0, 2, 1, 3).contiguous()
                v_flash = v.permute(0, 2, 1, 3).contiguous()
                attn_output, *_ = flash_attn_func(q_flash, k_flash, v_flash, softmax_scale=self.scale)
                attn_output = attn_output.permute(0, 2, 1, 3)
            except Exception:
                attn_output = None

        if attn_output is None and self.enable_flashattn2:
            try:  # pragma: no cover - optional dependency
                from flash_attn import flash_attn_func

                q_flash = q.permute(0, 2, 1, 3)
                k_flash = k.permute(0, 2, 1, 3)
                v_flash = v.permute(0, 2, 1, 3)
                attn_output = flash_attn_func(q_flash, k_flash, v_flash, dropout_p=0.0, softmax_scale=self.scale)
                attn_output = attn_output.permute(0, 2, 1, 3)
            except Exception:
                attn_output = None

        if attn_output is None and self.enable_xformers:
            try:  # pragma: no cover - optional dependency
                import xformers.ops

                q_x = q.permute(0, 2, 1, 3)
                k_x = k.permute(0, 2, 1, 3)
                v_x = v.permute(0, 2, 1, 3)
                attn_output = xformers.ops.memory_efficient_attention(q_x, k_x, v_x, attn_bias=None, op=None)
                attn_output = attn_output.permute(0, 2, 1, 3)
            except Exception:
                attn_output = None

        if attn_output is None:
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn_weights = attn_weights.softmax(dim=-1)
            attn_output = torch.matmul(attn_weights, v)

        return attn_output

    def forward(self, x: torch.Tensor, shape=None, num_cond_latents=None, return_kv=False) -> Union[torch.Tensor, Tuple]:
        B, N, C = x.shape
        qkv = self.qkv(x)

        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape).permute((2, 0, 3, 1, 4))  # [3, B, H, N, D]
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if return_kv:
            k_cache, v_cache = k.clone(), v.clone()

        q, k = self.rope_3d(q, k, shape)

        if num_cond_latents is not None and num_cond_latents > 0:
            num_cond_latents_thw = num_cond_latents * (N // shape[0])
            q_cond = q[:, :, :num_cond_latents_thw].contiguous()
            k_cond = k[:, :, :num_cond_latents_thw].contiguous()
            v_cond = v[:, :, :num_cond_latents_thw].contiguous()
            x_cond = self._flash_or_sdp_attention(q_cond, k_cond, v_cond, shape)
            q_noise = q[:, :, num_cond_latents_thw:].contiguous()
            x_noise = self._flash_or_sdp_attention(q_noise, k, v, shape)
            x = torch.cat([x_cond, x_noise], dim=2).contiguous()
        else:
            x = self._flash_or_sdp_attention(q, k, v, shape)

        x_output_shape = (B, N, C)
        x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)

        if return_kv:
            return x, (k_cache, v_cache)
        return x

    def forward_with_kv_cache(self, x: torch.Tensor, shape=None, num_cond_latents=None, kv_cache=None):
        B, N, C = x.shape
        qkv = self.qkv(x)

        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.view(qkv_shape).permute((2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        k_full, v_full = k, v
        if kv_cache is not None:
            cached_k, cached_v = kv_cache
            cached_k = cached_k.to(device=x.device)
            cached_v = cached_v.to(device=x.device)
            if cached_k.shape[0] == 1:
                cached_k = cached_k.repeat(B, 1, 1, 1)
                cached_v = cached_v.repeat(B, 1, 1, 1)
            k_full = torch.cat([cached_k, k], dim=2).contiguous()
            v_full = torch.cat([cached_v, v], dim=2).contiguous()

        q_rot, k_rot = self.rope_3d(q, k_full, shape)
        attn_out = self._flash_or_sdp_attention(q_rot, k_rot, v_full, shape)

        x = attn_out.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        enable_flashattn3=False,
        enable_flashattn2=False,
        enable_xformers=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "d_model must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)

        self.enable_flashattn3 = enable_flashattn3
        self.enable_flashattn2 = enable_flashattn2
        self.enable_xformers = enable_xformers

    def _process_cross_attn(self, q, k, v, kv_seqlen):
        B, H, N, _ = q.shape
        attn_output = None

        if self.enable_flashattn3:
            try:  # pragma: no cover - optional dependency
                from flash_attn_interface import flash_attn_func

                q_flash = q.permute(0, 2, 1, 3).contiguous()
                k_flash = k.permute(0, 2, 1, 3).contiguous()
                v_flash = v.permute(0, 2, 1, 3).contiguous()
                attn_output, *_ = flash_attn_func(q_flash, k_flash, v_flash)
                attn_output = attn_output.permute(0, 2, 1, 3)
            except Exception:
                attn_output = None

        if attn_output is None and self.enable_flashattn2:
            try:  # pragma: no cover - optional dependency
                from flash_attn import flash_attn_func

                q_flash = q.permute(0, 2, 1, 3)
                k_flash = k.permute(0, 2, 1, 3)
                v_flash = v.permute(0, 2, 1, 3)
                attn_output = flash_attn_func(q_flash, k_flash, v_flash, dropout_p=0.0)
                attn_output = attn_output.permute(0, 2, 1, 3)
            except Exception:
                attn_output = None

        if attn_output is None and self.enable_xformers:
            try:  # pragma: no cover - optional dependency
                import xformers.ops

                q_x = q.permute(0, 2, 1, 3)
                k_x = k.permute(0, 2, 1, 3)
                v_x = v.permute(0, 2, 1, 3)
                if kv_seqlen is not None:
                    attn_bias = xformers.ops.fmha.attn_bias.BlockDiagonalMask.from_seqlens([N] * B, kv_seqlen)
                else:
                    attn_bias = None
                attn_output = xformers.ops.memory_efficient_attention(q_x, k_x, v_x, attn_bias=attn_bias)
                attn_output = attn_output.permute(0, 2, 1, 3)
            except Exception:
                attn_output = None

        if attn_output is None:
            attn_mask = None
            if kv_seqlen is not None:
                max_k = k.shape[2]
                attn_mask = torch.zeros((B, 1, N, max_k), device=q.device, dtype=q.dtype)
                for idx, seqlen in enumerate(kv_seqlen):
                    if seqlen < max_k:
                        attn_mask[idx, :, :, seqlen:] = float("-inf")
            try:
                attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
            except Exception as exc:
                logger.error(
                    "LongCat cross-attn SDPA failed: q %s, k %s, v %s, attn_mask %s, kv_seqlen %s",
                    tuple(q.shape),
                    tuple(k.shape),
                    tuple(v.shape),
                    tuple(attn_mask.shape) if attn_mask is not None else None,
                    kv_seqlen,
                )
                # Log a quick head/token summary for debugging
                logger.error(
                    "SDPA debug: B=%s, heads=%s, q_tokens=%s, k_tokens=%s",
                    q.shape[0],
                    q.shape[1],
                    q.shape[2],
                    k.shape[2],
                )
                raise

        return attn_output

    def forward(self, x, cond, kv_seqlen, num_cond_latents=None, shape=None):
        B, N, C = x.shape
        assert C == self.dim and cond.shape[2] == self.dim

        q = self.q_linear(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_linear(cond).view(B, cond.shape[1], 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        kv_list = []
        for b in range(B):
            seq_len = kv_seqlen[b] if kv_seqlen is not None else cond.shape[1]
            kv_list.append(seq_len)
        kv_list = kv_list or [cond.shape[1]] * B

        attn_output = self._process_cross_attn(q, k, v, kv_list)
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)
        attn_output = self.proj(attn_output)
        return attn_output


class LongCatSingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params=None,
        cp_split_hw=None,
    ):
        super().__init__()

        self.hidden_size = hidden_size

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True))

        self.mod_norm_attn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mod_norm_ffn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)

        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
            enable_bsa=enable_bsa,
            bsa_params=bsa_params,
            cp_split_hw=cp_split_hw,
        )
        self.cross_attn = MultiHeadCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            enable_flashattn3=enable_flashattn3,
            enable_flashattn2=enable_flashattn2,
            enable_xformers=enable_xformers,
        )
        self.ffn = FeedForwardSwiGLU(dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio))

    def forward(
        self, x, y, t, y_seqlen, latent_shape, num_cond_latents=None, return_kv=False, kv_cache=None, skip_crs_attn=False
    ):
        x_dtype = x.dtype

        B, N, C = x.shape
        T, _, _ = latent_shape

        with torch.autocast(device_type=t.device.type, dtype=torch.float32, enabled=t.device.type == "cuda"):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(t).unsqueeze(2).chunk(6, dim=-1)
            )

        x_m = modulate_fp32(self.mod_norm_attn, x.view(B, T, -1, C), shift_msa, scale_msa).view(B, N, C)

        if kv_cache is not None:
            attn_outputs = self.attn.forward_with_kv_cache(
                x_m, shape=latent_shape, num_cond_latents=num_cond_latents, kv_cache=kv_cache
            )
            kv_cache = kv_cache
        else:
            attn_outputs = self.attn(x_m, shape=latent_shape, num_cond_latents=num_cond_latents, return_kv=return_kv)
        if return_kv:
            x_s, kv_cache = attn_outputs
        else:
            x_s = attn_outputs

        with torch.autocast(device_type=t.device.type, dtype=torch.float32, enabled=t.device.type == "cuda"):
            x = x + (gate_msa * x_s.view(B, -1, N // T, C)).view(B, -1, C)
        x = x.to(x_dtype)

        if not skip_crs_attn:
            if kv_cache is not None:
                num_cond_latents = None
            x = x + self.cross_attn(
                self.pre_crs_attn_norm(x), y, y_seqlen, num_cond_latents=num_cond_latents, shape=latent_shape
            )

        x = modulate_fp32(self.mod_norm_ffn, x.view(B, T, -1, C), shift_mlp, scale_mlp).view(B, N, C)

        with torch.autocast(device_type=t.device.type, dtype=torch.float32, enabled=t.device.type == "cuda"):
            x = x + gate_mlp * self.ffn(x.view(B, -1, N // T, C)).view(B, -1, C)
        x = x.to(x_dtype)

        if return_kv:
            return x, kv_cache
        return x


class LongCatVideoTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    The 3D DiT-style transformer used by LongCat-Video.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 4096,
        depth: int = 48,
        num_heads: int = 32,
        caption_channels: int = 4096,
        mlp_ratio: int = 4,
        adaln_tembed_dim: int = 512,
        frequency_embedding_size: int = 256,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        enable_flashattn3: bool = False,
        enable_flashattn2: bool = False,
        enable_xformers: bool = False,
        enable_bsa: bool = False,
        bsa_params: dict = None,
        cp_split_hw: Optional[List[int]] = None,
        text_tokens_zero_pad: bool = False,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cp_split_hw = cp_split_hw or [1, 1]

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(t_embed_dim=adaln_tembed_dim, frequency_embedding_size=frequency_embedding_size)
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size)

        self.blocks = nn.ModuleList(
            [
                LongCatSingleStreamBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    adaln_tembed_dim=adaln_tembed_dim,
                    enable_flashattn3=enable_flashattn3,
                    enable_flashattn2=enable_flashattn2,
                    enable_xformers=enable_xformers,
                    enable_bsa=enable_bsa,
                    bsa_params=bsa_params,
                    cp_split_hw=self.cp_split_hw,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer_FP32(hidden_size, math.prod(self.patch_size), out_channels, adaln_tembed_dim)

        self.gradient_checkpointing = False
        self.text_tokens_zero_pad = text_tokens_zero_pad

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        num_cond_latents=0,
        return_kv=False,
        kv_cache_dict=None,
        skip_crs_attn=False,
        offload_kv_cache=False,
        return_dict: bool = True,
    ):
        if kv_cache_dict is None:
            kv_cache_dict = {}

        B, _, T, H, W = hidden_states.shape
        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        if len(timestep.shape) == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t)

        dtype = self.x_embedder.proj.weight.dtype
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        hidden_states = self.x_embedder(hidden_states)

        with torch.autocast(device_type=timestep.device.type, dtype=torch.float32, enabled=timestep.device.type == "cuda"):
            t = self.t_embedder(timestep.float().flatten(), dtype=torch.float32).reshape(B, N_t, -1)

        encoder_hidden_states = self.y_embedder(encoder_hidden_states)

        def _normalize_mask(mask: torch.Tensor) -> torch.Tensor:
            if mask.dim() == 4 and mask.shape[1] == 1 and mask.shape[2] == 1:
                return mask.squeeze(1).squeeze(1)
            if mask.dim() == 3 and mask.shape[1] == 1:
                return mask.squeeze(1)
            if mask.dim() == 3 and mask.shape[2] == 1:
                return mask.squeeze(2)
            if mask.dim() == 2:
                return mask
            raise ValueError(f"Unexpected encoder_attention_mask shape: {mask.shape}")

        if self.text_tokens_zero_pad and encoder_attention_mask is not None:
            try:
                encoder_attention_mask = _normalize_mask(encoder_attention_mask)
                encoder_hidden_states = encoder_hidden_states * encoder_attention_mask.unsqueeze(-1).to(dtype)
            except Exception as exc:
                logger.error(
                    "LongCat encoder mask broadcast failed: encoder_hidden_states shape %s, mask shape %s",
                    tuple(encoder_hidden_states.shape),
                    tuple(encoder_attention_mask.shape) if encoder_attention_mask is not None else None,
                )
                raise
            encoder_attention_mask = (encoder_attention_mask * 0 + 1).to(encoder_attention_mask.dtype)

        if encoder_attention_mask is not None:
            encoder_attention_mask = _normalize_mask(encoder_attention_mask).to(dtype)
            y_seqlens = encoder_attention_mask.sum(dim=1).tolist()

            flat_mask = encoder_attention_mask.reshape(-1)
            encoder_hidden_states = encoder_hidden_states.reshape(
                B * encoder_hidden_states.shape[1], encoder_hidden_states.shape[2]
            )
            encoder_hidden_states = encoder_hidden_states[flat_mask != 0]
            encoder_hidden_states = encoder_hidden_states.view(1, -1, hidden_states.shape[-1])
        else:
            y_seqlens = [encoder_hidden_states.shape[1]] * encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.view(1, -1, hidden_states.shape[-1])

        if self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
            hidden_states = hidden_states.view(B, N_t, N_h, N_w, -1)
            hidden_states = hidden_states.view(B, -1, hidden_states.shape[-1])

        kv_cache_dict_ret = {}
        for i, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                block_outputs = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    t,
                    y_seqlens,
                    (N_t, N_h, N_w),
                    num_cond_latents,
                    return_kv,
                    kv_cache_dict.get(i, None),
                    skip_crs_attn,
                )
            else:
                block_outputs = block(
                    hidden_states,
                    encoder_hidden_states,
                    t,
                    y_seqlens,
                    (N_t, N_h, N_w),
                    num_cond_latents,
                    return_kv,
                    kv_cache_dict.get(i, None),
                    skip_crs_attn,
                )

            if return_kv:
                hidden_states, kv_cache = block_outputs
                if offload_kv_cache:
                    kv_cache_dict_ret[i] = (kv_cache[0].cpu(), kv_cache[1].cpu())
                else:
                    kv_cache_dict_ret[i] = (kv_cache[0].contiguous(), kv_cache[1].contiguous())
            else:
                hidden_states = block_outputs

        hidden_states = self.final_layer(hidden_states, t, (N_t, N_h, N_w))

        if self.cp_split_hw[0] * self.cp_split_hw[1] > 1:
            hidden_states = hidden_states.view(B, N_t, N_h, N_w, -1)
            hidden_states = hidden_states.view(B, -1, hidden_states.shape[-1])

        hidden_states = self.unpatchify(hidden_states, N_t, N_h, N_w)
        hidden_states = hidden_states.to(torch.float32)

        if return_kv:
            return hidden_states, kv_cache_dict_ret
        if not return_dict:
            return (hidden_states,)
        return Transformer2DModelOutput(sample=hidden_states)

    def unpatchify(self, x, N_t, N_h, N_w):
        T_p, H_p, W_p = self.patch_size
        x = x.view(
            x.shape[0],
            N_t,
            N_h,
            N_w,
            T_p,
            H_p,
            W_p,
            self.out_channels,
        )
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(x.shape[0], self.out_channels, N_t * T_p, N_h * H_p, N_w * W_p)
