from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.modeling_utils import ModelMixin
from huggingface_hub import hf_hub_download

from simpletuner.helpers.musubi_block_swap import MusubiBlockSwapManager
from simpletuner.helpers.training.tread import TREADRouter

logger = logging.getLogger(__name__)


def _store_hidden_state(buffer, key: str, hidden_states: torch.Tensor):
    if buffer is not None:
        buffer[key] = hidden_states


def _get_1d_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.outer(pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_interpolated_pos_embed(
    embed_dim: int,
    grid_size: int,
    image_resolution: int,
    base_image_resolution: int = 256,
) -> np.ndarray:
    scale = float(base_image_resolution) / float(image_resolution)
    grid_h = np.arange(grid_size, dtype=np.float32) * scale
    grid_w = np.arange(grid_size, dtype=np.float32) * scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1).astype(np.float32)


def _get_rectangular_pos_embed(
    embed_dim: int,
    grid_height: int,
    grid_width: int,
    image_height: int,
    image_width: int,
    base_image_resolution: int = 256,
) -> np.ndarray:
    scale_h = float(base_image_resolution) / float(image_height)
    scale_w = float(base_image_resolution) / float(image_width)
    grid_h = np.arange(grid_height, dtype=np.float32) * scale_h
    grid_w = np.arange(grid_width, dtype=np.float32) * scale_w
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_height, grid_width])
    emb_h = _get_1d_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1).astype(np.float32)


def _default_rope_axes_dims(head_dim: int) -> tuple[int, int, int]:
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even for RoPE.")
    time_dim = head_dim // 2
    if time_dim % 2 != 0:
        time_dim -= 1
    remaining = head_dim - time_dim
    row_dim = remaining // 2
    col_dim = remaining - row_dim
    if row_dim % 2 != 0:
        row_dim -= 1
        col_dim += 1
    if col_dim % 2 != 0:
        col_dim -= 1
        row_dim += 1
    if min(time_dim, row_dim, col_dim) <= 0:
        raise ValueError("Each RoPE axis must receive at least two dimensions.")
    return time_dim, row_dim, col_dim


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        x_float = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * self.scale.float()).to(dtype)


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        mean = x_float.mean(dim=-1, keepdim=True)
        var = (x_float - mean).square().mean(dim=-1, keepdim=True)
        x_float = (x_float - mean) * torch.rsqrt(var + self.eps)
        return (x_float * self.scale.float() + self.bias.float()).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, hidden_size: int, in_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.linear1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.linear2(F.silu(self.linear1(x)))


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, hidden_features: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(hidden_size, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class MlpBlock(nn.Module):
    def __init__(self, hidden_size: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, qk_norm: bool, use_rmsnorm: bool) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        norm = RMSNorm if use_rmsnorm else LayerNorm
        self.q_norm = norm(self.head_dim) if qk_norm else None
        self.k_norm = norm(self.head_dim) if qk_norm else None
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        return self.proj(out)


class TextEncoderAdapterTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        drop_text_prob: float,
        num_heads: int,
        mlp_ratio: float,
        use_qknorm: bool,
        use_swiglu: bool,
        use_rmsnorm: bool,
        token_len: int,
    ) -> None:
        super().__init__()
        del drop_text_prob
        self.learnable_null_caption = nn.Parameter(torch.zeros(1, token_len, in_channels))
        self.connector_in = nn.Linear(in_channels, hidden_size)
        norm = RMSNorm if use_rmsnorm else LayerNorm
        self.connector_norm1 = norm(hidden_size)
        self.connector_norm2 = norm(hidden_size)
        self.connector_attn = Attention(hidden_size, num_heads, use_qknorm, use_rmsnorm)
        hidden_features = int(2 / 3 * int(hidden_size * mlp_ratio)) if use_swiglu else int(hidden_size * mlp_ratio)
        self.connector_mlp = (
            SwiGLUFFN(hidden_size, hidden_features) if use_swiglu else MlpBlock(hidden_size, hidden_features)
        )
        self.connector_norm3 = norm(hidden_size)
        self.connector_norm4 = norm(hidden_size)
        self.connector_attn2 = Attention(hidden_size, num_heads, use_qknorm, use_rmsnorm)
        self.connector_mlp2 = (
            SwiGLUFFN(hidden_size, hidden_features) if use_swiglu else MlpBlock(hidden_size, hidden_features)
        )

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        x = self.connector_in(caption)
        x = x + self.connector_attn(self.connector_norm1(x))
        x = x + self.connector_mlp(self.connector_norm2(x))
        x = x + self.connector_attn2(self.connector_norm3(x))
        return x + self.connector_mlp2(self.connector_norm4(x))


class MultimodalRopeEmbedder(nn.Module):
    def __init__(
        self,
        axes_dims: tuple[int, ...],
        axes_lens: tuple[int, ...],
        axes_scales: tuple[float, ...],
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.axes_dims = axes_dims
        self.theta = theta
        cos_tables = []
        sin_tables = []
        for dim, axis_len, axis_scale in zip(axes_dims, axes_lens, axes_scales):
            cos_table, sin_table = self._build_table(dim, axis_len, axis_scale, device=None)
            cos_tables.append(cos_table)
            sin_tables.append(sin_table)
        self.cos_tables = nn.ParameterList([nn.Parameter(t, requires_grad=False) for t in cos_tables])
        self.sin_tables = nn.ParameterList([nn.Parameter(t, requires_grad=False) for t in sin_tables])

    def _build_table(
        self,
        dim: int,
        axis_len: int,
        axis_scale: float,
        device: Optional[torch.device],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        steps = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
        base = 1.0 / (self.theta ** (steps / dim))
        positions = torch.arange(axis_len, dtype=torch.float32, device=device) * axis_scale
        angles = positions[:, None] * base[None, :]
        return angles.cos(), angles.sin()

    def forward(
        self,
        position_ids: torch.Tensor,
        axes_scales: Optional[tuple[float, ...]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = []
        sin = []
        if axes_scales is None:
            table_pairs = zip(self.cos_tables, self.sin_tables)
        else:
            max_positions = position_ids.amax(dim=(0, 1)).tolist()
            table_pairs = [
                self._build_table(
                    self.axes_dims[axis_idx],
                    int(max_positions[axis_idx]) + 1,
                    axes_scales[axis_idx],
                    position_ids.device,
                )
                for axis_idx in range(len(self.axes_dims))
            ]
        for axis_idx, (cos_table, sin_table) in enumerate(table_pairs):
            pos = position_ids[:, :, axis_idx].clamp(0, cos_table.shape[0] - 1)
            cos.append(cos_table[pos])
            sin.append(sin_table[pos])
        return torch.cat(cos, dim=-1), torch.cat(sin, dim=-1)


def _apply_multimodal_rope(
    x: torch.Tensor,
    freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if freqs is None:
        return x
    cos, sin = freqs
    dtype = x.dtype
    x_pair = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x0, x1 = x_pair.unbind(dim=-1)
    cos = cos[:, None].float()
    sin = sin[:, None].float()
    out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return out.reshape_as(x).to(dtype)


@dataclass(frozen=True)
class i1DiTForwardCache:
    text_tokens: torch.Tensor
    text_mask: Optional[torch.Tensor]
    image_freqs: tuple[torch.Tensor, torch.Tensor]
    text_freqs: tuple[torch.Tensor, torch.Tensor]


class MMDiTAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, qk_norm: bool, use_rmsnorm: bool) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_image = nn.Linear(hidden_size, 3 * hidden_size)
        self.qkv_text = nn.Linear(hidden_size, 3 * hidden_size)
        norm = RMSNorm if use_rmsnorm else LayerNorm
        self.q_norm = norm(self.head_dim) if qk_norm else None
        self.k_norm = norm(self.head_dim) if qk_norm else None
        self.proj_image = nn.Linear(hidden_size, hidden_size)
        self.proj_text = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
        text_freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
        text_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, image_len, _ = image_tokens.shape
        text_len = text_tokens.shape[1]

        def project(linear: nn.Linear, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            qkv = linear(x).reshape(bsz, x.shape[1], 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q_image, k_image, v_image = project(self.qkv_image, image_tokens)
        q_text, k_text, v_text = project(self.qkv_text, text_tokens)
        if self.q_norm is not None:
            q_image = self.q_norm(q_image)
            k_image = self.k_norm(k_image)
            q_text = self.q_norm(q_text)
            k_text = self.k_norm(k_text)
        q_image = _apply_multimodal_rope(q_image, image_freqs)
        k_image = _apply_multimodal_rope(k_image, image_freqs)
        q_text = _apply_multimodal_rope(q_text, text_freqs)
        k_text = _apply_multimodal_rope(k_text, text_freqs)
        q = torch.cat([q_image, q_text], dim=2)
        k = torch.cat([k_image, k_text], dim=2)
        v = torch.cat([v_image, v_text], dim=2)
        key_mask = None
        attn_mask = None
        if text_mask is not None:
            image_mask = torch.ones((bsz, image_len), dtype=torch.bool, device=text_tokens.device)
            key_mask = torch.cat([image_mask, text_mask.bool()], dim=1)
            attn_mask = key_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(bsz, image_len + text_len, self.hidden_size)
        if key_mask is not None:
            out = out * key_mask[:, :, None].to(out.dtype)
        return self.proj_image(out[:, :image_len]), self.proj_text(out[:, image_len:])


class i1DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        use_qknorm: bool,
        use_swiglu: bool,
        use_rmsnorm: bool,
        use_skip: bool = False,
    ) -> None:
        super().__init__()
        self.use_skip = use_skip
        if use_skip:
            self.skip_linear_image = nn.Linear(2 * hidden_size, hidden_size)
            self.skip_linear_text = nn.Linear(2 * hidden_size, hidden_size)
        norm = RMSNorm if use_rmsnorm else LayerNorm
        self.norm1 = norm(hidden_size)
        self.norm2 = norm(hidden_size)
        self.norm3 = norm(hidden_size)
        self.norm4 = norm(hidden_size)
        self.attn = MMDiTAttention(hidden_size, num_heads, use_qknorm, use_rmsnorm)
        hidden_features = int(2 / 3 * int(hidden_size * mlp_ratio)) if use_swiglu else int(hidden_size * mlp_ratio)
        self.mlp_image = SwiGLUFFN(hidden_size, hidden_features) if use_swiglu else MlpBlock(hidden_size, hidden_features)
        self.mlp_text = SwiGLUFFN(hidden_size, hidden_features) if use_swiglu else MlpBlock(hidden_size, hidden_features)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
        text_freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
        text_mask: Optional[torch.Tensor],
        skip: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_skip:
            if skip is None:
                raise ValueError("Skip connection is required.")
            image_tokens = self.skip_linear_image(torch.cat([image_tokens, skip[0]], dim=-1))
            text_tokens = self.skip_linear_text(torch.cat([text_tokens, skip[1]], dim=-1))
        image_attn, text_attn = self.attn(
            self.norm1(image_tokens),
            self.norm1(text_tokens),
            image_freqs,
            text_freqs,
            text_mask,
        )
        image_tokens = image_tokens + self.norm3(image_attn)
        text_tokens = text_tokens + self.norm3(text_attn)
        image_tokens = image_tokens + self.norm4(self.mlp_image(self.norm2(image_tokens)))
        text_tokens = text_tokens + self.norm4(self.mlp_text(self.norm2(text_tokens)))
        if text_mask is not None:
            text_tokens = text_tokens * text_mask[:, :, None].to(text_tokens.dtype)
        return image_tokens, text_tokens


class FinalLayerNoAdaLN(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, use_rmsnorm: bool) -> None:
        super().__init__()
        norm = RMSNorm if use_rmsnorm else LayerNorm
        self.norm_final = norm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))


class ZlabI1Transformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True
    _tread_router: Optional[TREADRouter] = None
    _tread_routes: Optional[list[dict[str, Any]]] = None
    _cp_plan = {
        "x_embedder": {
            0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
        },
        "rope_embedder": {
            0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        },
        "final_layer": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(
        self,
        input_size: int = 1024 // 8,
        image_resolution: int = 1024,
        patch_size: int = 2,
        in_channels: int = 32,
        hidden_size: int = 2016,
        depth: int = 29,
        num_heads: int = 28,
        head_dim: Optional[int] = None,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 2304,
        text_num_tokens: int = 256,
        rope_theta: float = 10000.0,
        musubi_blocks_to_swap: int = 0,
        musubi_block_swap_device: str = "cpu",
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads, got hidden_size={hidden_size}, num_heads={num_heads}."
            )
        resolved_head_dim = hidden_size // num_heads
        if head_dim is not None and head_dim != resolved_head_dim:
            raise ValueError(f"head_dim must equal hidden_size // num_heads ({resolved_head_dim}), got {head_dim}.")
        if patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}.")
        if input_size % patch_size != 0:
            raise ValueError(
                f"input_size must be divisible by patch_size, got input_size={input_size}, patch_size={patch_size}."
            )
        self.register_to_config(head_dim=resolved_head_dim)
        self.head_dim = resolved_head_dim
        self.gradient_checkpointing = False
        self.input_size = input_size
        self.image_resolution = image_resolution
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.x_embedder = PatchEmbed(patch_size, hidden_size, in_channels)
        hw = input_size // patch_size
        self.hw = hw
        pos = _get_interpolated_pos_embed(hidden_size, hw, image_resolution)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos.reshape(1, hw * hw, hidden_size)))
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_embedder.requires_grad_(False)
        self.text_encoder_adapter = TextEncoderAdapterTransformer(
            text_embed_dim,
            hidden_size,
            0.1,
            num_heads,
            mlp_ratio,
            True,
            True,
            True,
            text_num_tokens,
        )
        axes_dims = _default_rope_axes_dims(resolved_head_dim)
        axes_lens = (text_num_tokens + 1, hw, hw)
        image_scale = 256.0 / image_resolution
        self.rope_embedder = MultimodalRopeEmbedder(
            axes_dims,
            axes_lens,
            (1.0, image_scale, image_scale),
            theta=rope_theta,
        )
        self.register_buffer("image_row_ids", torch.repeat_interleave(torch.arange(hw), hw), persistent=False)
        self.register_buffer("image_col_ids", torch.tile(torch.arange(hw), (hw,)), persistent=False)
        num_in_blocks = depth // 2
        self.in_blocks = nn.ModuleList(
            [
                i1DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    True,
                    True,
                    True,
                )
                for _ in range(num_in_blocks)
            ]
        )
        self.mid_block = i1DiTBlock(
            hidden_size,
            num_heads,
            mlp_ratio,
            True,
            True,
            True,
        )
        self.out_blocks = nn.ModuleList(
            [
                i1DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    True,
                    True,
                    True,
                    use_skip=True,
                )
                for _ in range(num_in_blocks)
            ]
        )
        self.final_layer = FinalLayerNoAdaLN(
            hidden_size,
            patch_size,
            self.out_channels,
            True,
        )
        self._musubi_block_swap = MusubiBlockSwapManager.build(
            depth=depth,
            blocks_to_swap=musubi_blocks_to_swap,
            swap_device=musubi_block_swap_device,
            logger=logger,
        )

    def set_router(self, router: TREADRouter, routes: list[dict[str, Any]]):
        self._tread_router = router
        self._tread_routes = routes

    def _image_position_ids(
        self,
        image_token_height: int,
        image_token_width: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if image_token_height == self.hw and image_token_width == self.hw:
            return self.image_row_ids, self.image_col_ids
        row_ids = torch.repeat_interleave(torch.arange(image_token_height, device=device), image_token_width)
        col_ids = torch.tile(torch.arange(image_token_width, device=device), (image_token_height,))
        return row_ids, col_ids

    def _position_axes_scales(
        self,
        image_token_height: int,
        image_token_width: int,
    ) -> Optional[tuple[float, float, float]]:
        if image_token_height == self.hw and image_token_width == self.hw:
            return None
        image_height = image_token_height * self.patch_size * 8
        image_width = image_token_width * self.patch_size * 8
        return (1.0, 256.0 / image_height, 256.0 / image_width)

    def _position_embedding(self, token_height: int, token_width: int, x: torch.Tensor) -> torch.Tensor:
        if token_height == self.hw and token_width == self.hw:
            return self.pos_embed.to(dtype=x.dtype, device=x.device)
        image_height = token_height * self.patch_size * 8
        image_width = token_width * self.patch_size * 8
        pos = _get_rectangular_pos_embed(
            self.pos_embed.shape[-1],
            token_height,
            token_width,
            image_height,
            image_width,
        )
        return torch.from_numpy(pos.reshape(1, token_height * token_width, self.pos_embed.shape[-1])).to(
            device=x.device,
            dtype=x.dtype,
        )

    def _build_position_ids(
        self,
        text_mask: torch.Tensor,
        text_lengths: torch.Tensor,
        image_token_height: int,
        image_token_width: int,
    ) -> torch.Tensor:
        bsz, text_len = text_mask.shape
        num_image_tokens = image_token_height * image_token_width
        caption_positions = torch.arange(text_len, dtype=torch.long, device=text_mask.device)[None].expand(bsz, text_len)
        caption_positions = torch.where(text_mask.bool(), caption_positions, torch.zeros_like(caption_positions))
        zeros = torch.zeros_like(caption_positions)
        caption_ids = torch.stack((caption_positions, zeros, zeros), dim=-1)
        image_row_ids, image_col_ids = self._image_position_ids(image_token_height, image_token_width, text_mask.device)
        row_ids = image_row_ids[None].expand(bsz, num_image_tokens)
        col_ids = image_col_ids[None].expand(bsz, num_image_tokens)
        image_time = text_lengths[:, None].expand(bsz, num_image_tokens)
        image_ids = torch.stack((image_time, row_ids, col_ids), dim=-1)
        return torch.cat([caption_ids, image_ids], dim=1)

    def prepare_forward_cache(
        self,
        caption: torch.Tensor,
        mask: Optional[torch.Tensor],
        image_token_height: int,
        image_token_width: int,
    ) -> i1DiTForwardCache:
        text_tokens = self.text_encoder_adapter(caption)
        text_mask = mask.bool() if mask is not None else None
        seq_text = text_tokens.shape[1]
        pos_mask = (
            text_mask
            if text_mask is not None
            else torch.ones((text_tokens.shape[0], seq_text), dtype=torch.bool, device=text_tokens.device)
        )
        text_lengths = pos_mask.to(torch.int32).sum(dim=1)
        num_image_tokens = image_token_height * image_token_width
        position_ids = self._build_position_ids(pos_mask, text_lengths, image_token_height, image_token_width)
        cos, sin = self.rope_embedder(
            position_ids, axes_scales=self._position_axes_scales(image_token_height, image_token_width)
        )
        text_freqs = (cos[:, :seq_text], sin[:, :seq_text])
        image_freqs = (cos[:, seq_text : seq_text + num_image_tokens], sin[:, seq_text : seq_text + num_image_tokens])
        return i1DiTForwardCache(text_tokens, text_mask, image_freqs, text_freqs)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        caption: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        forward_cache: Optional[i1DiTForwardCache] = None,
        skip_layers: Optional[list[int]] = None,
        force_keep_mask: Optional[torch.Tensor] = None,
        hidden_states_buffer: Optional[dict] = None,
    ) -> torch.Tensor:
        del t
        if x.shape[-2] % self.patch_size != 0 or x.shape[-1] % self.patch_size != 0:
            raise ValueError(
                f"i1 latent height and width must be divisible by patch_size={self.patch_size}, got {tuple(x.shape[-2:])}."
            )
        token_height = x.shape[-2] // self.patch_size
        token_width = x.shape[-1] // self.patch_size
        tokens = self.x_embedder(x) + self._position_embedding(token_height, token_width, x)
        cache = (
            forward_cache
            if forward_cache is not None
            else self.prepare_forward_cache(caption, mask, token_height, token_width)
        )
        text_tokens = cache.text_tokens
        text_mask = cache.text_mask
        text_freqs = cache.text_freqs
        image_freqs = cache.image_freqs
        image_tokens = tokens
        skips = []

        routes = self._tread_routes or []
        router = self._tread_router
        use_routing = self.training and len(routes) > 0 and torch.is_grad_enabled()
        if use_routing and router is None:
            raise ValueError("TREAD routing requested but no router has been configured. Call set_router before training.")
        if routes:
            total_layers = len(self.in_blocks) + 1 + len(self.out_blocks)

            def _to_pos(idx):
                return idx if idx >= 0 else total_layers + idx

            routes = [
                {
                    **route,
                    "start_layer_idx": _to_pos(route["start_layer_idx"]),
                    "end_layer_idx": _to_pos(route["end_layer_idx"]),
                }
                for route in routes
            ]

        route_ptr = 0
        routing_now = False
        tread_mask_info = None
        saved_image_tokens = None
        saved_image_freqs = None
        skip_set = set(skip_layers) if skip_layers is not None else set()
        global_idx = 0

        def maybe_start_route():
            nonlocal routing_now, tread_mask_info, saved_image_tokens, saved_image_freqs, image_tokens, image_freqs
            if not (use_routing and route_ptr < len(routes) and global_idx == routes[route_ptr]["start_layer_idx"]):
                return
            mask_ratio = routes[route_ptr]["selection_ratio"]
            tread_mask_info = router.get_mask(image_tokens, mask_ratio=mask_ratio, force_keep=force_keep_mask)
            saved_image_tokens = image_tokens.clone()
            saved_image_freqs = (image_freqs[0].clone(), image_freqs[1].clone())
            image_tokens = router.start_route(image_tokens, tread_mask_info)
            image_freqs = (
                router.start_route(image_freqs[0], tread_mask_info),
                router.start_route(image_freqs[1], tread_mask_info),
            )
            routing_now = True

        def maybe_end_route():
            nonlocal route_ptr, routing_now, image_tokens, image_freqs
            if not (routing_now and route_ptr < len(routes) and global_idx == routes[route_ptr]["end_layer_idx"]):
                return
            image_tokens = router.end_route(image_tokens, tread_mask_info, original_x=saved_image_tokens)
            image_freqs = (
                router.end_route(image_freqs[0], tread_mask_info, original_x=saved_image_freqs[0]),
                router.end_route(image_freqs[1], tread_mask_info, original_x=saved_image_freqs[1]),
            )
            routing_now = False
            route_ptr += 1

        def full_image_tokens_for_storage():
            if routing_now and tread_mask_info is not None and saved_image_tokens is not None:
                return router.end_route(image_tokens, tread_mask_info, original_x=saved_image_tokens)
            return image_tokens

        def run_block(block: i1DiTBlock, skip: Optional[tuple[torch.Tensor, torch.Tensor]] = None):
            block_skip = skip
            if (
                block_skip is not None
                and routing_now
                and tread_mask_info is not None
                and block_skip[0].shape[1] != image_tokens.shape[1]
            ):
                block_skip = (router.start_route(block_skip[0], tread_mask_info), block_skip[1])
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                inputs = (
                    block,
                    image_tokens,
                    text_tokens,
                    image_freqs,
                    text_freqs,
                    text_mask,
                )
                if block_skip is not None:
                    inputs = (*inputs, block_skip)
                return self._gradient_checkpointing_func(*inputs)
            return block(image_tokens, text_tokens, image_freqs, text_freqs, text_mask, block_skip)

        combined_blocks = list(self.in_blocks) + [self.mid_block] + list(self.out_blocks)
        musubi_manager = self._musubi_block_swap
        musubi_offload_active = False
        if musubi_manager is not None:
            musubi_offload_active = musubi_manager.activate(combined_blocks, x.device, torch.is_grad_enabled())

        for block in self.in_blocks:
            if musubi_offload_active and musubi_manager.is_managed_block(global_idx):
                musubi_manager.stream_in(block, x.device)
            maybe_start_route()
            if global_idx not in skip_set:
                image_tokens, text_tokens = run_block(block)
            image_tokens_for_storage = full_image_tokens_for_storage()
            skips.append((image_tokens_for_storage, text_tokens))
            maybe_end_route()
            _store_hidden_state(hidden_states_buffer, f"layer_{global_idx}", image_tokens_for_storage)
            if musubi_offload_active and musubi_manager.is_managed_block(global_idx):
                musubi_manager.stream_out(block)
            global_idx += 1

        if musubi_offload_active and musubi_manager.is_managed_block(global_idx):
            musubi_manager.stream_in(self.mid_block, x.device)
        maybe_start_route()
        if global_idx not in skip_set:
            image_tokens, text_tokens = run_block(self.mid_block)
        image_tokens_for_storage = full_image_tokens_for_storage()
        maybe_end_route()
        _store_hidden_state(hidden_states_buffer, f"layer_{global_idx}", image_tokens_for_storage)
        if musubi_offload_active and musubi_manager.is_managed_block(global_idx):
            musubi_manager.stream_out(self.mid_block)
        global_idx += 1

        for block in self.out_blocks:
            if musubi_offload_active and musubi_manager.is_managed_block(global_idx):
                musubi_manager.stream_in(block, x.device)
            maybe_start_route()
            skip = skips.pop()
            if global_idx not in skip_set:
                image_tokens, text_tokens = run_block(block, skip)
            image_tokens_for_storage = full_image_tokens_for_storage()
            maybe_end_route()
            _store_hidden_state(hidden_states_buffer, f"layer_{global_idx}", image_tokens_for_storage)
            if musubi_offload_active and musubi_manager.is_managed_block(global_idx):
                musubi_manager.stream_out(block)
            global_idx += 1
        tokens = self.final_layer(image_tokens)
        bsz = x.shape[0]
        h = token_height
        w = token_width
        p = self.patch_size
        tokens = tokens.reshape(bsz, h, w, p, p, self.out_channels)
        tokens = tokens.permute(0, 1, 3, 2, 4, 5).reshape(bsz, h * p, w * p, self.out_channels)
        image = tokens.permute(0, 3, 1, 2)
        return image

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, subfolder: Optional[str] = None, **kwargs):
        torch_dtype = kwargs.pop("torch_dtype", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        variant = kwargs.pop("variant", None)
        revision = kwargs.pop("revision", None)
        local_files_only = kwargs.pop("local_files_only", None)
        cache_dir = kwargs.pop("cache_dir", None)
        musubi_blocks_to_swap = kwargs.pop("musubi_blocks_to_swap", 0)
        musubi_block_swap_device = kwargs.pop("musubi_block_swap_device", "cpu")

        if os.path.isfile(pretrained_model_name_or_path):
            checkpoint_path = pretrained_model_name_or_path
        elif pretrained_model_name_or_path == "zlab-princeton/i1-3B":
            checkpoint_path = hf_hub_download(
                repo_id=pretrained_model_name_or_path,
                filename="1024_resolution_checkpoint_torch.pt",
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=bool(local_files_only) if local_files_only is not None else False,
            )
        else:
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *args,
                subfolder=subfolder,
                torch_dtype=torch_dtype,
                use_safetensors=use_safetensors,
                variant=variant,
                revision=revision,
                local_files_only=local_files_only,
                cache_dir=cache_dir,
                musubi_blocks_to_swap=musubi_blocks_to_swap,
                musubi_block_swap_device=musubi_block_swap_device,
                **kwargs,
            )

        model = cls(
            musubi_blocks_to_swap=musubi_blocks_to_swap,
            musubi_block_swap_device=musubi_block_swap_device,
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(state_dict, dict) and "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        if torch_dtype is not None:
            model.to(dtype=torch_dtype)
        return model


FLUX2_LATENTS_MEAN = [
    -0.06761776655912399,
    -0.07152235507965088,
    -0.07534133642911911,
    -0.07449393719434738,
    0.022278539836406708,
    0.017995379865169525,
    0.014197370037436485,
    0.01836133562028408,
    -6.275518535403535e-05,
    -0.006251443177461624,
    -0.00021015340462327003,
    -0.0031394739635288715,
    -0.027202727273106575,
    -0.02810601517558098,
    -0.027645578607916832,
    -0.029033277183771133,
    -0.0768895298242569,
    -0.06717019528150558,
    -0.09018829464912415,
    -0.08921381831169128,
    0.016836659982800484,
    0.015206480398774147,
    0.00790204294025898,
    0.008579261600971222,
    0.008347982540726662,
    0.0015409095212817192,
    0.0002583497844170779,
    -0.004281752277165651,
    -0.043877143412828445,
    -0.04189559817314148,
    -0.04378034919500351,
    -0.043148837983608246,
    -0.010246668942272663,
    -0.013186423107981682,
    -0.006620197091251612,
    -0.004766239318996668,
    -0.031062893569469452,
    -0.03055436909198761,
    -0.027904054149985313,
    -0.01795399747788906,
    0.0030211929697543383,
    0.001502539962530136,
    0.012592565268278122,
    0.0144742326810956,
    0.034720875322818756,
    0.03376586362719536,
    0.033663298934698105,
    0.02829528972506523,
    0.0019797170534729958,
    0.004728920292109251,
    0.004654144402593374,
    0.004963618237525225,
    0.012272646650671959,
    0.008096166886389256,
    0.00805679615586996,
    0.014576919376850128,
    0.06810732930898666,
    0.06790295243263245,
    0.07665354013442993,
    0.07318653911352158,
    -0.04621443152427673,
    -0.04739413782954216,
    -0.03918757662177086,
    -0.05109340697526932,
    -0.05277586728334427,
    -0.04773825407028198,
    -0.047003958374261856,
    -0.0517151840031147,
    -0.03170523792505264,
    -0.03163386881351471,
    -0.03446723148226738,
    -0.02825590781867504,
    0.050968676805496216,
    0.04450491443276405,
    0.057813018560409546,
    0.04580356180667877,
    -0.0411602221429348,
    -0.04582904279232025,
    -0.048741210252046585,
    -0.04673927649855614,
    -0.008838738314807415,
    -0.010627646930515766,
    -0.008805501274764538,
    -0.004613492637872696,
    -0.03758484125137329,
    -0.043219830840826035,
    -0.043574366718530655,
    -0.049890533089637756,
    0.011846445500850677,
    0.016636915504932404,
    0.020284568890929222,
    0.027899663895368576,
    0.011271224357187748,
    0.01290129590779543,
    0.0015593513380736113,
    0.007155619561672211,
    -0.01180021371692419,
    -0.0018362690461799502,
    -0.014141527935862541,
    -0.005370706785470247,
    -0.009097136557102203,
    -0.013795508071780205,
    -0.014467928558588028,
    -0.01869881898164749,
    0.03225415572524071,
    0.030501458793878555,
    0.02587026357650757,
    0.02995659038424492,
    0.05399540066719055,
    0.06144390255212784,
    0.049539074301719666,
    0.05898929387331009,
    -0.051080696284770966,
    -0.06032619997859001,
    -0.047775182873010635,
    -0.052397292107343674,
    -0.022676242515444756,
    -0.027419250458478928,
    -0.015365149825811386,
    -0.025462470948696136,
    -0.05720777437090874,
    -0.056476689875125885,
    -0.05176353082060814,
    -0.049556463956832886,
    0.011585467495024204,
    0.0054222596809268,
    0.01630038022994995,
    0.010384724475443363,
]
FLUX2_LATENTS_VAR = [
    3.2502119541168213,
    3.163407325744629,
    3.192434072494507,
    3.1813714504241943,
    3.1389076709747314,
    3.0941381454467773,
    3.1011831760406494,
    3.0550901889801025,
    3.0051753520965576,
    3.0179455280303955,
    3.0067572593688965,
    3.0076351165771484,
    3.4690163135528564,
    3.432523727416992,
    3.470231533050537,
    3.45538592338562,
    3.0949840545654297,
    3.071377754211426,
    3.0819239616394043,
    3.091344118118286,
    3.014709711074829,
    3.027461051940918,
    3.01198673248291,
    3.0252928733825684,
    3.0074563026428223,
    2.9741339683532715,
    3.024878978729248,
    2.9940483570098877,
    3.080418586730957,
    3.0669093132019043,
    3.0831477642059326,
    3.058147430419922,
    3.403618097305298,
    3.4055330753326416,
    3.44087290763855,
    3.435497283935547,
    3.326714277267456,
    3.1730010509490967,
    3.1874520778656006,
    3.22017240524292,
    3.2569847106933594,
    3.1953234672546387,
    3.130955457687378,
    3.124211549758911,
    3.1620266437530518,
    3.1209557056427,
    3.2129595279693604,
    3.185375690460205,
    3.090271472930908,
    3.030029058456421,
    3.0565788745880127,
    3.0162465572357178,
    3.225846767425537,
    3.2391276359558105,
    3.211076259613037,
    3.21309494972229,
    3.161032199859619,
    3.149500846862793,
    3.142376184463501,
    3.150174379348755,
    3.071641206741333,
    3.0439963340759277,
    3.1177477836608887,
    3.0607917308807373,
    3.1593689918518066,
    3.139946222305298,
    3.1729917526245117,
    3.1730189323425293,
    3.2984564304351807,
    3.244508981704712,
    3.248305559158325,
    3.251725673675537,
    3.0720319747924805,
    3.00360369682312,
    3.084465742111206,
    3.056194543838501,
    3.100954532623291,
    3.064960479736328,
    3.1261374950408936,
    3.102006435394287,
    3.120508909225464,
    3.0782599449157715,
    3.178100109100342,
    3.141893148422241,
    3.2024238109588623,
    3.2396669387817383,
    3.1909685134887695,
    3.1540026664733887,
    3.102187395095825,
    3.106377601623535,
    3.08341121673584,
    3.0892975330352783,
    3.1621134281158447,
    3.1226611137390137,
    3.1719861030578613,
    3.168121337890625,
    2.958735942840576,
    2.9129180908203125,
    2.980844497680664,
    2.9209375381469727,
    3.165689706802368,
    3.08971905708313,
    3.0632121562957764,
    3.0465474128723145,
    3.0928444862365723,
    3.0622732639312744,
    3.0709831714630127,
    3.014193534851074,
    3.103145122528076,
    3.087780714035034,
    3.042872667312622,
    3.0380074977874756,
    3.065497875213623,
    3.10084867477417,
    3.109544038772583,
    3.101743698120117,
    2.976869583129883,
    2.935845136642456,
    2.999986171722412,
    2.9673469066619873,
    3.1200692653656006,
    3.105872631072998,
    3.139338493347168,
    3.12007999420166,
    3.0474750995635986,
    3.0419390201568604,
    3.086534261703491,
    3.072920083999634,
]
