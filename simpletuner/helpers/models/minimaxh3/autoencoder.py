# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import re
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.autoencoders.vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import logging
from diffusers.utils.accelerate_utils import apply_forward_hook

from .activations import MiniMaxH3FeedForward

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class MiniMaxH3VideoCausalConv3d(nn.Conv3d):
    r"""
    3D convolution used throughout the MiniMax-H3 video encoder.

    Spatial padding is symmetric and uses `spatial_padding_mode` (`"reflect"` in the released checkpoint); temporal
    padding is causal, i.e. `kernel_size_t - 1` zero frames are prepended and nothing is appended.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_padding > 0:
            padding = self.spatial_padding
            hidden_states = F.pad(hidden_states, (padding, padding, padding, padding, 0, 0), mode=self.spatial_padding_mode)
        if self.temporal_padding > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0), mode="constant")
        return F.conv3d(hidden_states, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)


class MiniMaxH3VideoGroupNorm(nn.GroupNorm):
    r"""
    Group normalization applied to each latent frame in isolation (`use_t_isolated_gn` in the original config): the
    temporal axis is folded into the batch axis so statistics never mix across frames.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        hidden_states = hidden_states.view(batch_size * num_frames, num_channels, 1, height, width)
        hidden_states = super().forward(hidden_states)
        hidden_states = hidden_states.view(batch_size, num_frames, num_channels, height, width)
        return hidden_states.permute(0, 2, 1, 3, 4).contiguous()


class MiniMaxH3VideoResnetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = MiniMaxH3VideoGroupNorm(norm_num_groups, in_channels, eps=norm_eps, affine=True)
        self.conv1 = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm2 = MiniMaxH3VideoGroupNorm(norm_num_groups, out_channels, eps=norm_eps, affine=True)
        self.conv2 = MiniMaxH3VideoCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = MiniMaxH3VideoCausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.silu(self.norm1(hidden_states))
        hidden_states = self.conv1(hidden_states)
        hidden_states = F.silu(self.norm2(hidden_states))
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


class MiniMaxH3VideoDownsample3d(nn.Module):
    r"""
    Strided 3x3x3 downsampling convolution. A spatial stride of 2 is preceded by an asymmetric bottom/right pad of 1
    (the convolution itself carries no spatial padding), so the output is exactly `ceil(size / 2)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            spatial_padding=0,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_stride == 2:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1, 0, 0), mode=self.spatial_padding_mode)
        return self.conv(hidden_states)


class MiniMaxH3VideoDownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        temporal_downsample_factor: int,
        spatial_downsample_factor: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                MiniMaxH3VideoResnetBlock3d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(num_layers)
            ]
        )
        self.downsamplers = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsamplers = nn.ModuleList(
                [
                    MiniMaxH3VideoDownsample3d(
                        out_channels,
                        out_channels,
                        temporal_stride=temporal_downsample_factor,
                        spatial_stride=spatial_downsample_factor,
                        spatial_padding_mode=spatial_padding_mode,
                    )
                ]
            )

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)
            else:
                hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


class MiniMaxH3VideoEncoder3d(nn.Module):
    r"""
    Causal 3D CNN encoder. `block_out_channels` gives the channel count of every level; the per-level
    `spatial_downsample_factors` / `temporal_downsample_factors` multiply out to the total compression ratios.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 48,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()

        self.conv_in = MiniMaxH3VideoCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoDownBlock3d(
                    in_channels=block_in_channels[i],
                    out_channels=block_out_channels[i],
                    num_layers=layers_per_block,
                    temporal_downsample_factor=temporal_downsample_factors[i],
                    spatial_downsample_factor=spatial_downsample_factors[i],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(len(block_out_channels))
            ]
        )

        self.norm_out = MiniMaxH3VideoGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps, affine=True)
        self.conv_out = MiniMaxH3VideoCausalConv3d(
            block_out_channels[-1],
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        hidden_states = F.silu(self.norm_out(hidden_states))
        return self.conv_out(hidden_states)


class MiniMaxH3VideoRotaryPosEmbed(nn.Module):
    r"""
    3-axis rotary embedding for the ViT decoder. Coordinates are length-normalized to `[-1, 1)` per axis and scaled by
    `2 * pi`, and the resulting `(t, h, w)` angles are concatenated and then duplicated, so the first
    `rope_dim_ratio * attention_head_dim` channels of every head are rotated.
    """

    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3) -> None:
        super().__init__()
        if dim % (2 * num_axes) != 0:
            raise ValueError(f"`dim` {dim} must be divisible by `2 * num_axes` {2 * num_axes}.")
        inv_freq = 1.0 / theta ** torch.arange(0, 1, 2 * num_axes / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


class MiniMaxH3VideoAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "MiniMaxH3VideoAttention",
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))

        # The reference normalizes Q/K in float32 regardless of the compute dtype.
        query = attn.norm_q(query.float()).to(query.dtype)
        key = attn.norm_k(key.float()).to(key.dtype)

        if rotary_emb is not None:
            cos, sin = rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            rotary_dim = cos.shape[-1]
            query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
            key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
            query_first, query_second = query_rotary.chunk(2, dim=-1)
            key_first, key_second = key_rotary.chunk(2, dim=-1)
            query_rotated = torch.cat([-query_second, query_first], dim=-1)
            key_rotated = torch.cat([-key_second, key_first], dim=-1)
            query = torch.cat([query_rotary * cos + query_rotated * sin, query_pass], dim=-1)
            key = torch.cat([key_rotary * cos + key_rotated * sin, key_pass], dim=-1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        return attn.to_out[0](hidden_states)


class MiniMaxH3VideoAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = MiniMaxH3VideoAttnProcessor
    _available_processors = [MiniMaxH3VideoAttnProcessor]

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.use_bias = bias
        inner_dim = heads * dim_head

        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=bias), nn.Dropout(0.0)])

        self.set_processor(MiniMaxH3VideoAttnProcessor())

    def forward(
        self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, rotary_emb)


class MiniMaxH3VideoTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ffn_mult: int = 4,
        eps: float = 1e-5,
        bias: bool = True,
        swiglu_gate_first: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = MiniMaxH3VideoAttention(dim=dim, heads=heads, dim_head=dim_head, eps=eps, bias=bias)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = MiniMaxH3FeedForward(dim, mult=ffn_mult, bias=bias, gate_first=swiglu_gate_first)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(
        self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        # The reference normalizes in float32 regardless of the compute dtype.
        norm_hidden_states = self.norm1(hidden_states.float()).to(hidden_states.dtype)
        scale1 = self.scale1.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = hidden_states + self.attn(norm_hidden_states, rotary_emb) * scale1
        norm_hidden_states = self.norm2(hidden_states.float()).to(hidden_states.dtype)
        scale2 = self.scale2.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = hidden_states + self.ff(norm_hidden_states) * scale2
        return hidden_states


class MiniMaxH3VideoViTDecoder3d(nn.Module):
    r"""
    Non-causal ViT decoder. Every latent voxel becomes one token; `num_register_tokens` learned register tokens plus a
    single all-zero token are appended (all at position `0`), attended over with full self-attention, and dropped
    again before the patch projection expands each token into a `patch_size_t x patch_size x patch_size` pixel block.
    """

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        patch_size_t: int = 4,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        swiglu_gate_first: bool = True,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.75,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.rope = MiniMaxH3VideoRotaryPosEmbed(int(attention_head_dim * rope_dim_ratio), theta=rope_theta)
        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoTransformerBlock(
                    dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    ffn_mult=ffn_mult,
                    eps=norm_eps,
                    swiglu_gate_first=swiglu_gate_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(batch_size, num_frames * height * width, num_channels)
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]

        register_tokens = self.register_tokens.to(device=hidden_states.device, dtype=hidden_states.dtype).expand(
            batch_size, -1, -1
        )
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        grids = [
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
            for size in (num_frames, height, width)
        ]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
        position_ids = torch.cat([position_ids, suffix_ids], dim=1)
        rotary_emb = self.rope(position_ids)

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, rotary_emb)
            else:
                hidden_states = block(hidden_states, rotary_emb)

        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states[:, :num_patches, :]

        patch_size, patch_size_t = self.patch_size, self.patch_size_t
        hidden_states = hidden_states.view(
            batch_size,
            num_frames,
            height,
            width,
            self.out_channels,
            patch_size_t,
            patch_size,
            patch_size,
        )
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden_states.reshape(
            batch_size,
            self.out_channels,
            num_frames * patch_size_t,
            height * patch_size,
            width * patch_size,
        )


_MINIMAX_H3_VAE_DEFAULT_SPATIAL_DOWNSAMPLE_FACTORS = (2, 2, 2, 2, 1, 1)
_MINIMAX_H3_VAE_DEFAULT_TEMPORAL_DOWNSAMPLE_FACTORS = (1, 2, 2, 1, 1, 1)


def _strip_minimax_h3_vae_checkpoint_prefix(key: str) -> str:
    for prefix in ("vae.", "first_stage_model.", "model.first_stage_model.", "model.vae."):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _map_minimax_h3_vae_comfy_key_to_diffusers(key: str) -> list[str]:
    from simpletuner.helpers.models.minimaxh3.transformer import _COMFY_QUANT_METADATA_SUFFIXES

    if key.endswith(_COMFY_QUANT_METADATA_SUFFIXES):
        return []
    if key in {"decoder.mask_token", "latents_mean", "latents_std", "pixel_mean", "pixel_std"}:
        return []
    if key == "decoder.pos_embed.inv_freq":
        return ["decoder.rope.inv_freq"]

    block_match = re.match(r"^encoder\.down\.(\d+)\.block\.(\d+)\.(.+)$", key)
    if block_match is not None:
        level, block, rest = block_match.groups()
        rest = rest.replace("nin_shortcut.", "conv_shortcut.", 1)
        return [f"encoder.down_blocks.{level}.resnets.{block}.{rest}"]

    downsample_match = re.match(r"^encoder\.down\.(\d+)\.downsample\.conv\.(.+)$", key)
    if downsample_match is not None:
        level, rest = downsample_match.groups()
        return [f"encoder.down_blocks.{level}.downsamplers.0.conv.{rest}"]

    key = key.replace("decoder.x_embedder.", "decoder.proj_in.", 1)
    key = re.sub(r"\.attn\.to_out\.(?!0\.)", ".attn.to_out.0.", key)
    key = key.replace(".ff.w1.", ".ff.net.0.proj.")
    key = key.replace(".ff.w2.", ".ff.net.2.")
    if key.endswith(".attn.to_qkv.weight"):
        base = key.removesuffix(".attn.to_qkv.weight")
        return [
            f"{base}.attn.to_q.weight",
            f"{base}.attn.to_k.weight",
            f"{base}.attn.to_v.weight",
        ]
    if key.endswith(".attn.to_qkv.bias"):
        base = key.removesuffix(".attn.to_qkv.bias")
        return [
            f"{base}.attn.to_q.bias",
            f"{base}.attn.to_k.bias",
            f"{base}.attn.to_v.bias",
        ]
    return [key]


def _count_minimax_h3_vae_indexed_blocks(keys: set[str], prefix: str) -> int:
    indices = set()
    for key in keys:
        if not key.startswith(prefix):
            continue
        index = key[len(prefix) :].split(".", 1)[0]
        if index.isdigit():
            indices.add(int(index))
    return max(indices) + 1 if indices else 0


def _get_minimax_h3_vae_checkpoint_tensor(checkpoint, stripped_key: str) -> torch.Tensor:
    for raw_key in checkpoint.keys():
        if _strip_minimax_h3_vae_checkpoint_prefix(raw_key) == stripped_key:
            return checkpoint.get_tensor(raw_key)
    raise RuntimeError(f"MiniMax-H3 VAE checkpoint is missing required tensor: {stripped_key}")


def _infer_minimax_h3_vae_config_from_checkpoint(checkpoint) -> dict[str, Any]:
    raw_keys = {_strip_minimax_h3_vae_checkpoint_prefix(key) for key in checkpoint.keys()}
    if "encoder.conv_in.weight" not in raw_keys:
        raise RuntimeError("MiniMax-H3 VAE single-file checkpoint does not contain recognized encoder keys.")

    conv_in = _get_minimax_h3_vae_checkpoint_tensor(checkpoint, "encoder.conv_in.weight")
    conv_out = _get_minimax_h3_vae_checkpoint_tensor(checkpoint, "encoder.conv_out.weight")
    post_quant = _get_minimax_h3_vae_checkpoint_tensor(checkpoint, "post_quant_conv.weight")
    decoder_proj_in_key = (
        "decoder.x_embedder.weight" if "decoder.x_embedder.weight" in raw_keys else "decoder.proj_in.weight"
    )
    decoder_proj_in = _get_minimax_h3_vae_checkpoint_tensor(checkpoint, decoder_proj_in_key)
    decoder_proj_out = _get_minimax_h3_vae_checkpoint_tensor(checkpoint, "decoder.proj_out.weight")

    block_out_channels = []
    level_count = _count_minimax_h3_vae_indexed_blocks(raw_keys, "encoder.down.")
    diffusers_level_count = _count_minimax_h3_vae_indexed_blocks(raw_keys, "encoder.down_blocks.")
    for index in range(max(level_count, diffusers_level_count)):
        comfy_key = f"encoder.down.{index}.block.0.conv2.weight"
        diffusers_key = f"encoder.down_blocks.{index}.resnets.0.conv2.weight"
        if comfy_key in raw_keys:
            block_out_channels.append(_get_minimax_h3_vae_checkpoint_tensor(checkpoint, comfy_key).shape[0])
        elif diffusers_key in raw_keys:
            block_out_channels.append(_get_minimax_h3_vae_checkpoint_tensor(checkpoint, diffusers_key).shape[0])
    if not block_out_channels:
        block_out_channels = [conv_in.shape[0]]

    layers_per_block = _count_minimax_h3_vae_indexed_blocks(raw_keys, "encoder.down.0.block.")
    if layers_per_block == 0:
        layers_per_block = _count_minimax_h3_vae_indexed_blocks(raw_keys, "encoder.down_blocks.0.resnets.")
    layers_per_block = layers_per_block or 2

    num_levels = len(block_out_channels)
    if num_levels == len(_MINIMAX_H3_VAE_DEFAULT_SPATIAL_DOWNSAMPLE_FACTORS):
        spatial_downsample_factors = _MINIMAX_H3_VAE_DEFAULT_SPATIAL_DOWNSAMPLE_FACTORS
        temporal_downsample_factors = _MINIMAX_H3_VAE_DEFAULT_TEMPORAL_DOWNSAMPLE_FACTORS
    else:
        spatial_downsample_factors = tuple(
            (
                2
                if (
                    f"encoder.down.{index}.downsample.conv.weight" in raw_keys
                    or f"encoder.down_blocks.{index}.downsamplers.0.conv.weight" in raw_keys
                )
                else 1
            )
            for index in range(num_levels)
        )
        temporal_downsample_factors = (1,) * num_levels

    decoder_hidden_size = decoder_proj_in.shape[0]
    if "decoder.transformer_blocks.0.attn.to_qkv.weight" in raw_keys:
        decoder_qkv = _get_minimax_h3_vae_checkpoint_tensor(checkpoint, "decoder.transformer_blocks.0.attn.to_qkv.weight")
        decoder_hidden_size = decoder_qkv.shape[1]
    elif "decoder.transformer_blocks.0.attn.to_q.weight" in raw_keys:
        decoder_hidden_size = _get_minimax_h3_vae_checkpoint_tensor(
            checkpoint, "decoder.transformer_blocks.0.attn.to_q.weight"
        ).shape[1]

    if decoder_hidden_size % 64 == 0:
        decoder_attention_head_dim = 64
    else:
        decoder_attention_head_dim = decoder_hidden_size
    decoder_num_attention_heads = decoder_hidden_size // decoder_attention_head_dim

    has_raw_swiglu = "decoder.transformer_blocks.0.ff.w1.weight" in raw_keys
    has_diffusers_swiglu = "decoder.transformer_blocks.0.ff.net.0.proj.weight" in raw_keys
    if has_raw_swiglu and has_diffusers_swiglu:
        raise RuntimeError("MiniMax-H3 VAE checkpoint mixes raw and Diffusers SwiGLU key layouts.")
    if has_raw_swiglu:
        ffn_weight = _get_minimax_h3_vae_checkpoint_tensor(checkpoint, "decoder.transformer_blocks.0.ff.w1.weight")
    else:
        ffn_weight = _get_minimax_h3_vae_checkpoint_tensor(checkpoint, "decoder.transformer_blocks.0.ff.net.0.proj.weight")
    decoder_ffn_mult = max(ffn_weight.shape[0] // (2 * decoder_hidden_size), 1)

    decoder_num_register_tokens = 4
    if "decoder.register_tokens" in raw_keys:
        decoder_num_register_tokens = _get_minimax_h3_vae_checkpoint_tensor(checkpoint, "decoder.register_tokens").shape[1]

    latent_channels = decoder_proj_in.shape[1]
    norm_num_groups = 32
    if any(channel % norm_num_groups != 0 for channel in block_out_channels):
        norm_num_groups = 1
    inferred_config: dict[str, Any] = {
        "in_channels": conv_in.shape[1],
        "out_channels": 3,
        "latent_channels": latent_channels,
        "block_out_channels": tuple(block_out_channels),
        "layers_per_block": layers_per_block,
        "spatial_downsample_factors": tuple(spatial_downsample_factors),
        "temporal_downsample_factors": tuple(temporal_downsample_factors),
        "norm_num_groups": norm_num_groups,
        "decoder_num_layers": _count_minimax_h3_vae_indexed_blocks(raw_keys, "decoder.transformer_blocks."),
        "decoder_num_attention_heads": decoder_num_attention_heads,
        "decoder_attention_head_dim": decoder_attention_head_dim,
        "decoder_num_register_tokens": decoder_num_register_tokens,
        "decoder_ffn_mult": decoder_ffn_mult,
        # The official Diffusers conversion swaps raw `[gate; up]` tensors to `[up; gate]`.
        "decoder_swiglu_gate_first": has_raw_swiglu,
    }
    if tuple(conv_out.shape[:2]) != (2 * latent_channels, block_out_channels[-1]):
        raise RuntimeError(
            "MiniMax-H3 VAE checkpoint uses unsupported encoder/decoder latent channel wiring: "
            f"encoder.conv_out has shape {tuple(conv_out.shape)}, decoder.proj_in has {latent_channels} channels."
        )
    if tuple(post_quant.shape[:2]) != (latent_channels, latent_channels):
        raise RuntimeError(
            "MiniMax-H3 VAE checkpoint uses unsupported post_quant_conv shape "
            f"{tuple(post_quant.shape)}; SimpleTuner expects matching embed/z channels."
        )
    if inferred_config["decoder_num_layers"] <= 0:
        raise RuntimeError("MiniMax-H3 VAE checkpoint does not contain decoder transformer blocks.")

    for config_key in ("latents_mean", "latents_std"):
        if config_key not in raw_keys:
            continue
        value = _get_minimax_h3_vae_checkpoint_tensor(checkpoint, config_key).to(torch.float32).flatten()
        if value.shape[0] != latent_channels:
            raise RuntimeError(
                f"MiniMax-H3 VAE checkpoint {config_key} has {value.shape[0]} values, expected {latent_channels}."
            )
        inferred_config[config_key] = tuple(float(item) for item in value.tolist())
    inferred_config.setdefault("latents_mean", (0.0,) * latent_channels)
    inferred_config.setdefault("latents_std", (1.0,) * latent_channels)

    # Comfy fixes the decoder output channels at RGB. Allow explicit test/config overrides for nonstandard fixtures.
    patch_volume = math.prod(inferred_config["spatial_downsample_factors"]) ** 2 * math.prod(
        inferred_config["temporal_downsample_factors"]
    )
    if decoder_proj_out.shape[0] % patch_volume == 0:
        inferred_config["out_channels"] = decoder_proj_out.shape[0] // patch_volume
    return inferred_config


def _set_minimax_h3_vae_module_buffer(root: nn.Module, buffer_name: str, value: torch.Tensor) -> None:
    module = root
    parts = buffer_name.split(".")
    for part in parts[:-1]:
        module = getattr(module, part)
    module._buffers[parts[-1]] = value


def _normalize_minimax_h3_vae_convrot_scale(scale: torch.Tensor, out_features: int, key: str) -> torch.Tensor:
    scale = scale.to(torch.float32)
    if scale.ndim == 0 or scale.numel() == 1:
        return scale.reshape(1, 1).expand(out_features, 1).contiguous()
    if tuple(scale.shape) == (out_features,):
        return scale.reshape(out_features, 1).contiguous()
    if tuple(scale.shape) == (out_features, 1):
        return scale.contiguous()
    raise RuntimeError(
        f"MiniMax-H3 VAE ConvRot tensor {key}_scale has shape {tuple(scale.shape)}, " f"expected {(out_features, 1)}."
    )


class AutoencoderKLMiniMaxH3(ModelMixin, ConfigMixin, AttentionMixin, AutoencoderMixin):
    r"""
    A VAE model with a causal 3D CNN encoder and a non-causal ViT decoder, used in
    [MiniMax-H3](https://huggingface.co/MiniMaxAI).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Latents are normalized with per-channel `latents_mean` / `latents_std` rather than a `scaling_factor`; a pipeline
    encodes with `(latent - latents_mean) / latents_std` and decodes with `latent * latents_std + latents_mean`.

    The pixel convention is ImageNet-normalized RGB over a `[0, 1]` base range, not the usual `[-1, 1]`: `encode`
    expects `(pixel - imagenet_mean) / imagenet_std` and `decode` returns values in that same space, so a pipeline has
    to apply `sample * imagenet_std + imagenet_mean` (mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`) and
    clamp to `[0, 1]` before postprocessing.

    The temporal geometry is fixed by `clip_length` (17 pixel frames per encoder chunk) and `token_drop` (3 trailing
    latent frames dropped per encode): `17 * n + 5` pixel frames map to `5 * n + 2` latent frames. A single pixel frame
    is true image mode and maps to one latent frame.

    Unlike most autoencoders in the library, spatial tiling is **on by default**: MiniMax-H3 was released with tiling
    enabled for both encoding and decoding, and the released frames are the blended-tile ones, so disabling tiling
    changes the output. Use `enable_tiling` to change the tile geometry, `disable_tiling` to turn it off.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxH3VideoResnetBlock3d", "MiniMaxH3VideoTransformerBlock"]
    _repeated_blocks = ["MiniMaxH3VideoTransformerBlock"]
    _skip_layerwise_casting_patterns = ["norm"]
    # The released checkpoint is float32 and the verified decode recipe is float16 *autocast over float32 weights*
    # (see `decode`). A pipeline-level `torch_dtype=torch.bfloat16` must therefore not downcast the weights, so every
    # top-level module is pinned, mirroring the transformer's mixed-precision contract.
    _keep_in_fp32_modules = ["encoder", "decoder", "quant_conv", "post_quant_conv"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 24,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
        decoder_num_layers: int = 36,
        decoder_num_attention_heads: int = 32,
        decoder_attention_head_dim: int = 64,
        decoder_num_register_tokens: int = 4,
        decoder_ffn_mult: int = 4,
        decoder_swiglu_gate_first: bool = False,
        decoder_rope_theta: float = 100.0,
        decoder_rope_dim_ratio: float = 0.75,
        decoder_norm_eps: float = 1e-5,
        clip_length: int = 17,
        token_drop: int = 3,
        latents_mean: tuple[float, ...] = (0.0,) * 24,
        latents_std: tuple[float, ...] = (1.0,) * 24,
    ) -> None:
        super().__init__()

        self.spatial_compression_ratio = math.prod(spatial_downsample_factors)
        self.temporal_compression_ratio = math.prod(temporal_downsample_factors)

        self.encoder = MiniMaxH3VideoEncoder3d(
            in_channels=in_channels,
            out_channels=2 * latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            spatial_downsample_factors=spatial_downsample_factors,
            temporal_downsample_factors=temporal_downsample_factors,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)
        self.decoder = MiniMaxH3VideoViTDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            patch_size=self.spatial_compression_ratio,
            patch_size_t=self.temporal_compression_ratio,
            num_layers=decoder_num_layers,
            num_attention_heads=decoder_num_attention_heads,
            attention_head_dim=decoder_attention_head_dim,
            num_register_tokens=decoder_num_register_tokens,
            ffn_mult=decoder_ffn_mult,
            swiglu_gate_first=decoder_swiglu_gate_first,
            rope_theta=decoder_rope_theta,
            rope_dim_ratio=decoder_rope_dim_ratio,
            norm_eps=decoder_norm_eps,
        )

        # Derived temporal-chunking geometry. `clip_length` pixel frames are encoded at a time; because
        # `clip_length` is not a multiple of `temporal_compression_ratio`, the decoder has to re-derive the
        # implicit leading pad (`frame_pre_padding`) and the overlap that `token_drop` leaves behind.
        self.frame_pre_padding = (-clip_length) % self.temporal_compression_ratio
        self.tokens_chunk_size = math.ceil(clip_length / self.temporal_compression_ratio)
        self.token_overlap = (-token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.temporal_compression_ratio - self.frame_pre_padding, 0)
        self.use_temporal_chunking = True

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When encoding/decoding spatially large videos, the memory requirement is very high. By splitting the frames
        # into smaller tiles, running the encoder/decoder per tile and blending the overlaps, the memory requirement
        # can be lowered. MiniMax-H3 ships with tiling enabled.
        self.use_tiling = True

        # The tile size in pixel space, and the minimum overlap between two neighbouring tiles. The actual overlaps are
        # widened (in multiples of `spatial_compression_ratio`) so that the tiles cover the frame exactly.
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64

    @classmethod
    def from_single_file(
        cls,
        pretrained_model_link_or_path: str,
        *args: Any,
        filename: str | None = None,
        subfolder: str | None = None,
        revision: str | None = None,
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> "AutoencoderKLMiniMaxH3":
        del args
        from simpletuner.helpers.models.minimaxh3.transformer import (
            _COMFY_QUANT_METADATA_SUFFIXES,
            _open_minimax_h3_single_file,
            _resolve_minimax_h3_single_file_path,
        )

        checkpoint_path = _resolve_minimax_h3_single_file_path(
            pretrained_model_link_or_path,
            filename=filename,
            subfolder=subfolder,
            revision=revision,
        )
        non_quantized_state_dict: dict[str, torch.Tensor] = {}
        quantized_weights: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}

        with _open_minimax_h3_single_file(checkpoint_path) as checkpoint:
            inferred_config = _infer_minimax_h3_vae_config_from_checkpoint(checkpoint)
            init_config = {**inferred_config, **kwargs}
            with torch.device("meta"):
                model = cls(**init_config)
            expected_state_dict = model.state_dict()
            checkpoint_keys = set(checkpoint.keys())

            for raw_key in checkpoint.keys():
                key = _strip_minimax_h3_vae_checkpoint_prefix(raw_key)
                if key.endswith(_COMFY_QUANT_METADATA_SUFFIXES):
                    continue
                mapped_keys = _map_minimax_h3_vae_comfy_key_to_diffusers(key)
                if not mapped_keys:
                    continue

                tensor = checkpoint.get_tensor(raw_key)
                if tensor.dtype == torch.int8:
                    from simpletuner.helpers.models.z_image.quantized_loading import _decode_comfy_quant

                    scale_key = f"{raw_key}_scale"
                    if scale_key not in checkpoint_keys:
                        raise RuntimeError(f"MiniMax-H3 VAE ConvRot tensor {raw_key} is missing weight_scale")
                    quant_key = f"{raw_key.removesuffix('.weight')}.comfy_quant"
                    if quant_key not in checkpoint_keys:
                        raise RuntimeError(f"MiniMax-H3 VAE ConvRot tensor {raw_key} is missing comfy_quant metadata")
                    quant_metadata = _decode_comfy_quant(checkpoint.get_tensor(quant_key))
                    if not quant_metadata.get("convrot", False):
                        raise RuntimeError(f"MiniMax-H3 VAE INT8 tensor {raw_key} is not marked as ConvRot")
                    hadamard_group_size = int(quant_metadata.get("convrot_groupsize", 0))
                    if hadamard_group_size <= 0:
                        raise RuntimeError(f"MiniMax-H3 VAE ConvRot tensor {raw_key} has invalid convrot_groupsize")

                    scale = checkpoint.get_tensor(scale_key)
                    if len(mapped_keys) == 3:
                        if tensor.shape[0] % 3 != 0:
                            raise RuntimeError(f"MiniMax-H3 VAE ConvRot tensor {raw_key} cannot split into q/k/v tensors")
                        scale = _normalize_minimax_h3_vae_convrot_scale(scale, tensor.shape[0], raw_key)
                        for mapped_key, qkv_tensor, qkv_scale in zip(
                            mapped_keys,
                            tensor.split(tensor.shape[0] // 3, dim=0),
                            scale.split(scale.shape[0] // 3, dim=0),
                        ):
                            quantized_weights[mapped_key] = (
                                qkv_tensor.contiguous(),
                                qkv_scale.contiguous(),
                                hadamard_group_size,
                            )
                    elif len(mapped_keys) == 1:
                        scale = _normalize_minimax_h3_vae_convrot_scale(scale, tensor.shape[0], raw_key)
                        quantized_weights[mapped_keys[0]] = (tensor.contiguous(), scale, hadamard_group_size)
                    else:
                        raise RuntimeError(f"MiniMax-H3 VAE ConvRot tensor {raw_key} maps to multiple targets unexpectedly")
                    continue

                if not torch.is_floating_point(tensor):
                    raise RuntimeError(
                        f"MiniMax-H3 VAE tensor {raw_key} has unsupported dtype {tensor.dtype}. "
                        "Only floating-point and INT8 ConvRot single-file VAE tensors are supported."
                    )

                if len(mapped_keys) == 3:
                    if tensor.shape[0] % 3 != 0:
                        raise RuntimeError(f"MiniMax-H3 VAE fused QKV tensor {raw_key} cannot split into q/k/v tensors")
                    for mapped_key, qkv_tensor in zip(mapped_keys, tensor.split(tensor.shape[0] // 3, dim=0)):
                        non_quantized_state_dict[mapped_key] = qkv_tensor.contiguous()
                elif len(mapped_keys) == 1:
                    non_quantized_state_dict[mapped_keys[0]] = tensor
                else:
                    raise RuntimeError(f"MiniMax-H3 VAE tensor {raw_key} maps to multiple targets unexpectedly")

        decoder_rope_inv_freq = non_quantized_state_dict.pop("decoder.rope.inv_freq", None)
        if decoder_rope_inv_freq is not None and tuple(decoder_rope_inv_freq.shape) != tuple(
            model.decoder.rope.inv_freq.shape
        ):
            raise RuntimeError(
                f"MiniMax-H3 VAE tensor decoder.rope.inv_freq has shape {tuple(decoder_rope_inv_freq.shape)}, "
                f"expected {tuple(model.decoder.rope.inv_freq.shape)}"
            )

        expected_quantized_keys = set(quantized_weights)
        keep_fp32_patterns = tuple(getattr(cls, "_keep_in_fp32_modules", ()))
        for key, tensor in list(non_quantized_state_dict.items()):
            if key not in expected_state_dict:
                raise RuntimeError(f"MiniMax-H3 VAE checkpoint has unexpected tensor: {key}")
            expected_tensor = expected_state_dict[key]
            if tuple(tensor.shape) != tuple(expected_tensor.shape):
                raise RuntimeError(
                    f"MiniMax-H3 VAE tensor {key} has shape {tuple(tensor.shape)}, expected {tuple(expected_tensor.shape)}"
                )
            if torch_dtype is not None and not any(pattern in key for pattern in keep_fp32_patterns):
                tensor = tensor.to(torch_dtype)
            non_quantized_state_dict[key] = tensor

        missing, unexpected = model.load_state_dict(non_quantized_state_dict, strict=False, assign=True)
        real_missing = [key for key in missing if key not in expected_quantized_keys]
        if real_missing or unexpected:
            raise RuntimeError(
                "MiniMax-H3 VAE checkpoint does not match autoencoder architecture. "
                f"Missing: {len(real_missing)}, Unexpected: {len(unexpected)}"
            )

        hadamard_group_sizes: set[int] = set()
        if quantized_weights:
            from simpletuner.helpers.models.z_image.quantized_loading import _wrap_convrot_linear

            for weight_key, (weight, scale, hadamard_group_size) in quantized_weights.items():
                if weight_key not in expected_state_dict:
                    raise RuntimeError(f"MiniMax-H3 VAE ConvRot checkpoint has unexpected tensor: {weight_key}")
                expected_tensor = expected_state_dict[weight_key]
                if tuple(weight.shape) != tuple(expected_tensor.shape):
                    raise RuntimeError(
                        f"MiniMax-H3 VAE ConvRot tensor {weight_key} has shape {tuple(weight.shape)}, "
                        f"expected {tuple(expected_tensor.shape)}"
                    )
                hadamard_group_sizes.add(hadamard_group_size)
                _wrap_convrot_linear(
                    model,
                    weight_key.removesuffix(".weight"),
                    weight,
                    scale,
                    result_dtype=torch_dtype or torch.bfloat16,
                    hadamard_group_size=hadamard_group_size,
                )
            if len(hadamard_group_sizes) != 1:
                raise RuntimeError(
                    "MiniMax-H3 VAE ConvRot checkpoint uses multiple Hadamard group sizes: "
                    f"{sorted(hadamard_group_sizes)}"
                )
            group_size = hadamard_group_sizes.pop()
            model.quantization_method = "minimax_h3_vae_comfy_convrot_sdnq"
            model.quantization_config = {
                "quant_method": "sdnq_training",
                "weights_dtype": "int8",
                "quantized_matmul_dtype": "int8",
                "use_hadamard": True,
                "hadamard_group_size": group_size,
                "group_size": -1,
                "source_format": "comfy_minimax_h3_vae_convrot",
            }

        if decoder_rope_inv_freq is not None:
            _set_minimax_h3_vae_module_buffer(model, "decoder.rope.inv_freq", decoder_rope_inv_freq.to(torch.float32))
        elif model.decoder.rope.inv_freq.is_meta:
            rope_dim = int(model.config.decoder_attention_head_dim * model.config.decoder_rope_dim_ratio)
            inv_freq = MiniMaxH3VideoRotaryPosEmbed(
                rope_dim,
                theta=getattr(model.config, "decoder_rope_theta", 100.0),
            ).inv_freq
            _set_minimax_h3_vae_module_buffer(model, "decoder.rope.inv_freq", inv_freq)

        return model

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_overlap_height: int | None = None,
        tile_sample_min_overlap_width: int | None = None,
    ) -> None:
        r"""
        Enable tiled VAE encoding/decoding. When this option is enabled, the VAE splits the frames into tiles, encodes
        or decodes each tile separately and linearly blends the overlaps back together. This lowers the memory
        requirement and allows processing larger frames.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The tile height in pixel space. Frames taller than this are split along the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The tile width in pixel space. Frames wider than this are split along the width dimension.
            tile_sample_min_overlap_height (`int`, *optional*):
                The minimum overlap, in pixels, between two consecutive vertical tiles.
            tile_sample_min_overlap_width (`int`, *optional*):
                The minimum overlap, in pixels, between two consecutive horizontal tiles.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_overlap_height = tile_sample_min_overlap_height or self.tile_sample_min_overlap_height
        self.tile_sample_min_overlap_width = tile_sample_min_overlap_width or self.tile_sample_min_overlap_width

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def enable_slicing(self) -> None:
        self.use_slicing = True

    def disable_slicing(self) -> None:
        self.use_slicing = False

    def enable_temporal_chunking(self) -> None:
        self.use_temporal_chunking = True

    def _split_tiles(self, length: int, tile_size: int, min_overlap: int) -> tuple[list[int], list[int], list[int]]:
        r"""
        Lay `tile_size`-wide tiles over `length` pixels. The number of tiles is the smallest one whose union can cover
        `length` while keeping every overlap at least `min_overlap`; the slack is then distributed round-robin over the
        overlaps in whole `spatial_compression_ratio` steps so that every tile boundary stays latent-aligned.
        """
        if tile_size >= length:
            return [0], [length], []

        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1

        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // self.spatial_compression_ratio):
            overlaps[i % (num_tiles - 1)] += self.spatial_compression_ratio

        tile_start_indices = [0]
        for i in range(num_tiles - 1):
            tile_start_indices.append(tile_start_indices[-1] + tile_size - overlaps[i])
        return tile_start_indices, [tile_size] * num_tiles, overlaps

    def _blend(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).view(shape)
        weight_b = (positions / blend_extent).view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b

        if blend_extent == b.shape[dim]:
            return blended
        slice_rest = [slice(None)] * b.ndim
        slice_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)

    def _stitch_tiles(
        self,
        tiles: list[list[torch.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> torch.Tensor:
        result_rows = []
        for i, row in enumerate(tiles):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    @apply_forward_hook
    def _encode_clip(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Encode one temporal clip, spatially tiled when tiling is enabled.

        MiniMax-H3 encodes a keyframe or an image reference through this method rather than through [`~encode`],
        because a single frame must not go through the temporal chunking, so it carries the offload hook too.
        """
        if not self.use_tiling:
            return self.quant_conv(self.encoder(x))

        height, width = x.shape[-2], x.shape[-1]
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )

        rows = []
        for i_pos, i_len in zip(y_indices, y_lengths):
            row = []
            for j_pos, j_len in zip(x_indices, x_lengths):
                tile = x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
                row.append(self.quant_conv(self.encoder(tile)))
            rows.append(row)

        latent_y_overlaps = [overlap // self.spatial_compression_ratio for overlap in y_overlaps]
        latent_x_overlaps = [overlap // self.spatial_compression_ratio for overlap in x_overlaps]
        return self._stitch_tiles(rows, latent_y_overlaps, latent_x_overlaps)

    def _decode_clip(self, z: torch.Tensor) -> torch.Tensor:
        r"""Decode one temporal clip, spatially tiled when tiling is enabled."""
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(z))

        # Tiles are laid out in pixel space and then mapped back onto the latent grid.
        height = z.shape[-2] * self.spatial_compression_ratio
        width = z.shape[-1] * self.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )

        ratio = self.spatial_compression_ratio
        rows = []
        for i_pos, i_len in zip(y_indices, y_lengths):
            row = []
            for j_pos, j_len in zip(x_indices, x_lengths):
                tile = z[
                    ...,
                    i_pos // ratio : i_pos // ratio + i_len // ratio,
                    j_pos // ratio : j_pos // ratio + j_len // ratio,
                ]
                row.append(self.decoder(self.post_quant_conv(tile)))
            rows.append(row)

        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    @apply_forward_hook
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Encode a video in `clip_length`-frame chunks and drop the `token_drop` trailing latent frames.

        MiniMax-H3 encodes a video reference through this method rather than through [`~encode`], because the
        posterior is sampled under a fixed generator rather than through the distribution object, so it carries the
        offload hook too.
        """
        clip_length = self.config.clip_length
        num_frames = x.shape[2]
        if num_frames % clip_length != 0:
            pad_frames = x[:, :, -1:].repeat(1, 1, (-num_frames) % clip_length, 1, 1)
            x = torch.cat([x, pad_frames], dim=2)

        moments = torch.cat(
            [self._encode_clip(x[:, :, i * clip_length : (i + 1) * clip_length]) for i in range(x.shape[2] // clip_length)],
            dim=2,
        )
        if self.config.token_drop > 0:
            moments = moments[:, :, : -self.config.token_drop]
        return moments

    def _encode_image_or_video(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] == 1:
            return self._encode_clip(x)[:, :, -1:, :, :]
        return self._encode(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        r"""
        Decode a latent video, mirroring the chunking that `_encode` applied.

        `token_drop` removed the tail of every encoded chunk, so consecutive decoded chunks overlap by
        `frame_overlap` pixel frames and are linearly cross-faded. Latent frames are repeated at the end when the
        length is not a whole number of chunks; the extra pixel frames are cut off again at the end.
        """
        if z.shape[2] == 1:
            return self._decode_clip(z)[:, :, -1:, :, :]

        tokens_chunk_size = self.tokens_chunk_size
        token_drop = self.config.token_drop
        temporal_ratio = self.temporal_compression_ratio
        chunk_num_frames = tokens_chunk_size * temporal_ratio

        num_tokens = z.shape[2] + token_drop
        pad_tokens = (-num_tokens) % tokens_chunk_size
        num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
        if pad_tokens > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

        decoded_chunks = []
        overlap = None
        for i in range(num_chunks):
            start = i * tokens_chunk_size
            clip = self._decode_clip(z[:, :, start : start + tokens_chunk_size + self.token_overlap])
            for j in range(int(token_drop > 0) + 1):
                frame_start = j * chunk_num_frames
                chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                chunk = chunk[:, :, self.frame_pre_padding :]
                if j == 0:
                    if overlap is not None:
                        chunk = self._blend(overlap, chunk, self.frame_overlap, dim=-3)
                    decoded_chunks.append(chunk)
                else:
                    overlap = chunk
        if overlap is not None:
            decoded_chunks.append(overlap)

        dec = torch.cat(decoded_chunks, dim=2)

        # `pad_tokens` repeated latent frames produced trailing pixel frames that were never requested. A chunk's
        # last latent frame only covers `clip_length % temporal_ratio` pixel frames, the others cover `temporal_ratio`.
        if pad_tokens > 0:
            intra_tail = self.config.clip_length % temporal_ratio
            num_tokens_before_pad = z.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
                for k in range(pad_tokens)
            )
            dec = dec[:, :, :-pad_frames]
        return dec

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput | tuple[torch.Tensor]:
        r"""
        Encode a batch of videos into latents.

        Args:
            x (`torch.Tensor`):
                Input batch of videos, shape `(batch_size, in_channels, num_frames, height, width)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.autoencoder_kl.AutoencoderKLOutput`] instead of a plain
                tuple.

        Returns:
            The latent distribution of the encoded videos. Note that MiniMax-H3 normalizes the encoded latents with
            `latents_mean` / `latents_std` afterwards.
        """
        if self.use_slicing and x.shape[0] > 1:
            moments = torch.cat([self._encode_image_or_video(x_slice) for x_slice in x.split(1)])
        else:
            moments = self._encode_image_or_video(x)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        r"""
        Decode a batch of latent videos.

        Args:
            z (`torch.Tensor`):
                Input batch of latent videos, shape `(batch_size, latent_channels, num_latent_frames, height, width)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                The decoded videos, shape `(batch_size, out_channels, num_frames, height, width)`.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded = torch.cat([self._decode(z_slice) for z_slice in z.split(1)])
        else:
            decoded = self._decode(z)
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        r"""
        Encode then decode a batch of videos.

        Args:
            sample (`torch.Tensor`):
                Input batch of videos, shape `(batch_size, in_channels, num_frames, height, width)`.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample the posterior instead of taking its mode.
            generator (`torch.Generator`, *optional*):
                Generator used when `sample_posterior=True`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                The round-tripped videos, shape `(batch_size, out_channels, num_frames, height, width)`.
        """
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(z, return_dict=return_dict)
