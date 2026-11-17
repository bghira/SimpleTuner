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

import torch
import torch.nn as nn
from diffusers.configuration_utils import register_to_config
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin, get_parameter_dtype
from diffusers.utils.accelerate_utils import apply_forward_hook

from .autoencoder import AutoencoderKLMiniMaxH3
from .autoencoder import MiniMaxH3VideoAttention as MiniMaxH3ImageAttention
from .autoencoder import MiniMaxH3VideoEncoder3d as MiniMaxH3ImageEncoder3d
from .autoencoder import MiniMaxH3VideoRotaryPosEmbed as MiniMaxH3ImageRotaryPosEmbed
from .autoencoder import MiniMaxH3VideoTransformerBlock


class MiniMaxH3ImageTransformerBlock(MiniMaxH3VideoTransformerBlock):
    def forward(self, hidden_states, rotary_emb=None):
        compute_dtype = get_parameter_dtype(self.attn.to_q)
        normalized = self.norm1(hidden_states.float()).to(compute_dtype)
        hidden_states = hidden_states + self.attn(normalized, rotary_emb) * self.scale1
        normalized = self.norm2(hidden_states.float()).to(compute_dtype)
        return hidden_states + self.ff(normalized) * self.scale2


class MiniMaxH3ImageViTDecoder2d(nn.Module):
    """
    Non-causal 2D ViT decoder. Every latent pixel becomes one token; `num_register_tokens` learned register tokens plus
    a single all-zero token are appended (all at position `0`), attended over with full self-attention, and dropped
    again before the patch projection expands each token into a `patch_size x patch_size` pixel block. The rotary
    embedding carries `(h, w)` coordinates only.
    """

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        num_layers: int = 24,
        num_attention_heads: int = 16,
        attention_head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        swiglu_gate_first: bool = False,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.5,
        norm_eps: float = 1e-05,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens
        self.rope = MiniMaxH3ImageRotaryPosEmbed(int(attention_head_dim * rope_dim_ratio), theta=rope_theta, num_axes=2)
        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3ImageTransformerBlock(
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
        self.proj_out = nn.Linear(dim, out_channels * patch_size * patch_size)
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, num_channels)
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]
        register_tokens = self.register_tokens.expand(batch_size, -1, -1).to(hidden_states.dtype)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)
        grids = [
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
            for size in (height, width)
        ]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 1)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 2))
        position_ids = torch.cat([position_ids, suffix_ids], dim=1)
        rotary_emb = self.rope(position_ids)
        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, rotary_emb)
            else:
                hidden_states = block(hidden_states, rotary_emb)
        hidden_states = self.norm_out(hidden_states).to(get_parameter_dtype(self.proj_out))
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states[:, :num_patches, :]
        patch_size = self.patch_size
        hidden_states = hidden_states.view(batch_size, height, width, self.out_channels, patch_size, patch_size)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5).contiguous()
        return hidden_states.reshape(batch_size, self.out_channels, height * patch_size, width * patch_size)


class AutoencoderKLMiniMaxH3Image(AutoencoderKLMiniMaxH3):
    """H3 single-frame CNN encoder with the image checkpoint's 2D ViT decoder."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxH3VideoResnetBlock3d", "MiniMaxH3ImageTransformerBlock"]
    _repeated_blocks = ["MiniMaxH3ImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["norm"]
    _keep_in_fp32_modules = ["encoder", "quant_conv", "norm1", "norm2", "norm_out", "scale1", "scale2"]

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
        norm_eps: float = 1e-06,
        spatial_padding_mode: str = "reflect",
        decoder_num_layers: int = 24,
        decoder_num_attention_heads: int = 16,
        decoder_attention_head_dim: int = 64,
        decoder_num_register_tokens: int = 4,
        decoder_ffn_mult: int = 4,
        decoder_swiglu_gate_first: bool = False,
        decoder_rope_theta: float = 100.0,
        decoder_rope_dim_ratio: float = 0.5,
        decoder_norm_eps: float = 1e-05,
        latents_mean: tuple[float, ...] = (0.0,) * 24,
        latents_std: tuple[float, ...] = (1.0,) * 24,
    ) -> None:
        ModelMixin.__init__(self)
        self.spatial_compression_ratio = math.prod(spatial_downsample_factors)
        self.encoder = MiniMaxH3ImageEncoder3d(
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
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.decoder = MiniMaxH3ImageViTDecoder2d(
            in_channels=latent_channels,
            out_channels=out_channels,
            patch_size=self.spatial_compression_ratio,
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
        self.use_slicing = False
        self.use_tiling = True
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.quant_conv(self.encoder(x[:, :, None]))[:, :, -1]
        height, width = (x.shape[-2], x.shape[-1])
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
                row.append(self.quant_conv(self.encoder(tile[:, :, None]))[:, :, -1])
            rows.append(row)
        latent_y_overlaps = [overlap // self.spatial_compression_ratio for overlap in y_overlaps]
        latent_x_overlaps = [overlap // self.spatial_compression_ratio for overlap in x_overlaps]
        return self._stitch_tiles(rows, latent_y_overlaps, latent_x_overlaps)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(z))
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
                    ..., i_pos // ratio : i_pos // ratio + i_len // ratio, j_pos // ratio : j_pos // ratio + j_len // ratio
                ]
                row.append(self.decoder(self.post_quant_conv(tile)))
            rows.append(row)
        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput | tuple[torch.Tensor]:
        video_format = x.ndim == 5
        if video_format:
            if x.shape[2] != 1:
                raise ValueError("MiniMax-H3-Image VAE encodes exactly one frame.")
            x = x[:, :, 0]
        x = x.to(get_parameter_dtype(self.encoder))
        if self.use_slicing and x.shape[0] > 1:
            moments = torch.cat([self._encode(x_slice) for x_slice in x.split(1)])
        else:
            moments = self._encode(x)
        if video_format:
            moments = moments.unsqueeze(2)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        video_format = z.ndim == 5
        if video_format:
            if z.shape[2] != 1:
                raise ValueError("MiniMax-H3-Image VAE decodes exactly one frame.")
            z = z[:, :, 0]
        z = z.to(get_parameter_dtype(self.post_quant_conv))
        if self.use_slicing and z.shape[0] > 1:
            decoded = torch.cat([self._decode(z_slice) for z_slice in z.split(1)])
        else:
            decoded = self._decode(z)
        if video_format:
            decoded = decoded.unsqueeze(2)
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
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(z, return_dict=return_dict)
