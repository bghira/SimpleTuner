"""
Optimized HunyuanVideo VAE implementation.

Ported from Kandinsky-5 team's implementation with dynamic memory-aware tiling,
framewise encoding/decoding, and temporal blending for efficient video processing.

Used by: Kandinsky5 Video
"""

from math import ceil, sqrt
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import Attention
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook

# Memory limit for temporal chunking (512MB default)
MEMORY_LIMIT = 512 * 1024**2


def _find_temporal_split_indices(seq_len: int, part_num: int, stride: int) -> list:
    """Find optimal split indices for temporal chunking."""
    ideal_interval = seq_len / part_num
    possible_indices = list(range(0, seq_len, stride))
    selected_indices = []

    for i in range(1, part_num):
        closest = min(possible_indices, key=lambda x: abs(x - round(i * ideal_interval)))
        if closest not in selected_indices:
            selected_indices.append(closest)

    merged_indices = []
    prev_idx = 0
    for idx in selected_indices:
        if idx - prev_idx >= stride:
            merged_indices.append(idx)
            prev_idx = idx

    return merged_indices


def prepare_causal_attention_mask(f: int, s: int, dtype: torch.dtype, device: torch.device, b: int) -> torch.Tensor:
    return (
        torch.ones((f, f), dtype=dtype, device=device)
        .tril_()
        .log_()
        .repeat_interleave(s, dim=0)
        .repeat_interleave(s, dim=1)
        .unsqueeze(0)
        .expand(b, -1, -1)
        .contiguous()
    )


class HunyuanVideoCausalConv3d(nn.Module):
    """
    Causal 3D convolution with optional temporal chunking for memory efficiency.

    When enable_temporal_chunking=True, large tensors are automatically split
    along the temporal dimension to reduce peak VRAM usage.
    """

    # Class-level flag to enable temporal chunking globally
    _temporal_chunking_enabled: bool = False

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()

        kernel_size = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

        self.pad_mode = pad_mode
        self.time_causal_padding = (
            kernel_size[0] // 2,
            kernel_size[0] // 2,
            kernel_size[1] // 2,
            kernel_size[1] // 2,
            kernel_size[2] - 1,
            0,
        )

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(hidden_states, self.time_causal_padding, mode=self.pad_mode)

        if not self._temporal_chunking_enabled:
            return self.conv(hidden_states)

        # Temporal chunking for memory efficiency
        T = hidden_states.shape[2]
        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        memory_count = torch.prod(torch.tensor(hidden_states.shape)).item() * 2 / MEMORY_LIMIT
        part_num = int(memory_count / 2) + 1

        use_chunking = T > kernel_size and memory_count > 0.6 and part_num >= 2
        if not use_chunking or T <= 1:
            return self.conv(hidden_states)

        max_parts = max(1, T // kernel_size)
        if part_num > max_parts:
            part_num = max_parts

        split_indices = _find_temporal_split_indices(T, part_num, stride)
        if len(split_indices) == 0 or kernel_size == 1:
            input_chunks = torch.tensor_split(hidden_states, split_indices, dim=2) if split_indices else [hidden_states]
        else:
            boundaries = [0] + split_indices + [T]
            input_chunks = []
            for i in range(len(boundaries) - 1):
                start = boundaries[i]
                end = boundaries[i + 1]
                overlap_start = max(start - kernel_size + 1, 0)
                if i == 0:
                    input_chunks.append(hidden_states[:, :, start:end])
                else:
                    input_chunks.append(hidden_states[:, :, overlap_start:end])

        output_chunks = [self.conv(chunk) for chunk in input_chunks]
        return torch.cat(output_chunks, dim=2)


class HunyuanVideoUpsampleCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        upsample_factor: Tuple[float, float, float] = (2, 2, 2),
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        self.upsample_factor = upsample_factor

        self.conv = HunyuanVideoCausalConv3d(in_channels, out_channels, kernel_size, stride, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_frames = hidden_states.size(2)
        dtp = hidden_states.dtype
        first_frame, other_frames = hidden_states.split((1, num_frames - 1), dim=2)
        first_frame = (
            F.interpolate(
                first_frame.squeeze(2),
                scale_factor=self.upsample_factor[1:],
                mode="nearest",
            )
            .unsqueeze(2)
            .to(dtp)
        )

        if num_frames > 1:
            other_frames = other_frames.contiguous()
            other_frames = F.interpolate(other_frames, scale_factor=self.upsample_factor, mode="nearest").to(dtp)
            hidden_states = torch.cat((first_frame, other_frames), dim=2)
            del first_frame
            del other_frames
            torch.cuda.empty_cache()
        else:
            hidden_states = first_frame

        hidden_states = self.conv(hidden_states)
        return hidden_states


class HunyuanVideoDownsampleCausal3D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        padding: int = 1,
        kernel_size: int = 3,
        bias: bool = True,
        stride=2,
    ) -> None:
        super().__init__()
        out_channels = out_channels or channels

        self.conv = HunyuanVideoCausalConv3d(channels, out_channels, kernel_size, stride, padding, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        return hidden_states


class HunyuanVideoResnetBlockCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)
        self.conv1 = HunyuanVideoCausalConv3d(in_channels, out_channels, 3, 1, 0)

        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = HunyuanVideoCausalConv3d(out_channels, out_channels, 3, 1, 0)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = HunyuanVideoCausalConv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtp = hidden_states.dtype
        hidden_states = hidden_states.contiguous()
        residual = hidden_states

        hidden_states = self.norm1(hidden_states).to(dtp)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states).to(dtp)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        hidden_states = hidden_states + residual
        return hidden_states


class HunyuanVideoMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_head_dim: int = 1,
    ) -> None:
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        resnets = [
            HunyuanVideoResnetBlockCausal3D(
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                non_linearity=resnet_act_fn,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                HunyuanVideoResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                batch_size, _, num_frames, height, width = hidden_states.shape
                hidden_states = hidden_states.permute(0, 2, 3, 4, 1).flatten(1, 3)
                mask = prepare_causal_attention_mask(
                    num_frames,
                    height * width,
                    hidden_states.dtype,
                    hidden_states.device,
                    batch_size,
                )
                hidden_states = attn(hidden_states, attention_mask=mask)
                hidden_states = hidden_states.unflatten(1, (num_frames, height, width)).permute(0, 4, 1, 2, 3)

            hidden_states = resnet(hidden_states)

        return hidden_states


class HunyuanVideoDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_stride: int = 2,
        downsample_padding: int = 1,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                HunyuanVideoResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    HunyuanVideoDownsampleCausal3D(
                        out_channels,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        stride=downsample_stride,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class HunyuanVideoUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_upsample: bool = True,
        upsample_scale_factor: Tuple[int, int, int] = (2, 2, 2),
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                HunyuanVideoResnetBlockCausal3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    HunyuanVideoUpsampleCausal3D(
                        out_channels,
                        out_channels=out_channels,
                        upsample_factor=upsample_scale_factor,
                    )
                ]
            )
        else:
            self.upsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class HunyuanVideoEncoder3D(nn.Module):
    """Causal encoder for 3D video-like data."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        temporal_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
    ) -> None:
        super().__init__()

        self.conv_in = HunyuanVideoCausalConv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1)
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            if down_block_type != "HunyuanVideoDownBlock3D":
                raise ValueError(f"Unsupported down_block_type: {down_block_type}")

            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_downsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_downsample_layers = int(np.log2(temporal_compression_ratio))

            if temporal_compression_ratio == 4:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(
                    i >= (len(block_out_channels) - 1 - num_time_downsample_layers) and not is_final_block
                )
            elif temporal_compression_ratio == 8:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(i < num_time_downsample_layers)
            else:
                raise ValueError(f"Unsupported time_compression_ratio: {temporal_compression_ratio}")

            downsample_stride_HW = (2, 2) if add_spatial_downsample else (1, 1)
            downsample_stride_T = (2,) if add_time_downsample else (1,)
            downsample_stride = tuple(downsample_stride_T + downsample_stride_HW)

            down_block = HunyuanVideoDownBlock3D(
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=bool(add_spatial_downsample or add_time_downsample),
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                downsample_stride=downsample_stride,
                downsample_padding=0,
            )

            self.down_blocks.append(down_block)

        self.mid_block = HunyuanVideoMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = HunyuanVideoCausalConv3d(block_out_channels[-1], conv_out_channels, kernel_size=3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)

        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)

        hidden_states = self.mid_block(hidden_states)

        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class HunyuanVideoDecoder3D(nn.Module):
    """Causal decoder for 3D video-like data."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = (
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention=True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = HunyuanVideoCausalConv3d(in_channels, block_out_channels[-1], kernel_size=3, stride=1)
        self.up_blocks = nn.ModuleList([])

        self.mid_block = HunyuanVideoMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            if up_block_type != "HunyuanVideoUpBlock3D":
                raise ValueError(f"Unsupported up_block_type: {up_block_type}")

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_upsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_upsample_layers = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 4:
                add_spatial_upsample = bool(i < num_spatial_upsample_layers)
                add_time_upsample = bool(i >= len(block_out_channels) - 1 - num_time_upsample_layers and not is_final_block)
            else:
                raise ValueError(f"Unsupported time_compression_ratio: {time_compression_ratio}")

            upsample_scale_factor_HW = (2, 2) if add_spatial_upsample else (1, 1)
            upsample_scale_factor_T = (2,) if add_time_upsample else (1,)
            upsample_scale_factor = tuple(upsample_scale_factor_T + upsample_scale_factor_HW)

            up_block = HunyuanVideoUpBlock3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=bool(add_spatial_upsample or add_time_upsample),
                upsample_scale_factor=upsample_scale_factor,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = HunyuanVideoCausalConv3d(block_out_channels[0], out_channels, kernel_size=3)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtp = hidden_states.dtype
        hidden_states = self.conv_in(hidden_states)

        hidden_states = self.mid_block(hidden_states)

        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states)

        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states).to(dtp)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class AutoencoderKLHunyuanVideoOptimized(ModelMixin, ConfigMixin):
    """
    Optimized VAE for HunyuanVideo with dynamic memory-aware tiling,
    framewise encoding/decoding, and temporal blending.

    Ported from Kandinsky-5 team's implementation.
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        down_block_types: Tuple[str, ...] = (
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        scaling_factor: float = 0.476986,
        spatial_compression_ratio: int = 8,
        temporal_compression_ratio: int = 4,
        mid_block_add_attention: bool = True,
    ) -> None:
        super().__init__()

        self.time_compression_ratio = temporal_compression_ratio

        self.encoder = HunyuanVideoEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
        )

        self.decoder = HunyuanVideoDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            time_compression_ratio=temporal_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)

        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

        self.use_slicing = False
        self.use_tiling = True
        self.use_framewise_encoding = True
        self.use_framewise_decoding = True

        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_num_frames = 16

        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        self.tile_sample_stride_num_frames = 12

        self.tile_size = None

    def enable_temporal_chunking(self):
        """
        Enable temporal chunking in Conv3d layers for lower peak VRAM.

        This splits large convolution operations along the temporal dimension,
        reducing memory usage at the cost of slightly more compute.
        """
        HunyuanVideoCausalConv3d._temporal_chunking_enabled = True

    def disable_temporal_chunking(self):
        """Disable temporal chunking in Conv3d layers."""
        HunyuanVideoCausalConv3d._temporal_chunking_enabled = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, num_frames, height, width = x.shape

        if self.use_framewise_decoding and num_frames > (self.tile_sample_min_num_frames + 1):
            return self._temporal_tiled_encode(x)

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        x = self.encoder(x)
        enc = self.quant_conv(x)
        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, opt_tiling: bool = True, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        """Encode a batch of videos into latents."""
        if opt_tiling:
            tile_size, tile_stride = self.get_enc_optimal_tiling(x.shape)
        else:
            b, _, f, h, w = x.shape
            tile_size, tile_stride = (b, f, h, w), (f, h, w)
        if tile_size != self.tile_size:
            self.tile_size = tile_size
            self.apply_tiling(tile_size, tile_stride)

        h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        _, _, num_frames, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio

        if self.use_framewise_decoding and num_frames > (tile_latent_min_num_frames + 1):
            return self._temporal_tiled_decode(z, return_dict=return_dict)

        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        """Decode a batch of latents."""
        tile_size, tile_stride = self.get_dec_optimal_tiling(z.shape)
        if tile_size != self.tile_size:
            self.tile_size = tile_size
            self.apply_tiling(tile_size, tile_stride)

        decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of videos using a tiled encoder."""
        _, _, _, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        rows = []
        for i in range(0, height - self.tile_sample_min_height + 1, self.tile_sample_stride_height):
            row = []
            for j in range(0, width - self.tile_sample_min_width + 1, self.tile_sample_stride_width):
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_height,
                    j : j + self.tile_sample_min_width,
                ]
                tile = self.encoder(tile).clone()
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                height_lim = tile_latent_min_height if i == len(rows) - 1 else tile_latent_stride_height
                width_lim = tile_latent_min_width if j == len(row) - 1 else tile_latent_stride_width
                result_row.append(tile[:, :, :, :height_lim, :width_lim])
            result_rows.append(torch.cat(result_row, dim=4))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc

    def tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        """Decode a batch of latents using a tiled decoder."""
        _, _, _, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        rows = []
        for i in range(0, height - tile_latent_min_height + 1, tile_latent_stride_height):
            row = []
            for j in range(0, width - tile_latent_min_width + 1, tile_latent_stride_width):
                tile = z[
                    :,
                    :,
                    :,
                    i : i + tile_latent_min_height,
                    j : j + tile_latent_min_width,
                ]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile).clone()
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                height_lim = self.tile_sample_min_height if i == len(rows) - 1 else self.tile_sample_stride_height
                width_lim = self.tile_sample_min_width if j == len(row) - 1 else self.tile_sample_stride_width
                result_row.append(tile[:, :, :, :height_lim, :width_lim])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def _temporal_tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, num_frames, height, width = x.shape
        latent_num_frames = (num_frames - 1) // self.temporal_compression_ratio + 1

        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        tile_latent_stride_num_frames = self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        blend_num_frames = tile_latent_min_num_frames - tile_latent_stride_num_frames

        row = []
        for i in range(
            0,
            num_frames - self.tile_sample_min_num_frames + 1,
            self.tile_sample_stride_num_frames,
        ):
            tile = x[:, :, i : i + self.tile_sample_min_num_frames + 1, :, :]
            if self.use_tiling and (height > self.tile_sample_min_height or width > self.tile_sample_min_width):
                tile = self.tiled_encode(tile)
            else:
                tile = self.encoder(tile).clone()
                tile = self.quant_conv(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                t_lim = tile_latent_min_num_frames if i == len(row) - 1 else tile_latent_stride_num_frames
                result_row.append(tile[:, :, :t_lim, :, :])
            else:
                result_row.append(tile[:, :, : tile_latent_stride_num_frames + 1, :, :])

        enc = torch.cat(result_row, dim=2)[:, :, :latent_num_frames]
        return enc

    def _temporal_tiled_decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        _, _, num_frames, _, _ = z.shape
        num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_min_num_frames = self.tile_sample_min_num_frames // self.temporal_compression_ratio
        tile_latent_stride_num_frames = self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        blend_num_frames = self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames

        row = []
        for i in range(
            0,
            num_frames - tile_latent_min_num_frames + 1,
            tile_latent_stride_num_frames,
        ):
            tile = z[:, :, i : i + tile_latent_min_num_frames + 1, :, :]
            if self.use_tiling and (tile.shape[-1] > tile_latent_min_width or tile.shape[-2] > tile_latent_min_height):
                decoded = self.tiled_decode(tile, return_dict=True).sample
            else:
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile).clone()
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                t_lim = self.tile_sample_min_num_frames if i == len(row) - 1 else self.tile_sample_stride_num_frames
                result_row.append(tile[:, :, :t_lim, :, :])
            else:
                result_row.append(tile[:, :, : self.tile_sample_stride_num_frames + 1, :, :])

        dec = torch.cat(result_row, dim=2)[:, :, :num_sample_frames]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, return_dict=return_dict)
        return dec

    def apply_tiling(self, tile: Tuple[int, int, int, int], stride: Tuple[int, int, int]):
        """Applies tiling configuration."""
        _, ft, ht, wt = tile
        fs, hs, ws = stride

        self.use_tiling = True
        self.tile_sample_min_num_frames = ft - 1
        self.tile_sample_stride_num_frames = fs
        self.tile_sample_min_height = ht
        self.tile_sample_min_width = wt
        self.tile_sample_stride_height = hs
        self.tile_sample_stride_width = ws

    def get_enc_optimal_tiling(self, shape: List[int]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int]]:
        """Returns optimal tiling for given shape based on available memory."""
        h, w = shape[3:]

        if torch.cuda.is_available():
            free_mem = torch.cuda.mem_get_info()[0]
        else:
            # Fallback for non-CUDA devices
            free_mem = 8 * 1024**3  # Assume 8GB

        max_area = free_mem / 256 / 17 / 8
        num_vals = 256 * 17 * (h + 32) * (w + 32)

        if h * w < max_area and num_vals < 2**31:
            return (1, 17, h, w), (8, h, w)

        def factorize(n, k):
            a = sqrt(n / k)
            b = sqrt(n * k)
            return ceil(a), ceil(b)

        k = max(h / w, w / h)
        N = max(ceil(h * w / max_area), ceil(num_vals / 2**31))
        a, b = factorize(N, k)
        if h >= w:
            wn, hn = a, b
        else:
            wn, hn = b, a

        if wn > 1:
            wt = ceil(w / wn / 8) * 8 + 16
            ws = wt - 32
        else:
            wt = w
            ws = w
        if hn > 1:
            ht = ceil(h / hn / 8) * 8 + 16
            hs = ht - 32
        else:
            ht = h
            hs = h
        return (1, 17, ht, wt), (8, hs, ws)

    def get_dec_optimal_tiling(self, shape: List[int]) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int]]:
        """Returns optimal tiling for decoding given shape."""
        b, _, f, h, w = shape
        enc_inp_shape = [b, 3, 4 * (f - 1) + 1, 8 * h, 8 * w]
        return self.get_enc_optimal_tiling(enc_inp_shape)


def load_optimized_vae(
    pretrained_path: str,
    subfolder: str = "vae",
    torch_dtype: torch.dtype = torch.float16,
    enable_temporal_chunking: bool = False,
):
    """
    Load an optimized HunyuanVideo VAE from a pretrained path.

    This loads the weights from a diffusers-format checkpoint into our optimized VAE class.

    Args:
        pretrained_path: Path to the pretrained model (e.g., HuggingFace repo or local path)
        subfolder: Subfolder containing the VAE (default: "vae")
        torch_dtype: Data type for the model weights
        enable_temporal_chunking: If True, enables temporal chunking in Conv3d layers
            for lower peak VRAM usage. Useful for long videos or limited GPU memory.
    """
    from diffusers import AutoencoderKLHunyuanVideo as DiffusersVAE

    # Load the diffusers VAE to get the config and weights
    diffusers_vae = DiffusersVAE.from_pretrained(pretrained_path, subfolder=subfolder, torch_dtype=torch_dtype)

    # Create our optimized VAE with the same config
    config = diffusers_vae.config
    optimized_vae = AutoencoderKLHunyuanVideoOptimized(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        latent_channels=config.latent_channels,
        down_block_types=config.down_block_types,
        up_block_types=config.up_block_types,
        block_out_channels=config.block_out_channels,
        layers_per_block=config.layers_per_block,
        act_fn=config.act_fn,
        norm_num_groups=config.norm_num_groups,
        scaling_factor=config.scaling_factor,
        spatial_compression_ratio=config.spatial_compression_ratio,
        temporal_compression_ratio=config.temporal_compression_ratio,
        mid_block_add_attention=config.mid_block_add_attention,
    )

    # Copy the weights
    optimized_vae.load_state_dict(diffusers_vae.state_dict())
    optimized_vae = optimized_vae.to(torch_dtype)

    # Enable temporal chunking if requested
    if enable_temporal_chunking:
        optimized_vae.enable_temporal_chunking()

    del diffusers_vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return optimized_vae
