# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders.vae import BaseOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch import Tensor
from torch import distributed as dist
from torch import nn

from simpletuner.helpers.models.hunyuanvideo.commons.parallel_states import get_parallel_state

MEMORY_LIMIT = 512 * 1024**2  # 512MB


@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.FloatTensor
    posterior: Optional[DiagonalGaussianDistribution] = None


def swish(x: Tensor, inplace=False) -> Tensor:
    """Applies the swish activation function (SiLU) with optional inplace support."""
    return F.silu(x, inplace=inplace)


def forward_with_checkpointing(module, *inputs, use_checkpointing=False):
    """Forward with optional gradient checkpointing."""

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    if use_checkpointing:
        return torch.utils.checkpoint.checkpoint(create_custom_forward(module), *inputs, use_reentrant=False)
    else:
        return module(*inputs)


def torch_cat_if_needed(tensors, dim: int):
    """
    Concatenate a list of tensors if needed, otherwise return the single element.
    If tensors are off by a few elements along the concat dimension (e.g., from streaming pads),
    trim to the minimum length to keep shapes aligned.
    """
    if len(tensors) == 1:
        return tensors[0]
    min_len = min(t.shape[dim] for t in tensors)
    if any(t.shape[dim] != min_len for t in tensors):
        tensors = [t.narrow(dim, 0, min_len) for t in tensors]
    return torch.cat(tensors, dim=dim)


class CarriedCausalConv3d(nn.Module):
    """
    Causal 3D convolution that cooperates with conv_carry_causal_3d to reuse tail frames between chunks.
    Padding is applied externally by conv_carry_causal_3d.
    """

    def __init__(
        self,
        chan_in: int,
        chan_out: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode: str = "replicate",
        enable_patch_conv: bool = False,
        **kwargs,
    ):
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
        self.carry_frames = kernel_size[0] - 1

        conv_module = PatchCausalConv3d if enable_patch_conv else nn.Conv3d
        self.conv = conv_module(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, padding=0, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


def conv_carry_causal_3d(
    xl: list, op: nn.Module, conv_carry_in: Optional[list] = None, conv_carry_out: Optional[list] = None
):
    """
    Apply a convolution with optional temporal carry so consecutive chunks share boundary context.
    """
    x = xl[0]
    xl.clear()

    if isinstance(op, CarriedCausalConv3d):
        pad = op.time_causal_padding
        if conv_carry_in is None:
            x = F.pad(x, pad, mode=op.pad_mode)
        else:
            carry = conv_carry_in.pop(0)
            # Align spatial dims (H, W) from carry to current chunk to avoid concat mismatches when tiling.
            if carry.shape[3] != x.shape[3] or carry.shape[4] != x.shape[4]:
                # Crop if carry is larger, pad if smaller.
                h_delta = x.shape[3] - carry.shape[3]
                w_delta = x.shape[4] - carry.shape[4]
                if carry.shape[3] > x.shape[3]:
                    carry = carry[:, :, :, : x.shape[3], :]
                elif h_delta > 0:
                    carry = F.pad(carry, (0, 0, 0, h_delta, 0, 0), mode=op.pad_mode)
                if carry.shape[4] > x.shape[4]:
                    carry = carry[:, :, :, :, : x.shape[4]]
                elif w_delta > 0:
                    carry = F.pad(carry, (0, w_delta, 0, 0, 0, 0), mode=op.pad_mode)

            carry_len = carry.shape[2]
            x = torch.cat([carry, x], dim=2)
            # Adjust the leading temporal pad so we do not double-count carried frames.
            x = F.pad(x, (pad[0], pad[1], pad[2], pad[3], max(pad[4] - carry_len, 0), pad[5]), mode=op.pad_mode)

        if conv_carry_out is not None and op.carry_frames > 0:
            conv_carry_out.append(x[:, :, -op.carry_frames :, :, :].clone())

        return op(x)

    return op(x)


# Optimized implementation of CogVideoXSafeConv3d
# https://github.com/huggingface/diffusers/blob/c9ff360966327ace3faad3807dc871a4e5447501/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py#L38
class PatchCausalConv3d(nn.Conv3d):
    r"""Causal Conv3d with efficient patch processing for large tensors."""

    def find_split_indices(self, seq_len, part_num):
        ideal_interval = seq_len / part_num
        possible_indices = list(range(0, seq_len, self.stride[0]))
        selected_indices = []

        for i in range(1, part_num):
            closest = min(possible_indices, key=lambda x: abs(x - round(i * ideal_interval)))
            if closest not in selected_indices:
                selected_indices.append(closest)

        merged_indices = []
        prev_idx = 0
        for idx in selected_indices:
            if idx - prev_idx >= self.kernel_size[0]:
                merged_indices.append(idx)
                prev_idx = idx

        return merged_indices

    def forward(self, input):
        T = input.shape[2]  # input: NCTHW
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / MEMORY_LIMIT
        part_num = int(memory_count / 2) + 1

        if T > self.kernel_size[0] and memory_count > 0.6 and part_num >= 2:
            if part_num > T // self.kernel_size[0]:
                part_num = T // self.kernel_size[0]
            kernel_size = self.kernel_size[0]
            split_indices = self.find_split_indices(T, part_num)

            if len(split_indices) == 0 or kernel_size == 1:
                input_chunks = torch.tensor_split(input, split_indices, dim=2) if len(split_indices) > 0 else [input]
            else:
                boundaries = [0] + split_indices + [T]
                input_chunks = []
                for i in range(len(boundaries) - 1):
                    start = boundaries[i]
                    end = boundaries[i + 1]
                    overlap_start = max(start - kernel_size + 1, 0)
                    if i == 0:
                        input_chunks.append(input[:, :, start:end])
                    else:
                        input_chunks.append(input[:, :, overlap_start:end])
            output_chunks = []
            for input_chunk in input_chunks:
                output_chunks.append(super().forward(input_chunk))
            output = torch.cat(output_chunks, dim=2)
            return output
        else:
            return super().forward(input)


class PatchConv3d(nn.Conv3d):
    r"""Conv3d with efficient patch processing for large tensors."""

    def forward(self, input):
        assert (
            self.kernel_size[0] == 1 and self.kernel_size[1] == 1 and self.kernel_size[2] == 1
        ), "PatchConv3d only supports kernel_size=1 for now."
        assert (
            self.stride[0] == 1 and self.stride[1] == 1 and self.stride[2] == 1
        ), "PatchConv3d only supports stride=1 for now."
        assert (
            self.padding[0] == 0 and self.padding[1] == 0 and self.padding[2] == 0
        ), "PatchConv3d only supports padding=0 for now."

        T = input.shape[2]  # input: NCTHW
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / MEMORY_LIMIT
        part_num = int(memory_count / 2) + 1

        if T > self.kernel_size[0] and memory_count > 0.6 and part_num >= 2:
            input_chunks = torch.tensor_split(input, part_num, dim=2)
            output_chunks = []
            for input_chunk in input_chunks:
                output_chunks.append(super().forward(input_chunk))
            output = torch.cat(output_chunks, dim=2)
            return output
        return super().forward(input)


class RMS_norm(nn.Module):
    """Root Mean Square Layer Normalization for Channel-First or Last"""

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x):
        return F.normalize(x, dim=(1 if self.channel_first else -1)) * self.scale * self.gamma + self.bias


class CausalConv3d(nn.Module):
    """Causal Conv3d with configurable padding for temporal axis."""

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode="replicate",
        disable_causal=False,
        enable_patch_conv=False,
        **kwargs,
    ):
        super().__init__()

        self.pad_mode = pad_mode
        if disable_causal:
            padding = (
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
                kernel_size // 2,
            )
        else:
            padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size - 1, 0)  # W, H, T
        self.time_causal_padding = padding

        if enable_patch_conv:
            self.conv = PatchCausalConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)
        else:
            self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


def prepare_causal_attention_mask(n_frame: int, n_hw: int, dtype, device, batch_size: int = None):
    """Prepare a causal attention mask for 3D videos.

    Args:
        n_frame (int): Number of frames (temporal length).
        n_hw (int): Product of height and width.
        dtype: Desired mask dtype.
        device: Device for the mask.
        batch_size (int, optional): If set, expands for batch.

    Returns:
        torch.Tensor: Causal attention mask.
    """
    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class AttnBlock(nn.Module):
    """Self-attention block for 3D video tensors."""

    def __init__(self, in_channels: int, enable_patch_conv: bool = False):
        super().__init__()
        self.in_channels = in_channels

        self.norm = RMS_norm(in_channels, images=False)

        conv_module = PatchConv3d if enable_patch_conv else nn.Conv3d
        self.q = conv_module(in_channels, in_channels, kernel_size=1)
        self.k = conv_module(in_channels, in_channels, kernel_size=1)
        self.v = conv_module(in_channels, in_channels, kernel_size=1)
        self.proj_out = conv_module(in_channels, in_channels, kernel_size=1)

    def sliced_attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        seq_hw = h * w
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()

        out = torch.empty_like(q)
        for frame in range(f):
            frm_start = frame * seq_hw
            frm_end = frm_start + seq_hw
            q_slice = q[:, :, frm_start:frm_end, :]
            k_slice = k[:, :, :frm_end, :]
            v_slice = v[:, :, :frm_end, :]
            out[:, :, frm_start:frm_end, :] = nn.functional.scaled_dot_product_attention(q_slice, k_slice, v_slice)

        out = rearrange(out, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w)
        return out

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()
        attention_mask = prepare_causal_attention_mask(f, h * w, h_.dtype, h_.device, batch_size=b)
        h_ = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask.unsqueeze(1))

        return rearrange(h_, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.sliced_attention(x))


class ResnetBlock(nn.Module):
    """ResNet-style block for 3D video tensors."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        enable_patch_conv: bool = False,
        conv_op: Type[nn.Module] = CausalConv3d,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = RMS_norm(in_channels, images=False)
        self.conv1 = conv_op(in_channels, out_channels, kernel_size=3, enable_patch_conv=enable_patch_conv)

        self.norm2 = RMS_norm(out_channels, images=False)
        self.conv2 = conv_op(out_channels, out_channels, kernel_size=3, enable_patch_conv=enable_patch_conv)
        if self.in_channels != self.out_channels:
            conv_module = PatchConv3d if enable_patch_conv else nn.Conv3d
            self.nin_shortcut = conv_module(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = None

    def forward(self, x, conv_carry_in: Optional[list] = None, conv_carry_out: Optional[list] = None):
        h = x
        h = self.norm1(h)
        h = swish(h, inplace=True)
        h = conv_carry_causal_3d([h], self.conv1, conv_carry_in, conv_carry_out)

        h = self.norm2(h)
        h = swish(h, inplace=True)
        h = conv_carry_causal_3d([h], self.conv2, conv_carry_in, conv_carry_out)

        if self.nin_shortcut is not None:
            x = (
                self.nin_shortcut(x)
                if not isinstance(self.nin_shortcut, CarriedCausalConv3d)
                else conv_carry_causal_3d([x], self.nin_shortcut, conv_carry_in, conv_carry_out)
            )
        return x + h


class Downsample(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_temporal_downsample: bool = True,
        enable_patch_conv: bool = False,
        conv_op: Type[nn.Module] = CausalConv3d,
    ):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        self.conv = conv_op(in_channels, out_channels // factor, kernel_size=3, enable_patch_conv=enable_patch_conv)
        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def _forward_fast(self, x: Tensor, conv_carry_in: Optional[list] = None, conv_carry_out: Optional[list] = None):
        r1 = 2 if self.add_temporal_downsample else 1
        h = conv_carry_causal_3d([x], self.conv, conv_carry_in, conv_carry_out)
        # Ensure temporal length aligns with expected stride to avoid reshape issues.
        if self.add_temporal_downsample and h.shape[2] % r1 != 0:
            pad_t = r1 - (h.shape[2] % r1)
            h = F.pad(h, (0, 0, 0, 0, 0, pad_t), mode="replicate")
        if self.add_temporal_downsample:
            h_first = h[:, :, :1, :, :]
            h_first = rearrange(h_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            h_first = torch.cat([h_first, h_first], dim=1)
            h_next = h[:, :, 1:, :, :]
            h_next = rearrange(h_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)
            # shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = rearrange(x_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            B, C, T, H, W = x_first.shape
            x_first = x_first.view(B, h.shape[1], self.group_size // 2, T, H, W)
            if self.group_size <= 2:
                x_first = x_first[:, :, 0]
            elif self.group_size == 4:
                x_first = x_first[:, :, 0].add_(x_first[:, :, 1]).mul_(0.5)
            elif self.group_size == 8:
                x_first = x_first[:, :, 0].add_(x_first[:, :, 1]).add_(x_first[:, :, 2]).add_(x_first[:, :, 3]).mul_(0.25)
            else:
                assert False, f"Unsupported group_size: {self.group_size}"

            x_next = x[:, :, 1:, :, :]
            x_next = rearrange(x_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = x_next.shape
            x_next = x_next.view(B, h.shape[1], self.group_size, T, H, W)
            if self.group_size == 1:
                x_next = x_next[:, :, 0]
            elif self.group_size == 2:
                x_next = x_next[:, :, 0].add_(x_next[:, :, 1]).mul_(0.5)
            elif self.group_size == 4:
                x_next = x_next[:, :, 0].add_(x_next[:, :, 1]).add_(x_next[:, :, 2]).add_(x_next[:, :, 3]).mul_(0.25)
            elif self.group_size == 8:
                x_next = (
                    x_next[:, :, 0]
                    .add_(x_next[:, :, 1])
                    .add_(x_next[:, :, 2])
                    .add_(x_next[:, :, 3])
                    .add_(x_next[:, :, 4])
                    .add_(x_next[:, :, 5])
                    .add_(x_next[:, :, 6])
                    .add_(x_next[:, :, 7])
                    .mul_(0.125)
                )
            else:
                assert False, f"Unsupported group_size: {self.group_size}"
            shortcut = torch.cat([x_first, x_next], dim=2)
        else:
            h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = shortcut.shape
            shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W)
            if self.group_size == 1:
                shortcut = shortcut[:, :, 0]
            elif self.group_size == 2:
                shortcut = shortcut[:, :, 0].add_(shortcut[:, :, 1]).mul_(0.5)
            elif self.group_size == 4:
                shortcut = (
                    shortcut[:, :, 0].add_(shortcut[:, :, 1]).add_(shortcut[:, :, 2]).add_(shortcut[:, :, 3]).mul_(0.25)
                )
            elif self.group_size == 8:
                shortcut = (
                    shortcut[:, :, 0]
                    .add_(shortcut[:, :, 1])
                    .add_(shortcut[:, :, 2])
                    .add_(shortcut[:, :, 3])
                    .add_(shortcut[:, :, 4])
                    .add_(shortcut[:, :, 5])
                    .add_(shortcut[:, :, 6])
                    .add_(shortcut[:, :, 7])
                    .mul_(0.125)
                )
            else:
                assert False, f"Unsupported group_size: {self.group_size}"

        return h + shortcut

    def _forward(self, x: Tensor, conv_carry_in: Optional[list] = None, conv_carry_out: Optional[list] = None):
        r1 = 2 if self.add_temporal_downsample else 1
        h = conv_carry_causal_3d([x], self.conv, conv_carry_in, conv_carry_out)
        if self.add_temporal_downsample and h.shape[2] % r1 != 0:
            pad_t = r1 - (h.shape[2] % r1)
            h = F.pad(h, (0, 0, 0, 0, 0, pad_t), mode="replicate")
        if self.add_temporal_downsample:
            h_first = h[:, :, :1, :, :]
            h_first = rearrange(h_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            h_first = torch.cat([h_first, h_first], dim=1)
            h_next = h[:, :, 1:, :, :]
            h_next = rearrange(h_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)
            # shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = rearrange(x_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            B, C, T, H, W = x_first.shape
            x_first = x_first.view(B, h.shape[1], self.group_size // 2, T, H, W).mean(dim=2)

            x_next = x[:, :, 1:, :, :]
            x_next = rearrange(x_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = x_next.shape
            x_next = x_next.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
            shortcut = torch.cat([x_first, x_next], dim=2)
        else:
            h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = shortcut.shape
            shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)

        return h + shortcut

    def forward(self, x: Tensor, conv_carry_in: Optional[list] = None, conv_carry_out: Optional[list] = None):
        return self._forward_fast(x, conv_carry_in, conv_carry_out)


class Upsample(nn.Module):
    """Hierarchical upsampling with temporal/ spatial support."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        add_temporal_upsample: bool = True,
        enable_patch_conv: bool = False,
        conv_op: Type[nn.Module] = CausalConv3d,
    ):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = conv_op(in_channels, out_channels * factor, kernel_size=3, enable_patch_conv=enable_patch_conv)
        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(self, x: Tensor, conv_carry_in: Optional[list] = None, conv_carry_out: Optional[list] = None):
        r1 = 2 if self.add_temporal_upsample else 1
        h = conv_carry_causal_3d([x], self.conv, conv_carry_in, conv_carry_out)
        if self.add_temporal_upsample:
            h_first = h[:, :, :1, :, :]
            h_first = rearrange(h_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            h_first = h_first[:, : h_first.shape[1] // 2]
            h_next = h[:, :, 1:, :, :]
            h_next = rearrange(h_next, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)

            # shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = rearrange(x_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            x_first = x_first.repeat_interleave(repeats=self.repeats // 2, dim=1)

            x_next = x[:, :, 1:, :, :]
            x_next = rearrange(x_next, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            x_next = x_next.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = torch.cat([x_first, x_next], dim=2)

        else:
            h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = rearrange(shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        return h + shortcut


class Encoder(nn.Module):
    """Hierarchical video encoder with temporal and spatial factorization."""

    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        downsample_match_channel: bool = True,
        enable_patch_conv: bool = False,
        temporal_roll: bool = False,
    ):
        super().__init__()
        assert block_out_channels[-1] % (2 * z_channels) == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks
        self.temporal_roll = temporal_roll
        self.time_compress = 1

        conv_op: Type[nn.Module] = CarriedCausalConv3d if temporal_roll else CausalConv3d
        # downsampling
        self.conv_in = conv_op(in_channels, block_out_channels[0], kernel_size=3, enable_patch_conv=enable_patch_conv)

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        enable_patch_conv=enable_patch_conv,
                        conv_op=conv_op,
                    )
                )
                block_in = block_out
            down = nn.Module()
            down.block = block

            add_spatial_downsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and bool(
                i_level >= np.log2(ffactor_spatial // ffactor_temporal)
            )
            if add_spatial_downsample or add_temporal_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if downsample_match_channel else block_in
                if add_temporal_downsample:
                    self.time_compress *= 2
                down.downsample = Downsample(
                    block_in,
                    block_out,
                    add_temporal_downsample,
                    enable_patch_conv=enable_patch_conv,
                    conv_op=conv_op,
                )
                block_in = block_out
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, enable_patch_conv=enable_patch_conv, conv_op=conv_op
        )
        self.mid.attn_1 = AttnBlock(block_in, enable_patch_conv=enable_patch_conv)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, enable_patch_conv=enable_patch_conv, conv_op=conv_op
        )

        # end
        self.norm_out = RMS_norm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, 2 * z_channels, kernel_size=3, enable_patch_conv=enable_patch_conv)

        self.gradient_checkpointing = False

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the encoder."""
        if self.temporal_roll:
            return self._forward_temporal_roll(x)

        use_checkpointing = bool(self.training and self.gradient_checkpointing)

        # downsampling
        h = self.conv_in(x)
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks):
                h = forward_with_checkpointing(self.down[i_level].block[i_block], h, use_checkpointing=use_checkpointing)
            if hasattr(self.down[i_level], "downsample"):
                h = forward_with_checkpointing(self.down[i_level].downsample, h, use_checkpointing=use_checkpointing)

        # middle
        h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)

        # end
        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        shortcut = rearrange(h, "b (c r) f h w -> b c r f h w", r=group_size).mean(dim=2)
        h = self.norm_out(h)
        h = swish(h, inplace=True)
        h = self.conv_out(h)
        h += shortcut
        return h

    def _forward_temporal_roll(self, x: Tensor) -> Tensor:
        """
        Stream the VAE encoder over time, carrying the last frames between chunks to reduce VRAM.
        """
        segments = [x[:, :, :1, :, :]]
        if x.shape[2] > self.time_compress:
            tail = x[:, :, 1 : 1 + ((x.shape[2] - 1) // self.time_compress) * self.time_compress, :, :]
            segments.extend(torch.split(tail, self.time_compress * 2, dim=2))
        elif x.shape[2] > 1:
            segments.append(x[:, :, 1:, :, :])

        out = []
        conv_carry_in: Optional[list] = None

        for idx, seg in enumerate(segments):
            conv_carry_out: Optional[list] = [] if idx < len(segments) - 1 else None
            h = conv_carry_causal_3d([seg], self.conv_in, conv_carry_in, conv_carry_out)
            for i_level in range(len(self.block_out_channels)):
                for i_block in range(self.num_res_blocks):
                    h = self.down[i_level].block[i_block](h, conv_carry_in, conv_carry_out)
                if hasattr(self.down[i_level], "downsample"):
                    h = self.down[i_level].downsample(h, conv_carry_in, conv_carry_out)
            out.append(h)
            conv_carry_in = conv_carry_out

        h = torch_cat_if_needed(out, dim=2)

        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        shortcut = rearrange(h, "b (c r) f h w -> b c r f h w", r=group_size).mean(dim=2)
        h = self.norm_out(h)
        h = swish(h, inplace=True)
        h = self.conv_out(h)
        h += shortcut
        return h


class Decoder(nn.Module):
    """Hierarchical video decoder with upsampling factories."""

    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        upsample_match_channel: bool = True,
        enable_patch_conv: bool = False,
        temporal_roll: bool = False,
    ):
        super().__init__()
        assert block_out_channels[0] % z_channels == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks
        self.temporal_roll = temporal_roll

        block_in = block_out_channels[0]
        conv_op: Type[nn.Module] = CarriedCausalConv3d if temporal_roll else CausalConv3d

        self.conv_in = conv_op(z_channels, block_in, kernel_size=3, enable_patch_conv=enable_patch_conv)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, enable_patch_conv=enable_patch_conv, conv_op=conv_op
        )
        self.mid.attn_1 = AttnBlock(block_in, enable_patch_conv=enable_patch_conv)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in, out_channels=block_in, enable_patch_conv=enable_patch_conv, conv_op=conv_op
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        enable_patch_conv=enable_patch_conv,
                        conv_op=conv_op,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block

            add_spatial_upsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_upsample = bool(i_level < np.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if upsample_match_channel else block_in
                up.upsample = Upsample(
                    block_in,
                    block_out,
                    add_temporal_upsample,
                    enable_patch_conv=enable_patch_conv,
                    conv_op=conv_op,
                )
                block_in = block_out
            self.up.append(up)

        # end
        self.norm_out = RMS_norm(block_in, images=False)
        self.conv_out = conv_op(block_in, out_channels, kernel_size=3, enable_patch_conv=enable_patch_conv)

        self.gradient_checkpointing = False

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass through the decoder."""
        if self.temporal_roll:
            return self._forward_temporal_roll(z)

        use_checkpointing = bool(self.training and self.gradient_checkpointing)

        # z to block_in
        repeats = self.block_out_channels[0] // (self.z_channels)
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)

        # middle
        h = forward_with_checkpointing(self.mid.block_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.attn_1, h, use_checkpointing=use_checkpointing)
        h = forward_with_checkpointing(self.mid.block_2, h, use_checkpointing=use_checkpointing)

        # upsampling
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                h = forward_with_checkpointing(self.up[i_level].block[i_block], h, use_checkpointing=use_checkpointing)
            if hasattr(self.up[i_level], "upsample"):
                h = forward_with_checkpointing(self.up[i_level].upsample, h, use_checkpointing=use_checkpointing)

        # end
        h = self.norm_out(h)
        h = swish(h, inplace=True)
        h = self.conv_out(h)
        return h

    def _forward_temporal_roll(self, z: Tensor) -> Tensor:
        repeats = self.block_out_channels[0] // (self.z_channels)
        h = conv_carry_causal_3d([z], self.conv_in)
        h = h + z.repeat_interleave(repeats=repeats, dim=1)

        segments = torch.split(h, 2, dim=2) if h.shape[2] > 1 else (h,)
        out = []
        conv_carry_in: Optional[list] = None

        for idx, seg in enumerate(segments):
            conv_carry_out: Optional[list] = [] if idx < len(segments) - 1 else None

            h_seg = self.mid.block_1(seg, conv_carry_in, conv_carry_out)
            h_seg = self.mid.attn_1(h_seg)
            h_seg = self.mid.block_2(h_seg, conv_carry_in, conv_carry_out)

            for i_level in range(len(self.block_out_channels) - 1, -1, -1):
                for i_block in range(self.num_res_blocks + 1):
                    h_seg = self.up[i_level].block[i_block](h_seg, conv_carry_in, conv_carry_out)
                if hasattr(self.up[i_level], "upsample"):
                    h_seg = self.up[i_level].upsample(h_seg, conv_carry_in, conv_carry_out)

            h_seg = self.norm_out(h_seg)
            h_seg = swish(h_seg, inplace=True)
            h_seg = conv_carry_causal_3d([h_seg], self.conv_out, conv_carry_in, conv_carry_out)
            out.append(h_seg)

            conv_carry_in = conv_carry_out

        return torch_cat_if_needed(list(out), dim=2)


class AutoencoderKLConv3D(ModelMixin, ConfigMixin):
    """KL regularized 3D Conv VAE with advanced tiling and slicing strategies."""

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...],
        layers_per_block: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        sample_size: int,
        sample_tsize: int,
        scaling_factor: float = None,
        shift_factor: Optional[float] = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
        enable_patch_conv: bool = False,
        enable_temporal_roll: bool = False,
    ):
        super().__init__()
        self.ffactor_spatial = ffactor_spatial
        self.ffactor_temporal = ffactor_temporal
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor
        self.temporal_roll = enable_temporal_roll

        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=latent_channels,
            block_out_channels=block_out_channels,
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            downsample_match_channel=downsample_match_channel,
            enable_patch_conv=enable_patch_conv,
            temporal_roll=enable_temporal_roll,
        )
        self.decoder = Decoder(
            z_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=list(reversed(block_out_channels)),
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            upsample_match_channel=upsample_match_channel,
            enable_patch_conv=enable_patch_conv,
            temporal_roll=enable_temporal_roll,
        )

        self.use_slicing = False
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // ffactor_spatial
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // ffactor_temporal
        self.tile_overlap_factor = 0.25

        self._tile_parallelism_enabled = False

    def set_tile_sample_min_size(self, sample_size: int, tile_overlap_factor: float = 0.2):
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // self.ffactor_spatial
        self.tile_overlap_factor = tile_overlap_factor

        assert (
            self.tile_latent_min_size * self.tile_overlap_factor
        ).is_integer(), "self.tile_latent_min_size multiplied by tile_overlap_factor must be an integer"

    def _set_gradient_checkpointing(self, module, value=False):
        """Enable or disable gradient checkpointing on encoder and decoder."""
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    def enable_temporal_tiling(self, use_tiling: bool = True):
        raise RuntimeError("Temporal tiling is not supported for this VAE.")

    def disable_temporal_tiling(self):
        self.enable_temporal_tiling(False)

    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(self, use_tiling: bool = True):
        self.enable_spatial_tiling(use_tiling)

    def disable_tiling(self):
        self.disable_spatial_tiling()

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        """Blend tensor b horizontally into a at blend_extent region."""
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        """Blend tensor b vertically into a at blend_extent region."""
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        """Blend tensor b temporally into a at blend_extent region."""
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (
                x / blend_extent
            )
        return b

    def spatial_tiled_encode(self, x: torch.Tensor):
        """Tiled spatial encoding for large inputs via overlapping."""
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = x[:, :, :, i : i + self.tile_sample_min_size, j : j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        moments = torch.cat(result_rows, dim=-2)
        return moments

    def temporal_tiled_encode(self, x: torch.Tensor):
        """Tiled temporal encoding for large video sequences."""
        raise RuntimeError("Temporal tiling is not supported for this VAE.")

    def enable_tile_parallelism(self):
        self._tile_parallelism_enabled = True

    def disable_tile_parallelism(self):
        self._tile_parallelism_enabled = False

    def tile_parallel_spatial_tiled_decode(self, z: torch.Tensor):
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        rank = get_parallel_state().sp_rank
        world_size = get_parallel_state().sp

        num_rows = math.ceil(H / overlap_size)
        num_cols = math.ceil(W / overlap_size)
        total_tiles = num_rows * num_cols
        tiles_per_rank = math.ceil(total_tiles / world_size)

        my_linear_indices = list(range(rank, total_tiles, world_size))
        decoded_tiles = []
        decoded_metas = []
        H_out_std = self.tile_sample_min_size
        W_out_std = self.tile_sample_min_size
        for lin_idx in my_linear_indices:
            ri = lin_idx // num_cols
            rj = lin_idx % num_cols
            i = ri * overlap_size
            j = rj * overlap_size
            tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
            dec = self.decoder(tile)

            pad_h = max(0, H_out_std - dec.shape[-2])
            pad_w = max(0, W_out_std - dec.shape[-1])
            if pad_h > 0 or pad_w > 0:
                dec = F.pad(dec, (0, pad_w, 0, pad_h, 0, 0), "constant", 0)
            decoded_tiles.append(dec)
            decoded_metas.append(torch.tensor([ri, rj, pad_w, pad_h], device=z.device, dtype=torch.int64))

        while len(decoded_tiles) < tiles_per_rank:
            zero_tile = torch.zeros(
                [1, 3, (T - 1) * self.ffactor_temporal + 1, self.tile_sample_min_size, self.tile_sample_min_size],
                device=dec.device,
                dtype=dec.dtype,
            )
            decoded_tiles.append(zero_tile)
            meta_tensor = torch.tensor(
                [-1, -1, self.tile_sample_min_size, self.tile_sample_min_size], device=z.device, dtype=torch.int64
            )
            decoded_metas.append(meta_tensor)

        decoded_tiles = torch.stack(decoded_tiles, dim=0)
        decoded_metas = torch.stack(decoded_metas, dim=0)

        tiles_gather_list = [torch.empty_like(decoded_tiles) for _ in range(world_size)]
        metas_gather_list = [torch.empty_like(decoded_metas) for _ in range(world_size)]

        dist.all_gather(tiles_gather_list, decoded_tiles, group=get_parallel_state().sp_group)
        dist.all_gather(metas_gather_list, decoded_metas, group=get_parallel_state().sp_group)

        if rank != 0:
            return torch.empty(0, device=z.device)

        rows = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        for r in range(world_size):
            gathered_tiles_r = tiles_gather_list[r]  # [tiles_per_rank, B, C, T, H, W]
            gathered_metas_r = metas_gather_list[r]  # [tiles_per_rank, 4]
            for k in range(gathered_tiles_r.shape[0]):
                ri = int(gathered_metas_r[k][0])
                rj = int(gathered_metas_r[k][1])
                if ri < 0 or rj < 0:
                    continue
                if ri < num_rows and rj < num_cols:
                    # remove padding
                    pad_w = int(gathered_metas_r[k][2])
                    pad_h = int(gathered_metas_r[k][3])
                    h_end = None if pad_h == 0 else -pad_h
                    w_end = None if pad_w == 0 else -pad_w
                    rows[ri][rj] = gathered_tiles_r[k][:, :, :, :h_end, :w_end]

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if tile is None:
                    continue
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=-2)
        return dec

    def spatial_tiled_decode(self, z: torch.Tensor):
        if self._tile_parallelism_enabled:
            return self.tile_parallel_spatial_tiled_decode(z)

        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = z[:, :, :, i : i + self.tile_latent_min_size, j : j + self.tile_latent_min_size]
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=-2)
        return dec

    def temporal_tiled_decode(self, z: torch.Tensor):
        """Tiled temporal decoding for long sequence latents."""
        raise RuntimeError("Temporal tiling is not supported for this VAE.")

    def encode(self, x: Tensor, return_dict: bool = True):

        def _encode(x):
            if self.use_temporal_tiling and x.shape[-3] > self.tile_sample_min_tsize:
                return self.temporal_tiled_encode(x)
            if self.use_spatial_tiling and (
                x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size
            ):
                return self.spatial_tiled_encode(x)
            return self.encoder(x)

        assert len(x.shape) == 5  # (B, C, T, H, W)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [_encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = _encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: Tensor, return_dict: bool = True, generator=None):

        def _decode(z):
            if self.use_temporal_tiling and z.shape[-3] > self.tile_latent_min_tsize:
                return self.temporal_tiled_decode(z)
            if self.use_spatial_tiling and (
                z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size
            ):
                return self.spatial_tiled_decode(z)
            return self.decoder(z)

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [_decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = _decode(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(
        self, sample: torch.Tensor, sample_posterior: bool = False, return_posterior: bool = True, return_dict: bool = True
    ):
        """Forward autoencoder pass. Returns both reconstruction and optionally the posterior."""
        posterior = self.encode(sample).latent_dist
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z).sample
        return DecoderOutput(sample=dec, posterior=posterior) if return_dict else (dec, posterior)
