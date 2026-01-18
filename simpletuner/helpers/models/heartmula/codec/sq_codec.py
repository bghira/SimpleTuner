# Copyright 2025 SimpleTuner contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import InplaceFunction
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class Conv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        padding=None,
        causal: bool = False,
        w_init_gain=None,
    ):
        self.causal = causal
        if padding is None:
            if causal:
                padding = 0
                self.left_padding = dilation * (kernel_size - 1)
            else:
                padding = get_padding(kernel_size, dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias,
        )
        if w_init_gain is not None:
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        if self.causal:
            x = F.pad(x.unsqueeze(2), (self.left_padding, 0, 0, 0)).squeeze(2)
        return super().forward(x)


class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding=None,
        padding_mode: str = "zeros",
        causal: bool = False,
    ):
        if padding is None:
            padding = 0 if causal else (kernel_size - stride) // 2
        if causal:
            if padding != 0:
                raise ValueError("padding is not allowed in causal ConvTranspose1d.")
            if kernel_size != 2 * stride:
                raise ValueError("kernel_size must equal 2*stride in causal ConvTranspose1d.")
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )
        self.causal = causal
        self.stride = stride

    def forward(self, x):
        x = super().forward(x)
        if self.causal:
            x = x[:, :, : -self.stride]
        return x


class PreProcessor(nn.Module):
    def __init__(self, n_in, n_out, num_samples, kernel_size=7, causal=False):
        super().__init__()
        self.pooling = nn.AvgPool1d(kernel_size=num_samples)
        self.conv = Conv1d(n_in, n_out, kernel_size=kernel_size, causal=causal)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.activation(self.conv(x))
        return self.pooling(x)


class PostProcessor(nn.Module):
    def __init__(self, n_in, n_out, num_samples, kernel_size=7, causal=False):
        super().__init__()
        self.num_samples = num_samples
        self.conv = Conv1d(n_in, n_out, kernel_size=kernel_size, causal=causal)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = x.transpose(1, 2)
        batch, length, channels = x.size()
        x = x.repeat(1, 1, self.num_samples).view(batch, -1, channels)
        x = x.transpose(1, 2)
        return self.activation(self.conv(x))


class ResidualUnit(nn.Module):
    def __init__(self, n_in, n_out, dilation, res_kernel_size=7, causal=False):
        super().__init__()
        self.conv1 = weight_norm(Conv1d(n_in, n_out, kernel_size=res_kernel_size, dilation=dilation, causal=causal))
        self.conv2 = weight_norm(Conv1d(n_in, n_out, kernel_size=1, causal=causal))
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()

    def forward(self, x):
        residual = self.activation1(self.conv1(x))
        residual = self.activation2(self.conv2(residual))
        return residual + x


class ResEncoderBlock(nn.Module):
    def __init__(self, n_in, n_out, stride, down_kernel_size, res_kernel_size=7, causal=False):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                ResidualUnit(n_in, n_out // 2, dilation=1, res_kernel_size=res_kernel_size, causal=causal),
                ResidualUnit(n_out // 2, n_out // 2, dilation=3, res_kernel_size=res_kernel_size, causal=causal),
                ResidualUnit(n_out // 2, n_out // 2, dilation=5, res_kernel_size=res_kernel_size, causal=causal),
                ResidualUnit(n_out // 2, n_out // 2, dilation=7, res_kernel_size=res_kernel_size, causal=causal),
                ResidualUnit(n_out // 2, n_out // 2, dilation=9, res_kernel_size=res_kernel_size, causal=causal),
            ]
        )
        self.down_conv = DownsampleLayer(
            n_in,
            n_out,
            down_kernel_size,
            stride=stride,
            causal=causal,
            activation=nn.PReLU(),
            pooling=False,
        )

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return self.down_conv(x)


class ResDecoderBlock(nn.Module):
    def __init__(self, n_in, n_out, stride, up_kernel_size, res_kernel_size=7, causal=False):
        super().__init__()
        self.up_conv = UpsampleLayer(
            n_in,
            n_out,
            kernel_size=up_kernel_size,
            stride=stride,
            causal=causal,
            activation=None,
            repeat=False,
        )
        self.convs = nn.ModuleList(
            [
                ResidualUnit(n_out, n_out, dilation=1, res_kernel_size=res_kernel_size, causal=causal),
                ResidualUnit(n_out, n_out, dilation=3, res_kernel_size=res_kernel_size, causal=causal),
                ResidualUnit(n_out, n_out, dilation=5, res_kernel_size=res_kernel_size, causal=causal),
                ResidualUnit(n_out, n_out, dilation=7, res_kernel_size=res_kernel_size, causal=causal),
                ResidualUnit(n_out, n_out, dilation=9, res_kernel_size=res_kernel_size, causal=causal),
            ]
        )

    def forward(self, x):
        x = self.up_conv(x)
        for conv in self.convs:
            x = conv(x)
        return x


class DownsampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        activation=nn.PReLU(),
        use_weight_norm: bool = True,
        pooling: bool = False,
    ):
        super().__init__()
        self.pooling = pooling
        self.stride = stride
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        if pooling:
            layer = Conv1d(in_channels, out_channels, kernel_size, causal=causal)
            self.pooling = nn.AvgPool1d(kernel_size=stride)
        else:
            layer = Conv1d(in_channels, out_channels, kernel_size, stride=stride, causal=causal)
        if use_weight_norm:
            layer = weight_norm(layer)
        self.layer = layer

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x) if self.activation is not None else x
        if self.pooling:
            x = self.pooling(x)
        return x

    def remove_weight_norm(self):
        if self.use_weight_norm:
            remove_weight_norm(self.layer)


class UpsampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        causal: bool = False,
        activation=nn.PReLU(),
        use_weight_norm: bool = True,
        repeat: bool = False,
    ):
        super().__init__()
        self.repeat = repeat
        self.stride = stride
        self.activation = activation
        self.use_weight_norm = use_weight_norm
        if repeat:
            layer = Conv1d(in_channels, out_channels, kernel_size, causal=causal)
        else:
            layer = ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, causal=causal)
        if use_weight_norm:
            layer = weight_norm(layer)
        self.layer = layer

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x) if self.activation is not None else x
        if self.repeat:
            x = x.transpose(1, 2)
            batch, length, channels = x.size()
            x = x.repeat(1, 1, self.stride).view(batch, -1, channels)
            x = x.transpose(1, 2)
        return x

    def remove_weight_norm(self):
        if self.use_weight_norm:
            remove_weight_norm(self.layer)


class round_func9(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return torch.round(9 * input) / 9

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class ScalarModel(nn.Module):
    def __init__(
        self,
        num_bands,
        sample_rate,
        causal,
        num_samples,
        downsample_factors,
        downsample_kernel_sizes,
        upsample_factors,
        upsample_kernel_sizes,
        latent_hidden_dim,
        default_kernel_size,
        delay_kernel_size,
        init_channel,
        res_kernel_size,
        mode="pre_proj",
    ):
        super().__init__()
        self.vq = round_func9()
        self.mode = mode
        encoder = []
        decoder = []

        encoder.append(weight_norm(Conv1d(num_bands, init_channel, kernel_size=default_kernel_size, causal=causal)))
        if num_samples > 1:
            encoder.append(
                PreProcessor(
                    init_channel,
                    init_channel,
                    num_samples,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        for idx, down_factor in enumerate(downsample_factors):
            encoder.append(
                ResEncoderBlock(
                    init_channel * np.power(2, idx),
                    init_channel * np.power(2, idx + 1),
                    down_factor,
                    downsample_kernel_sizes[idx],
                    res_kernel_size,
                    causal=causal,
                )
            )
        encoder.append(
            weight_norm(
                Conv1d(
                    init_channel * np.power(2, len(downsample_factors)),
                    latent_hidden_dim,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        )

        decoder.append(
            weight_norm(
                Conv1d(
                    latent_hidden_dim,
                    init_channel * np.power(2, len(upsample_factors)),
                    kernel_size=delay_kernel_size,
                )
            )
        )
        for idx, upsample_factor in enumerate(upsample_factors):
            decoder.append(
                ResDecoderBlock(
                    init_channel * np.power(2, len(upsample_factors) - idx),
                    init_channel * np.power(2, len(upsample_factors) - idx - 1),
                    upsample_factor,
                    upsample_kernel_sizes[idx],
                    res_kernel_size,
                    causal=causal,
                )
            )
        if num_samples > 1:
            decoder.append(
                PostProcessor(
                    init_channel,
                    init_channel,
                    num_samples,
                    kernel_size=default_kernel_size,
                    causal=causal,
                )
            )
        decoder.append(weight_norm(Conv1d(init_channel, num_bands, kernel_size=default_kernel_size, causal=causal)))

        self.encoder = nn.ModuleList(encoder)
        self.decoder = nn.ModuleList(decoder)

    def _encode_core(self, x):
        for i, layer in enumerate(self.encoder):
            if i != len(self.encoder) - 1:
                x = layer(x)
            else:
                x = torch.tanh(layer(x))
        return x

    def forward(self, x):
        x = self._encode_core(x)
        x = self.vq.apply(x)
        for layer in self.decoder:
            x = layer(x)
        return x

    def inference(self, x):
        emb = self._encode_core(x)
        emb_quant = self.vq.apply(emb)
        x = emb_quant
        for layer in self.decoder:
            x = layer(x)
        return emb, emb_quant, x

    def encode(self, x):
        emb = self._encode_core(x)
        self.vq.apply(emb)
        return emb

    def decode(self, x):
        x = self.vq.apply(x)
        for layer in self.decoder:
            x = layer(x)
        return x
