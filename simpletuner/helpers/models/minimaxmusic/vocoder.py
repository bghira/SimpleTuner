# Copyright 2026 The MiniMax Team and The HuggingFace Team. All rights reserved.
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
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.nn.utils import weight_norm


class MiniMaxMusic3Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * hidden_states).pow(2)
        return hidden_states.reshape(shape)


class MiniMaxMusic3VocoderResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        pad = (7 - 1) * dilation // 2
        self.snake1 = MiniMaxMusic3Snake1d(dim)
        self.conv1 = weight_norm(nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad))
        self.snake2 = MiniMaxMusic3Snake1d(dim)
        self.conv2 = weight_norm(nn.Conv1d(dim, dim, kernel_size=1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.snake2(self.conv1(self.snake1(hidden_states))))
        return hidden_states + residual


class MiniMaxMusic3VocoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.snake1 = MiniMaxMusic3Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(input_dim, output_dim, kernel_size=2 * stride, stride=stride, padding=math.ceil(stride / 2))
        )
        self.res_unit1 = MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=1)
        self.res_unit2 = MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=3)
        self.res_unit3 = MiniMaxMusic3VocoderResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_t1(self.snake1(hidden_states))
        hidden_states = self.res_unit1(hidden_states)
        hidden_states = self.res_unit2(hidden_states)
        return self.res_unit3(hidden_states)


class MiniMaxMusic3Vocoder(ModelMixin, ConfigMixin):
    r"""
    The Flow-VAE waveform decoder of MiniMax Music 3 (a DAC-style decoder). It decodes flow-matched latents of shape
    `(batch, latent_channels, length)` into stereo waveforms at `sampling_rate`; the two audio channels are decoded as
    two folded `latent_channels // 2` streams.
    """

    @register_to_config
    def __init__(
        self,
        latent_channels: int = 128,
        decoder_input_dim: int = 1024,
        decoder_hidden_dim: int = 1536,
        upsampling_ratios: tuple = (8, 8, 4, 2),
        sampling_rate: int = 44100,
    ):
        super().__init__()
        self.dec_in_proj = nn.Conv1d(latent_channels // 2, decoder_input_dim, kernel_size=1)
        self.conv_in = weight_norm(nn.Conv1d(decoder_input_dim, decoder_hidden_dim, kernel_size=7, padding=3))
        blocks = []
        output_dim = decoder_hidden_dim
        for index, stride in enumerate(upsampling_ratios):
            input_dim = decoder_hidden_dim // (2**index)
            output_dim = decoder_hidden_dim // (2 ** (index + 1))
            blocks.append(MiniMaxMusic3VocoderBlock(input_dim, output_dim, stride))
        self.blocks = nn.ModuleList(blocks)
        self.snake_out = MiniMaxMusic3Snake1d(output_dim)
        self.conv_out = weight_norm(nn.Conv1d(output_dim, 1, kernel_size=7, padding=3))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            latents (`torch.Tensor` of shape `(batch, latent_channels, length)`):
                Flow-matched Flow-VAE latents.

        Returns:
            `torch.Tensor` of shape `(batch, 2, samples)`: the stereo waveform in `[-1, 1]`.
        """
        batch_size, _, length = latents.shape
        hidden_states = latents.reshape(batch_size * 2, self.config.latent_channels // 2, length)
        hidden_states = self.conv_in(self.dec_in_proj(hidden_states))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        waveform = torch.tanh(self.conv_out(self.snake_out(hidden_states)))
        return waveform.reshape(batch_size, 2, -1)


class MiniMaxMusic3DAVResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int):
        super().__init__()
        padding = 3 * dilation
        self.block = nn.Sequential(
            MiniMaxMusic3Snake1d(dim),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=padding)),
            MiniMaxMusic3Snake1d(dim),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=1)),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = self.block(hidden_states)
        if residual.shape[-1] != hidden_states.shape[-1]:
            padding = (hidden_states.shape[-1] - residual.shape[-1]) // 2
            hidden_states = hidden_states[..., padding : hidden_states.shape[-1] - padding]
        return hidden_states + residual


class MiniMaxMusic3DAVEncoderBlock(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            MiniMaxMusic3DAVResidualUnit(dim // 2, dilation=1),
            MiniMaxMusic3DAVResidualUnit(dim // 2, dilation=3),
            MiniMaxMusic3DAVResidualUnit(dim // 2, dilation=9),
            MiniMaxMusic3Snake1d(dim // 2),
            weight_norm(
                nn.Conv1d(
                    dim // 2,
                    dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                )
            ),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class MiniMaxMusic3DAVEncoder(nn.Module):
    def __init__(self, encoder_dim: int, encoder_rates: tuple[int, ...], latent_dim: int):
        super().__init__()
        block: list[nn.Module] = [weight_norm(nn.Conv1d(1, encoder_dim, kernel_size=7, padding=3))]
        for stride in encoder_rates:
            encoder_dim *= 2
            block.append(MiniMaxMusic3DAVEncoderBlock(encoder_dim, stride=stride))
        block.extend(
            (
                MiniMaxMusic3Snake1d(encoder_dim),
                weight_norm(nn.Conv1d(encoder_dim, latent_dim, kernel_size=3, padding=1)),
            )
        )
        self.block = nn.Sequential(*block)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class MiniMaxMusic3DAVDecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            MiniMaxMusic3Snake1d(input_dim),
            weight_norm(
                nn.ConvTranspose1d(
                    input_dim,
                    output_dim,
                    kernel_size=2 * stride,
                    stride=stride,
                    padding=math.ceil(stride / 2),
                )
            ),
            MiniMaxMusic3DAVResidualUnit(output_dim, dilation=1),
            MiniMaxMusic3DAVResidualUnit(output_dim, dilation=3),
            MiniMaxMusic3DAVResidualUnit(output_dim, dilation=9),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.block(hidden_states)


class MiniMaxMusic3DAVDecoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, upsampling_ratios: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = [weight_norm(nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3))]
        output_dim = hidden_dim
        for index, stride in enumerate(upsampling_ratios):
            input_channels = hidden_dim // (2**index)
            output_dim = hidden_dim // (2 ** (index + 1))
            layers.append(MiniMaxMusic3DAVDecoderBlock(input_channels, output_dim, stride=stride))
        layers.extend(
            (
                MiniMaxMusic3Snake1d(output_dim),
                weight_norm(nn.Conv1d(output_dim, 1, kernel_size=7, padding=3)),
                nn.Tanh(),
            )
        )
        self.model = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model(hidden_states)


class MiniMaxMusic3DAV(ModelMixin, ConfigMixin):
    r"""
    The original MiniMax Music 3 DAV checkpoint layout used for VAECache audio encoding and waveform decoding.

    The public Diffusers layout exposes the decoder as `vocoder/`; the original `dav.pth` also contains the DAC-style
    waveform encoder and posterior projection heads needed to cache raw audio examples.
    """

    @register_to_config
    def __init__(
        self,
        latent_channels: int = 128,
        channel_latent_channels: int = 64,
        encoder_dim: int = 64,
        encoder_rates: tuple[int, ...] = (2, 4, 8, 8),
        encoder_latent_dim: int = 1024,
        decoder_input_dim: int = 1024,
        decoder_hidden_dim: int = 1536,
        upsampling_ratios: tuple[int, ...] = (8, 8, 4, 2),
        sampling_rate: int = 44100,
    ):
        super().__init__()
        if channel_latent_channels * 2 != latent_channels:
            raise ValueError("MiniMax Music 3 DAV expects latent_channels to be exactly twice channel_latent_channels.")
        self.hop_length = math.prod(encoder_rates)
        self.encoder = MiniMaxMusic3DAVEncoder(
            encoder_dim=encoder_dim,
            encoder_rates=encoder_rates,
            latent_dim=encoder_latent_dim,
        )
        self.mean_proj = nn.Conv1d(encoder_latent_dim, channel_latent_channels, kernel_size=1)
        self.logs_proj = nn.Conv1d(encoder_latent_dim, channel_latent_channels, kernel_size=1)
        self.dec_in_proj = nn.Conv1d(channel_latent_channels, decoder_input_dim, kernel_size=1)
        self.decoder = MiniMaxMusic3DAVDecoder(
            input_dim=decoder_input_dim,
            hidden_dim=decoder_hidden_dim,
            upsampling_ratios=upsampling_ratios,
        )

    @classmethod
    def from_original_dav(cls, checkpoint_file: str) -> "MiniMaxMusic3DAV":
        model = cls()
        state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state_dict, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("flow.")]
        if incompatible.missing_keys or unexpected:
            details = []
            if incompatible.missing_keys:
                details.append(f"missing keys: {incompatible.missing_keys[:8]}")
            if unexpected:
                details.append(f"unexpected keys: {unexpected[:8]}")
            raise RuntimeError(f"Failed to load MiniMax Music 3 DAV checkpoint ({'; '.join(details)}).")
        return model

    def _prepare_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            raise ValueError("MiniMax Music 3 DAV expects audio tensors shaped [batch, channels, samples].")
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        elif waveform.shape[1] != 2:
            raise ValueError("MiniMax Music 3 DAV expects mono or stereo audio.")

        remainder = waveform.shape[-1] % self.hop_length
        if remainder:
            waveform = torch.nn.functional.pad(waveform, (0, self.hop_length - remainder))
        return waveform

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = self._prepare_waveform(waveform)
        batch_size, _, _ = waveform.shape
        hidden_states = waveform.reshape(batch_size * 2, 1, -1)
        hidden_states = self.encoder(hidden_states)
        latents = self.mean_proj(hidden_states)
        return latents.reshape(batch_size, self.config.latent_channels, -1)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim != 3:
            raise ValueError("MiniMax Music 3 DAV expects latents shaped [batch, channels, frames].")
        if latents.shape[1] != self.config.latent_channels:
            raise ValueError(f"MiniMax Music 3 DAV expects {self.config.latent_channels} latent channels.")
        batch_size, _, length = latents.shape
        hidden_states = latents.reshape(batch_size * 2, self.config.channel_latent_channels, length)
        waveform = self.decoder(self.dec_in_proj(hidden_states))
        return waveform.reshape(batch_size, 2, -1)

    forward = decode
