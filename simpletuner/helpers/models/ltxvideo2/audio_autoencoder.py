# Copyright 2025 The Lightricks team and The HuggingFace Team.
# All rights reserved.
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

from typing import List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders.vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.accelerate_utils import apply_forward_hook

LATENT_DOWNSAMPLE_FACTOR = 4


def _require_torchaudio():
    try:
        import torchaudio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torchaudio is required for LTX-2 audio encoding.") from exc
    return torchaudio


class LTX2AudioProcessor(nn.Module):
    """Waveform to log-mel processor used by the LTX-2 audio encoder."""

    def __init__(
        self,
        sample_rate: int,
        mel_bins: int,
        mel_hop_length: int,
        n_fft: int,
    ) -> None:
        super().__init__()
        torchaudio = _require_torchaudio()
        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=mel_hop_length,
            f_min=0.0,
            f_max=sample_rate / 2.0,
            n_mels=mel_bins,
            window_fn=torch.hann_window,
            center=True,
            pad_mode="reflect",
            power=1.0,
            mel_scale="slaney",
            norm="slaney",
        )

    def resample_waveform(self, waveform: torch.Tensor, source_rate: int, target_rate: int) -> torch.Tensor:
        if source_rate == target_rate:
            return waveform
        torchaudio = _require_torchaudio()
        resampled = torchaudio.functional.resample(waveform, source_rate, target_rate)
        return resampled.to(device=waveform.device, dtype=waveform.dtype)

    def waveform_to_mel(self, waveform: torch.Tensor, waveform_sample_rate: int) -> torch.Tensor:
        waveform = self.resample_waveform(waveform, waveform_sample_rate, self.sample_rate)
        mel = self.mel_transform(waveform)
        mel = torch.log(torch.clamp(mel, min=1e-5))
        mel = mel.to(device=waveform.device, dtype=waveform.dtype)
        return mel.permute(0, 1, 3, 2).contiguous()


class LTX2AudioPerChannelStatistics(nn.Module):
    """
    Per-channel statistics for normalizing and denormalizing the latent representation.
    """

    def __init__(self, latent_channels: int) -> None:
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.get_buffer("std-of-means").to(x)) + self.get_buffer("mean-of-means").to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer("mean-of-means").to(x)) / self.get_buffer("std-of-means").to(x)


class LTX2AudioCausalConv2d(nn.Module):
    """
    A causal 2D convolution that pads asymmetrically along the causal axis.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: int = 1,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        causality_axis: str = "height",
    ) -> None:
        super().__init__()

        self.causality_axis = causality_axis
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]

        if self.causality_axis == "none":
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        elif self.causality_axis in {"width", "width-compatibility"}:
            padding = (pad_w, 0, pad_h // 2, pad_h - pad_h // 2)
        elif self.causality_axis == "height":
            padding = (pad_w // 2, pad_w - pad_w // 2, pad_h, 0)
        else:
            raise ValueError(f"Invalid causality_axis: {causality_axis}")

        self.padding = padding
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding)
        return self.conv(x)


class LTX2AudioPixelNorm(nn.Module):
    """
    Per-pixel (per-location) RMS normalization layer.
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        return x / rms


class LTX2AudioAttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        norm_type: str = "group",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels

        if norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        elif norm_type == "pixel":
            self.norm = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        batch, channels, height, width = q.shape
        q = q.reshape(batch, channels, height * width).permute(0, 2, 1).contiguous()
        k = k.reshape(batch, channels, height * width).contiguous()
        attn = torch.bmm(q, k) * (int(channels) ** (-0.5))
        attn = torch.nn.functional.softmax(attn, dim=2)

        v = v.reshape(batch, channels, height * width)
        attn = attn.permute(0, 2, 1).contiguous()
        h_ = torch.bmm(v, attn).reshape(batch, channels, height, width)

        h_ = self.proj_out(h_)
        return x + h_


class LTX2AudioResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        norm_type: str = "group",
        causality_axis: str = "height",
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis

        if self.causality_axis is not None and self.causality_axis != "none" and norm_type == "group":
            raise ValueError("Causal ResnetBlock with GroupNorm is not supported.")
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        if norm_type == "group":
            self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        elif norm_type == "pixel":
            self.norm1 = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        self.non_linearity = nn.SiLU()
        if causality_axis is not None:
            self.conv1 = LTX2AudioCausalConv2d(
                in_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)
        if norm_type == "group":
            self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        elif norm_type == "pixel":
            self.norm2 = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {norm_type}")
        self.dropout = nn.Dropout(dropout)
        if causality_axis is not None:
            self.conv2 = LTX2AudioCausalConv2d(
                out_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis
            )
        else:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                if causality_axis is not None:
                    self.conv_shortcut = LTX2AudioCausalConv2d(
                        in_channels, out_channels, kernel_size=3, stride=1, causality_axis=causality_axis
                    )
                else:
                    self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                if causality_axis is not None:
                    self.nin_shortcut = LTX2AudioCausalConv2d(
                        in_channels, out_channels, kernel_size=1, stride=1, causality_axis=causality_axis
                    )
                else:
                    self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h = self.non_linearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.non_linearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.non_linearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)

        return x + h


class LTX2AudioUpsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool, causality_axis: Optional[str] = "height") -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.with_conv:
            if causality_axis is not None:
                self.conv = LTX2AudioCausalConv2d(
                    in_channels, in_channels, kernel_size=3, stride=1, causality_axis=causality_axis
                )
            else:
                self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
            if self.causality_axis is None or self.causality_axis == "none":
                pass
            elif self.causality_axis == "height":
                x = x[:, :, 1:, :]
            elif self.causality_axis == "width":
                x = x[:, :, :, 1:]
            elif self.causality_axis == "width-compatibility":
                pass
            else:
                raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

        return x


class LTX2AudioDownsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool, causality_axis: Optional[str] = "height") -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.causality_axis is not None and self.causality_axis != "none" and not self.with_conv:
            raise ValueError("Causal downsampling requires with_conv=True.")
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            if self.causality_axis is None or self.causality_axis == "none":
                pad = (0, 1, 0, 1)
            elif self.causality_axis == "width":
                pad = (2, 0, 0, 1)
            elif self.causality_axis == "height":
                pad = (0, 1, 2, 0)
            elif self.causality_axis == "width-compatibility":
                pad = (1, 0, 0, 1)
            else:
                raise ValueError(f"Invalid causality_axis: {self.causality_axis}")
            x = F.pad(x, pad, mode="constant", value=0)
            return self.conv(x)
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class LTX2AudioAudioPatchifier:
    """
    Patchifier for spectrogram/audio latents.
    """

    def __init__(
        self,
        patch_size: int,
        sample_rate: int = 16000,
        hop_length: int = 160,
        audio_latent_downsample_factor: int = 4,
        is_causal: bool = True,
    ):
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.audio_latent_downsample_factor = audio_latent_downsample_factor
        self.is_causal = is_causal
        self._patch_size = (1, patch_size, patch_size)

    def patchify(self, audio_latents: torch.Tensor) -> torch.Tensor:
        batch, channels, time, freq = audio_latents.shape
        return audio_latents.permute(0, 2, 1, 3).reshape(batch, time, channels * freq)

    def unpatchify(self, audio_latents: torch.Tensor, channels: int, mel_bins: int) -> torch.Tensor:
        batch, time, _ = audio_latents.shape
        return audio_latents.view(batch, time, channels, mel_bins).permute(0, 2, 1, 3)

    @property
    def patch_size(self) -> Tuple[int, int, int]:
        return self._patch_size


class LTX2AudioDecoder(nn.Module):
    """
    Symmetric decoder that reconstructs audio spectrograms from latent features.

    The decoder mirrors the encoder structure with configurable channel multipliers, attention resolutions, and causal
    convolutions.
    """

    def __init__(
        self,
        base_channels: int,
        output_channels: int,
        num_res_blocks: int,
        attn_resolutions: Set[int],
        in_channels: int,
        resolution: int,
        latent_channels: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        norm_type: str = "group",
        causality_axis: Optional[str] = "width",
        dropout: float = 0.0,
        mid_block_add_attention: bool = True,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.mel_bins = mel_bins
        self.patchifier = LTX2AudioAudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.base_channels = base_channels
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.out_ch = output_channels
        self.give_pre_end = False
        self.tanh_out = False
        self.norm_type = norm_type
        self.latent_channels = latent_channels
        self.channel_multipliers = ch_mult
        self.attn_resolutions = attn_resolutions
        self.causality_axis = causality_axis

        base_block_channels = base_channels * self.channel_multipliers[-1]
        base_resolution = resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, latent_channels, base_resolution, base_resolution)

        if self.causality_axis is not None:
            self.conv_in = LTX2AudioCausalConv2d(
                latent_channels, base_block_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
            )
        else:
            self.conv_in = nn.Conv2d(latent_channels, base_block_channels, kernel_size=3, stride=1, padding=1)
        self.non_linearity = nn.SiLU()
        self.mid = nn.Module()
        self.mid.block_1 = LTX2AudioResnetBlock(
            in_channels=base_block_channels,
            out_channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )
        if mid_block_add_attention:
            self.mid.attn_1 = LTX2AudioAttnBlock(base_block_channels, norm_type=self.norm_type)
        else:
            self.mid.attn_1 = nn.Identity()
        self.mid.block_2 = LTX2AudioResnetBlock(
            in_channels=base_block_channels,
            out_channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )

        self.up = nn.ModuleList()
        block_in = base_block_channels
        curr_res = self.resolution // (2 ** (self.num_resolutions - 1))

        for level in reversed(range(self.num_resolutions)):
            stage = nn.Module()
            stage.block = nn.ModuleList()
            stage.attn = nn.ModuleList()
            block_out = self.base_channels * self.channel_multipliers[level]

            for _ in range(self.num_res_blocks + 1):
                stage.block.append(
                    LTX2AudioResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        norm_type=self.norm_type,
                        causality_axis=self.causality_axis,
                    )
                )
                block_in = block_out
                if self.attn_resolutions:
                    if curr_res in self.attn_resolutions:
                        stage.attn.append(LTX2AudioAttnBlock(block_in, norm_type=self.norm_type))

            if level != 0:
                stage.upsample = LTX2AudioUpsample(block_in, True, causality_axis=self.causality_axis)
                curr_res *= 2

            self.up.insert(0, stage)

        final_block_channels = block_in

        if self.norm_type == "group":
            self.norm_out = nn.GroupNorm(num_groups=32, num_channels=final_block_channels, eps=1e-6, affine=True)
        elif self.norm_type == "pixel":
            self.norm_out = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {self.norm_type}")

        if self.causality_axis is not None:
            self.conv_out = LTX2AudioCausalConv2d(
                final_block_channels, output_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
            )
        else:
            self.conv_out = nn.Conv2d(final_block_channels, output_channels, kernel_size=3, stride=1, padding=1)

    def forward(
        self,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        _, _, frames, mel_bins = sample.shape

        target_frames = frames * LATENT_DOWNSAMPLE_FACTOR

        if self.causality_axis is not None:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        target_channels = self.out_ch
        target_mel_bins = self.mel_bins if self.mel_bins is not None else mel_bins

        hidden_features = self.conv_in(sample)
        hidden_features = self.mid.block_1(hidden_features, temb=None)
        hidden_features = self.mid.attn_1(hidden_features)
        hidden_features = self.mid.block_2(hidden_features, temb=None)

        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_idx, block in enumerate(stage.block):
                hidden_features = block(hidden_features, temb=None)
                if stage.attn:
                    hidden_features = stage.attn[block_idx](hidden_features)

            if level != 0 and hasattr(stage, "upsample"):
                hidden_features = stage.upsample(hidden_features)

        if self.give_pre_end:
            return hidden_features

        hidden = self.norm_out(hidden_features)
        hidden = self.non_linearity(hidden)
        decoded_output = self.conv_out(hidden)
        decoded_output = torch.tanh(decoded_output) if self.tanh_out else decoded_output

        _, _, current_time, current_freq = decoded_output.shape
        target_time = target_frames
        target_freq = target_mel_bins

        decoded_output = decoded_output[
            :, :target_channels, : min(current_time, target_time), : min(current_freq, target_freq)
        ]

        time_padding_needed = target_time - decoded_output.shape[2]
        freq_padding_needed = target_freq - decoded_output.shape[3]

        if time_padding_needed > 0 or freq_padding_needed > 0:
            padding = (
                0,
                max(freq_padding_needed, 0),
                0,
                max(time_padding_needed, 0),
            )
            decoded_output = F.pad(decoded_output, padding)

        decoded_output = decoded_output[:, :target_channels, :target_time, :target_freq]

        return decoded_output


class LTX2AudioEncoder(nn.Module):
    """
    Encoder that compresses audio spectrograms into latent representations.
    """

    def __init__(
        self,
        base_channels: int,
        num_res_blocks: int,
        attn_resolutions: Set[int],
        in_channels: int,
        resolution: int,
        latent_channels: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        norm_type: str = "group",
        causality_axis: Optional[str] = "height",
        dropout: float = 0.0,
        mid_block_add_attention: bool = True,
        resamp_with_conv: bool = True,
        double_z: bool = True,
    ) -> None:
        super().__init__()
        self.base_channels = base_channels
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.double_z = double_z
        self.norm_type = norm_type
        self.causality_axis = causality_axis
        self.attn_resolutions = attn_resolutions
        self.non_linearity = nn.SiLU()

        if self.causality_axis is not None:
            self.conv_in = LTX2AudioCausalConv2d(
                in_channels, base_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
            )
        else:
            self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList()
        curr_res = resolution
        in_ch_mult = (1, *tuple(ch_mult))
        block_in = base_channels

        for level in range(self.num_resolutions):
            stage = nn.Module()
            stage.block = nn.ModuleList()
            stage.attn = nn.ModuleList()
            block_in = base_channels * in_ch_mult[level]
            block_out = base_channels * ch_mult[level]

            for _ in range(self.num_res_blocks):
                stage.block.append(
                    LTX2AudioResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        norm_type=self.norm_type,
                        causality_axis=self.causality_axis,
                    )
                )
                block_in = block_out
                if curr_res in self.attn_resolutions:
                    stage.attn.append(LTX2AudioAttnBlock(block_in, norm_type=self.norm_type))

            if level != self.num_resolutions - 1:
                stage.downsample = LTX2AudioDownsample(block_in, resamp_with_conv, causality_axis=self.causality_axis)
                curr_res //= 2

            self.down.append(stage)

        self.mid = nn.Module()
        self.mid.block_1 = LTX2AudioResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )
        if mid_block_add_attention:
            self.mid.attn_1 = LTX2AudioAttnBlock(block_in, norm_type=self.norm_type)
        else:
            self.mid.attn_1 = nn.Identity()
        self.mid.block_2 = LTX2AudioResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
        )

        if self.norm_type == "group":
            self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        elif self.norm_type == "pixel":
            self.norm_out = LTX2AudioPixelNorm(dim=1, eps=1e-6)
        else:
            raise ValueError(f"Invalid normalization type: {self.norm_type}")

        out_channels = latent_channels * 2 if double_z else latent_channels
        if self.causality_axis is not None:
            self.conv_out = LTX2AudioCausalConv2d(
                block_in, out_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
            )
        else:
            self.conv_out = nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        hidden = self.conv_in(spectrogram)
        for level in range(self.num_resolutions):
            stage = self.down[level]
            for block_idx in range(self.num_res_blocks):
                hidden = stage.block[block_idx](hidden, temb=None)
                if stage.attn:
                    hidden = stage.attn[block_idx](hidden)
            if level != self.num_resolutions - 1 and hasattr(stage, "downsample"):
                hidden = stage.downsample(hidden)

        hidden = self.mid.block_1(hidden, temb=None)
        hidden = self.mid.attn_1(hidden)
        hidden = self.mid.block_2(hidden, temb=None)

        hidden = self.norm_out(hidden)
        hidden = self.non_linearity(hidden)
        return self.conv_out(hidden)


class AutoencoderKLLTX2Audio(ModelMixin, AutoencoderMixin, ConfigMixin):
    r"""
    LTX2 audio VAE with encoder and decoder support.
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        base_channels: int = 128,
        output_channels: int = 2,
        ch_mult: Tuple[int] = (1, 2, 4),
        num_res_blocks: int = 2,
        attn_resolutions: Optional[Tuple[int]] = None,
        in_channels: int = 2,
        resolution: int = 256,
        latent_channels: int = 8,
        double_z: bool = True,
        norm_type: str = "pixel",
        causality_axis: Optional[str] = "height",
        dropout: float = 0.0,
        mid_block_add_attention: bool = False,
        resamp_with_conv: bool = True,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        n_fft: int = 1024,
        is_causal: bool = True,
        mel_bins: Optional[int] = 64,
    ) -> None:
        super().__init__()

        supported_causality_axes = {"none", "width", "height", "width-compatibility"}
        if causality_axis not in supported_causality_axes:
            raise ValueError(f"{causality_axis=} is not valid. Supported values: {supported_causality_axes}")

        attn_resolution_set = set(attn_resolutions) if attn_resolutions else attn_resolutions

        self.encoder = LTX2AudioEncoder(
            base_channels=base_channels,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolution_set or set(),
            in_channels=in_channels,
            resolution=resolution,
            latent_channels=latent_channels,
            ch_mult=ch_mult,
            norm_type=norm_type,
            causality_axis=causality_axis,
            dropout=dropout,
            mid_block_add_attention=mid_block_add_attention,
            resamp_with_conv=resamp_with_conv,
            double_z=double_z,
        )
        self.decoder = LTX2AudioDecoder(
            base_channels=base_channels,
            output_channels=output_channels,
            ch_mult=ch_mult,
            num_res_blocks=num_res_blocks,
            attn_resolutions=attn_resolution_set,
            in_channels=in_channels,
            resolution=resolution,
            latent_channels=latent_channels,
            norm_type=norm_type,
            causality_axis=causality_axis,
            dropout=dropout,
            mid_block_add_attention=mid_block_add_attention,
            sample_rate=sample_rate,
            mel_hop_length=mel_hop_length,
            is_causal=is_causal,
            mel_bins=mel_bins,
        )

        latent_mel_bins = None
        if mel_bins is not None:
            mel_downsample_factor = 2 ** (len(ch_mult) - 1)
            latent_mel_bins = mel_bins // mel_downsample_factor
        per_channel_features = base_channels if latent_mel_bins is None else latent_channels * latent_mel_bins
        self.per_channel_statistics = LTX2AudioPerChannelStatistics(latent_channels=per_channel_features)
        self.register_buffer("latents_mean", torch.zeros((latent_channels,), requires_grad=False), persistent=True)
        self.register_buffer("latents_std", torch.ones((latent_channels,), requires_grad=False), persistent=True)
        self._latent_patchifier = LTX2AudioAudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        # TODO: calculate programmatically instead of hardcoding
        self.temporal_compression_ratio = LATENT_DOWNSAMPLE_FACTOR  # 4
        # TODO: confirm whether the mel compression ratio below is correct
        self.mel_compression_ratio = LATENT_DOWNSAMPLE_FACTOR
        self.use_slicing = False
        self._audio_processor = None
        self.n_fft = n_fft
        self.in_channels = in_channels
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.mel_bins = mel_bins

    def _normalize_latents(self, latent_output: torch.Tensor) -> torch.Tensor:
        means = latent_output
        if self.encoder.double_z:
            means = torch.chunk(latent_output, 2, dim=1)[0]
        _, channels, _, mel_bins = means.shape
        latent_patched = self._latent_patchifier.patchify(means)
        latent_normalized = self.per_channel_statistics.normalize(latent_patched)
        return self._latent_patchifier.unpatchify(latent_normalized, channels=channels, mel_bins=mel_bins)

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        _, channels, _, mel_bins = latents.shape
        latents_patched = self._latent_patchifier.patchify(latents)
        latents_denormalized = self.per_channel_statistics.un_normalize(latents_patched)
        return self._latent_patchifier.unpatchify(latents_denormalized, channels=channels, mel_bins=mel_bins)

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True):
        if x.ndim != 4:
            raise ValueError(
                "AutoencoderKLLTX2Audio.encode expects a spectrogram tensor with shape [B, C, T, F]. "
                "Use encode_waveform() to encode raw audio waveforms."
            )
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        if h.shape[1] == self.encoder.latent_channels:
            zeros = torch.zeros_like(h)
            h = torch.cat([h, zeros], dim=1)
        elif h.shape[1] != self.encoder.latent_channels * 2:
            raise ValueError(
                "Audio encoder output channels must be either latent_channels or 2 * latent_channels, " f"got {h.shape}."
            )
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        latent_output = self.encoder(x)
        return self._normalize_latents(latent_output)

    def _get_audio_processor(self) -> LTX2AudioProcessor:
        if self._audio_processor is None:
            mel_bins = self.mel_bins if self.mel_bins is not None else 64
            self._audio_processor = LTX2AudioProcessor(
                sample_rate=self.sample_rate,
                mel_bins=mel_bins,
                mel_hop_length=self.mel_hop_length,
                n_fft=self.n_fft,
            )
        return self._audio_processor

    def encode_waveform(
        self,
        waveform: torch.Tensor,
        sample_rates: Optional[Sequence[int]] = None,
        return_dict: bool = True,
    ):
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.ndim != 3:
            raise ValueError(f"Audio waveform must have shape [B, C, T] or [C, T], received {tuple(waveform.shape)}.")
        waveform = waveform.to(dtype=torch.float32)
        batch_size = waveform.shape[0]
        if sample_rates is None:
            sample_rates = [self.sample_rate] * batch_size
        if len(sample_rates) != batch_size:
            raise ValueError("Sample rates length must match batch size for audio encoding.")
        if waveform.shape[1] != self.in_channels:
            raise ValueError(
                f"Audio VAE expects {self.in_channels} channel(s), received {waveform.shape[1]}. "
                "Ensure the dataset audio channel count matches the audio VAE configuration."
            )

        processor = self._get_audio_processor()
        target_rate = self.sample_rate
        if all(int(rate) == target_rate for rate in sample_rates):
            mel = processor.waveform_to_mel(waveform, waveform_sample_rate=target_rate)
        else:
            mel_segments: List[torch.Tensor] = []
            max_frames = 0
            for idx, rate in enumerate(sample_rates):
                mel_item = processor.waveform_to_mel(waveform[idx : idx + 1], waveform_sample_rate=int(rate))
                mel_segments.append(mel_item)
                max_frames = max(max_frames, mel_item.shape[2])
            if max_frames == 0:
                raise ValueError("Unable to compute mel spectrograms for the audio batch.")
            padded = []
            for mel_item in mel_segments:
                pad_frames = max_frames - mel_item.shape[2]
                if pad_frames > 0:
                    mel_item = F.pad(mel_item, (0, 0, 0, pad_frames))
                padded.append(mel_item)
            mel = torch.cat(padded, dim=0)

        encoder_dtype = next(self.encoder.parameters()).dtype
        mel = mel.to(device=waveform.device, dtype=encoder_dtype)
        return self.encode(mel, return_dict=return_dict)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self._denormalize_latents(z)
        return self.decoder(z)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "This model doesn't have an encoder yet so we don't implement its `forward()`. Please use `decode()`."
        )
