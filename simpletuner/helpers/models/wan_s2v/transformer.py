# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, FP32LayerNorm
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers

from simpletuner.helpers.musubi_block_swap import MusubiBlockSwapManager
from simpletuner.helpers.training.tread import TREADRouter

logger = logging.get_logger(__name__)

WAN_S2V_FEED_FORWARD_CHUNK_SIZE = int(os.getenv("WAN_S2V_FEED_FORWARD_CHUNK_SIZE", "0") or 0)
WAN_S2V_FEED_FORWARD_CHUNK_DIM = int(os.getenv("WAN_S2V_FEED_FORWARD_CHUNK_DIM", "0") or 0)


# -----------------------------------------------------------------------------
# Attention Processor
# -----------------------------------------------------------------------------


class WanS2VAttnProcessor2_0:
    """Attention processor for WanS2V using PyTorch 2.0's scaled_dot_product_attention."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanS2VAttnProcessor2_0 requires PyTorch 2.0. Please upgrade PyTorch to 2.0 or higher.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        target_dtype = hidden_states.dtype
        target_device = hidden_states.device

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:
            query = self._apply_rotary_emb(query, rotary_emb)
            key = self._apply_rotary_emb(key, rotary_emb)

        # I2V task - image conditioning
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            if hasattr(attn, "norm_added_k") and attn.norm_added_k is not None:
                key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2)
        hidden_states = hidden_states.to(target_dtype)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img.to(target_dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states

    @staticmethod
    def _apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Apply rotary embeddings to hidden states."""
        batch_size = hidden_states.size(0)
        output = []
        for i in range(batch_size):
            s = hidden_states.size(2)
            x_i = torch.view_as_complex(hidden_states[i, :, :s].to(torch.float64).reshape(hidden_states.size(1), s, -1, 2))
            freqs_i = freqs[i, :s]
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            output.append(x_i)
        return torch.stack(output).transpose(1, 2).type_as(hidden_states)


# -----------------------------------------------------------------------------
# Audio Processing Modules
# -----------------------------------------------------------------------------


class WanS2VCausalConv1d(nn.Module):
    """Causal 1D convolution with padding."""

    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1, pad_mode="replicate", **kwargs):
        super().__init__()
        self.pad_mode = pad_mode
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)


class WanS2VCausalConvLayer(nn.Module):
    """Causal convolution with normalization and activation."""

    def __init__(self, chan_in, chan_out, kernel_size=3, stride=1, dilation=1, pad_mode="replicate", eps=1e-6, **kwargs):
        super().__init__()
        self.conv = WanS2VCausalConv1d(chan_in, chan_out, kernel_size, stride, dilation, pad_mode, **kwargs)
        self.norm = nn.LayerNorm(chan_out, elementwise_affine=False, eps=eps)
        self.act = nn.SiLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)
        x = self.act(x)
        return x


class WanS2VMotionEncoder(nn.Module):
    """Encodes motion/audio features using causal convolutions."""

    def __init__(self, in_dim: int, hidden_dim: int, num_attention_heads: int, need_global: bool = True):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.need_global = need_global

        self.conv1_local = WanS2VCausalConv1d(in_dim, hidden_dim // 4 * num_attention_heads, 3, stride=1)
        if need_global:
            self.conv1_global = WanS2VCausalConv1d(in_dim, hidden_dim // 4, 3, stride=1)
        self.conv2 = WanS2VCausalConvLayer(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.conv3 = WanS2VCausalConvLayer(hidden_dim // 2, hidden_dim, 3, stride=2)

        if need_global:
            self.final_linear = nn.Linear(hidden_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6)
        self.act = nn.SiLU()
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        residual = x.clone()
        batch_size, num_channels, seq_len = x.shape

        x = self.conv1_local(x)
        x = x.unflatten(1, (self.num_attention_heads, -1)).permute(0, 1, 3, 2).flatten(0, 1)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3)
        padding = self.padding_tokens.repeat(batch_size, x.shape[1], 1, 1)
        x = torch.cat([x, padding], dim=-2)
        x_local = x.clone()

        if not self.need_global:
            return x_local

        x = self.conv1_global(residual)
        x = x.permute(0, 2, 1)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final_linear(x)
        x = x.unflatten(0, (batch_size, -1)).permute(0, 2, 1, 3)

        return x, x_local


class WeightedAverageLayer(nn.Module):
    """Weighted average across Wav2Vec2 layers."""

    def __init__(self, num_layers):
        super().__init__()
        self.weights = nn.Parameter(torch.ones((1, num_layers, 1, 1)) * 0.01)
        self.act = nn.SiLU()

    def forward(self, features):
        # features: B * num_layers * dim * video_length
        weights = self.act(self.weights)
        weights_sum = weights.sum(dim=1, keepdims=True)
        weighted_feat = ((features * weights) / weights_sum).sum(dim=1)
        return weighted_feat


class CausalAudioEncoder(nn.Module):
    """Processes Wav2Vec2 features for S2V conditioning."""

    def __init__(self, dim=5120, num_weighted_avg_layers=25, out_dim=2048, num_audio_token=4, need_global=False):
        super().__init__()
        self.weighted_avg = WeightedAverageLayer(num_weighted_avg_layers)
        self.encoder = WanS2VMotionEncoder(
            in_dim=dim, hidden_dim=out_dim, num_attention_heads=num_audio_token, need_global=need_global
        )

    def forward(self, features):
        # features: B * num_layers * dim * video_length
        weighted_feat = self.weighted_avg(features)
        weighted_feat = weighted_feat.permute(0, 2, 1)  # B F dim
        res = self.encoder(weighted_feat)  # B F N dim
        return res


class AudioInjector(nn.Module):
    """Injects audio features into transformer blocks via cross-attention."""

    def __init__(
        self,
        num_injection_layers: int,
        inject_layers: Tuple[int],
        dim: int = 2048,
        num_heads: int = 32,
        enable_adain: bool = False,
        adain_mode: str = "attn_norm",
        adain_dim: int = 2048,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()
        self.enable_adain = enable_adain
        self.adain_mode = adain_mode
        self.injected_block_id = dict(zip(inject_layers, range(num_injection_layers)))

        # Cross-attention modules for audio injection
        self.injector = nn.ModuleList(
            [
                Attention(
                    query_dim=dim,
                    heads=num_heads,
                    dim_head=dim // num_heads,
                    eps=eps,
                    cross_attention_dim=dim,
                    processor=WanS2VAttnProcessor2_0(),
                )
                for _ in range(num_injection_layers)
            ]
        )

        self.injector_pre_norm_feat = nn.ModuleList(
            [nn.LayerNorm(dim, elementwise_affine=False, eps=eps) for _ in range(num_injection_layers)]
        )
        self.injector_pre_norm_vec = nn.ModuleList(
            [nn.LayerNorm(dim, elementwise_affine=False, eps=eps) for _ in range(num_injection_layers)]
        )

        if enable_adain:
            self.injector_adain_layers = nn.ModuleList(
                [AdaLayerNorm(embedding_dim=adain_dim, output_dim=dim * 2, chunk_dim=1) for _ in range(num_injection_layers)]
            )
            if adain_mode != "attn_norm":
                self.injector_adain_output_layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_injection_layers)])

    def forward(
        self,
        block_idx: int,
        hidden_states: torch.Tensor,
        original_sequence_length: int,
        merged_audio_emb_num_frames: int,
        attn_audio_emb: torch.Tensor,
        audio_emb_global: torch.Tensor,
    ) -> torch.Tensor:
        audio_attn_id = self.injected_block_id[block_idx]

        input_hidden_states = hidden_states[:, :original_sequence_length].clone()
        input_hidden_states = input_hidden_states.unflatten(1, (merged_audio_emb_num_frames, -1)).flatten(0, 1)

        if self.enable_adain and self.adain_mode == "attn_norm":
            attn_hidden_states = self.injector_adain_layers[audio_attn_id](input_hidden_states, temb=audio_emb_global[:, 0])
        else:
            attn_hidden_states = self.injector_pre_norm_feat[audio_attn_id](input_hidden_states)

        residual_out = self.injector[audio_attn_id](attn_hidden_states, attn_audio_emb, None, None)
        residual_out = residual_out.unflatten(0, (-1, merged_audio_emb_num_frames)).flatten(1, 2)
        hidden_states[:, :original_sequence_length] = hidden_states[:, :original_sequence_length] + residual_out

        return hidden_states


# -----------------------------------------------------------------------------
# Motion/Frame Packing
# -----------------------------------------------------------------------------


class WanS2VRotaryPosEmbed(nn.Module):
    """3D Rotary Position Embeddings for S2V."""

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        num_attention_heads: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.num_attention_heads = num_attention_heads

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim
        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs = []
        for dim in [t_dim, h_dim, w_dim]:
            freq = get_1d_rotary_pos_embed(
                dim, max_seq_len, theta, use_real=False, repeat_interleave_real=False, freqs_dtype=freqs_dtype
            )
            freqs.append(freq)

        self.register_buffer("freqs", torch.cat(freqs, dim=1), persistent=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_latents: Optional[torch.Tensor] = None,
        grid_sizes: Optional[List[List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        if grid_sizes is None:
            batch_size, num_channels, num_frames, height, width = hidden_states.shape
            p_t, p_h, p_w = self.patch_size
            ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

            grid_sizes = torch.tensor([ppf, pph, ppw]).unsqueeze(0).repeat(batch_size, 1)
            grid_sizes = [torch.zeros_like(grid_sizes), grid_sizes, grid_sizes]

            image_grid_sizes = [
                torch.tensor([30, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
                torch.tensor([31, image_latents.shape[3] // p_h, image_latents.shape[4] // p_w])
                .unsqueeze(0)
                .repeat(batch_size, 1),
                torch.tensor([1, image_latents.shape[3] // p_h, image_latents.shape[4] // p_w])
                .unsqueeze(0)
                .repeat(batch_size, 1),
            ]

            grids = [grid_sizes, image_grid_sizes]
            S = ppf * pph * ppw + image_latents.shape[3] // p_h * image_latents.shape[4] // p_w
        else:
            batch_size, S, _, _ = hidden_states.shape
            grids = grid_sizes

        split_sizes = [
            self.attention_head_dim // 2 - 2 * (self.attention_head_dim // 6),
            self.attention_head_dim // 6,
            self.attention_head_dim // 6,
        ]
        freqs = self.freqs.split(split_sizes, dim=1)

        output = torch.view_as_complex(
            torch.zeros(
                (batch_size, S, self.num_attention_heads, self.attention_head_dim // 2, 2),
                device=hidden_states.device,
                dtype=torch.float64,
            )
        )

        seq_bucket = [0]
        for g in grids:
            if not isinstance(g, list):
                g = [torch.zeros_like(g), g]
            batch_size = g[0].shape[0]
            for i in range(batch_size):
                f_o, h_o, w_o = g[0][i]
                f, h, w = g[1][i]
                t_f, t_h, t_w = g[2][i]
                seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
                seq_len = int(seq_f * seq_h * seq_w)

                if seq_len > 0 and t_f > 0:
                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1, seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(), (-t_f - f_o).item() + 1, seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1, seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1, seq_w).astype(int).tolist()

                    freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][f_sam].conj()
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1)

                    freqs_i = torch.cat(
                        [
                            freqs_0.expand(seq_f, seq_h, seq_w, -1),
                            freqs[1][h_sam].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                            freqs[2][w_sam].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1),
                        ],
                        dim=-1,
                    ).reshape(seq_len, 1, -1)

                    output[i, seq_bucket[-1] : seq_bucket[-1] + seq_len] = freqs_i
            seq_bucket.append(seq_bucket[-1] + seq_len)

        return output


class FramePackMotioner(nn.Module):
    """Packs motion frames at multiple temporal resolutions."""

    def __init__(
        self,
        inner_dim: int = 1024,
        num_attention_heads: int = 16,
        zip_frame_buckets: List[int] = [1, 2, 16],
        drop_mode: str = "drop",
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 16,
    ):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_attention_heads = num_attention_heads
        self.in_channels = in_channels
        self.drop_mode = drop_mode

        self.proj = nn.Conv3d(in_channels, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(in_channels, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(in_channels, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.register_buffer("zip_frame_buckets", torch.tensor(zip_frame_buckets, dtype=torch.long), persistent=False)

        self.rope = WanS2VRotaryPosEmbed(
            inner_dim // num_attention_heads,
            patch_size=patch_size,
            max_seq_len=1024,
            num_attention_heads=num_attention_heads,
        )

    def forward(self, motion_latents: torch.Tensor, add_last_motion: int = 2):
        latent_height, latent_width = motion_latents.shape[3], motion_latents.shape[4]
        zip_sum = self.zip_frame_buckets.sum().item()

        padd_latent = torch.zeros(
            (motion_latents.shape[0], self.in_channels, zip_sum, latent_height, latent_width),
            device=motion_latents.device,
            dtype=motion_latents.dtype,
        )
        overlap_frame = min(zip_sum, motion_latents.shape[2])
        if overlap_frame > 0:
            padd_latent[:, :, -overlap_frame:] = motion_latents[:, :, -overlap_frame:]

        if add_last_motion < 2 and self.drop_mode != "drop":
            zero_end_frame = self.zip_frame_buckets[: len(self.zip_frame_buckets) - add_last_motion - 1].sum()
            padd_latent[:, :, -zero_end_frame:] = 0

        clean_latents_4x, clean_latents_2x, clean_latents_post = padd_latent[:, :, -zip_sum:, :, :].split(
            list(self.zip_frame_buckets.tolist())[::-1], dim=2
        )

        clean_latents_post = self.proj(clean_latents_post).flatten(2).transpose(1, 2)
        clean_latents_2x = self.proj_2x(clean_latents_2x).flatten(2).transpose(1, 2)
        clean_latents_4x = self.proj_4x(clean_latents_4x).flatten(2).transpose(1, 2)

        if add_last_motion < 2 and self.drop_mode == "drop":
            clean_latents_post = clean_latents_post[:, :0]
            clean_latents_2x = clean_latents_2x[:, :0] if add_last_motion < 1 else clean_latents_2x

        motion_lat = torch.cat([clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)

        # Build grid sizes for RoPE
        grid_sizes = self._build_grid_sizes(add_last_motion, latent_height, latent_width)

        motion_rope_emb = self.rope(
            motion_lat.detach().view(
                motion_lat.shape[0],
                motion_lat.shape[1],
                self.num_attention_heads,
                self.inner_dim // self.num_attention_heads,
            ),
            grid_sizes=grid_sizes,
        )

        return motion_lat, motion_rope_emb

    def _build_grid_sizes(self, add_last_motion: int, latent_height: int, latent_width: int) -> List:
        zfb = self.zip_frame_buckets.tolist()
        grid_sizes = []

        if not (add_last_motion < 2 and self.drop_mode == "drop"):
            start_time_id = -zfb[0]
            end_time_id = start_time_id + zfb[0]
            grid_sizes.append(
                [
                    torch.tensor([start_time_id, 0, 0]).unsqueeze(0),
                    torch.tensor([end_time_id, latent_height // 2, latent_width // 2]).unsqueeze(0),
                    torch.tensor([zfb[0], latent_height // 2, latent_width // 2]).unsqueeze(0),
                ]
            )

        if not (add_last_motion < 1 and self.drop_mode == "drop"):
            start_time_id = -(zfb[0] + zfb[1])
            end_time_id = start_time_id + zfb[1] // 2
            grid_sizes.append(
                [
                    torch.tensor([start_time_id, 0, 0]).unsqueeze(0),
                    torch.tensor([end_time_id, latent_height // 4, latent_width // 4]).unsqueeze(0),
                    torch.tensor([zfb[1], latent_height // 2, latent_width // 2]).unsqueeze(0),
                ]
            )

        start_time_id = -sum(zfb)
        end_time_id = start_time_id + zfb[2] // 4
        grid_sizes.append(
            [
                torch.tensor([start_time_id, 0, 0]).unsqueeze(0),
                torch.tensor([end_time_id, latent_height // 8, latent_width // 8]).unsqueeze(0),
                torch.tensor([zfb[2], latent_height // 2, latent_width // 2]).unsqueeze(0),
            ]
        )

        return grid_sizes


class Motioner(nn.Module):
    """Simple motion encoder without frame packing."""

    def __init__(
        self,
        inner_dim: int,
        num_attention_heads: int,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 16,
        rope_max_seq_len: int = 1024,
    ):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_attention_heads = num_attention_heads

        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)
        self.rope = WanS2VRotaryPosEmbed(inner_dim // num_attention_heads, patch_size, rope_max_seq_len, num_attention_heads)

    def forward(self, motion_latents: torch.Tensor):
        latent_motion_frames = motion_latents.shape[2]
        mot = self.patch_embedding(motion_latents)

        height, width = mot.shape[3], mot.shape[4]
        flat_mot = mot.flatten(2).transpose(1, 2).contiguous()
        motion_grid_sizes = [
            [
                torch.tensor([-latent_motion_frames, 0, 0]).unsqueeze(0),
                torch.tensor([0, height, width]).unsqueeze(0),
                torch.tensor([latent_motion_frames, height, width]).unsqueeze(0),
            ]
        ]
        motion_rope_emb = self.rope(
            flat_mot.detach().view(
                flat_mot.shape[0],
                flat_mot.shape[1],
                self.num_attention_heads,
                self.inner_dim // self.num_attention_heads,
            ),
            grid_sizes=motion_grid_sizes,
        )

        return flat_mot, motion_rope_emb


# -----------------------------------------------------------------------------
# Condition Embeddings
# -----------------------------------------------------------------------------


class WanTimeTextAudioPoseEmbedding(nn.Module):
    """Unified embedding for timestep, text, audio, and pose conditions."""

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        audio_embed_dim: int,
        pose_embed_dim: int,
        patch_size: Tuple[int, int, int],
        enable_adain: bool,
        num_weighted_avg_layers: int,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")
        self.causal_audio_encoder = CausalAudioEncoder(
            dim=audio_embed_dim,
            num_weighted_avg_layers=num_weighted_avg_layers,
            out_dim=dim,
            num_audio_token=4,
            need_global=enable_adain,
        )
        self.pose_embedder = nn.Conv3d(pose_embed_dim, dim, kernel_size=patch_size, stride=patch_size)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        pose_hidden_states: Optional[torch.Tensor] = None,
        timestep_seq_len: Optional[int] = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        audio_hidden_states = self.causal_audio_encoder(audio_hidden_states)
        pose_hidden_states = self.pose_embedder(pose_hidden_states)

        return temb, timestep_proj, encoder_hidden_states, audio_hidden_states, pose_hidden_states


# -----------------------------------------------------------------------------
# Transformer Block
# -----------------------------------------------------------------------------


class WanS2VTransformerBlock(nn.Module):
    """Single transformer block for WanS2V with segment-aware modulation."""

    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
    ):
        super().__init__()

        # Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            processor=WanS2VAttnProcessor2_0(),
        )

        # Cross-attention
        self.attn2 = Attention(
            query_dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim=dim,
            added_kv_proj_dim=added_kv_proj_dim,
            processor=WanS2VAttnProcessor2_0(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: Tuple[torch.Tensor, int],
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        # Segment handling for different timestep embeddings on ref vs motion frames
        seg_idx = temb[1].item() if isinstance(temb[1], torch.Tensor) else temb[1]
        seg_idx = min(max(0, seg_idx), hidden_states.shape[1])
        seg_idx = [0, seg_idx, hidden_states.shape[1]]
        temb = temb[0]

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.unsqueeze(2) + temb.float()
        ).chunk(6, dim=1)

        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)

        norm_hidden_states = self.norm1(hidden_states.float())
        parts = []
        for i in range(2):
            parts.append(
                norm_hidden_states[:, seg_idx[i] : seg_idx[i + 1]] * (1 + scale_msa[:, i : i + 1]) + shift_msa[:, i : i + 1]
            )
        norm_hidden_states = torch.cat(parts, dim=1).type_as(hidden_states)

        # Self-attention
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        z = []
        for i in range(2):
            z.append(attn_output[:, seg_idx[i] : seg_idx[i + 1]] * gate_msa[:, i : i + 1])
        attn_output = torch.cat(z, dim=1)
        hidden_states = (hidden_states.float() + attn_output).type_as(hidden_states)

        # Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # Feed-forward
        norm3_hidden_states = self.norm3(hidden_states.float())
        parts = []
        for i in range(2):
            parts.append(
                norm3_hidden_states[:, seg_idx[i] : seg_idx[i + 1]] * (1 + c_scale_msa[:, i : i + 1])
                + c_shift_msa[:, i : i + 1]
            )
        norm3_hidden_states = torch.cat(parts, dim=1).type_as(hidden_states)
        ff_output = self.ffn(norm3_hidden_states)
        z = []
        for i in range(2):
            z.append(ff_output[:, seg_idx[i] : seg_idx[i + 1]] * c_gate_msa[:, i : i + 1])
        ff_output = torch.cat(z, dim=1)
        hidden_states = (hidden_states.float() + ff_output.float()).type_as(hidden_states)

        return hidden_states


# -----------------------------------------------------------------------------
# Main Transformer Model
# -----------------------------------------------------------------------------


class WanS2VTransformer3DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    Transformer model for Wan2.2-S2V (Speech-to-Video) generation.

    This model extends the base Wan transformer with audio conditioning via
    Wav2Vec2 features and optional pose conditioning.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["WanS2VTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3", "causal_audio_encoder"]
    _tread_router: Optional[TREADRouter] = None
    _tread_routes: Optional[List[Dict[str, Any]]] = None

    @register_to_config
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        audio_dim: int = 1024,
        audio_inject_layers: Tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39),
        enable_adain: bool = True,
        adain_mode: str = "attn_norm",
        pose_dim: int = 16,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        num_weighted_avg_layers: int = 25,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        enable_framepack: bool = True,
        framepack_drop_mode: str = "padd",
        add_last_motion: bool = True,
        zero_timestep: bool = True,
        musubi_blocks_to_swap: int = 0,
        musubi_block_swap_device: str = "cpu",
    ):
        super().__init__()

        self.inner_dim = inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # Patch & position embedding
        self.rope = WanS2VRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len, num_attention_heads)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # Motion handling
        if enable_framepack:
            self.frame_packer = FramePackMotioner(
                inner_dim=inner_dim,
                num_attention_heads=num_attention_heads,
                zip_frame_buckets=[1, 2, 16],
                drop_mode=framepack_drop_mode,
                patch_size=patch_size,
                in_channels=in_channels,
            )
        else:
            self.motion_in = Motioner(
                inner_dim=inner_dim,
                num_attention_heads=num_attention_heads,
                patch_size=patch_size,
                in_channels=in_channels,
                rope_max_seq_len=rope_max_seq_len,
            )

        self.trainable_condition_mask = nn.Embedding(3, inner_dim)

        # Condition embeddings
        self.condition_embedder = WanTimeTextAudioPoseEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            audio_embed_dim=audio_dim,
            pose_embed_dim=pose_dim,
            patch_size=patch_size,
            enable_adain=enable_adain,
            num_weighted_avg_layers=num_weighted_avg_layers,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanS2VTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # Audio injector
        self.audio_injector = AudioInjector(
            num_injection_layers=len(audio_inject_layers),
            inject_layers=audio_inject_layers,
            dim=inner_dim,
            num_heads=num_attention_heads,
            enable_adain=enable_adain,
            adain_dim=inner_dim,
            adain_mode=adain_mode,
            eps=eps,
        )

        # Output
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False
        self._musubi_block_swap = MusubiBlockSwapManager.build(
            depth=num_layers,
            blocks_to_swap=musubi_blocks_to_swap,
            swap_device=musubi_block_swap_device,
            logger=logger,
        )

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def set_router(self, router: TREADRouter, routes: List[Dict[str, Any]]):
        """Set the TREAD router and routing configuration."""
        self._tread_router = router
        self._tread_routes = routes

    def process_motion(self, motion_latents: torch.Tensor, drop_motion_frames: bool = False):
        if drop_motion_frames or motion_latents.shape[2] == 0:
            return None, None
        return self.motion_in(motion_latents)

    def process_motion_frame_pack(
        self, motion_latents: torch.Tensor, drop_motion_frames: bool = False, add_last_motion: int = 2
    ):
        flat_mot, mot_remb = self.frame_packer(motion_latents, add_last_motion)
        if drop_motion_frames:
            return flat_mot[:, :0], mot_remb[:, :0]
        return flat_mot, mot_remb

    def inject_motion(
        self,
        hidden_states: torch.Tensor,
        seq_lens: torch.Tensor,
        rope_embs: torch.Tensor,
        mask_input: torch.Tensor,
        motion_latents: torch.Tensor,
        drop_motion_frames: bool = False,
        add_last_motion: int = 2,
    ):
        if self.config.enable_framepack:
            mot, mot_remb = self.process_motion_frame_pack(motion_latents, drop_motion_frames, add_last_motion)
        else:
            mot, mot_remb = self.process_motion(motion_latents, drop_motion_frames)

        if mot is not None and mot.shape[1] > 0:
            hidden_states = torch.cat([hidden_states, mot], dim=1)
            seq_lens = seq_lens + torch.tensor([mot.shape[1]], dtype=torch.long, device=seq_lens.device)
            rope_embs = torch.cat([rope_embs, mot_remb], dim=1)
            mask_input = torch.cat(
                [
                    mask_input,
                    2
                    * torch.ones(
                        [1, hidden_states.shape[1] - mask_input.shape[1]],
                        device=mask_input.device,
                        dtype=mask_input.dtype,
                    ),
                ],
                dim=1,
            )

        return hidden_states, seq_lens, rope_embs, mask_input

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        motion_latents: torch.Tensor,
        audio_embeds: torch.Tensor,
        image_latents: torch.Tensor,
        pose_latents: torch.Tensor,
        motion_frames: List[int] = [17, 5],
        drop_motion_frames: bool = False,
        add_last_motion: int = 2,
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        output_hidden_states: bool = False,
        hidden_state_layer: Optional[int] = None,
        hidden_states_buffer: Optional[Dict[str, torch.Tensor]] = None,
        force_keep_mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        """
        Forward pass for S2V generation.

        Args:
            hidden_states: Noisy latents [B, C, T, H, W]
            timestep: Diffusion timestep [B]
            encoder_hidden_states: Text embeddings [B, L, D]
            motion_latents: Previous frame latents for continuity [B, C, T_m, H, W]
            audio_embeds: Wav2Vec2 features [B, num_layers, audio_dim, T_audio]
            image_latents: Reference image latents [B, C, 1, H, W]
            pose_latents: Pose conditioning [B, C, T, H, W]
            motion_frames: [num_motion_frames, latent_motion_frames]
            drop_motion_frames: Whether to drop motion conditioning
            add_last_motion: Motion frame usage level (0, 1, or 2)
        """
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        add_last_motion = self.config.add_last_motion * add_last_motion

        # Rotary embeddings
        rotary_emb = self.rope(hidden_states, image_latents)

        # Patch embeddings
        hidden_states = self.patch_embedding(hidden_states)
        image_latents = self.patch_embedding(image_latents)

        # Condition embeddings
        audio_embeds = torch.cat(
            [audio_embeds[..., 0].unsqueeze(-1).repeat(1, 1, 1, motion_frames[0]), audio_embeds], dim=-1
        )

        if self.config.zero_timestep:
            timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])

        temb, timestep_proj, encoder_hidden_states, audio_hidden_states, pose_hidden_states = self.condition_embedder(
            timestep, encoder_hidden_states, audio_embeds, pose_latents
        )
        timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if self.config.enable_adain:
            audio_emb_global, audio_emb = audio_hidden_states
            audio_emb_global = audio_emb_global[:, motion_frames[1] :].clone()
        else:
            audio_emb = audio_hidden_states
            audio_emb_global = None
        merged_audio_emb = audio_emb[:, motion_frames[1] :, :]

        hidden_states = hidden_states + pose_hidden_states
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        image_latents = image_latents.flatten(2).transpose(1, 2)

        sequence_length = torch.tensor([hidden_states.shape[1]], dtype=torch.long, device=hidden_states.device)
        original_sequence_length = sequence_length.clone()
        sequence_length = sequence_length + torch.tensor(
            [image_latents.shape[1]], dtype=torch.long, device=hidden_states.device
        )
        hidden_states = torch.cat([hidden_states, image_latents], dim=1)

        mask_input = torch.zeros([1, hidden_states.shape[1]], dtype=torch.long, device=hidden_states.device)
        mask_input[:, original_sequence_length:] = 1

        hidden_states, sequence_length, rotary_emb, mask_input = self.inject_motion(
            hidden_states,
            sequence_length,
            rotary_emb,
            mask_input,
            motion_latents,
            drop_motion_frames,
            add_last_motion,
        )

        hidden_states = hidden_states + self.trainable_condition_mask(mask_input).to(hidden_states.dtype)

        if self.config.zero_timestep:
            temb = temb[:-1]
            zero_timestep_proj = timestep_proj[-1:]
            timestep_proj = timestep_proj[:-1]
            timestep_proj = torch.cat(
                [timestep_proj.unsqueeze(2), zero_timestep_proj.unsqueeze(2).repeat(timestep_proj.shape[0], 1, 1, 1)], dim=2
            )
            timestep_proj = [timestep_proj, original_sequence_length]
        else:
            timestep_proj = timestep_proj.unsqueeze(2).repeat(1, 1, 2, 1)
            timestep_proj = [timestep_proj, torch.tensor(0)]

        merged_audio_emb_num_frames = merged_audio_emb.shape[1]
        attn_audio_emb = merged_audio_emb.flatten(0, 1).to(hidden_states.dtype)
        if audio_emb_global is not None:
            audio_emb_global = audio_emb_global.flatten(0, 1).to(hidden_states.dtype)

        # TREAD initialization
        routes = self._tread_routes or []
        router = self._tread_router
        use_routing = self.training and len(routes) > 0 and torch.is_grad_enabled()
        route_ptr = 0
        routing_now = False
        tread_mask_info = None
        saved_tokens = None
        current_rope = rotary_emb

        # Handle negative route indices
        if routes:
            num_blocks = len(self.blocks)

            def _to_pos(idx: int) -> int:
                return idx if idx >= 0 else num_blocks + idx

            routes = [
                {
                    **r,
                    "start_layer_idx": _to_pos(r["start_layer_idx"]),
                    "end_layer_idx": _to_pos(r["end_layer_idx"]),
                }
                for r in routes
            ]

        captured_frame_hidden: Optional[torch.Tensor] = None

        # Musubi block swap activation
        grad_enabled = torch.is_grad_enabled()
        musubi_manager = self._musubi_block_swap
        musubi_offload_active = False
        if musubi_manager is not None:
            musubi_offload_active = musubi_manager.activate(self.blocks, hidden_states.device, grad_enabled)

        # Transformer blocks with TREAD routing
        for block_idx, block in enumerate(self.blocks):
            # TREAD: START a route?
            if use_routing and route_ptr < len(routes) and block_idx == routes[route_ptr]["start_layer_idx"]:
                mask_ratio = routes[route_ptr]["selection_ratio"]

                tread_mask_info = router.get_mask(
                    hidden_states,
                    mask_ratio=mask_ratio,
                    force_keep=force_keep_mask,
                )
                saved_tokens = hidden_states.clone()
                hidden_states = router.start_route(hidden_states, tread_mask_info)
                current_rope = router.route_rope(rotary_emb, tread_mask_info)
                routing_now = True

            # Musubi: stream in block if managed
            if musubi_offload_active and musubi_manager.is_managed_block(block_idx):
                musubi_manager.stream_in(block, hidden_states.device)

            if self.training and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, timestep_proj, current_rope
                )
            else:
                hidden_states = block(hidden_states, encoder_hidden_states, timestep_proj, current_rope)

            # TREAD: END the current route?
            if routing_now and block_idx == routes[route_ptr]["end_layer_idx"]:
                hidden_states = router.end_route(
                    hidden_states,
                    tread_mask_info,
                    original_x=saved_tokens,
                )
                routing_now = False
                route_ptr += 1
                current_rope = rotary_emb

            # Capture hidden states for CREPA
            if output_hidden_states and (hidden_state_layer is None or block_idx == hidden_state_layer):
                captured_frame_hidden = hidden_states[:, : original_sequence_length.item()].reshape(
                    batch_size,
                    post_patch_num_frames,
                    post_patch_height * post_patch_width,
                    -1,
                )
                if hidden_state_layer is not None and block_idx == hidden_state_layer:
                    output_hidden_states = False

            # Store in hidden states buffer for LayerSync
            if hidden_states_buffer is not None:
                tokens_view = hidden_states[:, : original_sequence_length.item()].reshape(
                    batch_size,
                    post_patch_num_frames,
                    post_patch_height * post_patch_width,
                    -1,
                )
                hidden_states_buffer[f"layer_{block_idx}"] = tokens_view

            if block_idx in self.audio_injector.injected_block_id:
                hidden_states = self.audio_injector(
                    block_idx,
                    hidden_states,
                    original_sequence_length.item(),
                    merged_audio_emb_num_frames,
                    attn_audio_emb,
                    audio_emb_global,
                )

            # Musubi: stream out block if managed
            if musubi_offload_active and musubi_manager.is_managed_block(block_idx):
                musubi_manager.stream_out(block)

        hidden_states = hidden_states[:, : original_sequence_length.item()]

        # Output projection
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        # Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_num_frames, post_patch_height, post_patch_width, p_t, p_h, p_w, -1
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            if captured_frame_hidden is None:
                return (output,)
            return (output, captured_frame_hidden)

        result = Transformer2DModelOutput(sample=output)
        if captured_frame_hidden is not None:
            result.crepa_hidden_states = captured_frame_hidden
        return result
