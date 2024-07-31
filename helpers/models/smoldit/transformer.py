# Copyright 2024 Lumina, Hunyuan DiT, PixArt, The HuggingFace Team. All rights reserved.
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
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PatchEmbed,
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
    apply_rotary_emb,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, FP32LayerNorm
from diffusers.models.transformers.hunyuan_transformer_2d import AdaLayerNormShift
from diffusers.utils import logging


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SmolDiTAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        cross_attention_dim,
        dim_head,
        num_heads,
        kv_heads,
        sliding_window=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * num_heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.num_heads = num_heads
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )

        self.scale = dim_head**-0.5
        self.sliding_window = sliding_window

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=False)
        self.to_v = nn.Linear(self.cross_attention_dim, self.inner_kv_dim, bias=False)

        self.to_out = nn.Linear(self.inner_dim, query_dim, bias=False)

    # this mask processing utility is taken from the `prepare_attention_mask()`
    # function from diffusers. it is here for self-containment.
    def prepare_attention_mask(self, hidden_states, attention_mask):
        sequence_length = hidden_states.shape[1]
        current_length = attention_mask.shape[-1]
        batch_size = hidden_states.shape[0]
        if current_length != sequence_length:
            if attention_mask.device.type == "mps":
                padding_shape = (
                    attention_mask.shape[0],
                    attention_mask.shape[1],
                    sequence_length,
                )
                padding = torch.zeros(
                    padding_shape,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                attention_mask = F.pad(attention_mask, (0, sequence_length), value=0.0)

        if attention_mask.shape[0] < batch_size * self.num_heads:
            attention_mask = attention_mask.repeat_interleave(self.num_heads, dim=0)

        return attention_mask

    def sliding_window_attention_mask(
        self,
        sequence_length: int,
        window_size: int,
        batch_size: int,
        num_heads: int,
        device,
    ) -> torch.Tensor:
        mask = torch.zeros(
            (batch_size, num_heads, sequence_length, sequence_length), device=device
        )
        for i in range(sequence_length):
            start = max(0, i - window_size)
            end = min(sequence_length, i + window_size + 1)
            mask[:, :, i, start:end] = 1
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        batch_size, _, _ = hidden_states.shape
        encoder_hidden_states = (
            hidden_states if encoder_hidden_states is None else encoder_hidden_states
        )

        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = None
        if encoder_attention_mask is not None:
            encoder_attention_mask = self.prepare_attention_mask(
                encoder_hidden_states, encoder_attention_mask
            )
            encoder_attention_mask = encoder_attention_mask.view(
                batch_size, self.num_heads, -1, encoder_attention_mask.shape[-1]
            )
            attention_mask = encoder_attention_mask
        elif self.sliding_window:
            attention_mask = self.sliding_window_attention_mask(
                sequence_length=hidden_states.shape[1],
                window_size=self.sliding_window,
                batch_size=batch_size,
                num_heads=self.num_heads,
                device=hidden_states.device,
            )

        # Projections.
        query = self.to_q(hidden_states)
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // self.num_heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim
        query = query.view(batch_size, -1, self.num_heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        # GQA
        if kv_heads != self.num_heads:
            # if GQA or MQA, repeat the key/value heads to reach the number of query heads.
            heads_per_kv_head = self.num_heads // kv_heads
            key = torch.repeat_interleave(key, heads_per_kv_head, dim=1)
            value = torch.repeat_interleave(value, heads_per_kv_head, dim=1)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            query = query.to(dtype)
            if not self.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)
                key = query.to(dtype)

        # the output of sdpa = (batch, num_heads, seq_len, head_dim)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=self.scale
        )

        # out
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = self.to_out(hidden_states)
        return hidden_states


class SmolDiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        num_kv_heads: int,
        ff_inner_dim: int,
        cross_attention_dim: int = 1024,
        activation_fn: str = "gelu-approximate",
        layer_idx: int = None,
        sliding_window: int = None,
    ):
        super().__init__()

        # 1. Self-Attn
        self.norm1 = AdaLayerNormShift(dim, elementwise_affine=True, eps=1e-6)
        if layer_idx is not None and sliding_window is not None:
            sliding_window = sliding_window if not bool(layer_idx % 2) else None
        else:
            sliding_window = None

        self.attn1 = SmolDiTAttention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=dim // num_attention_heads,
            num_heads=num_attention_heads,
            kv_heads=num_kv_heads,
            sliding_window=sliding_window,
        )

        # 2. Cross-Attn
        self.norm2 = FP32LayerNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn2 = SmolDiTAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            dim_head=dim // num_attention_heads,
            num_heads=num_attention_heads,
            kv_heads=num_kv_heads,
        )

        # 3. Feed-forward
        self.ff = FeedForward(
            dim,
            activation_fn=activation_fn,
            inner_dim=ff_inner_dim,
            bias=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # 1. Self-Attention
        norm_hidden_states = self.norm1(hidden_states, temb)
        attn_output = self.attn1(
            norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + attn_output

        # 2. Cross-Attention
        hidden_states = hidden_states + self.attn2(
            self.norm2(hidden_states),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

        # FFN Layer
        hidden_states = hidden_states + self.ff(hidden_states)

        return hidden_states


class SmolDiT2DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        num_attention_heads: int = 16,
        num_kv_heads: int = 8,
        attention_head_dim: int = 88,
        in_channels: int = 4,
        out_channels: int = 4,
        activation_fn: str = "gelu-approximate",
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        cross_attention_dim: int = 1024,
        sliding_window: int = None,
    ):
        super().__init__()
        self.inner_dim = num_attention_heads * attention_head_dim

        self.time_proj = Timesteps(
            num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=self.inner_dim
        )

        self.text_embedder = PixArtAlphaTextProjection(
            in_features=cross_attention_dim,
            hidden_size=cross_attention_dim * 4,
            out_features=cross_attention_dim,
            act_fn="silu_fp32",
        )

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            patch_size=patch_size,
            pos_embed_type=None,
        )

        # SmolDiT Blocks
        self.blocks = nn.ModuleList(
            [
                SmolDiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    num_kv_heads=num_kv_heads,
                    ff_inner_dim=int(self.inner_dim * mlp_ratio),
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    layer_idx=layer_idx,
                    sliding_window=(
                        sliding_window if sliding_window is not None else None
                    ),
                )
                for layer_idx in range(num_layers)
            ]
        )

        self.out_channels = out_channels
        self.norm_out = AdaLayerNormContinuous(
            self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6
        )
        self.proj_out = nn.Linear(
            self.inner_dim, patch_size * patch_size * out_channels
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict=True,
    ):
        height, width = hidden_states.shape[-2:]
        hidden_dtype = hidden_states.dtype

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # patch embed
        hidden_states = self.pos_embed(hidden_states)

        # timestep
        batch_size = hidden_states.shape[0]
        timesteps_proj = self.time_proj(timestep)
        temb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, 256)

        # text projection
        batch_size, sequence_length, _ = encoder_hidden_states.shape
        encoder_hidden_states = self.text_embedder(
            encoder_hidden_states.view(-1, encoder_hidden_states.shape[-1])
        )
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size, sequence_length, -1
        )

        for _, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                image_rotary_emb=image_rotary_emb,
            )  # (N, L, D)

        # final layer
        hidden_states = self.norm_out(hidden_states, temb.to(torch.float32))
        hidden_states = self.proj_out(hidden_states)
        # (N, L, patch_size ** 2 * out_channels)

        # unpatchify: (N, out_channels, H, W)
        patch_size = self.pos_embed.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                self.out_channels,
                height * patch_size,
                width * patch_size,
            )
        )
        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
