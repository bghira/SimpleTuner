# Copyright 2025 The NVIDIA Team and The HuggingFace Team. All rights reserved.
# Copyright 2025 bghira (SimpleTuner)
# - Added support for PEFT LoRA
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

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import FromOriginalModelMixin
from diffusers.loaders.peft import PeftAdapterMixin
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm
from diffusers.utils import USE_PEFT_BACKEND, is_torchvision_available, logging, scale_lora_layers, unscale_lora_layers

from simpletuner.helpers.musubi_block_swap import MusubiBlockSwapManager
from simpletuner.helpers.training.qk_clip_logging import publish_attention_max_logits
from simpletuner.helpers.training.tread import TREADRouter
from simpletuner.helpers.utils.patching import MutableModuleList, PatchableModule

logger = logging.get_logger(__name__)


def _store_hidden_state(buffer, key: str, hidden_states: torch.Tensor, image_tokens_start: int | None = None):
    if buffer is None:
        return
    if image_tokens_start is not None and hidden_states.dim() >= 3:
        buffer[key] = hidden_states[:, image_tokens_start:, ...]
    else:
        buffer[key] = hidden_states


if is_torchvision_available():
    from torchvision import transforms


class CosmosPatchEmbed(PatchableModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: Tuple[int, int, int],
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2],
            out_channels,
            bias=bias,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        hidden_states = hidden_states.reshape(
            batch_size,
            num_channels,
            num_frames // p_t,
            p_t,
            height // p_h,
            p_h,
            width // p_w,
            p_w,
        )
        hidden_states = hidden_states.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class CosmosTimestepEmbedding(PatchableModule):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features, out_features, bias=False)
        self.activation = nn.SiLU()
        self.linear_2 = nn.Linear(out_features, 3 * out_features, bias=False)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        emb = self.linear_1(timesteps)
        emb = self.activation(emb)
        emb = self.linear_2(emb)
        return emb


class CosmosEmbedding(PatchableModule):
    def __init__(self, embedding_dim: int, condition_dim: int) -> None:
        super().__init__()

        self.time_proj = Timesteps(embedding_dim, flip_sin_to_cos=True, downscale_freq_shift=0.0)
        self.t_embedder = CosmosTimestepEmbedding(embedding_dim, condition_dim)
        self.norm = RMSNorm(embedding_dim, eps=1e-6, elementwise_affine=True)

    def forward(self, hidden_states: torch.Tensor, timestep: torch.LongTensor) -> torch.Tensor:
        timesteps_proj = self.time_proj(timestep).type_as(hidden_states)
        temb = self.t_embedder(timesteps_proj)
        embedded_timestep = self.norm(timesteps_proj)
        return temb, embedded_timestep


class CosmosAdaLayerNorm(PatchableModule):
    def __init__(self, in_features: int, hidden_features: int) -> None:
        super().__init__()
        self.embedding_dim = in_features

        self.activation = nn.SiLU()
        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)
        self.linear_2 = nn.Linear(hidden_features, 2 * in_features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedded_timestep = self.activation(embedded_timestep)
        embedded_timestep = self.linear_1(embedded_timestep)
        embedded_timestep = self.linear_2(embedded_timestep)

        if temb is not None:
            embedded_timestep = embedded_timestep + temb[..., : 2 * self.embedding_dim]

        shift, scale = embedded_timestep.chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)

        if embedded_timestep.ndim == 2:
            shift, scale = (x.unsqueeze(1) for x in (shift, scale))

        hidden_states = hidden_states * (1 + scale) + shift
        return hidden_states


class CosmosAdaLayerNormZero(PatchableModule):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(in_features, elementwise_affine=False, eps=1e-6)
        self.activation = nn.SiLU()

        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features

        if hidden_features == in_features:
            self.linear_1 = nn.Identity()
        else:
            self.linear_1 = nn.Linear(in_features, hidden_features, bias=False)

        self.linear_2 = nn.Linear(hidden_features, 3 * in_features, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        embedded_timestep = self.activation(embedded_timestep)
        embedded_timestep = self.linear_1(embedded_timestep)
        embedded_timestep = self.linear_2(embedded_timestep)

        if temb is not None:
            embedded_timestep = embedded_timestep + temb

        shift, scale, gate = embedded_timestep.chunk(3, dim=-1)
        hidden_states = self.norm(hidden_states)

        expanded = embedded_timestep.ndim == 2
        if expanded:
            shift, scale, gate = (x.unsqueeze(1) for x in (shift, scale, gate))

        hidden_states = hidden_states * (1 + scale) + shift
        if expanded:
            gate = gate.squeeze(1)
        return hidden_states, gate


class CosmosAttnProcessor2_0:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention") or not callable(F.scaled_dot_product_attention):
            raise ImportError("CosmosAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 1. QKV projections
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        target_dtype = query.dtype
        key = key.to(target_dtype)
        value = value.to(target_dtype)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        # 2. QK normalization
        query = attn.norm_q(query).to(target_dtype)
        key = attn.norm_k(key).to(target_dtype)

        # 3. Apply RoPE
        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
            key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)

        # 4. Prepare for GQA
        if torch.onnx.is_in_onnx_export():
            query_idx = torch.tensor(query.size(3), device=query.device)
            key_idx = torch.tensor(key.size(3), device=key.device)
            value_idx = torch.tensor(value.size(3), device=value.device)

        else:
            query_idx = query.size(3)
            key_idx = key.size(3)
            value_idx = value.size(3)
        key = key.repeat_interleave(query_idx // key_idx, dim=3)
        value = value.repeat_interleave(query_idx // value_idx, dim=3)

        # 5. Attention
        publish_attention_max_logits(
            query,
            key,
            attention_mask,
            getattr(attn, "to_q", None) and attn.to_q.weight,
            getattr(attn, "to_k", None) and attn.to_k.weight,
        )
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        # 6. Output projection
        hidden_states = hidden_states.to(hidden_states.dtype if hidden_states.dtype == query.dtype else query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class CosmosTransformerBlock(PatchableModule):
    def __init__(
        self,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        mlp_ratio: float = 4.0,
        adaln_lora_dim: int = 256,
        qk_norm: str = "rms_norm",
        out_bias: bool = False,
    ) -> None:
        super().__init__()

        hidden_size = num_attention_heads * attention_head_dim

        self.norm1 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.attn1 = Attention(
            query_dim=hidden_size,
            cross_attention_dim=None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm=qk_norm,
            elementwise_affine=True,
            out_bias=out_bias,
            processor=CosmosAttnProcessor2_0(),
        )

        self.norm2 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.attn2 = Attention(
            query_dim=hidden_size,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm=qk_norm,
            elementwise_affine=True,
            out_bias=out_bias,
            processor=CosmosAttnProcessor2_0(),
        )

        self.norm3 = CosmosAdaLayerNormZero(in_features=hidden_size, hidden_features=adaln_lora_dim)
        self.ff = FeedForward(hidden_size, mult=mlp_ratio, activation_fn="gelu", bias=out_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        extra_pos_emb: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if extra_pos_emb is not None:
            hidden_states = hidden_states + extra_pos_emb

        # 1. Self Attention
        norm_hidden_states, gate = self._extract_norm_outputs(
            self.norm1(hidden_states, embedded_timestep, temb), hidden_states
        )
        attn_output = self.attn1(norm_hidden_states, image_rotary_emb=image_rotary_emb)
        hidden_states = hidden_states + gate * attn_output

        # 2. Cross Attention
        norm_hidden_states, gate = self._extract_norm_outputs(
            self.norm2(hidden_states, embedded_timestep, temb), hidden_states
        )
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + gate * attn_output

        # 3. Feed Forward
        norm_hidden_states, gate = self._extract_norm_outputs(
            self.norm3(hidden_states, embedded_timestep, temb), hidden_states
        )
        ff_output = self.ff(norm_hidden_states)
        hidden_states = hidden_states + gate * ff_output

        return hidden_states

    @staticmethod
    def _extract_norm_outputs(output, reference):
        if isinstance(output, tuple):
            if len(output) >= 2:
                norm_hidden_states, gate = output[0], output[1]
                if gate.ndim == reference.ndim - 1:
                    gate = gate.unsqueeze(1)
                elif gate.ndim < reference.ndim - 1:
                    # bring to (B,1,H)
                    while gate.ndim < reference.ndim - 1:
                        gate = gate.unsqueeze(0)
                    gate = gate.unsqueeze(1)
                return norm_hidden_states, gate
            raise ValueError("Cosmos AdaLayerNorm output tuple must have at least two elements.")

        # Fallback for legacy behaviours that returned just the normalized tensor
        hidden_states = output
        if hidden_states.ndim >= 2:
            batch = hidden_states.size(0)
            hidden = hidden_states.size(-1)
            gate = torch.ones(batch, 1, hidden, device=hidden_states.device, dtype=hidden_states.dtype)
        else:
            gate = torch.ones_like(hidden_states).unsqueeze(0)
        return hidden_states, gate


class CosmosRotaryPosEmbed(PatchableModule):
    def __init__(
        self,
        hidden_size: int,
        max_size: Tuple[int, int, int] = (128, 240, 240),
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        base_fps: int = 24,
        rope_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.max_size = [size // patch for size, patch in zip(max_size, patch_size)]
        self.patch_size = patch_size
        self.base_fps = base_fps

        self.dim_h = hidden_size // 6 * 2
        self.dim_w = hidden_size // 6 * 2
        self.dim_t = hidden_size - self.dim_h - self.dim_w

        self.h_ntk_factor = rope_scale[1] ** (self.dim_h / (self.dim_h - 2))
        self.w_ntk_factor = rope_scale[2] ** (self.dim_w / (self.dim_w - 2))
        self.t_ntk_factor = rope_scale[0] ** (self.dim_t / (self.dim_t - 2))

    def forward(self, hidden_states: torch.Tensor, fps: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        pe_size = [
            num_frames // self.patch_size[0],
            height // self.patch_size[1],
            width // self.patch_size[2],
        ]
        device = hidden_states.device

        h_theta = 10000.0 * self.h_ntk_factor
        w_theta = 10000.0 * self.w_ntk_factor
        t_theta = 10000.0 * self.t_ntk_factor

        seq = torch.arange(max(self.max_size), device=device, dtype=torch.float32)
        dim_h_range = torch.arange(0, self.dim_h, 2, device=device, dtype=torch.float32)[: (self.dim_h // 2)] / self.dim_h
        dim_w_range = torch.arange(0, self.dim_w, 2, device=device, dtype=torch.float32)[: (self.dim_w // 2)] / self.dim_w
        dim_t_range = torch.arange(0, self.dim_t, 2, device=device, dtype=torch.float32)[: (self.dim_t // 2)] / self.dim_t
        h_spatial_freqs = 1.0 / (h_theta**dim_h_range)
        w_spatial_freqs = 1.0 / (w_theta**dim_w_range)
        temporal_freqs = 1.0 / (t_theta**dim_t_range)

        emb_h = torch.outer(seq[: pe_size[1]], h_spatial_freqs)[None, :, None, :].repeat(pe_size[0], 1, pe_size[2], 1)
        emb_w = torch.outer(seq[: pe_size[2]], w_spatial_freqs)[None, None, :, :].repeat(pe_size[0], pe_size[1], 1, 1)

        # Apply sequence scaling in temporal dimension
        if fps is None:
            # Images
            emb_t = torch.outer(seq[: pe_size[0]], temporal_freqs)
        else:
            # Videos
            emb_t = torch.outer(seq[: pe_size[0]] / fps * self.base_fps, temporal_freqs)

        emb_t = emb_t[:, None, None, :].repeat(1, pe_size[1], pe_size[2], 1)
        freqs = torch.cat([emb_t, emb_h, emb_w] * 2, dim=-1).flatten(0, 2).float()
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin


class CosmosLearnablePositionalEmbed(PatchableModule):
    def __init__(
        self,
        hidden_size: int,
        max_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.max_size = [size // patch for size, patch in zip(max_size, patch_size)]
        self.patch_size = patch_size
        self.eps = eps

        self.pos_emb_t = nn.Parameter(torch.empty(self.max_size[0], hidden_size))
        self.pos_emb_h = nn.Parameter(torch.empty(self.max_size[1], hidden_size))
        self.pos_emb_w = nn.Parameter(torch.empty(self.max_size[2], hidden_size))

        for param in (self.pos_emb_t, self.pos_emb_h, self.pos_emb_w):
            nn.init.normal_(param, mean=0.0, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        pe_size = [
            num_frames // self.patch_size[0],
            height // self.patch_size[1],
            width // self.patch_size[2],
        ]

        emb_t = self.pos_emb_t[: pe_size[0]][None, :, None, None, :].repeat(batch_size, 1, pe_size[1], pe_size[2], 1)
        emb_h = self.pos_emb_h[: pe_size[1]][None, None, :, None, :].repeat(batch_size, pe_size[0], 1, pe_size[2], 1)
        emb_w = self.pos_emb_w[: pe_size[2]][None, None, None, :, :].repeat(batch_size, pe_size[0], pe_size[1], 1, 1)
        emb = emb_t + emb_h + emb_w
        emb = emb.flatten(1, 3)

        norm = torch.linalg.vector_norm(emb, dim=-1, keepdim=True, dtype=torch.float32)
        norm = torch.add(self.eps, norm, alpha=np.sqrt(norm.numel() / emb.numel()))
        return (emb / norm).type_as(hidden_states)


class CosmosTransformer3DModel(PatchableModule, ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    r"""
    A Transformer model for video-like data used in [Cosmos](https://github.com/NVIDIA/Cosmos).

    Args:
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `32`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each attention head.
        num_layers (`int`, defaults to `28`):
            The number of layers of transformer blocks to use.
        mlp_ratio (`float`, defaults to `4.0`):
            The ratio of the hidden layer size to the input size in the feedforward network.
        text_embed_dim (`int`, defaults to `4096`):
            Input dimension of text embeddings from the text encoder.
        adaln_lora_dim (`int`, defaults to `256`):
            The hidden dimension of the Adaptive LayerNorm LoRA layer.
        max_size (`Tuple[int, int, int]`, defaults to `(128, 240, 240)`):
            The maximum size of the input latent tensors in the temporal, height, and width dimensions.
        patch_size (`Tuple[int, int, int]`, defaults to `(1, 2, 2)`):
            The patch size to use for patchifying the input latent tensors in the temporal, height, and width
            dimensions.
        rope_scale (`Tuple[float, float, float]`, defaults to `(2.0, 1.0, 1.0)`):
            The scaling factor to use for RoPE in the temporal, height, and width dimensions.
        concat_padding_mask (`bool`, defaults to `True`):
            Whether to concatenate the padding mask to the input latent tensors.
        extra_pos_embed_type (`str`, *optional*, defaults to `learnable`):
            The type of extra positional embeddings to use. Can be one of `None` or `learnable`.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embed", "final_layer", "norm"]
    _no_split_modules = ["CosmosTransformerBlock"]
    _keep_in_fp32_modules = ["learnable_pos_embed"]
    _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        max_size: Tuple[int, int, int] = (128, 240, 240),
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        rope_scale: Tuple[float, float, float] = (2.0, 1.0, 1.0),
        concat_padding_mask: bool = True,
        extra_pos_embed_type: Optional[str] = "learnable",
        musubi_blocks_to_swap: int = 0,
        musubi_block_swap_device: str = "cpu",
    ) -> None:
        super().__init__()
        self.register_to_config(
            in_channels=in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            text_embed_dim=text_embed_dim,
            adaln_lora_dim=adaln_lora_dim,
            max_size=max_size,
            patch_size=patch_size,
            rope_scale=rope_scale,
            concat_padding_mask=concat_padding_mask,
            extra_pos_embed_type=extra_pos_embed_type,
            musubi_blocks_to_swap=musubi_blocks_to_swap,
            musubi_block_swap_device=musubi_block_swap_device,
        )
        hidden_size = num_attention_heads * attention_head_dim

        # 1. Patch Embedding
        patch_embed_in_channels = in_channels + 1 if concat_padding_mask else in_channels
        self.patch_embed = CosmosPatchEmbed(patch_embed_in_channels, hidden_size, patch_size, bias=False)

        # 2. Positional Embedding
        self.rope = CosmosRotaryPosEmbed(
            hidden_size=attention_head_dim,
            max_size=max_size,
            patch_size=patch_size,
            rope_scale=rope_scale,
        )

        self.learnable_pos_embed = None
        if extra_pos_embed_type == "learnable":
            self.learnable_pos_embed = CosmosLearnablePositionalEmbed(
                hidden_size=hidden_size,
                max_size=max_size,
                patch_size=patch_size,
            )

        # 3. Time Embedding
        self.time_embed = CosmosEmbedding(hidden_size, hidden_size)

        # 4. Transformer Blocks
        self.transformer_blocks = MutableModuleList(
            [
                CosmosTransformerBlock(
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=text_embed_dim,
                    mlp_ratio=mlp_ratio,
                    adaln_lora_dim=adaln_lora_dim,
                    qk_norm="rms_norm",
                    out_bias=False,
                )
                for _ in range(num_layers)
            ]
        )

        # 5. Output norm & projection
        self.norm_out = CosmosAdaLayerNorm(hidden_size, adaln_lora_dim)
        self.proj_out = nn.Linear(
            hidden_size,
            patch_size[0] * patch_size[1] * patch_size[2] * out_channels,
            bias=False,
        )

        self.gradient_checkpointing = False
        self._musubi_block_swap = MusubiBlockSwapManager.build(
            depth=num_layers,
            blocks_to_swap=musubi_blocks_to_swap,
            swap_device=musubi_block_swap_device,
            logger=logger,
        )

        # TREAD support
        self._tread_router = None
        self._tread_routes = None

    def set_router(self, router: TREADRouter, routes: Optional[List[Dict]] = None):
        """Set TREAD router and routes for token reduction during training."""
        self._tread_router = router
        self._tread_routes = routes

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        fps: Optional[int] = None,
        condition_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        force_keep_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        hidden_states_buffer: Optional[dict] = None,
    ) -> torch.Tensor:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)

        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        # 1. Concatenate padding mask if needed & prepare attention mask
        if condition_mask is not None:
            hidden_states = torch.cat([hidden_states, condition_mask], dim=1)

        if self.config.concat_padding_mask:
            if padding_mask.ndim > 5:
                padding_mask = padding_mask.reshape(
                    padding_mask.shape[0],
                    padding_mask.shape[1],
                    -1,
                    padding_mask.shape[-2],
                    padding_mask.shape[-1],
                )
            if padding_mask.ndim == 4:
                padding_mask = transforms.functional.resize(
                    padding_mask,
                    list(hidden_states.shape[-2:]),
                    interpolation=transforms.InterpolationMode.NEAREST,
                )
                if padding_mask.ndim == 4:
                    padding_mask = padding_mask.unsqueeze(2)
                elif padding_mask.ndim == 5:
                    pass
                else:
                    raise ValueError(f"Unexpected padding mask dimensions after resize: {padding_mask.shape}")
            elif padding_mask.ndim == 5:
                # ensure spatial dimensions match later via interpolation
                pass
            else:
                raise ValueError(f"Padding mask must be 4D or 5D, received shape {padding_mask.shape}.")

            if (
                padding_mask.shape[2] != num_frames
                or padding_mask.shape[-2] != hidden_states.shape[-2]
                or padding_mask.shape[-1] != hidden_states.shape[-1]
            ):
                padding_mask = torch.nn.functional.interpolate(
                    padding_mask,
                    size=(num_frames, hidden_states.shape[-2], hidden_states.shape[-1]),
                    mode="nearest",
                )

            target_shape = (
                batch_size,
                padding_mask.shape[1],
                num_frames,
                hidden_states.shape[-2],
                hidden_states.shape[-1],
            )
            expand_dims = []
            for current, target in zip(padding_mask.shape, target_shape):
                if current == target:
                    expand_dims.append(current)
                elif current == 1:
                    expand_dims.append(target)
                else:
                    raise ValueError(
                        f"Cannot broadcast padding mask dimension {current} to {target} for shape {padding_mask.shape}."
                    )

            padding_mask = padding_mask.expand(*expand_dims)
            hidden_states = torch.cat([hidden_states, padding_mask], dim=1)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, S]

        # 2. Generate positional embeddings
        image_rotary_emb = self.rope(hidden_states, fps=fps)
        extra_pos_emb = self.learnable_pos_embed(hidden_states) if self.config.extra_pos_embed_type else None

        # 3. Patchify input
        p_t, p_h, p_w = self.config.patch_size
        expected_num_frames = num_frames // p_t
        expected_height = height // p_h
        expected_width = width // p_w
        hidden_states = self.patch_embed(hidden_states)
        if hidden_states.ndim == 5:
            hidden_states = hidden_states.flatten(1, 3)  # [B, T, H, W, C] -> [B, THW, C]
        elif hidden_states.ndim != 3:
            raise ValueError(
                f"Cosmos patch_embed expected to return a tensor with 3 or 5 dimensions, got {hidden_states.ndim}."
            )
        token_count = hidden_states.shape[1]
        spatial_tokens = expected_height * expected_width
        if spatial_tokens == 0:
            raise ValueError("Spatial token count cannot be zero.")

        if token_count % spatial_tokens != 0:
            raise ValueError(f"Token count {token_count} is incompatible with spatial tokens {spatial_tokens}.")

        post_patch_num_frames = token_count // spatial_tokens
        post_patch_height = expected_height
        post_patch_width = expected_width

        # 4. Timestep embeddings
        if timestep.ndim == 1:
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
        elif timestep.ndim == 5:
            assert timestep.shape == (
                batch_size,
                1,
                num_frames,
                1,
                1,
            ), f"Expected timestep to have shape [B, 1, T, 1, 1], but got {timestep.shape}"
            timestep = timestep.flatten()
            temb, embedded_timestep = self.time_embed(hidden_states, timestep)
            # We can do this because num_frames == post_patch_num_frames, as p_t is 1
            temb, embedded_timestep = (
                x.view(batch_size, post_patch_num_frames, 1, 1, -1)
                .expand(-1, -1, post_patch_height, post_patch_width, -1)
                .flatten(1, 3)
                for x in (temb, embedded_timestep)
            )  # [BT, C] -> [B, T, 1, 1, C] -> [B, T, H, W, C] -> [B, THW, C]
        else:
            assert False

        # 5. Transformer blocks
        # TREAD initialization
        routes = self._tread_routes or []
        router = self._tread_router
        use_routing = self.training and len(routes) > 0 and torch.is_grad_enabled()

        grad_enabled = torch.is_grad_enabled()
        musubi_manager = self._musubi_block_swap
        musubi_offload_active = False
        if musubi_manager is not None:
            musubi_offload_active = musubi_manager.activate(self.transformer_blocks, hidden_states.device, grad_enabled)

        capture_idx = 0
        for bid, block in enumerate(self.transformer_blocks):
            # TREAD routing for this layer
            if use_routing:
                # Check if this layer should use routing
                for route in routes:
                    start_idx = route["start_layer_idx"]
                    end_idx = route["end_layer_idx"]
                    # Handle negative indices
                    if start_idx < 0:
                        start_idx = len(self.transformer_blocks) + start_idx
                    if end_idx < 0:
                        end_idx = len(self.transformer_blocks) + end_idx

                    if start_idx <= bid <= end_idx:
                        mask_info = router.get_mask(
                            hidden_states.shape[1], route["selection_ratio"], force_keep_mask=force_keep_mask
                        )
                        hidden_states = router.start_route(hidden_states, mask_info)
                        break
            if musubi_offload_active and musubi_manager.is_managed_block(bid):
                musubi_manager.stream_in(block, hidden_states.device)
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    embedded_timestep,
                    temb,
                    image_rotary_emb,
                    extra_pos_emb,
                    attention_mask,
                )
            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    embedded_timestep=embedded_timestep,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    extra_pos_emb=extra_pos_emb,
                    attention_mask=attention_mask,
                )

            # TREAD end routing for this layer
            if use_routing:
                # Check if this layer should end routing
                for route in routes:
                    start_idx = route["start_layer_idx"]
                    end_idx = route["end_layer_idx"]
                    # Handle negative indices
                    if start_idx < 0:
                        start_idx = len(self.transformer_blocks) + start_idx
                    if end_idx < 0:
                        end_idx = len(self.transformer_blocks) + end_idx

                    if start_idx <= bid <= end_idx:
                        mask_info = router.get_mask(
                            hidden_states.shape[1], route["selection_ratio"], force_keep_mask=force_keep_mask
                        )
                        hidden_states = router.end_route(hidden_states, mask_info)
                        break

            if musubi_offload_active and musubi_manager.is_managed_block(bid):
                musubi_manager.stream_out(block)
            _store_hidden_state(hidden_states_buffer, f"layer_{capture_idx}", hidden_states)
            capture_idx += 1

        # 6. Output norm & projection & unpatchify
        hidden_states = self.norm_out(hidden_states, embedded_timestep, temb)
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.unflatten(2, (p_h, p_w, p_t, -1))
        hidden_states = hidden_states.unflatten(1, (post_patch_num_frames, post_patch_height, post_patch_width))
        # NOTE: The permutation order here is not the inverse operation of what happens when patching as usually expected.
        # It might be a source of confusion to the reader, but this is correct
        hidden_states = hidden_states.permute(0, 7, 1, 6, 2, 4, 3, 5)
        hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer2DModelOutput(sample=hidden_states)
