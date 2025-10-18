# Copyright 2025 Qwen-Image Team, The HuggingFace Team, and 2025 bghira. All rights reserved.
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

import functools
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.attention_processor import Attention
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, RMSNorm
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph

from simpletuner.helpers.training.tread import TREADRouter
from simpletuner.helpers.utils.patching import MutableModuleList, PatchableModule

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _enable_safe_half_full():
    if not torch.backends.mps.is_available():
        return

    if getattr(torch.full, "__wrapped_safe_half_full__", False):
        return

    original_full = torch.full

    def safe_full(*args, **kwargs):
        dtype = kwargs.get("dtype")
        if dtype is torch.float16:
            try:
                return original_full(*args, **kwargs)
            except RuntimeError as exc:  # pragma: no cover
                if "cannot be converted to type at::Half" in str(exc):
                    kwargs_fp32 = dict(kwargs)
                    kwargs_fp32["dtype"] = torch.float32
                    result = original_full(*args, **kwargs_fp32)
                    return result.to(torch.float16)
                raise
        return original_full(*args, **kwargs)

    safe_full.__wrapped_safe_half_full__ = True
    torch.full = safe_full


_enable_safe_half_full()


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> torch.Tensor:
    # sinusoidal timestep embeddings from DDPM
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def apply_rotary_emb_qwen(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos.unsqueeze(0).unsqueeze(2).to(device=x.device, dtype=x.dtype)
        sin = sin.unsqueeze(0).unsqueeze(2).to(device=x.device, dtype=x.dtype)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x * cos + x_rotated * sin).to(x.dtype)

        return out
    else:
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freq_shape = freqs_cis.shape[-1]
        if freq_shape != x_rotated.shape[-1]:
            freqs_cis = freqs_cis[..., : x_rotated.shape[-1]]
        freqs_cis = freqs_cis.to(x_rotated.device)
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


class QwenTimestepProjEmbeddings(PatchableModule):
    def __init__(self, embedding_dim):
        super().__init__()

        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0, scale=1000)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.timestep_embedder.time_embed_dim = embedding_dim
        self.time_embed_dim = embedding_dim

    def forward(self, timestep, *states, guidance=None, hidden_states=None):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj)

        target_tensor: Optional[torch.Tensor] = None
        guidance_tensor = guidance if isinstance(guidance, torch.Tensor) else None

        collected_states: List[torch.Tensor] = list(states)
        if hidden_states is not None:
            collected_states.append(hidden_states)

        for state in collected_states:
            if not isinstance(state, torch.Tensor):
                continue
            if state.dim() == 1 and guidance_tensor is None:
                guidance_tensor = state
                continue
            target_tensor = state
            break

        if target_tensor is None and guidance_tensor is not None:
            target_tensor = guidance_tensor

        if target_tensor is None:
            target_tensor = timesteps_emb

        conditioning = timesteps_emb.to(device=target_tensor.device, dtype=target_tensor.dtype)

        if guidance_tensor is not None:
            guidance_embed = guidance_tensor.to(device=conditioning.device, dtype=conditioning.dtype)
            guidance_embed = guidance_embed.unsqueeze(-1).expand_as(conditioning)
            conditioning = conditioning + guidance_embed

        return conditioning


class QwenEmbedRope(PatchableModule):
    def __init__(self, theta: int, axes_dim: List[int], scale_rope=False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        self.pos_freqs = torch.cat(
            [
                self.rope_params(pos_index, self.axes_dim[0], self.theta),
                self.rope_params(pos_index, self.axes_dim[1], self.theta),
                self.rope_params(pos_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.neg_freqs = torch.cat(
            [
                self.rope_params(neg_index, self.axes_dim[0], self.theta),
                self.rope_params(neg_index, self.axes_dim[1], self.theta),
                self.rope_params(neg_index, self.axes_dim[2], self.theta),
            ],
            dim=1,
        )
        self.rope_cache = {}

        # DO NOT USING REGISTER BUFFER HERE, IT WILL CAUSE COMPLEX NUMBERS LOSE ITS IMAGINARY PART
        self.scale_rope = scale_rope

    def rope_params(self, index, dim, theta=10000):
        assert dim % 2 == 0
        freqs = torch.outer(index, 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)))
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def forward(self, video_fhw, txt_seq_lens, device):
        if self.pos_freqs.device != device:
            self.pos_freqs = self.pos_freqs.to(device)
            self.neg_freqs = self.neg_freqs.to(device)

        # Normalise input so that we always iterate over a list of (frame, height, width) tuples
        if isinstance(video_fhw, (tuple, list)):
            # ``video_fhw`` can be provided either as a single triple or as a list of triples.
            if len(video_fhw) == 0:
                video_fhw = []
            elif isinstance(video_fhw[0], (list, tuple)) and len(video_fhw[0]) == 3:
                video_fhw = [tuple(v) for v in video_fhw]
            elif len(video_fhw) == 3:
                video_fhw = [tuple(video_fhw)]
            else:
                video_fhw = [tuple(video_fhw)]
        else:
            video_fhw = [video_fhw]

        vid_freqs = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            rope_key = f"{idx}_{height}_{width}"

            if not torch.compiler.is_compiling():
                if rope_key not in self.rope_cache:
                    self.rope_cache[rope_key] = self._compute_video_freqs(frame, height, width, idx)
                video_freq = self.rope_cache[rope_key]
            else:
                video_freq = self._compute_video_freqs(frame, height, width, idx)
            video_freq = video_freq.to(device)
            vid_freqs.append(video_freq)

            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        max_len = max(txt_seq_lens)
        txt_freqs = self.pos_freqs[max_vid_index : max_vid_index + max_len, ...]
        vid_freqs = torch.cat(vid_freqs, dim=0)

        return vid_freqs, txt_freqs

    @functools.lru_cache(maxsize=None)
    def _compute_video_freqs(self, frame, height, width, idx=0):
        seq_lens = frame * height * width
        freqs_pos = self.pos_freqs.split([x // 2 for x in self.axes_dim], dim=1)
        freqs_neg = self.neg_freqs.split([x // 2 for x in self.axes_dim], dim=1)

        freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
        if self.scale_rope:
            freqs_height = torch.cat([freqs_neg[1][-(height - height // 2) :], freqs_pos[1][: height // 2]], dim=0)
            freqs_height = freqs_height.view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = torch.cat([freqs_neg[2][-(width - width // 2) :], freqs_pos[2][: width // 2]], dim=0)
            freqs_width = freqs_width.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            freqs_height = freqs_pos[1][:height].view(1, height, 1, -1).expand(frame, height, width, -1)
            freqs_width = freqs_pos[2][:width].view(1, 1, width, -1).expand(frame, height, width, -1)

        freqs = torch.cat([freqs_frame, freqs_height, freqs_width], dim=-1).reshape(seq_lens, -1)
        return freqs.clone().contiguous()


class QwenDoubleStreamAttnProcessor2_0:
    # joint attention for text and image streams

    _attention_backend = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention") or not callable(F.scaled_dot_product_attention):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)
        img_key = attn.to_k(hidden_states)
        img_value = attn.to_v(hidden_states)

        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)
        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb
            img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        joint_hidden_states = dispatch_attention_fn(
            joint_query,
            joint_key,
            joint_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        # Reshape back
        joint_hidden_states = joint_hidden_states.flatten(2, 3)
        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output


@maybe_allow_in_graph
class QwenImageTransformerBlock(PatchableModule):
    def __init__(
        self, dim: int, num_attention_heads: int, attention_head_dim: int, qk_norm: str = "rms_norm", eps: float = 1e-6
    ):
        super().__init__()

        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim

        # Image processing modules
        self.img_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,  # Enable cross attention for joint computation
            added_kv_proj_dim=dim,  # Enable added KV projections for text stream
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            processor=QwenDoubleStreamAttnProcessor2_0(),
            qk_norm=qk_norm,
            eps=eps,
        )
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.img_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        # Text processing modules
        self.txt_mod = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),  # For scale, shift, gate for norm1 and norm2
        )
        self.txt_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        # Text doesn't need separate attention - it's handled by img_attn joint computation
        self.txt_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.txt_mlp = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

    def _modulate(self, x, mod_params):
        shift, scale, gate = mod_params.chunk(3, dim=-1)
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1), gate.unsqueeze(1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get modulation parameters for both streams
        if temb.shape[-1] == self.dim:
            mod_dtype = self.img_mod[1].weight.dtype
            img_mod_params = self.img_mod(temb.to(mod_dtype)).to(temb.dtype)
            txt_mod_params = self.txt_mod(temb.to(mod_dtype)).to(temb.dtype)
        elif temb.shape[-1] == 6 * self.dim:
            img_mod_params = temb.to(hidden_states.dtype)
            txt_mod_params = temb.to(encoder_hidden_states.dtype)
        else:
            raise ValueError(f"Expected modulation embedding of size {self.dim} or {6 * self.dim}, got {temb.shape[-1]}")

        # Split modulation parameters for norm1 and norm2
        img_mod1, img_mod2 = img_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]
        txt_mod1, txt_mod2 = txt_mod_params.chunk(2, dim=-1)  # Each [B, 3*dim]

        # Process image stream - norm1 + modulation
        img_normed = self.img_norm1(hidden_states)
        img_modulated, img_gate1 = self._modulate(img_normed, img_mod1)

        # Process text stream - norm1 + modulation
        txt_normed = self.txt_norm1(encoder_hidden_states)
        txt_modulated, txt_gate1 = self._modulate(txt_normed, txt_mod1)

        # joint attention: compute QKV, apply norm/rope, concat, split
        attn_inputs = {
            "hidden_states": img_modulated,
            "encoder_hidden_states": txt_modulated,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "image_rotary_emb": image_rotary_emb,
        }
        if joint_attention_kwargs:
            attn_inputs.update(joint_attention_kwargs)

        attn_output = self.attn(**attn_inputs)

        if not hasattr(self.attn, "call_args"):
            self.attn.call_args = SimpleNamespace(args=(), kwargs={k: v for k, v in attn_inputs.items()})

        # attention processor returns (img_output, txt_output)
        img_attn_output, txt_attn_output = attn_output

        # apply gates and residual
        hidden_states = hidden_states + img_gate1 * img_attn_output
        encoder_hidden_states = encoder_hidden_states + txt_gate1 * txt_attn_output

        # Process image stream - norm2 + MLP
        img_normed2 = self.img_norm2(hidden_states)
        img_modulated2, img_gate2 = self._modulate(img_normed2, img_mod2)
        img_mlp_output = self.img_mlp(img_modulated2.to(torch.float32)).to(img_modulated2.dtype)
        hidden_states = hidden_states + img_gate2 * img_mlp_output

        # Process text stream - norm2 + MLP
        txt_normed2 = self.txt_norm2(encoder_hidden_states)
        txt_modulated2, txt_gate2 = self._modulate(txt_normed2, txt_mod2)
        txt_mlp_output = self.txt_mlp(txt_modulated2.to(torch.float32)).to(txt_modulated2.dtype)
        encoder_hidden_states = encoder_hidden_states + txt_gate2 * txt_mlp_output

        # clip for fp16 overflow prevention
        if torch.isnan(encoder_hidden_states).any() or torch.isinf(encoder_hidden_states).any():
            encoder_hidden_states = torch.nan_to_num(encoder_hidden_states, nan=0.0, posinf=65504, neginf=-65504)
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
        if torch.isnan(hidden_states).any() or torch.isinf(hidden_states).any():
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0, posinf=65504, neginf=-65504)
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class QwenImageTransformer2DModel(
    PatchableModule, ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    # qwen dual-stream transformer model

    _supports_gradient_checkpointing = True
    _no_split_modules = ["QwenImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]
    _repeated_blocks = ["QwenImageTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 64,
        out_channels: Optional[int] = 16,
        num_layers: int = 60,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,  # TODO: this should probably be removed
        axes_dims_rope: Tuple[int, int, int] = (16, 56, 56),
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = QwenEmbedRope(theta=10000, axes_dim=list(axes_dims_rope), scale_rope=True)

        self.time_text_embed = QwenTimestepProjEmbeddings(embedding_dim=self.inner_dim)

        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)

        self._patch_area = patch_size * patch_size
        self.img_in = nn.Linear(in_channels, self.inner_dim)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)

        self.transformer_blocks = MutableModuleList(
            [
                QwenImageTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

        # TREAD support
        self._tread_router = None
        self._tread_routes = None

    def set_router(self, router: TREADRouter, routes: Optional[List[Dict]] = None):
        self._tread_router = router
        self._tread_routes = routes

    def _flatten_image_latents(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        if hidden_states.ndim != 4:
            return hidden_states, None, None

        batch_size, channels, height, width = hidden_states.shape
        patch_size = self.config.patch_size
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(f"Height ({height}) and width ({width}) must be divisible by patch_size ({patch_size}).")

        patches = torch.nn.functional.unfold(
            hidden_states,
            kernel_size=patch_size,
            stride=patch_size,
        )
        patches = patches.transpose(1, 2)
        return patches, height // patch_size, width // patch_size

    def _unflatten_image_latents(
        self,
        hidden_states: torch.Tensor,
        img_shapes: Optional[List[Tuple[int, int, int]]],
        patch_grid: Tuple[int, int],
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            return hidden_states

        batch_size = hidden_states.shape[0]
        if not img_shapes:
            raise ValueError("img_shapes must be provided to reconstruct image latents.")

        if len(img_shapes) == 1 and batch_size > 1:
            img_shapes = img_shapes * batch_size

        patch_size = self.config.patch_size
        out_channels = self.out_channels
        expected_features = self._patch_area * out_channels
        if hidden_states.shape[-1] != expected_features:
            raise ValueError(f"Expected last dimension to be {expected_features}, got {hidden_states.shape[-1]}.")

        outputs: List[torch.Tensor] = []
        patch_height, patch_width = patch_grid

        for idx, sample in enumerate(hidden_states):
            frames, latent_h, latent_w = img_shapes[idx]
            tokens_expected = frames * latent_h * latent_w
            if sample.shape[0] != tokens_expected:
                raise ValueError(
                    f"Token count mismatch for sample {idx}: expected {tokens_expected}, got {sample.shape[0]}."
                )

            sample = sample.view(frames, latent_h, latent_w, patch_size, patch_size, out_channels)
            sample = sample.permute(0, 5, 1, 3, 2, 4)
            sample = sample.reshape(frames * out_channels, latent_h * patch_size, latent_w * patch_size)
            outputs.append(sample)

        output = torch.stack(outputs, dim=0)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,  # TODO: this should probably be removed
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        force_keep_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight lora layers
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        hidden_states, patch_h, patch_w = self._flatten_image_latents(hidden_states)

        if img_shapes is None:
            if patch_h is None or patch_w is None:
                raise ValueError("img_shapes must be provided when hidden_states are already flattened.")
            img_shapes = [(1, patch_h, patch_w)] * hidden_states.shape[0]

        if patch_h is None or patch_w is None:
            patch_h = img_shapes[0][1]
            patch_w = img_shapes[0][2]

        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, hidden_states)
            if guidance is None
            else self.time_text_embed(timestep, guidance, hidden_states)
        )

        image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)

        # tread routing setup
        routes = self._tread_routes or []
        router = self._tread_router
        use_routing = self.training and len(routes) > 0 and torch.is_grad_enabled()

        for index_block, block in enumerate(self.transformer_blocks):
            # tread routing
            if use_routing:
                # check layer routing
                for route in routes:
                    start_idx = route["start_layer_idx"]
                    end_idx = route["end_layer_idx"]
                    # handle negative indices
                    if start_idx < 0:
                        start_idx = len(self.transformer_blocks) + start_idx
                    if end_idx < 0:
                        end_idx = len(self.transformer_blocks) + end_idx

                    if start_idx <= index_block <= end_idx:
                        mask_info = router.get_mask(
                            hidden_states.shape[1], route["selection_ratio"], force_keep_mask=force_keep_mask
                        )
                        hidden_states = router.start_route(hidden_states, mask_info)
                        break
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=attention_kwargs,
                )

            # tread end routing
            if use_routing:
                # check end routing
                for route in routes:
                    start_idx = route["start_layer_idx"]
                    end_idx = route["end_layer_idx"]
                    # handle negative indices
                    if start_idx < 0:
                        start_idx = len(self.transformer_blocks) + start_idx
                    if end_idx < 0:
                        end_idx = len(self.transformer_blocks) + end_idx

                    if start_idx <= index_block <= end_idx:
                        mask_info = router.get_mask(
                            hidden_states.shape[1], route["selection_ratio"], force_keep_mask=force_keep_mask
                        )
                        hidden_states = router.end_route(hidden_states, mask_info)
                        break

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        # use image part from dual-stream
        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        output = self._unflatten_image_latents(output, img_shapes, (patch_h, patch_w))

        if USE_PEFT_BACKEND:
            # remove lora scale
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
