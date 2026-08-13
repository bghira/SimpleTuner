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
from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import TimestepEmbedding
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils.torch_utils import lru_cache_unless_export

from simpletuner.helpers.models.flowmap import (
    blend_flowmap_embeddings,
    clone_flowmap_embedder,
    prepare_flowmap_delta_timestep,
    register_flowmap_config,
    register_flowmap_gate_buffer,
    set_flowmap_gate,
    validate_flowmap_deltatime_type,
)


class MiniMaxMusic3FourierEmbedding(nn.Module):
    """Random Fourier features over the flow-matching time in `[0, 1]`. The projection is a trained checkpoint weight."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(embedding_dim // 2, 1))

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        angles = 2.0 * math.pi * timestep.unsqueeze(-1) @ self.weight.T
        return torch.cat((angles.cos(), angles.sin()), dim=-1)


class MiniMaxMusic3RotaryEmbedding(nn.Module):
    """Partial rotary embedding: only the first `rotary_dim` dimensions of each head rotate."""

    def __init__(self, rotary_dim: int, theta: float = 10000.0):
        super().__init__()
        self.rotary_dim = rotary_dim
        self.theta = theta

    @lru_cache_unless_export(maxsize=32)
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.rotary_dim, 2, device=device).float() / self.rotary_dim))
        steps = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(steps, inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos().contiguous(), freqs.sin().contiguous()


def _apply_partial_rotary_emb(hidden_states: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    # hidden_states: [batch, seq, heads, head_dim]; only the leading rotary dims rotate.
    cos, sin = rotary_emb
    rotary_dim = cos.shape[-1]
    cos = cos[:, None, :].to(hidden_states.dtype)
    sin = sin[:, None, :].to(hidden_states.dtype)
    rotated = hidden_states[..., :rotary_dim]
    half_first, half_second = rotated.chunk(2, dim=-1)
    rotate_half = torch.cat((-half_second, half_first), dim=-1)
    rotated = rotated * cos + rotate_half * sin
    return torch.cat((rotated, hidden_states[..., rotary_dim:]), dim=-1)


class MiniMaxMusic3AttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "MiniMaxMusic3Attention",
        hidden_states: torch.Tensor,
        rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.view(batch_size, seq_len, attn.heads, attn.head_dim)
        key = key.view(batch_size, seq_len, attn.heads, attn.head_dim)
        value = value.view(batch_size, seq_len, attn.heads, attn.head_dim)

        query = _apply_partial_rotary_emb(query, rotary_emb)
        key = _apply_partial_rotary_emb(key, rotary_emb)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class MiniMaxMusic3Attention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = MiniMaxMusic3AttnProcessor
    _available_processors = [MiniMaxMusic3AttnProcessor]

    def __init__(self, dim: int, heads: int, head_dim: int, processor: Optional[MiniMaxMusic3AttnProcessor] = None):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.inner_dim = heads * head_dim
        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, dim, bias=False), nn.Dropout(0.0)])
        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(self, hidden_states: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.processor(self, hidden_states, rotary_emb=rotary_emb)


class MiniMaxMusic3TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, head_dim: int, ff_inner_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MiniMaxMusic3Attention(dim, heads, head_dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff_in = nn.Linear(dim, ff_inner_dim * 2)
        self.ff_out = nn.Linear(ff_inner_dim, dim)

    def forward(self, hidden_states: torch.Tensor, rotary_emb: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), rotary_emb)
        gate_states, gate = self.ff_in(self.norm2(hidden_states)).chunk(2, dim=-1)
        hidden_states = hidden_states + self.ff_out(gate_states * torch.nn.functional.silu(gate))
        return hidden_states


class MiniMaxMusic3Transformer1DModel(ModelMixin, ConfigMixin):
    r"""
    The flow-matching diffusion transformer of MiniMax Music 3. It denoises Flow-VAE audio latents conditioned on
    per-frame hidden states produced by the autoregressive language-model stage.

    Inputs are 1D latent sequences of shape `(batch, in_channels, length)`. The conditioning signal
    (`encoder_hidden_states`, shape `(batch, length, condition_dim)`) must already be aligned to the latent timeline —
    see `MiniMaxMusic3ConditionEncoder`. The flow-matching `timestep` runs from 0 (noise) to 1 (data).
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxMusic3TransformerBlock"]
    _repeated_blocks = ["MiniMaxMusic3TransformerBlock"]
    _skip_layerwise_casting_patterns = ["time_proj", "norm"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        condition_dim: int = 2048,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        ff_inner_dim: int = 8192,
        rotary_dim: int = 32,
        fourier_embedding_dim: int = 256,
        enable_time_sign_embed: bool = False,
        gate_value: Optional[float] = None,
        deltatime_type: Optional[str] = None,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim
        # The transformer input concatenates [latent, zeros(in_channels), condition] along channels.
        concat_channels = 2 * in_channels + condition_dim

        self.time_proj = MiniMaxMusic3FourierEmbedding(fourier_embedding_dim)
        self.time_embed = TimestepEmbedding(fourier_embedding_dim, inner_dim)
        self.delta_time_embed: Optional[nn.Module] = None
        self.flowmap_deltatime_type: Optional[str] = None
        register_flowmap_gate_buffer(self, gate_value=0.25 if gate_value is None else float(gate_value))
        if deltatime_type is not None:
            self.enable_flowmap_time_conditioning(
                gate_value=0.25 if gate_value is None else float(gate_value),
                deltatime_type=deltatime_type,
            )
        self.time_sign_embed = None
        if enable_time_sign_embed:
            self.time_sign_embed = nn.Embedding(2, inner_dim)
            nn.init.zeros_(self.time_sign_embed.weight)

        self.preprocess_conv = nn.Conv1d(concat_channels, concat_channels, 1, bias=False)
        self.proj_in = nn.Linear(concat_channels, inner_dim, bias=False)
        self.rotary_emb = MiniMaxMusic3RotaryEmbedding(rotary_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxMusic3TransformerBlock(inner_dim, num_attention_heads, attention_head_dim, ff_inner_dim)
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, in_channels, bias=False)
        self.postprocess_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)

        self.gradient_checkpointing = False

    def enable_flowmap_time_conditioning(self, gate_value: float = 0.25, deltatime_type: str = "r") -> None:
        self.flowmap_deltatime_type = validate_flowmap_deltatime_type(deltatime_type, model_name="MiniMax Music 3")
        if self.delta_time_embed is None:
            self.delta_time_embed = clone_flowmap_embedder(self.time_embed)
        set_flowmap_gate(self, gate_value)
        register_flowmap_config(self, gate_value, deltatime_type)

    def _time_embedding(
        self,
        timestep: torch.Tensor,
        *,
        timestep_sign: Optional[torch.Tensor] = None,
        r_timestep: Optional[torch.Tensor] = None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        timestep_proj = self.time_proj(timestep.reshape(-1)).to(dtype=dtype)
        temb = self.time_embed(timestep_proj)
        if timestep.ndim > 1:
            temb = temb.view(*timestep.shape, -1)

        if r_timestep is not None:
            if self.delta_time_embed is None or self.flowmap_deltatime_type is None:
                raise ValueError(
                    "MiniMax Music 3 FlowMap conditioning requires `enable_flowmap_time_conditioning()` before training."
                )
            delta_timestep = prepare_flowmap_delta_timestep(
                timestep,
                r_timestep,
                self.flowmap_deltatime_type,
                model_name="MiniMax Music 3",
            )
            delta_proj = self.time_proj(delta_timestep.reshape(-1)).to(dtype=dtype)
            delta_temb = self.delta_time_embed(delta_proj)
            if delta_timestep.ndim > 1:
                delta_temb = delta_temb.view(*delta_timestep.shape, -1)
            temb = blend_flowmap_embeddings(temb, delta_temb, self.flowmap_delta_emb_gate)

        if timestep_sign is not None:
            if self.time_sign_embed is None:
                raise ValueError(
                    "timestep_sign was provided but the model was loaded without `enable_time_sign_embed=True`. "
                    "Enable TwinFlow before loading the MiniMax Music 3 transformer."
                )
            sign_tensor = timestep_sign.to(device=temb.device)
            expected_shape = temb.shape[:-1]
            if sign_tensor.ndim == 0:
                sign_tensor = sign_tensor.expand(expected_shape)
            elif sign_tensor.shape == expected_shape:
                pass
            elif sign_tensor.ndim == 1 and len(expected_shape) == 1:
                if sign_tensor.shape[0] == 1:
                    sign_tensor = sign_tensor.expand(expected_shape)
                elif sign_tensor.shape[0] != expected_shape[0]:
                    raise ValueError(
                        f"MiniMax Music 3 timestep_sign expected 1 or {expected_shape[0]} values, got {sign_tensor.shape[0]}."
                    )
            elif sign_tensor.ndim == 1 and len(expected_shape) == 2:
                if sign_tensor.shape[0] == 1:
                    sign_tensor = sign_tensor.expand(expected_shape[0])
                if sign_tensor.shape[0] == expected_shape[0]:
                    sign_tensor = sign_tensor[:, None].expand(expected_shape)
                else:
                    raise ValueError(
                        f"MiniMax Music 3 timestep_sign expected 1, {expected_shape[0]}, or {expected_shape} values, "
                        f"got {tuple(sign_tensor.shape)}."
                    )
            else:
                raise ValueError(
                    "MiniMax Music 3 timestep_sign expected scalar, batch, or tokenwise values matching timesteps, "
                    f"got shape {tuple(sign_tensor.shape)}."
                )
            sign_idx = (sign_tensor < 0).long().to(device=temb.device)
            temb = temb + self.time_sign_embed(sign_idx).to(device=temb.device, dtype=temb.dtype)

        return temb

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
        timestep_sign: Optional[torch.Tensor] = None,
        r_timestep: Optional[torch.Tensor] = None,
        skip_layers: Optional[list[int]] = None,
        hidden_states_buffer: Optional[dict[str, torch.Tensor]] = None,
        output_hidden_states: bool = False,
        hidden_state_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor] | Transformer2DModelOutput:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch, in_channels, length)`):
                Noisy Flow-VAE latents.
            timestep (`torch.Tensor` of shape `(batch,)`):
                Flow-matching time in `[0, 1]`, where 0 is pure noise and 1 is data.
            encoder_hidden_states (`torch.Tensor` of shape `(batch, length, condition_dim)`):
                Frame-aligned conditioning from `MiniMaxMusic3ConditionEncoder`. Pass zeros for the unconditional
                branch of classifier-free guidance.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`~models.transformers.transformer_minimax_music3.Transformer2DModelOutput`]
                instead of a plain tuple.
            timestep_sign (`torch.Tensor`, *optional*):
                TwinFlow time-sign values. Requires `enable_time_sign_embed=True`.
            r_timestep (`torch.Tensor`, *optional*):
                FlowMap/AnyFlow reference timestep. Requires `enable_flowmap_time_conditioning()`.
            skip_layers (`list` of `int`, *optional*):
                Transformer block indexes to skip for LayerSkip/LayerSync workflows.
            hidden_states_buffer (`dict`, *optional*):
                Buffer used by CREPA and LayerSync to capture per-layer latent-token hidden states.
            output_hidden_states (`bool`, defaults to `False`):
                Whether to include captured hidden states in tuple outputs.
            hidden_state_layer (`int`, *optional*):
                Single block index to return as the captured hidden state.

        Returns:
            The predicted flow-matching velocity with the same shape as `hidden_states`.
        """
        zeros = torch.zeros_like(hidden_states)
        hidden_states = torch.cat((hidden_states, zeros, encoder_hidden_states.transpose(1, 2)), dim=1)
        hidden_states = self.preprocess_conv(hidden_states) + hidden_states
        hidden_states = hidden_states.transpose(1, 2)

        temb = self._time_embedding(
            timestep,
            timestep_sign=timestep_sign,
            r_timestep=r_timestep,
            dtype=hidden_states.dtype,
        )

        hidden_states = self.proj_in(hidden_states)
        # The timestep embedding is prepended as one extra token and removed after the blocks.
        if temb.ndim == 2:
            time_token = temb.unsqueeze(1)
        elif temb.ndim == 3:
            if temb.shape[:2] != hidden_states.shape[:2]:
                raise ValueError(
                    "MiniMax Music 3 tokenwise timestep embeddings must match latent token shape "
                    f"{tuple(hidden_states.shape[:2])}, got {tuple(temb.shape[:2])}."
                )
            hidden_states = hidden_states + temb.to(device=hidden_states.device, dtype=hidden_states.dtype)
            time_token = temb.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"MiniMax Music 3 timestep embedding must be 2-D or 3-D, got {tuple(temb.shape)}.")
        hidden_states = torch.cat((time_token, hidden_states), dim=1)
        rotary_emb = self.rotary_emb(hidden_states.shape[1], hidden_states.device)

        skip_set = set(skip_layers) if skip_layers is not None else set()
        capture_layer = int(hidden_state_layer) if hidden_state_layer is not None else None
        captured_hidden_states = None
        for block_idx, block in enumerate(self.transformer_blocks):
            if block_idx in skip_set:
                continue
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, rotary_emb)
            else:
                hidden_states = block(hidden_states, rotary_emb)
            if hidden_states_buffer is not None:
                hidden_states_buffer[f"layer_{block_idx}"] = hidden_states[:, 1:]
            if output_hidden_states and (capture_layer is None or block_idx == capture_layer):
                captured_hidden_states = hidden_states[:, 1:]
                if capture_layer is not None:
                    output_hidden_states = False

        hidden_states = self.proj_out(hidden_states[:, 1:])
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.postprocess_conv(hidden_states) + hidden_states

        if not return_dict:
            if captured_hidden_states is not None:
                return (hidden_states, captured_hidden_states)
            return (hidden_states,)
        output = Transformer2DModelOutput(sample=hidden_states)
        output.hidden_states = captured_hidden_states
        return output
