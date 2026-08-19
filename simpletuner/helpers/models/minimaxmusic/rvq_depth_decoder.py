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

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import RMSNorm


class MiniMaxMusic3DepthAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn: "MiniMaxMusic3DepthAttention", hidden_states: torch.Tensor) -> torch.Tensor:
        query, key, value = self._project(attn, hidden_states)
        return self._attention(attn, query, key, value, is_causal=True)

    def forward_with_cache(
        self,
        attn: "MiniMaxMusic3DepthAttention",
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        query, key, value = self._project(attn, hidden_states)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=1)
            value = torch.cat((past_key_value[1], value), dim=1)
        output = self._attention(attn, query, key, value, is_causal=past_key_value is None)
        return output, (key, value)

    @staticmethod
    def _project(
        attn: "MiniMaxMusic3DepthAttention", hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = query.view(batch_size, seq_len, attn.heads, attn.head_dim)
        key = key.view(batch_size, seq_len, attn.heads, attn.head_dim)
        value = value.view(batch_size, seq_len, attn.heads, attn.head_dim)
        return query, key, value

    def _attention(
        self,
        attn: "MiniMaxMusic3DepthAttention",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            is_causal=is_causal,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3).to(query.dtype)
        return attn.to_out(hidden_states)


class MiniMaxMusic3DepthAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = MiniMaxMusic3DepthAttnProcessor
    _available_processors = [MiniMaxMusic3DepthAttnProcessor]

    def __init__(self, dim: int, heads: int, processor: Optional[MiniMaxMusic3DepthAttnProcessor] = None):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.processor(self, hidden_states)

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        return self.processor.forward_with_cache(self, hidden_states, past_key_value)


class MiniMaxMusic3DepthDecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, intermediate_size: int):
        super().__init__()
        self.input_layernorm = RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.attn = MiniMaxMusic3DepthAttention(dim, heads)
        self.post_attention_layernorm = RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(self.input_layernorm(hidden_states))
        norm_states = self.post_attention_layernorm(hidden_states)
        return hidden_states + self.down_proj(F.silu(self.gate_proj(norm_states)) * self.up_proj(norm_states))

    def forward_with_cache(
        self,
        hidden_states: torch.Tensor,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attention_output, present_key_value = self.attn.forward_with_cache(
            self.input_layernorm(hidden_states), past_key_value
        )
        hidden_states = hidden_states + attention_output
        norm_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + self.down_proj(F.silu(self.gate_proj(norm_states)) * self.up_proj(norm_states))
        return hidden_states, present_key_value


class MiniMaxMusic3RVQDepthDecoder(ModelMixin, ConfigMixin):
    r"""
    The local language model of MiniMax Music 3. Within each audio frame it autoregressively predicts the seven
    residual RVQ codebooks (c1..c7) from the global language model's hidden state and the frame's semantic code, and
    exposes the per-step hidden states that condition the flow-matching transformer.

    It also owns the embedding table for the residual codebooks, which the pipeline uses to embed complete frames for
    the global language model's feedback loop.
    """

    @register_to_config
    def __init__(
        self,
        hidden_size: int = 4096,
        num_layers: int = 4,
        num_attention_heads: int = 16,
        intermediate_size: int = 6144,
        audio_vocab_size: int = 1024,
        num_codebooks: int = 8,
        max_position_embeddings: int = 16,
    ):
        super().__init__()
        self.audio_embeddings = nn.Embedding(audio_vocab_size * (num_codebooks - 1), hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.pos_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.layers = nn.ModuleList(
            [MiniMaxMusic3DepthDecoderBlock(hidden_size, num_attention_heads, intermediate_size) for _ in range(num_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=1e-6, elementwise_affine=True)
        self.audio_heads = nn.ModuleList(
            [nn.Linear(hidden_size, audio_vocab_size, bias=False) for _ in range(num_codebooks - 1)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            inputs_embeds (`torch.Tensor` of shape `(batch, steps, hidden_size)`):
                Projected depth-sequence embeddings: the global hidden state followed by the embedded codes sampled so
                far, each passed through `projection`.

        Returns:
            `torch.Tensor` of shape `(batch, steps, hidden_size)`: normalized hidden states; the last step feeds the
            next codebook head.
        """
        positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
        hidden_states = inputs_embeds + self.pos_embedding(positions).unsqueeze(0)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)

    def forward_with_cache(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past key values must contain one entry per depth-decoder layer")
        past_length = 0 if past_key_values is None else past_key_values[0][0].shape[1]
        positions = torch.arange(past_length, past_length + inputs_embeds.shape[1], device=inputs_embeds.device)
        if positions[-1] >= self.config.max_position_embeddings:
            raise ValueError("depth-decoder sequence exceeds max_position_embeddings")
        hidden_states = inputs_embeds + self.pos_embedding(positions).unsqueeze(0)
        present_key_values = []
        for index, layer in enumerate(self.layers):
            past_key_value = None if past_key_values is None else past_key_values[index]
            hidden_states, present_key_value = layer.forward_with_cache(hidden_states, past_key_value)
            present_key_values.append(present_key_value)
        return self.norm(hidden_states), tuple(present_key_values)
