# Copyright 2025 SimpleTuner contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import logging
import os
from typing import Dict

import torch
import torch.nn as nn

try:
    from safetensors.torch import load_file
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise ImportError("HeartMuLa requires safetensors for weight loading.") from exc

from transformers import LlamaConfig, LlamaModel
from transformers.modeling_utils import PreTrainedModel

from .configuration_heartmula import HeartMuLaConfig

logger = logging.getLogger(__name__)

_LLAMA_FLAVORS: Dict[str, dict] = {
    "llama-3B": {
        "num_layers": 28,
        "num_heads": 24,
        "num_kv_heads": 8,
        "embed_dim": 3072,
        "max_seq_len": 8192,
        "intermediate_dim": 8192,
    },
    "llama-300M": {
        "num_layers": 3,
        "num_heads": 8,
        "num_kv_heads": 4,
        "embed_dim": 3072,
        "max_seq_len": 2048,
        "intermediate_dim": 8192,
    },
    "llama-7B": {
        "num_layers": 32,
        "num_heads": 32,
        "num_kv_heads": 8,
        "embed_dim": 4096,
        "max_seq_len": 8192,
        "intermediate_dim": 14336,
    },
    "llama-400M": {
        "num_layers": 4,
        "num_heads": 8,
        "num_kv_heads": 4,
        "embed_dim": 3072,
        "max_seq_len": 2048,
        "intermediate_dim": 8192,
    },
}


def _build_llama_config(flavor: str, vocab_size: int) -> LlamaConfig:
    if flavor not in _LLAMA_FLAVORS:
        raise ValueError(f"Unsupported HeartMuLa Llama flavor: {flavor}")
    cfg = _LLAMA_FLAVORS[flavor]
    max_seq_len = int(cfg["max_seq_len"])
    rope_scaling = {
        "rope_type": "llama3",
        "factor": 32.0,
        "original_max_position_embeddings": max_seq_len,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
    }
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=int(cfg["embed_dim"]),
        intermediate_size=int(cfg["intermediate_dim"]),
        num_hidden_layers=int(cfg["num_layers"]),
        num_attention_heads=int(cfg["num_heads"]),
        num_key_value_heads=int(cfg["num_kv_heads"]),
        max_position_embeddings=max_seq_len,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        rope_scaling=rope_scaling,
        attention_bias=False,
        mlp_bias=False,
        use_cache=True,
        tie_word_embeddings=False,
    )


def _map_llama_key(key: str, prefix: str) -> str:
    key = key.replace(f"{prefix}layers.", f"{prefix}layers.")
    key = key.replace(".attn.q_proj.", ".self_attn.q_proj.")
    key = key.replace(".attn.k_proj.", ".self_attn.k_proj.")
    key = key.replace(".attn.v_proj.", ".self_attn.v_proj.")
    key = key.replace(".attn.output_proj.", ".self_attn.o_proj.")
    key = key.replace(".mlp.w1.", ".mlp.gate_proj.")
    key = key.replace(".mlp.w2.", ".mlp.down_proj.")
    key = key.replace(".mlp.w3.", ".mlp.up_proj.")
    key = key.replace(".sa_norm.scale", ".input_layernorm.weight")
    key = key.replace(".mlp_norm.scale", ".post_attention_layernorm.weight")
    key = key.replace(".norm.scale", ".norm.weight")
    return key


def _load_sharded_state_dict(model_path: str) -> Dict[str, torch.Tensor]:
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as fp:
            index = json.load(fp)
        weight_map = index.get("weight_map", {})
        shard_files = sorted({os.path.join(model_path, fname) for fname in weight_map.values()})
    else:
        single_file = os.path.join(model_path, "model.safetensors")
        if not os.path.isfile(single_file):
            raise FileNotFoundError(f"No HeartMuLa safetensors found in {model_path}.")
        shard_files = [single_file]

    state_dict: Dict[str, torch.Tensor] = {}
    for shard in shard_files:
        shard_state = load_file(shard)
        state_dict.update(shard_state)
    return state_dict


class HeartMuLaModel(PreTrainedModel):
    config_class = HeartMuLaConfig

    def __init__(self, config: HeartMuLaConfig):
        super().__init__(config)

        backbone_config = _build_llama_config(config.backbone_flavor, config.text_vocab_size)
        decoder_config = _build_llama_config(config.decoder_flavor, config.text_vocab_size)

        self.backbone = LlamaModel(backbone_config)
        self.decoder = LlamaModel(decoder_config)

        backbone_dim = backbone_config.hidden_size
        decoder_dim = decoder_config.hidden_size

        self.text_embeddings = nn.Embedding(config.text_vocab_size, backbone_dim)
        self.audio_embeddings = nn.Embedding(config.audio_vocab_size * config.audio_num_codebooks, backbone_dim)
        self.unconditional_text_embedding = nn.Embedding(1, backbone_dim)

        self.projection = nn.Linear(backbone_dim, decoder_dim, bias=False)
        self.codebook0_head = nn.Linear(backbone_dim, config.audio_vocab_size, bias=False)
        self.audio_head = nn.Parameter(torch.empty(config.audio_num_codebooks - 1, decoder_dim, config.audio_vocab_size))
        self.muq_linear = nn.Linear(config.muq_dim, backbone_dim)

        self.post_init()

    def _embed_audio(self, codebook: int, tokens: torch.Tensor) -> torch.Tensor:
        return self.audio_embeddings(tokens + codebook * self.config.audio_vocab_size)

    def _embed_local_audio(self, tokens: torch.Tensor) -> torch.Tensor:
        audio_tokens = tokens + (
            self.config.audio_vocab_size * torch.arange(self.config.audio_num_codebooks - 1, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            tokens.size(0), tokens.size(1), self.config.audio_num_codebooks - 1, -1
        )
        return audio_embeds

    def _embed_tokens(self, tokens: torch.Tensor, uncond_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, _ = tokens.size()
        text_embeds = self.text_embeddings(tokens[:, :, -1])

        if uncond_mask is not None:
            uncond_text_embed = self.unconditional_text_embedding(torch.zeros(1, device=tokens.device, dtype=torch.long))
            mask_expanded = uncond_mask.view(batch, 1, 1).expand_as(text_embeds)
            text_embeds = torch.where(mask_expanded, uncond_text_embed, text_embeds)

        text_embeds = text_embeds.unsqueeze(-2)

        audio_tokens = tokens[:, :, :-1] + (
            self.config.audio_vocab_size * torch.arange(self.config.audio_num_codebooks, device=tokens.device)
        )
        audio_embeds = self.audio_embeddings(audio_tokens.view(-1)).reshape(
            batch, seq_len, self.config.audio_num_codebooks, -1
        )
        return torch.cat([audio_embeds, text_embeds], dim=-2)

    def _build_backbone_inputs(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        *,
        uncond_mask: torch.Tensor | None = None,
        continuous_segments: torch.Tensor | None = None,
        starts: list[int] | None = None,
    ) -> torch.Tensor:
        embeds = self._embed_tokens(tokens, uncond_mask=uncond_mask)
        masked_embeds = embeds * tokens_mask.unsqueeze(-1).to(dtype=embeds.dtype)
        hidden = masked_embeds.sum(dim=2, dtype=embeds.dtype)

        if continuous_segments is not None:
            if starts is None:
                raise ValueError("continuous_segments provided but no start indices were supplied.")
            segments = self.muq_linear(continuous_segments)
            if uncond_mask is not None:
                uncond_embed = self.unconditional_text_embedding(torch.zeros(1, device=tokens.device, dtype=torch.long))
                mask_expanded = uncond_mask.view(hidden.shape[0], 1).expand_as(segments)
                segments = torch.where(mask_expanded, uncond_embed, segments)
            batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
            hidden[batch_indices, starts] = segments

        return hidden

    def forward_backbone(
        self,
        inputs_embeds: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = True,
    ):
        return self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

    def forward_decoder(
        self,
        inputs_embeds: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool = True,
    ):
        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        tokens_mask: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if kwargs:
            logger.warning(
                "HeartMuLaModel.forward received unexpected keyword arguments: %s",
                ", ".join(sorted(kwargs.keys())),
            )
        if attention_mask is None:
            attention_mask = tokens_mask.any(dim=-1).to(dtype=torch.long)
        hidden = self._build_backbone_inputs(tokens, tokens_mask)
        outputs = self.forward_backbone(inputs_embeds=hidden, attention_mask=attention_mask, use_cache=False)
        hidden_states = outputs.last_hidden_state

        codebook0_logits, codebook_logits = self._predict_audio_tokens(hidden_states, tokens)
        return {
            "codebook0_logits": codebook0_logits,
            "codebook_logits": codebook_logits,
            "hidden_states": hidden_states,
        }

    def _predict_audio_tokens(self, hidden_states: torch.Tensor, tokens: torch.Tensor):
        if hidden_states.dim() != 3:
            raise ValueError(
                f"HeartMuLa backbone must return [batch, seq_len, hidden] activations, got {hidden_states.shape}."
            )
        if tokens.dim() != 3:
            raise ValueError(f"HeartMuLa tokens must have shape [batch, seq_len, num_codebooks+1], got {tokens.shape}.")

        codebook0_logits = self.codebook0_head(hidden_states[:, :-1, :])

        target_audio = tokens[:, 1:, :-1]
        context = hidden_states[:, :-1, :]
        codebook_inputs = target_audio[:, :, : self.config.audio_num_codebooks - 1]
        codebook_embeds = self._embed_local_audio(codebook_inputs)

        decoder_inputs = torch.cat([context.unsqueeze(2), codebook_embeds], dim=2)
        decoder_inputs = self.projection(decoder_inputs)
        batch, frames, seq_len, embed_dim = decoder_inputs.shape
        decoder_inputs = decoder_inputs.reshape(batch * frames, seq_len, embed_dim)
        decoder_outputs = self.forward_decoder(inputs_embeds=decoder_inputs, use_cache=False)
        decoder_hidden = decoder_outputs.last_hidden_state[:, 1:, :]
        logits = torch.einsum("bqd, qdv -> bqv", decoder_hidden, self.audio_head)
        logits = logits.view(batch, frames, self.config.audio_num_codebooks - 1, self.config.audio_vocab_size)

        return codebook0_logits, logits

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        torch_dtype: torch.dtype | None = None,
        **kwargs,
    ) -> "HeartMuLaModel":
        config = HeartMuLaConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config, *model_args)

        state_dict = _load_sharded_state_dict(pretrained_model_name_or_path)
        mapped: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                mapped[_map_llama_key(key, "backbone.")] = value
            elif key.startswith("decoder."):
                mapped[_map_llama_key(key, "decoder.")] = value
            else:
                mapped[key] = value

        missing, unexpected = model.load_state_dict(mapped, strict=False)
        if unexpected:
            logger.debug("HeartMuLa unexpected keys: %s", unexpected)
        if missing:
            logger.debug("HeartMuLa missing keys: %s", missing)

        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        return model
