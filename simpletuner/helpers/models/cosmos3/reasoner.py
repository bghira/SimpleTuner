from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.normalization import RMSNorm
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import load_file

from simpletuner.helpers.models.cosmos3.transformer import (
    Cosmos3NemotronRMSNorm,
    Cosmos3ReasonerMemoryState,
    Cosmos3VLTextMLP,
    Cosmos3VLTextRotaryEmbedding,
)

logger = logging.getLogger(__name__)

COSMOS3_REASONER_COMPONENTS = {
    "edge": "SimpleTuner/cosmos3-component-reasoning-layers-bf16-edge",
    "nano": "SimpleTuner/cosmos3-component-reasoning-layers-bf16-nano",
    "super": "SimpleTuner/cosmos3-component-reasoning-layers-bf16-super",
    "super-i2v": "SimpleTuner/cosmos3-component-reasoning-layers-bf16-super-i2v",
    "super-t2i": "SimpleTuner/cosmos3-component-reasoning-layers-bf16-super-t2i",
}

COSMOS3_GENERATOR_COMPONENTS = {
    "edge": "SimpleTuner/cosmos3-component-generation-layers-bf16-edge",
    "nano": "SimpleTuner/cosmos3-component-generation-layers-bf16-nano",
    "super": "SimpleTuner/cosmos3-component-generation-layers-bf16-super",
    "super-i2v": "SimpleTuner/cosmos3-component-generation-layers-bf16-super-i2v",
    "super-t2i": "SimpleTuner/cosmos3-component-generation-layers-bf16-super-t2i",
}


@dataclass
class Cosmos3ReasonerConfig:
    attention_bias: bool = False
    head_dim: int = 128
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    num_hidden_layers: int = 36
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-6
    rope_scaling: dict | None = None
    rope_theta: float = 5000000.0
    vocab_size: int = 151936
    hidden_act: str = "silu"
    qk_norm_for_text: bool = True
    use_und_k_norm_for_gen: bool = False
    rope_axes_dim: tuple[int, int, int] | list[int] | None = None
    component: str = "cosmos3_reasoner"
    source_model_id: str | None = None
    source_revision: str | None = None
    dtype: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Cosmos3ReasonerConfig":
        values = {field: data[field] for field in cls.__dataclass_fields__ if field in data}
        if values.get("rope_axes_dim") is not None:
            values["rope_axes_dim"] = tuple(values["rope_axes_dim"])
        return cls(**values)

    def to_dict(self) -> dict:
        output = dict(self.__dict__)
        if output["rope_axes_dim"] is not None:
            output["rope_axes_dim"] = list(output["rope_axes_dim"])
        return output


class Cosmos3ReasonerAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        head_dim: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_bias: bool,
        rms_norm_eps: float,
        qk_norm_for_text: bool,
        use_und_k_norm_for_gen: bool,
        norm_type: str,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.to_q = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=attention_bias)
        self.to_k = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.to_v = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=attention_bias)
        self.to_out = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=attention_bias)
        if not qk_norm_for_text:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        elif norm_type == "nemotron_rms_norm":
            self.norm_q = Cosmos3NemotronRMSNorm(head_dim, eps=rms_norm_eps)
            self.norm_k = Cosmos3NemotronRMSNorm(head_dim, eps=rms_norm_eps)
        else:
            self.norm_q = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, bias=False)
            self.norm_k = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, bias=False)

        if use_und_k_norm_for_gen and not qk_norm_for_text:
            if norm_type == "nemotron_rms_norm":
                self.k_norm_und_for_gen = Cosmos3NemotronRMSNorm(head_dim, eps=rms_norm_eps)
            else:
                self.k_norm_und_for_gen = RMSNorm(head_dim, eps=rms_norm_eps, elementwise_affine=True, bias=False)
        else:
            self.k_norm_und_for_gen = None

    def forward(
        self,
        und_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        q_und = self.to_q(und_seq).view(-1, self.num_attention_heads, self.head_dim)
        k_und = self.to_k(und_seq).view(-1, self.num_key_value_heads, self.head_dim)
        v_und = self.to_v(und_seq).view(-1, self.num_key_value_heads, self.head_dim)
        q_und = self.norm_q(q_und)
        k_und = self.norm_k(k_und)
        k_und_for_gen = self.k_norm_und_for_gen(k_und) if self.k_norm_und_for_gen is not None else None

        cos_und, sin_und = rotary_emb
        cos_und = cos_und.unsqueeze(1)
        sin_und = sin_und.unsqueeze(1)
        q_und = q_und * cos_und + self._rotate_half(q_und) * sin_und
        k_und = k_und * cos_und + self._rotate_half(k_und) * sin_und
        if k_und_for_gen is not None:
            k_und_for_gen = k_und_for_gen * cos_und + self._rotate_half(k_und_for_gen) * sin_und

        causal_out = dispatch_attention_fn(
            q_und.unsqueeze(0),
            k_und.unsqueeze(0),
            v_und.unsqueeze(0),
            is_causal=True,
            enable_gqa=True,
        )
        causal_out = causal_out.squeeze(0).flatten(-2, -1)
        reasoner_kv = {"k": k_und, "v": v_und}
        if k_und_for_gen is not None:
            reasoner_kv["k_for_gen"] = k_und_for_gen
        return self.to_out(causal_out), reasoner_kv

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Cosmos3ReasonerLayer(nn.Module):
    def __init__(self, config: Cosmos3ReasonerConfig) -> None:
        super().__init__()
        norm_type = "nemotron_rms_norm" if config.hidden_act == "relu2" else "rms_norm"
        self.self_attn = Cosmos3ReasonerAttention(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            attention_bias=config.attention_bias,
            rms_norm_eps=config.rms_norm_eps,
            qk_norm_for_text=config.qk_norm_for_text,
            use_und_k_norm_for_gen=config.use_und_k_norm_for_gen,
            norm_type=norm_type,
        )
        self.mlp = Cosmos3VLTextMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        if norm_type == "nemotron_rms_norm":
            self.input_layernorm = Cosmos3NemotronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = Cosmos3NemotronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True, bias=False)
            self.post_attention_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True, bias=False
            )

    def forward(
        self,
        und_seq: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        und_norm = self.input_layernorm(und_seq)
        und_attn_out, reasoner_kv = self.self_attn(und_norm, rotary_emb)
        residual_und = und_seq + und_attn_out
        mlp_out = self.mlp(self.post_attention_layernorm(residual_und))
        return residual_und + mlp_out, reasoner_kv


class Cosmos3Reasoner(nn.Module):
    def __init__(self, config: Cosmos3ReasonerConfig | None = None) -> None:
        super().__init__()
        self.config = config or Cosmos3ReasonerConfig()
        if self.config.rope_axes_dim is None:
            self.config.rope_axes_dim = (
                self.config.rope_scaling.get("mrope_section", [24, 20, 20])
                if self.config.rope_scaling is not None
                else [24, 20, 20]
            )
        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList([Cosmos3ReasonerLayer(self.config) for _ in range(self.config.num_hidden_layers)])
        if self.config.hidden_act == "relu2":
            self.norm = Cosmos3NemotronRMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        else:
            self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps, elementwise_affine=True, bias=False)
        self.rotary_emb = Cosmos3VLTextRotaryEmbedding(
            head_dim=self.config.head_dim,
            rope_theta=self.config.rope_theta,
            rope_axes_dim=tuple(self.config.rope_axes_dim),
        )

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        revision: str | None = None,
    ) -> "Cosmos3Reasoner":
        config_path = cls._resolve_component_file(repo_id, "config.json", revision=revision)
        with open(config_path, "r", encoding="utf-8") as handle:
            config = Cosmos3ReasonerConfig.from_dict(json.load(handle))
        if config.component != "cosmos3_reasoner":
            raise ValueError(f"Expected Cosmos3 reasoner component, got {config.component!r}.")
        reasoner = cls(config)
        logger.info("Loading Cosmos3 reasoner weights from %s", repo_id)
        state_dict = cls._load_component_state_dict(repo_id, revision=revision)
        logger.info("Loaded Cosmos3 reasoner state dict with %s tensors from %s", len(state_dict), repo_id)
        reasoner.load_state_dict(state_dict, strict=True)
        reasoner.to(device=device, dtype=dtype)
        reasoner.requires_grad_(False)
        reasoner.eval()
        return reasoner

    @classmethod
    def _load_component_state_dict(cls, repo_id: str, revision: str | None = None) -> dict[str, torch.Tensor]:
        repo_path = Path(repo_id)
        if repo_path.exists():
            index_path_obj = repo_path / "model.safetensors.index.json"
            if not index_path_obj.is_file():
                return load_file(cls._resolve_component_file(repo_id, "model.safetensors", revision=revision))
            index_path = str(index_path_obj)
        else:
            try:
                index_path = cls._resolve_component_file(repo_id, "model.safetensors.index.json", revision=revision)
            except EntryNotFoundError:
                return load_file(cls._resolve_component_file(repo_id, "model.safetensors", revision=revision))

        with open(index_path, "r", encoding="utf-8") as handle:
            index = json.load(handle)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ValueError(f"Index file {index_path} does not contain a weight_map object.")

        state_dict = {}
        shard_names = sorted(set(weight_map.values()))
        logger.info("Loading Cosmos3 reasoner state dict from %s shard(s)", len(shard_names))
        for shard_name in shard_names:
            logger.info("Loading Cosmos3 reasoner shard %s", shard_name)
            state_dict.update(load_file(cls._resolve_component_file(repo_id, shard_name, revision=revision)))
        return state_dict

    @staticmethod
    def _resolve_component_file(repo_id: str, filename: str, revision: str | None = None) -> str:
        local_path = Path(repo_id) / filename
        if local_path.is_file():
            return str(local_path)
        return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> Cosmos3ReasonerMemoryState:
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        position_ids = position_ids.to(device=self.device)
        und_seq = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(
            position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1),
            device=und_seq.device,
            dtype=und_seq.dtype,
        )
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)

        layer_kv = []
        rotary_emb = (cos, sin)
        for layer in self.layers:
            und_seq, reasoner_kv = layer(und_seq, rotary_emb)
            layer_kv.append({key: value.detach().cpu() for key, value in reasoner_kv.items()})
        return Cosmos3ReasonerMemoryState(layer_kv=layer_kv)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return next(self.buffers()).device
