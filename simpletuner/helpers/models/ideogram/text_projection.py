from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from simpletuner.helpers.models.ideogram.constants import QWEN3_VL_ACTIVATION_LAYERS
from simpletuner.helpers.models.ideogram.quantized_loading import (
    is_bnb4bit_state_dict,
    is_fp8_state_dict,
    load_bnb4bit_state_dict,
    load_fp8_state_dict,
    swap_linears_to_bnb4bit,
    swap_linears_to_fp8,
)
from simpletuner.helpers.models.ideogram.transformer import Ideogram4RMSNorm

TEXT_PROJECTION_KEY_PREFIXES = ("llm_cond_norm.", "llm_cond_proj.")
DEFAULT_LLM_FEATURES_DIM = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)
DEFAULT_EMB_DIM = 4608
DEFAULT_NORM_EPS = 1e-6


@dataclass
class Ideogram4TextProjectionConfig:
    llm_features_dim: int = DEFAULT_LLM_FEATURES_DIM
    emb_dim: int = DEFAULT_EMB_DIM
    norm_eps: float = DEFAULT_NORM_EPS
    activation_layers: tuple[int, ...] = QWEN3_VL_ACTIVATION_LAYERS
    source_repo: str | None = None
    source_revision: str | None = None
    quantization: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "Ideogram4TextProjectionConfig":
        activation_layers = data.get("activation_layers", QWEN3_VL_ACTIVATION_LAYERS)
        return cls(
            llm_features_dim=int(data.get("llm_features_dim", DEFAULT_LLM_FEATURES_DIM)),
            emb_dim=int(data.get("emb_dim", DEFAULT_EMB_DIM)),
            norm_eps=float(data.get("norm_eps", DEFAULT_NORM_EPS)),
            activation_layers=tuple(int(layer) for layer in activation_layers),
            source_repo=data.get("source_repo"),
            source_revision=data.get("source_revision"),
            quantization=data.get("quantization"),
        )

    def to_dict(self) -> dict:
        return {
            "llm_features_dim": self.llm_features_dim,
            "emb_dim": self.emb_dim,
            "norm_eps": self.norm_eps,
            "activation_layers": list(self.activation_layers),
            "source_repo": self.source_repo,
            "source_revision": self.source_revision,
            "quantization": self.quantization,
        }


class Ideogram4TextProjection(nn.Module):
    """Ideogram text-conditioning projection used to shrink cached text embeds."""

    def __init__(self, config: Ideogram4TextProjectionConfig | None = None) -> None:
        super().__init__()
        self.config = config or Ideogram4TextProjectionConfig()
        self.llm_cond_norm = Ideogram4RMSNorm(self.config.llm_features_dim, eps=self.config.norm_eps)
        self.llm_cond_proj = nn.Linear(self.config.llm_features_dim, self.config.emb_dim, bias=True)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        revision: str | None = None,
    ) -> "Ideogram4TextProjection":
        config_path = cls._resolve_component_file(repo_id, "config.json", revision=revision)
        with open(config_path, "r", encoding="utf-8") as handle:
            config = Ideogram4TextProjectionConfig.from_dict(json.load(handle))

        model = cls(config)
        state_dict = load_file(cls._resolve_component_file(repo_id, "model.safetensors", revision=revision))
        model.load_component_state_dict(state_dict, device=device, dtype=dtype)
        model.eval()
        return model

    @staticmethod
    def _resolve_component_file(repo_id: str, filename: str, revision: str | None = None) -> str:
        local_path = Path(repo_id) / filename
        if local_path.is_file():
            return str(local_path)
        return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)

    def load_component_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if is_fp8_state_dict(state_dict):
            swap_linears_to_fp8(self, state_dict, compute_dtype=dtype)
            load_fp8_state_dict(self, state_dict, device=device, dtype=dtype)
        elif is_bnb4bit_state_dict(state_dict):
            swap_linears_to_bnb4bit(self, compute_dtype=dtype)
            load_bnb4bit_state_dict(self, state_dict, device=device, dtype=dtype)
        else:
            self.load_state_dict(state_dict)
            self.to(device=device, dtype=dtype)

    def forward(self, prompt_embeds: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        param_dtype = getattr(self.llm_cond_proj, "compute_dtype", None) or self.llm_cond_proj.weight.dtype
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=param_dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=prompt_embeds.device, dtype=torch.bool)
            prompt_embeds = prompt_embeds * attention_mask.to(prompt_embeds.dtype).unsqueeze(-1)
        projected = self.llm_cond_proj(self.llm_cond_norm(prompt_embeds))
        if attention_mask is not None:
            projected = projected * attention_mask.to(projected.dtype).unsqueeze(-1)
        return projected

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return next(self.buffers()).device
