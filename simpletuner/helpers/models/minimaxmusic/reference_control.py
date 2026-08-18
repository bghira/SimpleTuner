# Copyright 2026 SimpleTuner contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import dataclasses
import math
import weakref
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass(frozen=True)
class ReferenceControlConfig:
    hidden_size: int = 4096
    control_dim: int = 512
    num_heads: int = 8
    layer_indices: tuple[int, ...] = (5, 11, 17, 23, 29, 35)
    window_frames: int = 100
    gate_init: float = 1e-3

    def __post_init__(self) -> None:
        if self.control_dim % self.num_heads:
            raise ValueError("control_dim must be divisible by num_heads")
        if not self.layer_indices:
            raise ValueError("layer_indices must not be empty")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("layer_indices must not contain duplicates")
        if self.window_frames < 0:
            raise ValueError("window_frames must be non-negative")
        if self.gate_init < 0:
            raise ValueError("gate_init must be non-negative")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ReferenceControlConfig":
        normalized = dict(values)
        if "layer_indices" in normalized:
            normalized["layer_indices"] = tuple(normalized["layer_indices"])
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(normalized) - known)
        if unknown:
            raise ValueError(f"Unknown reference-control configuration fields: {unknown}")
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        values = dataclasses.asdict(self)
        values["layer_indices"] = list(self.layer_indices)
        return values


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtype = hidden_states.dtype
        values = hidden_states.float()
        values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + self.eps)
        return values.to(dtype) * self.weight.to(dtype)


class ReferenceMemoryEncoder(nn.Module):
    def __init__(self, hidden_size: int, control_dim: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, control_dim, bias=False)
        self.temporal = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        control_dim,
                        control_dim,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=control_dim,
                    ),
                    nn.SiLU(),
                    nn.Conv1d(control_dim, control_dim, kernel_size=1),
                )
                for dilation in (1, 3, 9)
            ]
        )

    def forward(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        if frame_embeddings.ndim != 3:
            raise ValueError("frame_embeddings must have shape [batch, frames, hidden_size]")
        hidden_states = self.proj(self.norm(frame_embeddings)).transpose(1, 2)
        for block in self.temporal:
            hidden_states = hidden_states + block(hidden_states)
        return hidden_states.transpose(1, 2)


def aligned_reference_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    window_frames: int,
) -> torch.Tensor:
    if query_positions.ndim != 2 or key_positions.ndim != 2:
        raise ValueError("query_positions and key_positions must have shape [batch, frames]")
    if query_positions.shape[0] != key_positions.shape[0]:
        raise ValueError("query_positions and key_positions must have the same batch size")
    mask = (query_positions.unsqueeze(-1) - key_positions.unsqueeze(-2)).abs() <= window_frames
    if not mask.any(dim=-1).all():
        raise ValueError("every query position must have at least one reference key in its attention window")
    return mask


class AlignedReferenceCrossAttention(nn.Module):
    def __init__(self, config: ReferenceControlConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.control_dim = config.control_dim
        self.num_heads = config.num_heads
        self.head_dim = config.control_dim // config.num_heads
        self.window_frames = config.window_frames
        self.norm = RMSNorm(config.hidden_size)
        self.to_q = nn.Linear(config.hidden_size, config.control_dim, bias=False)
        self.to_k = nn.Linear(config.control_dim, config.control_dim, bias=False)
        self.to_v = nn.Linear(config.control_dim, config.control_dim, bias=False)
        self.to_out = nn.Linear(config.control_dim, config.hidden_size, bias=False)
        self.gate = nn.Parameter(torch.tensor(config.gate_init, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        reference_memory: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape
        key_length = reference_memory.shape[1]
        if reference_memory.shape[0] != batch_size:
            raise ValueError("hidden_states and reference_memory must have the same batch size")
        if query_positions.shape != (batch_size, query_length):
            raise ValueError("query_positions do not match hidden_states")
        if key_positions.shape != (batch_size, key_length):
            raise ValueError("key_positions do not match reference_memory")

        query = self.to_q(self.norm(hidden_states))
        key = self.to_k(reference_memory)
        value = self.to_v(reference_memory)
        query = query.view(batch_size, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, key_length, self.num_heads, self.head_dim).transpose(1, 2)
        mask = aligned_reference_mask(query_positions, key_positions, self.window_frames).unsqueeze(1)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        attended = attended.transpose(1, 2).reshape(batch_size, query_length, self.control_dim)
        return hidden_states + self.gate.to(hidden_states.dtype) * self.to_out(attended)


class ReferenceConditionedDecoderLayer(nn.Module):
    def __init__(self, base_layer: nn.Module, control: AlignedReferenceCrossAttention):
        super().__init__()
        self.base_layer = base_layer
        object.__setattr__(self, "_control", weakref.proxy(control))

    @property
    def control(self) -> AlignedReferenceCrossAttention:
        return object.__getattribute__(self, "_control")

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        reference_memory = kwargs.pop("reference_memory", None)
        query_positions = kwargs.pop("reference_query_positions", None)
        key_positions = kwargs.pop("reference_key_positions", None)
        query_start = kwargs.pop("reference_query_start", None)
        hidden_states = self.base_layer(hidden_states, *args, **kwargs)
        if reference_memory is None:
            return hidden_states
        if query_positions is None or key_positions is None or query_start is None:
            raise ValueError("reference conditioning requires query positions, key positions, and query_start")
        query_length = query_positions.shape[1]
        query_end = query_start + query_length
        if query_start < 0 or query_end > hidden_states.shape[1]:
            raise ValueError("reference-conditioned query range is outside the decoder sequence")
        conditioned = self.control(
            hidden_states[:, query_start:query_end],
            reference_memory,
            query_positions,
            key_positions,
        )
        return torch.cat((hidden_states[:, :query_start], conditioned, hidden_states[:, query_end:]), dim=1)


class MiniMaxMusic3ReferenceControlAdapter(nn.Module):
    def __init__(self, config: ReferenceControlConfig):
        super().__init__()
        self.config = config
        self.memory_encoder = ReferenceMemoryEncoder(config.hidden_size, config.control_dim)
        self.controls = nn.ModuleList([AlignedReferenceCrossAttention(config) for _ in config.layer_indices])
        self._installed_layers: list[tuple[int, nn.Module]] = []

    @property
    def installed(self) -> bool:
        return bool(self._installed_layers)

    def encode_reference(self, frame_embeddings: torch.Tensor) -> torch.Tensor:
        return self.memory_encoder(frame_embeddings)

    def install(self, language_model) -> None:
        if self.installed:
            raise RuntimeError("reference-control adapter is already installed")
        layers = language_model.model.layers
        if self.config.hidden_size != language_model.config.hidden_size:
            raise ValueError(
                f"adapter hidden_size {self.config.hidden_size} does not match LM hidden_size "
                f"{language_model.config.hidden_size}"
            )
        if min(self.config.layer_indices) < 0 or max(self.config.layer_indices) >= len(layers):
            raise ValueError(f"layer_indices must be within the LM's {len(layers)} decoder layers")
        for layer_index, control in zip(self.config.layer_indices, self.controls, strict=True):
            base_layer = layers[layer_index]
            if isinstance(base_layer, ReferenceConditionedDecoderLayer):
                raise RuntimeError(f"decoder layer {layer_index} is already reference-conditioned")
            layers[layer_index] = ReferenceConditionedDecoderLayer(base_layer, control)
            self._installed_layers.append((layer_index, base_layer))

    def uninstall(self, language_model) -> None:
        if not self.installed:
            return
        for layer_index, base_layer in self._installed_layers:
            current = language_model.model.layers[layer_index]
            if not isinstance(current, ReferenceConditionedDecoderLayer):
                raise RuntimeError(f"decoder layer {layer_index} was replaced after adapter installation")
            language_model.model.layers[layer_index] = base_layer
        self._installed_layers.clear()


@dataclasses.dataclass(frozen=True)
class ControlLoRAConfig:
    hidden_size: int = 4096
    residual_rank: int = 16
    layer_indices: tuple[int, ...] = tuple(range(36))

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.residual_rank < 1:
            raise ValueError("residual_rank must be positive")
        if not self.layer_indices:
            raise ValueError("layer_indices must not be empty")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("layer_indices must not contain duplicates")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ControlLoRAConfig":
        normalized = dict(values)
        if "layer_indices" in normalized:
            normalized["layer_indices"] = tuple(normalized["layer_indices"])
        known = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(normalized) - known)
        if unknown:
            raise ValueError(f"Unknown ControlLoRA configuration fields: {unknown}")
        return cls(**normalized)

    def to_dict(self) -> dict[str, Any]:
        values = dataclasses.asdict(self)
        values["layer_indices"] = list(self.layer_indices)
        return values


class ControlResidualProjector(nn.Module):
    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.multiplier = 1.0
        nn.init.zeros_(self.up.weight)

    def set_multiplier(self, value: float) -> None:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("ControlLoRA multiplier must be finite and non-negative")
        self.multiplier = float(value)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(F.silu(self.down(self.norm(hidden_states)))) * self.multiplier


class ControlLoRADecoderLayer(nn.Module):
    def __init__(self, base_layer: nn.Module, residual: ControlResidualProjector, layer_index: int):
        super().__init__()
        self.base_layer = base_layer
        self.layer_index = layer_index
        object.__setattr__(self, "_residual", weakref.proxy(residual))

    @property
    def residual(self) -> ControlResidualProjector:
        return object.__getattribute__(self, "_residual")

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        control_hidden_states = kwargs.pop("control_hidden_states", None)
        query_start = kwargs.pop("control_query_start", None)
        control_scale = kwargs.pop("control_scale", 1.0)
        hidden_states = self.base_layer(hidden_states, *args, **kwargs)
        if control_hidden_states is None:
            return hidden_states
        if query_start is None:
            raise ValueError("ControlLoRA residual injection requires control_query_start")
        if not 0 <= query_start < hidden_states.shape[1]:
            raise ValueError("control_query_start is outside the decoder sequence")
        if not math.isfinite(control_scale) or control_scale < 0.0:
            raise ValueError("control_scale must be finite and non-negative")
        control = control_hidden_states[self.layer_index]
        target_hidden_states = hidden_states[:, query_start:]
        if control.shape == hidden_states.shape:
            control = control[:, query_start:]
        elif control.shape != target_hidden_states.shape:
            raise ValueError(
                f"control hidden shape {tuple(control.shape)} does not match layer output "
                f"{tuple(hidden_states.shape)} or controlled suffix {tuple(target_hidden_states.shape)}"
            )
        controlled = target_hidden_states + control_scale * self.residual(control)
        return torch.cat((hidden_states[:, :query_start], controlled), dim=1)


class MiniMaxMusic3ControlLoRAAdapter(nn.Module):
    def __init__(self, config: ControlLoRAConfig):
        super().__init__()
        self.config = config
        self.residuals = nn.ModuleList(
            [ControlResidualProjector(config.hidden_size, config.residual_rank) for _ in config.layer_indices]
        )
        self._installed_layers: list[tuple[int, nn.Module]] = []

    @property
    def installed(self) -> bool:
        return bool(self._installed_layers)

    def set_multiplier(self, value: float) -> None:
        for residual in self.residuals:
            residual.set_multiplier(value)

    def install(self, language_model) -> None:
        if self.installed:
            raise RuntimeError("ControlLoRA adapter is already installed")
        layers = language_model.model.layers
        if self.config.hidden_size != language_model.config.hidden_size:
            raise ValueError(
                f"adapter hidden_size {self.config.hidden_size} does not match LM hidden_size "
                f"{language_model.config.hidden_size}"
            )
        if min(self.config.layer_indices) < 0 or max(self.config.layer_indices) >= len(layers):
            raise ValueError(f"layer_indices must be within the LM's {len(layers)} decoder layers")
        for layer_index, residual in zip(self.config.layer_indices, self.residuals, strict=True):
            base_layer = layers[layer_index]
            if isinstance(base_layer, (ControlLoRADecoderLayer, ReferenceConditionedDecoderLayer)):
                raise RuntimeError(f"decoder layer {layer_index} is already wrapped")
            layers[layer_index] = ControlLoRADecoderLayer(base_layer, residual, layer_index)
            self._installed_layers.append((layer_index, base_layer))

    def uninstall(self, language_model) -> None:
        if not self.installed:
            return
        for layer_index, base_layer in self._installed_layers:
            current = language_model.model.layers[layer_index]
            if not isinstance(current, ControlLoRADecoderLayer):
                raise RuntimeError(f"decoder layer {layer_index} was replaced after adapter installation")
            language_model.model.layers[layer_index] = base_layer
        self._installed_layers.clear()


def embed_rvq_frames(language_model, rvq_depth_decoder, codes: torch.Tensor) -> torch.Tensor:
    if codes.ndim != 3 or codes.shape[-1] != rvq_depth_decoder.config.num_codebooks:
        raise ValueError("codes must have shape [batch, frames, num_codebooks]")
    semantic = language_model.model.embed_tokens(codes[..., :1] + 151_675).squeeze(-2)
    offsets = torch.arange(codes.shape[-1] - 1, device=codes.device) * rvq_depth_decoder.config.audio_vocab_size
    acoustic = rvq_depth_decoder.audio_embeddings(codes[..., 1:] + offsets).sum(dim=-2)
    return (semantic + acoustic.to(semantic.dtype)) * (codes.shape[-1] ** -0.5)


def create_qwen_oftv2_adapter(
    language_model,
    *,
    block_size: int = 64,
    constraint: float = 0.0,
    rescaled: bool = False,
):
    """Attach LyCORIS Diag-OFT modules to Qwen self-attention query/output projections."""
    if any(isinstance(layer, ReferenceConditionedDecoderLayer) for layer in language_model.model.layers):
        raise RuntimeError("create the OFTv2 adapter before installing reference-control decoder wrappers")
    try:
        from lycoris import LycorisNetwork, create_lycoris
    except ImportError as exc:
        raise ImportError("LyCORIS OFTv2 requires `pip install lycoris-lora>=3.4.0`") from exc

    LycorisNetwork.apply_preset(
        {
            "target_module": [],
            "target_name": [
                "model.layers.*.self_attn.q_proj",
                "model.layers.*.self_attn.o_proj",
            ],
            "use_fnmatch": True,
        }
    )
    network = create_lycoris(
        language_model,
        multiplier=1.0,
        linear_dim=block_size,
        linear_alpha=1.0,
        algo="diag-oft",
        constraint=constraint,
        rescaled=rescaled,
    )
    expected_modules = len(language_model.model.layers) * 2
    if len(network.loras) != expected_modules:
        raise RuntimeError(f"expected {expected_modules} Qwen OFTv2 modules, created {len(network.loras)}")
    network.apply_to()
    return network


_QWEN_ATTENTION_FFN_TARGETS = [
    "model.layers.*.self_attn.q_proj",
    "model.layers.*.self_attn.k_proj",
    "model.layers.*.self_attn.v_proj",
    "model.layers.*.self_attn.o_proj",
    "model.layers.*.mlp.gate_proj",
    "model.layers.*.mlp.up_proj",
    "model.layers.*.mlp.down_proj",
]


def _apply_qwen_attention_ffn_preset(LycorisNetwork) -> None:
    LycorisNetwork.apply_preset(
        {
            "target_module": [],
            "target_name": _QWEN_ATTENTION_FFN_TARGETS,
            "use_fnmatch": True,
        }
    )


def create_qwen_lokr_adapter(
    language_model,
    *,
    rank: int = 16,
    alpha: float = 16.0,
):
    """Attach LyCORIS LoKr modules to every Qwen attention and gated-MLP projection."""
    if any(isinstance(layer, ReferenceConditionedDecoderLayer) for layer in language_model.model.layers):
        raise RuntimeError("create the LoKr adapter before installing reference-control decoder wrappers")
    try:
        from lycoris import LycorisNetwork, create_lycoris
    except ImportError as exc:
        raise ImportError("LyCORIS LoKr requires `pip install lycoris-lora>=3.4.0`") from exc

    _apply_qwen_attention_ffn_preset(LycorisNetwork)
    network = create_lycoris(
        language_model,
        multiplier=1.0,
        linear_dim=rank,
        linear_alpha=alpha,
        algo="lokr",
    )
    expected_modules = len(language_model.model.layers) * 7
    if len(network.loras) != expected_modules:
        raise RuntimeError(f"expected {expected_modules} Qwen LoKr modules, created {len(network.loras)}")
    network.apply_to()
    return network


def prefix_adapter_checkpoint_filename(adapter_type: str) -> str:
    if adapter_type == "oftv2":
        return "qwen_prefix_oftv2.safetensors"
    if adapter_type == "lokr":
        return "qwen_prefix_lokr.safetensors"
    raise ValueError(f"Unsupported prefix adapter type: {adapter_type}")
