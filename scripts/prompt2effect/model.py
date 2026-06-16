from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from scripts.prompt2effect.lora_utils import normalized_frobenius_loss


@dataclass
class Prompt2EffectConfig:
    rank: int
    hidden_dim: int
    text_hidden_dim: int
    compressed_tokens: int
    num_heads: int
    num_layers: int
    dropout: float
    layer_count: int
    module_types: list[str]
    layer_shapes: list[tuple[int, int]]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layer_shapes"] = [list(shape) for shape in self.layer_shapes]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Prompt2EffectConfig":
        data = dict(payload)
        data["layer_shapes"] = [tuple(shape) for shape in data["layer_shapes"]]
        return cls(**data)


class Prompt2EffectBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_1 = nn.LayerNorm(hidden_dim)
        self.norm_2 = nn.LayerNorm(hidden_dim)
        self.norm_3 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, text_hidden: torch.Tensor, text_key_padding_mask=None) -> torch.Tensor:
        residual = tokens
        x = self.norm_1(tokens)
        x, _ = self.self_attn(x, x, x, need_weights=False)
        tokens = residual + self.dropout(x)

        residual = tokens
        x = self.norm_2(tokens)
        x, _ = self.cross_attn(
            x,
            text_hidden,
            text_hidden,
            key_padding_mask=text_key_padding_mask,
            need_weights=False,
        )
        tokens = residual + self.dropout(x)

        residual = tokens
        x = self.ff(self.norm_3(tokens))
        return residual + self.dropout(x)


class Prompt2EffectHyperNetwork(nn.Module):
    def __init__(self, config: Prompt2EffectConfig):
        super().__init__()
        if config.rank <= 0:
            raise ValueError("Prompt2Effect rank must be greater than zero.")
        if config.hidden_dim % config.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.config = config
        self.text_projection = nn.Linear(config.text_hidden_dim, config.hidden_dim)
        self.layer_embedding = nn.Embedding(config.layer_count, config.hidden_dim)
        self.module_type_to_idx = {module_type: idx for idx, module_type in enumerate(config.module_types)}
        self.module_type_embedding = nn.Embedding(len(config.module_types), config.hidden_dim)
        self.compress_queries = nn.Parameter(torch.randn(config.compressed_tokens, config.hidden_dim) * 0.02)
        self.compress_attn = nn.MultiheadAttention(config.hidden_dim, config.num_heads, batch_first=True)
        self.decode_attn = nn.MultiheadAttention(config.hidden_dim, config.num_heads, batch_first=True)
        self.blocks = nn.ModuleList(
            [Prompt2EffectBlock(config.hidden_dim, config.num_heads, config.dropout) for _ in range(config.num_layers)]
        )
        self.projections = nn.ModuleDict()
        for out_dim, in_dim in sorted(set(config.layer_shapes)):
            self.projections[self._shape_key(out_dim, in_dim)] = nn.ModuleDict(
                {
                    "row": nn.Linear(in_dim, config.hidden_dim),
                    "col": nn.Linear(out_dim, config.hidden_dim),
                }
            )
        self.head_b = nn.Linear(config.hidden_dim, config.rank)
        self.head_a = nn.Linear(config.hidden_dim, config.rank)

    @staticmethod
    def _shape_key(out_dim: int, in_dim: int) -> str:
        return f"out{int(out_dim)}_in{int(in_dim)}"

    def _weight_tokens(self, weight: torch.Tensor) -> torch.Tensor:
        out_dim, in_dim = weight.shape
        projections = self.projections[self._shape_key(out_dim, in_dim)]
        weight = weight.to(device=self.compress_queries.device, dtype=self.compress_queries.dtype)
        row_tokens = projections["row"](weight)
        col_tokens = projections["col"](weight.transpose(0, 1))
        return torch.cat([row_tokens, col_tokens], dim=0)

    def forward(
        self,
        text_hidden: torch.Tensor,
        base_weights: list[torch.Tensor],
        *,
        module_types_for_layer: list[str],
        text_attention_mask: torch.Tensor | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        if len(base_weights) != self.config.layer_count:
            raise ValueError(f"Expected {self.config.layer_count} base weights, got {len(base_weights)}.")
        if len(module_types_for_layer) != self.config.layer_count:
            raise ValueError(f"Expected {self.config.layer_count} module type entries, got {len(module_types_for_layer)}.")

        batch_size = text_hidden.shape[0]
        text_hidden = self.text_projection(text_hidden.to(dtype=self.compress_queries.dtype))
        text_key_padding_mask = None
        if text_attention_mask is not None:
            text_key_padding_mask = ~text_attention_mask.to(device=text_hidden.device, dtype=torch.bool)

        compressed = []
        weight_tokens: list[torch.Tensor] = []
        for layer_idx, weight in enumerate(base_weights):
            tokens = self._weight_tokens(weight)
            weight_tokens.append(tokens)
            queries = self.compress_queries.unsqueeze(0)
            layer_tokens, _ = self.compress_attn(queries, tokens.unsqueeze(0), tokens.unsqueeze(0), need_weights=False)
            layer_tokens = layer_tokens.squeeze(0)
            layer_tokens = layer_tokens + self.layer_embedding.weight[layer_idx].unsqueeze(0)
            module_type_idx = self.module_type_to_idx[module_types_for_layer[layer_idx]]
            layer_tokens = layer_tokens + self.module_type_embedding.weight[module_type_idx].unsqueeze(0)
            compressed.append(layer_tokens)

        tokens = torch.stack(compressed, dim=0).reshape(1, self.config.layer_count * self.config.compressed_tokens, -1)
        tokens = tokens.expand(batch_size, -1, -1).contiguous()
        for block in self.blocks:
            tokens = block(tokens, text_hidden, text_key_padding_mask=text_key_padding_mask)
        layer_latents = tokens.reshape(batch_size, self.config.layer_count, self.config.compressed_tokens, -1)

        predictions: list[dict[str, torch.Tensor]] = []
        for layer_idx, weight in enumerate(base_weights):
            out_dim, in_dim = weight.shape
            queries = weight_tokens[layer_idx].unsqueeze(0).expand(batch_size, -1, -1).contiguous()
            decoded, _ = self.decode_attn(
                queries,
                layer_latents[:, layer_idx],
                layer_latents[:, layer_idx],
                need_weights=False,
            )
            row_tokens = decoded[:, :out_dim]
            col_tokens = decoded[:, out_dim:]
            b_hat = self.head_b(row_tokens)
            a_hat = self.head_a(col_tokens).transpose(1, 2).contiguous()
            predictions.append({"A": a_hat, "B": b_hat})
        return predictions


def prompt2effect_loss(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    if len(predictions) != len(targets):
        raise ValueError(f"Prediction/target layer count mismatch: {len(predictions)} vs {len(targets)}.")
    losses = []
    for prediction, target in zip(predictions, targets):
        losses.append(normalized_frobenius_loss(prediction["A"], target["A"].to(prediction["A"].device), eps=eps))
        losses.append(normalized_frobenius_loss(prediction["B"], target["B"].to(prediction["B"].device), eps=eps))
    return torch.stack(losses).mean()
