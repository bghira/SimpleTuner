from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _config_value(config, name: str):
    if isinstance(config, dict):
        return config.get(name)
    try:
        values = vars(config)
    except TypeError:
        values = None
    if values is not None:
        return values.get(name)
    return getattr(config, name, None)


def nextlat_enabled_from_config(config) -> bool:
    return bool(_config_value(config, "nextlat_enabled") or False)


def _resolve_module_path(model: nn.Module, path: str):
    current = model
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def infer_nextlat_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    if config is not None:
        heads = _config_value(config, "num_attention_heads")
        head_dim = _config_value(config, "attention_head_dim")
        if heads is not None and head_dim is not None:
            return int(heads * head_dim)
        for attribute in ("hidden_size", "d_model", "model_dim", "dim", "inner_dim", "embed_dim", "emb_dim", "n_embd"):
            value = _config_value(config, attribute)
            if value is not None:
                return int(value)

    for module in model.modules():
        if isinstance(module, nn.LayerNorm) and module.normalized_shape:
            return int(module.normalized_shape[0])
    raise ValueError("NextLat could not infer the transformer's hidden size.")


def infer_nextlat_block_count(model: nn.Module) -> int:
    candidates = (
        ("transformer_blocks",),
        ("transformer_blocks", "single_transformer_blocks"),
        ("joint_transformer_blocks", "single_transformer_blocks"),
        ("double_stream_layers", "single_stream_layers"),
        ("visual_transformer_blocks",),
        ("model.layers",),
        ("decoder.layers",),
        ("transformer.h",),
        ("layers",),
        ("blocks",),
        ("h",),
    )
    for names in candidates:
        modules = [_resolve_module_path(model, name) for name in names]
        if modules and all(isinstance(module, (nn.ModuleList, nn.Sequential)) for module in modules):
            count = sum(len(module) for module in modules)
            if count:
                return count
    raise ValueError("NextLat could not determine the transformer's block count.")


class NextLatPredictor(nn.Module):
    def __init__(self, hidden_size: int, block_index: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.up = nn.Linear(hidden_size, hidden_size * 2)
        self.down = nn.Linear(hidden_size * 2, hidden_size)
        self.register_buffer("block_index", torch.tensor(int(block_index), dtype=torch.int64), persistent=True)
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim < 3:
            raise ValueError("NextLat expects hidden states with batch, token, and feature dimensions.")
        tokens = hidden_states.reshape(hidden_states.shape[0], -1, hidden_states.shape[-1])
        parameter_dtype = self.norm.weight.dtype
        if tokens.dtype != parameter_dtype:
            tokens = tokens.to(dtype=parameter_dtype)
        return self.down(F.gelu(self.up(self.norm(tokens))))


class NextLatRegularizer:
    MODULE_NAME = "nextlat_predictor"

    def __init__(self, config, accelerator, hidden_size: int, block_count: int):
        self.config = config
        self.device = accelerator.device
        self.enabled = bool(_config_value(config, "nextlat_enabled") or False)
        self.weight = float(_config_value(config, "nextlat_weight") or 0.0)
        if self.enabled and self.weight <= 0:
            raise ValueError("nextlat_weight must be greater than zero when NextLat is enabled.")

        configured_block = int(_config_value(config, "nextlat_block_index") or -1)
        self.block_index = block_count - 1 if configured_block < 0 else configured_block
        if not 0 <= self.block_index < block_count:
            raise ValueError(f"nextlat_block_index must be within [-1, {block_count - 1}], got {configured_block}.")

        self.state_loss = str(_config_value(config, "nextlat_state_loss") or "smooth_l1")
        if self.state_loss not in ("smooth_l1", "mse"):
            raise ValueError("nextlat_state_loss must be 'smooth_l1' or 'mse'.")
        self.kl_weight = float(_config_value(config, "nextlat_kl_weight") or 0.0)
        if self.kl_weight < 0:
            raise ValueError("nextlat_kl_weight must be non-negative.")

        self.predictor = NextLatPredictor(hidden_size, self.block_index)
        self.model: Optional[nn.Module] = None

    def attach_to_model(self, model: nn.Module, dtype: torch.dtype) -> None:
        if hasattr(model, self.MODULE_NAME):
            self.predictor = getattr(model, self.MODULE_NAME)
        else:
            setattr(model, self.MODULE_NAME, self.predictor)
        self.model = model
        self.predictor.to(device=self.device, dtype=dtype)

    def wants_hidden_states(self) -> bool:
        return self.enabled

    def _attached_predictor(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError("NextLat predictor is not attached to a transformer.")
        return getattr(self.model, self.MODULE_NAME)

    def compute_loss(self, hidden_states_buffer, model_output: dict) -> tuple[torch.Tensor, dict]:
        if hidden_states_buffer is None:
            raise ValueError("NextLat is enabled but the model did not return a hidden-state buffer.")
        hidden_states = hidden_states_buffer.get(f"layer_{self.block_index}")
        if hidden_states is None:
            raise ValueError(f"NextLat requested layer {self.block_index}, but that layer was not captured.")
        hidden_states = hidden_states.reshape(hidden_states.shape[0], -1, hidden_states.shape[-1])
        if hidden_states.shape[1] < 2:
            raise ValueError("NextLat requires at least two hidden tokens to predict the next latent state.")

        predictor = self._attached_predictor()
        prediction = predictor(hidden_states[:, :-1])
        target = hidden_states[:, 1:].detach()
        if self.state_loss == "smooth_l1":
            state_loss = F.smooth_l1_loss(prediction.float(), target.float(), reduction="mean")
        else:
            state_loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")

        total_loss = state_loss * self.weight
        logs = {
            "nextlat_loss": total_loss.detach().item(),
            "nextlat_state_loss": state_loss.detach().item(),
        }

        if self.kl_weight > 0:
            kl_loss = self._compute_optional_kl(prediction, target, model_output)
            total_loss = total_loss + kl_loss * self.kl_weight
            logs["nextlat_kl_loss"] = kl_loss.detach().item()

        return total_loss, logs

    @staticmethod
    def _compute_optional_kl(
        prediction: torch.Tensor,
        target: torch.Tensor,
        model_output: dict,
    ) -> torch.Tensor:
        head = model_output.get("nextlat_logits_head")
        if head is None:
            raise ValueError("nextlat_kl_weight requires model_output['nextlat_logits_head'].")
        pred_logits = head(prediction)
        with torch.no_grad():
            target_logits = head(target)
            target_probs = target_logits.float().softmax(dim=-1)
        pred_log_probs = pred_logits.float().log_softmax(dim=-1)
        return F.kl_div(pred_log_probs, target_probs, reduction="batchmean")
