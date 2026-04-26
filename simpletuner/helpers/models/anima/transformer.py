# Vendored from diffusers-anima: /src/diffusers-anima/src/diffusers_anima/models/transformers/modeling_anima_transformer.py
# Adapted for SimpleTuner local imports.

from __future__ import annotations

import json
import numbers
import os
import re
from typing import Any, Optional

import torch
import torch.nn.functional as F
from diffusers import CosmosTransformer3DModel, ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm
from diffusers.utils import USE_PEFT_BACKEND, set_weights_and_activate_adapters
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn

DEFAULT_ANIMA_TRANSFORMER_FILENAME = "anima-preview.safetensors"
DIFFUSERS_LLM_ADAPTER_FILENAME = "llm_adapter/diffusion_pytorch_model.safetensors"
DIFFUSERS_LLM_ADAPTER_CONFIG_FILENAME = "llm_adapter/config.json"


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    feature_dim = x.shape[-1]
    if feature_dim % 2 != 0:
        raise ValueError(f"RoPE rotate_half expects even feature dim, got {feature_dim}.")
    half = feature_dim // 2
    paired = x.reshape(*x.shape[:-1], 2, half)
    first, second = paired.unbind(dim=-2)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos.unsqueeze(1)) + (_rotate_half(x) * sin.unsqueeze(1))


def _expand_attention_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    casted = mask.to(torch.bool)
    if casted.ndim == 2:
        return casted[:, None, None, :]
    return casted


def _build_position_ids(batch_size: int, length: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(length, device=device, dtype=torch.long)
    return base.unsqueeze(0).expand(batch_size, -1)


def _pad_to_length(hidden_states: torch.Tensor, target_length: int) -> torch.Tensor:
    pad_tokens = target_length - hidden_states.shape[1]
    if pad_tokens <= 0:
        return hidden_states
    return F.pad(hidden_states, (0, 0, 0, pad_tokens))


def _default_padding_mask(hidden_states: torch.Tensor) -> torch.Tensor:
    return torch.zeros(
        (1, 1, hidden_states.shape[-2], hidden_states.shape[-1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )


class _AnimaRMSNorm(nn.Module):
    """RMSNorm implementation used by the Anima adapter blocks."""

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-6,
        *,
        elementwise_affine: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (int(normalized_shape),)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.register_parameter("weight", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_diffusers(cls, module: DiffusersRMSNorm) -> "_AnimaRMSNorm":
        patched = cls(
            tuple(module.dim),
            eps=float(module.eps),
            elementwise_affine=module.weight is not None,
            bias=module.bias is not None,
        )
        with torch.no_grad():
            if module.weight is not None:
                patched.weight.copy_(module.weight)
            if module.bias is not None and patched.bias is not None:
                patched.bias.copy_(module.bias)
        return patched

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            out = F.rms_norm(x, self.normalized_shape, eps=self.eps)
        else:
            out = F.rms_norm(
                x,
                self.normalized_shape,
                weight=self.weight.to(dtype=x.dtype, device=x.device),
                eps=self.eps,
            )
        if self.bias is not None:
            out = out + self.bias.to(dtype=out.dtype, device=out.device)
        return out


def _patch_diffusers_rmsnorm_to_anima(module: nn.Module) -> None:
    """Recursively replace Diffusers RMSNorm modules with Anima RMSNorm."""
    for child_name, child in list(module.named_children()):
        if isinstance(child, DiffusersRMSNorm):
            setattr(module, child_name, _AnimaRMSNorm.from_diffusers(child))
            continue
        _patch_diffusers_rmsnorm_to_anima(child)


def _store_hidden_state(buffer, key: str, hidden_states: torch.Tensor) -> None:
    if buffer is None:
        return
    capture_layers = getattr(buffer, "capture_layers", None)
    if capture_layers is not None:
        try:
            layer_idx = int(key.rsplit("_", maxsplit=1)[-1])
        except ValueError:
            layer_idx = None
        if layer_idx is not None and layer_idx not in capture_layers:
            return
    buffer[key] = hidden_states


class _RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even, got {head_dim}.")
        half_dim = head_dim // 2
        index = torch.arange(half_dim, dtype=torch.float32)
        exponent = (2.0 / float(head_dim)) * index
        inv = torch.reciprocal(torch.pow(torch.tensor(theta, dtype=torch.float32), exponent))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = positions.to(device=x.device, dtype=torch.float32)
        inv = self.inv_freq.to(device=x.device, dtype=torch.float32)
        freqs = torch.einsum("bl,d->bld", pos, inv)
        emb = freqs.repeat(1, 1, 2)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


class _AdapterAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, heads: int):
        super().__init__()
        if query_dim % heads != 0:
            raise ValueError(f"Adapter attention query_dim must be divisible by heads, got {query_dim} and {heads}.")
        inner = query_dim
        head_dim = inner // heads
        self.heads = heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.q_norm = _AnimaRMSNorm(head_dim)
        self.k_norm = _AnimaRMSNorm(head_dim)
        self.o_proj = nn.Linear(inner, query_dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        pos_q: tuple[torch.Tensor, torch.Tensor] | None = None,
        pos_k: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        context = x if context is None else context

        q = self.q_proj(x).view(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(context).view(context.shape[0], context.shape[1], self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(context).view(context.shape[0], context.shape[1], self.heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if pos_q is not None and pos_k is not None:
            q = _apply_rope(q, *pos_q)
            k = _apply_rope(k, *pos_k)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1).contiguous()
        return self.o_proj(y)


class _AdapterBlock(nn.Module):
    def __init__(self, model_dim: int = 1024, context_dim: int = 1024, heads: int = 16):
        super().__init__()
        self.norm_self_attn = _AnimaRMSNorm(model_dim)
        self.self_attn = _AdapterAttention(model_dim, model_dim, heads)
        self.norm_cross_attn = _AnimaRMSNorm(model_dim)
        self.cross_attn = _AdapterAttention(model_dim, context_dim, heads)
        self.norm_mlp = _AnimaRMSNorm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4, bias=True),
            nn.GELU(),
            nn.Linear(model_dim * 4, model_dim, bias=True),
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        context: torch.Tensor,
        target_mask: torch.Tensor | None,
        source_mask: torch.Tensor | None,
        pos_target: tuple[torch.Tensor, torch.Tensor],
        pos_source: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.self_attn(
            self.norm_self_attn(x),
            attn_mask=target_mask,
            pos_q=pos_target,
            pos_k=pos_target,
        )
        x = x + self.cross_attn(
            self.norm_cross_attn(x),
            context=context,
            attn_mask=source_mask,
            pos_q=pos_target,
            pos_k=pos_source,
        )
        x = x + self.mlp(self.norm_mlp(x))
        return x


class _LLMAdapter(nn.Module):
    def __init__(self, vocab_size: int = 32128, dim: int = 1024, layers: int = 6, heads: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([_AdapterBlock(model_dim=dim, context_dim=dim, heads=heads) for _ in range(layers)])
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.norm = _AnimaRMSNorm(dim)
        self.rope = _RotaryEmbedding(dim // heads)

    def forward(
        self,
        source_hidden_states: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor | None = None,
        source_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_target_mask = _expand_attention_mask(target_attention_mask)
        normalized_source_mask = _expand_attention_mask(source_attention_mask)

        target_hidden_states = self.embed(target_input_ids)
        source_context = source_hidden_states

        target_position_ids = _build_position_ids(
            batch_size=target_hidden_states.shape[0],
            length=target_hidden_states.shape[1],
            device=target_hidden_states.device,
        )
        source_position_ids = _build_position_ids(
            batch_size=source_context.shape[0],
            length=source_context.shape[1],
            device=source_context.device,
        )

        target_position_embed = self.rope(target_hidden_states, target_position_ids)
        source_position_embed = self.rope(source_context, source_position_ids)

        hidden_states = target_hidden_states
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                context=source_context,
                target_mask=normalized_target_mask,
                source_mask=normalized_source_mask,
                pos_target=target_position_embed,
                pos_source=source_position_embed,
            )
        return self.norm(self.out_proj(hidden_states))


class AnimaTransformerModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["CosmosTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        # CosmosTransformer3DModel core parameters
        in_channels: int = 16,
        out_channels: int = 16,
        num_attention_heads: int = 16,
        attention_head_dim: int = 128,
        num_layers: int = 28,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 1024,
        adaln_lora_dim: int = 256,
        max_size: tuple[int, int, int] | list[int] = (128, 240, 240),
        patch_size: tuple[int, int, int] | list[int] = (1, 2, 2),
        rope_scale: tuple[float, float, float] | list[float] = (1.0, 4.0, 4.0),
        # LLMAdapter parameters
        adapter_vocab_size: int = 32128,
        adapter_dim: int = 1024,
        adapter_layers: int = 6,
        adapter_heads: int = 16,
    ):
        super().__init__()
        core = _create_anima_transformer_core_model(
            in_channels=in_channels,
            out_channels=out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            text_embed_dim=text_embed_dim,
            adaln_lora_dim=adaln_lora_dim,
            max_size=tuple(max_size),
            patch_size=tuple(patch_size),
            rope_scale=tuple(rope_scale),
        )
        _patch_diffusers_rmsnorm_to_anima(core)
        self.core = core
        self.llm_adapter = _LLMAdapter(
            vocab_size=adapter_vocab_size,
            dim=adapter_dim,
            layers=adapter_layers,
            heads=adapter_heads,
        )

    def preprocess_text_embeds(
        self,
        text_embeds: torch.Tensor,
        text_ids: torch.Tensor | None,
        t5xxl_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if text_ids is None:
            return text_embeds

        adapted = self.llm_adapter(text_embeds, text_ids)
        if t5xxl_weights is not None:
            adapted = adapted * t5xxl_weights
        return _pad_to_length(adapted, 512)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Transformer2DModelOutput | tuple[torch.Tensor]:
        t5xxl_ids = kwargs.pop("t5xxl_ids", None)
        t5xxl_weights = kwargs.pop("t5xxl_weights", None)
        hidden_states_buffer = kwargs.pop("hidden_states_buffer", None)
        if t5xxl_ids is not None:
            encoder_hidden_states = self.preprocess_text_embeds(
                encoder_hidden_states, t5xxl_ids, t5xxl_weights=t5xxl_weights
            )

        padding_mask = kwargs.pop("padding_mask", None)
        if padding_mask is None:
            # CosmosTransformer3DModel internally repeats this per batch, so keep batch=1 here.
            padding_mask = _default_padding_mask(hidden_states)

        hook_handles = []
        if hidden_states_buffer is not None:
            for block_idx, block in enumerate(getattr(self.core, "transformer_blocks", [])):
                hook_handles.append(
                    block.register_forward_hook(
                        lambda _module, _inputs, output, idx=block_idx: _store_hidden_state(
                            hidden_states_buffer,
                            f"layer_{idx}",
                            output[0] if isinstance(output, tuple) else output,
                        )
                    )
                )

        try:
            sample = self.core(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                padding_mask=padding_mask,
                return_dict=False,
            )[0]
        finally:
            for handle in hook_handles:
                handle.remove()

        if not return_dict:
            return (sample,)
        return Transformer2DModelOutput(sample=sample)

    def set_adapters(
        self,
        adapter_names: list[str] | str,
        weights: float | dict[str, float] | list[float | dict[str, float] | None] | None = None,
    ) -> None:
        """Set active LoRA adapters without relying on Diffusers private model-name mappings."""
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `set_adapters()`.")

        normalized_names = [adapter_names] if isinstance(adapter_names, str) else list(adapter_names)
        if not isinstance(weights, list):
            normalized_weights = [weights] * len(normalized_names)
        else:
            normalized_weights = list(weights)

        if len(normalized_names) != len(normalized_weights):
            raise ValueError(
                f"Length of adapter names {len(normalized_names)} is not equal to the length of their weights "
                f"{len(normalized_weights)}."
            )

        resolved_weights = [weight if weight is not None else 1.0 for weight in normalized_weights]
        set_weights_and_activate_adapters(self, normalized_names, resolved_weights)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *args: Any,
        **kwargs: Any,
    ) -> "AnimaTransformerModel":
        subfolder = kwargs.get("subfolder")
        resolved_dir = None
        if isinstance(pretrained_model_name_or_path, str) and os.path.isdir(pretrained_model_name_or_path):
            resolved_dir = (
                os.path.join(pretrained_model_name_or_path, subfolder) if subfolder else pretrained_model_name_or_path
            )
        if resolved_dir and os.path.isfile(os.path.join(resolved_dir, "config.json")):
            core = CosmosTransformer3DModel.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
            return cls._from_diffusers_components(core, pretrained_model_name_or_path, **kwargs)

        if subfolder == "transformer":
            try:
                core = CosmosTransformer3DModel.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
            except (OSError, ValueError):
                pass
            else:
                return cls._from_diffusers_components(core, pretrained_model_name_or_path, **kwargs)

        return cls.from_single_file(
            pretrained_model_name_or_path,
            filename=kwargs.pop("filename", DEFAULT_ANIMA_TRANSFORMER_FILENAME),
            subfolder=kwargs.pop("subfolder", None),
            revision=kwargs.pop("revision", None),
            torch_dtype=kwargs.pop("torch_dtype", None),
        )

    @staticmethod
    def _diffusers_repo_root(pretrained_model_name_or_path: str, subfolder: Optional[str] = None) -> str:
        if not os.path.isdir(pretrained_model_name_or_path):
            return pretrained_model_name_or_path
        if subfolder:
            return pretrained_model_name_or_path
        normalized_path = os.path.abspath(pretrained_model_name_or_path)
        if os.path.basename(normalized_path) == "transformer" and os.path.isfile(
            os.path.join(normalized_path, "config.json")
        ):
            return os.path.dirname(normalized_path)
        return pretrained_model_name_or_path

    @staticmethod
    def _resolve_diffusers_llm_adapter_path(
        pretrained_model_name_or_path: str,
        *,
        subfolder: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> str:
        if os.path.isdir(pretrained_model_name_or_path):
            repo_root = AnimaTransformerModel._diffusers_repo_root(pretrained_model_name_or_path, subfolder=subfolder)
            return os.path.join(repo_root, DIFFUSERS_LLM_ADAPTER_FILENAME)
        normalized_token = None if token is False else token
        return hf_hub_download(
            pretrained_model_name_or_path,
            filename=DIFFUSERS_LLM_ADAPTER_FILENAME,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=normalized_token,
        )

    @staticmethod
    def _resolve_diffusers_llm_adapter_config_path(
        pretrained_model_name_or_path: str,
        *,
        subfolder: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
    ) -> str:
        if os.path.isdir(pretrained_model_name_or_path):
            repo_root = AnimaTransformerModel._diffusers_repo_root(pretrained_model_name_or_path, subfolder=subfolder)
            return os.path.join(repo_root, DIFFUSERS_LLM_ADAPTER_CONFIG_FILENAME)
        normalized_token = None if token is False else token
        return hf_hub_download(
            pretrained_model_name_or_path,
            filename=DIFFUSERS_LLM_ADAPTER_CONFIG_FILENAME,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=normalized_token,
        )

    @classmethod
    def _load_diffusers_llm_adapter_config(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        config_path = cls._resolve_diffusers_llm_adapter_config_path(
            pretrained_model_name_or_path,
            subfolder=kwargs.get("subfolder"),
            revision=kwargs.get("revision"),
            cache_dir=kwargs.get("cache_dir"),
            force_download=bool(kwargs.get("force_download", False)),
            local_files_only=bool(kwargs.get("local_files_only", False)),
            token=kwargs.get("token"),
        )
        with open(config_path, encoding="utf-8") as handle:
            adapter_config = json.load(handle)

        model_dim = int(adapter_config["model_dim"])
        if (
            int(adapter_config.get("source_dim", model_dim)) != model_dim
            or int(adapter_config.get("target_dim", model_dim)) != model_dim
        ):
            raise ValueError("Anima llm_adapter source_dim, target_dim, and model_dim must match.")
        return adapter_config

    @classmethod
    def _from_diffusers_components(
        cls,
        core: CosmosTransformer3DModel,
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> "AnimaTransformerModel":
        adapter_config = cls._load_diffusers_llm_adapter_config(pretrained_model_name_or_path, **kwargs)
        config = core.config
        transformer = cls(
            in_channels=int(config.in_channels),
            out_channels=int(config.out_channels),
            num_attention_heads=int(config.num_attention_heads),
            attention_head_dim=int(config.attention_head_dim),
            num_layers=int(config.num_layers),
            mlp_ratio=float(config.mlp_ratio),
            text_embed_dim=int(config.text_embed_dim),
            adaln_lora_dim=int(config.adaln_lora_dim),
            max_size=tuple(config.max_size),
            patch_size=tuple(config.patch_size),
            rope_scale=tuple(config.rope_scale),
            adapter_vocab_size=int(adapter_config["vocab_size"]),
            adapter_dim=int(adapter_config["model_dim"]),
            adapter_layers=int(adapter_config["num_layers"]),
            adapter_heads=int(adapter_config["num_heads"]),
        )
        _patch_diffusers_rmsnorm_to_anima(core)
        transformer.core = core
        cls._load_diffusers_llm_adapter(transformer, pretrained_model_name_or_path, **kwargs)
        torch_dtype = kwargs.get("torch_dtype")
        if isinstance(torch_dtype, torch.dtype):
            transformer.to(dtype=torch_dtype)
        return transformer

    @classmethod
    def _load_diffusers_llm_adapter(
        cls,
        transformer: "AnimaTransformerModel",
        pretrained_model_name_or_path: str,
        **kwargs: Any,
    ) -> None:
        adapter_path = cls._resolve_diffusers_llm_adapter_path(
            pretrained_model_name_or_path,
            subfolder=kwargs.get("subfolder"),
            revision=kwargs.get("revision"),
            cache_dir=kwargs.get("cache_dir"),
            force_download=bool(kwargs.get("force_download", False)),
            local_files_only=bool(kwargs.get("local_files_only", False)),
            token=kwargs.get("token"),
        )
        state_dict = load_file(adapter_path, device="cpu")
        if all(key.startswith("llm_adapter.") for key in state_dict):
            state_dict = {key.removeprefix("llm_adapter."): value for key, value in state_dict.items()}
        missing, unexpected = transformer.llm_adapter.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                "Anima Diffusers-format llm_adapter weights do not match expected architecture. "
                f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
            )

    @classmethod
    def from_single_file(
        cls,
        pretrained_model_link_or_path: str,
        *args: Any,
        filename: str = DEFAULT_ANIMA_TRANSFORMER_FILENAME,
        subfolder: Optional[str] = None,
        revision: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs: Any,
    ) -> "AnimaTransformerModel":
        del args, kwargs
        file_path = _resolve_transformer_weight_path(
            pretrained_model_link_or_path,
            filename=filename,
            subfolder=subfolder,
            revision=revision,
        )
        state_dict = load_file(file_path, device="cpu")
        state_dict = _strip_net_prefix(state_dict)
        core_state_dict, llm_adapter_state_dict = _convert_anima_state_dict_to_diffusers(state_dict)

        transformer = cls()
        missing, unexpected = transformer.load_state_dict(
            {**core_state_dict, **llm_adapter_state_dict},
            strict=False,
        )
        if missing or unexpected:
            raise RuntimeError(
                "Anima checkpoint does not match transformer architecture. "
                f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
            )
        if torch_dtype is not None:
            transformer.to(dtype=torch_dtype)
        return transformer


def _create_anima_transformer_core_model(
    in_channels: int = 16,
    out_channels: int = 16,
    num_attention_heads: int = 16,
    attention_head_dim: int = 128,
    num_layers: int = 28,
    mlp_ratio: float = 4.0,
    text_embed_dim: int = 1024,
    adaln_lora_dim: int = 256,
    max_size: tuple[int, int, int] = (128, 240, 240),
    patch_size: tuple[int, int, int] = (1, 2, 2),
    rope_scale: tuple[float, float, float] = (1.0, 4.0, 4.0),
) -> CosmosTransformer3DModel:
    return CosmosTransformer3DModel(
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
        concat_padding_mask=True,
        extra_pos_embed_type=None,
    )


def _resolve_transformer_weight_path(
    pretrained_model_name_or_path: str,
    *,
    filename: str,
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    if pretrained_model_name_or_path is None:
        raise ValueError("pretrained_model_name_or_path is required")
    if os.path.isfile(pretrained_model_name_or_path):
        return pretrained_model_name_or_path
    if os.path.isdir(pretrained_model_name_or_path):
        base = os.path.join(pretrained_model_name_or_path, subfolder) if subfolder else pretrained_model_name_or_path
        return os.path.join(base, filename)
    relative = os.path.join(subfolder, filename) if subfolder else filename
    return hf_hub_download(pretrained_model_name_or_path, filename=relative, revision=revision)


def _strip_net_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if all(key.startswith("net.") for key in state_dict):
        return {key[4:]: value for key, value in state_dict.items()}
    return dict(state_dict)


def _convert_anima_state_dict_to_diffusers(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    core: dict[str, torch.Tensor] = {}
    adapter: dict[str, torch.Tensor] = {}

    root_map = {
        "x_embedder.proj.1.weight": "core.patch_embed.proj.weight",
        "t_embedder.1.linear_1.weight": "core.time_embed.t_embedder.linear_1.weight",
        "t_embedder.1.linear_2.weight": "core.time_embed.t_embedder.linear_2.weight",
        "t_embedding_norm.weight": "core.time_embed.norm.weight",
        "final_layer.adaln_modulation.1.weight": "core.norm_out.linear_1.weight",
        "final_layer.adaln_modulation.2.weight": "core.norm_out.linear_2.weight",
        "final_layer.linear.weight": "core.proj_out.weight",
    }

    block_maps = {
        "adaln_modulation_self_attn.1.weight": "norm1.linear_1.weight",
        "adaln_modulation_self_attn.2.weight": "norm1.linear_2.weight",
        "adaln_modulation_cross_attn.1.weight": "norm2.linear_1.weight",
        "adaln_modulation_cross_attn.2.weight": "norm2.linear_2.weight",
        "adaln_modulation_mlp.1.weight": "norm3.linear_1.weight",
        "adaln_modulation_mlp.2.weight": "norm3.linear_2.weight",
        "self_attn.q_norm.weight": "attn1.norm_q.weight",
        "self_attn.k_norm.weight": "attn1.norm_k.weight",
        "self_attn.q_proj.weight": "attn1.to_q.weight",
        "self_attn.k_proj.weight": "attn1.to_k.weight",
        "self_attn.v_proj.weight": "attn1.to_v.weight",
        "self_attn.output_proj.weight": "attn1.to_out.0.weight",
        "cross_attn.q_norm.weight": "attn2.norm_q.weight",
        "cross_attn.k_norm.weight": "attn2.norm_k.weight",
        "cross_attn.q_proj.weight": "attn2.to_q.weight",
        "cross_attn.k_proj.weight": "attn2.to_k.weight",
        "cross_attn.v_proj.weight": "attn2.to_v.weight",
        "cross_attn.output_proj.weight": "attn2.to_out.0.weight",
        "mlp.layer1.weight": "ff.net.0.proj.weight",
        "mlp.layer2.weight": "ff.net.2.weight",
    }

    block_re = re.compile(r"^blocks\.(\d+)\.(.+)$")
    for key, value in state_dict.items():
        if key.startswith("llm_adapter."):
            adapter[".".join(["llm_adapter", key.removeprefix("llm_adapter.")])] = value
            continue

        mapped = root_map.get(key)
        if mapped is not None:
            core[mapped] = value
            continue

        m = block_re.match(key)
        if m is not None:
            block_index = m.group(1)
            tail = m.group(2)
            mapped_tail = block_maps.get(tail)
            if mapped_tail is None:
                raise RuntimeError(f"Unsupported Anima checkpoint key in blocks: {key}")
            core[f"core.transformer_blocks.{block_index}.{mapped_tail}"] = value
            continue

        raise RuntimeError(f"Unsupported Anima checkpoint key: {key}")

    return core, adapter


AnimaDiT = AnimaTransformerModel
