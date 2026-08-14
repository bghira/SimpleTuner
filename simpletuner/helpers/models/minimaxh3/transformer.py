# Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
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

import json
import math
import os
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models._modeling_parallel import ContextParallelInput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry, dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, apply_lora_scale, logging
from huggingface_hub import hf_hub_download
from safetensors import SafetensorError, safe_open

from simpletuner.helpers.models.flowmap import (
    blend_flowmap_embeddings,
    clone_flowmap_embedder,
    flowmap_timestep_embedding,
    prepare_flowmap_delta_timestep,
    register_flowmap_config,
    register_flowmap_gate_buffer,
    set_flowmap_gate,
    validate_flowmap_deltatime_type,
)
from simpletuner.helpers.musubi_block_swap import MusubiBlockSwapManager
from simpletuner.helpers.training.context_parallel_tensors import context_parallel_config, prepare_cp_attention_mask
from simpletuner.helpers.training.gradient_checkpointing_interval import checkpoint_sequential_state, should_checkpoint_block
from simpletuner.helpers.training.offloaded_gradient_checkpointer import activation_offload_context
from simpletuner.helpers.training.tread import TREADRouter

from .activations import MiniMaxH3FeedForward
from .sparse_attention import (
    MiniMaxH3SparseAttentionConfig,
    MiniMaxH3SparseAttentionLayout,
    initialize_minimax_h3_flex_attention,
    minimax_h3_sparse_attention,
    minimax_h3_sparse_attention_ulysses,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# MiniMax-H3 tags every row of the packed sequence with the modality it belongs to and keeps one set of AdaLN
# modulation parameters per (timestep, modality) pair: 0 = video, 1 = text, 2 = audio.
MINIMAX_H3_MODALITY_NUM = 3


class _MiniMaxH3AllGather(torch.autograd.Function):
    """Gather sequence shards without PyTorch's unsupported NCCL coalesced path."""

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, dim: int, group):
        ctx.dim = dim
        ctx.group = group
        ctx.world_size = torch.distributed.get_world_size(group)
        ctx.rank = torch.distributed.get_rank(group)
        shards = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
        torch.distributed.all_gather(shards, tensor.contiguous(), group=group)
        return torch.cat(shards, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        local_gradient = torch.chunk(grad_output, ctx.world_size, dim=ctx.dim)[ctx.rank]
        return local_gradient.contiguous(), None, None


def _gather_h3_context_parallel_output(tensor: torch.Tensor, context_config: Any, dim: int = 1) -> torch.Tensor:
    if context_config is None:
        return tensor
    mesh = getattr(context_config, "_flattened_mesh", None)
    if mesh is None:
        raise RuntimeError("MiniMax-H3 context parallel output gathering requires an initialized flattened CP mesh.")
    group = mesh.get_group()
    if torch.distributed.get_world_size(group) <= 1:
        return tensor
    return _MiniMaxH3AllGather.apply(tensor, dim, group)


def _pad_h3_context_parallel_layout(
    position_ids: torch.Tensor,
    token_tags: torch.Tensor,
    timestep_indices: torch.Tensor,
    degree: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad packed layout rows so Diffusers can shard them evenly across the CP mesh."""
    if degree < 1:
        raise ValueError(f"MiniMax-H3 context parallel degree must be positive, got {degree}.")
    padding = (-position_ids.shape[0]) % degree
    if padding == 0:
        return position_ids, token_tags, timestep_indices

    position_ids = torch.cat([position_ids, position_ids.new_zeros((padding, position_ids.shape[1]))], dim=0)
    token_tags = torch.cat([token_tags, token_tags.new_full((padding,), -1)], dim=0)
    timestep_indices = torch.cat([timestep_indices, timestep_indices.new_zeros((padding,))], dim=0)
    return position_ids, token_tags, timestep_indices


def _interpolate_adaln_curve(table: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
    table = table.to(device=timestep.device)
    position = timestep.to(device=timestep.device, dtype=torch.float32).clamp(0.0, 1.0) * (table.shape[0] - 1)
    lower = position.floor().long().clamp(max=table.shape[0] - 2)
    weight = (position - lower).to(dtype=table.dtype).unsqueeze(-1)
    return torch.lerp(table[lower], table[lower + 1], weight)


class MiniMaxH3AdaLNCurveEmbedder(nn.Module):
    """Independent trainable copy of H3's sampled AdaLN timestep curve."""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.weight = nn.Parameter(table.detach().clone())

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return _interpolate_adaln_curve(self.weight, timestep)


class H3_REFERENCE_MODE(str, Enum):
    Vanilla = "vanilla"
    CachedKV = "cached_kv"


def resolve_h3_reference_mode(
    value: str | H3_REFERENCE_MODE | None,
) -> H3_REFERENCE_MODE:
    if value is None:
        return H3_REFERENCE_MODE.Vanilla
    if isinstance(value, H3_REFERENCE_MODE):
        return value
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "vanilla": H3_REFERENCE_MODE.Vanilla,
        "cached_kv": H3_REFERENCE_MODE.CachedKV,
        "cachedkv": H3_REFERENCE_MODE.CachedKV,
        "cached": H3_REFERENCE_MODE.CachedKV,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        choices = ", ".join(mode.value for mode in H3_REFERENCE_MODE)
        raise ValueError(f"MiniMax-H3 reference mode must be one of: {choices}; got {value!r}.") from exc


def _resolve_minimax_h3_single_file_path(
    pretrained_model_link_or_path: str,
    *,
    filename: str | None = None,
    subfolder: str | None = None,
    revision: str | None = None,
) -> str:
    if pretrained_model_link_or_path is None:
        raise ValueError("pretrained_model_link_or_path is required")
    if os.path.isfile(pretrained_model_link_or_path):
        return pretrained_model_link_or_path
    if os.path.isdir(pretrained_model_link_or_path):
        if filename is None:
            raise ValueError("filename is required when loading a MiniMax-H3 single-file checkpoint from a directory")
        base = os.path.join(pretrained_model_link_or_path, subfolder) if subfolder else pretrained_model_link_or_path
        return os.path.join(base, filename)

    parsed = urlparse(pretrained_model_link_or_path)
    if parsed.scheme in {"http", "https"}:
        if parsed.netloc != "huggingface.co":
            raise ValueError("MiniMax-H3 single-file loading only supports local files or Hugging Face Hub URLs")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 5 or parts[2] not in {"resolve", "blob"}:
            raise ValueError(f"Unsupported Hugging Face single-file URL: {pretrained_model_link_or_path}")
        repo_id = "/".join(parts[:2])
        url_revision = parts[3]
        url_filename = "/".join(parts[4:])
        return hf_hub_download(repo_id=repo_id, filename=url_filename, revision=revision or url_revision)

    if filename is None:
        raise ValueError("filename is required when loading a MiniMax-H3 single-file checkpoint from a Hub repo")
    relative = os.path.join(subfolder, filename) if subfolder else filename
    return hf_hub_download(repo_id=pretrained_model_link_or_path, filename=relative, revision=revision)


_SAFETENSORS_DTYPE_MAP = {
    "BOOL": torch.bool,
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E4M3FN": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
}

_COMFY_QUANT_METADATA_SUFFIXES = (
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".pre_quant_scale",
    ".comfy_quant",
)

_COMFY_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2}


class _MiniMaxH3TrailingSafetensorsReader:
    def __init__(self, path: str):
        self.path = path
        self.file_size = os.path.getsize(path)
        with open(path, "rb") as handle:
            raw_header_len = handle.read(8)
            if len(raw_header_len) != 8:
                raise RuntimeError(f"MiniMax-H3 safetensors checkpoint {path} is too small to contain a header.")
            self.header_len = struct.unpack("<Q", raw_header_len)[0]
            header_bytes = handle.read(self.header_len)
            if len(header_bytes) != self.header_len:
                raise RuntimeError(f"MiniMax-H3 safetensors checkpoint {path} ended before its header was complete.")
        header = json.loads(header_bytes)
        self.entries = {key: value for key, value in header.items() if key != "__metadata__"}
        cursor = 0
        for key, value in sorted(self.entries.items(), key=lambda item: item[1]["data_offsets"][0]):
            start, end = value["data_offsets"]
            if start != cursor:
                raise RuntimeError(
                    f"MiniMax-H3 safetensors checkpoint {path} has non-contiguous tensor data at {key}: "
                    f"expected offset {cursor}, got {start}."
                )
            if end < start:
                raise RuntimeError(
                    f"MiniMax-H3 safetensors checkpoint {path} has invalid offsets for {key}: {value['data_offsets']}."
                )
            cursor = end
        self.covered_size = 8 + self.header_len + cursor
        self.trailing_bytes = self.file_size - self.covered_size
        if self.trailing_bytes <= 0:
            raise RuntimeError(f"MiniMax-H3 safetensors checkpoint {path} has no trailing bytes to trim.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def keys(self):
        return self.entries.keys()

    def get_tensor(self, key: str) -> torch.Tensor:
        value = self.entries[key]
        dtype = _SAFETENSORS_DTYPE_MAP.get(value["dtype"])
        if dtype is None:
            raise RuntimeError(f"MiniMax-H3 safetensors checkpoint {self.path} uses unsupported dtype {value['dtype']}.")
        shape = tuple(value["shape"])
        start, end = value["data_offsets"]
        byte_count = end - start
        with open(self.path, "rb") as handle:
            handle.seek(8 + self.header_len + start)
            buffer = bytearray(handle.read(byte_count))
        if len(buffer) != byte_count:
            raise RuntimeError(f"MiniMax-H3 safetensors checkpoint {self.path} ended while reading tensor {key}.")
        tensor = torch.frombuffer(buffer, dtype=dtype)
        expected_numel = 1
        for dim in shape:
            expected_numel *= dim
        if tensor.numel() != expected_numel:
            raise RuntimeError(
                f"MiniMax-H3 safetensors tensor {key} has {tensor.numel()} elements, expected {expected_numel}."
            )
        return tensor.reshape(shape)


def _open_minimax_h3_single_file(path: str):
    try:
        return safe_open(path, framework="pt", device="cpu")
    except SafetensorError as exc:
        if "file not fully covered" not in str(exc):
            raise
        reader = _MiniMaxH3TrailingSafetensorsReader(path)
        logger.warning(
            "MiniMax-H3 safetensors checkpoint %s has %s trailing bytes beyond declared tensor data; "
            "using the H3 compatibility reader.",
            path,
            reader.trailing_bytes,
        )
        return reader


def _strip_minimax_h3_checkpoint_prefix(key: str) -> str:
    for prefix in (
        "transformer.",
        "transformer_ref.",
        "model.diffusion_model.",
        "diffusion_model.",
    ):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _map_minimax_h3_comfy_key_to_diffusers(key: str) -> list[str]:
    if key.endswith(_COMFY_QUANT_METADATA_SUFFIXES):
        return []
    direct_map = {
        "video_patch_proj.": "proj_in.",
        "audio_patch_proj.": "audio_proj_in.",
        "condition_proj.": "context_embedder.",
        "time_embedder.proj_in.": "time_embedder.linear_1.",
        "time_embedder.proj_out.": "time_embedder.linear_2.",
        "final_layer.norm.": "norm_out.norm.",
        "final_layer.adaln_proj.": "norm_out.",
        "final_layer.video_out.": "proj_out.",
        "final_layer.audio_out.": "audio_proj_out.",
    }
    for source, target in direct_map.items():
        if key.startswith(source):
            return [key.replace(source, target, 1)]

    if key.startswith("token_refiner.blocks."):
        key = key.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.", 1)
    elif key.startswith("blocks."):
        key = key.replace("blocks.", "transformer_blocks.", 1)
    key = key.replace(".attn.q_norm.", ".attn.norm_q.")
    key = key.replace(".attn.k_norm.", ".attn.norm_k.")
    key = key.replace(".attn.out_proj.", ".attn.to_out.0.")
    key = key.replace(".mlp.fc1.", ".ff.net.0.proj.")
    key = key.replace(".mlp.fc2.", ".ff.net.2.")
    if key.endswith(".attn.qkv_proj.weight"):
        base = key.removesuffix(".attn.qkv_proj.weight")
        return [
            f"{base}.attn.to_q.weight",
            f"{base}.attn.to_k.weight",
            f"{base}.attn.to_v.weight",
        ]
    return [key]


def _convert_minimax_h3_native_swiglu_to_diffusers(key: str, tensor: torch.Tensor) -> torch.Tensor:
    """Convert native H3 ``[gate; value]`` FFN rows to Diffusers' ``[value; gate]`` order."""
    if not key.endswith(".mlp.fc1.weight"):
        return tensor
    if tensor.ndim == 0 or tensor.shape[0] % 2 != 0:
        raise RuntimeError(f"MiniMax-H3 SwiGLU tensor {key} cannot be split into gate/value rows")
    gate, value = tensor.chunk(2, dim=0)
    return torch.cat((value, gate), dim=0).contiguous()


def _convert_minimax_h3_native_swiglu_scale_to_diffusers(key: str, scale: torch.Tensor) -> torch.Tensor:
    """Apply the native H3 FFN row conversion to a per-output-row quantization scale."""
    if not key.endswith(".mlp.fc1.weight") or scale.ndim == 0 or scale.numel() == 1:
        return scale
    if scale.shape[0] % 2 != 0:
        raise RuntimeError(f"MiniMax-H3 SwiGLU scale for {key} cannot be split into gate/value rows")
    gate, value = scale.chunk(2, dim=0)
    return torch.cat((value, gate), dim=0).contiguous()


def _count_indexed_blocks(keys: set[str], prefix: str) -> int:
    indices = set()
    for key in keys:
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        index = rest.split(".", 1)[0]
        if index.isdigit():
            indices.add(int(index))
    return max(indices) + 1 if indices else 0


def _get_checkpoint_tensor(checkpoint, stripped_key: str) -> torch.Tensor:
    for raw_key in checkpoint.keys():
        if _strip_minimax_h3_checkpoint_prefix(raw_key) == stripped_key:
            return checkpoint.get_tensor(raw_key)
    raise RuntimeError(f"MiniMax-H3 checkpoint is missing required tensor: {stripped_key}")


def _infer_minimax_h3_config_from_checkpoint(checkpoint) -> dict[str, Any]:
    raw_keys = {_strip_minimax_h3_checkpoint_prefix(key) for key in checkpoint.keys()}
    if "video_patch_proj.weight" in raw_keys:
        video_patch_weight = _get_checkpoint_tensor(checkpoint, "video_patch_proj.weight")
        audio_patch_weight = _get_checkpoint_tensor(checkpoint, "audio_patch_proj.weight")
        condition_weight = _get_checkpoint_tensor(checkpoint, "condition_proj.weight")
        q_norm_weight = _get_checkpoint_tensor(checkpoint, "blocks.0.attn.q_norm.weight")
        qkv_weight = _get_checkpoint_tensor(checkpoint, "blocks.0.attn.qkv_proj.weight")
        ffn_weight = _get_checkpoint_tensor(checkpoint, "blocks.0.mlp.fc1.weight")
        has_adaln_curve = "adaln_t_table" in raw_keys
        adaln_curve_table = _get_checkpoint_tensor(checkpoint, "adaln_t_table") if has_adaln_curve else None
        has_time_weight = "time_embedder.proj_in.weight" in raw_keys
        has_time_out = "time_embedder.proj_out.weight" in raw_keys
        has_rope = "rope.inv_freq" in raw_keys
        return {
            "hidden_size": video_patch_weight.shape[0],
            "num_layers": _count_indexed_blocks(raw_keys, "blocks."),
            "num_refiner_layers": _count_indexed_blocks(raw_keys, "token_refiner.blocks."),
            "ffn_dim": ffn_weight.shape[0] // 2,
            "in_channels": video_patch_weight.shape[1] // 4,
            "audio_in_channels": audio_patch_weight.shape[1],
            "text_dim": condition_weight.shape[1],
            "attention_head_dim": q_norm_weight.shape[0],
            "num_attention_heads": qkv_weight.shape[0] // (3 * q_norm_weight.shape[0]),
            "freq_dim": (
                _get_checkpoint_tensor(checkpoint, "time_embedder.proj_in.weight").shape[1] if has_time_weight else 256
            ),
            "time_embed_hidden_dim": (
                _get_checkpoint_tensor(checkpoint, "time_embedder.proj_in.weight").shape[0] if has_time_weight else 5376
            ),
            "time_embed_dim": (
                adaln_curve_table.shape[1]
                if has_adaln_curve
                else _get_checkpoint_tensor(checkpoint, "time_embedder.proj_out.weight").shape[0] if has_time_out else 2688
            ),
            "rope_freq_dim": _get_checkpoint_tensor(checkpoint, "rope.inv_freq").shape[0] if has_rope else 16,
            "adaln_curve_grid": adaln_curve_table.shape[0] if has_adaln_curve else None,
            "swiglu_gate_first": False,
        }

    if "proj_in.weight" in raw_keys:
        proj_weight = _get_checkpoint_tensor(checkpoint, "proj_in.weight")
        audio_weight = _get_checkpoint_tensor(checkpoint, "audio_proj_in.weight")
        context_weight = _get_checkpoint_tensor(checkpoint, "context_embedder.weight")
        q_norm_weight = _get_checkpoint_tensor(checkpoint, "transformer_blocks.0.attn.norm_q.weight")
        q_weight = _get_checkpoint_tensor(checkpoint, "transformer_blocks.0.attn.to_q.weight")
        ffn_weight = _get_checkpoint_tensor(checkpoint, "transformer_blocks.0.ff.net.0.proj.weight")
        has_adaln_curve = "adaln_t_table" in raw_keys
        adaln_curve_table = _get_checkpoint_tensor(checkpoint, "adaln_t_table") if has_adaln_curve else None
        time_in = None if has_adaln_curve else _get_checkpoint_tensor(checkpoint, "time_embedder.linear_1.weight")
        time_out = None if has_adaln_curve else _get_checkpoint_tensor(checkpoint, "time_embedder.linear_2.weight")
        has_rope = "rope.inv_freq" in raw_keys
        return {
            "hidden_size": proj_weight.shape[0],
            "num_layers": _count_indexed_blocks(raw_keys, "transformer_blocks."),
            "num_refiner_layers": _count_indexed_blocks(raw_keys, "token_refiner.refiner_blocks."),
            "ffn_dim": ffn_weight.shape[0] // 2,
            "in_channels": proj_weight.shape[1] // 4,
            "audio_in_channels": audio_weight.shape[1],
            "text_dim": context_weight.shape[1],
            "attention_head_dim": q_norm_weight.shape[0],
            "num_attention_heads": q_weight.shape[0] // q_norm_weight.shape[0],
            "freq_dim": time_in.shape[1] if time_in is not None else 256,
            "time_embed_hidden_dim": time_in.shape[0] if time_in is not None else 5376,
            "time_embed_dim": adaln_curve_table.shape[1] if has_adaln_curve else time_out.shape[0],
            "rope_freq_dim": _get_checkpoint_tensor(checkpoint, "rope.inv_freq").shape[0] if has_rope else 16,
            "adaln_curve_grid": adaln_curve_table.shape[0] if has_adaln_curve else None,
        }

    raise RuntimeError("MiniMax-H3 single-file checkpoint does not contain recognized H3 transformer keys.")


def _set_module_buffer(root: nn.Module, buffer_name: str, value: torch.Tensor) -> None:
    module = root
    parts = buffer_name.split(".")
    for part in parts[:-1]:
        module = getattr(module, part)
    module._buffers[parts[-1]] = value


@dataclass
class MiniMaxH3TransformerOutput(BaseOutput):
    r"""
    The output of [`MiniMaxH3Transformer3DModel`].

    Args:
        sample (`torch.Tensor` of shape `(batch_size, num_video_tokens, in_channels * prod(patch_size))`):
            The video velocity prediction for the rows addressed by `video_indices`, in the same order. Conditioning
            rows are returned unmasked — masking them out before the scheduler step is the caller's job.
        audio_sample (`torch.Tensor` of shape `(batch_size, num_audio_tokens, audio_in_channels)`):
            The audio velocity prediction for the rows addressed by `audio_indices`, in the same order.
        crepa_hidden_states (`torch.Tensor`, *optional*):
            Captured generated-video hidden states for CREPA/LayerSync, shaped
            `(batch_size, post_patch_frames, post_patch_height * post_patch_width, hidden_dim)`.
    """

    sample: torch.Tensor | None = None
    audio_sample: torch.Tensor | None = None
    crepa_hidden_states: torch.Tensor | None = None


def _apply_rotary_emb(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    r"""
    Rotate the leading `rotary_dim` channels of every head and pass the remaining channels through unchanged.
    `hidden_states` is `(batch_size, seq_len, num_heads, head_dim)` and `cos`/`sin` are either
    `(seq_len, rotary_dim)` or `(batch_size, seq_len, rotary_dim)`.
    """
    rotary_dim = cos.shape[-1]
    hidden_states_rotary = hidden_states[..., :rotary_dim]
    hidden_states_pass = hidden_states[..., rotary_dim:]

    cos = cos.to(hidden_states.dtype)
    sin = sin.to(hidden_states.dtype)
    if cos.ndim == 2:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    elif cos.ndim == 3:
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
    else:
        raise ValueError(f"MiniMax-H3 rotary embeddings must be 2-D or 3-D, got {list(cos.shape)}.")
    x1, x2 = hidden_states_rotary.chunk(2, dim=-1)
    hidden_states_rotated = torch.cat((-x2, x1), dim=-1)
    hidden_states_rotary = hidden_states_rotary * cos + hidden_states_rotated * sin
    return torch.cat((hidden_states_rotary, hidden_states_pass), dim=-1).contiguous()


def _select_modulation(table: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.ndim == 1:
        return table.index_select(0, indices)
    if indices.ndim != 2:
        raise ValueError(f"MiniMax-H3 modulation indices must be 1-D or 2-D, got {list(indices.shape)}.")
    return table.index_select(0, indices.reshape(-1)).view(*indices.shape, table.shape[-1])


def _is_reduced_adaln_projection_key(key: str) -> bool:
    return key.startswith("norm_out.linear.") or (key.startswith("transformer_blocks.") and ".adaln_proj.linear." in key)


def _slice_sequence_rows(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if value.ndim == 2:
        return value.index_select(0, indices)
    if value.ndim == 3:
        return value.index_select(1, indices)
    raise ValueError(f"MiniMax-H3 sequence tensor must be 2-D or 3-D, got {list(value.shape)}.")


def _slice_rotary_emb(
    rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if rotary_emb is None:
        return None
    return tuple(_slice_sequence_rows(value, indices) for value in rotary_emb)


def _cache_tensor_signature(value: torch.Tensor) -> tuple[Any, ...]:
    detached = value.detach()
    summary = detached.float()
    flat = summary.reshape(-1)
    if flat.numel() == 0:
        first = last = 0.0
    else:
        first = float(flat[0].cpu())
        last = float(flat[-1].cpu())
    return (
        tuple(detached.shape),
        str(detached.device),
        str(detached.dtype),
        float(summary.sum().cpu()),
        float(summary.square().sum().cpu()),
        first,
        last,
    )


def _cache_rotary_signature(
    rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[Any, ...] | None:
    if rotary_emb is None:
        return None
    return tuple(_cache_tensor_signature(value) for value in rotary_emb)


class MiniMaxH3RotaryPosEmbed(nn.Module):
    r"""
    3-axis rotary embedding over the `(t, h, w)` coordinates of the packed sequence.

    A single `inv_freq` buffer of `rope_freq_dim` frequencies is shared by the three axes. Each axis contributes
    `rope_freq_dim` angles, the three blocks are concatenated to `3 * rope_freq_dim` and then concatenated with
    themselves so that the `rotate_half` convention rotates `2 * 3 * rope_freq_dim` of the `head_dim` channels.
    """

    def __init__(self, rope_freq_dim: int = 16, rope_theta: float = 10000.0):
        super().__init__()
        self.rope_freq_dim = rope_freq_dim
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, 2 * rope_freq_dim, 2, dtype=torch.float32) / (2 * rope_freq_dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (..., seq_len, 3) -> cos/sin: (..., seq_len, 2 * 3 * rope_freq_dim)
        position_ids = position_ids.to(torch.float32)
        inv_freq = self.inv_freq.to(device=position_ids.device)
        freqs = position_ids.unsqueeze(-1) * inv_freq.view(1, 1, -1)
        freqs_t, freqs_h, freqs_w = freqs.unbind(dim=-2)
        freqs = torch.cat((freqs_t, freqs_h, freqs_w), dim=-1)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()


class MiniMaxH3AdaLayerNormModulation(nn.Module):
    r"""
    Projects the shared timestep embedding into the six per-(timestep, modality) modulation parameters of one
    transformer block.

    `(num_timesteps, time_embed_dim)` -> six tensors of shape `(num_timesteps * MINIMAX_H3_MODALITY_NUM,
    hidden_size)`, in the diffusers `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp` order. The row
    layout of the returned tensors is `[t0_mod0, t0_mod1, t0_mod2, t1_mod0, ...]`, which is what `timestep_indices *
    MINIMAX_H3_MODALITY_NUM + token_tags` addresses.

    A single projection is shared by `norm1` and `norm2` and by the three modalities, so it cannot be folded into
    either norm the way [`~models.normalization.AdaLayerNormZero`] does. It is therefore a block-level module of its
    own, named after the checkpoint's `adaln_proj`, with the modulation projection under the `linear` name diffusers
    uses inside every AdaLN module.
    """

    def __init__(self, time_embed_dim: int, hidden_size: int, apply_silu: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.apply_silu = apply_silu
        self.linear = nn.Linear(time_embed_dim, 6 * hidden_size * MINIMAX_H3_MODALITY_NUM, bias=True)

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # The activation runs at `temb`'s own precision and only the projection input is aligned to the projection
        # weight. Every block reads the same `temb`, so early rounding biases every block's modulation coherently.
        temb = nn.functional.silu(temb) if self.apply_silu else temb
        temb = self.linear(temb.to(self.linear.weight.dtype))
        temb = temb.view(-1, 6 * self.hidden_size)
        return temb.chunk(6, dim=-1)


class MiniMaxH3AdaLayerNormOut(nn.Module):
    r"""
    Final norm of the packed sequence, shift/scale modulated per row.

    Same module layout and checkpoint keys as [`~models.normalization.AdaLayerNormContinuous`] (`norm` plus a `linear`
    projecting the conditioning embedding to `2 * hidden_size`), with two MiniMax-H3 specifics: the modulation table
    holds one row per *timestep* and is addressed per row of the packed sequence rather than per batch item, and the
    two halves of the projection are `shift` then `scale`, the order `LTX2Transformer3DModel` and
    `WanTransformer3DModel` also use in their output layers.
    """

    def __init__(self, hidden_size: int, time_embed_dim: int, eps: float, apply_silu: bool = True):
        super().__init__()
        self.apply_silu = apply_silu
        self.norm = nn.RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(time_embed_dim, 2 * hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep_indices: torch.Tensor,
    ) -> torch.Tensor:
        # As in `MiniMaxH3AdaLayerNormModulation`: activate at `temb`'s precision, cast to the projection's dtype after.
        temb = nn.functional.silu(temb) if self.apply_silu else temb
        shift, scale = self.linear(temb.to(self.linear.weight.dtype)).chunk(2, dim=-1)
        activation_dtype = hidden_states.dtype
        hidden_states = self.norm(hidden_states)
        shift = _select_modulation(shift, timestep_indices).to(dtype=activation_dtype)
        scale = _select_modulation(scale, timestep_indices).to(dtype=activation_dtype)
        # The modulation itself stays at the block stack's precision; `forward` casts to the output heads' dtype.
        return hidden_states * (1.0 + scale) + shift


class MiniMaxH3AttnProcessor:
    r"""
    Full self-attention over one packed sequence. There is no cross-attention anywhere in MiniMax-H3.
    """

    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "MiniMaxH3Attention",
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
        reference_kv_cache: dict[Any, tuple[torch.Tensor, torch.Tensor]] | None = None,
        reference_kv_stats: dict[str, int] | None = None,
        sparse_attention_layout: MiniMaxH3SparseAttentionLayout | None = None,
    ) -> torch.Tensor:
        use_reference_cache = (
            reference_mask is not None
            and reference_kv_cache is not None
            and not attn.fused_projections
            and not torch.is_grad_enabled()
            and bool(reference_mask.any())
        )
        if use_reference_cache:
            query = attn.to_q(hidden_states)
            query = query.unflatten(-1, (attn.heads, -1))
            query = attn.norm_q(query)
            if rotary_emb is not None:
                query = _apply_rotary_emb(query, *rotary_emb)

            reference_indices = reference_mask.nonzero(as_tuple=False).flatten()
            dynamic_indices = (~reference_mask).nonzero(as_tuple=False).flatten()
            key = query.new_empty(query.shape)
            value = query.new_empty(query.shape)

            if dynamic_indices.numel() > 0:
                dynamic_hidden_states = hidden_states.index_select(1, dynamic_indices)
                dynamic_key = attn.to_k(dynamic_hidden_states).unflatten(-1, (attn.heads, -1))
                dynamic_value = attn.to_v(dynamic_hidden_states).unflatten(-1, (attn.heads, -1))
                dynamic_key = attn.norm_k(dynamic_key)
                dynamic_rotary_emb = _slice_rotary_emb(rotary_emb, dynamic_indices)
                if dynamic_rotary_emb is not None:
                    dynamic_key = _apply_rotary_emb(dynamic_key, *dynamic_rotary_emb)
                key.index_copy_(1, dynamic_indices, dynamic_key)
                value.index_copy_(1, dynamic_indices, dynamic_value)

            reference_hidden_states = hidden_states.index_select(1, reference_indices)
            reference_rotary_emb = _slice_rotary_emb(rotary_emb, reference_indices)
            cache_key = (
                "kv",
                id(attn),
                _cache_tensor_signature(reference_hidden_states),
                _cache_rotary_signature(reference_rotary_emb),
            )
            cached = reference_kv_cache.get(cache_key)
            if cached is None:
                reference_key = attn.to_k(reference_hidden_states).unflatten(-1, (attn.heads, -1))
                reference_value = attn.to_v(reference_hidden_states).unflatten(-1, (attn.heads, -1))
                reference_key = attn.norm_k(reference_key)
                if reference_rotary_emb is not None:
                    reference_key = _apply_rotary_emb(reference_key, *reference_rotary_emb)
                cached = (reference_key.detach(), reference_value.detach())
                reference_kv_cache[cache_key] = cached
                if reference_kv_stats is not None:
                    reference_kv_stats["misses"] = reference_kv_stats.get("misses", 0) + 1
            else:
                if reference_kv_stats is not None:
                    reference_kv_stats["hits"] = reference_kv_stats.get("hits", 0) + 1
            reference_key, reference_value = cached
            key.index_copy_(
                1,
                reference_indices,
                reference_key.to(device=key.device, dtype=key.dtype),
            )
            value.index_copy_(
                1,
                reference_indices,
                reference_value.to(device=value.device, dtype=value.dtype),
            )
        elif attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))

            query = attn.norm_q(query)
            key = attn.norm_k(key)

            if rotary_emb is not None:
                query = _apply_rotary_emb(query, *rotary_emb)
                key = _apply_rotary_emb(key, *rotary_emb)
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))

            query = attn.norm_q(query)
            key = attn.norm_k(key)

            if rotary_emb is not None:
                query = _apply_rotary_emb(query, *rotary_emb)
                key = _apply_rotary_emb(key, *rotary_emb)

        # Without padding rows the packed sequence is a single attention document and no mask is needed (passing an
        # all-zero float mask here would hard-fail the flash / sage backends). When padding rows are present, the
        # caller supplies a boolean mask that keeps them in their own attention document, mirroring the reference's
        # `cu_seqlens = [0, used, S]` split; masked backends (SDPA & co.) are required in that case.
        if attention_mask is not None and context_parallel_config(self._parallel_config) is not None:
            attention_mask = prepare_cp_attention_mask(
                attention_mask,
                query.shape[1],
                self._parallel_config,
                model_name="MiniMax-H3",
            )
        sparse_config = getattr(attn, "_h3_sparse_attention_config", None)
        use_sparse_attention = sparse_attention_layout is not None and sparse_config is not None and sparse_config.enabled
        if use_sparse_attention:
            sparse_layer_index = int(getattr(attn, "_h3_sparse_layer_index", -1))
            use_sparse_attention = sparse_layer_index >= sparse_config.start_layer
        if use_sparse_attention:
            cp_config = context_parallel_config(self._parallel_config)
            if attention_mask is not None and cp_config is None and sparse_attention_layout.packed_valid_mask is None:
                raise ValueError("MiniMax-H3 sparse attention cannot be combined with a packed attention mask yet.")
            if reference_mask is not None and attention_mask is not None:
                raise ValueError("MiniMax-H3 sparse attention cannot combine batched padding with CachedKV masking yet.")
            if cp_config is not None:
                hidden_states = minimax_h3_sparse_attention_ulysses(
                    query,
                    key,
                    value,
                    layout=sparse_attention_layout,
                    config=sparse_config,
                    process_group=cp_config._ulysses_mesh.get_group(),
                )
            else:
                hidden_states = minimax_h3_sparse_attention(
                    query,
                    key,
                    value,
                    layout=sparse_attention_layout,
                    config=sparse_config,
                )
        else:
            hidden_states = dispatch_attention_fn(
                query,
                key,
                value,
                attn_mask=attention_mask,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                parallel_config=self._parallel_config,
            )
        hidden_states = hidden_states.flatten(2, 3).type_as(query)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class MiniMaxH3Attention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = MiniMaxH3AttnProcessor
    _available_processors = [MiniMaxH3AttnProcessor]

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        dim_head: int,
        qk_norm_eps: float = 1e-5,
        processor=None,
    ):
        super().__init__()
        self.heads = heads
        self.head_dim = dim_head
        self.inner_dim = heads * dim_head
        self.use_bias = False

        self.to_q = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_k = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.to_v = nn.Linear(hidden_size, self.inner_dim, bias=False)
        self.norm_q = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.norm_k = nn.RMSNorm(dim_head, eps=qk_norm_eps)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, hidden_size, bias=False), nn.Dropout(0.0)])

        if processor is None:
            processor = self._default_processor_cls()
        self.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
        reference_kv_cache: dict[Any, tuple[torch.Tensor, torch.Tensor]] | None = None,
        reference_kv_stats: dict[str, int] | None = None,
        sparse_attention_layout: MiniMaxH3SparseAttentionLayout | None = None,
    ) -> torch.Tensor:
        return self.processor(
            self,
            hidden_states,
            rotary_emb,
            attention_mask,
            reference_mask=reference_mask,
            reference_kv_cache=reference_kv_cache,
            reference_kv_stats=reference_kv_stats,
            sparse_attention_layout=sparse_attention_layout,
        )


class MiniMaxH3TokenRefinerBlock(nn.Module):
    r"""
    Plain pre-norm transformer block used to refine the projected text stream. No AdaLN and no rotary embedding.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        swiglu_gate_first: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        if swiglu_gate_first:
            self.ff = MiniMaxH3FeedForward(hidden_size, inner_dim=ffn_dim, bias=False, gate_first=True)
        else:
            self.ff = FeedForward(hidden_size, inner_dim=ffn_dim, activation_fn="swiglu", bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        checkpoint_ffn: bool = False,
        checkpoint_fn: Any | None = None,
        offload_attention: bool = False,
    ) -> torch.Tensor:
        with activation_offload_context(offload_attention, label=f"{self.__class__.__qualname__}:attention"):
            hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask=attention_mask)
        if checkpoint_ffn:
            if checkpoint_fn is None:
                raise ValueError("checkpoint_fn is required when checkpoint_ffn=True")
            ff_output = checkpoint_fn(self._ff_forward, hidden_states, use_reentrant=False)
        else:
            ff_output = self._ff_forward(hidden_states)
        hidden_states = hidden_states + ff_output
        return hidden_states

    def _ff_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.ff(self.norm2(hidden_states))


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        num_layers: int,
        norm_eps: float,
        qk_norm_eps: float,
        final_norm_eps: float,
        swiglu_gate_first: bool = False,
    ):
        super().__init__()
        self.refiner_blocks = nn.ModuleList(
            [
                MiniMaxH3TokenRefinerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                    swiglu_gate_first=swiglu_gate_first,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(hidden_size, eps=final_norm_eps)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        checkpoint_backend: str = "torch",
        offload_attention: bool = False,
    ) -> torch.Tensor:
        checkpoint_fn = None
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            if checkpoint_backend.startswith("unsloth"):
                from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

                checkpoint_fn = offloaded_checkpoint
            else:
                checkpoint_fn = torch.utils.checkpoint.checkpoint

        for block in self.refiner_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                if checkpoint_backend.endswith("-ffn"):
                    hidden_states = block(
                        hidden_states,
                        attention_mask=attention_mask,
                        checkpoint_ffn=True,
                        checkpoint_fn=checkpoint_fn,
                        offload_attention=offload_attention,
                    )
                else:

                    def run_checkpointed_block(checkpoint_hidden_states, checkpoint_block=block):
                        return checkpoint_block(
                            checkpoint_hidden_states,
                            attention_mask=attention_mask,
                            offload_attention=offload_attention,
                        )

                    hidden_states = checkpoint_fn(run_checkpointed_block, hidden_states, use_reentrant=False)
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    offload_attention=offload_attention,
                )
        return self.final_norm(hidden_states)


class MiniMaxH3TransformerBlock(nn.Module):
    r"""
    MiniMax-H3 block: pre-norm self-attention and feed-forward, each modulated by AdaLN parameters selected per row of
    the packed sequence from the `(timestep, modality)` table.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_head_dim: int,
        ffn_dim: int,
        time_embed_dim: int,
        norm_eps: float,
        qk_norm_eps: float,
        adaln_apply_silu: bool = True,
        swiglu_gate_first: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(hidden_size, eps=norm_eps)
        self.attn = MiniMaxH3Attention(
            hidden_size=hidden_size,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            qk_norm_eps=qk_norm_eps,
        )
        self.norm2 = nn.RMSNorm(hidden_size, eps=norm_eps)
        if swiglu_gate_first:
            self.ff = MiniMaxH3FeedForward(hidden_size, inner_dim=ffn_dim, bias=False, gate_first=True)
        else:
            self.ff = FeedForward(hidden_size, inner_dim=ffn_dim, activation_fn="swiglu", bias=False)
        self.adaln_proj = MiniMaxH3AdaLayerNormModulation(
            time_embed_dim=time_embed_dim,
            hidden_size=hidden_size,
            apply_silu=adaln_apply_silu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        adaln_indices: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        checkpoint_ffn: bool = False,
        checkpoint_fn: Any | None = None,
        offload_attention: bool = False,
        reference_mask: torch.Tensor | None = None,
        reference_kv_cache: dict[Any, tuple[torch.Tensor, torch.Tensor] | torch.Tensor] | None = None,
        reference_kv_stats: dict[str, int] | None = None,
        sparse_attention_layout: MiniMaxH3SparseAttentionLayout | None = None,
    ) -> torch.Tensor:
        cached_reference_hidden_states = None
        reference_post_cache_key = None
        if reference_mask is not None and reference_kv_cache is not None and bool(reference_mask.any()):
            reference_hidden_states = hidden_states[:, reference_mask, :]
            reference_post_cache_key = (
                "post",
                id(self),
                _cache_tensor_signature(reference_hidden_states),
            )
            cached_reference_hidden_states = reference_kv_cache.get(reference_post_cache_key)
            if cached_reference_hidden_states is not None and reference_kv_stats is not None:
                reference_kv_stats["post_hits"] = reference_kv_stats.get("post_hits", 0) + 1

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_proj(temb)
        activation_dtype = hidden_states.dtype
        shift_msa = _select_modulation(shift_msa, adaln_indices).to(dtype=activation_dtype)
        scale_msa = _select_modulation(scale_msa, adaln_indices).to(dtype=activation_dtype)
        gate_msa = _select_modulation(gate_msa, adaln_indices).to(dtype=activation_dtype)
        shift_mlp = _select_modulation(shift_mlp, adaln_indices).to(dtype=activation_dtype)
        scale_mlp = _select_modulation(scale_mlp, adaln_indices).to(dtype=activation_dtype)
        gate_mlp = _select_modulation(gate_mlp, adaln_indices).to(dtype=activation_dtype)

        residual = hidden_states
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1.0 + scale_msa) + shift_msa
        with activation_offload_context(offload_attention, label=f"{self.__class__.__qualname__}:attention"):
            attn_output = self.attn(
                norm_hidden_states,
                rotary_emb,
                attention_mask,
                reference_mask=reference_mask,
                reference_kv_cache=reference_kv_cache,
                reference_kv_stats=reference_kv_stats,
                sparse_attention_layout=sparse_attention_layout,
            )
        hidden_states = residual + gate_msa * attn_output

        residual = hidden_states
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1.0 + scale_mlp) + shift_mlp
        if checkpoint_ffn:
            if checkpoint_fn is None:
                raise ValueError("checkpoint_fn is required when checkpoint_ffn=True")
            ff_output = checkpoint_fn(self.ff, norm_hidden_states, use_reentrant=False)
        else:
            ff_output = self.ff(norm_hidden_states)
        hidden_states = residual + gate_mlp * ff_output

        if reference_post_cache_key is not None:
            if cached_reference_hidden_states is None:
                reference_kv_cache[reference_post_cache_key] = hidden_states[:, reference_mask, :].detach()
                if reference_kv_stats is not None:
                    reference_kv_stats["post_misses"] = reference_kv_stats.get("post_misses", 0) + 1
            else:
                hidden_states = hidden_states.clone()
                hidden_states[:, reference_mask, :] = cached_reference_hidden_states.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )

        return hidden_states


class MiniMaxH3Transformer3DModel(ModelMixin, ConfigMixin, AttentionMixin, PeftAdapterMixin, CacheMixin):
    r"""
    A Transformer model for joint video + audio generation, introduced in MiniMax-H3.

    MiniMax-H3 runs a single stack of blocks over **one packed 1-D sequence** that holds the text condition, the
    conditioning image / video rows, the audio rows and the target video rows. Attention is full self-attention over
    that sequence; there is no cross-attention and no per-modality block weights. Modality-specific behaviour comes
    only from the two input patch projections, the per-row AdaLN modality tag, and the two output heads.

    The caller is responsible for building the packed layout: patchifying the video latents, ordering the rows, and
    producing the `(t, h, w)` position grid, the per-row modality tags and the per-row timestep indices. Padding rows
    (tag `-1`) are kept in a separate attention document, matching the reference implementation, which pads to a
    multiple of 64 for FlashAttention with `cu_seqlens = [0, used, S]`. Prefer dropping them — a padless sequence
    needs no attention mask, keeping the unmasked attention backends available.

    The batch axis is a pure replication axis: the structural arguments (`timestep`, `timestep_indices`, `token_tags`,
    `position_ids` and the three index tensors) describe one packed layout that every batch item shares, and each item
    is a single attention document.

    Args:
        num_attention_heads (`int`, defaults to `56`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each attention head. Note that `num_attention_heads * attention_head_dim` is
            *larger* than `hidden_size` in MiniMax-H3.
        hidden_size (`int`, defaults to `5376`):
            The number of channels of the packed sequence (the residual stream).
        num_layers (`int`, defaults to `50`):
            The number of transformer blocks.
        num_refiner_layers (`int`, defaults to `2`):
            The number of token refiner blocks applied to the projected text stream.
        ffn_dim (`int`, defaults to `14336`):
            The inner dimension of the SwiGLU feed-forward layers.
        in_channels (`int`, defaults to `24`):
            The number of channels of the video latents.
        audio_in_channels (`int`, defaults to `32`):
            The number of channels of the audio latents.
        patch_size (`tuple[int, int, int]`, defaults to `(1, 2, 2)`):
            The `(t, h, w)` patch used to pack the video latents into rows.
        text_dim (`int`, defaults to `5120`):
            The number of channels of the text conditioning produced by the text encoder.
        freq_dim (`int`, defaults to `256`):
            The dimension of the sinusoidal timestep embedding. Timesteps are consumed unscaled in `[0, 1]`.
        time_embed_hidden_dim (`int`, defaults to `5376`):
            The inner dimension of the timestep MLP.
        time_embed_dim (`int`, defaults to `2688`):
            The output dimension of the timestep MLP, i.e. the input of every AdaLN projection.
        rope_freq_dim (`int`, defaults to `16`):
            The number of rotary frequencies per axis. The `(t, h, w)` axes share one `inv_freq` buffer of this length
            and `2 * 3 * rope_freq_dim` of the `attention_head_dim` channels are rotated.
        rope_theta (`float`, defaults to `10000.0`):
            The base of the rotary frequency schedule the `rope.inv_freq` buffer is computed from.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon of the pre-attention and pre-feed-forward norms.
        qk_norm_eps (`float`, defaults to `1e-5`):
            Epsilon of the per-head query/key norms.
        final_norm_eps (`float`, defaults to `1e-5`):
            Epsilon of the token refiner output norm and of `norm_out`.
        enable_time_sign_embed (`bool`, defaults to `False`):
            Adds a zero-initialized signed-timestep embedding for TwinFlow-compatible training.
        gate_value (`float`, *optional*):
            FlowMap/AnyFlow delta embedding blend gate, used when `deltatime_type` is set.
        deltatime_type (`str`, *optional*):
            Enables FlowMap/AnyFlow interval conditioning. Must be `"r"` or `"t-r"`.
    """

    _supports_gradient_checkpointing = True
    _supports_ffn_gradient_checkpointing = True
    _supports_attention_activation_offload = True
    _tread_router: TREADRouter | None = None
    _no_split_modules = [
        "MiniMaxH3TransformerBlock",
        "MiniMaxH3TokenRefinerBlock",
        "MiniMaxH3AdaLayerNormOut",
    ]
    _repeated_blocks = ["MiniMaxH3TransformerBlock", "MiniMaxH3TokenRefinerBlock"]
    # MiniMax-H3 builds one packed sequence inside forward, so CP sharding starts at the first transformer block.
    # The modality heads are gathered explicitly in forward because Diffusers' generic output hook uses a functional
    # coalesced all-gather that NCCL does not support for hybrid CP subgroups in current PyTorch releases.
    _cp_plan = {
        "rope": {
            0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
            1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        },
        "transformer_blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "transformer_blocks.*": {
            "adaln_indices": ContextParallelInput(split_dim=0, expected_dims=1, split_output=False),
        },
        "norm_out": {
            "timestep_indices": ContextParallelInput(split_dim=0, expected_dims=1, split_output=False),
        },
    }
    _skip_layerwise_casting_patterns = ["norm"]
    # MiniMax-H3 ships a mixed-precision checkpoint: the two input patch projections, the timestep MLP and the two
    # output heads are float32 while everything else (including the AdaLN projections) is bfloat16. The `rope.inv_freq`
    # and optional `adaln_t_table` buffers are kept float32 for the same reason the reference ships them float32.
    # Entries are matched as substrings of the parameter name, so `proj_in` / `proj_out` also cover the audio heads.
    _keep_in_fp32_modules = [
        "proj_in",
        "audio_proj_in",
        "adaln_t_table",
        "delta_adaln_embedder",
        "time_embedder",
        "proj_out",
        "audio_proj_out",
        "rope",
    ]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 56,
        attention_head_dim: int = 128,
        hidden_size: int = 5376,
        num_layers: int = 50,
        num_refiner_layers: int = 2,
        ffn_dim: int = 14336,
        in_channels: int = 24,
        audio_in_channels: int = 32,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        text_dim: int = 5120,
        freq_dim: int = 256,
        time_embed_hidden_dim: int = 5376,
        time_embed_dim: int = 2688,
        rope_freq_dim: int = 16,
        rope_theta: float = 10000.0,
        norm_eps: float = 1e-5,
        qk_norm_eps: float = 1e-5,
        final_norm_eps: float = 1e-5,
        enable_time_sign_embed: bool = False,
        gate_value: float | None = None,
        deltatime_type: str | None = None,
        adaln_curve_grid: int | None = None,
        swiglu_gate_first: bool = False,
        musubi_blocks_to_swap: int = 0,
        musubi_block_swap_device: str = "cpu",
    ) -> None:
        super().__init__()

        video_patch_dim = in_channels * patch_size[0] * patch_size[1] * patch_size[2]
        self.use_adaln_curves = adaln_curve_grid is not None
        if self.use_adaln_curves and enable_time_sign_embed:
            raise ValueError("MiniMax-H3 adaln_t_table checkpoints do not support TwinFlow time-sign embeddings.")

        # 1. Per-modality input projections
        self.proj_in = nn.Linear(video_patch_dim, hidden_size, bias=True)
        self.audio_proj_in = nn.Linear(audio_in_channels, hidden_size, bias=True)
        self.context_embedder = nn.Linear(text_dim, hidden_size, bias=True)

        # 2. Timestep embedding, shared by every AdaLN projection
        if self.use_adaln_curves:
            self.time_proj = None
            self.time_embedder = None
            self.register_buffer(
                "adaln_t_table",
                torch.empty(adaln_curve_grid, time_embed_dim, dtype=torch.float32),
            )
        else:
            self.time_proj = Timesteps(num_channels=freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.time_embedder = TimestepEmbedding(
                in_channels=freq_dim,
                time_embed_dim=time_embed_hidden_dim,
                out_dim=time_embed_dim,
            )
        self.time_sign_embed = None
        if enable_time_sign_embed:
            self.time_sign_embed = nn.Embedding(2, time_embed_dim)
            nn.init.zeros_(self.time_sign_embed.weight)
        self.delta_time_embedder = None
        self.delta_adaln_embedder = None
        self.flowmap_deltatime_type = None
        register_flowmap_gate_buffer(self, gate_value=0.25 if gate_value is None else float(gate_value))
        if deltatime_type is not None:
            self.enable_flowmap_time_conditioning(
                gate_value=0.25 if gate_value is None else float(gate_value),
                deltatime_type=deltatime_type,
            )

        # 3. Rotary embedding over the packed (t, h, w) grid
        self.rope = MiniMaxH3RotaryPosEmbed(rope_freq_dim=rope_freq_dim, rope_theta=rope_theta)

        # 4. Text stream refiner
        self.token_refiner = MiniMaxH3TokenRefiner(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            ffn_dim=ffn_dim,
            num_layers=num_refiner_layers,
            norm_eps=norm_eps,
            qk_norm_eps=qk_norm_eps,
            final_norm_eps=final_norm_eps,
            swiglu_gate_first=swiglu_gate_first,
        )

        # 5. The block stack
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3TransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    ffn_dim=ffn_dim,
                    time_embed_dim=time_embed_dim,
                    norm_eps=norm_eps,
                    qk_norm_eps=qk_norm_eps,
                    adaln_apply_silu=not self.use_adaln_curves,
                    swiglu_gate_first=swiglu_gate_first,
                )
                for _ in range(num_layers)
            ]
        )
        self._h3_sparse_attention_config = MiniMaxH3SparseAttentionConfig()
        for layer_index, block in enumerate(self.transformer_blocks):
            block.attn._h3_sparse_layer_index = layer_index
            block.attn._h3_sparse_attention_config = self._h3_sparse_attention_config

        # 6. Shared output norm and the two per-modality output heads. Both heads run over every row of the packed
        # sequence; the rows of each modality are selected afterwards.
        self.norm_out = MiniMaxH3AdaLayerNormOut(
            hidden_size=hidden_size,
            time_embed_dim=time_embed_dim,
            eps=final_norm_eps,
            apply_silu=not self.use_adaln_curves,
        )
        self.proj_out = nn.Linear(hidden_size, video_patch_dim, bias=True)
        self.audio_proj_out = nn.Linear(hidden_size, audio_in_channels, bias=True)

        self.gradient_checkpointing = False
        self.gradient_checkpointing_backend = "torch"
        self.gradient_checkpointing_offload_attention = False
        self.gradient_checkpointing_interval = None
        self.gradient_checkpointing_segment_stride = None
        self._tread_router = None
        self._tread_routes = []
        self._h3_reference_kv_cache: dict[Any, tuple[torch.Tensor, torch.Tensor] | torch.Tensor] = {}
        self._h3_reference_kv_stats: dict[str, int] = {}
        self._musubi_block_swap = MusubiBlockSwapManager.build(
            depth=num_layers,
            blocks_to_swap=musubi_blocks_to_swap,
            swap_device=musubi_block_swap_device,
            logger=logger,
        )

    def clear_h3_reference_kv_cache(self) -> None:
        self._h3_reference_kv_cache.clear()
        self._h3_reference_kv_stats.clear()

    def get_h3_reference_kv_stats(self) -> dict[str, int]:
        return dict(self._h3_reference_kv_stats)

    def configure_h3_sparse_attention(
        self,
        *,
        mode: str = "disabled",
        block_shape: str | tuple[int, int, int] | list[int] = (1, 8, 16),
        video_kv_fraction: float = 0.5,
        share_across_heads: bool = False,
        start_layer: int = 0,
    ) -> MiniMaxH3SparseAttentionConfig:
        config = MiniMaxH3SparseAttentionConfig(
            mode=mode,
            block_shape=block_shape,
            video_kv_fraction=video_kv_fraction,
            share_across_heads=share_across_heads,
            start_layer=start_layer,
        )
        if config.start_layer >= len(self.transformer_blocks) and config.enabled:
            raise ValueError(
                f"MiniMax-H3 sparse start layer {config.start_layer} is outside the "
                f"{len(self.transformer_blocks)}-layer transformer."
            )
        if config.enabled:
            initialize_minimax_h3_flex_attention()
        self._h3_sparse_attention_config = config
        for layer_index, block in enumerate(self.transformer_blocks):
            block.attn._h3_sparse_layer_index = layer_index
            block.attn._h3_sparse_attention_config = config
        return config

    def enable_parallelism(self, *, config, cp_plan=None):
        context_config = getattr(config, "context_parallel_config", None)
        if context_config is None and hasattr(config, "ring_degree"):
            context_config = config
        if context_config is not None:
            processor = self.transformer_blocks[0].attn.processor
            backend = processor._attention_backend
            if backend is None:
                backend, _ = _AttentionBackendRegistry.get_active_backend()
            else:
                backend = AttentionBackendName(backend)
            if not _AttentionBackendRegistry._is_context_parallel_available(backend):
                logger.warning(
                    "MiniMax-H3 context parallelism cannot use attention backend %s; "
                    "falling back to the native FlashAttention backend.",
                    backend.value,
                )
                self.set_attention_backend("_native_flash")
        return super().enable_parallelism(config=config, cp_plan=cp_plan)

    @staticmethod
    def _build_reference_mask(
        sequence_length: int,
        *,
        device: torch.device,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        num_condition_video_rows: int = 0,
        num_condition_audio_rows: int = 0,
    ) -> torch.Tensor | None:
        num_condition_video_rows = int(num_condition_video_rows or 0)
        num_condition_audio_rows = int(num_condition_audio_rows or 0)
        if num_condition_video_rows < 0 or num_condition_audio_rows < 0:
            raise ValueError("MiniMax-H3 conditioning row counts must be non-negative.")
        if num_condition_video_rows > int(video_indices.shape[0]):
            raise ValueError(
                "MiniMax-H3 num_condition_video_rows exceeds the number of video rows: "
                f"{num_condition_video_rows} > {int(video_indices.shape[0])}."
            )
        if num_condition_audio_rows > int(audio_indices.shape[0]):
            raise ValueError(
                "MiniMax-H3 num_condition_audio_rows exceeds the number of audio rows: "
                f"{num_condition_audio_rows} > {int(audio_indices.shape[0])}."
            )
        static_row_count = int(text_indices.shape[0]) + num_condition_video_rows + num_condition_audio_rows
        if static_row_count == 0:
            return None

        reference_mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
        if text_indices.numel() > 0:
            reference_mask.index_fill_(0, text_indices.to(device=device), True)
        if num_condition_video_rows:
            reference_mask.index_fill_(0, video_indices[:num_condition_video_rows].to(device=device), True)
        if num_condition_audio_rows:
            reference_mask.index_fill_(0, audio_indices[:num_condition_audio_rows].to(device=device), True)
        return reference_mask

    def enable_flowmap_time_conditioning(self, gate_value: float = 0.25, deltatime_type: str = "r") -> None:
        self.flowmap_deltatime_type = validate_flowmap_deltatime_type(deltatime_type, model_name="MiniMax-H3")
        if self.time_embedder is None and self.delta_adaln_embedder is None:
            self.delta_adaln_embedder = MiniMaxH3AdaLNCurveEmbedder(self.adaln_t_table)
        elif self.time_embedder is not None and self.delta_time_embedder is None:
            self.delta_time_embedder = clone_flowmap_embedder(self.time_embedder)
        set_flowmap_gate(self, gate_value)
        register_flowmap_config(self, gate_value, deltatime_type)

    def _adaln_curve_embedding(self, timestep: torch.Tensor) -> torch.Tensor:
        return _interpolate_adaln_curve(self.adaln_t_table, timestep)

    def _time_embedding(
        self,
        timestep: torch.Tensor,
        timestep_sign: torch.Tensor | None = None,
        r_timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.time_embedder is None:
            if timestep_sign is not None:
                raise ValueError("MiniMax-H3 adaln_t_table checkpoints do not support TwinFlow time-sign embeddings.")
            temb = self._adaln_curve_embedding(timestep)
            if r_timestep is not None:
                if self.flowmap_deltatime_type is None:
                    raise ValueError(
                        "MiniMax-H3 FlowMap conditioning requires `enable_flowmap_time_conditioning()` before training."
                    )
                if self.flowmap_deltatime_type == "t-r":
                    # H3 time is data-ward, so a denoising endpoint r is greater than the current t.
                    delta_timestep = prepare_flowmap_delta_timestep(
                        r_timestep,
                        timestep,
                        self.flowmap_deltatime_type,
                        model_name="MiniMax-H3",
                    )
                else:
                    delta_timestep = prepare_flowmap_delta_timestep(
                        timestep,
                        r_timestep,
                        self.flowmap_deltatime_type,
                        model_name="MiniMax-H3",
                    )
                if self.delta_adaln_embedder is None:
                    raise ValueError(
                        "MiniMax-H3 FlowMap conditioning requires `enable_flowmap_time_conditioning()` before training."
                    )
                delta_temb = self.delta_adaln_embedder(delta_timestep)
                temb = blend_flowmap_embeddings(temb, delta_temb, self.flowmap_delta_emb_gate)
            return temb

        dtype = self.time_embedder.linear_1.weight.dtype
        temb = flowmap_timestep_embedding(
            time_proj=self.time_proj,
            timestep_embedder=self.time_embedder,
            timestep=timestep,
            dtype=dtype,
        )
        if r_timestep is not None:
            if self.delta_time_embedder is None or self.flowmap_deltatime_type is None:
                raise ValueError(
                    "MiniMax-H3 FlowMap conditioning requires `enable_flowmap_time_conditioning()` before training."
                )
            delta_timestep = prepare_flowmap_delta_timestep(
                timestep,
                r_timestep,
                self.flowmap_deltatime_type,
                model_name="MiniMax-H3",
            )
            delta_temb = flowmap_timestep_embedding(
                time_proj=self.time_proj,
                timestep_embedder=self.delta_time_embedder,
                timestep=delta_timestep,
                dtype=dtype,
            )
            temb = blend_flowmap_embeddings(temb, delta_temb, self.flowmap_delta_emb_gate)
        if timestep_sign is not None:
            if self.time_sign_embed is None:
                raise ValueError(
                    "timestep_sign was provided but the model was loaded without `enable_time_sign_embed=True`. "
                    "Enable TwinFlow before loading the MiniMax-H3 transformer."
                )
            sign_tensor = timestep_sign.to(device=temb.device)
            if temb.ndim != 2:
                raise ValueError(f"MiniMax-H3 timestep embedding must be 2-D, got {tuple(temb.shape)}.")
            if sign_tensor.ndim == 0:
                sign_tensor = sign_tensor.expand(temb.shape[0])
            elif sign_tensor.ndim == 1:
                if sign_tensor.shape[0] == 1:
                    sign_tensor = sign_tensor.expand(temb.shape[0])
                elif sign_tensor.shape[0] != temb.shape[0]:
                    raise ValueError(
                        f"MiniMax-H3 timestep_sign expected 1 or {temb.shape[0]} values, got {sign_tensor.shape[0]}."
                    )
            else:
                raise ValueError(
                    "MiniMax-H3 timestep_sign expected scalar or 1-D tensor matching unique timesteps, "
                    f"got shape {tuple(sign_tensor.shape)}."
                )
            sign_idx = (sign_tensor.reshape(-1) < 0).long().to(device=temb.device)
            temb = temb + self.time_sign_embed(sign_idx).to(device=temb.device, dtype=temb.dtype)
        return temb

    def set_gradient_checkpointing_backend(self, backend: str):
        self.gradient_checkpointing_backend = backend

    def set_gradient_checkpointing_offload_attention(self, enabled: bool):
        self.gradient_checkpointing_offload_attention = bool(enabled)

    def set_gradient_checkpointing_interval(self, interval: int):
        self.gradient_checkpointing_interval = interval

    def set_gradient_checkpointing_segment_stride(self, segment_stride: int | None):
        self.gradient_checkpointing_segment_stride = segment_stride

    def set_router(self, router: TREADRouter, routes: list[dict[str, Any]]):
        self._tread_router = router
        self._tread_routes = routes

    @staticmethod
    def _route_token_tensor(tensor: torch.Tensor, info, keep_len: int, batch_size: int) -> torch.Tensor:
        tensor = tensor.to(device=info.ids_shuffle.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0).expand(batch_size, -1)
        if tensor.ndim != 2:
            raise ValueError(f"MiniMax-H3 TREAD metadata tensors must be 1-D or 2-D, got {list(tensor.shape)}.")
        routed = torch.take_along_dim(tensor, info.ids_shuffle, dim=1)
        return routed[:, :keep_len]

    @staticmethod
    def _route_rotary(
        rotary_emb: tuple[torch.Tensor, torch.Tensor], info, keep_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def route_one(value: torch.Tensor) -> torch.Tensor:
            if value.ndim == 2:
                value = value.unsqueeze(0).expand(info.ids_shuffle.shape[0], -1, -1)
            if value.ndim != 3:
                raise ValueError(f"MiniMax-H3 rotary tensors must be 2-D or 3-D, got {list(value.shape)}.")
            routed = torch.take_along_dim(
                value,
                info.ids_shuffle.unsqueeze(-1).expand(-1, -1, value.shape[-1]),
                dim=1,
            )
            return routed[:, :keep_len]

        return tuple(route_one(value) for value in rotary_emb)

    @classmethod
    def from_single_file(
        cls,
        pretrained_model_link_or_path: str,
        *args: Any,
        filename: str | None = None,
        subfolder: str | None = None,
        revision: str | None = None,
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> "MiniMaxH3Transformer3DModel":
        del args
        checkpoint_path = _resolve_minimax_h3_single_file_path(
            pretrained_model_link_or_path,
            filename=filename,
            subfolder=subfolder,
            revision=revision,
        )
        init_passthrough = {
            key: kwargs[key]
            for key in (
                "enable_time_sign_embed",
                "gate_value",
                "deltatime_type",
                "musubi_blocks_to_swap",
                "musubi_block_swap_device",
            )
            if key in kwargs
        }

        non_quantized_state_dict: dict[str, torch.Tensor] = {}
        fp8_state_dict: dict[str, torch.Tensor] = {}
        quantized_weights: dict[str, tuple[torch.Tensor, torch.Tensor, int]] = {}

        with _open_minimax_h3_single_file(checkpoint_path) as checkpoint:
            inferred_config = _infer_minimax_h3_config_from_checkpoint(checkpoint)
            with torch.device("meta"):
                model = cls(**inferred_config, **init_passthrough)
            expected_state_dict = model.state_dict()
            checkpoint_keys = set(checkpoint.keys())

            for raw_key in checkpoint.keys():
                key = _strip_minimax_h3_checkpoint_prefix(raw_key)
                if key.endswith(_COMFY_QUANT_METADATA_SUFFIXES):
                    continue
                mapped_keys = _map_minimax_h3_comfy_key_to_diffusers(key)
                if not mapped_keys:
                    continue

                tensor = checkpoint.get_tensor(raw_key)
                tensor = _convert_minimax_h3_native_swiglu_to_diffusers(key, tensor)
                if tensor.dtype in _COMFY_FP8_DTYPES:
                    from simpletuner.helpers.models.z_image.quantized_loading import _decode_comfy_quant

                    scale_key = f"{raw_key}_scale"
                    if scale_key not in checkpoint_keys:
                        raise RuntimeError(f"MiniMax-H3 FP8 tensor {raw_key} is missing weight_scale")
                    quant_key = f"{raw_key.removesuffix('.weight')}.comfy_quant"
                    if quant_key not in checkpoint_keys:
                        raise RuntimeError(f"MiniMax-H3 FP8 tensor {raw_key} is missing comfy_quant metadata")
                    quant_metadata = _decode_comfy_quant(checkpoint.get_tensor(quant_key))
                    if quant_metadata.get("format") not in {
                        "float8_e4m3fn",
                        "float8_e5m2",
                    }:
                        raise RuntimeError(
                            f"MiniMax-H3 FP8 tensor {raw_key} has unsupported comfy_quant format "
                            f"{quant_metadata.get('format')!r}"
                        )
                    scale = checkpoint.get_tensor(scale_key).to(torch.float32)
                    scale = _convert_minimax_h3_native_swiglu_scale_to_diffusers(key, scale)
                    if len(mapped_keys) == 3:
                        if tensor.shape[0] % 3 != 0:
                            raise RuntimeError(f"MiniMax-H3 FP8 tensor {raw_key} cannot be split into q/k/v tensors")
                        split_size = tensor.shape[0] // 3
                        qkv_tensors = tensor.split(split_size, dim=0)
                        if scale.ndim == 0 or scale.numel() == 1:
                            qkv_scales = [scale.reshape(())] * 3
                        elif scale.shape[0] % 3 == 0:
                            qkv_scales = scale.split(scale.shape[0] // 3, dim=0)
                        else:
                            raise RuntimeError(
                                f"MiniMax-H3 FP8 tensor {raw_key} has weight_scale shape {tuple(scale.shape)} "
                                "that cannot be split into q/k/v tensors"
                            )
                        for mapped_key, qkv_tensor, qkv_scale in zip(mapped_keys, qkv_tensors, qkv_scales):
                            fp8_state_dict[mapped_key] = qkv_tensor.contiguous()
                            fp8_state_dict[f"{mapped_key.removesuffix('.weight')}.weight_scale"] = qkv_scale.contiguous()
                    elif len(mapped_keys) == 1:
                        fp8_state_dict[mapped_keys[0]] = tensor
                        fp8_state_dict[f"{mapped_keys[0].removesuffix('.weight')}.weight_scale"] = scale
                    else:
                        raise RuntimeError(f"MiniMax-H3 FP8 tensor {raw_key} maps to multiple targets unexpectedly")
                    continue

                if tensor.dtype in {torch.int8}:
                    from simpletuner.helpers.models.z_image.quantized_loading import _decode_comfy_quant

                    scale_key = f"{raw_key}_scale"
                    if scale_key not in checkpoint_keys:
                        raise RuntimeError(f"MiniMax-H3 ConvRot tensor {raw_key} is missing weight_scale")
                    quant_key = f"{raw_key.removesuffix('.weight')}.comfy_quant"
                    if quant_key not in checkpoint_keys:
                        raise RuntimeError(f"MiniMax-H3 ConvRot tensor {raw_key} is missing comfy_quant metadata")
                    quant_metadata = _decode_comfy_quant(checkpoint.get_tensor(quant_key))
                    if not quant_metadata.get("convrot", False):
                        raise RuntimeError(f"MiniMax-H3 INT8 tensor {raw_key} is not marked as ConvRot")
                    hadamard_group_size = int(quant_metadata.get("convrot_groupsize", 0))
                    if hadamard_group_size <= 0:
                        raise RuntimeError(f"MiniMax-H3 ConvRot tensor {raw_key} has invalid convrot_groupsize")
                    scale = checkpoint.get_tensor(scale_key)
                    scale = _convert_minimax_h3_native_swiglu_scale_to_diffusers(key, scale)
                    if len(mapped_keys) == 3:
                        if tensor.shape[0] % 3 != 0 or scale.shape[0] % 3 != 0:
                            raise RuntimeError(f"MiniMax-H3 ConvRot tensor {raw_key} cannot be split into q/k/v tensors")
                        for mapped_key, qkv_tensor, qkv_scale in zip(
                            mapped_keys,
                            tensor.split(tensor.shape[0] // 3, dim=0),
                            scale.split(scale.shape[0] // 3, dim=0),
                        ):
                            quantized_weights[mapped_key] = (
                                qkv_tensor.contiguous(),
                                qkv_scale.contiguous(),
                                hadamard_group_size,
                            )
                    elif len(mapped_keys) == 1:
                        quantized_weights[mapped_keys[0]] = (
                            tensor,
                            scale,
                            hadamard_group_size,
                        )
                    else:
                        raise RuntimeError(f"MiniMax-H3 ConvRot tensor {raw_key} maps to multiple targets unexpectedly")
                    continue

                if not torch.is_floating_point(tensor):
                    raise RuntimeError(
                        f"MiniMax-H3 tensor {raw_key} has unsupported dtype {tensor.dtype}. "
                        "INT4/NVFP4 single-file loading needs a model-specific quantized tensor wrapper."
                    )

                if len(mapped_keys) == 3:
                    if tensor.shape[0] % 3 != 0:
                        raise RuntimeError(f"MiniMax-H3 fused QKV tensor {raw_key} cannot be split into q/k/v tensors")
                    qkv_tensors = tensor.split(tensor.shape[0] // 3, dim=0)
                    for mapped_key, qkv_tensor in zip(mapped_keys, qkv_tensors):
                        non_quantized_state_dict[mapped_key] = qkv_tensor.contiguous()
                elif len(mapped_keys) == 1:
                    non_quantized_state_dict[mapped_keys[0]] = tensor
                else:
                    raise RuntimeError(f"MiniMax-H3 tensor {raw_key} maps to multiple targets unexpectedly")

        rope_inv_freq = non_quantized_state_dict.pop("rope.inv_freq", None)
        if rope_inv_freq is not None and tuple(rope_inv_freq.shape) != tuple(model.rope.inv_freq.shape):
            raise RuntimeError(
                f"MiniMax-H3 tensor rope.inv_freq has shape {tuple(rope_inv_freq.shape)}, "
                f"expected {tuple(model.rope.inv_freq.shape)}"
            )

        expected_quantized_keys = set(quantized_weights)
        keep_fp32_patterns = tuple(getattr(cls, "_keep_in_fp32_modules", ()))
        fp8_weight_keys = {key for key in fp8_state_dict if key.endswith(".weight")}
        for key in sorted(fp8_weight_keys):
            tensor = fp8_state_dict[key]
            if key not in expected_state_dict:
                raise RuntimeError(f"MiniMax-H3 FP8 checkpoint has unexpected tensor: {key}")
            expected_tensor = expected_state_dict[key]
            if tuple(tensor.shape) != tuple(expected_tensor.shape):
                raise RuntimeError(
                    f"MiniMax-H3 FP8 tensor {key} has shape {tuple(tensor.shape)}, expected {tuple(expected_tensor.shape)}"
                )
            scale_key = f"{key.removesuffix('.weight')}.weight_scale"
            scale = fp8_state_dict[scale_key].to(torch.float32)
            out_features = expected_tensor.shape[0]
            if scale.ndim == 0 or scale.numel() == 1:
                scale = scale.reshape(1).expand(out_features).contiguous()
            elif tuple(scale.shape) == (out_features, 1):
                scale = scale.reshape(out_features).contiguous()
            elif tuple(scale.shape) != (out_features,):
                raise RuntimeError(
                    f"MiniMax-H3 FP8 tensor {scale_key} has shape {tuple(scale.shape)}, expected ({out_features},)"
                )
            fp8_state_dict[scale_key] = scale

        for key, tensor in list(non_quantized_state_dict.items()):
            if key not in expected_state_dict:
                raise RuntimeError(f"MiniMax-H3 checkpoint has unexpected tensor: {key}")
            expected_tensor = expected_state_dict[key]
            if tuple(tensor.shape) != tuple(expected_tensor.shape):
                raise RuntimeError(
                    f"MiniMax-H3 tensor {key} has shape {tuple(tensor.shape)}, expected {tuple(expected_tensor.shape)}"
                )
            if model.use_adaln_curves and _is_reduced_adaln_projection_key(key):
                tensor = tensor.to(torch.float32)
            elif torch_dtype is not None and not any(pattern in key for pattern in keep_fp32_patterns):
                tensor = tensor.to(torch_dtype)
            non_quantized_state_dict[key] = tensor

        if fp8_state_dict:
            from simpletuner.helpers.models.ideogram.quantized_loading import swap_linears_to_fp8

            with torch.device("meta"):
                swap_linears_to_fp8(model, fp8_state_dict, compute_dtype=torch_dtype or torch.bfloat16)

        load_state_dict = {**non_quantized_state_dict, **fp8_state_dict}
        missing, unexpected = model.load_state_dict(load_state_dict, strict=False, assign=True)
        real_missing = [key for key in missing if key not in expected_quantized_keys and key != "rope.inv_freq"]
        if real_missing or unexpected:
            raise RuntimeError(
                "MiniMax-H3 checkpoint does not match transformer architecture. "
                f"Missing: {len(real_missing)}, Unexpected: {len(unexpected)}"
            )

        hadamard_group_sizes: set[int] = set()
        if quantized_weights:
            from simpletuner.helpers.models.z_image.quantized_loading import _wrap_convrot_linear

            for weight_key, (
                weight,
                scale,
                hadamard_group_size,
            ) in quantized_weights.items():
                if weight_key not in expected_state_dict:
                    raise RuntimeError(f"MiniMax-H3 ConvRot checkpoint has unexpected tensor: {weight_key}")
                expected_tensor = expected_state_dict[weight_key]
                if tuple(weight.shape) != tuple(expected_tensor.shape):
                    raise RuntimeError(
                        f"MiniMax-H3 ConvRot tensor {weight_key} has shape {tuple(weight.shape)}, "
                        f"expected {tuple(expected_tensor.shape)}"
                    )
                hadamard_group_sizes.add(hadamard_group_size)
                _wrap_convrot_linear(
                    model,
                    weight_key.removesuffix(".weight"),
                    weight,
                    scale,
                    result_dtype=torch_dtype or torch.bfloat16,
                    hadamard_group_size=hadamard_group_size,
                )
            if len(hadamard_group_sizes) != 1:
                raise RuntimeError(
                    f"MiniMax-H3 ConvRot checkpoint uses multiple Hadamard group sizes: {sorted(hadamard_group_sizes)}"
                )
            group_size = hadamard_group_sizes.pop()
            model.quantization_method = "minimax_h3_comfy_convrot_sdnq"
            model.quantization_config = {
                "quant_method": "sdnq_training",
                "weights_dtype": "int8",
                "quantized_matmul_dtype": "int8",
                "use_hadamard": True,
                "hadamard_group_size": group_size,
                "group_size": -1,
                "source_format": "comfy_minimax_h3_convrot",
            }
        elif fp8_state_dict:
            model.quantization_method = "minimax_h3_comfy_fp8"
            model.quantization_config = {
                "quant_method": "fp8_weight_only",
                "weights_dtype": "float8_e4m3fn",
                "source_format": "comfy_minimax_h3_fp8",
            }

        if rope_inv_freq is not None:
            _set_module_buffer(model, "rope.inv_freq", rope_inv_freq.to(torch.float32))
        elif model.rope.inv_freq.is_meta:
            inv_freq = MiniMaxH3RotaryPosEmbed(
                rope_freq_dim=inferred_config["rope_freq_dim"],
                rope_theta=getattr(model.config, "rope_theta", 10000.0),
            ).inv_freq
            _set_module_buffer(model, "rope.inv_freq", inv_freq)

        return model

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        packed_valid_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        skip_layers: list[int] | None = None,
        force_keep_mask: torch.Tensor | None = None,
        timestep_sign: torch.Tensor | None = None,
        r_timestep: torch.Tensor | None = None,
        hidden_states_buffer: dict[str, torch.Tensor] | None = None,
        output_hidden_states: bool = False,
        hidden_state_layer: int | None = None,
        video_hidden_shape: tuple[int, int, int] | None = None,
        num_condition_video_rows: int = 0,
        num_condition_audio_rows: int = 0,
        minimax_h3_reference_mode: str | H3_REFERENCE_MODE | None = None,
        return_dict: bool = True,
    ) -> MiniMaxH3TransformerOutput | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, num_video_tokens, in_channels * prod(patch_size))`):
                Patchified video latent rows — conditioning rows and target rows — ordered as they appear in the packed
                sequence, i.e. matching `video_indices`.
            audio_hidden_states (`torch.Tensor` of shape `(batch_size, num_audio_tokens, audio_in_channels)`):
                Audio latent rows, ordered to match `audio_indices`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, num_text_tokens, text_dim)`):
                Text conditioning, ordered to match `text_indices`.
            timestep (`torch.Tensor` of shape `(num_timesteps,)`):
                The *distinct* timestep values present in the packed sequence, in `[0, 1]` and unscaled. One forward
                serves rows at different noise levels (target video, target audio, conditioning rows).
            timestep_indices (`torch.Tensor` of shape `(seq_len,)` or `(batch_size, seq_len)`):
                For every row of the packed sequence, the index of its timestep in `timestep`.
            token_tags (`torch.Tensor` of shape `(seq_len,)`):
                For every row of the packed sequence, its modality: `0` video, `1` text, `2` audio, `-1` padding.
                Padding rows form their own attention document and never reach the outputs.
            position_ids (`torch.Tensor` of shape `(seq_len, 3)` or `(batch_size, seq_len, 3)`):
                The `(t, h, w)` rotary coordinates of every row of the packed sequence. Batched coordinates preserve
                each sample's unpadded text-length origin.
            video_indices (`torch.Tensor` of shape `(num_video_tokens,)`):
                Positions of the video rows in the packed sequence.
            audio_indices (`torch.Tensor` of shape `(num_audio_tokens,)`):
                Positions of the audio rows in the packed sequence.
            text_indices (`torch.Tensor` of shape `(num_text_tokens,)`):
                Positions of the text rows in the packed sequence.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that, if specified, may carry a `scale` entry which is applied to the LoRA layers.
            minimax_h3_reference_mode (`str` or `H3_REFERENCE_MODE`, *optional*):
                `vanilla` keeps the stock full packed self-attention path. `cached_kv` treats text plus conditioning
                video/audio rows as static memory during no-grad inference, prevents those rows from attending to
                denoised rows, and reuses their per-block K/V projections across calls.
            timestep_sign (`torch.Tensor`, *optional*):
                Signed-time marker for TwinFlow-compatible training. Requires `enable_time_sign_embed=True`.
            r_timestep (`torch.Tensor`, *optional*):
                FlowMap/AnyFlow reference timestep. Requires `enable_flowmap_time_conditioning()`.
            return_dict (`bool`, defaults to `True`):
                Whether to return a [`MiniMaxH3TransformerOutput`] instead of a plain tuple.

        Returns:
            [`MiniMaxH3TransformerOutput`] or `tuple`:
                The video velocity of shape `(batch_size, num_video_tokens, in_channels * prod(patch_size))` and the
                audio velocity of shape `(batch_size, num_audio_tokens, audio_in_channels)`, in the row order of
                `video_indices` and `audio_indices`.
        """
        # `attention_kwargs` is consumed by the `@apply_lora_scale` decorator on this method.
        valid_position_ids = position_ids.ndim == 2 or (
            position_ids.ndim == 3 and position_ids.shape[0] == hidden_states.shape[0]
        )
        if not valid_position_ids or position_ids.shape[-1] != 3:
            raise ValueError(
                "`position_ids` must be `(seq_len, 3)` or `(batch, seq_len, 3)`, got " f"{list(position_ids.shape)}."
            )
        sequence_length = position_ids.shape[-2]
        valid_timestep_indices = timestep_indices.shape == (sequence_length,) or timestep_indices.shape == (
            hidden_states.shape[0],
            sequence_length,
        )
        if token_tags.shape != (sequence_length,) or not valid_timestep_indices:
            raise ValueError(
                "`token_tags` must be `(seq_len,)` and `timestep_indices` must be `(seq_len,)` or "
                "`(batch, seq_len)` tensors matching `position_ids`, got "
                f"{list(token_tags.shape)} and {list(timestep_indices.shape)} for seq_len={sequence_length}."
            )
        if packed_valid_mask is not None:
            packed_valid_mask = packed_valid_mask.to(device=hidden_states.device, dtype=torch.bool)
            if packed_valid_mask.shape != (hidden_states.shape[0], sequence_length):
                raise ValueError(
                    "`packed_valid_mask` must be `(batch, seq_len)`, got "
                    f"{list(packed_valid_mask.shape)} for batch={hidden_states.shape[0]}, seq_len={sequence_length}."
                )

        parallel_config = getattr(self, "_parallel_config", None)
        cp_config = context_parallel_config(parallel_config)
        cp_active = cp_config is not None
        if cp_active and (position_ids.ndim == 3 or timestep_indices.ndim == 2 or packed_valid_mask is not None):
            raise ValueError("MiniMax-H3 batched packed layouts are not supported with context parallelism yet.")
        sparse_config = self._h3_sparse_attention_config
        if sparse_config.enabled and cp_active:
            ring_degree = int(getattr(cp_config, "ring_degree", 1) or 1)
            ulysses_degree = int(getattr(cp_config, "ulysses_degree", 1) or 1)
            if ring_degree != 1 or ulysses_degree <= 1:
                raise ValueError(
                    "MiniMax-H3 sparse attention supports context parallelism only with "
                    "context_parallel_strategy=alltoall."
                )
        unpadded_sequence_length = sequence_length
        if cp_active:
            cp_degree = int(getattr(cp_config, "ring_degree", 1) or 1) * int(getattr(cp_config, "ulysses_degree", 1) or 1)
            position_ids, token_tags, timestep_indices = _pad_h3_context_parallel_layout(
                position_ids,
                token_tags,
                timestep_indices,
                cp_degree,
            )
            sequence_length = position_ids.shape[0]

        rotary_emb = self.rope(position_ids)

        # 1. Project each modality and scatter the rows into the packed sequence buffer. The checkpoint is
        # mixed-precision (the two patch projections are float32 while `context_embedder` and the block stack are
        # bfloat16 — see `_keep_in_fp32_modules`), so every input is aligned with its projection's parameter dtype,
        # mirroring the reference's explicit casts. The text stream sets the dtype of the packed sequence.
        video_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
        text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
        self.token_refiner.gradient_checkpointing = self.gradient_checkpointing
        text_attention_mask = None
        if packed_valid_mask is not None:
            text_is_pad = ~packed_valid_mask.index_select(1, text_indices)
            if bool(text_is_pad.any()):
                text_attention_mask = text_is_pad[:, None, :, None] == text_is_pad[:, None, None, :]
        text_embeds = self.token_refiner(
            text_embeds,
            attention_mask=text_attention_mask,
            checkpoint_backend=self.gradient_checkpointing_backend,
            offload_attention=self.gradient_checkpointing_offload_attention,
        )

        hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
        hidden_states = hidden_states.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        hidden_states = hidden_states.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))

        # 2. One timestep embedding per distinct noise level. `temb` is shared by all AdaLN projections, which are
        # bfloat16 in the checkpoint while `time_embedder` is float32, so it stays at the time embedder's precision:
        # each AdaLN module applies its own activation to it and casts to its projection's dtype afterwards.
        temb = self._time_embedding(timestep, timestep_sign=timestep_sign, r_timestep=r_timestep)

        # 3. Row -> AdaLN table row. `clamp(min=0)` mirrors the reference, where padding rows carry the tag `-1`; the
        # clamp keeps the `-1` from indexing backwards (padding rows never reach the outputs, which are selected by
        # `video_indices` / `audio_indices`).
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)

        # 4. Padding rows (tag `-1`) must not exchange attention with live rows: the reference keeps the padding tail
        # as a separate attention document (`cu_seqlens = [0, used, S]`). A boolean mask that pairs live rows with live
        # rows and padding rows with padding rows reproduces that split exactly. Padless sequences keep `None` so the
        # unmasked fast paths (flash & co.) stay available. Under CP, a key-only mask is sufficient: padding queries
        # are discarded, and excluding padding keys keeps every live row exact.
        attention_mask = None
        is_pad = token_tags < 0
        if packed_valid_mask is not None:
            is_pad = (~packed_valid_mask) | is_pad.unsqueeze(0)
        if bool(is_pad.any()):
            if cp_active:
                attention_mask = (~is_pad)[None, None, None, :]
            elif is_pad.ndim == 2:
                attention_mask = is_pad[:, None, :, None] == is_pad[:, None, None, :]
            else:
                attention_mask = is_pad[None, :] == is_pad[:, None]

        grad_enabled = torch.is_grad_enabled()
        if grad_enabled and self._h3_reference_kv_cache:
            self.clear_h3_reference_kv_cache()
        reference_mode = resolve_h3_reference_mode(minimax_h3_reference_mode)
        reference_mask = None
        reference_kv_cache = None
        reference_kv_stats = None
        if reference_mode is H3_REFERENCE_MODE.CachedKV and not grad_enabled:
            if packed_valid_mask is not None:
                raise ValueError("MiniMax-H3 CachedKV reference mode does not support batched text padding yet.")
            if cp_active:
                raise ValueError("MiniMax-H3 CachedKV reference mode does not support context_parallel_size yet.")
            reference_mask = self._build_reference_mask(
                sequence_length,
                device=token_tags.device,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
                num_condition_video_rows=num_condition_video_rows,
                num_condition_audio_rows=num_condition_audio_rows,
            )
            if reference_mask is not None:
                live_mask = token_tags >= 0
                dynamic_mask = live_mask & ~reference_mask
                if bool(dynamic_mask.any()):
                    reference_attention_mask = torch.ones(
                        (sequence_length, sequence_length),
                        dtype=torch.bool,
                        device=token_tags.device,
                    )
                    reference_rows = reference_mask.nonzero(as_tuple=False).flatten()
                    dynamic_rows = dynamic_mask.nonzero(as_tuple=False).flatten()
                    reference_attention_mask[reference_rows[:, None], dynamic_rows[None, :]] = False
                    attention_mask = (
                        reference_attention_mask
                        if attention_mask is None
                        else attention_mask.to(device=token_tags.device) & reference_attention_mask
                    )
                reference_kv_cache = self._h3_reference_kv_cache
                reference_kv_stats = self._h3_reference_kv_stats
        routes = self._tread_routes or []
        router = self._tread_router
        use_routing = self.training and len(routes) > 0 and torch.is_grad_enabled()
        if cp_active and use_routing:
            raise ValueError("MiniMax-H3 TREAD routing is not supported together with context_parallel_size.")
        if use_routing and router is None:
            raise ValueError("TREAD routing requested but no router has been configured. Call set_router before training.")
        if use_routing and attention_mask is not None:
            raise ValueError("MiniMax-H3 TREAD routing does not support padded packed sequences.")
        if routes:
            total_layers = len(self.transformer_blocks)

            def _to_pos(idx):
                return idx if idx >= 0 else total_layers + idx

            routes = [
                {
                    **route,
                    "start_layer_idx": _to_pos(route["start_layer_idx"]),
                    "end_layer_idx": _to_pos(route["end_layer_idx"]),
                }
                for route in routes
            ]

        sparse_attention_layout = None
        if sparse_config.enabled:
            if video_hidden_shape is None:
                raise ValueError("MiniMax-H3 sparse attention requires `video_hidden_shape`.")
            target_shape = tuple(int(dim) for dim in video_hidden_shape)
            target_video_rows = int(video_indices.shape[0]) - int(num_condition_video_rows or 0)
            expected_target_rows = math.prod(target_shape)
            if target_video_rows != expected_target_rows:
                raise ValueError(
                    "MiniMax-H3 sparse attention target shape does not match the packed target-video rows: "
                    f"shape={target_shape} ({expected_target_rows} rows), packed={target_video_rows}."
                )
            sparse_attention_layout = MiniMaxH3SparseAttentionLayout(
                target_start=unpadded_sequence_length - target_video_rows,
                target_shape=target_shape,
                trailing_padding=sequence_length - unpadded_sequence_length,
                packed_valid_mask=packed_valid_mask,
            )
            if use_routing:
                raise ValueError("MiniMax-H3 sparse attention cannot be combined with TREAD routing yet.")

        need_hidden_state_capture = hidden_states_buffer is not None or output_hidden_states
        captured_frame_hidden = None
        if cp_active and need_hidden_state_capture:
            raise ValueError("MiniMax-H3 hidden-state capture is not supported together with context_parallel_size.")
        if need_hidden_state_capture:
            if video_hidden_shape is None:
                raise ValueError("MiniMax-H3 hidden-state capture requires `video_hidden_shape`.")
            post_patch_frames, post_patch_height, post_patch_width = tuple(int(dim) for dim in video_hidden_shape)
            target_video_rows = int(video_indices.shape[0]) - int(num_condition_video_rows or 0)
            expected_video_rows = post_patch_frames * post_patch_height * post_patch_width
            if target_video_rows != expected_video_rows:
                raise ValueError(
                    "MiniMax-H3 hidden-state capture expected "
                    f"{expected_video_rows} generated video rows from shape {video_hidden_shape}, "
                    f"but got {target_video_rows}."
                )
            if use_routing:
                raise ValueError("MiniMax-H3 hidden-state capture is not compatible with TREAD-routed partial sequences.")

        capture_layer = int(hidden_state_layer) if hidden_state_layer is not None else None
        block_rotary_emb = rotary_emb
        block_adaln_indices = adaln_indices
        block_reference_mask = reference_mask
        block_sparse_attention_layout = sparse_attention_layout

        def capture_layer_hidden(block_idx: int, block_hidden_states: torch.Tensor) -> None:
            nonlocal captured_frame_hidden, output_hidden_states
            if not need_hidden_state_capture:
                return
            layer_video_hidden = block_hidden_states.index_select(1, video_indices)
            if num_condition_video_rows:
                layer_video_hidden = layer_video_hidden[:, int(num_condition_video_rows) :, :]
            layer_video_hidden = layer_video_hidden.reshape(
                layer_video_hidden.shape[0],
                post_patch_frames,
                post_patch_height * post_patch_width,
                layer_video_hidden.shape[-1],
            )
            if hidden_states_buffer is not None:
                hidden_states_buffer[f"layer_{block_idx}"] = layer_video_hidden
            if output_hidden_states and (capture_layer is None or block_idx == capture_layer):
                captured_frame_hidden = layer_video_hidden
                if capture_layer is not None:
                    output_hidden_states = False

        skip_set = set(skip_layers) if skip_layers is not None else set()
        musubi_manager = self._musubi_block_swap
        musubi_offload_active = False
        if musubi_manager is not None:
            musubi_offload_active = musubi_manager.activate(self.transformer_blocks, hidden_states.device, grad_enabled)

        use_segmented_checkpointing = (
            grad_enabled
            and self.gradient_checkpointing
            and self.gradient_checkpointing_interval is not None
            and self.gradient_checkpointing_interval > 1
            and not self.gradient_checkpointing_backend.endswith("-ffn")
            and not use_routing
            and not skip_set
            and not musubi_offload_active
            and not need_hidden_state_capture
        )
        if use_segmented_checkpointing:
            if self.gradient_checkpointing_backend.startswith("unsloth"):
                from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

                checkpoint_fn = offloaded_checkpoint
            else:
                checkpoint_fn = torch.utils.checkpoint.checkpoint

            def run_h3_block(_block_index, segment_block, segment_hidden_states):
                return segment_block(
                    segment_hidden_states,
                    temb,
                    block_adaln_indices,
                    block_rotary_emb,
                    attention_mask,
                    reference_mask=block_reference_mask,
                    reference_kv_cache=reference_kv_cache,
                    reference_kv_stats=reference_kv_stats,
                    sparse_attention_layout=block_sparse_attention_layout,
                    offload_attention=self.gradient_checkpointing_offload_attention,
                )

            (hidden_states,) = checkpoint_sequential_state(
                self.transformer_blocks,
                self.gradient_checkpointing_interval,
                (hidden_states,),
                run_h3_block,
                checkpoint_fn,
                {"use_reentrant": False},
                segment_stride=self.gradient_checkpointing_segment_stride,
            )
        else:
            route_ptr = 0
            routing_now = False
            tread_mask_info = None
            saved_tokens = None
            current_rotary_emb = block_rotary_emb
            current_adaln_indices = block_adaln_indices

            for block_idx, block in enumerate(self.transformer_blocks):
                if use_routing and route_ptr < len(routes) and block_idx == routes[route_ptr]["start_layer_idx"]:
                    mask_ratio = routes[route_ptr]["selection_ratio"]
                    base_force_keep = (
                        token_tags.ne(0).unsqueeze(0).expand(hidden_states.shape[0], -1).to(device=hidden_states.device)
                    )
                    if force_keep_mask is not None:
                        force_keep = force_keep_mask.to(device=hidden_states.device, dtype=torch.bool)
                        if force_keep.ndim == 1:
                            force_keep = force_keep.unsqueeze(0).expand(hidden_states.shape[0], -1)
                        if force_keep.shape != base_force_keep.shape:
                            raise ValueError(
                                "MiniMax-H3 force_keep_mask must have shape `(seq_len,)` or `(batch, seq_len)`, got "
                                f"{list(force_keep.shape)} for expected {list(base_force_keep.shape)}."
                            )
                        base_force_keep = base_force_keep | force_keep

                    tread_mask_info = router.get_mask(
                        hidden_states,
                        mask_ratio=mask_ratio,
                        force_keep=base_force_keep,
                    )
                    saved_tokens = hidden_states.clone()
                    hidden_states = router.start_route(hidden_states, tread_mask_info)
                    keep_len = hidden_states.shape[1]
                    current_rotary_emb = self._route_rotary(rotary_emb, tread_mask_info, keep_len)
                    current_adaln_indices = self._route_token_tensor(
                        adaln_indices, tread_mask_info, keep_len, hidden_states.shape[0]
                    )
                    routing_now = True

                if block_idx in skip_set:
                    continue

                if musubi_offload_active and musubi_manager.is_managed_block(block_idx):
                    musubi_manager.stream_in(block, hidden_states.device)

                if grad_enabled and should_checkpoint_block(
                    block_idx,
                    self.gradient_checkpointing,
                    self.gradient_checkpointing_interval,
                    self.gradient_checkpointing_segment_stride,
                ):
                    if self.gradient_checkpointing_backend.startswith("unsloth"):
                        from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

                        checkpoint_fn = offloaded_checkpoint
                    else:
                        checkpoint_fn = torch.utils.checkpoint.checkpoint

                    if self.gradient_checkpointing_backend.endswith("-ffn"):
                        hidden_states = block(
                            hidden_states,
                            temb,
                            current_adaln_indices,
                            current_rotary_emb,
                            attention_mask,
                            checkpoint_ffn=True,
                            checkpoint_fn=checkpoint_fn,
                            offload_attention=self.gradient_checkpointing_offload_attention,
                            reference_mask=block_reference_mask,
                            reference_kv_cache=reference_kv_cache,
                            reference_kv_stats=reference_kv_stats,
                            sparse_attention_layout=block_sparse_attention_layout,
                        )
                    else:

                        def run_checkpointed_block(
                            checkpoint_hidden_states,
                            checkpoint_block=block,
                            checkpoint_adaln_indices=current_adaln_indices,
                            checkpoint_rotary_emb=current_rotary_emb,
                        ):
                            return checkpoint_block(
                                checkpoint_hidden_states,
                                temb,
                                checkpoint_adaln_indices,
                                checkpoint_rotary_emb,
                                attention_mask,
                                offload_attention=self.gradient_checkpointing_offload_attention,
                                reference_mask=block_reference_mask,
                                reference_kv_cache=reference_kv_cache,
                                reference_kv_stats=reference_kv_stats,
                                sparse_attention_layout=block_sparse_attention_layout,
                            )

                        hidden_states = checkpoint_fn(run_checkpointed_block, hidden_states, use_reentrant=False)
                else:
                    hidden_states = block(
                        hidden_states,
                        temb,
                        current_adaln_indices,
                        current_rotary_emb,
                        attention_mask,
                        offload_attention=self.gradient_checkpointing_offload_attention,
                        reference_mask=block_reference_mask,
                        reference_kv_cache=reference_kv_cache,
                        reference_kv_stats=reference_kv_stats,
                        sparse_attention_layout=block_sparse_attention_layout,
                    )

                if musubi_offload_active and musubi_manager.is_managed_block(block_idx):
                    musubi_manager.stream_out(block)

                if routing_now and route_ptr < len(routes) and block_idx == routes[route_ptr]["end_layer_idx"]:
                    hidden_states = router.end_route(hidden_states, tread_mask_info, original_x=saved_tokens)
                    current_rotary_emb = block_rotary_emb
                    current_adaln_indices = block_adaln_indices
                    routing_now = False
                    route_ptr += 1
                capture_layer_hidden(block_idx, hidden_states)
        # 5. Both heads run over every row, then the rows of each modality are selected. The heads are listed in
        # `_keep_in_fp32_modules`, so they stay float32 while the block stack runs in the requested `torch_dtype`;
        # align the activation with their parameter dtype.
        hidden_states = self.norm_out(hidden_states, temb, timestep_indices).to(self.proj_out.weight.dtype)
        video_output = _gather_h3_context_parallel_output(self.proj_out(hidden_states), cp_config, dim=1).index_select(
            1, video_indices.to(hidden_states.device)
        )
        if audio_indices.numel():
            audio_output = _gather_h3_context_parallel_output(
                self.audio_proj_out(hidden_states), cp_config, dim=1
            ).index_select(1, audio_indices.to(hidden_states.device))
        else:
            audio_output = hidden_states.new_empty(
                hidden_states.shape[0],
                0,
                self.audio_proj_out.out_features,
            )

        if not return_dict:
            if captured_frame_hidden is not None:
                return (video_output, audio_output, captured_frame_hidden)
            return (video_output, audio_output)
        return MiniMaxH3TransformerOutput(
            sample=video_output,
            audio_sample=audio_output,
            crepa_hidden_states=captured_frame_hidden,
        )
