# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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

import os
from typing import Any, Callable, Dict, Optional, Union

import torch
from diffusers.loaders.lora_base import LoraBaseMixin, _fetch_state_dict
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline
from diffusers.utils import USE_PEFT_BACKEND, is_peft_version, logging

from simpletuner.helpers.training.lora_format import (
    PEFTLoRAFormat,
    _resolve_alpha_for_module,
    collect_lora_alphas,
    collect_lora_ranks,
    detect_state_dict_format,
    normalize_lora_format,
    peft_lora_config_kwargs_from_state_dict,
    synthesize_missing_lora_alphas_from_ranks,
)

from .transformer import _map_minimax_h3_comfy_key_to_diffusers

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


_LORA_A_SUFFIXES = (".lora_A.weight", ".lora.down.weight", ".lora_down.weight")
_LORA_B_SUFFIXES = (".lora_B.weight", ".lora.up.weight", ".lora_up.weight")
_COMPONENT_PREFIXES = ("text_encoder.", "text_encoder_2.", "controlnet.", "unet.", "transformer.", "transformer_ref.")
_COMFY_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
_MINIMAX_H3_NATIVE_LORA_PREFIXES = (
    "blocks.",
    "token_refiner.blocks.",
    "final_layer.",
    "video_patch_proj.",
    "audio_patch_proj.",
    "condition_proj.",
    "time_embedder.",
)
MINIMAX_H3_SWIGLU_GATE_FIRST_METADATA_KEY = "swiglu_gate_first"
MINIMAX_H3_FLOWMAP_GATE_METADATA_KEY = "flowmap_gate_value"
MINIMAX_H3_FLOWMAP_DELTATIME_METADATA_KEY = "flowmap_deltatime_type"
_MINIMAX_H3_CUSTOM_ADAPTER_METADATA_KEYS = {
    MINIMAX_H3_SWIGLU_GATE_FIRST_METADATA_KEY,
    MINIMAX_H3_FLOWMAP_GATE_METADATA_KEY,
    MINIMAX_H3_FLOWMAP_DELTATIME_METADATA_KEY,
}


def _strip_comfy_lora_prefix(key: str) -> str:
    for prefix in _COMFY_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def _split_lora_suffix(key: str) -> tuple[str, str] | None:
    for suffix in _LORA_A_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], ".lora.down.weight"
    for suffix in _LORA_B_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], ".lora.up.weight"
    return None


def _is_minimax_h3_native_lora_state_dict(state_dict: Dict[str, Any]) -> bool:
    for raw_key in state_dict:
        key = _strip_comfy_lora_prefix(raw_key)
        if ".lora" not in key:
            continue
        if key.startswith(_MINIMAX_H3_NATIVE_LORA_PREFIXES):
            return True
    return False


def _minimax_h3_swiglu_gate_first_from_metadata(
    metadata: Optional[dict],
    *,
    target_prefix: str,
) -> Optional[bool]:
    if not metadata:
        return None

    candidate_keys = (
        f"{target_prefix}.{MINIMAX_H3_SWIGLU_GATE_FIRST_METADATA_KEY}",
        f"transformer.{MINIMAX_H3_SWIGLU_GATE_FIRST_METADATA_KEY}",
        f"transformer_ref.{MINIMAX_H3_SWIGLU_GATE_FIRST_METADATA_KEY}",
        MINIMAX_H3_SWIGLU_GATE_FIRST_METADATA_KEY,
    )
    for key in candidate_keys:
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise ValueError(f"MiniMax-H3 LoRA metadata `{key}` must be a boolean, received {value!r}.")
    return None


def _minimax_h3_adapter_metadata_value(metadata: Optional[dict], key: str, *, target_prefix: str):
    if not metadata:
        return None
    for candidate in (f"{target_prefix}.{key}", f"transformer.{key}", f"transformer_ref.{key}", key):
        if candidate in metadata:
            return metadata[candidate]
    return None


def _minimax_h3_peft_adapter_metadata(metadata: Optional[dict], *, target_prefix: str) -> Optional[dict]:
    if not metadata:
        return None
    filtered = {}
    for key, value in metadata.items():
        if key.rsplit(".", 1)[-1] in _MINIMAX_H3_CUSTOM_ADAPTER_METADATA_KEYS:
            continue
        for source_prefix in ("transformer.", "transformer_ref."):
            if key.startswith(source_prefix):
                key = f"{target_prefix}.{key.removeprefix(source_prefix)}"
                break
        else:
            key = f"{target_prefix}.{key}"
        filtered[key] = value
    return filtered or None


def _convert_minimax_h3_diffusers_swiglu_lora_layout(
    state_dict: Dict[str, Any],
    *,
    source_gate_first: bool,
    target_gate_first: bool,
) -> Dict[str, Any]:
    if source_gate_first == target_gate_first:
        return state_dict

    converted = dict(state_dict)
    for key, value in state_dict.items():
        if ".ff.net.0.proj" not in key or not key.endswith(_LORA_B_SUFFIXES) or not torch.is_tensor(value):
            continue
        if value.ndim == 0 or value.shape[0] % 2 != 0:
            raise ValueError(f"MiniMax-H3 SwiGLU LoRA tensor `{key}` cannot be split into gate/value rows.")
        first, second = value.chunk(2, dim=0)
        converted[key] = torch.cat((second, first), dim=0).contiguous()
    return converted


def _map_comfy_lora_module(module_key: str, target_prefix: str) -> list[str]:
    if module_key.startswith(f"{target_prefix}."):
        return [module_key]
    if module_key.startswith(_COMPONENT_PREFIXES):
        return [module_key]

    mapped_weight_keys = _map_minimax_h3_comfy_key_to_diffusers(f"{module_key}.weight")
    mapped_modules = [key.removesuffix(".weight") for key in mapped_weight_keys]
    return [f"{target_prefix}.{module}" for module in mapped_modules]


def _convert_minimax_h3_comfy_lora_to_diffusers(
    state_dict: Dict[str, Any],
    *,
    target_prefix: str,
    target_swiglu_gate_first: bool = False,
) -> tuple[Dict[str, Any], Dict[str, float]]:
    converted: Dict[str, Any] = {}
    network_alphas: Dict[str, float] = {}

    lora_pairs: dict[str, dict[str, torch.Tensor]] = {}
    for raw_key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        split_key = _split_lora_suffix(_strip_comfy_lora_prefix(raw_key))
        if split_key is None:
            continue
        module_key, target_suffix = split_key
        lora_pairs.setdefault(module_key, {})[target_suffix] = value

    split_qkv: dict[str, dict[str, list[torch.Tensor] | list[int]]] = {}
    for module_key, tensors in lora_pairs.items():
        down = tensors.get(".lora.down.weight")
        up = tensors.get(".lora.up.weight")
        mapped_modules = _map_comfy_lora_module(module_key, target_prefix)
        if (
            len(mapped_modules) != 3
            or down is None
            or up is None
            or down.ndim != 2
            or up.ndim != 2
            or up.shape[0] % 3 != 0
            or up.shape[1] != down.shape[0]
        ):
            continue

        up_chunks = list(up.chunk(3, dim=0))
        total_rank = int(down.shape[0])
        rank_indices = None

        if total_rank % 3 == 0:
            rank = total_rank // 3
            equal_indices = [torch.arange(index * rank, (index + 1) * rank, device=up.device) for index in range(3)]
            is_block_diagonal = True
            for projection_index, chunk in enumerate(up_chunks):
                outside = torch.ones(total_rank, dtype=torch.bool, device=up.device)
                outside[equal_indices[projection_index]] = False
                if torch.count_nonzero(chunk[:, outside]).item() != 0:
                    is_block_diagonal = False
                    break
            if is_block_diagonal:
                rank_indices = equal_indices

        if rank_indices is None:
            active_columns = [torch.any(chunk != 0, dim=0) for chunk in up_chunks]
            active_count = torch.stack(active_columns).sum(dim=0)
            if torch.all(active_count == 1).item():
                inferred_indices = [torch.nonzero(mask, as_tuple=False).flatten() for mask in active_columns]
                if all(indices.numel() > 0 for indices in inferred_indices):
                    rank_indices = inferred_indices

        if rank_indices is None:
            continue

        split_qkv[module_key] = {
            ".lora.down.weight": [down.index_select(0, indices).contiguous() for indices in rank_indices],
            ".lora.up.weight": [
                chunk.index_select(1, indices).contiguous() for chunk, indices in zip(up_chunks, rank_indices)
            ],
            "ranks": [int(indices.numel()) for indices in rank_indices],
        }

    for raw_key, value in state_dict.items():
        key = _strip_comfy_lora_prefix(raw_key)
        split_key = _split_lora_suffix(key)
        if split_key is None:
            if key.endswith(".alpha"):
                module_key = key[: -len(".alpha")]
                mapped_modules = _map_comfy_lora_module(module_key, target_prefix)
                qkv_split = split_qkv.get(module_key)
                for index, mapped_module in enumerate(mapped_modules):
                    if qkv_split is not None:
                        network_alphas[f"{mapped_module}.alpha"] = float(qkv_split["ranks"][index])
                        continue
                    if torch.is_tensor(value):
                        network_alphas[f"{mapped_module}.alpha"] = float(value.detach().float().cpu().item())
                    else:
                        network_alphas[f"{mapped_module}.alpha"] = float(value)
                continue
            if key.startswith(f"{target_prefix}.") or key.startswith(_COMPONENT_PREFIXES):
                converted[key] = value
            else:
                converted[f"{target_prefix}.{key}"] = value
            continue

        module_key, target_suffix = split_key
        mapped_modules = _map_comfy_lora_module(module_key, target_prefix)
        qkv_split = split_qkv.get(module_key)
        if qkv_split is not None:
            values = qkv_split[target_suffix]
        elif len(mapped_modules) == 3 and target_suffix == ".lora.up.weight":
            if value.shape[0] % 3 != 0:
                raise ValueError(f"MiniMax-H3 fused QKV LoRA tensor {raw_key} cannot be split into q/k/v tensors.")
            values = value.split(value.shape[0] // 3, dim=0)
        elif (
            target_suffix == ".lora.up.weight"
            and module_key.endswith(".mlp.fc1")
            and not target_swiglu_gate_first
            and torch.is_tensor(value)
        ):
            if value.shape[0] % 2 != 0:
                raise ValueError(f"MiniMax-H3 SwiGLU LoRA tensor {raw_key} cannot be split into gate/value tensors.")
            gate, hidden = value.split(value.shape[0] // 2, dim=0)
            values = (torch.cat([hidden, gate], dim=0).contiguous(),)
        else:
            values = (value,) * len(mapped_modules)

        for mapped_module, mapped_value in zip(mapped_modules, values):
            converted[f"{mapped_module}{target_suffix}"] = (
                mapped_value.contiguous() if torch.is_tensor(mapped_value) else mapped_value
            )

    return converted, network_alphas


def _map_minimax_h3_diffusers_lora_module_to_native(module_key: str) -> str:
    for prefix in ("transformer.", "transformer_ref."):
        if module_key.startswith(prefix):
            module_key = module_key[len(prefix) :]
            break

    direct_map = {
        "proj_in": "video_patch_proj",
        "audio_proj_in": "audio_patch_proj",
        "context_embedder": "condition_proj",
        "time_embedder.linear_1": "time_embedder.proj_in",
        "time_embedder.linear_2": "time_embedder.proj_out",
        "norm_out.norm": "final_layer.norm",
        "norm_out.linear": "final_layer.adaln_proj.linear",
        "proj_out": "final_layer.video_out",
        "audio_proj_out": "final_layer.audio_out",
    }
    for source, target in direct_map.items():
        if module_key == source or module_key.startswith(f"{source}."):
            return f"{target}{module_key[len(source):]}"

    if module_key.startswith("token_refiner.refiner_blocks."):
        module_key = module_key.replace("token_refiner.refiner_blocks.", "token_refiner.blocks.", 1)
    elif module_key.startswith("transformer_blocks."):
        module_key = module_key.replace("transformer_blocks.", "blocks.", 1)
    module_key = module_key.replace(".attn.norm_q", ".attn.q_norm")
    module_key = module_key.replace(".attn.norm_k", ".attn.k_norm")
    module_key = module_key.replace(".attn.to_qkv", ".attn.qkv_proj")
    module_key = module_key.replace(".attn.to_out.0", ".attn.out_proj")
    module_key = module_key.replace(".ff.net.0.proj", ".mlp.fc1")
    module_key = module_key.replace(".ff.net.2", ".mlp.fc2")
    return module_key


def _convert_minimax_h3_diffusers_lora_to_comfyui(
    state_dict: Dict[str, Any],
    *,
    adapter_metadata: Optional[dict] = None,
    source_swiglu_gate_first: bool = False,
) -> Dict[str, Any]:
    """Convert split Diffusers H3 LoRA weights to the native fused ComfyUI layout without approximation."""
    lora_modules: dict[str, dict[str, torch.Tensor]] = {}
    passthrough: dict[str, Any] = {}
    suffixes = {
        ".lora.down.weight": "down",
        ".lora_A.weight": "down",
        ".lora.up.weight": "up",
        ".lora_B.weight": "up",
    }
    for key, value in state_dict.items():
        matched_suffix = next((suffix for suffix in suffixes if key.endswith(suffix)), None)
        if matched_suffix is None:
            if not key.endswith((".alpha", ".lora_alpha")):
                native_key = _map_minimax_h3_diffusers_lora_module_to_native(key)
                passthrough[f"diffusion_model.{native_key}"] = value
            continue
        module_key = key[: -len(matched_suffix)]
        lora_modules.setdefault(module_key, {})[suffixes[matched_suffix]] = value

    converted: dict[str, Any] = dict(passthrough)
    qkv_groups: dict[str, dict[str, tuple[str, dict[str, torch.Tensor]]]] = {}
    regular_modules: list[tuple[str, dict[str, torch.Tensor]]] = []
    for module_key, tensors in lora_modules.items():
        projection = next((name for name in ("to_q", "to_k", "to_v") if module_key.endswith(f".attn.{name}")), None)
        if projection is None:
            regular_modules.append((module_key, tensors))
            continue
        group_key = module_key.removesuffix(f".{projection}")
        qkv_groups.setdefault(group_key, {})[projection] = (module_key, tensors)

    def module_alpha(module_key: str, down: torch.Tensor) -> float:
        stripped_module_key = module_key
        for prefix in ("transformer.", "transformer_ref."):
            if stripped_module_key.startswith(prefix):
                stripped_module_key = stripped_module_key[len(prefix) :]
                break
        alpha = _resolve_alpha_for_module(stripped_module_key, down, adapter_metadata)
        return float(down.shape[0] if alpha is None else alpha)

    for module_key, tensors in regular_modules:
        if "down" not in tensors or "up" not in tensors:
            raise ValueError(f"Incomplete MiniMax-H3 LoRA module `{module_key}` while exporting ComfyUI weights.")
        down, up = tensors["down"], tensors["up"]
        native_module = _map_minimax_h3_diffusers_lora_module_to_native(module_key)
        if native_module.endswith(".mlp.fc1") and not source_swiglu_gate_first:
            if up.shape[0] % 2 != 0:
                raise ValueError(f"MiniMax-H3 SwiGLU LoRA tensor for `{module_key}` has an odd output size.")
            hidden, gate = up.chunk(2, dim=0)
            up = torch.cat((gate, hidden), dim=0).contiguous()
        prefix = f"diffusion_model.{native_module}"
        converted[f"{prefix}.lora_A.weight"] = down
        converted[f"{prefix}.lora_B.weight"] = up
        converted[f"{prefix}.alpha"] = torch.tensor(module_alpha(module_key, down), dtype=torch.float32)

    for group_key, projections in qkv_groups.items():
        ordered = [projections.get(name) for name in ("to_q", "to_k", "to_v")]
        present = [entry for entry in ordered if entry is not None]
        if not present:
            continue
        for module_key, tensors in present:
            if "down" not in tensors or "up" not in tensors:
                raise ValueError(f"Incomplete MiniMax-H3 LoRA module `{module_key}` while exporting fused QKV weights.")

        output_sizes = {int(tensors["up"].shape[0]) for _module_key, tensors in present}
        input_sizes = {int(tensors["down"].shape[1]) for _module_key, tensors in present}
        if len(output_sizes) != 1 or len(input_sizes) != 1:
            raise ValueError(f"MiniMax-H3 QKV LoRA modules under `{group_key}` have incompatible dimensions.")
        output_size = output_sizes.pop()
        input_size = input_sizes.pop()

        shared_down = len(present) == 3 and all(
            torch.equal(present[0][1]["down"], entry[1]["down"]) for entry in present[1:]
        )
        scales = [module_alpha(module_key, tensors["down"]) / tensors["down"].shape[0] for module_key, tensors in present]
        shared_scale = shared_down and all(abs(scales[0] - scale) <= 1e-12 for scale in scales[1:])
        if shared_scale:
            fused_down = present[0][1]["down"]
            fused_up = torch.cat([entry[1]["up"] for entry in present], dim=0)
            fused_alpha = scales[0] * fused_down.shape[0]
        else:
            ranks = [int(tensors["down"].shape[0]) for _module_key, tensors in present]
            total_rank = sum(ranks)
            fused_down = torch.cat([tensors["down"] for _module_key, tensors in present], dim=0)
            fused_up = present[0][1]["up"].new_zeros((3 * output_size, total_rank))
            rank_offset = 0
            for projection_index, entry in enumerate(ordered):
                if entry is None:
                    continue
                module_key, tensors = entry
                rank = int(tensors["down"].shape[0])
                scale = module_alpha(module_key, tensors["down"]) / rank
                fused_up[
                    projection_index * output_size : (projection_index + 1) * output_size,
                    rank_offset : rank_offset + rank,
                ] = (
                    tensors["up"] * scale
                )
                rank_offset += rank
            fused_alpha = float(total_rank)
        if fused_down.shape[1] != input_size:
            raise ValueError(f"MiniMax-H3 fused QKV LoRA under `{group_key}` has an invalid input dimension.")

        native_group = _map_minimax_h3_diffusers_lora_module_to_native(group_key)
        prefix = f"diffusion_model.{native_group}.qkv_proj"
        converted[f"{prefix}.lora_A.weight"] = fused_down.contiguous()
        converted[f"{prefix}.lora_B.weight"] = fused_up.contiguous()
        converted[f"{prefix}.alpha"] = torch.tensor(fused_alpha, dtype=torch.float32)

    return converted


class MiniMaxH3LoraLoaderMixin(LoraBaseMixin):
    _lora_loadable_modules = ["transformer"]
    transformer_name = "transformer"

    @staticmethod
    def _transformer_uses_gate_first_swiglu(transformer) -> bool:
        config = getattr(transformer, "config", None)
        return bool(getattr(config, "swiglu_gate_first", False))

    @classmethod
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        **kwargs,
    ):
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        return_lora_metadata = kwargs.pop("return_lora_metadata", False)
        metadata = kwargs.pop("metadata", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }
        state_dict, metadata = _fetch_state_dict(
            pretrained_model_name_or_path_or_dict=pretrained_model_name_or_path_or_dict,
            weight_name=weight_name,
            use_safetensors=use_safetensors,
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            allow_pickle=allow_pickle,
            metadata=metadata,
        )
        return (state_dict, metadata) if return_lora_metadata else state_dict

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[str] = None,
        hotswap: bool = False,
        **kwargs,
    ):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for MiniMax-H3 LoRA loading.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", False)
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        lora_format = normalize_lora_format(kwargs.pop("lora_format", None))
        loaded = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict,
            return_lora_metadata=True,
            **kwargs,
        )
        if isinstance(loaded, tuple):
            state_dict, adapter_metadata = loaded
        else:
            state_dict, adapter_metadata = loaded, None
        if not isinstance(state_dict, dict):
            raise ValueError("MiniMax-H3 LoRA checkpoint did not return a state dict.")

        transformer = getattr(self, self.transformer_name, None)
        if transformer is None:
            raise ValueError(f"MiniMax-H3 pipeline has no `{self.transformer_name}` component to load LoRA weights into.")

        detected_format = detect_state_dict_format(state_dict)
        if lora_format == PEFTLoRAFormat.DIFFUSERS and (
            detected_format == PEFTLoRAFormat.COMFYUI or _is_minimax_h3_native_lora_state_dict(state_dict)
        ):
            lora_format = PEFTLoRAFormat.COMFYUI

        network_alphas = None
        if lora_format == PEFTLoRAFormat.COMFYUI:
            state_dict, network_alphas = _convert_minimax_h3_comfy_lora_to_diffusers(
                state_dict,
                target_prefix=self.transformer_name,
                target_swiglu_gate_first=self._transformer_uses_gate_first_swiglu(transformer),
            )
        else:
            source_gate_first = _minimax_h3_swiglu_gate_first_from_metadata(
                adapter_metadata,
                target_prefix=self.transformer_name,
            )
            if source_gate_first is not None:
                state_dict = _convert_minimax_h3_diffusers_swiglu_lora_layout(
                    state_dict,
                    source_gate_first=source_gate_first,
                    target_gate_first=self._transformer_uses_gate_first_swiglu(transformer),
                )

        transformer_prefix = f"{self.transformer_name}."
        transformer_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith(transformer_prefix):
                transformer_state_dict[key] = value
            elif self.transformer_name == "transformer_ref" and key.startswith("transformer."):
                transformer_state_dict[f"transformer_ref.{key.removeprefix('transformer.')}"] = value
            elif not key.startswith(_COMPONENT_PREFIXES):
                transformer_state_dict[f"{transformer_prefix}{key}"] = value

        if not transformer_state_dict:
            raise ValueError("No transformer LoRA weights found for MiniMax-H3.")

        if not network_alphas:
            explicit_alphas = collect_lora_alphas(transformer_state_dict)
            if explicit_alphas:
                network_alphas = {f"{module_key}.alpha": alpha for module_key, alpha in explicit_alphas.items()}
        if not network_alphas:
            inferred_alphas = synthesize_missing_lora_alphas_from_ranks(transformer_state_dict)
            if inferred_alphas:
                network_alphas = inferred_alphas
        transformer_state_dict = {
            key: value for key, value in transformer_state_dict.items() if not key.endswith((".alpha", ".lora_alpha"))
        }

        has_delta_adaln = any(".delta_adaln_embedder." in key for key in transformer_state_dict)
        has_delta_time = any(".delta_time_embedder." in key for key in transformer_state_dict)
        gate_value = _minimax_h3_adapter_metadata_value(
            adapter_metadata,
            MINIMAX_H3_FLOWMAP_GATE_METADATA_KEY,
            target_prefix=self.transformer_name,
        )
        deltatime_type = _minimax_h3_adapter_metadata_value(
            adapter_metadata,
            MINIMAX_H3_FLOWMAP_DELTATIME_METADATA_KEY,
            target_prefix=self.transformer_name,
        )
        if has_delta_adaln or has_delta_time or gate_value is not None or deltatime_type is not None:
            transformer.enable_flowmap_time_conditioning(
                gate_value=0.25 if gate_value is None else float(gate_value),
                deltatime_type="r" if deltatime_type is None else str(deltatime_type),
            )

        peft_metadata = _minimax_h3_peft_adapter_metadata(
            adapter_metadata,
            target_prefix=self.transformer_name,
        )
        if has_delta_adaln and (
            peft_metadata is None
            or _minimax_h3_adapter_metadata_value(
                peft_metadata,
                "modules_to_save",
                target_prefix=self.transformer_name,
            )
            is None
        ):
            local_state_dict = {
                key.removeprefix(f"{self.transformer_name}."): value for key, value in transformer_state_dict.items()
            }
            for key, alpha in (network_alphas or {}).items():
                local_key = key.removeprefix(f"{self.transformer_name}.")
                local_state_dict[local_key] = torch.tensor(alpha, dtype=torch.float32)
            inferred_metadata = peft_lora_config_kwargs_from_state_dict(local_state_dict)
            inferred_metadata["target_modules"] = sorted(collect_lora_ranks(local_state_dict))
            inferred_metadata["modules_to_save"] = ["delta_adaln_embedder"]
            inferred_metadata.setdefault("rank_pattern", {})
            inferred_metadata.setdefault("alpha_pattern", {})
            peft_metadata = {f"{self.transformer_name}.{key}": value for key, value in inferred_metadata.items()}

        if peft_metadata is not None:
            network_alphas = None

        transformer.load_lora_adapter(
            transformer_state_dict,
            adapter_name=adapter_name,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
            prefix=self.transformer_name,
            network_alphas=network_alphas or None,
            metadata=peft_metadata,
        )

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        transformer_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        transformer_lora_adapter_metadata: Optional[dict] = None,
        **kwargs,
    ):
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise ValueError(
                f"MiniMax-H3 LoRA saving only supports transformer weights, got unsupported keys: {unsupported}."
            )
        if not transformer_lora_layers:
            raise ValueError("You must pass `transformer_lora_layers` to save MiniMax-H3 LoRA weights.")

        state_dict = cls.pack_weights(transformer_lora_layers, cls.transformer_name)
        lora_adapter_metadata = {}
        if transformer_lora_adapter_metadata:
            lora_adapter_metadata.update(cls.pack_weights(transformer_lora_adapter_metadata, cls.transformer_name))

        cls.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
            lora_adapter_metadata=lora_adapter_metadata,
        )


class MiniMaxH3ModularPipeline(MiniMaxH3LoraLoaderMixin, ModularPipeline):
    """
    A ModularPipeline for joint video + audio generation with MiniMax-H3, covering the `t2va` (text only) and `fl2va`
    (first and/or last keyframe) tasks of the FL2VA checkpoint.

    MiniMax-H3 denoises **one packed sequence** that holds the text conditioning, the keyframe conditioning latents,
    the audio latents and the video latents at once, which is why the blocks pass a row layout around rather than
    per-modality tensors, and why the pipeline carries two schedulers (`shift = 12.0` for video, `shift = 3.0` for
    audio) that are stepped inside a single transformer call.

    The released checkpoint is guidance-distilled, so the default path runs one forward pass per step. SimpleTuner's
    vendored pipeline can optionally run an explicit negative branch for real CFG and skipped-layer guidance.

    MiniMax-H3 is modular only: this pipeline and its blocks are the whole integration, there is no
    `DiffusionPipeline` half. This class carries the config-derived geometry the blocks read off the components, the
    packed-sequence geometry lives in `modular_pipelines.minimax_h3.packing`, and the conditioning, encoding and noise
    contracts live on the blocks themselves.

    ```py
    import torch
    from diffusers import ModularPipeline

    pipe = ModularPipeline.from_pretrained("MiniMaxAI/MiniMax-H3")
    pipe.load_components(dtype=torch.bfloat16)
    ```

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "MiniMaxH3Blocks"

    @property
    def vae_spatial_compression_ratio(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.spatial_compression_ratio
        return 16

    @property
    def vae_latent_channels(self):
        if getattr(self, "vae", None) is not None:
            return self.vae.config.latent_channels
        return 24

    @property
    def audio_sampling_rate(self):
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.config.sampling_rate
        return 32000

    @property
    def audio_latent_channels(self):
        if getattr(self, "audio_vae", None) is not None:
            return self.audio_vae.config.latent_channels
        return 32

    @property
    def patch_size(self):
        if getattr(self, "transformer", None) is not None:
            return tuple(self.transformer.config.patch_size)
        return (1, 2, 2)


class MiniMaxH3Ref2VAModularPipeline(MiniMaxH3ModularPipeline):
    """
    A ModularPipeline for joint video + audio generation from omni-references with MiniMax-H3, the `ref2va` task of the
    Ref2VA checkpoint.

    A request carries an ordered list of references — up to 9 images, 3 videos and 3 audio clips, 12 in total — and
    MiniMax-H3 packs one block per reference in front of the generated rows. The order is semantic: it labels the
    references in the prompt presentation and it advances the shared audio/video rotary clock, so reordering the same
    references is a different request.

    The transformer is registered as `transformer_ref`, so one repository can hold both checkpoint partitions
    (`transformer/` for [`MiniMaxH3ModularPipeline`], `transformer_ref/` for this one) and either pipeline loads only
    its own weights. One repository also means one `modular_model_index.json`, which names the `t2va` / `fl2va` half;
    load this one through its blocks instead, which reads the very same file:

    ```py
    pipe = MiniMaxH3Ref2VABlocks().init_pipeline("MiniMaxAI/MiniMax-H3")
    pipe.load_components(dtype=torch.bfloat16)
    ```

    The blocks carry the `ref2va` conditioning, encoding and noise contracts themselves.

    > [!WARNING] > This is an experimental feature and is likely to change in the future.
    """

    default_blocks_name = "MiniMaxH3Ref2VABlocks"
    _lora_loadable_modules = ["transformer_ref"]
    transformer_name = "transformer_ref"

    @property
    def patch_size(self):
        if getattr(self, "transformer_ref", None) is not None:
            return tuple(self.transformer_ref.config.patch_size)
        return (1, 2, 2)
