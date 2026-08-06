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
    collect_lora_alphas,
    detect_state_dict_format,
    normalize_lora_format,
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

    for raw_key, value in state_dict.items():
        key = _strip_comfy_lora_prefix(raw_key)
        split_key = _split_lora_suffix(key)
        if split_key is None:
            if key.endswith(".alpha"):
                module_key = key[: -len(".alpha")]
                for mapped_module in _map_comfy_lora_module(module_key, target_prefix):
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
        if len(mapped_modules) == 3 and target_suffix == ".lora.up.weight":
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

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }
        state_dict, _ = _fetch_state_dict(
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
        )
        return state_dict

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
        state_dict = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)
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

        transformer.load_lora_adapter(
            transformer_state_dict,
            adapter_name=adapter_name,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
            prefix=self.transformer_name,
            network_alphas=network_alphas or None,
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
