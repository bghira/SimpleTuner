"""Krea 2 LoRA loader vendored from huggingface/diffusers#14046."""

import os
from typing import Callable

import torch
from diffusers.loaders.lora_base import LoraBaseMixin, _fetch_state_dict
from diffusers.utils import (
    USE_PEFT_BACKEND,
    convert_unet_state_dict_to_peft,
    get_adapter_name,
    get_peft_kwargs,
    is_peft_available,
    is_peft_version,
    is_torch_version,
    is_transformers_available,
    is_transformers_version,
    logging,
)
from huggingface_hub.utils import validate_hf_hub_args

from simpletuner.helpers.utils.offloading import restore_offload_state, unpack_offload_state

_LOW_CPU_MEM_USAGE_DEFAULT_LORA = False
if is_torch_version(">=", "1.9.0"):
    if (
        is_peft_available()
        and is_peft_version(">=", "0.13.1")
        and is_transformers_available()
        and is_transformers_version(">", "4.45.2")
    ):
        _LOW_CPU_MEM_USAGE_DEFAULT_LORA = True

TRANSFORMER_NAME = "transformer"
KREA2_LORA_TRANSFORMER_PREFIXES = ("transformer", "diffusion_model")
KREA2_EXTERNAL_LORA_MODULE_MAP = (
    ("blocks.", "transformer_blocks."),
    (".attn.wq.", ".attn.to_q."),
    (".attn.wk.", ".attn.to_k."),
    (".attn.wv.", ".attn.to_v."),
    (".attn.wo.", ".attn.to_out.0."),
    (".attn.gate.", ".attn.to_gate."),
    (".mlp.gate.", ".ff.gate."),
    (".mlp.up.", ".ff.up."),
    (".mlp.down.", ".ff.down."),
)

logger = logging.get_logger(__name__)


def _normalize_krea2_lora_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], str | None]:
    """Return KREA2 transformer LoRA keys without their serialization prefix."""
    for prefix in KREA2_LORA_TRANSFORMER_PREFIXES:
        prefix_with_dot = f"{prefix}."
        matches = {
            _translate_krea2_lora_module_key(key.removeprefix(prefix_with_dot)): value
            for key, value in state_dict.items()
            if key.startswith(prefix_with_dot)
        }
        if matches:
            return matches, prefix

    return {_translate_krea2_lora_module_key(key): value for key, value in state_dict.items()}, None


def _translate_krea2_lora_module_key(key: str) -> str:
    """Map external KREA LoRA module names onto SimpleTuner's KREA2 module graph."""
    translated = key
    for old, new in KREA2_EXTERNAL_LORA_MODULE_MAP:
        if old == "blocks.":
            if translated.startswith(old):
                translated = translated.replace(old, new, 1)
            continue
        translated = translated.replace(old, new)
    return translated


def _infer_krea2_lora_target_modules(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Infer exact model-relative module paths from PEFT-style LoRA tensor keys."""
    target_modules: dict[str, None] = {}
    for key in state_dict:
        for marker in (".lora_A.", ".lora_B."):
            if marker in key:
                target_modules[key.split(marker, 1)[0]] = None
                break
    return list(target_modules)


class Krea2LoraLoaderMixin(LoraBaseMixin):
    r"""
    Load LoRA layers into [`Krea2Transformer2DModel`]. Specific to [`Krea2Pipeline`].
    """

    _lora_loadable_modules = ["transformer"]
    transformer_name = TRANSFORMER_NAME

    @classmethod
    @validate_hf_hub_args
    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.lora_state_dict
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: str | dict[str, torch.Tensor],
        **kwargs,
    ):
        r"""
        See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`] for more details.
        """
        # Load the main state dict first which has the LoRA layers for either of
        # transformer and text encoder or both.
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

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {"file_type": "attn_procs_weights", "framework": "pytorch"}

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
        )

        is_dora_scale_present = any("dora_scale" in k for k in state_dict)
        if is_dora_scale_present:
            warn_msg = "It seems like you are using a DoRA checkpoint that is not compatible in Diffusers at the moment. So, we are going to filter out the keys associated to 'dora_scale` from the state dict. If you think this is a mistake please open an issue https://github.com/huggingface/diffusers/issues/new."
            logger.warning(warn_msg)
            state_dict = {k: v for k, v in state_dict.items() if "dora_scale" not in k}

        out = (state_dict, metadata) if return_lora_metadata else state_dict
        return out

    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.load_lora_weights
    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: str | dict[str, torch.Tensor],
        adapter_name: str | None = None,
        hotswap: bool = False,
        **kwargs,
    ):
        """
        See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`] for more details.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA)
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # if a dict is passed, copy it instead of modifying it inplace
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        kwargs["return_lora_metadata"] = True
        state_dict, metadata = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)

        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint. Make sure all LoRA param names contain `'lora'` substring.")

        self.load_lora_into_transformer(
            state_dict,
            transformer=getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer,
            adapter_name=adapter_name,
            metadata=metadata,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
            hotswap=hotswap,
        )

    @classmethod
    # Copied from diffusers.loaders.lora_pipeline.SD3LoraLoaderMixin.load_lora_into_transformer with SD3Transformer2DModel->Krea2Transformer2DModel
    def load_lora_into_transformer(
        cls,
        state_dict,
        transformer,
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
        hotswap: bool = False,
        metadata=None,
    ):
        """
        See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_unet`] for more details.
        """
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        if hotswap:
            # Keep the upstream hot-swap implementation for compiled pipelines. Normal validation
            # adapter loading uses the explicit PEFT path below so KREA module names are targeted.
            logger.info(f"Loading {cls.transformer_name}.")
            transformer.load_lora_adapter(
                state_dict,
                network_alphas=None,
                adapter_name=adapter_name,
                metadata=metadata,
                _pipeline=_pipeline,
                low_cpu_mem_usage=low_cpu_mem_usage,
                hotswap=hotswap,
            )
            return

        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

        state_dict, source_prefix = _normalize_krea2_lora_state_dict(state_dict)
        if not state_dict:
            supported = ", ".join(f"{prefix}.*" for prefix in KREA2_LORA_TRANSFORMER_PREFIXES)
            raise ValueError(f"No KREA2 transformer LoRA keys were found. Expected one of: {supported}.")

        first_key = next(iter(state_dict.keys()))
        if "lora_A" not in first_key:
            state_dict = convert_unet_state_dict_to_peft(state_dict)

        if adapter_name in getattr(transformer, "peft_config", {}):
            raise ValueError(
                f"Adapter name {adapter_name} already in use in the transformer - please select a new adapter name."
            )

        rank = {}
        for key, val in state_dict.items():
            if "lora_B" in key and getattr(val, "ndim", 0) > 1:
                rank[key] = val.shape[1]

        target_modules = _infer_krea2_lora_target_modules(state_dict)
        if not target_modules:
            raise ValueError(
                "Could not infer KREA2 LoRA target modules from the adapter state dict. "
                "Expected keys ending in .lora_A.weight and .lora_B.weight."
            )

        lora_config_kwargs = get_peft_kwargs(rank, network_alpha_dict=None, peft_state_dict=state_dict)
        lora_config_kwargs["target_modules"] = target_modules
        if "use_dora" in lora_config_kwargs:
            if lora_config_kwargs["use_dora"] and is_peft_version("<", "0.9.0"):
                raise ValueError(
                    "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. "
                    "Please upgrade your installation of `peft`."
                )
            if not lora_config_kwargs["use_dora"]:
                lora_config_kwargs.pop("use_dora")
        lora_config = LoraConfig(**lora_config_kwargs)

        if adapter_name is None:
            adapter_name = get_adapter_name(transformer)

        logger.info(
            f"Loading {cls.transformer_name} LoRA adapter '{adapter_name}' "
            f"from {source_prefix or 'unprefixed'} keys with {len(target_modules)} target module(s)."
        )

        offload_state = cls._optionally_disable_offloading(_pipeline)
        (
            is_model_cpu_offload,
            is_sequential_cpu_offload,
            is_group_offload,
        ) = unpack_offload_state(offload_state)

        peft_kwargs = {}
        if is_peft_version(">=", "0.13.1"):
            peft_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

        try:
            inject_adapter_in_model(lora_config, transformer, adapter_name=adapter_name, **peft_kwargs)
            incompatible_keys = set_peft_model_state_dict(transformer, state_dict, adapter_name, **peft_kwargs)
            if incompatible_keys is not None:
                logger.info(f"Loaded KREA2 LoRA with incompatible keys: {incompatible_keys}")
        finally:
            restore_offload_state(_pipeline, is_model_cpu_offload, is_sequential_cpu_offload, is_group_offload)

    @classmethod
    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.save_lora_weights
    def save_lora_weights(
        cls,
        save_directory: str | os.PathLike,
        transformer_lora_layers: dict[str, torch.nn.Module | torch.Tensor] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        transformer_lora_adapter_metadata: dict | None = None,
    ):
        r"""
        See [`~loaders.StableDiffusionLoraLoaderMixin.save_lora_weights`] for more information.
        """
        lora_layers = {}
        lora_metadata = {}

        if transformer_lora_layers:
            lora_layers[cls.transformer_name] = transformer_lora_layers
            lora_metadata[cls.transformer_name] = transformer_lora_adapter_metadata

        if not lora_layers:
            raise ValueError("You must pass at least one of `transformer_lora_layers` or `text_encoder_lora_layers`.")

        cls._save_lora_weights(
            save_directory=save_directory,
            lora_layers=lora_layers,
            lora_metadata=lora_metadata,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )

    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.fuse_lora
    def fuse_lora(
        self,
        components: list[str] = ["transformer"],
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
        adapter_names: list[str] | None = None,
        **kwargs,
    ):
        r"""
        See [`~loaders.StableDiffusionLoraLoaderMixin.fuse_lora`] for more details.
        """
        super().fuse_lora(
            components=components,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
            adapter_names=adapter_names,
            **kwargs,
        )

    # Copied from diffusers.loaders.lora_pipeline.CogVideoXLoraLoaderMixin.unfuse_lora
    def unfuse_lora(self, components: list[str] = ["transformer"], **kwargs):
        r"""
        See [`~loaders.StableDiffusionLoraLoaderMixin.unfuse_lora`] for more details.
        """
        super().unfuse_lora(components=components, **kwargs)
