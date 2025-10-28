# Copyright 2025 Black Forest Labs and The HuggingFace Team.
# Licensed under the Apache License, Version 2.0 (the "License").

import inspect
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL.Image
import torch
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxIPAdapterMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.loaders.lora_base import LoraBaseMixin, _fetch_state_dict
from diffusers.loaders.lora_conversion_utils import (
    _convert_bfl_flux_control_lora_to_diffusers,
    _convert_kohya_flux_lora_to_diffusers,
    _convert_xlabs_flux_lora_to_diffusers,
)
from diffusers.models import AutoencoderKL
from diffusers.models.lora import text_encoder_attn_modules, text_encoder_mlp_modules
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    USE_PEFT_BACKEND,
    BaseOutput,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    convert_unet_state_dict_to_peft,
    deprecate,
    get_adapter_name,
    get_peft_kwargs,
    is_peft_available,
    is_peft_version,
    is_torch_version,
    is_torch_xla_available,
    is_transformers_available,
    is_transformers_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from huggingface_hub.utils import validate_hf_hub_args
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection, T5EncoderModel, T5TokenizerFast

from simpletuner.helpers.models.chroma.transformer import ChromaTransformer2DModel
from simpletuner.helpers.utils.offloading import restore_offload_state, unpack_offload_state

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ChromaPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


_LOW_CPU_MEM_USAGE_DEFAULT_LORA = False
if is_torch_version(">=", "1.9.0"):
    if (
        is_peft_available()
        and is_peft_version(">=", "0.13.1")
        and is_transformers_available()
        and is_transformers_version(">", "4.45.2")
    ):
        _LOW_CPU_MEM_USAGE_DEFAULT_LORA = True


class ChromaLoraLoaderMixin(LoraBaseMixin):
    r"""
    Load LoRA layers into [`ChromaTransformer2DModel`],
    [`T5EncoderModel`](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel).

    Specific to [`ChromaPipeline`].
    """

    _lora_loadable_modules = ["transformer", "text_encoder"]
    transformer_name = "transformer"
    text_encoder_name = "text_encoder"
    controlnet_name = "controlnet"

    @classmethod
    @validate_hf_hub_args
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        return_alphas: bool = False,
        **kwargs,
    ):
        r"""
        Return state dict for lora weights and the network alphas.

        <Tip warning={true}>

        We support loading A1111 formatted LoRA checkpoints in a limited capacity.

        This function is experimental and might change in the future.

        </Tip>

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.

            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.

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
        )

        is_dora_scale_present = any("dora_scale" in k for k in state_dict)
        if is_dora_scale_present:
            warn_msg = "It seems like you are using a DoRA checkpoint that is not compatible in Diffusers at the moment. So, we are going to filter out the keys associated to 'dora_scale` from the state dict. If you think this is a mistake please open an issue https://github.com/huggingface/diffusers/issues/new."
            logger.warning(warn_msg)
            state_dict = {k: v for k, v in state_dict.items() if "dora_scale" not in k}

        # TODO (sayakpaul): to a follow-up to clean and try to unify the conditions.
        is_kohya = any(".lora_down.weight" in k for k in state_dict)
        if is_kohya:
            state_dict = _convert_kohya_flux_lora_to_diffusers(state_dict)
            # Kohya already takes care of scaling the LoRA parameters with alpha.
            return (state_dict, None) if return_alphas else state_dict

        is_xlabs = any("processor" in k for k in state_dict)
        if is_xlabs:
            state_dict = _convert_xlabs_flux_lora_to_diffusers(state_dict)
            # xlabs doesn't use `alpha`.
            return (state_dict, None) if return_alphas else state_dict

        # For state dicts like
        # https://huggingface.co/TheLastBen/Jon_Snow_Flux_LoRA
        keys = list(state_dict.keys())
        network_alphas = {}
        for k in keys:
            if "alpha" in k:
                alpha_value = state_dict.get(k)
                if (torch.is_tensor(alpha_value) and torch.is_floating_point(alpha_value)) or isinstance(alpha_value, float):
                    network_alphas[k] = state_dict.pop(k)
                else:
                    raise ValueError(
                        f"The alpha key ({k}) seems to be incorrect. If you think this error is unexpected, please open as issue."
                    )

        if return_alphas:
            return state_dict, network_alphas
        else:
            return state_dict

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name=None,
        **kwargs,
    ):
        """
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.transformer`,
        `self.text_encoder`, and optionally `self.controlnet`.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA)
        if low_cpu_mem_usage and not is_peft_version(">=", "0.13.1"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # if a dict is passed, copy it instead of modifying it inplace
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict, network_alphas = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict, return_alphas=True, **kwargs
        )

        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        # Separate transformer, text encoder, and controlnet weights
        transformer_state_dict = {}
        text_encoder_state_dict = {}
        controlnet_state_dict = {}

        for k, v in state_dict.items():
            if k.startswith("text_encoder."):
                text_encoder_state_dict[k] = v
            elif k.startswith("controlnet."):
                controlnet_state_dict[k] = v
            else:
                # Assume transformer weights
                transformer_state_dict[k] = v

        # Load transformer weights
        if transformer_state_dict:
            self.load_lora_into_transformer(
                transformer_state_dict,
                network_alphas=network_alphas,
                transformer=(getattr(self, self.transformer_name) if not hasattr(self, "transformer") else self.transformer),
                adapter_name=adapter_name,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )

        # Load text encoder weights
        if text_encoder_state_dict:
            self.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=network_alphas,
                text_encoder=self.text_encoder,
                prefix="text_encoder",
                lora_scale=self.lora_scale,
                adapter_name=adapter_name,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )

        # Load controlnet weights if present
        if controlnet_state_dict and hasattr(self, "controlnet"):
            self.load_lora_into_controlnet(
                controlnet_state_dict,
                network_alphas=network_alphas,
                controlnet=self.controlnet,
                adapter_name=adapter_name,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )

    @classmethod
    def load_lora_into_controlnet(
        cls,
        state_dict,
        network_alphas,
        controlnet,
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
    ):
        """
        Load LoRA layers into the controlnet.
        """
        if low_cpu_mem_usage and not is_peft_version(">=", "0.13.1"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

        keys = list(state_dict.keys())
        controlnet_key = getattr(cls, "controlnet_name", "controlnet")

        controlnet_keys = [k for k in keys if k.startswith(controlnet_key)]
        state_dict = {k.replace(f"{controlnet_key}.", ""): v for k, v in state_dict.items() if k in controlnet_keys}

        if len(state_dict.keys()) > 0:
            # check with first key if is not in peft format
            first_key = next(iter(state_dict.keys()))
            if "lora_A" not in first_key:
                state_dict = convert_unet_state_dict_to_peft(state_dict)

            if adapter_name in getattr(controlnet, "peft_config", {}):
                raise ValueError(
                    f"Adapter name {adapter_name} already in use in the controlnet - please select a new adapter name."
                )

            rank = {}
            for key, val in state_dict.items():
                if "lora_B" in key:
                    rank[key] = val.shape[1]

            if network_alphas is not None and len(network_alphas) >= 1:
                alpha_keys = [k for k in network_alphas.keys() if k.startswith(controlnet_key)]
                network_alphas = {
                    k.replace(f"{controlnet_key}.", ""): v for k, v in network_alphas.items() if k in alpha_keys
                }

            lora_config_kwargs = get_peft_kwargs(rank, network_alpha_dict=network_alphas, peft_state_dict=state_dict)
            if "use_dora" in lora_config_kwargs:
                if lora_config_kwargs["use_dora"] and is_peft_version("<", "0.9.0"):
                    raise ValueError(
                        "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
                    )
                else:
                    lora_config_kwargs.pop("use_dora")
            lora_config = LoraConfig(**lora_config_kwargs)

            # adapter_name
            if adapter_name is None:
                adapter_name = get_adapter_name(controlnet)

            # In case the pipeline has been already offloaded to CPU - temporarily remove the hooks
            # otherwise loading LoRA weights will lead to an error
            offload_state = cls._optionally_disable_offloading(_pipeline)
            (
                is_model_cpu_offload,
                is_sequential_cpu_offload,
                is_group_offload,
            ) = unpack_offload_state(offload_state)

            peft_kwargs = {}
            if is_peft_version(">=", "0.13.1"):
                peft_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

            inject_adapter_in_model(lora_config, controlnet, adapter_name=adapter_name, **peft_kwargs)
            incompatible_keys = set_peft_model_state_dict(controlnet, state_dict, adapter_name, **peft_kwargs)

            warn_msg = ""
            if incompatible_keys is not None:
                # Check only for unexpected keys.
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    lora_unexpected_keys = [k for k in unexpected_keys if "lora_" in k and adapter_name in k]
                    if lora_unexpected_keys:
                        warn_msg = (
                            f"Loading adapter weights from state_dict led to unexpected keys found in the model:"
                            f" {', '.join(lora_unexpected_keys)}. "
                        )

            if warn_msg:
                logger.warning(warn_msg)

            # Offload back.
            restore_offload_state(_pipeline, is_model_cpu_offload, is_sequential_cpu_offload, is_group_offload)

    @classmethod
    def load_lora_into_transformer(
        cls,
        state_dict,
        network_alphas,
        transformer,
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
    ):
        """
        This will load the LoRA layers specified in `state_dict` into `transformer`.

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The keys can either be indexed directly
                into the unet or prefixed with an additional `unet` which can be used to distinguish between text
                encoder lora layers.
            network_alphas (`Dict[str, float]`):
                The value of the network alpha used for stable learning and preventing underflow. This value has the
                same meaning as the `--network_alpha` option in the kohya-ss trainer script. Refer to [this
                link](https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning).
            transformer (`SD3Transformer2DModel`):
                The Transformer model to load the LoRA layers into.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            Speed up model loading by only loading the pretrained LoRA weights and not initializing the random weights.:
        """
        if low_cpu_mem_usage and not is_peft_version(">=", "0.13.1"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

        keys = list(state_dict.keys())

        transformer_keys = [k for k in keys if k.startswith(cls.transformer_name)]
        state_dict = {k.replace(f"{cls.transformer_name}.", ""): v for k, v in state_dict.items() if k in transformer_keys}

        if len(state_dict.keys()) > 0:
            # check with first key if is not in peft format
            first_key = next(iter(state_dict.keys()))
            if "lora_A" not in first_key:
                state_dict = convert_unet_state_dict_to_peft(state_dict)

            if adapter_name in getattr(transformer, "peft_config", {}):
                raise ValueError(
                    f"Adapter name {adapter_name} already in use in the transformer - please select a new adapter name."
                )

            rank = {}
            for key, val in state_dict.items():
                if "lora_B" in key:
                    rank[key] = val.shape[1]

            if network_alphas is not None and len(network_alphas) >= 1:
                prefix = cls.transformer_name
                alpha_keys = [k for k in network_alphas.keys() if k.startswith(prefix) and k.split(".")[0] == prefix]
                network_alphas = {k.replace(f"{prefix}.", ""): v for k, v in network_alphas.items() if k in alpha_keys}

            lora_config_kwargs = get_peft_kwargs(rank, network_alpha_dict=network_alphas, peft_state_dict=state_dict)
            if "use_dora" in lora_config_kwargs:
                if lora_config_kwargs["use_dora"] and is_peft_version("<", "0.9.0"):
                    raise ValueError(
                        "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
                    )
                else:
                    lora_config_kwargs.pop("use_dora")
            lora_config = LoraConfig(**lora_config_kwargs)

            # adapter_name
            if adapter_name is None:
                adapter_name = get_adapter_name(transformer)

            # In case the pipeline has been already offloaded to CPU - temporarily remove the hooks
            # otherwise loading LoRA weights will lead to an error
            offload_state = cls._optionally_disable_offloading(_pipeline)
            (
                is_model_cpu_offload,
                is_sequential_cpu_offload,
                is_group_offload,
            ) = unpack_offload_state(offload_state)

            peft_kwargs = {}
            if is_peft_version(">=", "0.13.1"):
                peft_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

            inject_adapter_in_model(lora_config, transformer, adapter_name=adapter_name, **peft_kwargs)
            incompatible_keys = set_peft_model_state_dict(transformer, state_dict, adapter_name, **peft_kwargs)

            warn_msg = ""
            if incompatible_keys is not None:
                # Check only for unexpected keys.
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    lora_unexpected_keys = [k for k in unexpected_keys if "lora_" in k and adapter_name in k]
                    if lora_unexpected_keys:
                        warn_msg = (
                            f"Loading adapter weights from state_dict led to unexpected keys found in the model:"
                            f" {', '.join(lora_unexpected_keys)}. "
                        )

            if warn_msg:
                logger.warning(warn_msg)

            # Offload back.
            restore_offload_state(_pipeline, is_model_cpu_offload, is_sequential_cpu_offload, is_group_offload)

    # Copied from diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.load_lora_into_text_encoder
    def load_lora_into_text_encoder(
        self,
        pretrained_model_name_or_path_or_dict,
        network_alphas,
        text_encoder,
        adapter_name=None,
        lora_scale=1.0,
        prefix: str = "text_encoder",
        _pipeline=None,
        low_cpu_mem_usage=False,
    ):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

        if adapter_name in getattr(text_encoder, "peft_config", {}):
            raise ValueError(
                f"Adapter name {adapter_name} already in use in the text encoder - please select a new adapter name."
            )

        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            state_dict, network_alphas = pretrained_model_name_or_path_or_dict, network_alphas
        else:
            state_dict, network_alphas = self._get_text_encoder_lora_state_dicts(
                pretrained_model_name_or_path_or_dict,
                network_alphas,
                prefix=prefix,
                weight_name=None,
            )

        rank = {}
        for key, val in state_dict.items():
            if "lora_B" in key:
                rank[key] = val.shape[1]

        if network_alphas is not None and len(network_alphas) >= 1:
            network_alphas = {k.replace(f"{prefix}.", ""): v for k, v in network_alphas.items() if prefix in k}

        lora_config_kwargs = get_peft_kwargs(rank, network_alpha_dict=network_alphas, peft_state_dict=state_dict)
        if "use_dora" in lora_config_kwargs:
            if lora_config_kwargs["use_dora"] and is_peft_version("<", "0.9.0"):
                raise ValueError(
                    "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
                )
            else:
                lora_config_kwargs.pop("use_dora")

        lora_config = LoraConfig(
            target_modules=text_encoder_attn_modules(text_encoder)
            + text_encoder_mlp_modules(text_encoder, lora_config_kwargs.get("key_shared")),
            init_lora_weights="gaussian",
            **lora_config_kwargs,
        )

        if adapter_name is None:
            adapter_name = get_adapter_name(text_encoder)

        peft_kwargs = {}
        if is_peft_version(">=", "0.13.1"):
            peft_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

        inject_adapter_in_model(lora_config, text_encoder, adapter_name=adapter_name, **peft_kwargs)
        incompatible_keys = set_peft_model_state_dict(text_encoder, state_dict, adapter_name, **peft_kwargs)

        warn_msg = ""
        if incompatible_keys is not None:
            # Check only for unexpected keys.
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                lora_unexpected_keys = [k for k in unexpected_keys if "lora_" in k and adapter_name in k]
                if lora_unexpected_keys:
                    warn_msg = (
                        f"Loading adapter weights from state_dict led to unexpected keys found in the model:"
                        f" {', '.join(lora_unexpected_keys)}. "
                    )

        if warn_msg:
            logger.warning(warn_msg)

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        transformer_lora_layers: Union[Dict[str, torch.nn.Module], Dict[str, torch.Tensor]] = None,
        text_encoder_lora_layers: Union[Dict[str, torch.nn.Module], Dict[str, torch.Tensor]] = None,
        controlnet_lora_layers: Union[Dict[str, torch.nn.Module], Dict[str, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
    ):
        r"""
        Save the LoRA parameters of the UNet, text encoder and optionally the ControlNet.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            transformer_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `transformer`.
            text_encoder_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder`. Must explicitly pass the text
                encoder LoRA state dict because it comes from 🤗 Transformers.
            controlnet_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `controlnet`.
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or the traditional PyTorch way with `pickle`.
        """
        state_dict = {}

        if not (transformer_lora_layers or text_encoder_lora_layers or controlnet_lora_layers):
            raise ValueError(
                "You must pass at least one of `transformer_lora_layers`, `text_encoder_lora_layers`, or `controlnet_lora_layers`."
            )

        if transformer_lora_layers:
            state_dict.update(cls.pack_weights(transformer_lora_layers, cls.transformer_name))

        if text_encoder_lora_layers:
            state_dict.update(cls.pack_weights(text_encoder_lora_layers, cls.text_encoder_name))

        if controlnet_lora_layers:
            controlnet_prefix = "controlnet"
            state_dict.update(cls.pack_weights(controlnet_lora_layers, controlnet_prefix))

        # Save the model
        cls.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
        )

    # Copied from diffusers.loaders.lora_pipeline.StableDiffusionLoraLoaderMixin.fuse_lora with unet->transformer
    def fuse_lora(
        self,
        components: List[str] = ["transformer", "text_encoder"],
        lora_scale: float = 1.0,
        safe_fusing: bool = False,
        adapter_names: Optional[List[str]] = None,
        **kwargs,
    ):
        r"""
        Fuses the LoRA parameters into the original parameters of the corresponding blocks.

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            components: (`List[str]`): List of LoRA-injectable components to fuse the LoRAs into.
            lora_scale (`float`, defaults to 1.0):
                Controls how much to influence the outputs with the LoRA parameters.
            safe_fusing (`bool`, defaults to `False`):
                Whether to check fused weights for NaN values before fusing and if values are NaN not fusing them.
            adapter_names (`List[str]`, *optional*):
                Adapter names to be used for fusing. If nothing is passed, all active adapters will be fused.

        Example:

        ```py
        from diffusers import DiffusionPipeline
        import torch

        pipeline = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.fuse_lora(lora_scale=0.7)
        ```
        """
        super().fuse_lora(
            components=components,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
            adapter_names=adapter_names,
        )

    def unfuse_lora(self, components: List[str] = ["transformer", "text_encoder"], **kwargs):
        r"""
        Reverses the effect of
        [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraBaseMixin.fuse_lora).

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            components (`List[str]`): List of LoRA-injectable components to unfuse LoRA from.
        """
        super().unfuse_lora(components=components)


class ChromaPipeline(
    DiffusionPipeline,
    ChromaLoraLoaderMixin,
    FromSingleFileMixin,
    TextualInversionLoaderMixin,
    FluxIPAdapterMixin,
):
    r"""
    The Chroma pipeline for text-to-image generation.
    """

    model_cpu_offload_seq = "text_encoder->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: T5EncoderModel,
        tokenizer: T5TokenizerFast,
        transformer: ChromaTransformer2DModel,
        image_encoder: CLIPVisionModelWithProjection = None,
        feature_extractor: CLIPImageProcessor = None,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.default_sample_size = 128

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if isinstance(self, TextualInversionLoaderMixin):
            prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        tokenizer_mask = text_inputs.attention_mask
        tokenizer_mask_device = tokenizer_mask.to(device)

        # unlike Flux, Chroma uses the tokenizer's attention mask when generating the T5 embeddings
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device),
            output_hidden_states=False,
            attention_mask=tokenizer_mask_device,
        )[0]

        dtype = self.text_encoder.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # for the text tokens, Chroma requires that all except the first padding token are masked out
        seq_lengths = tokenizer_mask_device.sum(dim=1)
        mask_indices = torch.arange(tokenizer_mask_device.size(1), device=device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = (mask_indices <= seq_lengths.unsqueeze(1)).to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        attention_mask = attention_mask.repeat(1, num_images_per_prompt)
        attention_mask = attention_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        do_classifier_free_guidance: bool = True,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        device = device or self._execution_device

        if lora_scale is not None and isinstance(self, ChromaLoraLoaderMixin):
            self._lora_scale = lora_scale

            if self.text_encoder is not None and USE_PEFT_BACKEND:
                scale_lora_layers(self.text_encoder, lora_scale)

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
            )

        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        negative_text_ids = None

        if do_classifier_free_guidance:
            if negative_prompt_embeds is None:
                negative_prompt = negative_prompt or ""
                negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

                if prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )

                negative_prompt_embeds, negative_prompt_attention_mask = self._get_t5_prompt_embeds(
                    prompt=negative_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                )

            negative_text_ids = torch.zeros(negative_prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

        if self.text_encoder is not None:
            if isinstance(self, ChromaLoraLoaderMixin) and USE_PEFT_BACKEND:
                unscale_lora_layers(self.text_encoder, lora_scale)

        return (
            prompt_embeds,
            text_ids,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_text_ids,
            negative_prompt_attention_mask,
        )

    def encode_image(self, image, device, num_images_per_prompt):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.feature_extractor(image, return_tensors="pt").pixel_values

        image = image.to(device=device, dtype=dtype)
        image_embeds = self.image_encoder(image).image_embeds
        image_embeds = image_embeds.repeat_interleave(num_images_per_prompt, dim=0)
        return image_embeds

    def prepare_ip_adapter_image_embeds(self, ip_adapter_image, ip_adapter_image_embeds, device, num_images_per_prompt):
        image_embeds = []
        if ip_adapter_image_embeds is None:
            if not isinstance(ip_adapter_image, list):
                ip_adapter_image = [ip_adapter_image]

            if len(ip_adapter_image) != self.transformer.encoder_hid_proj.num_ip_adapters:
                raise ValueError(
                    f"`ip_adapter_image` must have same length as the number of IP Adapters. Got {len(ip_adapter_image)} images and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_ip_adapter_image in ip_adapter_image:
                single_image_embeds = self.encode_image(single_ip_adapter_image, device, 1)
                image_embeds.append(single_image_embeds[None, :])
        else:
            if not isinstance(ip_adapter_image_embeds, list):
                ip_adapter_image_embeds = [ip_adapter_image_embeds]

            if len(ip_adapter_image_embeds) != self.transformer.encoder_hid_proj.num_ip_adapters:
                raise ValueError(
                    f"`ip_adapter_image_embeds` must have same length as the number of IP Adapters. Got {len(ip_adapter_image_embeds)} image embeds and {self.transformer.encoder_hid_proj.num_ip_adapters} IP Adapters."
                )

            for single_image_embeds in ip_adapter_image_embeds:
                image_embeds.append(single_image_embeds)

        ip_adapter_image_embeds = []
        for single_image_embeds in image_embeds:
            single_image_embeds = torch.cat([single_image_embeds] * num_images_per_prompt, dim=0)
            single_image_embeds = single_image_embeds.to(device=device)
            ip_adapter_image_embeds.append(single_image_embeds)

        return ip_adapter_image_embeds

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_embeds=None,
        negative_prompt_attention_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
            )

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and prompt_attention_mask is None:
            raise ValueError("Cannot provide `prompt_embeds` without also providing `prompt_attention_mask")

        if negative_prompt_embeds is not None and negative_prompt_attention_mask is None:
            raise ValueError(
                "Cannot provide `negative_prompt_embeds` without also providing `negative_prompt_attention_mask"
            )

        if max_sequence_length is not None and max_sequence_length > 512:
            raise ValueError(f"`max_sequence_length` cannot be greater than 512 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(latent_image_id_height * latent_image_id_width, latent_image_id_channels)

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

        return latents

    def enable_vae_slicing(self):
        deprecate(
            "enable_vae_slicing",
            "0.40.0",
            f"Calling `enable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_slicing()`.",
        )
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        deprecate(
            "disable_vae_slicing",
            "0.40.0",
            f"Calling `disable_vae_slicing()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_slicing()`.",
        )
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        deprecate(
            "enable_vae_tiling",
            "0.40.0",
            f"Calling `enable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.enable_tiling()`.",
        )
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        deprecate(
            "disable_vae_tiling",
            "0.40.0",
            f"Calling `disable_vae_tiling()` on a `{self.__class__.__name__}` is deprecated and this method will be removed in a future version. Please use `pipe.vae.disable_tiling()`.",
        )
        self.vae.disable_tiling()

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is not None:
            latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
            return latents.to(device=device, dtype=dtype), latent_image_ids

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)

        latent_image_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)

        return latents, latent_image_ids

    def _prepare_attention_mask(
        self,
        batch_size,
        sequence_length,
        dtype,
        attention_mask=None,
    ):
        if attention_mask is None:
            return attention_mask

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    batch_size,
                    sequence_length,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )

        return attention_mask

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 35,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        skip_guidance_layers: Optional[List[int]] = None,
        skip_layer_guidance_scale: float = 2.8,
        skip_layer_guidance_stop: float = 0.2,
        skip_layer_guidance_start: float = 0.01,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._skip_layer_guidance_scale = skip_layer_guidance_scale
        self._current_timestep = None
        self._interrupt = False
        if skip_guidance_layers is not None and not isinstance(skip_guidance_layers, list):
            skip_guidance_layers = list(skip_guidance_layers)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        (
            prompt_embeds,
            text_ids,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_text_ids,
            negative_prompt_attention_mask,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        original_prompt_embeds = prompt_embeds
        original_text_ids = text_ids

        num_channels_latents = self.transformer.config.in_channels // 4
        latents, latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )

        attention_mask = self._prepare_attention_mask(
            batch_size=latents.shape[0],
            sequence_length=image_seq_len,
            dtype=latents.dtype,
            attention_mask=prompt_attention_mask,
        )
        negative_attention_mask = self._prepare_attention_mask(
            batch_size=latents.shape[0],
            sequence_length=image_seq_len,
            dtype=latents.dtype,
            attention_mask=negative_prompt_attention_mask,
        )

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                timestep = t.expand(latents.shape[0]).to(latents.dtype)

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    attention_mask=attention_mask,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                if self.do_classifier_free_guidance:
                    noise_pred_text = noise_pred
                    if negative_image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                    neg_noise_pred = self.transformer(
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_image_ids,
                        attention_mask=negative_attention_mask,
                        joint_attention_kwargs=self.joint_attention_kwargs,
                        return_dict=False,
                    )[0]
                    if image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred_text - neg_noise_pred)

                    should_skip_layers = (
                        skip_guidance_layers is not None
                        and i > num_inference_steps * skip_layer_guidance_start
                        and i < num_inference_steps * skip_layer_guidance_stop
                    )
                    if should_skip_layers:
                        skip_noise_pred = self.transformer(
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            encoder_hidden_states=original_prompt_embeds,
                            txt_ids=original_text_ids,
                            img_ids=latent_image_ids,
                            attention_mask=attention_mask,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                            skip_layers=skip_guidance_layers,
                        )[0]
                        noise_pred = noise_pred + (noise_pred_text - skip_noise_pred) * self._skip_layer_guidance_scale

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    if skip_guidance_layers is not None:
                        original_prompt_embeds = prompt_embeds

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ChromaPipelineOutput(images=image)
