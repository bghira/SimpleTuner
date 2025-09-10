import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import math
import os
import einops
import torch
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5Tokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
)
from huggingface_hub.utils import validate_hf_hub_args
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin
from diffusers.loaders.lora_base import LoraBaseMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.schedulers import (
    FlowMatchEulerDiscreteScheduler,
    UniPCMultistepScheduler,
)
from diffusers.models.lora import (
    text_encoder_attn_modules,
    text_encoder_mlp_modules,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    scale_lora_layers,
    unscale_lora_layers,
    is_peft_available,
    is_peft_version,
    is_torch_version,
    is_transformers_available,
    is_transformers_version,
    get_peft_kwargs,
    get_adapter_name,
    convert_unet_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from helpers.models.hidream.schedule import FlowUniPCMultistepScheduler
from diffusers.loaders.lora_base import _fetch_state_dict
from diffusers.loaders.lora_conversion_utils import (
    _convert_non_diffusers_hidream_lora_to_diffusers,
)

from dataclasses import dataclass
from typing import List, Union
from diffusers.utils import BaseOutput

import numpy as np
import PIL.Image

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

_LOW_CPU_MEM_USAGE_DEFAULT_LORA = False
if is_torch_version(">=", "1.9.0"):
    if (
        is_peft_available()
        and is_peft_version(">=", "0.13.1")
        and is_transformers_available()
        and is_transformers_version(">", "4.45.2")
    ):
        _LOW_CPU_MEM_USAGE_DEFAULT_LORA = True

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
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


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
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


class HiDreamImageLoraLoaderMixin(LoraBaseMixin):
    r"""
    Load LoRA layers into [`HiDreamImageTransformer2DModel`], text encoders, and optionally [`HiDreamControlNetModel`].
    Specific to [`HiDreamImagePipeline`].
    """

    _lora_loadable_modules = [
        "transformer",
        "text_encoder",
        "text_encoder_2",
        "text_encoder_3",
        "text_encoder_4",
        "controlnet",
    ]
    transformer_name = "transformer"
    text_encoder_name = "text_encoder"
    controlnet_name = "controlnet"

    @classmethod
    @validate_hf_hub_args
    def lora_state_dict(
        cls,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
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
            return_lora_metadata (`bool`, *optional*, defaults to False):
                When enabled, additionally return the LoRA adapter metadata, typically found in the state dict.
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

        is_non_diffusers_format = any("diffusion_model" in k for k in state_dict)
        if is_non_diffusers_format:
            state_dict = _convert_non_diffusers_hidream_lora_to_diffusers(state_dict)

        out = (state_dict, metadata) if return_lora_metadata else state_dict
        return out

    def load_lora_weights(
        self,
        pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
        adapter_name: Optional[str] = None,
        hotswap: bool = False,
        **kwargs,
    ):
        """
        Load LoRA weights specified in `pretrained_model_name_or_path_or_dict` into `self.transformer`,
        `self.text_encoder`, `self.text_encoder_2`, `self.text_encoder_3`, `self.text_encoder_4`,
        and optionally `self.controlnet`.

        All kwargs are forwarded to `self.lora_state_dict`. See
        [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`] for more details on how the state dict is loaded.
        See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_into_transformer`] for more details on how the state
        dict is loaded into `self.transformer`.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            low_cpu_mem_usage (`bool`, *optional*):
                Speed up model loading by only loading the pretrained LoRA weights and not initializing the random
                weights.
            hotswap (`bool`, *optional*):
                See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`].
            kwargs (`dict`, *optional*):
                See [`~loaders.StableDiffusionLoraLoaderMixin.lora_state_dict`].
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        low_cpu_mem_usage = kwargs.pop(
            "low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA
        )
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # if a dict is passed, copy it instead of modifying it inplace
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = (
                pretrained_model_name_or_path_or_dict.copy()
            )

        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        kwargs["return_lora_metadata"] = True
        state_dict, metadata = self.lora_state_dict(
            pretrained_model_name_or_path_or_dict, **kwargs
        )

        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        # Separate transformer, text encoder, and controlnet weights
        transformer_state_dict = {}
        text_encoder_state_dict = {}
        text_encoder_2_state_dict = {}
        text_encoder_3_state_dict = {}
        text_encoder_4_state_dict = {}
        controlnet_state_dict = {}

        for k, v in state_dict.items():
            if k.startswith("text_encoder_4."):
                text_encoder_4_state_dict[k] = v
            elif k.startswith("text_encoder_3."):
                text_encoder_3_state_dict[k] = v
            elif k.startswith("text_encoder_2."):
                text_encoder_2_state_dict[k] = v
            elif k.startswith("text_encoder."):
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
                transformer=(
                    getattr(self, self.transformer_name)
                    if not hasattr(self, "transformer")
                    else self.transformer
                ),
                adapter_name=adapter_name,
                metadata=metadata,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
                hotswap=hotswap,
            )

        # Load text encoder weights
        if text_encoder_state_dict and hasattr(self, "text_encoder"):
            self.load_lora_into_text_encoder(
                text_encoder_state_dict,
                network_alphas=None,
                text_encoder=self.text_encoder,
                prefix="text_encoder",
                lora_scale=self.lora_scale,
                adapter_name=adapter_name,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )

        # Load text encoder 2 weights
        if text_encoder_2_state_dict and hasattr(self, "text_encoder_2"):
            self.load_lora_into_text_encoder(
                text_encoder_2_state_dict,
                network_alphas=None,
                text_encoder=self.text_encoder_2,
                prefix="text_encoder_2",
                lora_scale=self.lora_scale,
                adapter_name=adapter_name,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )

        # Load text encoder 3 weights (T5)
        if text_encoder_3_state_dict and hasattr(self, "text_encoder_3"):
            self.load_lora_into_text_encoder(
                text_encoder_3_state_dict,
                network_alphas=None,
                text_encoder=self.text_encoder_3,
                prefix="text_encoder_3",
                lora_scale=self.lora_scale,
                adapter_name=adapter_name,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )

        # Load text encoder 4 weights (Llama)
        if text_encoder_4_state_dict and hasattr(self, "text_encoder_4"):
            self.load_lora_into_text_encoder(
                text_encoder_4_state_dict,
                network_alphas=None,
                text_encoder=self.text_encoder_4,
                prefix="text_encoder_4",
                lora_scale=self.lora_scale,
                adapter_name=adapter_name,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )

        # Load controlnet weights if present
        if controlnet_state_dict and hasattr(self, "controlnet"):
            self.load_lora_into_controlnet(
                controlnet_state_dict,
                network_alphas=None,
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
        Load LoRA layers into the HiDream ControlNet.

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters for the controlnet.
            network_alphas (`Dict[str, float]`):
                The value of the network alpha used for stable learning and preventing underflow.
            controlnet (`HiDreamControlNetModel`):
                The ControlNet model to load the LoRA layers into.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            low_cpu_mem_usage (`bool`, *optional*):
                Speed up model loading by only loading the pretrained LoRA weights and not initializing the random weights.
        """
        if low_cpu_mem_usage and not is_peft_version(">=", "0.13.1"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict

        keys = list(state_dict.keys())
        controlnet_key = cls.controlnet_name

        controlnet_keys = [k for k in keys if k.startswith(controlnet_key)]
        state_dict = {
            k.replace(f"{controlnet_key}.", ""): v
            for k, v in state_dict.items()
            if k in controlnet_keys
        }

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
                alpha_keys = [
                    k for k in network_alphas.keys() if k.startswith(controlnet_key)
                ]
                network_alphas = {
                    k.replace(f"{controlnet_key}.", ""): v
                    for k, v in network_alphas.items()
                    if k in alpha_keys
                }

            lora_config_kwargs = get_peft_kwargs(
                rank, network_alpha_dict=network_alphas, peft_state_dict=state_dict
            )
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
            is_model_cpu_offload, is_sequential_cpu_offload = (
                cls._optionally_disable_offloading(_pipeline)
            )

            peft_kwargs = {}
            if is_peft_version(">=", "0.13.1"):
                peft_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

            inject_adapter_in_model(
                lora_config, controlnet, adapter_name=adapter_name, **peft_kwargs
            )
            incompatible_keys = set_peft_model_state_dict(
                controlnet, state_dict, adapter_name, **peft_kwargs
            )

            if incompatible_keys is not None:
                logger.info(
                    f"Loaded ControlNet LoRA with incompatible keys: {incompatible_keys}"
                )

            # Offload back.
            if is_model_cpu_offload:
                _pipeline.enable_model_cpu_offload()
            elif is_sequential_cpu_offload:
                _pipeline.enable_sequential_cpu_offload()

    @classmethod
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
        This will load the LoRA layers specified in `state_dict` into `transformer`.

        Parameters:
            state_dict (`dict`):
                A standard state dict containing the lora layer parameters. The keys can either be indexed directly
                into the unet or prefixed with an additional `unet` which can be used to distinguish between text
                encoder lora layers.
            transformer (`HiDreamImageTransformer2DModel`):
                The Transformer model to load the LoRA layers into.
            adapter_name (`str`, *optional*):
                Adapter name to be used for referencing the loaded adapter model. If not specified, it will use
                `default_{i}` where i is the total number of adapters being loaded.
            low_cpu_mem_usage (`bool`, *optional*):
                Speed up model loading by only loading the pretrained LoRA weights and not initializing the random
                weights.
            hotswap (`bool`, *optional*):
                See [`~loaders.StableDiffusionLoraLoaderMixin.load_lora_weights`].
            metadata (`dict`):
                Optional LoRA adapter metadata. When supplied, the `LoraConfig` arguments of `peft` won't be derived
                from the state dict.
        """
        if low_cpu_mem_usage and is_peft_version("<", "0.13.0"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # Load the layers corresponding to transformer.
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

    @classmethod
    def load_lora_into_text_encoder(
        cls,
        state_dict,
        network_alphas,
        text_encoder,
        prefix=None,
        lora_scale=1.0,
        adapter_name=None,
        _pipeline=None,
        low_cpu_mem_usage=False,
    ):
        """
        Load LoRA layers into a HiDream text encoder.
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        peft_kwargs = {}
        if low_cpu_mem_usage:
            if not is_peft_version(">=", "0.13.1"):
                raise ValueError(
                    "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
                )
            if not is_transformers_version(">", "4.45.2"):
                raise ValueError(
                    "`low_cpu_mem_usage=True` is not compatible with this `transformers` version. Please update it with `pip install -U transformers`."
                )
            peft_kwargs["low_cpu_mem_usage"] = low_cpu_mem_usage

        from peft import LoraConfig

        # Filter keys for the specific text encoder
        keys = list(state_dict.keys())
        if prefix is None:
            raise ValueError("Prefix must be specified for text encoder LoRA loading")

        text_encoder_keys = [
            k for k in keys if k.startswith(prefix) and k.split(".")[0] == prefix
        ]
        text_encoder_lora_state_dict = {
            k.replace(f"{prefix}.", ""): v
            for k, v in state_dict.items()
            if k in text_encoder_keys
        }

        if len(text_encoder_lora_state_dict) > 0:
            logger.info(f"Loading {prefix}.")
            rank = {}
            text_encoder_lora_state_dict = convert_state_dict_to_diffusers(
                text_encoder_lora_state_dict
            )

            # convert state dict
            text_encoder_lora_state_dict = convert_state_dict_to_peft(
                text_encoder_lora_state_dict
            )

            for name, _ in text_encoder_attn_modules(text_encoder):
                for module in ("out_proj", "q_proj", "k_proj", "v_proj"):
                    rank_key = f"{name}.{module}.lora_B.weight"
                    if rank_key not in text_encoder_lora_state_dict:
                        continue
                    rank[rank_key] = text_encoder_lora_state_dict[rank_key].shape[1]

            for name, _ in text_encoder_mlp_modules(text_encoder):
                for module in ("fc1", "fc2"):
                    rank_key = f"{name}.{module}.lora_B.weight"
                    if rank_key not in text_encoder_lora_state_dict:
                        continue
                    rank[rank_key] = text_encoder_lora_state_dict[rank_key].shape[1]

            if network_alphas is not None:
                alpha_keys = [
                    k
                    for k in network_alphas.keys()
                    if k.startswith(prefix) and k.split(".")[0] == prefix
                ]
                network_alphas = {
                    k.replace(f"{prefix}.", ""): v
                    for k, v in network_alphas.items()
                    if k in alpha_keys
                }

            lora_config_kwargs = get_peft_kwargs(
                rank, network_alphas, text_encoder_lora_state_dict, is_unet=False
            )
            if "use_dora" in lora_config_kwargs:
                if lora_config_kwargs["use_dora"]:
                    if is_peft_version("<", "0.9.0"):
                        raise ValueError(
                            "You need `peft` 0.9.0 at least to use DoRA-enabled LoRAs. Please upgrade your installation of `peft`."
                        )
                else:
                    if is_peft_version("<", "0.9.0"):
                        lora_config_kwargs.pop("use_dora")
            lora_config = LoraConfig(**lora_config_kwargs)

            # adapter_name
            if adapter_name is None:
                adapter_name = get_adapter_name(text_encoder)

            is_model_cpu_offload, is_sequential_cpu_offload = (
                cls._optionally_disable_offloading(_pipeline)
            )

            # inject LoRA layers and load the state dict
            # in transformers we automatically check whether the adapter name is already in use or not
            text_encoder.load_adapter(
                adapter_name=adapter_name,
                adapter_state_dict=text_encoder_lora_state_dict,
                peft_config=lora_config,
                **peft_kwargs,
            )

            # scale LoRA layers with `lora_scale`
            scale_lora_layers(text_encoder, weight=lora_scale)

            text_encoder.to(device=text_encoder.device, dtype=text_encoder.dtype)

            # Offload back.
            if is_model_cpu_offload:
                _pipeline.enable_model_cpu_offload()
            elif is_sequential_cpu_offload:
                _pipeline.enable_sequential_cpu_offload()

    @classmethod
    def save_lora_weights(
        cls,
        save_directory: Union[str, os.PathLike],
        transformer_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        text_encoder_lora_layers: Dict[
            str, Union[torch.nn.Module, torch.Tensor]
        ] = None,
        text_encoder_2_lora_layers: Dict[
            str, Union[torch.nn.Module, torch.Tensor]
        ] = None,
        text_encoder_3_lora_layers: Dict[
            str, Union[torch.nn.Module, torch.Tensor]
        ] = None,
        text_encoder_4_lora_layers: Dict[
            str, Union[torch.nn.Module, torch.Tensor]
        ] = None,
        controlnet_lora_layers: Dict[str, Union[torch.nn.Module, torch.Tensor]] = None,
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        transformer_lora_adapter_metadata: Optional[dict] = None,
    ):
        r"""
        Save the LoRA parameters corresponding to the transformer, text encoders, and optionally controlnet.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save LoRA parameters to. Will be created if it doesn't exist.
            transformer_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `transformer`.
            text_encoder_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder`.
            text_encoder_2_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder_2`.
            text_encoder_3_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder_3`.
            text_encoder_4_lora_layers (`Dict[str, torch.nn.Module]` or `Dict[str, torch.Tensor]`):
                State dict of the LoRA layers corresponding to the `text_encoder_4`.
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
            transformer_lora_adapter_metadata:
                LoRA adapter metadata associated with the transformer to be serialized with the state dict.
        """
        state_dict = {}
        lora_adapter_metadata = {}

        if not (
            transformer_lora_layers
            or text_encoder_lora_layers
            or text_encoder_2_lora_layers
            or text_encoder_3_lora_layers
            or text_encoder_4_lora_layers
            or controlnet_lora_layers
        ):
            raise ValueError("You must pass at least one set of LoRA layers.")

        if transformer_lora_layers:
            state_dict.update(
                cls.pack_weights(transformer_lora_layers, cls.transformer_name)
            )

        if text_encoder_lora_layers:
            state_dict.update(
                cls.pack_weights(text_encoder_lora_layers, "text_encoder")
            )

        if text_encoder_2_lora_layers:
            state_dict.update(
                cls.pack_weights(text_encoder_2_lora_layers, "text_encoder_2")
            )

        if text_encoder_3_lora_layers:
            state_dict.update(
                cls.pack_weights(text_encoder_3_lora_layers, "text_encoder_3")
            )

        if text_encoder_4_lora_layers:
            state_dict.update(
                cls.pack_weights(text_encoder_4_lora_layers, "text_encoder_4")
            )

        if controlnet_lora_layers:
            state_dict.update(
                cls.pack_weights(controlnet_lora_layers, cls.controlnet_name)
            )

        if transformer_lora_adapter_metadata is not None:
            lora_adapter_metadata.update(
                cls.pack_weights(
                    transformer_lora_adapter_metadata, cls.transformer_name
                )
            )

        # Save the model
        cls.write_lora_layers(
            state_dict=state_dict,
            save_directory=save_directory,
            is_main_process=is_main_process,
            weight_name=weight_name,
            save_function=save_function,
            safe_serialization=safe_serialization,
            lora_adapter_metadata=lora_adapter_metadata,
        )

    def fuse_lora(
        self,
        components: List[str] = [
            "transformer",
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "text_encoder_4",
            "controlnet",
        ],
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
            "HiDream-ai/HiDream-I1-Full", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights("path/to/lora", adapter_name="my_lora")
        pipeline.fuse_lora(lora_scale=0.7)
        ```
        """
        super().fuse_lora(
            components=components,
            lora_scale=lora_scale,
            safe_fusing=safe_fusing,
            adapter_names=adapter_names,
            **kwargs,
        )

    def unfuse_lora(
        self,
        components: List[str] = [
            "transformer",
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "text_encoder_4",
            "controlnet",
        ],
        **kwargs,
    ):
        r"""
        Reverses the effect of
        [`pipe.fuse_lora()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraBaseMixin.fuse_lora).

        <Tip warning={true}>

        This is an experimental API.

        </Tip>

        Args:
            components (`List[str]`): List of LoRA-injectable components to unfuse LoRA from.
        """
        super().unfuse_lora(components=components, **kwargs)


@dataclass
class HiDreamImagePipelineOutput(BaseOutput):
    """
    Output class for HiDreamImage pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]


class HiDreamImagePipeline(
    DiffusionPipeline, FromSingleFileMixin, HiDreamImageLoraLoaderMixin
):
    model_cpu_offload_seq = "text_encoder->text_encoder_2->text_encoder_3->text_encoder_4->image_encoder->transformer->vae"
    _optional_components = ["image_encoder", "feature_extractor"]
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        transformer,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5Tokenizer,
        text_encoder_4: LlamaForCausalLM,
        tokenizer_4: PreTrainedTokenizerFast,
    ):
        super().__init__()

        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            text_encoder_3=text_encoder_3,
            text_encoder_4=text_encoder_4,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            tokenizer_3=tokenizer_3,
            tokenizer_4=tokenizer_4,
            scheduler=scheduler,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
            if hasattr(self, "vae") and self.vae is not None
            else 8
        )
        # HiDreamImage latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor * 2
        )
        self.default_sample_size = 128
        self.tokenizer_4.pad_token = self.tokenizer_4.eos_token

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Get T5 text encoder embeddings for the given prompt.

        Args:
            prompt: Text prompt to encode
            num_images_per_prompt: Number of images to generate per prompt
            max_sequence_length: Maximum sequence length for tokenization
            device: Device to place embeddings on
            dtype: Data type for embeddings

        Returns:
            T5 embeddings tensor of shape [batch_size, seq_len, dim]
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_3.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, self.tokenizer_3.model_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer_3(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer_3.batch_decode(
                untruncated_ids[
                    :,
                    min(max_sequence_length, self.tokenizer_3.model_max_length)
                    - 1 : -1,
                ]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {min(max_sequence_length, self.tokenizer_3.model_max_length)} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder_3(
            text_input_ids.to(device), attention_mask=attention_mask.to(device)
        )[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_images_per_prompt, seq_len, -1
        )
        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        tokenizer,
        text_encoder,
        prompt: Union[str, List[str]],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Get CLIP text encoder embeddings for the given prompt.

        Args:
            tokenizer: CLIP tokenizer
            text_encoder: CLIP text encoder
            prompt: Text prompt to encode
            num_images_per_prompt: Number of images to generate per prompt
            max_sequence_length: Maximum sequence length for tokenization
            device: Device to place embeddings on
            dtype: Data type for embeddings

        Returns:
            CLIP embeddings tensor of shape [batch_size, embedding_dim]
        """
        device = device or self._execution_device
        dtype = dtype or text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, 128),
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        untruncated_ids = tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = tokenizer.batch_decode(untruncated_ids[:, 128 - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {128} tokens: {removed_text}"
            )
        prompt_embeds = text_encoder(
            text_input_ids.to(device), output_hidden_states=True
        )

        # Use pooled output of CLIPTextModel
        prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds

    def _get_llama3_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 128,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Get Llama text encoder embeddings for the given prompt.

        Args:
            prompt: Text prompt to encode
            num_images_per_prompt: Number of images to generate per prompt
            max_sequence_length: Maximum sequence length for tokenization
            device: Device to place embeddings on
            dtype: Data type for embeddings

        Returns:
            Llama embeddings tensor of shape [num_layers, batch_size, seq_len, dim]
        """
        device = device or self._execution_device
        dtype = dtype or self.text_encoder_4.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer_4(
            prompt,
            padding="max_length",
            max_length=min(max_sequence_length, self.tokenizer_4.model_max_length),
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer_4(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer_4.batch_decode(
                untruncated_ids[
                    :,
                    min(max_sequence_length, self.tokenizer_4.model_max_length)
                    - 1 : -1,
                ]
            )
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {min(max_sequence_length, self.tokenizer_4.model_max_length)} tokens: {removed_text}"
            )

        outputs = self.text_encoder_4(
            text_input_ids.to(device),
            attention_mask=attention_mask.to(device),
            output_hidden_states=True,
            output_attentions=True,
        )

        # Get all hidden states (layers) and stack them for the transformer to select
        prompt_embeds = outputs.hidden_states[1:]
        prompt_embeds = torch.stack(prompt_embeds, dim=0)
        _, _, seq_len, dim = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, 1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            -1, batch_size * num_images_per_prompt, seq_len, dim
        )
        return prompt_embeds

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
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
        """
        Prepare latents for denoising.

        Args:
            batch_size: Batch size
            num_channels_latents: Number of channels in latents
            height: Image height
            width: Image width
            dtype: Data type for latents
            device: Device to place latents on
            generator: Random number generator
            latents: Optional existing latents to use

        Returns:
            Prepared latent tensors
        """
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            if latents.shape != shape:
                raise ValueError(
                    f"Unexpected latents shape, got {latents.shape}, expected {shape}"
                )
            latents = latents.to(device)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        prompt_4: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt_4: Optional[Union[str, List[str]]] = None,
        t5_prompt_embeds: Optional[torch.FloatTensor] = None,
        llama_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_t5_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_llama_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 128,
        lora_scale: Optional[float] = None,
    ):
        """
        Encode prompt into model embeddings needed for transformer.

        Args:
            prompt: Main text prompt (for CLIP L/14)
            prompt_2: Secondary text prompt (for CLIP G/14)
            prompt_3: Text prompt for T5 encoder
            prompt_4: Text prompt for Llama encoder
            device: Device to place embeddings on
            dtype: Data type for embeddings
            num_images_per_prompt: Number of images to generate per prompt
            do_classifier_free_guidance: Whether to use classifier-free guidance
            negative_prompt: Negative prompt for CLIP L/14
            negative_prompt_2: Negative prompt for CLIP G/14
            negative_prompt_3: Negative prompt for T5
            negative_prompt_4: Negative prompt for Llama
            t5_prompt_embeds: Pre-computed T5 prompt embeddings
            llama_prompt_embeds: Pre-computed Llama prompt embeddings
            negative_t5_prompt_embeds: Pre-computed negative T5 prompt embeddings
            negative_llama_prompt_embeds: Pre-computed negative Llama prompt embeddings
            pooled_prompt_embeds: Pre-computed pooled prompt embeddings
            negative_pooled_prompt_embeds: Pre-computed negative pooled prompt embeddings
            max_sequence_length: Maximum sequence length for tokenization
            lora_scale: Scale for LoRA weights

        Returns:
            Tuple containing:
            - t5_prompt_embeds: T5 encoder embeddings
            - llama_prompt_embeds: Llama encoder embeddings
            - negative_t5_prompt_embeds: Negative T5 encoder embeddings (if using guidance)
            - negative_llama_prompt_embeds: Negative Llama encoder embeddings (if using guidance)
            - pooled_prompt_embeds: Pooled CLIP embeddings
            - negative_pooled_prompt_embeds: Negative pooled CLIP embeddings (if using guidance)
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            # If no prompt is provided, determine batch size from embeddings
            if t5_prompt_embeds is not None:
                batch_size = t5_prompt_embeds.shape[0]
            elif llama_prompt_embeds is not None:
                # Handle based on expected shape format
                if len(llama_prompt_embeds.shape) == 4:  # [num_layers, batch, seq, dim]
                    batch_size = llama_prompt_embeds.shape[1]
                elif (
                    len(llama_prompt_embeds.shape) == 5
                ):  # [batch, num_layers, 1, seq, dim]
                    batch_size = llama_prompt_embeds.shape[0]
                else:
                    raise ValueError(
                        f"Unexpected llama embedding shape: {llama_prompt_embeds.shape}"
                    )
            else:
                raise ValueError(
                    "Either prompt or pre-computed embeddings must be provided"
                )

        # Check if we need to compute embeddings or use provided ones
        if (
            t5_prompt_embeds is None
            or llama_prompt_embeds is None
            or pooled_prompt_embeds is None
        ):
            t5_prompt_embeds, llama_prompt_embeds, pooled_prompt_embeds = (
                self._encode_prompt(
                    prompt=prompt,
                    prompt_2=prompt_2,
                    prompt_3=prompt_3,
                    prompt_4=prompt_4,
                    device=device,
                    dtype=dtype,
                    num_images_per_prompt=num_images_per_prompt,
                    t5_prompt_embeds=t5_prompt_embeds,
                    llama_prompt_embeds=llama_prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    max_sequence_length=max_sequence_length,
                )
            )

        # Handle negative embeddings for classifier-free guidance
        if do_classifier_free_guidance:
            if (
                negative_t5_prompt_embeds is None
                or negative_llama_prompt_embeds is None
                or negative_pooled_prompt_embeds is None
            ):
                negative_prompt = negative_prompt or ""
                negative_prompt_2 = negative_prompt_2 or negative_prompt
                negative_prompt_3 = negative_prompt_3 or negative_prompt
                negative_prompt_4 = negative_prompt_4 or negative_prompt

                # normalize str to list
                negative_prompt = (
                    batch_size * [negative_prompt]
                    if isinstance(negative_prompt, str)
                    else negative_prompt
                )
                negative_prompt_2 = (
                    batch_size * [negative_prompt_2]
                    if isinstance(negative_prompt_2, str)
                    else negative_prompt_2
                )
                negative_prompt_3 = (
                    batch_size * [negative_prompt_3]
                    if isinstance(negative_prompt_3, str)
                    else negative_prompt_3
                )
                negative_prompt_4 = (
                    batch_size * [negative_prompt_4]
                    if isinstance(negative_prompt_4, str)
                    else negative_prompt_4
                )

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

                (
                    negative_t5_prompt_embeds,
                    negative_llama_prompt_embeds,
                    negative_pooled_prompt_embeds,
                ) = self._encode_prompt(
                    prompt=negative_prompt,
                    prompt_2=negative_prompt_2,
                    prompt_3=negative_prompt_3,
                    prompt_4=negative_prompt_4,
                    device=device,
                    dtype=dtype,
                    num_images_per_prompt=num_images_per_prompt,
                    t5_prompt_embeds=negative_t5_prompt_embeds,
                    llama_prompt_embeds=negative_llama_prompt_embeds,
                    pooled_prompt_embeds=negative_pooled_prompt_embeds,
                    max_sequence_length=max_sequence_length,
                )

        return (
            t5_prompt_embeds,
            llama_prompt_embeds,
            negative_t5_prompt_embeds,
            negative_llama_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        )

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        prompt_2: Union[str, List[str]],
        prompt_3: Union[str, List[str]],
        prompt_4: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_images_per_prompt: int = 1,
        t5_prompt_embeds: Optional[torch.FloatTensor] = None,
        llama_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 128,
    ):
        """
        Internal method to encode prompts to embeddings for the model.

        Args:
            prompt: Main text prompt (for CLIP L/14)
            prompt_2: Secondary text prompt (for CLIP G/14)
            prompt_3: Text prompt for T5 encoder
            prompt_4: Text prompt for Llama encoder
            device: Device to place embeddings on
            dtype: Data type for embeddings
            num_images_per_prompt: Number of images to generate per prompt
            t5_prompt_embeds: Pre-computed T5 prompt embeddings
            llama_prompt_embeds: Pre-computed Llama prompt embeddings
            pooled_prompt_embeds: Pre-computed pooled prompt embeddings
            max_sequence_length: Maximum sequence length for tokenization

        Returns:
            Tuple containing:
            - t5_prompt_embeds: T5 encoder embeddings
            - llama_prompt_embeds: Llama encoder embeddings
            - pooled_prompt_embeds: Pooled CLIP embeddings
        """
        device = device or self._execution_device

        # Check if we need to compute any embeddings
        need_pooled = pooled_prompt_embeds is None
        need_t5 = t5_prompt_embeds is None
        need_llama = llama_prompt_embeds is None

        if need_pooled or need_t5 or need_llama:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2

            prompt_3 = prompt_3 or prompt
            prompt_3 = [prompt_3] if isinstance(prompt_3, str) else prompt_3

            prompt_4 = prompt_4 or prompt
            prompt_4 = [prompt_4] if isinstance(prompt_4, str) else prompt_4

            # Get CLIP embeddings for pooled embeddings if needed
            if need_pooled:
                # Get CLIP L/14 embeddings
                pooled_prompt_embeds_1 = self._get_clip_prompt_embeds(
                    self.tokenizer,
                    self.text_encoder,
                    prompt=prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )

                # Get CLIP G/14 embeddings
                pooled_prompt_embeds_2 = self._get_clip_prompt_embeds(
                    self.tokenizer_2,
                    self.text_encoder_2,
                    prompt=prompt_2,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )

                # Concatenate CLIP embeddings
                pooled_prompt_embeds = torch.cat(
                    [pooled_prompt_embeds_1, pooled_prompt_embeds_2], dim=-1
                )

            # Get T5 embeddings if needed
            if need_t5:
                t5_prompt_embeds = self._get_t5_prompt_embeds(
                    prompt=prompt_3,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )

            # Get Llama embeddings if needed
            if need_llama:
                llama_prompt_embeds = self._get_llama3_prompt_embeds(
                    prompt=prompt_4,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                    device=device,
                    dtype=dtype,
                )

        return t5_prompt_embeds, llama_prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        prompt_3: Optional[Union[str, List[str]]] = None,
        prompt_4: Optional[Union[str, List[str]]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt_3: Optional[Union[str, List[str]]] = None,
        negative_prompt_4: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        t5_prompt_embeds: Optional[torch.FloatTensor] = None,
        llama_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_t5_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_llama_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 128,
    ):
        """
        Generate images based on text prompts.

        Args:
            prompt: Main text prompt (for CLIP L/14)
            prompt_2: Secondary text prompt (for CLIP G/14)
            prompt_3: Text prompt for T5 encoder
            prompt_4: Text prompt for Llama encoder
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            sigmas: Optional custom sigmas for scheduler
            guidance_scale: Scale for classifier-free guidance
            negative_prompt: Negative prompt for CLIP L/14
            negative_prompt_2: Negative prompt for CLIP G/14
            negative_prompt_3: Negative prompt for T5
            negative_prompt_4: Negative prompt for Llama
            num_images_per_prompt: Number of images to generate per prompt
            generator: Random number generator
            latents: Optional existing latents to use
            t5_prompt_embeds: Pre-computed T5 prompt embeddings
            llama_prompt_embeds: Pre-computed Llama prompt embeddings
            negative_t5_prompt_embeds: Pre-computed negative T5 prompt embeddings
            negative_llama_prompt_embeds: Pre-computed negative Llama prompt embeddings
            pooled_prompt_embeds: Pre-computed pooled prompt embeddings
            negative_pooled_prompt_embeds: Pre-computed negative pooled prompt embeddings
            output_type: Output type - "pil", "latent", or "pt"
            return_dict: Whether to return as a dictionary
            joint_attention_kwargs: Additional attention parameters
            callback_on_step_end: Callback after each denoising step
            callback_on_step_end_tensor_inputs: Tensor inputs to pass to callback
            max_sequence_length: Maximum sequence length for tokenization

        Returns:
            Generated images
        """
        # 1. Set up image dimensions and scales
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        division = self.vae_scale_factor * 2
        S_max = (self.default_sample_size * self.vae_scale_factor) ** 2
        scale = S_max / (width * height)
        scale = math.sqrt(scale)
        width, height = int(width * scale // division * division), int(
            height * scale // division * division
        )

        # 2. Set up parameters for generation
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        # 3. Determine batch size
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            # Determine batch size from embeddings
            if t5_prompt_embeds is not None:
                batch_size = t5_prompt_embeds.shape[0]
            elif llama_prompt_embeds is not None:
                # Handle based on expected shape format
                if len(llama_prompt_embeds.shape) == 4:  # [num_layers, batch, seq, dim]
                    batch_size = llama_prompt_embeds.shape[1]
                elif (
                    len(llama_prompt_embeds.shape) == 5
                ):  # [batch, num_layers, 1, seq, dim]
                    batch_size = llama_prompt_embeds.shape[0]
                else:
                    raise ValueError(
                        f"Unexpected llama embedding shape: {llama_prompt_embeds.shape}"
                    )
            else:
                raise ValueError("Either prompt or embeddings must be provided")

        device = self.transformer.device

        # 4. Encode prompts
        lora_scale = (
            self.joint_attention_kwargs.get("scale", None)
            if self.joint_attention_kwargs is not None
            else None
        )
        (
            t5_prompt_embeds,
            llama_prompt_embeds,
            negative_t5_prompt_embeds,
            negative_llama_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_3=prompt_3,
            prompt_4=prompt_4,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            negative_prompt_3=negative_prompt_3,
            negative_prompt_4=negative_prompt_4,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            t5_prompt_embeds=t5_prompt_embeds,
            llama_prompt_embeds=llama_prompt_embeds,
            negative_t5_prompt_embeds=negative_t5_prompt_embeds,
            negative_llama_prompt_embeds=negative_llama_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        # 5. Prepare embeddings for guidance if needed
        if self.do_classifier_free_guidance:
            # Format embeddings for the transformer which expects separate inputs
            # Handle T5 embeddings (shape: [batch, seq_len, dim])
            if negative_t5_prompt_embeds is not None:
                t5_embeds_input = torch.cat(
                    [negative_t5_prompt_embeds, t5_prompt_embeds], dim=0
                )
            else:
                t5_embeds_input = t5_prompt_embeds

            # Handle Llama embeddings
            if negative_llama_prompt_embeds is not None:
                # The shape handling depends on the format of llama embeddings
                if len(llama_prompt_embeds.shape) == 4:  # [num_layers, batch, seq, dim]
                    llama_embeds_input = torch.cat(
                        [negative_llama_prompt_embeds, llama_prompt_embeds], dim=1
                    )
                elif (
                    len(llama_prompt_embeds.shape) == 5
                ):  # [batch, num_layers, 1, seq, dim]
                    llama_embeds_input = torch.cat(
                        [negative_llama_prompt_embeds, llama_prompt_embeds], dim=0
                    )
                else:
                    raise ValueError(
                        f"Unexpected llama embedding shape: {llama_prompt_embeds.shape}"
                    )
            else:
                llama_embeds_input = llama_prompt_embeds

            # Combine embeddings for passing to transformer
            pooled_embeds_input = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )
        else:
            # If not using guidance, use the embeddings directly
            t5_embeds_input = t5_prompt_embeds
            llama_embeds_input = llama_prompt_embeds
            pooled_embeds_input = pooled_prompt_embeds

        # 6. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            pooled_embeds_input.dtype,
            device,
            generator,
            latents,
        )

        # 7. Prepare spatial data for non-square images
        if latents.shape[-2] != latents.shape[-1]:
            B, C, H, W = latents.shape
            pH, pW = (
                H // self.transformer.config.patch_size,
                W // self.transformer.config.patch_size,
            )

            img_sizes = torch.tensor([pH, pW], dtype=torch.int64).reshape(-1)
            img_ids = torch.zeros(pH, pW, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW)[None, :]
            img_ids = img_ids.reshape(pH * pW, -1)
            img_ids_pad = torch.zeros(self.transformer.max_seq, 3)
            img_ids_pad[: pH * pW, :] = img_ids

            img_sizes = img_sizes.unsqueeze(0).to(latents.device)
            img_ids = img_ids_pad.unsqueeze(0).to(latents.device)
            if self.do_classifier_free_guidance:
                img_sizes = img_sizes.repeat(2 * B, 1)
                img_ids = img_ids.repeat(2 * B, 1, 1)
        else:
            img_sizes = img_ids = None

        # 8. Prepare timesteps
        mu = calculate_shift(self.transformer.max_seq)
        scheduler_kwargs = {"mu": mu}
        if isinstance(self.scheduler, FlowUniPCMultistepScheduler):
            self.scheduler.set_timesteps(
                num_inference_steps, device=device, shift=math.exp(mu)
            )
            timesteps = self.scheduler.timesteps
        elif isinstance(self.scheduler, UniPCMultistepScheduler):
            self.scheduler.set_timesteps(
                num_inference_steps, device=device
            )  # , shift=math.exp(mu))
            timesteps = self.scheduler.timesteps
        else:
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                **scheduler_kwargs,
            )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 9. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # Reshape latents for transformer if needed
                if latent_model_input.shape[-2] != latent_model_input.shape[-1]:
                    B, C, H, W = latent_model_input.shape
                    patch_size = self.transformer.config.patch_size
                    pH, pW = H // patch_size, W // patch_size
                    out = torch.zeros(
                        (B, C, self.transformer.max_seq, patch_size * patch_size),
                        dtype=latent_model_input.dtype,
                        device=latent_model_input.device,
                    )
                    latent_model_input = einops.rearrange(
                        latent_model_input,
                        "B C (H p1) (W p2) -> B C (H W) (p1 p2)",
                        p1=patch_size,
                        p2=patch_size,
                    )
                    out[:, :, 0 : pH * pW] = latent_model_input
                    latent_model_input = out

                # Call transformer with the updated input format
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timesteps=timestep,
                    t5_hidden_states=t5_embeds_input,
                    llama_hidden_states=llama_embeds_input,
                    pooled_embeds=pooled_embeds_input,
                    img_sizes=img_sizes,
                    img_ids=img_ids,
                    return_dict=False,
                )[0]
                noise_pred = -noise_pred

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    t5_embeds_input = callback_outputs.pop(
                        "t5_embeds_input", t5_embeds_input
                    )
                    llama_embeds_input = callback_outputs.pop(
                        "llama_embeds_input", llama_embeds_input
                    )
                    pooled_embeds_input = callback_outputs.pop(
                        "pooled_embeds_input", pooled_embeds_input
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        # 10. Post-processing
        if output_type == "latent":
            image = latents
        else:
            latents = (
                latents / self.vae.config.scaling_factor
            ) + self.vae.config.shift_factor

            image = self.vae.decode(
                latents.to(dtype=self.vae.dtype, device=self.vae.device),
                return_dict=False,
            )[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return HiDreamImagePipelineOutput(images=image)
